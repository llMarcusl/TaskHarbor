"""Managed background workers; the database queue survives process restarts."""

from __future__ import annotations

import atexit
import logging
import threading
import time
from functools import lru_cache

from langchain_core.messages import AIMessage, ToolMessage

from .extractor import MemoryExtractor, evidence_text
from .repository import LostLease, MemoryRepository
from .schemas import SessionMemory, digest, empty_long_term

logger = logging.getLogger(__name__)


def raw_records(messages):
    """Archive complete raw messages; compaction must mark its synthetic summary.

    LangGraph's completed-node hooks provide stable IDs. Stream chunks are not
    evidence, and a shortened state is only another batch of observations.
    """
    result = []
    for message in messages:
        if message.type not in ("human", "ai", "tool"):
            continue
        metadata = message.additional_kwargs
        if (
            metadata.get("memory_internal")
            or metadata.get("lc_source") == "async_subagent"
            or metadata.get("deepagents_notification")
        ):
            continue
        payload = message.model_dump(mode="json")
        # Keep evidence IDs compatible while ignoring provider usage/latency metadata.
        identity = {
            k: v for k, v in payload.items() if k not in ("response_metadata", "usage_metadata")
        }
        result.append({"event_id": digest(identity), "payload": payload})
    return result


def complete_boundary(messages):
    pending = set()
    for message in messages:
        if isinstance(message, AIMessage):
            pending.update(c["id"] for c in message.tool_calls)
        elif isinstance(message, ToolMessage):
            if message.tool_call_id not in pending:
                return False
            pending.remove(message.tool_call_id)
    return not pending


class MemoryService:
    def __init__(self, repository, settings, extractor_factory=MemoryExtractor):
        self.repository = repository
        self.settings = settings
        self.extractor_factory = extractor_factory
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._threads = []
        self._started = False
        self.lease_seconds = max(60, 2 * settings.MEMORY_TIMEOUT_SECONDS + 30)

    def start(self):
        with self._lock:
            if self._started and not self._stop.is_set():
                return
            if self._stop.is_set() and any(t.is_alive() for t in self._threads):
                raise RuntimeError("memory_workers_still_stopping")
            self.repository.setup()
            self._stop.clear()
            self._threads = [
                threading.Thread(target=self._worker, name=f"memory-worker-{i}", daemon=True)
                for i in range(self.settings.MEMORY_WORKERS)
            ]
            self._started = True
            for thread in self._threads:
                thread.start()

    def close(self):
        with self._lock:
            self._stop.set()
            self._wake.set()
            deadline = time.monotonic() + self.settings.MEMORY_TIMEOUT_SECONDS + 5
            for thread in self._threads:
                thread.join(timeout=max(0, deadline - time.monotonic()))
            self._threads = [t for t in self._threads if t.is_alive()]
            self._started = bool(self._threads)

    def observe(self, kind, messages, thread_id, *, schedule):
        """Short archive/queue operation only. Model calls run in worker threads."""
        if not thread_id:
            return
        self.start()
        owner = self.repository.owner_for_thread(thread_id)
        if not owner:
            logger.warning("Memory extraction skipped: no server-side thread owner")
            return
        records = raw_records(messages)
        end_seq = self.repository.capture(owner, thread_id, records)
        if schedule and complete_boundary(messages):
            debounce = self.settings.LONG_TERM_MEMORY_DEBOUNCE_SECONDS if kind == "long_term" else 0
            self.repository.enqueue(
                owner,
                thread_id,
                kind,
                end_seq,
                debounce,
                self.settings.LONG_TERM_MEMORY_MAX_DELAY_SECONDS,
            )
            self._wake.set()

    def process(self, job, extractor):
        target, doc = self.repository.load(job)
        after_seq = job["cursor_seq"]
        kind = job["kind"]
        body = doc["body"]

        def heartbeat():
            if self._stop.is_set():
                raise LostLease("memory_worker_stopping")
            self.repository.heartbeat(job, self.lease_seconds)

        if kind == "session":
            # Approximate character-based trigger, never accumulated API input_tokens.
            threshold = (
                self.settings.SESSION_MEMORY_STEP_TOKENS
                if body
                else self.settings.SESSION_MEMORY_INITIAL_TOKENS
            )
            growth = 0
            for page in self.repository.pages(job, target, after_seq):
                heartbeat()
                growth += sum(len(evidence_text(r["payload"])) for r in page)
                if growth >= threshold * 3:
                    break
            if growth < threshold * 3:
                self.repository.finish(job, None, doc["revision"], {})
                return
            body = body or SessionMemory().model_dump()
        else:
            body = body or empty_long_term()

        # Each page is released before loading the next. Partial results remain
        # local: any page failure leaves both stored memory and cursor unchanged.
        for page in self.repository.pages(job, target, after_seq):
            heartbeat()
            body = extractor.extract(kind, body, page, job["thread_id"], heartbeat)
        heartbeat()
        self.repository.finish(
            job,
            body,
            doc["revision"],
            {
                "source_thread_id": job["thread_id"],
                "target_seq": job["target_seq"],
                "model": self.settings.MEMORY_MODEL,
                "provider": self.settings.MEMORY_MODEL_PROVIDER,
            },
        )

    def _worker(self):
        extractor = self.extractor_factory(self.settings)
        kinds = []
        if self.settings.SESSION_MEMORY_ENABLED:
            kinds.append("session")
        if self.settings.LONG_TERM_MEMORY_ENABLED:
            kinds.append("long_term")
        while not self._stop.is_set():
            job = None
            try:
                job = self.repository.claim(self.lease_seconds, kinds)
                if job is None:
                    self._wake.wait(self.settings.MEMORY_POLL_SECONDS)
                    self._wake.clear()
                    continue
                self.process(job, extractor)
            except Exception as exc:
                # Only exception types are logged/persisted, never provider payloads or keys.
                logger.warning("Memory extraction failed (%s)", type(exc).__name__)
                if job is not None:
                    try:
                        self.repository.fail(
                            job, type(exc).__name__, self.settings.MEMORY_MAX_ATTEMPTS
                        )
                    except Exception:
                        logger.warning("Memory failure status unavailable; lease will expire")
                self._stop.wait(self.settings.MEMORY_POLL_SECONDS)


_service_lock = threading.Lock()


def get_memory_service():
    # functools' cache alone can create duplicate services on concurrent misses.
    with _service_lock:
        return _create_memory_service()


@lru_cache(maxsize=1)
def _create_memory_service():
    from taskharbor.clients.infra import get_db_pool
    from taskharbor.settings import get_settings

    settings = get_settings()
    service = MemoryService(
        MemoryRepository(get_db_pool(), standalone_owner=settings.MEMORY_STANDALONE_OWNER), settings
    )
    atexit.register(service.close)
    return service
