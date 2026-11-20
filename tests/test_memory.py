"""Memory contracts; PostgreSQL cases use an isolated disposable test schema.

Set MEMORY_TEST_DSN explicitly to run the database cases. Never uses application
credentials or Aegra's live tables. No real LLM or Docker sandbox calls are made.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
from uuid import uuid4

import psycopg
import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import ValidationError

from taskharbor.agent import factory
from taskharbor.agent.memory.extractor import MemoryExtractor, batches
from taskharbor.agent.memory.middleware import (
    LongTermMemoryMiddleware,
    SessionMemoryMiddleware,
)
from taskharbor.agent.memory.repository import LostLease, MemoryRepository
from taskharbor.agent.memory.schemas import LongTermDelta, SessionMemory, merge_long_term
from taskharbor.agent.memory.service import (
    MemoryService,
    complete_boundary,
    raw_records,
)
from taskharbor.settings import Settings


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        SUBAGENTS_ENABLED=False,
        CHAT_MODEL_API_KEY="unused",
        POSTGRES_USER="unused",
        POSTGRES_PASSWORD="unused",
        POSTGRES_DB="unused",
        POSTGRES_HOST="localhost",
        POSTGRES_PORT="5432",
        MEMORY_MODEL="test-small",
        MEMORY_MODEL_PROVIDER="openai",
        SESSION_MEMORY_ENABLED=True,
        LONG_TERM_MEMORY_ENABLED=True,
        LONG_TERM_MEMORY_DEBOUNCE_SECONDS=0,
        SESSION_MEMORY_INITIAL_TOKENS=1,
        SESSION_MEMORY_STEP_TOKENS=1,
        MEMORY_POLL_SECONDS=0.05,
    )


def fact(source="human", **overrides):
    return {
        "action": "add",
        "content": "Prefers Chinese",
        "category": "preference",
        "confidence": 0.95,
        "scope": "user",
        "durability": "durable",
        "authority": "descriptive",
        "source_ids": [source],
        **overrides,
    }


def merge(changes, old=None, **kwargs):
    return merge_long_term(
        old,
        LongTermDelta.model_validate({"facts": changes}),
        user_sources={"human"},
        thread_id="t",
        threshold=0.7,
        max_facts=100,
        **kwargs,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"confidence": 0.69},
        {"scope": "project"},
        {"scope": "session"},
        {"durability": "temporary"},
        {"authority": "instruction"},
        {"source_ids": ["assistant"]},
        {"category": "correction", "confidence": 0.9},
    ],
)
def test_long_term_write_gates(overrides):
    assert merge([fact(**overrides)])["facts"] == []


def test_fact_correction_confirmation_and_idempotency():
    original = merge([fact()])
    assert merge([fact()], original) == original
    target = original["facts"][0]["id"]
    assert merge([fact(action="update", target_id=target, content="English")], original) == original
    corrected = merge(
        [fact(action="update", target_id=target, content="English", explicit_correction=True)],
        original,
    )
    assert [f["content"] for f in corrected["facts"]] == ["English"]
    assert (
        merge([fact(action="retract", target_id=target, explicit_correction=True)], original)[
            "facts"
        ]
        == []
    )
    # An older concurrent extraction must not undo a later correction.
    original["facts"][0]["observed_at"] = "2026-11-02T12:00:00+00:00"
    assert (
        merge(
            [fact(action="retract", target_id=target, explicit_correction=True)],
            original,
            source_times={"human": "2026-11-02T11:00:00+00:00"},
        )
        == original
    )


def test_section_gates_apply_to_all_three_layers():
    delta = LongTermDelta.model_validate(
        {
            "sections": [
                {
                    "name": "workContext",
                    "content": "one-off request",
                    "source_ids": ["human"],
                    "confidence": 0.99,
                    "scope": "session",
                    "durability": "temporary",
                    "authority": "descriptive",
                }
            ]
        }
    )
    result = merge_long_term(
        None, delta, user_sources={"human"}, thread_id="t", threshold=0.7, max_facts=100
    )
    assert result["user_context"] == {}


def test_bounds_and_required_model_configuration(settings):
    with pytest.raises(ValidationError):
        SessionMemory(goals=["x"] * 13)
    with pytest.raises(ValidationError):
        Settings(**{**settings.model_dump(), "MEMORY_MODEL": None}, _env_file=None)


def test_raw_archive_filters_internal_data_and_preserves_tools():
    source = [
        SystemMessage(content="secret system prompt"),
        HumanMessage(content="question", id="u"),
        AIMessage(content="summary", additional_kwargs={"memory_internal": True}),
        AIMessage(content="", tool_calls=[{"name": "run", "args": {}, "id": "c"}]),
        ToolMessage(content="session_id=s1", tool_call_id="c"),
    ]
    records = raw_records(source)
    assert len(records) == 3
    assert records[-1]["payload"]["content"] == "session_id=s1"
    assert complete_boundary(source)
    assert not complete_boundary(source[:-1])
    assert not complete_boundary([source[-1]])
    assert raw_records(source) == records


def test_oversized_evidence_is_split_without_loss():
    text = 'x\n"\\' * 5000
    record = raw_records([HumanMessage(content=text, id="large")])[0]
    parts = list(batches([record], 2048, "session"))
    assert len(parts) > 1
    assert "".join(p["text"] for b in parts for p in b) == text
    assert all(len(json.dumps(b, ensure_ascii=False)) <= 2050 for b in parts)
    assert {p["source_id"] for b in parts for p in b} == {record["event_id"]}


def test_extractor_filters_tool_binary_payload_and_old_memory_as_evidence(settings):
    records = raw_records(
        [
            HumanMessage(content="以后用中文", id="u"),
            AIMessage(content="", tool_calls=[{"name": "run", "args": {}, "id": "c"}]),
            ToolMessage(
                content=json.dumps(
                    {
                        "stdout": "s1",
                        "display_outputs": [{"type": "image/png", "data": "base64-payload"}],
                    }
                ),
                tool_call_id="c",
            ),
            AIMessage(content="final answer"),
        ]
    )

    class Model:
        def __init__(self):
            self.seen = []

        def invoke(self, messages, config):
            self.seen.append(messages)
            return AIMessage(content='{"runtime_facts":["session_id=s1"]}')

    model = Model()
    result = MemoryExtractor(settings, model).extract("session", {}, records, "t", lambda: None)
    assert result["runtime_facts"] == ["session_id=s1"]
    assert "base64-payload" not in model.seen[0][-1].content
    selected = [p for batch in batches(records, 4096, "long_term") for p in batch]
    assert [p["role"] for p in selected] == ["human", "ai"]
    assert records[2]["payload"]["content"].find("base64-payload") >= 0


@pytest.mark.parametrize("async_mode", [False, True])
def test_real_graph_hooks_leave_checkpoint_unchanged(settings, monkeypatch, async_mode):
    from tests.test_agent_refactor import ScriptedModel

    monkeypatch.setattr(factory, "get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.agent.configuration.get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.crypto.resolve_api_key", lambda: "unused")
    service = MagicMock()
    graph = factory.build_agent(
        manager=MagicMock(),
        llm=ScriptedModel(responses=[AIMessage(content="answer", id="a")]),
        checkpointer=InMemorySaver(),
        memory_service=service,
    )
    cfg = {"configurable": {"thread_id": "thread-a", "user_id": "untrusted"}}
    state = {"messages": [HumanMessage(content="以后用中文", id="u")]}
    if async_mode:
        result = asyncio.run(graph.ainvoke(state, cfg))
    else:
        result = graph.invoke(state, cfg)
    assert [m.content for m in result["messages"]] == ["以后用中文", "answer"]
    assert {c.args[0] for c in service.observe.call_args_list} == {"session", "long_term"}
    final = [
        c
        for c in service.observe.call_args_list
        if c.args[0] == "long_term" and c.kwargs["schedule"]
    ]
    assert len(final) == 1 and final[0].args[1][-1].content == "answer"
    assert all(c.args[2] == "thread-a" for c in service.observe.call_args_list)
    assert [m.content for m in graph.get_state(cfg).values["messages"]] == ["以后用中文", "answer"]


@pytest.mark.parametrize("cls", [SessionMemoryMiddleware, LongTermMemoryMiddleware])
def test_archive_failure_is_best_effort(cls):
    service = MagicMock()
    service.observe.side_effect = RuntimeError("database down")
    middleware = cls(service)
    middleware._observe({"messages": [HumanMessage(content="hi")]}, schedule=True, thread_id="t")


@pytest.fixture
def repo():
    dsn = os.environ.get("MEMORY_TEST_DSN")
    if not dsn:
        pytest.skip("Set MEMORY_TEST_DSN to run PostgreSQL memory integration tests")
    namespace = "memory_test_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(namespace)))
    pool = ConnectionPool(
        dsn,
        min_size=1,
        max_size=5,
        timeout=5,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "options": f"-c search_path={namespace}",
        },
        open=True,
    )
    repository = MemoryRepository(pool, standalone_owner="owner")
    repository.setup()
    try:
        yield repository
    finally:
        pool.close()
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(namespace)))


def queued(repo, thread="t", kind="long_term", text="以后用中文"):
    records = raw_records(
        [
            HumanMessage(content=text, id="user-" + thread),
            AIMessage(content="ok", id="answer-" + thread),
        ]
    )
    target_seq = repo.capture("owner", thread, records)
    repo.enqueue("owner", thread, kind, target_seq, 0, 300)
    return records, target_seq


def test_postgres_archive_and_atomic_idempotent_commit(repo):
    records, target_seq = queued(repo)
    assert repo.capture("owner", "t", records) == target_seq
    job = repo.claim(60)
    snapshot, doc = repo.load(job)
    loaded = [r for page in repo.pages(job, repo.load(job)[0], job["cursor_seq"]) for r in page]
    assert [r["event_id"] for r in loaded] == [r["event_id"] for r in records]
    assert repo.finish(job, {"facts": ["one"]}, doc["revision"], {})
    assert not repo.finish(job, {"facts": ["duplicate"]}, doc["revision"], {})
    assert repo.read("owner", "user")["body"] == {"facts": ["one"]}
    assert repo.claim(60) is None
    with repo.pool.connection() as conn:
        assert conn.execute("SELECT count(*) AS n FROM sam_memory_versions").fetchone()["n"] == 1
        row = conn.execute("SELECT cursor_seq,status FROM sam_memory_jobs").fetchone()
        assert row["status"] == "done" and row["cursor_seq"] == len(records)


def test_postgres_compaction_scope_and_history(repo):
    records, end = queued(repo, kind="session")
    job = repo.claim(60)
    assert repo.finish(job, {"goals": ["original"]}, 0, {})
    reference = repo.save_compaction(
        "owner", "t", {"summary": "compressed"}, {"covered_until_seq": end}, "compact-test",
    )
    assert reference == {"scope": "compaction:t", "revision": 1}
    assert repo.save_compaction("owner", "t", {}, {}, "compact-test") == reference
    assert repo.read("owner", "session:t")["body"] == {"goals": ["original"]}
    assert repo.read("owner", "compaction:t")["body"] == {"summary": "compressed"}
    assert repo.session_version("owner", "t", end)["revision"] == 1
    assert repo.session_version("owner", "t", end - 1) is None
    assert len(repo.history_page("owner", "t", 0, 1)) == 1
    assert repo.history_page("other", "t") == []
    ids = [r["payload"]["id"] for r in records]
    assert repo.message_sequences("owner", "t", ids) == dict(zip(ids, [1, 2]))
    with repo.pool.connection() as conn:
        row = conn.execute("SELECT cursor_seq,status FROM sam_memory_jobs").fetchone()
        assert row["cursor_seq"] == end and row["status"] == "done"


def test_postgres_claims_are_exclusive_and_expired_worker_cannot_commit(repo):
    records, _ = queued(repo)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: repo.claim(60), range(2)))
    old = next(c for c in claims if c)
    assert sum(c is not None for c in claims) == 1
    with repo.pool.connection() as conn:
        conn.execute("UPDATE sam_memory_jobs SET lease_until=now()-interval '1 second'")
    new = repo.claim(60)
    assert new["lease_token"] != old["lease_token"]
    with pytest.raises(LostLease):
        repo.heartbeat(old, 60)
    assert not repo.finish(old, {}, 0, {})
    assert repo.finish(new, {}, 0, {})


def test_postgres_concurrent_user_writes_recompute_against_new_revision(repo, settings):
    queued(repo, "a", text="以后用中文")
    queued(repo, "b", text="我是工程师")
    jobs = [repo.claim(60), repo.claim(60)]
    service = MemoryService(repo, settings)
    barrier = threading.Barrier(2)

    class Extractor:
        def __init__(self, synchronize):
            self.synchronize = synchronize

        def extract(self, kind, old, records, thread_id, heartbeat):
            if self.synchronize:
                barrier.wait(timeout=5)
            source = next(r["event_id"] for r in records if r["payload"]["type"] == "human")
            return merge_long_term(
                old,
                LongTermDelta(facts=[fact(source, content=thread_id)]),
                user_sources={source},
                thread_id=thread_id,
                threshold=0.7,
                max_facts=100,
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(service.process, j, Extractor(True)) for j in jobs]
        for future in futures:
            future.result(timeout=10)
    retry = repo.claim(60)
    assert retry is not None
    service.process(retry, Extractor(False))
    doc = repo.read("owner", "user")
    assert doc["revision"] == 2
    assert {f["content"] for f in doc["body"]["facts"]} == {"a", "b"}


def test_postgres_session_threshold_and_independent_consumers(repo, settings):
    settings.SESSION_MEMORY_INITIAL_TOKENS = 10000
    records, target_seq = queued(repo, kind="session")
    repo.enqueue("owner", "t", "long_term", target_seq, 0, 300)
    service = MemoryService(repo, settings)
    jobs = [repo.claim(60), repo.claim(60)]
    session = next(j for j in jobs if j["kind"] == "session")
    long_term = next(j for j in jobs if j["kind"] == "long_term")
    extractor = MagicMock()
    service.process(session, extractor)
    extractor.extract.assert_not_called()
    repo.fail(long_term, "TimeoutError", 1)
    with repo.pool.connection() as conn:
        rows = conn.execute("SELECT kind,status,cursor_seq FROM sam_memory_jobs").fetchall()
    assert {r["kind"]: r["status"] for r in rows} == {"session": "done", "long_term": "failed"}
    assert all(r["cursor_seq"] == 0 for r in rows)


def test_postgres_resolves_owner_from_server_thread_record(repo):
    repo.standalone_owner = None
    with repo.pool.connection() as conn:
        conn.execute('CREATE TABLE "thread" (thread_id text PRIMARY KEY,user_id text)')
        conn.execute('INSERT INTO "thread" VALUES (%s,%s)', ("t", "real-owner"))
    assert repo.owner_for_thread("t") == "real-owner"
    assert repo.owner_for_thread("missing") is None


def test_postgres_deleted_thread_cancels_inflight_extraction(repo):
    with repo.pool.connection() as conn:
        conn.execute('CREATE TABLE "thread" (thread_id text PRIMARY KEY,user_id text)')
        conn.execute('INSERT INTO "thread" VALUES (%s,%s)', ("t", "owner"))
    repo.standalone_owner = None
    records, _ = queued(repo)
    job = repo.claim(60)
    with repo.pool.connection() as conn:
        conn.execute('DELETE FROM "thread" WHERE thread_id=%s', ("t",))
    assert not repo.finish(job, {}, 0, {})
    assert repo.read("owner", "user")["revision"] == 0
    with repo.pool.connection() as conn:
        assert (
            conn.execute("SELECT status FROM sam_memory_jobs").fetchone()["status"] == "cancelled"
        )


def test_postgres_worker_recovers_pending_work_after_restart(repo, settings):
    queued(repo, kind="session")
    done = threading.Event()

    class Extractor:
        def __init__(self, _):
            pass

        def extract(self, kind, old, records, thread_id, heartbeat):
            done.set()
            return SessionMemory(goals=["remember"]).model_dump()

    service = MemoryService(repo, settings, Extractor)
    service.start()
    assert done.wait(timeout=5)
    service.close()
    # Closing may stop a result before commit, so the durable queue must retain it.
    with repo.pool.connection() as conn:
        job = conn.execute("SELECT status,cursor_seq FROM sam_memory_jobs").fetchone()
    assert job["status"] in ("pending", "done")
    assert not service._threads


def test_postgres_incremental_session(repo, settings):
    records, _ = queued(repo, kind="session")
    service = MemoryService(repo, settings)
    seen = []

    class Extractor:
        def extract(self, kind, old, delta, thread_id, heartbeat):
            seen.append((old, delta))
            return SessionMemory(goals=["goal"]).model_dump()

    service.process(repo.claim(60), Extractor())
    appended = raw_records(
        [HumanMessage(content="new input", id="u2"), AIMessage(content="new answer", id="a2")]
    )
    target_seq = repo.capture("owner", "t", records + appended)
    repo.enqueue("owner", "t", "session", target_seq, 0, 300)
    service.process(repo.claim(60), Extractor())
    assert len(seen[-1][1]) == 2
    assert seen[-1][0]["goals"] == ["goal"]
    assert repo.read("owner", "session:t")["revision"] == 2


def test_postgres_incomplete_new_boundary_does_not_busy_retry_old_job(repo):
    records, _ = queued(repo, kind="session")
    job = repo.claim(60)
    pending = raw_records(
        [AIMessage(content="", id="pending", tool_calls=[{"name": "run", "args": {}, "id": "c"}])]
    )
    repo.capture("owner", "t", records + pending)
    assert repo.finish(job, {}, 0, {})
    assert repo.claim(60) is None
    with repo.pool.connection() as conn:
        row = conn.execute("SELECT cursor_seq FROM sam_memory_jobs").fetchone()
        assert row["cursor_seq"] == 2


@pytest.mark.parametrize("async_mode", [False, True])
def test_blocked_small_model_does_not_block_graph(repo, settings, monkeypatch, async_mode):
    from tests.test_agent_refactor import ScriptedModel

    repo.standalone_owner = "owner"
    started, release = threading.Event(), threading.Event()

    class Extractor:
        def __init__(self, _):
            pass

        def extract(self, kind, old, records, thread_id, heartbeat):
            started.set()
            assert release.wait(timeout=5)
            return SessionMemory(goals=["goal"]).model_dump() if kind == "session" else old

    service = MemoryService(repo, settings, Extractor)
    monkeypatch.setattr(factory, "get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.agent.configuration.get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.crypto.resolve_api_key", lambda: "unused")
    graph = factory.build_agent(
        manager=MagicMock(),
        llm=ScriptedModel(responses=[AIMessage(content="returned while extractor blocked")]),
        checkpointer=InMemorySaver(),
        memory_service=service,
    )
    try:
        kwargs = (
            {"messages": [HumanMessage(content="remember this", id="u")]},
            {"configurable": {"thread_id": "t"}},
        )
        result = asyncio.run(graph.ainvoke(*kwargs)) if async_mode else graph.invoke(*kwargs)
        assert started.wait(timeout=5)
        assert result["messages"][-1].content == "returned while extractor blocked"
        assert repo.read("owner", "session:t")["revision"] == 0
    finally:
        release.set()
        service.close()


def test_postgres_debounce_is_bounded_and_duplicate_enqueue_does_not_extend(repo):
    records, target_seq = queued(repo)
    with repo.pool.connection() as conn:
        conn.execute("UPDATE sam_memory_jobs SET due_at=now()+interval '30 seconds'")
        before = conn.execute("SELECT due_at FROM sam_memory_jobs").fetchone()["due_at"]
    repo.enqueue("owner", "t", "long_term", target_seq, 30, 300)
    with repo.pool.connection() as conn:
        assert conn.execute("SELECT due_at FROM sam_memory_jobs").fetchone()["due_at"] == before
        conn.execute("UPDATE sam_memory_jobs SET queued_at=now()-interval '299 seconds'")
    target_seq = repo.capture(
        "owner", "t", records + raw_records([HumanMessage(content="another user turn", id="next")])
    )
    repo.enqueue("owner", "t", "long_term", target_seq, 30, 300)
    with repo.pool.connection() as conn:
        row = conn.execute(
            "SELECT extract(epoch FROM (due_at-now())) AS delay FROM sam_memory_jobs"
        ).fetchone()
    assert 0 < row["delay"] <= 1.1


def test_postgres_malformed_model_result_does_not_advance_cursor(repo, settings):
    records, _ = queued(repo, kind="session")
    job = repo.claim(60)
    assert repo.finish(job, SessionMemory(goals=["old memory"]).model_dump(), 0, {})
    records += raw_records([HumanMessage(content="new evidence", id="next")])
    target_seq = repo.capture("owner", "t", records)
    repo.enqueue("owner", "t", "session", target_seq, 0, 300)
    job = repo.claim(60)
    model = MagicMock()
    model.invoke.return_value = AIMessage(content='{"goals": "invalid schema"}')
    service = MemoryService(repo, settings)
    with pytest.raises(ValidationError):
        service.process(job, MemoryExtractor(settings, model))
    repo.fail(job, "ValidationError", 1)
    assert repo.read("owner", "session:t")["body"]["goals"] == ["old memory"]
    with repo.pool.connection() as conn:
        row = conn.execute("SELECT * FROM sam_memory_jobs").fetchone()
    assert row["status"] == "failed" and row["cursor_seq"] == 2


def test_postgres_owner_documents_and_consumers_are_isolated(repo):
    records, target_seq = queued(repo)
    second = repo.capture("other-owner", "t", records)
    repo.enqueue("other-owner", "t", "long_term", second, 0, 300)
    job = repo.claim(60, ("long_term",))
    assert job["owner_id"] == "owner"
    assert repo.finish(job, {"private": "owner"}, 0, {})
    assert repo.read("other-owner", "user")["revision"] == 0
    assert repo.claim(60, ("session",)) is None


def test_session_output_budget_and_extraction_config_are_enforced(settings):
    model = MagicMock()
    model.invoke.return_value = AIMessage(content=json.dumps({"goals": ["x" * 600] * 12}))
    settings.SESSION_MEMORY_MAX_CHARS = 1000
    extractor = MemoryExtractor(settings, model)
    with pytest.raises(ValueError, match="session_memory_budget_exceeded"):
        extractor.extract(
            "session", {}, raw_records([HumanMessage(content="work")]), "t", lambda: None
        )
    config = model.invoke.call_args.kwargs["config"]
    assert config["callbacks"] == [] and config["tags"] == ["memory-extraction"]


def test_http_lifespan_starts_and_closes_memory_service(settings, monkeypatch):
    from starlette.testclient import TestClient

    from taskharbor.http_app import app

    service = MagicMock()
    monkeypatch.setattr("taskharbor.settings.get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.agent.memory.service.get_memory_service", lambda: service)
    with TestClient(app):
        service.start.assert_called_once()
        service.close.assert_not_called()
    service.close.assert_called_once()


def test_postgres_long_history_storage_and_incremental_pages(repo, settings):
    records = raw_records([HumanMessage(content=f"message {i}", id=str(i)) for i in range(1025)])
    for count in range(25, 1025, 25):
        repo.capture("owner", "t", records[:count])
    target_seq = repo.capture("owner", "t", records)
    repo.enqueue("owner", "t", "session", target_seq, 0, 300)
    seen = []

    class Extractor:
        def extract(self, kind, old, page, thread_id, heartbeat):
            assert len(page) <= repo.PAGE_SIZE
            seen.extend(r["seq"] for r in page)
            return SessionMemory(goals=[str(page[-1]["seq"])]).model_dump()

    service = MemoryService(repo, settings)
    service.process(repo.claim(60), Extractor())
    assert seen == list(range(1, 1026))
    with repo.pool.connection() as conn:
        assert conn.execute("SELECT count(*) AS n FROM sam_memory_events").fetchone()["n"] == 1025
        assert conn.execute("SELECT count(*) AS n FROM sam_memory_heads").fetchone()["n"] == 1
        assert (
            conn.execute("SELECT to_regclass('sam_memory_snapshots') AS t").fetchone()["t"] is None
        )
        meta = conn.execute("SELECT metadata FROM sam_memory_versions").fetchone()["metadata"]
        assert meta["covered_range"] == {"end_seq": 1025}
        assert "covered_refs" not in meta
    seen.clear()
    records += raw_records([HumanMessage(content="only new message", id="next")])
    target_seq = repo.capture("owner", "t", records)
    repo.enqueue("owner", "t", "session", target_seq, 0, 300)
    service.process(repo.claim(60), Extractor())
    assert seen == [1026]


def test_postgres_later_page_failure_keeps_memory_and_cursor_atomic(repo, settings):
    records, _ = queued(repo, kind="session")
    job = repo.claim(60)
    old = SessionMemory(goals=["old"]).model_dump()
    assert repo.finish(job, old, 0, {})
    records += raw_records([HumanMessage(content=str(i), id=str(i)) for i in range(260)])
    target_seq = repo.capture("owner", "t", records)
    repo.enqueue("owner", "t", "session", target_seq, 0, 300)
    calls = []

    class Extractor:
        def extract(self, kind, body, page, thread_id, heartbeat):
            calls.append(page[0]["seq"])
            if len(calls) == 2:
                raise TimeoutError()
            return SessionMemory(goals=["partial"]).model_dump()

    job = repo.claim(60)
    with pytest.raises(TimeoutError):
        MemoryService(repo, settings).process(job, Extractor())
    repo.fail(job, "TimeoutError", 1)
    assert calls == [3, 131]
    assert repo.read("owner", "session:t")["body"] == old
    with repo.pool.connection() as conn:
        row = conn.execute("SELECT cursor_seq,status FROM sam_memory_jobs").fetchone()
        assert row == {"cursor_seq": 2, "status": "failed"}
        assert conn.execute("SELECT count(*) AS n FROM sam_memory_versions").fetchone()["n"] == 1


@pytest.mark.parametrize("body", [None, {"goals": ["done"]}])
def test_postgres_new_target_during_extraction_remains_pending(repo, body):
    records, _ = queued(repo, kind="session")
    job = repo.claim(60)
    records += raw_records([HumanMessage(content="next", id="next")])
    end = repo.capture("owner", "t", records)
    repo.enqueue("owner", "t", "session", end, 0, 300)
    assert [
        r["seq"] for page in repo.pages(job, repo.load(job)[0], job["cursor_seq"]) for r in page
    ] == [1, 2]
    assert repo.finish(job, body, 0, {})
    next_job = repo.claim(60)
    assert next_job["target_seq"] == 3
    assert next_job["cursor_seq"] == (2 if body is not None else 0)
    assert [
        r["seq"]
        for page in repo.pages(next_job, repo.load(next_job)[0], next_job["cursor_seq"])
        for r in page
    ] == ([3] if body is not None else [1, 2, 3])


def test_postgres_concurrent_capture_deduplicates_message_ids(repo):
    records = raw_records([HumanMessage(content="same content", id="same")]) * 3
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: repo.capture("owner", "t", records), range(2))) == [1, 1]
    repo.enqueue("owner", "t", "session", 1, 0, 300)
    job = repo.claim(60)
    assert job["target_seq"] == 1
    loaded = [r for page in repo.pages(job, repo.load(job)[0], job["cursor_seq"]) for r in page]
    assert [r["seq"] for r in loaded] == [1]
    assert len({r["event_id"] for r in loaded}) == 1
    with repo.pool.connection() as conn:
        assert conn.execute("SELECT end_seq FROM sam_memory_heads").fetchone()["end_seq"] == 1
        assert conn.execute("SELECT count(*) AS n FROM sam_memory_events").fetchone()["n"] == 1


def seed_previous_schema(repo, version):
    """Replace only empty tables in the fixture's disposable schema with v1/v2 tables."""
    records = raw_records([HumanMessage(content=str(i), id=str(i)) for i in range(4)])
    refs = [r["event_id"] for r in records]
    with repo.pool.connection() as conn, conn.transaction():
        for table in ("jobs", "heads", "events"):
            assert (
                conn.execute(f"SELECT count(*) AS n FROM sam_memory_{table}").fetchone()["n"] == 0
            )
            conn.execute(f"DROP TABLE sam_memory_{table}")
        conn.execute("""CREATE TABLE sam_memory_events (
            owner_id text,thread_id text,event_id text,payload jsonb,
            created_at timestamptz DEFAULT now(),PRIMARY KEY(owner_id,thread_id,event_id))""")
        for r in records:
            conn.execute(
                "INSERT INTO sam_memory_events (owner_id,thread_id,event_id,payload) "
                "VALUES ('owner','t',%s,%s)",
                (r["event_id"], Jsonb(r["payload"])),
            )
        extra = "refs jsonb" if version == "refs" else "generation bigint,end_seq bigint"
        for table in ("heads", "snapshots"):
            conn.execute(
                f"CREATE TABLE sam_memory_{table} "
                f"(owner_id text,thread_id text,revision bigint,{extra})"
            )
        conn.execute("""CREATE TABLE sam_memory_jobs (
            owner_id text,thread_id text,kind text,desired_revision bigint,done_revision bigint,
            status text DEFAULT 'pending',due_at timestamptz DEFAULT now(),
            queued_at timestamptz DEFAULT now(),lease_token text,lease_until timestamptz,
            attempts integer DEFAULT 0,last_error text,
            PRIMARY KEY (owner_id,thread_id,kind))""")
        conn.execute("CREATE INDEX sam_memory_jobs_ready ON sam_memory_jobs(due_at)")
        if version == "refs":
            conn.execute("ALTER TABLE sam_memory_jobs ADD COLUMN cursor_refs jsonb DEFAULT '[]'")
            conn.execute("INSERT INTO sam_memory_heads VALUES ('owner','t',2,%s)", (Jsonb(refs),))
            for rev, end in ((1, 2), (2, 4)):
                conn.execute(
                    "INSERT INTO sam_memory_snapshots VALUES ('owner','t',%s,%s)",
                    (rev, Jsonb(refs[:end])),
                )
        else:
            conn.execute(
                "ALTER TABLE sam_memory_jobs ADD COLUMN cursor_generation bigint DEFAULT 1, "
                "ADD COLUMN cursor_seq bigint DEFAULT 2"
            )
            conn.execute(
                "CREATE TABLE sam_memory_positions "
                "(owner_id text,thread_id text,generation bigint,seq bigint,event_id text)"
            )
            conn.execute("INSERT INTO sam_memory_heads VALUES ('owner','t',2,1,4)")
            for rev, end in ((1, 2), (2, 4)):
                conn.execute(
                    "INSERT INTO sam_memory_snapshots VALUES ('owner','t',%s,1,%s)", (rev, end)
                )
            for seq, ref in enumerate(refs, 1):
                conn.execute(
                    "INSERT INTO sam_memory_positions VALUES ('owner','t',1,%s,%s)", (seq, ref)
                )
        if version == "ranges":
            conn.execute("ALTER TABLE sam_memory_heads ADD COLUMN history_hash text DEFAULT ''")
            conn.execute("UPDATE sam_memory_heads SET history_hash=%s", ("old-history-hash",))
        conn.execute(
            "INSERT INTO sam_memory_jobs "
            "(owner_id,thread_id,kind,desired_revision,done_revision,status,lease_token) "
            "VALUES ('owner','t','session',2,1,'running','old-lease')"
        )
        conn.execute("UPDATE sam_memory_jobs SET lease_until=now()-interval '1 second'")
        if version == "refs":
            conn.execute("UPDATE sam_memory_jobs SET cursor_refs=%s", (Jsonb(refs[:2]),))
        conn.execute(
            "INSERT INTO sam_memory_documents (owner_id,scope,revision,body) "
            "VALUES ('owner','session:t',1,%s)",
            (Jsonb(SessionMemory(goals=["existing"]).model_dump()),),
        )
        conn.execute(
            "INSERT INTO sam_memory_versions "
            "(owner_id,scope,revision,extraction_id,body,metadata) "
            "VALUES ('owner','session:t',1,'old','{}',%s)",
            (Jsonb({"history_revision": 1}),),
        )
    return records


@pytest.mark.parametrize("version", ["refs", "ranges"])
def test_postgres_upgrade_preserves_archive_memory_cursor_and_pending_work(repo, version):
    records = seed_previous_schema(repo, version)
    repo.setup()
    repo.setup()
    job = repo.claim(60)
    assert job["target_seq"] == 4 and job["cursor_seq"] == 2
    assert job["lease_token"] != "old-lease"
    assert "cursor_generation" not in job and "desired_revision" not in job
    assert [
        r["event_id"]
        for page in repo.pages(job, repo.load(job)[0], job["cursor_seq"])
        for r in page
    ] == [r["event_id"] for r in records[2:]]
    assert repo.load(job)[1]["body"]["goals"] == ["existing"]
    assert repo.finish(job, SessionMemory(goals=["updated"]).model_dump(), 1, {})
    with repo.pool.connection() as conn:
        assert conn.execute("SELECT count(*) AS n FROM sam_memory_events").fetchone()["n"] == 4
        assert conn.execute("SELECT count(*) AS n FROM sam_memory_versions").fetchone()["n"] == 2
        assert (
            conn.execute("SELECT to_regclass('sam_memory_snapshots') AS t").fetchone()["t"] is None
        )
        assert (
            conn.execute(
                "SELECT tablename FROM pg_indexes WHERE schemaname=current_schema() "
                "AND indexname='sam_memory_jobs_ready'"
            ).fetchone()["tablename"]
            == "sam_memory_jobs"
        )


def test_postgres_upgrade_failure_rolls_back_and_preserves_old_tables(repo):
    records = seed_previous_schema(repo, "refs")
    with repo.pool.connection() as conn:
        conn.execute("DELETE FROM sam_memory_events WHERE event_id=%s", (records[2]["event_id"],))
    with pytest.raises(ValueError, match="legacy_memory_events_missing"):
        repo.setup()
    with repo.pool.connection() as conn:
        assert "refs" in conn.execute("SELECT * FROM sam_memory_heads").fetchone()
        assert (
            conn.execute("SELECT to_regclass('sam_memory_events_pre_append_v4') AS t").fetchone()[
                "t"
            ]
            is None
        )
        conn.execute(
            "INSERT INTO sam_memory_events (owner_id,thread_id,event_id,payload) "
            "VALUES ('owner','t',%s,%s)",
            (records[2]["event_id"], Jsonb(records[2]["payload"])),
        )
    repo.setup()


def test_independent_confirmation_keeps_sources_from_both_threads():
    confirmed = merge_long_term(
        merge([fact()]),
        LongTermDelta(facts=[fact("other-human")]),
        user_sources={"other-human"},
        thread_id="other",
        threshold=0.7,
        max_facts=100,
    )
    assert len(confirmed["facts"]) == 1
    assert {s["source_thread_id"] for s in confirmed["facts"][0]["sources"]} == {"t", "other"}


def test_postgres_session_threshold_keeps_unconsumed_messages_for_next_job(repo, settings):
    settings.SESSION_MEMORY_INITIAL_TOKENS = 4
    records, _ = queued(repo, kind="session", text="a")
    service = MemoryService(repo, settings)
    extractor = MagicMock()
    service.process(repo.claim(60), extractor)
    extractor.extract.assert_not_called()
    records += raw_records([HumanMessage(content="x" * 20, id="next")])
    target_seq = repo.capture("owner", "t", records)
    repo.enqueue("owner", "t", "session", target_seq, 0, 300)
    extractor.extract.return_value = SessionMemory(goals=["all evidence"]).model_dump()
    job = repo.claim(60)
    assert job["cursor_seq"] == 0
    service.process(job, extractor)
    assert [r["seq"] for r in extractor.extract.call_args.args[2]] == [1, 2, 3]
    with repo.pool.connection() as conn:
        assert conn.execute("SELECT cursor_seq FROM sam_memory_jobs").fetchone()["cursor_seq"] == 3


def seed_append_only_schema(repo):
    """Populate the simplified version's actual schema, only inside the test schema."""
    records = raw_records([HumanMessage(content=str(i), id=str(i)) for i in range(3)])
    records.append(records[0])  # The sequence must preserve repeated occurrences.
    with repo.pool.connection() as conn, conn.transaction():
        for table in ("jobs", "heads", "events"):
            assert (
                conn.execute(f"SELECT count(*) AS n FROM sam_memory_{table}").fetchone()["n"] == 0
            )
            conn.execute(f"DROP TABLE sam_memory_{table}")
        conn.execute("""CREATE TABLE sam_memory_events (
            owner_id text,thread_id text,seq bigint,event_id text,payload jsonb,
            created_at timestamptz DEFAULT now(),PRIMARY KEY(owner_id,thread_id,seq))""")
        conn.execute("""CREATE TABLE sam_memory_heads (
            owner_id text,thread_id text,last_seq bigint,PRIMARY KEY(owner_id,thread_id))""")
        conn.execute("""CREATE TABLE sam_memory_jobs (
            owner_id text,thread_id text,kind text,target_seq bigint,cursor_seq bigint,
            status text,due_at timestamptz DEFAULT now(),queued_at timestamptz DEFAULT now(),
            lease_token text,lease_until timestamptz,attempts integer DEFAULT 0,last_error text,
            PRIMARY KEY(owner_id,thread_id,kind))""")
        conn.execute("CREATE INDEX sam_memory_jobs_ready ON sam_memory_jobs(due_at)")
        for seq, record in enumerate(records, 1):
            conn.execute(
                "INSERT INTO sam_memory_events "
                "(owner_id,thread_id,seq,event_id,payload) VALUES ('owner','t',%s,%s,%s)",
                (seq, record["event_id"], Jsonb(record["payload"])),
            )
        conn.execute("INSERT INTO sam_memory_heads VALUES ('owner','t',4)")
        conn.execute(
            "INSERT INTO sam_memory_jobs "
            "(owner_id,thread_id,kind,target_seq,cursor_seq,status,lease_token) "
            "VALUES ('owner','t','session',4,2,'running','old-token')"
        )
        conn.execute(
            "INSERT INTO sam_memory_documents (owner_id,scope,revision,body) "
            "VALUES ('owner','session:t',1,%s)",
            (Jsonb(SessionMemory(goals=["existing"]).model_dump()),),
        )
        conn.execute(
            "INSERT INTO sam_memory_versions "
            "(owner_id,scope,revision,extraction_id,body,metadata) "
            "VALUES ('owner','session:t',1,'old','{}','{\"schema_version\":3}')"
        )
        # A previous upgrade may already have retained legacy tables; never overwrite them.
        conn.execute("CREATE TABLE sam_memory_heads_legacy (marker text)")
        conn.execute("INSERT INTO sam_memory_heads_legacy VALUES ('preserve')")
    return records


def test_upgrade_from_simplified_schema_deduplicates_without_losing_cursor(repo):
    records = seed_append_only_schema(repo)
    repo.setup()
    repo.setup()
    job = repo.claim(60)
    target, doc = repo.load(job)
    assert target == {"end_seq": 3}
    assert job["cursor_seq"] == 2
    assert job["lease_token"] != "old-token"
    assert doc["body"]["goals"] == ["existing"]
    assert [r["event_id"] for page in repo.pages(job, target, 2) for r in page] == [
        records[2]["event_id"]
    ]
    assert repo.capture("owner", "t", records) == 3
    assert repo.finish(job, SessionMemory(goals=["updated"]).model_dump(), 1, {})
    with repo.pool.connection() as conn:
        assert (
            conn.execute("SELECT count(*) AS n FROM sam_memory_events_pre_append_v4").fetchone()[
                "n"
            ]
            == 4
        )
        assert (
            conn.execute("SELECT marker FROM sam_memory_heads_legacy").fetchone()["marker"]
            == "preserve"
        )
        assert conn.execute("SELECT count(*) AS n FROM sam_memory_versions").fetchone()["n"] == 2


def test_restore_from_simplified_schema_rolls_back_on_missing_original(repo):
    records = seed_append_only_schema(repo)
    with repo.pool.connection() as conn:
        conn.execute("DELETE FROM sam_memory_events WHERE seq=3")
    with pytest.raises(ValueError, match="legacy_memory_events_missing"):
        repo.setup()
    with repo.pool.connection() as conn:
        assert "last_seq" in conn.execute("SELECT * FROM sam_memory_heads").fetchone()
        assert (
            conn.execute("SELECT to_regclass('sam_memory_events_pre_append_v4') AS t").fetchone()[
                "t"
            ]
            is None
        )
        conn.execute(
            "INSERT INTO sam_memory_events "
            "(owner_id,thread_id,seq,event_id,payload) VALUES ('owner','t',3,%s,%s)",
            (records[2]["event_id"], Jsonb(records[2]["payload"])),
        )
    repo.setup()


@pytest.mark.parametrize("extract_before_compaction", [False, True])
@pytest.mark.parametrize("async_mode", [False, True])
def test_real_graph_compaction_keeps_archive_and_incremental_memory(
    repo, settings, monkeypatch, extract_before_compaction, async_mode
):
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    from tests.test_agent_refactor import ScriptedModel

    monkeypatch.setattr(factory, "get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.agent.configuration.get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.crypto.resolve_api_key", lambda: "unused")
    service = MemoryService(repo, settings)
    monkeypatch.setattr(service, "start", lambda: None)
    graph = factory.build_agent(
        manager=MagicMock(),
        llm=ScriptedModel(
            responses=[AIMessage(content=f"answer-{i}", id=f"a{i}") for i in range(3)]
        ),
        checkpointer=InMemorySaver(),
        memory_service=service,
    )
    cfg = {"configurable": {"thread_id": "t"}}

    def invoke(text, identity):
        state = {"messages": [HumanMessage(content=text, id=identity)]}
        return asyncio.run(graph.ainvoke(state, cfg)) if async_mode else graph.invoke(state, cfg)

    invoke("first original", "u0")
    invoke("second original", "u1")
    originals = graph.get_state(cfg).values["messages"]
    assert len(originals) == 4
    old_source = raw_records(originals)[0]["event_id"]
    old_long = merge_long_term(
        {},
        LongTermDelta(facts=[fact(old_source)]),
        user_sources={old_source},
        thread_id="t",
        threshold=0.7,
        max_facts=100,
    )
    if extract_before_compaction:
        for kind, body in (
            ("session", SessionMemory(goals=["existing goal"]).model_dump()),
            ("long_term", old_long),
        ):
            assert repo.finish(repo.claim(60, (kind,)), body, 0, {})
    summary = HumanMessage(
        content="compressed context", id="summary", additional_kwargs={"memory_internal": True}
    )
    graph.update_state(
        cfg,
        {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), summary, *originals[-2:]]},
        as_node="model",
    )
    invoke("after compaction", "u2")
    current = graph.get_state(cfg).values["messages"]
    assert len(current) == 5 and current[0].id == "summary"
    expected = raw_records(originals + current[-2:])
    for kind in ("session", "long_term"):
        job = repo.claim(60, (kind,))
        assert job["target_seq"] == 6
        assert job["cursor_seq"] == (4 if extract_before_compaction else 0)
        extractor = MagicMock()
        extractor.extract.side_effect = lambda kind, body, page, thread, heartbeat: body
        service.process(job, extractor)
        body, page = extractor.extract.call_args.args[1:3]
        assert [r["event_id"] for r in page] == [
            r["event_id"] for r in expected[job["cursor_seq"] :]
        ]
        if extract_before_compaction:
            if kind == "session":
                assert body["goals"] == ["existing goal"]
            else:
                assert body["facts"][0]["source_ids"] == [old_source]
        assert [r["event_id"] for page in repo.pages(job, {"end_seq": 6}) for r in page] == [
            r["event_id"] for r in expected
        ]
    assert graph.get_state(cfg).values["messages"] == current
    # Empty/short state and repeated observation do not erase history or requeue completed work.
    service.observe("session", [], "t", schedule=True)
    service.observe("session", current, "t", schedule=True)
    assert repo.claim(60) is None
    with repo.pool.connection() as conn:
        assert conn.execute("SELECT count(*) AS n FROM sam_memory_events").fetchone()["n"] == 6
        assert all(
            r["cursor_seq"] == 6
            for r in conn.execute("SELECT cursor_seq FROM sam_memory_jobs").fetchall()
        )


def test_identity_dedup_ignores_metadata_and_retains_distinct_equal_text(repo):
    first = HumanMessage(content="same text", id="u1")
    second = HumanMessage(content="same text", id="u2")
    assert repo.capture("owner", "t", raw_records([first, second])) == 2
    first.response_metadata["latency"] = 123
    assert repo.capture("owner", "t", raw_records([first])) == 2
    # ID-less observations have a deterministic content-identity fallback.
    no_id = HumanMessage(content="legacy without id")
    assert repo.capture("owner", "t", raw_records([no_id, no_id])) == 3
    # No editing support: an existing ID cannot rewrite archived evidence.
    assert repo.capture("owner", "t", raw_records([HumanMessage(content="changed", id="u1")])) == 3
    with repo.pool.connection() as conn:
        assert (
            conn.execute("SELECT payload FROM sam_memory_events WHERE seq=1").fetchone()["payload"][
                "content"
            ]
            == "same text"
        )


def test_raw_records_skips_partial_chunks_and_internal_summary():
    assert (
        raw_records(
            [
                AIMessageChunk(content="partial", id="a"),
                HumanMessage(
                    content="summary", id="s", additional_kwargs={"memory_internal": True}
                ),
                HumanMessage(
                    content="worker result", id="worker",
                    additional_kwargs={"lc_source": "async_subagent"},
                ),
                HumanMessage(
                    content="legacy worker event", id="legacy-worker",
                    additional_kwargs={"deepagents_notification": {"task_id": "child"}},
                ),
            ]
        )
        == []
    )
    assert len(raw_records([AIMessage(content="complete", id="a")])) == 1


def test_concurrent_appends_allocate_contiguous_sequences(repo):
    shared = raw_records([HumanMessage(content="shared", id="shared")])

    def append(i):
        repo.capture("owner", "t", shared + raw_records([HumanMessage(content=str(i), id=str(i))]))

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(append, range(12)))
    with repo.pool.connection() as conn:
        rows = conn.execute("SELECT seq,message_key FROM sam_memory_events ORDER BY seq").fetchall()
    assert [r["seq"] for r in rows] == list(range(1, 14))
    assert len({r["message_key"] for r in rows}) == 13


def test_upgrade_refuses_unsettled_edit_without_modifying_old_tables(repo):
    seed_previous_schema(repo, "ranges")
    with repo.pool.connection() as conn:
        conn.execute("UPDATE sam_memory_jobs SET cursor_generation=99")
    with pytest.raises(ValueError, match="legacy_memory_unsettled_edit"):
        repo.setup()
    with repo.pool.connection() as conn:
        assert conn.execute("SELECT generation FROM sam_memory_heads").fetchone()["generation"] == 1
        assert (
            conn.execute("SELECT to_regclass('sam_memory_events_pre_append_v4') AS t").fetchone()[
                "t"
            ]
            is None
        )


def test_upgrade_preserves_old_branch_but_only_imports_current_history(repo):
    records = seed_previous_schema(repo, "ranges")
    replacement = raw_records([HumanMessage(content="current branch", id="replacement")])[0]
    with repo.pool.connection() as conn, conn.transaction():
        conn.execute(
            "INSERT INTO sam_memory_events (owner_id,thread_id,event_id,payload) "
            "VALUES ('owner','t',%s,%s)",
            (replacement["event_id"], Jsonb(replacement["payload"])),
        )
        conn.execute("UPDATE sam_memory_heads SET revision=3,generation=3,end_seq=3")
        conn.execute("INSERT INTO sam_memory_snapshots VALUES ('owner','t',3,3,3)")
        for seq, record in enumerate([*records[:2], replacement], 1):
            conn.execute(
                "INSERT INTO sam_memory_positions VALUES ('owner','t',3,%s,%s)",
                (seq, record["event_id"]),
            )
        # The old edit-aware worker already consumed the current branch.
        conn.execute(
            "UPDATE sam_memory_jobs SET desired_revision=3,done_revision=3,"
            "cursor_generation=3,cursor_seq=3,status='done'"
        )
    repo.setup()
    with repo.pool.connection() as conn:
        events = conn.execute("SELECT event_id FROM sam_memory_events ORDER BY seq").fetchall()
        assert [r["event_id"] for r in events] == [
            records[0]["event_id"],
            records[1]["event_id"],
            replacement["event_id"],
        ]
        assert conn.execute("SELECT cursor_seq FROM sam_memory_jobs").fetchone()["cursor_seq"] == 3
        assert (
            conn.execute("SELECT count(*) AS n FROM sam_memory_events_pre_append_v4").fetchone()[
                "n"
            ]
            == 5
        )
