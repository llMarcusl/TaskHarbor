"""Aegra's HTTP Agent Protocol adapter for Deep Agents async subagents.

The five tools and their task state belong to Deep Agents 0.7.38. This module
adapts its standard SDK transport to Aegra 0.6 and authenticates each caller with
a capability restricted to the launching user's parent thread.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)
_HEADER = "x-sandbox-subagent-capability"
_PREFIX = "/internal/subagents"
_TERMINAL = {"success", "error", "timeout", "interrupted"}
_RUN_PATH = re.compile(r"^/threads/([^/]+)/runs(?:/(?:stream|wait))?$")
_THREAD_PATH = re.compile(r"^/threads/([^/]+)$")
_WORKER_METADATA = {
    "sandbox_worker",
    "sandbox_relationship",
    "sandbox_parent_turn",
    "sandbox_task_deadline",
    "parent_thread_id",
    "subagent_name",
}
_MODEL_CONFIG = {
    "chat_model",
    "chat_model_provider",
    "chat_model_api_key",
    "chat_model_base_url",
    "chat_model_supports_vision",
}
_service_loop = None
_background_tasks = set()


@lru_cache(maxsize=8)
def _signing_key(storage_dir: str, encryption_key: str) -> bytes:
    if encryption_key:
        return hashlib.sha256(("sandbox-subagents-v1:" + encryption_key).encode()).digest()
    root = Path(storage_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / ".subagent-signing-key"
    if not target.exists():
        descriptor, temporary = tempfile.mkstemp(prefix=".subagent-key-", dir=root)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(secrets.token_bytes(32))
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                pass
        finally:
            Path(temporary).unlink(missing_ok=True)
    key = target.read_bytes()
    if len(key) != 32:
        raise RuntimeError("Invalid subagent signing key")
    return key


def _key(settings=None) -> bytes:
    if settings is None:
        from taskharbor.settings import get_settings

        settings = get_settings()
    return _signing_key(str(settings.STORAGE_DIR), settings.ENCRYPTION_KEY)


def sign_payload(payload: dict[str, Any], *, purpose: str, settings=None) -> str:
    """Sign a service-issued capability or file manifest with a purpose boundary."""
    encoded = (
        base64.urlsafe_b64encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        .decode()
        .rstrip("=")
    )
    signature = hmac.new(_key(settings), f"{purpose}.{encoded}".encode(), hashlib.sha256)
    return f"{encoded}.{signature.hexdigest()}"


def verify_payload(token: str, *, purpose: str, settings=None) -> dict[str, Any]:
    try:
        encoded, signature = token.rsplit(".", 1)
        expected = hmac.new(
            _key(settings), f"{purpose}.{encoded}".encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("Invalid signature")
        payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        if not isinstance(payload, dict):
            raise ValueError("Invalid payload")
        return payload
    except (ValueError, TypeError, UnicodeError) as exc:
        raise PermissionError("Invalid subagent capability") from exc


class _RuntimeHeaders(Mapping[str, str]):
    """Resolve public SDK headers within the current LangGraph invocation.

    Deep Agents calls ``dict(spec['headers'])`` before choosing its cached client.
    A Mapping keeps clients isolated by the authenticated parent without changing
    Deep Agents' tools, client cache or SDK implementation.
    """

    def __init__(self, manager, settings):
        self.manager = manager
        self.settings = settings

    def __iter__(self):
        return iter((_HEADER,))

    def __len__(self):
        return 1

    def __getitem__(self, name):
        if name != _HEADER:
            raise KeyError(name)
        from langgraph.config import get_config

        config = get_config()
        thread = str((config.get("configurable") or {}).get("thread_id") or "")
        if not thread:
            raise PermissionError("Async tasks require a persisted parent thread")
        owner = self.manager.resolve_owner(thread)
        from aegra_api.core.auth_ctx import get_auth_ctx

        auth = get_auth_ctx()
        if auth is not None and auth.user.identity != owner:
            raise PermissionError("Parent thread does not belong to the authenticated user")
        return sign_payload(
            {"owner": owner, "parent_thread_id": thread},
            purpose="protocol",
            settings=self.settings,
        )


def async_subagent_spec(manager, settings) -> dict[str, Any]:
    return {
        "name": "taskharbor-worker",
        "description": "Execute a bounded task in an independent sandbox and return its result.",
        "graph_id": "taskharbor-worker",
        "url": getattr(settings, "SUBAGENT_PROTOCOL_URL", None)
        or settings.API_BASE_URL.rstrip("/") + _PREFIX,
        "headers": _RuntimeHeaders(manager, settings),
        "mode": "isolated",
        "notify_parent": False,
    }


def _relationship_payload(owner: str, parent: str, child: str) -> dict[str, str]:
    return {"owner": owner, "parent": parent, "child": child, "agent": "taskharbor-worker"}


def trusted_parent(owner: str, thread_id: str, metadata: dict) -> str | None:
    """Verify a relationship even when public thread metadata was user-edited."""
    parent = metadata.get("parent_thread_id")
    token = metadata.get("sandbox_relationship")
    if metadata.get("sandbox_worker") is not True or not isinstance(parent, str):
        return None
    if not isinstance(token, str):
        return None
    try:
        payload = verify_payload(token, purpose="relationship")
    except PermissionError:
        return None
    return parent if payload == _relationship_payload(owner, parent, thread_id) else None


def _maker():
    from aegra_api.core.orm import _get_session_maker

    return _get_session_maker()


async def _owned_thread(session, owner: str, thread_id: str, *, lock=False):
    from aegra_api.core.orm import Thread
    from sqlalchemy import select

    statement = select(Thread).where(Thread.thread_id == thread_id, Thread.user_id == owner)
    if lock:
        statement = statement.with_for_update()
    row = await session.scalar(statement)
    if row is None:
        raise HTTPException(404, "Thread not found")
    return row


async def worker_parent(owner: str, thread_id: str) -> str | None:
    async with _maker()() as session:
        child = await _owned_thread(session, owner, thread_id)
        parent = trusted_parent(owner, thread_id, child.metadata_json or {})
        if parent:
            await _owned_thread(session, owner, parent)
        return parent


def _model_config(run) -> dict:
    configurable = (run.config or {}).get("configurable") or {}
    context = run.context or {}
    return {
        key: value for key, value in {**configurable, **context}.items() if key in _MODEL_CONFIG
    }


def _cancel_worker_approvals(owner, thread_id):
    from taskharbor.security.repository import get_approval_store

    store = get_approval_store()
    store.cancel_thread(owner, thread_id)
    scopes = {
        (row.get("run_id"), row.get("task_id"))
        for row in store.list_thread(owner, thread_id)
        if row.get("result") is None
        and row["status"] in {"cancelled", "expired", "denied", "superseded"}
    }
    for run_id, task_id in scopes:
        store.cancel_task(owner, thread_id, run_id, task_id)


class WorkerResumeConfigMiddleware(BaseHTTPMiddleware):
    """Preserve service-owned relationships and model settings through resumes."""

    async def dispatch(self, request, call_next):
        match = _RUN_PATH.match(request.url.path) if request.method == "POST" else None
        thread_match = _THREAD_PATH.match(request.url.path) if request.method == "PATCH" else None
        create = request.method == "POST" and request.url.path == "/threads"
        if not (match or thread_match or create):
            return await call_next(request)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return await call_next(request)
        if not isinstance(body, dict):
            return await call_next(request)
        metadata = body.get("metadata") or {}
        if (thread_match or create) and isinstance(metadata, dict):
            reserved = metadata.keys() & _WORKER_METADATA
            if reserved:
                if create:
                    return JSONResponse(
                        {"detail": "Worker relationships are service-managed"}, status_code=403
                    )
                from taskharbor.http_app import _authorized_thread_owner

                try:
                    thread_id = thread_match.group(1)
                    owner = await _authorized_thread_owner(request, thread_id)
                    async with _maker()() as session:
                        row = await _owned_thread(session, owner, thread_id)
                        current = row.metadata_json or {}
                        if not trusted_parent(owner, thread_id, current) or any(
                            metadata[key] != current.get(key) for key in reserved
                        ):
                            raise HTTPException(403, "Worker relationships are service-managed")
                except HTTPException as exc:
                    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        if not match:
            return await call_next(request)
        command = body.get("command") or {}
        if not isinstance(command, dict) or not command.get("resume"):
            return await call_next(request)
        config = body.get("config") or {}
        if not isinstance(config, dict) or not isinstance(config.get("configurable") or {}, dict):
            return await call_next(request)
        from aegra_api.core.orm import Run
        from sqlalchemy import select

        from taskharbor.http_app import _authorized_thread_owner

        try:
            thread_id = match.group(1)
            owner = await _authorized_thread_owner(request, thread_id)
            async with _maker()() as session:
                thread = await _owned_thread(session, owner, thread_id)
                if trusted_parent(owner, thread_id, thread.metadata_json or {}):
                    previous = await session.scalar(
                        select(Run)
                        .where(Run.thread_id == thread_id, Run.user_id == owner)
                        .order_by(Run.created_at.desc())
                        .limit(1)
                    )
                    if previous:
                        config["configurable"] = {
                            **_model_config(previous),
                            **(config.get("configurable") or {}),
                        }
                        from taskharbor.settings import get_settings

                        config["recursion_limit"] = get_settings().SUBAGENT_MAX_STEPS
                        body["config"] = config
                        request._body = json.dumps(body).encode()
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        return await call_next(request)


async def related_threads(owner: str, parent_thread: str) -> list[str]:
    from aegra_api.core.orm import Thread
    from sqlalchemy import select

    async with _maker()() as session:
        await _owned_thread(session, owner, parent_thread)
        rows = await session.scalars(
            select(Thread).where(
                Thread.user_id == owner,
                Thread.metadata_json["parent_thread_id"].astext == parent_thread,
            )
        )
        return [
            row.thread_id
            for row in rows
            if trusted_parent(owner, row.thread_id, row.metadata_json or {}) == parent_thread
        ]


async def _state(session, owner: str, thread_id: str) -> dict[str, Any]:
    from aegra_api.api.threads import get_thread_state
    from aegra_api.models import User

    state = await get_thread_state(
        thread_id, subgraphs=True, checkpoint_ns=None, user=User(identity=owner), session=session
    )
    return state.model_dump(mode="json")


def flatten_interrupts(state: dict) -> list[dict]:
    """Flatten Aegra's current and nested task interrupts, preserving their IDs."""
    result = {}

    def visit(value):
        if isinstance(value, dict):
            if "id" in value and "value" in value:
                result[str(value["id"])] = value
            for key in ("interrupts", "tasks", "state"):
                visit(value.get(key))
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(state)
    return list(result.values())


async def thread_interrupts(owner: str, thread_id: str) -> list[dict]:
    async with _maker()() as session:
        await _owned_thread(session, owner, thread_id)
        return flatten_interrupts(await _state(session, owner, thread_id))


def normalize_run(run, thread, *, latest_run_id=None, has_interrupts=False) -> dict[str, Any]:
    """Aegra marks checkpoint pauses interrupted, Agent Server marks them success."""
    result = run.model_dump(mode="json")
    latest = latest_run_id is None or result.get("run_id") == latest_run_id
    if result["status"] == "interrupted" and latest and (result.get("output") or has_interrupts):
        result["status"] = "success"
    result["error"] = result.get("error_message")
    return result


async def _missing_run(session, capability, thread_id, run_id):
    state = await _state(session, capability["owner"], capability["parent_thread_id"])
    task = (state.get("values", {}).get("async_tasks") or {}).get(thread_id)
    if (
        not isinstance(task, dict)
        or task.get("thread_id") != thread_id
        or task.get("run_id") != run_id
    ):
        raise HTTPException(404, "Task not found")
    return {
        "run_id": run_id,
        "thread_id": thread_id,
        "status": "error",
        "error": "Worker thread is unavailable or deleted. Its result is unconfirmed.",
    }


async def _capability(request: Request) -> dict:
    try:
        capability = verify_payload(request.headers.get(_HEADER, ""), purpose="protocol")
        owner = capability["owner"]
        parent = capability["parent_thread_id"]
        if not isinstance(owner, str) or not isinstance(parent, str):
            raise PermissionError("Invalid identity")
    except (PermissionError, KeyError) as exc:
        raise HTTPException(401, "Invalid subagent capability") from exc
    async with _maker()() as session:
        row = await _owned_thread(session, owner, parent)
        if trusted_parent(owner, parent, row.metadata_json or {}):
            raise HTTPException(403, "Workers cannot delegate nested tasks")
    return capability


async def _child(session, capability, thread_id, *, lock=False):
    row = await _owned_thread(session, capability["owner"], thread_id, lock=lock)
    if (
        trusted_parent(row.user_id, thread_id, row.metadata_json or {})
        != capability["parent_thread_id"]
    ):
        raise HTTPException(404, "Task not found")
    return row


async def _worker_rows(session, owner, parent):
    from aegra_api.core.orm import Thread
    from sqlalchemy import select

    rows = await session.scalars(
        select(Thread).where(
            Thread.user_id == owner, Thread.metadata_json["parent_thread_id"].astext == parent
        )
    )
    return [
        row
        for row in rows
        if trusted_parent(owner, row.thread_id, row.metadata_json or {}) == parent
    ]


def _occupies_slot(row, settings):
    if row.status in {"busy", "interrupted"}:
        return True
    return (
        not (row.metadata_json or {}).get("graph_id")
        and (datetime.now(UTC) - row.created_at).total_seconds() < settings.SUBAGENT_TIMEOUT_SECONDS
    )


async def _launch_parent_state(session, owner: str, parent: str) -> dict:
    """Order launches against cancellation and reject a cancelled logical turn."""
    await _owned_thread(session, owner, parent, lock=True)
    state = await _state(session, owner, parent)
    expected = (state.get("values") or {}).get("security_generation")
    if expected is not None:
        from taskharbor.security.repository import get_approval_store

        generation = await asyncio.to_thread(get_approval_store().generation, owner, parent)
        if generation != expected:
            raise HTTPException(409, "The parent turn was cancelled")
    return state


def create_protocol_router() -> APIRouter:
    from aegra_api.core.orm import Run as RunORM
    from aegra_api.core.orm import Thread
    from aegra_api.models import Run, RunCreate, User
    from sqlalchemy import select

    router = APIRouter(prefix=_PREFIX, tags=["Async subagents"])

    @router.post("/threads")
    async def create_thread(body: dict, capability=Depends(_capability)):
        from aegra_api.api.threads import _serialize_thread

        from taskharbor.settings import get_settings

        settings = get_settings()
        owner, parent = capability["owner"], capability["parent_thread_id"]
        metadata = body.get("metadata") or {}
        if (
            metadata.get("parent_thread_id") != parent
            or metadata.get("subagent_name") != "taskharbor-worker"
        ):
            raise HTTPException(403, "Invalid task parent or agent")
        async with _maker()() as session:
            parent_state = await _launch_parent_state(session, owner, parent)
            rows = await _worker_rows(session, owner, parent)
            active = [row for row in rows if _occupies_slot(row, settings)]
            if len(active) >= settings.SUBAGENT_MAX_CONCURRENCY:
                raise HTTPException(409, "Subagent concurrency limit reached")
            values = parent_state.get("values") or {}
            turn_id = values.get("security_turn_id") or values.get("sandbox_turn_id")
            if not turn_id:
                latest_parent = await session.scalar(
                    select(RunORM)
                    .where(RunORM.thread_id == parent, RunORM.user_id == owner)
                    .order_by(RunORM.created_at.desc())
                    .limit(1)
                )
                turn_id = latest_parent.run_id if latest_parent else parent
            if (
                sum((row.metadata_json or {}).get("sandbox_parent_turn") == turn_id for row in rows)
                >= settings.SUBAGENT_MAX_TASKS_PER_RUN
            ):
                raise HTTPException(409, "Subagent task budget reached")
            child = str(uuid4())
            metadata = {
                "owner": owner,
                "sandbox_worker": True,
                "parent_thread_id": parent,
                "subagent_name": "taskharbor-worker",
                "sandbox_parent_turn": turn_id,
                "sandbox_relationship": sign_payload(
                    _relationship_payload(owner, parent, child), purpose="relationship"
                ),
            }
            row = Thread(thread_id=child, status="idle", metadata_json=metadata, user_id=owner)
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _serialize_thread(row).model_dump(mode="json")

    @router.get("/threads/{thread_id}")
    async def get_thread(thread_id: str, capability=Depends(_capability)):
        from aegra_api.api.threads import _serialize_thread

        async with _maker()() as session:
            row = await _child(session, capability, thread_id)
            state = await _state(session, row.user_id, thread_id)
            result = _serialize_thread(row).model_dump(mode="json")
            result.update(
                values=state.get("values", {}), interrupts={"": flatten_interrupts(state)}
            )
            return result

    @router.get("/threads/{thread_id}/state")
    async def get_state(thread_id: str, capability=Depends(_capability)):
        async with _maker()() as session:
            await _child(session, capability, thread_id)
            return await _state(session, capability["owner"], thread_id)

    @router.post("/threads/{thread_id}/runs")
    async def create_run(thread_id: str, body: dict, capability=Depends(_capability)):
        from aegra_api.api.runs import active_runs, cancel_run_endpoint
        from aegra_api.api.runs import create_run as aegra_create

        from taskharbor.settings import get_settings

        if body.get("assistant_id") != "taskharbor-worker":
            raise HTTPException(403, "Invalid worker graph")
        if body.get("command"):
            raise HTTPException(403, "Task management cannot answer approvals")
        if body.get("multitask_strategy") == "enqueue":
            raise HTTPException(400, "Queued updates are unavailable on this Aegra version")
        owner = capability["owner"]
        user = User(identity=owner)
        from taskharbor.security.http import thread_run_lock

        async with thread_run_lock(owner, thread_id), _maker()() as session:
            parent_state = await _launch_parent_state(
                session, owner, capability["parent_thread_id"]
            )
            child = await _child(session, capability, thread_id, lock=True)
            pending = list(
                await session.scalars(
                    select(RunORM).where(
                        RunORM.thread_id == thread_id,
                        RunORM.user_id == owner,
                        RunORM.status.in_(["running", "pending"]),
                    )
                )
            )
            if pending and body.get("multitask_strategy", "reject") != "interrupt":
                raise HTTPException(409, "Task is already running")
            for previous in pending:
                await cancel_run_endpoint(
                    thread_id, previous.run_id, wait=1, action="cancel", user=user, session=session
                )
            if pending:
                parent_state = await _launch_parent_state(
                    session, owner, capability["parent_thread_id"]
                )
                child = await _child(session, capability, thread_id, lock=True)
                await session.refresh(child)
                newly_started = await session.scalar(
                    select(RunORM)
                    .where(
                        RunORM.thread_id == thread_id,
                        RunORM.user_id == owner,
                        RunORM.status.in_(["running", "pending"]),
                    )
                    .limit(1)
                )
                if newly_started is not None:
                    raise HTTPException(409, "Another update started while this task was stopping")
            settings = get_settings()
            if not _occupies_slot(child, settings):
                rows = await _worker_rows(session, owner, capability["parent_thread_id"])
                active = [
                    row
                    for row in rows
                    if row.thread_id != thread_id and _occupies_slot(row, settings)
                ]
                if len(active) >= settings.SUBAGENT_MAX_CONCURRENCY:
                    raise HTTPException(409, "Subagent concurrency limit reached")
            if child.status == "interrupted":
                from taskharbor.http_app import _cancel_security_work

                await asyncio.to_thread(_cancel_worker_approvals, owner, thread_id)
                await asyncio.to_thread(_cancel_security_work, owner, thread_id)
            parent_turn = (parent_state.get("values") or {}).get("security_turn_id")
            child.metadata_json = {
                **(child.metadata_json or {}),
                "sandbox_task_deadline": (
                    datetime.now(UTC) + timedelta(seconds=settings.SUBAGENT_TIMEOUT_SECONDS)
                ).isoformat(),
            }
            if parent_turn:
                child.metadata_json = {
                    **(child.metadata_json or {}),
                    "sandbox_parent_turn": parent_turn,
                }
            parent_run = await session.scalar(
                select(RunORM)
                .where(RunORM.thread_id == capability["parent_thread_id"], RunORM.user_id == owner)
                .order_by(RunORM.created_at.desc())
                .limit(1)
            )
            config = {
                "recursion_limit": settings.SUBAGENT_MAX_STEPS,
                "configurable": _model_config(parent_run) if parent_run else {},
            }
            request = RunCreate(
                assistant_id="taskharbor-worker",
                input=body.get("input") or {},
                config=config,
                stream_mode=["values", "updates", "messages", "custom"],
                stream_subgraphs=True,
                multitask_strategy="reject",
            )
            run = await aegra_create(thread_id, request, user=user, session=session)
            task = active_runs.get(run.run_id)
            if task is not None:
                task.add_done_callback(lambda _: _schedule_reconcile())
            return run.model_dump(mode="json")

    @router.get("/threads/{thread_id}/runs/{run_id}")
    async def get_run(thread_id: str, run_id: str, capability=Depends(_capability)):
        async with _maker()() as session:
            try:
                row = await _child(session, capability, thread_id)
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
                return await _missing_run(session, capability, thread_id, run_id)
            run = await session.scalar(
                select(RunORM).where(
                    RunORM.thread_id == thread_id,
                    RunORM.run_id == run_id,
                    RunORM.user_id == row.user_id,
                )
            )
            if run is None:
                return await _missing_run(session, capability, thread_id, run_id)
            latest = await session.scalar(
                select(RunORM)
                .where(RunORM.thread_id == thread_id, RunORM.user_id == row.user_id)
                .order_by(RunORM.created_at.desc())
                .limit(1)
            )
            paused = False
            if (
                run.status == "interrupted"
                and not run.output
                and row.status == "interrupted"
                and latest.run_id == run_id
            ):
                paused = bool(flatten_interrupts(await _state(session, row.user_id, thread_id)))
            return normalize_run(
                Run.model_validate(run), row, latest_run_id=latest.run_id, has_interrupts=paused
            )

    @router.get("/threads/{thread_id}/runs")
    async def list_runs(
        thread_id: str, limit: int = Query(10, ge=1, le=100), capability=Depends(_capability)
    ):
        async with _maker()() as session:
            row = await _child(session, capability, thread_id)
            runs = list(
                await session.scalars(
                    select(RunORM)
                    .where(RunORM.thread_id == thread_id, RunORM.user_id == row.user_id)
                    .order_by(RunORM.created_at.desc())
                    .limit(limit)
                )
            )
            if not runs:
                return []
            paused = False
            if (
                runs[0].status == "interrupted"
                and not runs[0].output
                and row.status == "interrupted"
            ):
                paused = bool(flatten_interrupts(await _state(session, row.user_id, thread_id)))
            return [
                normalize_run(
                    Run.model_validate(run),
                    row,
                    latest_run_id=runs[0].run_id,
                    has_interrupts=paused,
                )
                for run in runs
            ]

    @router.post("/threads/{thread_id}/runs/{run_id}/cancel")
    async def cancel_run(thread_id: str, run_id: str, capability=Depends(_capability)):
        from aegra_api.api.runs import cancel_run_endpoint

        from taskharbor.security.http import thread_run_lock

        async with thread_run_lock(capability["owner"], thread_id), _maker()() as session:
            row = await _child(session, capability, thread_id)
            user = User(identity=row.user_id)
            result = await cancel_run_endpoint(
                thread_id, run_id, wait=1, action="cancel", user=user, session=session
            )
            await asyncio.to_thread(_cancel_worker_approvals, row.user_id, thread_id)
            from taskharbor.http_app import _cancel_security_work

            await asyncio.to_thread(_cancel_security_work, row.user_id, thread_id)
            row.status = "idle"
            run = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
            run.output = {}
            await session.commit()
            _schedule_reconcile()
            return result.model_dump(mode="json")

    return router


def _schedule_reconcile():
    _start_background(reconcile_waiting_parents())


def _start_background(coroutine):
    task = asyncio.create_task(coroutine)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    task.add_done_callback(_log_background_error)
    return task


def _log_background_error(task):
    if task.cancelled():
        return
    if error := task.exception():
        logger.warning("Subagent reconciliation failed (%s)", type(error).__name__)


async def reconcile_expired_approvals():
    """Deliver expired decisions through their checkpoint, never approve an operation."""
    from aegra_api.api.runs import create_run
    from aegra_api.core.orm import Run, Thread
    from aegra_api.models import RunCreate, User
    from sqlalchemy import select

    from taskharbor.security.http import _has_active_run, thread_run_lock
    from taskharbor.security.repository import get_approval_store
    from taskharbor.settings import get_settings

    settings = get_settings()
    store = get_approval_store()
    async with _maker()() as session:
        candidates = list(
            await session.scalars(select(Thread).where(Thread.status == "interrupted"))
        )
    for candidate in candidates:
        worker = trusted_parent(
            candidate.user_id, candidate.thread_id, candidate.metadata_json or {}
        )
        if not worker and (candidate.metadata_json or {}).get("graph_id") != "taskharbor":
            continue
        async with thread_run_lock(candidate.user_id, candidate.thread_id), _maker()() as session:
            try:
                thread = await _owned_thread(
                    session, candidate.user_id, candidate.thread_id, lock=True
                )
                if thread.status != "interrupted" or await asyncio.to_thread(
                    _has_active_run, thread.user_id, thread.thread_id
                ):
                    continue
                state = await _state(session, thread.user_id, thread.thread_id)
                expected = (state.get("values") or {}).get("security_generation")
                if expected is None or expected != await asyncio.to_thread(
                    store.generation, thread.user_id, thread.thread_id
                ):
                    continue
                # Reading the repository also transitions overdue pending/approved rows.
                records = await asyncio.to_thread(
                    store.list_thread, thread.user_id, thread.thread_id
                )
                expired = {
                    row["id"]: row for row in records
                    if row["status"] == "expired" and row.get("result") is None
                    and row.get("generation") == expected
                }
                resume = {}
                for item in flatten_interrupts(state):
                    value = item.get("value")
                    row = expired.get(value.get("approval_id")) if isinstance(value, dict) else None
                    if row and row["tool_call_id"] == value.get("tool_call_id"):
                        resume[item["id"]] = {"approval_id": row["id"]}
                if not resume:
                    continue
                previous = await session.scalar(
                    select(Run).where(
                        Run.thread_id == thread.thread_id, Run.user_id == thread.user_id
                    ).order_by(Run.created_at.desc()).limit(1)
                )
                await create_run(
                    thread.thread_id,
                    RunCreate(
                        assistant_id="taskharbor-worker" if worker else (
                            (thread.metadata_json or {}).get("assistant_id") or "taskharbor"
                        ),
                        command={"resume": resume},
                        config={
                            "configurable": _model_config(previous) if previous else {},
                            "recursion_limit": settings.SUBAGENT_MAX_STEPS if worker
                            else settings.AGENT_MAX_STEPS,
                        },
                        stream_mode=["values", "updates", "messages", "custom"],
                        stream_subgraphs=True,
                        multitask_strategy="reject",
                    ),
                    user=User(identity=thread.user_id),
                    session=session,
                )
            except HTTPException as exc:
                if exc.status_code not in {404, 409}:
                    raise


async def reconcile_waiting_parents():
    """Resume only a persisted business wait, leaving user decisions to the user."""
    from aegra_api.api.runs import create_run
    from aegra_api.core.orm import Run, Thread
    from aegra_api.models import RunCreate, User
    from sqlalchemy import select

    from taskharbor.security.http import _has_active_run, thread_run_lock

    async with _maker()() as session:
        parents = list(await session.scalars(select(Thread).where(Thread.status == "interrupted")))
    for candidate in parents:
        if (candidate.metadata_json or {}).get("graph_id") != "taskharbor":
            continue
        async with thread_run_lock(candidate.user_id, candidate.thread_id), _maker()() as session:
            try:
                parent = await _owned_thread(
                    session, candidate.user_id, candidate.thread_id, lock=True
                )
            except HTTPException as exc:
                if exc.status_code == 404:
                    continue
                raise
            if parent.status != "interrupted" or await asyncio.to_thread(
                _has_active_run, parent.user_id, parent.thread_id
            ):
                continue
            state = await _state(session, parent.user_id, parent.thread_id)
            interrupts = flatten_interrupts(state)
            waits = [
                item
                for item in interrupts
                if isinstance(item.get("value"), dict)
                and item["value"].get("kind") == "awaiting_subtasks"
            ]
            if not waits or len(waits) != len(interrupts):
                continue
            resume = {}
            for item in waits:
                task_ids = item["value"].get("task_ids") or []
                ready = True
                for child_id in task_ids:
                    tracked = (state.get("values", {}).get("async_tasks") or {}).get(child_id)
                    if not isinstance(tracked, dict) or tracked.get("thread_id") != child_id:
                        ready = False
                        break
                    try:
                        child = await _owned_thread(session, parent.user_id, child_id)
                    except HTTPException as exc:
                        if exc.status_code != 404:
                            raise
                        continue
                    if (
                        trusted_parent(parent.user_id, child_id, child.metadata_json or {})
                        != parent.thread_id
                    ):
                        ready = False
                        break
                    latest = await session.scalar(
                        select(Run)
                        .where(Run.thread_id == child_id)
                        .order_by(Run.created_at.desc())
                        .limit(1)
                    )
                    if (
                        latest is None
                        or latest.status not in _TERMINAL
                        or child.status in {"busy", "interrupted"}
                    ):
                        ready = False
                        break
                if ready:
                    resume[item["id"]] = {"subtasks_ready": True}
            if len(resume) == len(waits):
                previous = await session.scalar(
                    select(Run)
                    .where(Run.thread_id == parent.thread_id, Run.user_id == parent.user_id)
                    .order_by(Run.created_at.desc())
                    .limit(1)
                )
                await create_run(
                    parent.thread_id,
                    RunCreate(
                        assistant_id=(parent.metadata_json or {}).get("assistant_id")
                        or "taskharbor",
                        command={"resume": resume},
                        stream_mode=["values", "updates", "messages", "custom"],
                        config={"configurable": _model_config(previous) if previous else {}},
                        stream_subgraphs=True,
                        multitask_strategy="reject",
                    ),
                    user=User(identity=parent.user_id),
                    session=session,
                )


async def reconcile_orphaned_runs():
    """An interrupted process leaves no live coroutine, report that execution clearly."""
    from aegra_api.api.runs import active_runs
    from aegra_api.core.orm import Run, Thread
    from sqlalchemy import select

    from taskharbor.security.http import thread_run_lock

    async with _maker()() as session:
        runs = list(
            await session.scalars(select(Run).where(Run.status.in_(["pending", "running"])))
        )
        for run in runs:
            async with thread_run_lock(run.user_id, run.thread_id):
                await session.refresh(run)
                if run.run_id in active_runs or run.status not in {"pending", "running"}:
                    continue
                thread = await session.scalar(
                    select(Thread).where(Thread.thread_id == run.thread_id)
                )
                if thread is None:
                    continue
                worker = trusted_parent(run.user_id, run.thread_id, thread.metadata_json or {})
                if not worker and (thread.metadata_json or {}).get("graph_id") != "taskharbor":
                    continue
                state = await _state(session, run.user_id, run.thread_id)
                if flatten_interrupts(state):
                    run.status, thread.status = "interrupted", "interrupted"
                    run.output = state.get("values") or {}
                else:
                    run.status, thread.status = "error", "error"
                    run.error_message = (
                        "Execution stopped when the service restarted. "
                        "Its checkpoint and workspace remain available."
                    )
                run.updated_at = datetime.now(UTC)

        # Aegra commits the final run status and thread status separately. A
        # process can stop after the run commits while its thread is still busy.
        busy_threads = list(
            await session.scalars(select(Thread).where(Thread.status == "busy"))
        )
        for thread in busy_threads:
            worker = trusted_parent(thread.user_id, thread.thread_id, thread.metadata_json or {})
            if not worker and (thread.metadata_json or {}).get("graph_id") != "taskharbor":
                continue
            async with thread_run_lock(thread.user_id, thread.thread_id):
                await session.refresh(thread)
                if thread.status != "busy":
                    continue
                latest = await session.scalar(
                    select(Run)
                    .where(Run.thread_id == thread.thread_id, Run.user_id == thread.user_id)
                    .order_by(Run.created_at.desc())
                    .limit(1)
                )
                if latest is not None and latest.run_id in active_runs:
                    continue
                state = await _state(session, thread.user_id, thread.thread_id)
                if flatten_interrupts(state):
                    thread.status = "interrupted"
                    if latest is not None:
                        latest.status = "interrupted"
                        latest.output = state.get("values") or {}
                elif latest is not None and latest.status in {"success", "interrupted"}:
                    thread.status = "idle"
                else:
                    thread.status = "error"
                thread.updated_at = datetime.now(UTC)
        await session.commit()


async def _expire_workers():
    from aegra_api.api.runs import cancel_run_endpoint
    from aegra_api.core.orm import Run, Thread
    from aegra_api.models import User
    from sqlalchemy import select

    from taskharbor.security.http import thread_run_lock
    from taskharbor.settings import get_settings

    settings = get_settings()
    async with _maker()() as session:
        rows = list(
            await session.scalars(select(Thread).where(Thread.status.in_(["busy", "interrupted"])))
        )
        for thread in rows:
            if not trusted_parent(thread.user_id, thread.thread_id, thread.metadata_json or {}):
                continue
            async with thread_run_lock(thread.user_id, thread.thread_id):
                await session.refresh(thread)
                if thread.status not in {"busy", "interrupted"}:
                    continue
                run = await session.scalar(
                    select(Run)
                    .where(Run.thread_id == thread.thread_id)
                    .order_by(Run.created_at.desc())
                    .limit(1)
                )
                if run is None:
                    continue
                try:
                    deadline = datetime.fromisoformat(
                        (thread.metadata_json or {}).get("sandbox_task_deadline")
                    )
                    if deadline.tzinfo is None:
                        raise ValueError("Task deadline requires a timezone")
                except (ValueError, TypeError):
                    # Existing worker threads predate persistent task deadlines.
                    deadline = run.created_at + timedelta(
                        seconds=settings.SUBAGENT_TIMEOUT_SECONDS
                    )
                if datetime.now(UTC) <= deadline:
                    continue
                await cancel_run_endpoint(
                    thread.thread_id,
                    run.run_id,
                    wait=1,
                    action="cancel",
                    user=User(identity=thread.user_id),
                    session=session,
                )
                run.status, run.error_message, run.output = (
                    "timeout",
                    "Subagent deadline exceeded",
                    {},
                )
                thread.status = "idle"
                await session.commit()
                from taskharbor.http_app import _cancel_security_work

                await asyncio.to_thread(_cancel_worker_approvals, thread.user_id, thread.thread_id)
                await asyncio.to_thread(_cancel_security_work, thread.user_id, thread.thread_id)


async def _retire_finished_workers():
    from aegra_api.core.orm import Thread
    from sqlalchemy import select

    from taskharbor.http_app import _get_manager
    from taskharbor.security.http import thread_run_lock

    manager = _get_manager()
    active_threads = {info.thread_id for info in list(manager.sessions.values()) if info.thread_id}
    if not active_threads:
        return
    async with _maker()() as session:
        threads = await session.scalars(
            select(Thread).where(
                Thread.thread_id.in_(active_threads), Thread.status.in_(["idle", "error"])
            )
        )
        for thread in threads:
            if trusted_parent(thread.user_id, thread.thread_id, thread.metadata_json or {}):
                async with thread_run_lock(thread.user_id, thread.thread_id):
                    await session.refresh(thread)
                    if thread.status in {"idle", "error"}:
                        await asyncio.to_thread(
                            manager.cleanup_thread_sessions,
                            thread.thread_id,
                            user_id=thread.user_id,
                            delete_files=False,
                        )


async def cancel_child_tasks(owner: str, parent: str, *, task_ids=None):
    from aegra_api.api.runs import cancel_run_endpoint
    from aegra_api.core.orm import Run, Thread
    from aegra_api.models import User
    from sqlalchemy import select

    from taskharbor.http_app import _cancel_security_work
    from taskharbor.security.http import thread_run_lock

    async with thread_run_lock(owner, parent), _maker()() as session:
        try:
            await _owned_thread(session, owner, parent, lock=True)
        except HTTPException as exc:
            if exc.status_code == 404:
                return
            raise
        children = list(
            await session.scalars(
                select(Thread).where(
                    Thread.user_id == owner,
                    Thread.metadata_json["parent_thread_id"].astext == parent,
                )
            )
        )
        turn_id = None
        if children and task_ids is None:
            try:
                state = await _state(session, owner, parent)
            except HTTPException as exc:
                if exc.status_code == 404:
                    return
                raise
            turn_id = (state.get("values") or {}).get("security_turn_id")
        for child in children:
            if task_ids is not None and child.thread_id not in task_ids:
                continue
            if trusted_parent(owner, child.thread_id, child.metadata_json or {}) != parent:
                continue
            if turn_id and (child.metadata_json or {}).get("sandbox_parent_turn") != turn_id:
                continue
            async with thread_run_lock(owner, child.thread_id):
                await session.refresh(child)
                if turn_id and (child.metadata_json or {}).get("sandbox_parent_turn") != turn_id:
                    continue
                if child.status not in {"busy", "interrupted"}:
                    continue
                latest = await session.scalar(
                    select(Run)
                    .where(Run.thread_id == child.thread_id)
                    .order_by(Run.created_at.desc())
                    .limit(1)
                )
                if latest:
                    await cancel_run_endpoint(
                        child.thread_id,
                        latest.run_id,
                        wait=1,
                        action="cancel",
                        user=User(identity=owner),
                        session=session,
                    )
                    latest.output = {}
                    child.status = "idle"
                    await session.commit()
                await asyncio.to_thread(_cancel_worker_approvals, owner, child.thread_id)
                await asyncio.to_thread(_cancel_security_work, owner, child.thread_id)


async def _cancel_waiting_tasks():
    """Upstream cancels waiting tasks in parent state without an HTTP request."""
    from aegra_api.core.orm import Thread
    from sqlalchemy import select

    async with _maker()() as session:
        children = list(
            await session.scalars(select(Thread).where(Thread.status.in_(["busy", "interrupted"])))
        )
        parents = {}
        for child in children:
            parent = trusted_parent(child.user_id, child.thread_id, child.metadata_json or {})
            if parent is None:
                continue
            key = (child.user_id, parent)
            if key not in parents:
                parents[key] = await _state(session, child.user_id, parent)
            tasks = parents[key].get("values", {}).get("async_tasks") or {}
            if (tasks.get(child.thread_id) or {}).get("status") == "cancelled":
                await cancel_child_tasks(child.user_id, parent, task_ids={child.thread_id})


def schedule_cancel_children(owner: str, parent: str):
    """Bridge the existing synchronous cancellation callback to the service loop."""
    loop = _service_loop
    if loop is not None and loop.is_running():
        def submit():
            if _service_loop is loop:
                _start_background(cancel_child_tasks(owner, parent))

        loop.call_soon_threadsafe(submit)


async def cleanup_deleted_children(owner: str, parent: str, child_ids: list[str]):
    """Use captured, signed relationships after the parent's row has been deleted."""
    from aegra_api.api.threads import delete_thread
    from aegra_api.models import User
    from aegra_api.services.langgraph_service import get_langgraph_service

    from taskharbor.http_app import _get_manager
    from taskharbor.security.http import thread_run_lock
    from taskharbor.security.repository import get_approval_store

    async with _maker()() as session:
        # Include a launch committed between the pre-delete snapshot and deletion.
        current = await _worker_rows(session, owner, parent)
        child_ids = sorted(set(child_ids) | {row.thread_id for row in current})
    for child_id in child_ids:
        try:
            async with thread_run_lock(owner, child_id), _maker()() as session:
                child = await _owned_thread(session, owner, child_id)
                if trusted_parent(owner, child_id, child.metadata_json or {}) != parent:
                    continue
                graph_id = (child.metadata_json or {}).get("graph_id") or "taskharbor-worker"
                await delete_thread(child_id, user=User(identity=owner), session=session)
            async with get_langgraph_service().get_graph(graph_id) as graph:
                if graph.checkpointer is not None:
                    await graph.checkpointer.adelete_thread(child_id)
            await asyncio.to_thread(get_approval_store().revoke_thread, owner, child_id)
            await asyncio.to_thread(
                _get_manager().cleanup_thread_sessions,
                child_id,
                user_id=owner,
                delete_files=True,
            )
        except HTTPException as exc:
            if exc.status_code != 404:
                logger.warning("Child thread deletion failed for %s", child_id, exc_info=True)
        except Exception:
            logger.warning("Child thread cleanup failed for %s", child_id, exc_info=True)


def schedule_deleted_children(owner: str, parent: str, child_ids: list[str]):
    _start_background(cleanup_deleted_children(owner, parent, child_ids))


async def close_protocol():
    global _service_loop
    _service_loop = None
    tasks = list(_background_tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def run_reconciler():
    global _service_loop
    _service_loop = asyncio.get_running_loop()
    startup_pending = True
    while True:
        try:
            if startup_pending:
                await reconcile_orphaned_runs()
                startup_pending = False
            await _cancel_waiting_tasks()
            await _expire_workers()
            await reconcile_expired_approvals()
            await _retire_finished_workers()
            await reconcile_waiting_parents()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Subagent reconciliation will retry", exc_info=True)
        await asyncio.sleep(1)
