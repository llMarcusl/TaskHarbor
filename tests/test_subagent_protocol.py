"""The service adapter preserves upstream SDK and authenticated task semantics."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from starlette.testclient import TestClient

from taskharbor.agent.subagents import protocol


@pytest.fixture
def settings(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        STORAGE_DIR=tmp_path,
        ENCRYPTION_KEY="test-service-secret",
        API_BASE_URL="http://test",
        SUBAGENT_PROTOCOL_URL=None,
        SUBAGENT_MAX_CONCURRENCY=3,
        SUBAGENT_MAX_TASKS_PER_RUN=12,
        SUBAGENT_TIMEOUT_SECONDS=900,
        SUBAGENT_MAX_STEPS=80,
        AGENT_MAX_STEPS=1000,
    )
    monkeypatch.setattr("taskharbor.settings.get_settings", lambda: settings)
    return settings


def test_signatures_bind_payload_purpose_and_secret(settings):
    payload = {"owner": "alice", "parent": "parent", "child": "child"}
    token = protocol.sign_payload(payload, purpose="relationship")
    assert protocol.verify_payload(token, purpose="relationship") == payload
    with pytest.raises(PermissionError):
        protocol.verify_payload(token, purpose="file-manifest")
    with pytest.raises(PermissionError):
        protocol.verify_payload(
            token[:-1] + ("0" if token[-1] != "0" else "1"), purpose="relationship"
        )
    settings.ENCRYPTION_KEY = "different-secret"
    with pytest.raises(PermissionError):
        protocol.verify_payload(token, purpose="relationship")


def test_generated_signing_key_survives_process_cache_reset(settings):
    settings.ENCRYPTION_KEY = ""
    token = protocol.sign_payload({"value": 1}, purpose="file-manifest")
    protocol._signing_key.cache_clear()
    assert protocol.verify_payload(token, purpose="file-manifest") == {"value": 1}
    assert len((settings.STORAGE_DIR / ".subagent-signing-key").read_bytes()) == 32


def relationship(owner="alice", parent="parent", child="child"):
    return {
        "sandbox_worker": True,
        "parent_thread_id": parent,
        "subagent_name": "taskharbor-worker",
        "sandbox_relationship": protocol.sign_payload(
            protocol._relationship_payload(owner, parent, child), purpose="relationship"
        ),
    }


def test_shutdown_drains_cancellation_submitted_from_a_sync_callback(monkeypatch):
    async def scenario():
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def cancel(owner, parent):
            assert (owner, parent) == ("alice", "parent")
            started.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.set()

        monkeypatch.setattr(protocol, "cancel_child_tasks", cancel)
        monkeypatch.setattr(protocol, "_service_loop", asyncio.get_running_loop())
        monkeypatch.setattr(protocol, "_background_tasks", set())
        await asyncio.to_thread(protocol.schedule_cancel_children, "alice", "parent")
        await asyncio.wait_for(started.wait(), 1)
        await protocol.close_protocol()
        assert cleaned.is_set()
        assert not protocol._background_tasks

    asyncio.run(scenario())


def test_shutdown_rejects_a_cancellation_callback_that_has_not_started(monkeypatch):
    async def scenario():
        cancel = AsyncMock()
        monkeypatch.setattr(protocol, "cancel_child_tasks", cancel)
        monkeypatch.setattr(protocol, "_service_loop", asyncio.get_running_loop())
        monkeypatch.setattr(protocol, "_background_tasks", set())
        protocol.schedule_cancel_children("alice", "parent")
        await protocol.close_protocol()
        await asyncio.sleep(0)
        cancel.assert_not_awaited()
        assert not protocol._background_tasks

    asyncio.run(scenario())


def test_public_metadata_cannot_forge_or_copy_relationship(settings):
    metadata = relationship()
    assert protocol.trusted_parent("alice", "child", metadata) == "parent"
    assert protocol.trusted_parent("bob", "child", metadata) is None
    assert protocol.trusted_parent("alice", "other-child", metadata) is None
    assert (
        protocol.trusted_parent("alice", "child", {**metadata, "parent_thread_id": "other"}) is None
    )
    assert (
        protocol.trusted_parent(
            "alice", "child", {"sandbox_worker": True, "parent_thread_id": "parent"}
        )
        is None
    )


def test_runtime_headers_use_current_thread_and_authoritative_owner(settings, monkeypatch):
    state = {"configurable": {"thread_id": "parent", "run_id": "run-1"}}
    monkeypatch.setattr("langgraph.config.get_config", lambda: state)
    monkeypatch.setattr(
        "aegra_api.core.auth_ctx.get_auth_ctx",
        lambda: SimpleNamespace(user=SimpleNamespace(identity="alice")),
    )
    manager = SimpleNamespace(resolve_owner=Mock(return_value="alice"))
    headers = protocol.async_subagent_spec(manager, settings)["headers"]
    first = dict(headers)
    assert protocol.verify_payload(first[protocol._HEADER], purpose="protocol") == {
        "owner": "alice",
        "parent_thread_id": "parent",
    }
    state["configurable"]["run_id"] = "resumed-run"
    assert dict(headers) == first
    state["configurable"]["thread_id"] = "other-parent"
    assert dict(headers) != first
    manager.resolve_owner.return_value = "bob"
    with pytest.raises(PermissionError):
        dict(headers)


def test_nested_interrupts_flatten_and_deduplicate():
    first = {"id": "one", "value": {"approval_id": "approval-1"}}
    second = {"id": "two", "value": {"approval_id": "approval-2"}}
    state = {
        "interrupts": [first],
        "tasks": [{"interrupts": [first], "state": {"interrupts": [second]}}],
    }
    assert protocol.flatten_interrupts(state) == [first, second]


@pytest.mark.parametrize(
    ("status", "output", "expected"),
    [
        ("interrupted", {"messages": [{"content": "pending"}]}, "success"),
        ("interrupted", {}, "interrupted"),
        ("running", {}, "running"),
        ("timeout", {}, "timeout"),
    ],
)
def test_checkpoint_pause_and_cancellation_have_distinct_protocol_status(status, output, expected):
    run = SimpleNamespace(
        model_dump=lambda **_: {"status": status, "output": output, "error_message": None}
    )
    assert protocol.normalize_run(run, None)["status"] == expected


class FakeSession:
    def __init__(self):
        self.rows = []
        self.run = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def scalars(self, statement):
        return self.rows

    async def scalar(self, statement):
        return self.run

    def add(self, row):
        self.rows.append(row)

    async def commit(self):
        pass

    async def refresh(self, row, **_):
        row.created_at = row.updated_at = datetime.now(UTC)


@pytest.fixture
def api(settings, monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(protocol, "_maker", lambda: lambda: session)
    monkeypatch.setattr("taskharbor.security.http._has_active_run", lambda *_: False)
    capability = {"owner": "alice", "parent_thread_id": "parent", "turn_id": "turn"}
    parent = SimpleNamespace(thread_id="parent", user_id="alice", metadata_json={}, status="busy")

    async def owned(session, owner, thread_id, **_):
        for row in session.rows:
            if row.user_id == owner and row.thread_id == thread_id:
                return row
        if owner == "alice" and thread_id == "parent":
            return parent
        raise HTTPException(404, "Thread not found")

    monkeypatch.setattr(protocol, "_owned_thread", owned)

    async def state(*_):
        return {"values": {"security_turn_id": "turn"}}

    monkeypatch.setattr(protocol, "_state", state)
    app = FastAPI()
    app.include_router(protocol.create_protocol_router())
    app.dependency_overrides[protocol._capability] = lambda: capability
    return app, session, capability


def test_http_creates_signed_child_and_rejects_another_parent(api):
    app, session, _ = api
    with TestClient(app) as client:
        response = client.post(
            protocol._PREFIX + "/threads",
            json={
                "metadata": {
                    "parent_thread_id": "parent",
                    "subagent_name": "taskharbor-worker",
                    "owner": "forged-owner",
                    "sandbox_relationship": "forged-token",
                }
            },
        )
        assert response.status_code == 200
        row = response.json()
        assert row["user_id"] == "alice"
        assert protocol.trusted_parent("alice", row["thread_id"], row["metadata"]) == "parent"
        denied = client.post(
            protocol._PREFIX + "/threads",
            json={
                "metadata": {
                    "parent_thread_id": "other",
                    "subagent_name": "taskharbor-worker",
                }
            },
        )
        assert denied.status_code == 403
        assert len(session.rows) == 1


def test_concurrency_and_per_turn_budget_apply_before_launch(api, settings):
    app, _, _ = api
    settings.SUBAGENT_MAX_CONCURRENCY = 1
    payload = {"metadata": {"parent_thread_id": "parent", "subagent_name": "taskharbor-worker"}}
    with TestClient(app) as client:
        assert client.post(protocol._PREFIX + "/threads", json=payload).status_code == 200
        assert client.post(protocol._PREFIX + "/threads", json=payload).status_code == 409


def test_actual_sdk_reads_values_and_interrupts_from_adapter(api, monkeypatch):
    app, session, _ = api
    from aegra_api.core.orm import Thread
    from langgraph_sdk.client import LangGraphClient

    session.rows.append(
        Thread(
            thread_id="child", user_id="alice", status="interrupted", metadata_json=relationship()
        )
    )

    async def state(*_):
        return {
            "values": {"messages": [{"role": "assistant", "content": "result"}]},
            "interrupts": [{"id": "approval", "value": {"approval_id": "approval-1"}}],
        }

    monkeypatch.setattr(protocol, "_state", state)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test" + protocol._PREFIX
        ) as http:
            client = LangGraphClient(http)
            thread = await client.threads.get("child")
            assert thread["values"]["messages"][0]["content"] == "result"
            assert thread["interrupts"][""][0]["id"] == "approval"
            with pytest.raises(httpx.HTTPStatusError):
                await client.threads.get("foreign-child")

    asyncio.run(exercise())


def test_protocol_requires_service_capability(settings):
    app = FastAPI()
    app.include_router(protocol.create_protocol_router())
    with TestClient(app) as client:
        assert client.get(protocol._PREFIX + "/threads/child").status_code == 401
        assert (
            client.get(
                protocol._PREFIX + "/threads/child", headers={protocol._HEADER: "forged"}
            ).status_code
            == 401
        )


def test_old_paused_run_hands_over_to_newer_run():
    run = SimpleNamespace(
        model_dump=lambda **_: {
            "run_id": "old",
            "status": "interrupted",
            "output": {"messages": []},
        }
    )
    assert protocol.normalize_run(run, None, latest_run_id="new")["status"] == "interrupted"
    assert protocol.normalize_run(run, None, latest_run_id="old")["status"] == "success"


def test_empty_output_checkpoint_pause_is_still_waiting():
    run = SimpleNamespace(
        model_dump=lambda **_: {"run_id": "pause", "status": "interrupted", "output": {}}
    )
    assert (
        protocol.normalize_run(run, None, latest_run_id="pause", has_interrupts=True)["status"]
        == "success"
    )


def test_deleted_tracked_child_becomes_explicit_error(api, monkeypatch):
    app, _, _ = api

    async def state(*_):
        return {
            "values": {
                "async_tasks": {
                    "deleted": {"task_id": "deleted", "thread_id": "deleted", "run_id": "run"}
                }
            }
        }

    monkeypatch.setattr(protocol, "_state", state)
    with TestClient(app) as client:
        response = client.get(protocol._PREFIX + "/threads/deleted/runs/run")
        assert response.status_code == 200
        assert response.json()["status"] == "error"
        assert "unconfirmed" in response.json()["error"]
        assert client.get(protocol._PREFIX + "/threads/untracked/runs/run").status_code == 404
        assert client.get(protocol._PREFIX + "/threads/deleted/runs/untracked").status_code == 404


@pytest.mark.parametrize(
    "child_status,user_approval,expected",
    [
        ("idle", False, True),
        ("interrupted", False, False),
        ("idle", True, False),
    ],
)
def test_business_wait_resumes_only_after_all_children_and_without_user_approval(
    api, monkeypatch, child_status, user_approval, expected
):
    _, session, _ = api
    now = datetime.now(UTC)
    parent = SimpleNamespace(
        thread_id="parent",
        user_id="alice",
        status="interrupted",
        metadata_json={"graph_id": "taskharbor", "assistant_id": "taskharbor"},
        created_at=now,
        updated_at=now,
    )
    child = SimpleNamespace(
        thread_id="child",
        user_id="alice",
        status=child_status,
        metadata_json=relationship(),
        created_at=now,
        updated_at=now,
    )
    session.rows = [parent, child]
    session.run = SimpleNamespace(
        run_id="run",
        status="success",
        config={"configurable": {"chat_model": "chosen"}},
        context={},
    )
    interrupts = [{"id": "wait", "value": {"kind": "awaiting_subtasks", "task_ids": ["child"]}}]
    if user_approval:
        interrupts.append({"id": "approval", "value": {"approval_id": "approval-1"}})

    async def state(*_):
        return {
            "interrupts": interrupts,
            "values": {"async_tasks": {"child": {"thread_id": "child", "run_id": "run"}}},
        }

    resumed = []

    async def create(thread_id, request, **_):
        resumed.append((thread_id, request))

    monkeypatch.setattr(protocol, "_state", state)
    monkeypatch.setattr("aegra_api.api.runs.create_run", create)
    asyncio.run(protocol.reconcile_waiting_parents())
    assert bool(resumed) is expected
    if expected:
        assert resumed[0][0] == "parent"
        assert resumed[0][1].command == {"resume": {"wait": {"subtasks_ready": True}}}
        assert resumed[0][1].config["configurable"]["chat_model"] == "chosen"


@pytest.mark.parametrize(
    "worker,paused", [(True, False), (False, False), (True, True), (False, True)]
)
def test_restart_reconciles_root_and_worker_without_live_coroutines(
    settings, monkeypatch, worker, paused
):
    session = FakeSession()
    run = SimpleNamespace(
        run_id="lost-run",
        thread_id="child" if worker else "parent",
        user_id="alice",
        status="running",
        output=None,
        error_message=None,
    )
    thread = SimpleNamespace(
        thread_id=run.thread_id,
        user_id="alice",
        status="busy",
        metadata_json=relationship() if worker else {"graph_id": "taskharbor"},
    )
    session.rows, session.run = [run], thread

    async def rows(statement):
        from aegra_api.core.orm import Run

        return [run] if statement.column_descriptions[0]["entity"] is Run else []

    monkeypatch.setattr(session, "scalars", rows)
    monkeypatch.setattr(protocol, "_maker", lambda: lambda: session)
    monkeypatch.setattr("aegra_api.api.runs.active_runs", {})

    async def state(*_):
        return {
            "interrupts": [{"id": "approval", "value": {"approval_id": "approval"}}]
            if paused
            else [],
            "values": {"messages": [{"content": "persisted"}]},
        }

    monkeypatch.setattr(protocol, "_state", state)
    asyncio.run(protocol.reconcile_orphaned_runs())
    assert run.status == ("interrupted" if paused else "error")
    assert thread.status == run.status
    if paused:
        assert run.output["messages"][0]["content"] == "persisted"
    else:
        assert "checkpoint" in run.error_message


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize(
    ("status", "paused", "expected"),
    [
        ("interrupted", True, "interrupted"),
        ("success", True, "interrupted"),
        ("success", False, "idle"),
        ("interrupted", False, "idle"),
        ("error", False, "error"),
        ("timeout", False, "error"),
        (None, False, "error"),
    ],
)
def test_restart_repairs_busy_thread_after_final_run_status_commit(
    settings, monkeypatch, worker, status, paused, expected,
):
    from aegra_api.core.orm import Thread

    session = FakeSession()
    thread_id = "child" if worker else "parent"
    thread = SimpleNamespace(
        thread_id=thread_id, user_id="alice", status="busy",
        metadata_json=relationship() if worker else {"graph_id": "taskharbor"},
    )
    latest = SimpleNamespace(run_id="final-run", status=status, output=None) if status else None
    session.run = latest

    async def rows(statement):
        return [thread] if statement.column_descriptions[0]["entity"] is Thread else []

    async def state(*_):
        return {
            "interrupts": [{"id": "saved-wait", "value": {"kind": "awaiting_subtasks"}}]
            if paused else [],
            "values": {"messages": [{"content": "persisted"}]},
        }

    monkeypatch.setattr(session, "scalars", rows)
    monkeypatch.setattr(protocol, "_maker", lambda: lambda: session)
    monkeypatch.setattr(protocol, "_state", state)
    monkeypatch.setattr("aegra_api.api.runs.active_runs", {})
    asyncio.run(protocol.reconcile_orphaned_runs())
    assert thread.status == expected
    if paused:
        assert latest.status == "interrupted"
        assert latest.output["messages"][0]["content"] == "persisted"


def test_restart_does_not_repair_busy_thread_owned_by_live_run(settings, monkeypatch):
    from aegra_api.core.orm import Thread

    session = FakeSession()
    thread = SimpleNamespace(
        thread_id="parent", user_id="alice", status="busy",
        metadata_json={"graph_id": "taskharbor"},
    )
    session.run = SimpleNamespace(run_id="still-live", status="success")

    async def rows(statement):
        return [thread] if statement.column_descriptions[0]["entity"] is Thread else []

    state = AsyncMock()
    monkeypatch.setattr(session, "scalars", rows)
    monkeypatch.setattr(protocol, "_maker", lambda: lambda: session)
    monkeypatch.setattr(protocol, "_state", state)
    monkeypatch.setattr("aegra_api.api.runs.active_runs", {"still-live": object()})
    asyncio.run(protocol.reconcile_orphaned_runs())
    assert thread.status == "busy"
    state.assert_not_called()


def test_public_requests_preserve_service_owned_relationship_metadata(api, monkeypatch):
    app, session, _ = api
    child_metadata = relationship()
    session.rows.append(
        SimpleNamespace(thread_id="child", user_id="alice", metadata_json=child_metadata)
    )

    async def authorize(request, thread_id):
        return "alice"

    monkeypatch.setattr("taskharbor.http_app._authorized_thread_owner", authorize)

    @app.post("/threads")
    async def create(body: dict):
        return body

    @app.patch("/threads/{thread_id}")
    async def patch(thread_id: str, body: dict):
        return body

    app.add_middleware(protocol.WorkerResumeConfigMiddleware)
    with TestClient(app) as client:
        assert (
            client.post("/threads", json={"metadata": {"sandbox_worker": True}}).status_code == 403
        )
        assert (
            client.post("/threads", json={"metadata": {"thread_name": "Chat"}}).status_code == 200
        )
        assert (
            client.patch(
                "/threads/child", json={"metadata": {"parent_thread_id": "other"}}
            ).status_code
            == 403
        )
        assert client.patch("/threads/child", json={"metadata": child_metadata}).status_code == 200
        assert (
            client.patch(
                "/threads/child", json={"metadata": {"thread_name": "Renamed"}}
            ).status_code
            == 200
        )


def test_worker_cancellation_consumes_pending_requests_without_claiming_execution(monkeypatch):
    from taskharbor.security.repository import InMemoryApprovalStore

    store = InMemoryApprovalStore()
    row = store.prepare(
        {
            "invocation_key": "key",
            "owner": "alice",
            "thread_id": "child",
            "run_id": "run",
            "task_id": "child",
            "tool_call_id": "call",
            "tool_name": "execute_code",
            "group": "execution",
            "executor": "sandbox",
            "arguments": {},
            "request_digest": "digest",
            "reason": "Approval required",
        }
    )
    monkeypatch.setattr("taskharbor.security.repository.get_approval_store", lambda: store)
    protocol._cancel_worker_approvals("alice", "child")
    finished = store.get(row["id"], "alice", "child")
    assert finished["status"] == "cancelled"
    assert finished["result"] is not None
    assert not store.blocked_thread("alice", "child")


def test_upstream_state_only_cancellation_stops_exact_task(api, monkeypatch):
    _, session, _ = api
    session.rows = [
        SimpleNamespace(
            thread_id="cancelled",
            user_id="alice",
            status="interrupted",
            metadata_json=relationship(child="cancelled"),
        ),
        SimpleNamespace(
            thread_id="sibling",
            user_id="alice",
            status="busy",
            metadata_json=relationship(child="sibling"),
        ),
    ]
    cancelled = []

    async def state(*_):
        return {
            "values": {
                "async_tasks": {
                    "cancelled": {"status": "cancelled"},
                    "sibling": {"status": "running"},
                }
            }
        }

    async def cancel(owner, parent, *, task_ids):
        cancelled.append((owner, parent, task_ids))

    monkeypatch.setattr(protocol, "_state", state)
    monkeypatch.setattr(protocol, "cancel_child_tasks", cancel)
    asyncio.run(protocol._cancel_waiting_tasks())
    assert cancelled == [("alice", "parent", {"cancelled"})]


def test_parent_deletion_cleans_signed_children_after_parent_row_is_gone(api, monkeypatch):
    _, session, _ = api
    session.rows = [
        SimpleNamespace(thread_id=child, user_id="alice", metadata_json=relationship(child=child))
        for child in ("captured", "late")
    ]
    session.rows.append(
        SimpleNamespace(
            thread_id="unrelated",
            user_id="alice",
            metadata_json=relationship(parent="other", child="unrelated"),
        )
    )
    deleted = []

    async def delete(thread_id, **_):
        deleted.append(thread_id)

    saver = SimpleNamespace(adelete_thread=AsyncMock())

    @asynccontextmanager
    async def graph(graph_id):
        yield SimpleNamespace(checkpointer=saver)

    manager = SimpleNamespace(cleanup_thread_sessions=Mock())
    store = SimpleNamespace(revoke_thread=Mock())
    monkeypatch.setattr("aegra_api.api.threads.delete_thread", delete)
    monkeypatch.setattr(
        "aegra_api.services.langgraph_service.get_langgraph_service",
        lambda: SimpleNamespace(get_graph=graph),
    )
    monkeypatch.setattr("taskharbor.http_app._get_manager", lambda: manager)
    monkeypatch.setattr("taskharbor.security.repository.get_approval_store", lambda: store)
    asyncio.run(protocol.cleanup_deleted_children("alice", "parent", ["captured"]))
    assert deleted == ["captured", "late"]
    assert saver.adelete_thread.await_count == 2
    assert store.revoke_thread.call_count == 2
    assert manager.cleanup_thread_sessions.call_count == 2
    assert manager.cleanup_thread_sessions.call_args.kwargs == {
        "user_id": "alice",
        "delete_files": True,
    }


def test_parent_cancellation_is_limited_to_current_logical_turn(api, monkeypatch):
    _, session, _ = api
    children = []
    for child, turn in (("current", "this-turn"), ("earlier", "previous-turn")):
        children.append(
            SimpleNamespace(
                thread_id=child,
                user_id="alice",
                status="busy",
                metadata_json={**relationship(child=child), "sandbox_parent_turn": turn},
            )
        )
    session.rows = children
    session.run = SimpleNamespace(run_id="run", output={"partial": True})
    cancelled = []

    async def state(*_):
        return {"values": {"security_turn_id": "this-turn"}}

    async def cancel(thread_id, run_id, **_):
        cancelled.append(thread_id)

    monkeypatch.setattr(protocol, "_state", state)
    monkeypatch.setattr(protocol, "_cancel_worker_approvals", Mock())
    monkeypatch.setattr("taskharbor.http_app._cancel_security_work", Mock())
    monkeypatch.setattr("aegra_api.api.runs.cancel_run_endpoint", cancel)
    asyncio.run(protocol.cancel_child_tasks("alice", "parent"))
    assert cancelled == ["current"]
    assert children[1].status == "busy"


def test_concurrent_completion_notifications_submit_one_parent_resume(api, monkeypatch):
    _, session, _ = api
    parent = SimpleNamespace(
        thread_id="parent",
        user_id="alice",
        status="interrupted",
        metadata_json={"graph_id": "taskharbor"},
    )
    child = SimpleNamespace(
        thread_id="child", user_id="alice", status="idle", metadata_json=relationship()
    )
    session.rows = [parent, child]
    session.run = SimpleNamespace(run_id="run", status="success", config={}, context={})
    submissions = []

    async def state(*_):
        return {
            "interrupts": [
                {"id": "wait", "value": {"kind": "awaiting_subtasks", "task_ids": ["child"]}}
            ],
            "values": {"async_tasks": {"child": {"thread_id": "child", "run_id": "run"}}},
        }

    async def create(thread_id, request, **_):
        await asyncio.sleep(0)
        submissions.append(thread_id)
        parent.status = "busy"

    monkeypatch.setattr(protocol, "_state", state)
    monkeypatch.setattr("aegra_api.api.runs.create_run", create)

    async def run():
        await asyncio.gather(
            protocol.reconcile_waiting_parents(), protocol.reconcile_waiting_parents()
        )

    asyncio.run(run())
    assert submissions == ["parent"]


@pytest.mark.parametrize("operation", ["thread", "run"])
def test_cancelled_parent_turn_rejects_in_flight_launches(api, monkeypatch, operation):
    from aegra_api.core.orm import Thread

    from taskharbor.security.repository import InMemoryApprovalStore

    app, session, _ = api
    store = InMemoryApprovalStore()
    store.cancel_thread("alice", "parent")
    monkeypatch.setattr("taskharbor.security.repository.get_approval_store", lambda: store)

    async def state(*_):
        return {"values": {"security_turn_id": "turn", "security_generation": 0}}

    monkeypatch.setattr(protocol, "_state", state)
    create = AsyncMock()
    monkeypatch.setattr("aegra_api.api.runs.create_run", create)
    if operation == "run":
        session.rows.append(Thread(
            thread_id="child", user_id="alice", status="idle", metadata_json=relationship(),
        ))
        route = protocol._PREFIX + "/threads/child/runs"
        body = {"assistant_id": "taskharbor-worker", "input": {"messages": []}}
    else:
        route = protocol._PREFIX + "/threads"
        body = {"metadata": {"parent_thread_id": "parent", "subagent_name": "taskharbor-worker"}}
    with TestClient(app) as client:
        response = client.post(route, json=body)
    assert response.status_code == 409
    assert response.json()["detail"] == "The parent turn was cancelled"
    assert len(session.rows) == (1 if operation == "run" else 0)
    create.assert_not_called()


def test_cascade_cancellation_locks_parent_before_snapshot(api, monkeypatch):
    _, session, _ = api
    events = []
    owned = protocol._owned_thread
    scalars = session.scalars

    async def lock_parent(session, owner, thread_id, **options):
        events.append((thread_id, options.get("lock")))
        return await owned(session, owner, thread_id, **options)

    async def snapshot(statement):
        events.append("snapshot")
        return await scalars(statement)

    monkeypatch.setattr(protocol, "_owned_thread", lock_parent)
    monkeypatch.setattr(session, "scalars", snapshot)
    asyncio.run(protocol.cancel_child_tasks("alice", "parent"))
    assert events == [("parent", True), "snapshot"]


@pytest.mark.parametrize("update", [False, True])
def test_new_instruction_sets_persistent_task_deadline(api, settings, monkeypatch, update):
    app, session, _ = api
    metadata = {**relationship(), "graph_id": "taskharbor-worker"}
    if update:
        metadata["sandbox_task_deadline"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    child = SimpleNamespace(
        thread_id="child", user_id="alice", status="idle", metadata_json=metadata,
    )
    session.rows = [child]
    session.run = SimpleNamespace(config={}, context={})

    async def no_active_runs(statement):
        return []

    async def create(thread_id, request, **_):
        assert thread_id == "child"
        return SimpleNamespace(
            run_id="new-run", model_dump=lambda **_: {"run_id": "new-run"},
        )

    monkeypatch.setattr(session, "scalars", no_active_runs)
    monkeypatch.setattr("aegra_api.api.runs.create_run", create)
    before = datetime.now(UTC)
    with TestClient(app) as client:
        response = client.post(
            protocol._PREFIX + "/threads/child/runs",
            json={"assistant_id": "taskharbor-worker", "input": {"messages": [
                {"role": "user", "content": "new instruction"},
            ]}},
        )
    assert response.status_code == 200
    deadline = datetime.fromisoformat(child.metadata_json["sandbox_task_deadline"])
    assert before + timedelta(seconds=settings.SUBAGENT_TIMEOUT_SECONDS) <= deadline
    assert deadline <= datetime.now(UTC) + timedelta(seconds=settings.SUBAGENT_TIMEOUT_SECONDS)


def test_approval_resume_keeps_task_deadline_and_public_metadata_cannot_extend_it(api, monkeypatch):
    app, session, _ = api
    deadline = (datetime.now(UTC) + timedelta(seconds=30)).isoformat()
    child = SimpleNamespace(
        thread_id="child", user_id="alice", status="interrupted",
        metadata_json={**relationship(), "sandbox_task_deadline": deadline},
    )
    session.rows = [child]
    session.run = SimpleNamespace(config={"configurable": {"chat_model": "selected"}}, context={})

    async def authorize(request, thread_id):
        return "alice"

    monkeypatch.setattr("taskharbor.http_app._authorized_thread_owner", authorize)

    @app.post("/threads/{thread_id}/runs")
    async def resume(thread_id: str, body: dict):
        return body

    @app.patch("/threads/{thread_id}")
    async def patch(thread_id: str, body: dict):
        return body

    app.add_middleware(protocol.WorkerResumeConfigMiddleware)
    with TestClient(app) as client:
        resumed = client.post("/threads/child/runs", json={
            "assistant_id": "taskharbor-worker", "command": {"resume": {"approval": True}},
        })
        assert resumed.status_code == 200
        assert resumed.json()["config"]["configurable"]["chat_model"] == "selected"
        assert child.metadata_json["sandbox_task_deadline"] == deadline
        changed = client.patch("/threads/child", json={"metadata": {
            "sandbox_task_deadline": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
        }})
        assert changed.status_code == 403


@pytest.mark.parametrize(
    ("deadline", "old_run", "expired"),
    [
        ("past", False, True),
        ("future", True, False),
        (None, True, True),
        (None, False, False),
        ("invalid", True, True),
    ],
)
def test_expiry_preserves_deadline_across_resume_and_supports_legacy_workers(
    api, settings, monkeypatch, deadline, old_run, expired,
):
    _, session, _ = api
    now = datetime.now(UTC)
    metadata = relationship()
    if deadline:
        metadata["sandbox_task_deadline"] = (
            (now + timedelta(seconds=60 if deadline == "future" else -60)).isoformat()
            if deadline != "invalid" else "invalid"
        )
    child = SimpleNamespace(
        thread_id="child", user_id="alice", status="interrupted", metadata_json=metadata,
    )
    latest = SimpleNamespace(
        run_id="latest-resume", status="interrupted", output={"partial": True},
        created_at=now - timedelta(seconds=settings.SUBAGENT_TIMEOUT_SECONDS + 30)
        if old_run else now,
    )
    session.rows, session.run = [child], latest
    cancel = AsyncMock()
    monkeypatch.setattr("aegra_api.api.runs.cancel_run_endpoint", cancel)
    monkeypatch.setattr(protocol, "_cancel_worker_approvals", Mock())
    monkeypatch.setattr("taskharbor.http_app._cancel_security_work", Mock())
    asyncio.run(protocol._expire_workers())
    assert bool(cancel.await_count) is expired
    assert latest.status == ("timeout" if expired else "interrupted")
    assert child.status == ("idle" if expired else "interrupted")


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize(
    "case", ["expired", "pending", "cancelled", "active", "wrong_call", "old_generation"]
)
def test_expired_approval_auto_resume_is_scoped_and_never_approves(api, monkeypatch, worker, case):
    from taskharbor.security.repository import InMemoryApprovalStore

    _, session, _ = api
    now = [datetime(2026, 11, 19, tzinfo=UTC)]
    store = InMemoryApprovalStore(clock=lambda: now[0])
    thread_id = "child" if worker else "parent"
    metadata = relationship() if worker else {"graph_id": "taskharbor"}
    thread = SimpleNamespace(
        thread_id=thread_id, user_id="alice", status="interrupted", metadata_json=metadata
    )
    session.rows = [thread]
    session.run = SimpleNamespace(config={"configurable": {"chat_model": "chosen"}}, context={})
    record = store.prepare({
        "invocation_key": "expired-call", "owner": "alice", "thread_id": thread_id,
        "tool_call_id": "call-expired", "tool_name": "execute_terminal",
        "arguments": {}, "request_digest": "digest", "generation": 0,
        "group": "sandbox", "executor": "sandbox",
        "expires_at": (now[0] + timedelta(seconds=10)).isoformat(),
    })
    if case != "pending":
        now[0] += timedelta(seconds=11)
    if case in {"cancelled", "old_generation"}:
        store.cancel_thread("alice", thread_id)

    async def state(*_):
        return {
            "values": {"security_generation": 0},
            "interrupts": [
                {"id": "expired-interrupt", "value": {
                    "approval_id": record["id"],
                    "tool_call_id": "other" if case == "wrong_call" else "call-expired",
                }},
                {"id": "unanswered-sibling", "value": {"approval_id": "not-expired"}},
            ],
        }

    monkeypatch.setattr(protocol, "_state", state)
    monkeypatch.setattr("taskharbor.security.repository.get_approval_store", lambda: store)
    monkeypatch.setattr("taskharbor.security.http._has_active_run", lambda *_: case == "active")
    create = AsyncMock()
    monkeypatch.setattr("aegra_api.api.runs.create_run", create)
    asyncio.run(protocol.reconcile_expired_approvals())
    if case == "expired":
        create.assert_awaited_once()
        args = create.call_args.args
        assert args[0] == thread_id
        request = args[1]
        assert request.command == {"resume": {"expired-interrupt": {"approval_id": record["id"]}}}
        assert request.assistant_id == ("taskharbor-worker" if worker else "taskharbor")
        assert request.config["configurable"]["chat_model"] == "chosen"
        assert store.get(record["id"], "alice", thread_id)["status"] == "expired"
    else:
        create.assert_not_awaited()
