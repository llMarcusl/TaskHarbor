"""HTTP checks for active runs, pending approvals, and cancellation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.testclient import TestClient

from taskharbor.security.http import SecurityRunMiddleware, _has_active_run
from taskharbor.security.repository import InMemoryApprovalStore


@pytest.fixture
def guard(monkeypatch):
    store = InMemoryApprovalStore()
    identity = {"owner": "alice"}
    settings = SimpleNamespace(HOST_COMMANDS_ENABLED=True, TOOL_SECURITY_RULES=[])
    active_run = Mock(return_value=False)
    cleanup = Mock()
    calls = []

    async def authorize(request, thread_id):
        if identity["owner"] is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        if identity["owner"] != "alice" or thread_id != "t":
            raise HTTPException(status_code=404, detail="Thread not found")
        return "alice"

    monkeypatch.setattr("taskharbor.http_app._authorized_thread_owner", authorize)
    monkeypatch.setattr("taskharbor.http_app._cancel_security_work", cleanup)
    monkeypatch.setattr("taskharbor.settings.get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.security.http.get_approval_store", lambda: store)
    monkeypatch.setattr("taskharbor.security.http._has_active_run", active_run)
    app = FastAPI()

    @app.post("/threads/{thread_id}/runs")
    @app.post("/threads/{thread_id}/runs/stream")
    @app.post("/threads/{thread_id}/runs/wait")
    @app.post("/threads/{thread_id}/runs/{run_id}/cancel")
    async def downstream(thread_id: str, request: Request):
        calls.append(request.url.path)
        if request.path_params.get("run_id") == "missing":
            raise HTTPException(status_code=404, detail="Run not found")
        return {"status": "accepted"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.add_middleware(SecurityRunMiddleware)
    record = store.prepare(
        {
            "invocation_key": "operation",
            "owner": "alice",
            "thread_id": "t",
            "run_id": "run",
            "tool_call_id": "call",
            "tool_name": "custom_action",
            "group": "integration",
            "executor": "connector",
            "arguments": {},
            "request_digest": "digest",
        }
    )
    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client,
            store=store,
            record=record,
            identity=identity,
            settings=settings,
            active_run=active_run,
            cleanup=cleanup,
            calls=calls,
        )


@pytest.mark.parametrize("suffix", ["", "/stream", "/wait"])
def test_pending_decision_blocks_all_new_run_routes(guard, suffix):
    response = guard.client.post(f"/threads/t/runs{suffix}", json={"input": {"messages": []}})
    assert response.status_code == 409
    assert guard.calls == []
    guard.active_run.assert_called_once_with("alice", "t")


def test_denied_decision_blocks_new_run_until_graph_consumes_it(guard):
    guard.store.decide(guard.record["id"], "alice", "t", "deny")
    assert guard.client.post("/threads/t/runs", json={}).status_code == 409
    guard.store.finish(guard.record["id"], "denied", {"type": "tool", "content": "denied"})
    assert guard.client.post("/threads/t/runs", json={}).status_code == 200


def test_resume_is_admitted_and_preserves_approval(guard):
    guard.store.decide(guard.record["id"], "alice", "t", "allow")
    response = guard.client.post(
        "/threads/t/runs/stream", json={"command": {"resume": {"approval_id": guard.record["id"]}}}
    )
    assert response.status_code == 200
    assert guard.store.get(guard.record["id"], "alice", "t")["status"] == "approved"


@pytest.mark.parametrize("owner,status", [(None, 401), ("bob", 404)])
def test_unauthorized_run_cannot_query_or_cancel_work(guard, owner, status):
    guard.identity["owner"] = owner
    assert guard.client.post("/threads/t/runs", json={}).status_code == status
    guard.active_run.assert_not_called()
    guard.cleanup.assert_not_called()
    assert guard.store.get(guard.record["id"], "alice", "t")["status"] == "pending"


def test_active_run_blocks_new_request_and_resume(guard):
    guard.active_run.return_value = True
    for body in ({}, {"command": {"resume": {"approval_id": guard.record["id"]}}}):
        assert guard.client.post("/threads/t/runs", json=body).status_code == 409
    assert not guard.calls


def test_run_lookup_failure_returns_service_unavailable(guard):
    guard.active_run.side_effect = RuntimeError("secret DB connection details")
    response = guard.client.post("/threads/t/runs", json={})
    assert response.status_code == 503
    assert "secret" not in response.text
    assert not guard.calls


def test_cancel_invalidates_approval_before_stopping_execution(guard):
    guard.active_run.return_value = True

    def cleanup(owner, thread):
        assert guard.store.get(guard.record["id"], owner, thread)["status"] == "cancelled"
        assert guard.calls == []

    guard.cleanup.side_effect = cleanup
    assert guard.client.post("/threads/t/runs/run/cancel", json={}).status_code == 200
    assert guard.store.generation("alice", "t") == 1
    guard.active_run.assert_called_once_with("alice", "t", run_id="run")
    guard.cleanup.assert_called_once_with("alice", "t")


@pytest.mark.parametrize("run_id,status", [("old", 200), ("missing", 404)])
def test_inactive_cancel_uses_aegra_response_without_cancelling_new_work(guard, run_id, status):
    response = guard.client.post(f"/threads/t/runs/{run_id}/cancel", json={})
    assert response.status_code == status
    assert guard.calls == [f"/threads/t/runs/{run_id}/cancel"]
    guard.active_run.assert_called_once_with("alice", "t", run_id=run_id)
    assert guard.store.generation("alice", "t") == 0
    assert guard.store.get(guard.record["id"], "alice", "t")["status"] == "pending"
    guard.cleanup.assert_not_called()


def test_cancel_lookup_failure_does_not_cancel_or_call_aegra(guard):
    guard.active_run.side_effect = RuntimeError("secret DB connection details")
    response = guard.client.post("/threads/t/runs/run/cancel", json={})
    assert response.status_code == 503
    assert "secret" not in response.text
    assert not guard.calls
    assert guard.store.generation("alice", "t") == 0
    guard.cleanup.assert_not_called()


def test_cancel_storage_failure_does_not_stop_execution(guard, monkeypatch):
    guard.active_run.return_value = True
    monkeypatch.setattr(guard.store, "cancel_thread", Mock(side_effect=RuntimeError("DB failed")))
    assert guard.client.post("/threads/t/runs/run/cancel", json={}).status_code == 503
    assert not guard.calls
    guard.cleanup.assert_not_called()


@pytest.mark.parametrize("run_id", [None, "run"])
@pytest.mark.parametrize("active", [False, True])
def test_active_lookup_returns_connection_and_checks_owned_run(monkeypatch, run_id, active):
    connection = Mock()
    connection.execute.return_value.fetchone.return_value = {"exists": 1} if active else None
    checkout = MagicMock()
    checkout.__enter__.return_value = connection
    pool = Mock()
    pool.connection.return_value = checkout
    monkeypatch.setattr("taskharbor.clients.infra.get_db_pool", lambda: pool)

    assert _has_active_run("alice", "t", run_id=run_id) is active
    query, parameters = connection.execute.call_args.args
    assert "user_id=%s AND thread_id=%s" in query
    assert "status IN ('pending','running')" in query
    assert parameters == (("alice", "t", run_id) if run_id else ("alice", "t"))
    assert ("run_id=%s" in query) is (run_id is not None)
    checkout.__exit__.assert_called_once_with(None, None, None)


def test_active_lookup_returns_connection_on_query_error(monkeypatch):
    connection = Mock()
    error = RuntimeError("query failed")
    connection.execute.side_effect = error
    checkout = MagicMock()
    checkout.__enter__.return_value = connection
    pool = Mock()
    pool.connection.return_value = checkout
    monkeypatch.setattr("taskharbor.clients.infra.get_db_pool", lambda: pool)

    with pytest.raises(RuntimeError, match="query failed"):
        _has_active_run("alice", "t")
    checkout.__exit__.assert_called_once()
    assert checkout.__exit__.call_args.args[:2] == (RuntimeError, error)


def test_default_sandbox_only_requests_do_not_query_approval_storage(guard):
    guard.settings.HOST_COMMANDS_ENABLED = False
    response = guard.client.post("/threads/t/runs", json={})
    assert response.status_code == 200
    guard.active_run.assert_not_called()


def test_unrelated_routes_skip_security_checks(guard):
    assert guard.client.get("/health").status_code == 200
    guard.active_run.assert_not_called()


@pytest.mark.parametrize(
    "configurable",
    [
        {"thread_id": "another-thread"},
        {"user_id": "bob"},
        {"langgraph_auth_user": {"identity": "bob"}},
    ],
)
@pytest.mark.parametrize("security_enabled", [False, True])
@pytest.mark.parametrize("placement", ["configurable", "context"])
def test_run_cannot_override_server_identity(guard, configurable, security_enabled, placement):
    guard.settings.HOST_COMMANDS_ENABLED = security_enabled
    body = (
        {"context": configurable}
        if placement == "context"
        else {"config": {"configurable": configurable}}
    )
    response = guard.client.post(
        "/threads/t/runs",
        json=body,
    )
    assert response.status_code == 400
    assert guard.calls == []
    guard.active_run.assert_not_called()


def test_subagents_serialize_runs_when_host_security_is_disabled(guard):
    guard.settings.HOST_COMMANDS_ENABLED = False
    guard.settings.SUBAGENTS_ENABLED = True
    guard.active_run.return_value = True
    assert guard.client.post("/threads/t/runs", json={}).status_code == 409
    assert guard.calls == []
