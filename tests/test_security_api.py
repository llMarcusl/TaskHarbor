"""Approval HTTP endpoints authenticate, scope records and never execute tools."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from starlette.testclient import TestClient

from taskharbor.security.api import create_security_router
from taskharbor.security.repository import InMemoryApprovalStore


@pytest.fixture
def api():
    now = [datetime(2026, 11, 10, tzinfo=UTC)]
    store = InMemoryApprovalStore(clock=lambda: now[0])
    identity = {"owner": "alice", "exists": True}
    cleanup = Mock()

    async def authenticate(request):
        if identity["owner"] is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        return identity["owner"]

    async def authorize(request, thread_id):
        owner = await authenticate(request)
        if owner != "alice" or thread_id != "thread-1" or not identity["exists"]:
            raise HTTPException(status_code=404, detail="Thread not found")
        return owner

    app = FastAPI()
    app.include_router(
        create_security_router(
            lambda: store,
            authorize,
            authenticate,
            on_cancel=cleanup,
            is_run_active=lambda owner, thread_id: identity.get("active", False),
        )
    )
    record = store.prepare(
        {
            "invocation_key": "invocation-1",
            "owner": "alice",
            "thread_id": "thread-1",
            "run_id": "run-1",
            "tool_call_id": "call-1",
            "batch_index": 0,
            "tool_name": "execute_host_command",
            "group": "terminal",
            "executor": "host",
            "arguments": {"command": "Get-Date"},
            "request_digest": "digest-1",
            "reason": "Host command requires confirmation",
            "expires_at": (now[0] + timedelta(minutes=30)).isoformat(),
        }
    )
    with TestClient(app) as client:
        yield client, store, record, identity, now, cleanup


def decision_url(record, thread_id="thread-1"):
    return f"/threads/{thread_id}/approvals/{record['id']}/decision"


def test_list_hides_authorization_internals_and_preserves_display_details(api):
    client, _, row, _, _, _ = api
    response = client.get("/threads/thread-1/approvals")
    assert response.status_code == 200
    record = response.json()["approvals"][0]
    assert record["id"] == row["id"]
    assert record["arguments"] == {"command": "Get-Date"}
    assert record["status"] == "pending"
    assert record["batch_index"] == 0
    assert not {"owner", "invocation_key", "request_digest", "generation"} & record.keys()


@pytest.mark.parametrize("operation", ["list", "decision", "cancel"])
def test_endpoints_require_authentication(api, operation):
    client, store, row, identity, _, cleanup = api
    identity["owner"] = None
    if operation == "list":
        response = client.get("/threads/thread-1/approvals")
    elif operation == "decision":
        response = client.post(decision_url(row), json={"decision": "allow"})
    else:
        response = client.post("/threads/thread-1/approvals/cancel")
    assert response.status_code == 401
    assert store.get(row["id"], "alice", "thread-1")["status"] == "pending"
    cleanup.assert_not_called()


@pytest.mark.parametrize("wrong_identity", ["owner", "thread"])
def test_decision_does_not_leak_foreign_records(api, wrong_identity):
    client, store, row, identity, _, _ = api
    if wrong_identity == "owner":
        identity["owner"] = "bob"
    url = decision_url(row, "thread-2" if wrong_identity == "thread" else "thread-1")
    assert client.post(url, json={"decision": "allow"}).status_code == 404
    assert store.get(row["id"], "alice", "thread-1")["status"] == "pending"


def test_decision_only_records_never_claims_or_executes(api):
    client, store, row, _, _, cleanup = api
    response = client.post(decision_url(row), json={"decision": "allow"})
    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    assert store.get(row["id"], "alice", "thread-1")["status"] == "approved"
    cleanup.assert_not_called()
    assert client.post(decision_url(row), json={"decision": "allow"}).status_code == 200
    assert client.post(decision_url(row), json={"decision": "deny"}).status_code == 409


def test_resume_or_approval_flags_cannot_forge_decision(api):
    client, _, row, _, _, _ = api
    assert client.post(decision_url(row), json={"approved": True}).status_code == 422
    assert (
        client.post(
            decision_url(row), json={"decision": "allow", "arguments": {"command": "other"}}
        ).status_code
        == 422
    )


def test_decision_and_poll_report_active_execution_without_starting_another_run(api):
    client, store, row, identity, _, cleanup = api
    identity["active"] = True
    response = client.post(decision_url(row), json={"decision": "allow"})
    assert response.status_code == 200
    assert response.json()["execution_active"] is True
    assert response.json()["status"] == "approved"
    assert client.get("/threads/thread-1/approvals").json()["approvals"][0]["execution_active"]
    identity["active"] = False
    assert not client.get("/threads/thread-1/approvals").json()["approvals"][0]["execution_active"]
    assert store.get(row["id"], "alice", "thread-1")["status"] == "approved"
    cleanup.assert_not_called()


def test_expired_request_returns_gone_and_cannot_execute(api):
    client, store, row, _, now, _ = api
    now[0] += timedelta(minutes=31)
    assert client.post(decision_url(row), json={"decision": "allow"}).status_code == 410
    assert store.get(row["id"], "alice", "thread-1")["status"] == "expired"


def test_cancel_records_cancellation_before_cleanup_and_allows_next_generation(api):
    client, store, row, _, _, cleanup = api

    def check_cancelled(owner, thread_id):
        assert store.get(row["id"], owner, thread_id)["status"] == "cancelled"

    cleanup.side_effect = check_cancelled
    response = client.post("/threads/thread-1/approvals/cancel")
    assert response.status_code == 200
    assert store.generation("alice", "thread-1") == 1
    cleanup.assert_called_once_with("alice", "thread-1")
    assert client.post(decision_url(row), json={"decision": "allow"}).status_code == 409


def test_deleted_thread_cannot_be_approved_before_delayed_revocation(api):
    client, store, row, identity, _, _ = api
    identity["exists"] = False
    assert client.get("/threads/thread-1/approvals").status_code == 404
    assert client.post(decision_url(row), json={"decision": "allow"}).status_code == 404
    assert store.get(row["id"], "alice", "thread-1")["status"] == "pending"


def test_storage_failure_fails_closed_without_leaking_details(api, monkeypatch):
    client, store, row, _, _, _ = api

    def unavailable(*args):
        raise RuntimeError("postgresql://secret-password@database")

    monkeypatch.setattr(store, "decide", unavailable)
    response = client.post(decision_url(row), json={"decision": "allow"})
    assert response.status_code == 503
    assert "secret-password" not in response.text
    assert store.get(row["id"], "alice", "thread-1")["status"] == "pending"


def test_cleanup_failure_does_not_reverse_cancellation(api):
    client, store, row, _, _, cleanup = api
    cleanup.side_effect = RuntimeError("Docker is unavailable")
    assert client.post("/threads/thread-1/approvals/cancel").status_code == 503
    assert store.get(row["id"], "alice", "thread-1")["status"] == "cancelled"


def test_parent_routes_child_approval_to_its_execution_thread():
    store = InMemoryApprovalStore()
    child = store.prepare(
        {
            "invocation_key": "child-call",
            "owner": "alice",
            "thread_id": "child",
            "task_id": "child",
            "tool_call_id": "call",
            "tool_name": "custom_action",
            "group": "integration",
            "executor": "connector",
            "arguments": {},
            "request_digest": "digest",
        }
    )
    unrelated = store.prepare(
        {
            **child,
            "id": "unrelated-approval",
            "invocation_key": "unrelated-call",
            "thread_id": "unrelated",
        }
    )
    association = {"children": ["child"]}

    async def authorize(request, thread_id):
        if thread_id != "parent":
            raise HTTPException(status_code=404, detail="Thread not found")
        return "alice"

    async def children(owner, parent_thread):
        assert (owner, parent_thread) == ("alice", "parent")
        return association["children"]

    async def interrupts(owner, thread_id):
        assert owner == "alice"
        return [{"id": "interrupt-child", "value": {"approval_id": child["id"]}}]

    app = FastAPI()
    app.include_router(
        create_security_router(
            lambda: store,
            authorize,
            authorize,
            related_threads=children,
            get_interrupts=interrupts,
        )
    )
    with TestClient(app) as client:
        rows = client.get("/threads/parent/approvals").json()["approvals"]
        assert len(rows) == 1
        assert rows[0]["thread_id"] == "parent"
        assert rows[0]["execution_thread_id"] == "child"
        assert rows[0]["assistant_id"] == "taskharbor-worker"
        assert rows[0]["interrupt_id"] == "interrupt-child"
        response = client.post(
            f"/threads/parent/approvals/{child['id']}/decision",
            json={"decision": "allow"},
        )
        assert response.status_code == 200
        assert response.json()["execution_thread_id"] == "child"
        assert store.get(child["id"], "alice", "child")["status"] == "approved"
        assert (
            client.post(
                f"/threads/parent/approvals/{unrelated['id']}/decision",
                json={"decision": "allow"},
            ).status_code
            == 404
        )
        association["children"] = []
        assert (
            client.post(
                f"/threads/parent/approvals/{child['id']}/decision",
                json={"decision": "allow"},
            ).status_code
            == 404
        )
