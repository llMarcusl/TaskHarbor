"""Ownership and turn leases at the Agent and HTTP boundaries."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from starlette.testclient import TestClient

from taskharbor.agent.middleware import SandboxLifecycleMiddleware, ThreadContextMiddleware
from taskharbor.sandbox.manager import SandboxManager, current_thread_id, current_user_id
from taskharbor.sandbox.storage import WorkspaceStore


@pytest.fixture
def http_context(monkeypatch, tmp_path):
    from taskharbor.http_app import app

    manager = MagicMock(spec=SandboxManager)
    manager.client = MagicMock()
    manager.resolve_owner.return_value = "alice"
    store = WorkspaceStore(tmp_path)
    manager._get_storage.return_value = store
    auth = AsyncMock(return_value=SimpleNamespace(identity="alice"))
    monkeypatch.setattr("taskharbor.http_app._get_manager", lambda: manager)
    monkeypatch.setattr("aegra_api.core.auth_deps.require_auth", auth)
    monkeypatch.setattr(
        "taskharbor.settings.get_settings",
        lambda: SimpleNamespace(
            SESSION_MEMORY_ENABLED=False,
            LONG_TERM_MEMORY_ENABLED=False,
            CONTEXT_COMPACTION_ENABLED=False,
        ),
    )
    with TestClient(app) as client:
        yield client, manager, store, auth


def test_other_user_cannot_download_or_upload(http_context):
    client, manager, _, auth = http_context
    auth.return_value.identity = "mallory"
    response = client.get(
        "/threads/t/files/download", params={"session_id": "s", "path": "/workspace/a"}
    )
    assert response.status_code == 404
    response = client.post("/threads/t/files/upload", files={"files": ("a.txt", b"secret")})
    assert response.status_code == 404
    manager.is_file_exported.assert_not_called()
    manager._get_storage.assert_not_called()


@pytest.mark.parametrize("operation", ["download", "upload", "list", "kill"])
def test_file_and_session_routes_require_auth(http_context, operation):
    client, manager, _, auth = http_context
    auth.side_effect = HTTPException(status_code=401, detail="Authentication required")
    if operation == "download":
        response = client.get(
            "/threads/t/files/download", params={"session_id": "s", "path": "/workspace/a"}
        )
    elif operation == "upload":
        response = client.post("/threads/t/files/upload", files={"files": ("a.txt", b"a")})
    elif operation == "list":
        response = client.get("/sessions")
    else:
        response = client.delete("/sessions/s")
    assert response.status_code == 401
    manager.resolve_owner.assert_not_called()
    manager.stop_session.assert_not_called()


def test_uploads_are_scoped_and_duplicate_originals_are_preserved(http_context):
    client, _, store, _ = http_context
    first = client.post("/threads/t/files/upload", files={"files": ("input.csv", b"first")})
    second = client.post("/threads/t/files/upload", files={"files": ("input.csv", b"second")})
    assert first.status_code == second.status_code == 200
    first_path = first.json()[0]["path"]
    second_path = second.json()[0]["path"]
    assert first_path != second_path
    uploads = store.scope("alice", "t", "python").uploads
    assert sorted(p.read_bytes() for p in uploads.iterdir()) == [b"first", b"second"]
    assert not list(store.scope("bob", "t", "python").uploads.iterdir())


@pytest.mark.parametrize("filename", ["../secret", "/outside", "C:outside", "dir/../../a"])
def test_upload_rejects_paths_outside_its_scope(http_context, filename):
    client, _, store, _ = http_context
    response = client.post("/threads/t/files/upload", files={"files": (filename, b"a")})
    assert response.status_code == 400
    assert not list(store.scope("alice", "t", "python").uploads.iterdir())


def test_session_kill_cannot_fall_back_to_foreign_orphan_container(http_context):
    client, manager, _, _ = http_context
    manager.sessions = {"foreign": SimpleNamespace(user_id="bob", thread_id="b")}
    assert client.delete("/sessions/foreign").status_code == 404
    assert client.delete("/sessions/untracked").status_code == 404
    manager.stop_session.assert_not_called()
    manager.client.containers.list.assert_not_called()


def test_download_passes_trusted_owner_and_immutable_artifact_id(http_context):
    client, manager, _, _ = http_context
    manager.is_file_exported.return_value = True
    manager.is_exported_path_file.return_value = True
    manager.stream_exported_file.return_value = iter([b"published"])
    response = client.get(
        "/threads/t/files/download",
        params={"session_id": "s", "path": "/workspace/report.txt", "artifact_id": "a1"},
    )
    assert response.content == b"published"
    manager.stream_exported_file.assert_called_once_with(
        "t", "s", "/workspace/report.txt", user_id="alice", artifact_id="a1"
    )


def test_tool_context_ignores_client_identity_and_restores_outer_context():
    manager = MagicMock(spec=SandboxManager)
    manager.resolve_owner.return_value = "alice"
    middleware = ThreadContextMiddleware(manager)
    request = SimpleNamespace(
        runtime=SimpleNamespace(config={"configurable": {"thread_id": "t", "user_id": "bob"}})
    )
    thread_token = current_thread_id.set("outer-thread")
    owner_token = current_user_id.set("outer-owner")
    try:
        def handler(_):
            assert current_thread_id.get() == "t"
            assert current_user_id.get() == "alice"
            raise RuntimeError("tool failed")

        with pytest.raises(RuntimeError):
            middleware.wrap_tool_call(request, handler)
        assert current_thread_id.get() == "outer-thread"
        assert current_user_id.get() == "outer-owner"
    finally:
        current_user_id.reset(owner_token)
        current_thread_id.reset(thread_token)


@pytest.mark.parametrize("async_mode", [False, True])
def test_turn_lease_heartbeat_stops_and_releases_on_failure(async_mode):
    manager = MagicMock(spec=SandboxManager)
    manager.resolve_owner.return_value = "alice"
    renewed = threading.Event()
    calls = []

    def renew(*identity, execution_scope=None):
        assert execution_scope is None
        calls.append(identity)
        if len(calls) > 1:
            renewed.set()

    manager.renew_turn.side_effect = renew
    middleware = SandboxLifecycleMiddleware(manager, lease_ttl_seconds=0.03)
    runtime = SimpleNamespace(config={"configurable": {"thread_id": "t", "user_id": "bob"}})
    state = middleware.before_agent({"sandbox_turn_id": "old-checkpoint"}, runtime)
    assert state["sandbox_turn_id"] != "old-checkpoint"
    request = SimpleNamespace(state=state, runtime=runtime)

    def fail(_):
        assert renewed.wait(1), "lease must remain live while waiting for the model"
        raise RuntimeError("model unavailable")

    async def afail(_):
        await asyncio.to_thread(renewed.wait, 1)
        return fail(None)

    with pytest.raises(RuntimeError, match="model unavailable"):
        if async_mode:
            asyncio.run(middleware.awrap_model_call(request, afail))
        else:
            middleware.wrap_model_call(request, fail)
    manager.begin_turn.assert_called_once_with(
        "alice", "t", state["sandbox_turn_id"], execution_scope=None,
    )
    manager.end_turn.assert_called_once_with(
        "alice", "t", state["sandbox_turn_id"], execution_scope=None,
    )
    assert not any(t.name == "sandbox-turn-lease" and t.is_alive() for t in threading.enumerate())



def test_async_cancellation_cancels_underlying_sandbox_work_before_release():
    manager = MagicMock()
    manager.resolve_owner.return_value = "alice"
    middleware = SandboxLifecycleMiddleware(manager)
    runtime = SimpleNamespace(config={"configurable": {"thread_id": "t"}})
    state = middleware.before_agent({}, runtime)
    request = SimpleNamespace(state=state, runtime=runtime)

    async def cancelled_handler(_):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(middleware.awrap_tool_call(request, cancelled_handler))
    identity = ("alice", "t", state["sandbox_turn_id"])
    manager.cancel_turn.assert_called_once_with(*identity, execution_scope=None)
    manager.end_turn.assert_called_once_with(*identity, execution_scope=None)
    calls = [call[0] for call in manager.mock_calls]
    assert calls.index("cancel_turn") < calls.index("end_turn")
