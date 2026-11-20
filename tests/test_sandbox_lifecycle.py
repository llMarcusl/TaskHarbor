"""Warm lifecycle and durable mounts without requiring a Docker daemon."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from taskharbor.sandbox.manager import SandboxManager, current_thread_id, current_user_id


@pytest.fixture
def lifecycle_manager(tmp_path, monkeypatch):
    import taskharbor.sandbox.manager as module

    settings = SimpleNamespace(
        STORAGE_DIR=tmp_path,
        SANDBOX_STANDALONE_OWNER=None,
        SANDBOX_LEASE_TTL_SECONDS=10,
        SANDBOX_QUEUE_TIMEOUT_SECONDS=0.15,
        SANDBOX_MAX_WAITERS=4,
        SESSION_IDLE_TTL_SECONDS=600,
        SESSION_MAX_LIFETIME_SECONDS=7200,
        SESSION_GC_INTERVAL_SECONDS=60,
        CONTAINER_MAX_SESSIONS=2,
        CONTAINER_MAX_SESSIONS_PER_THREAD=3,
        CONTAINER_MEMORY_LIMIT="128m",
        CONTAINER_CPU_QUOTA=100000,
        CONTAINER_PIDS_LIMIT=64,
        CONTAINER_TMPFS_SIZE="16m",
        CONTAINER_NETWORK_ENABLED=False,
    )
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    client = MagicMock()

    def run(*args, **kwargs):
        container = MagicMock(id=kwargs["name"])
        container.status = "running"
        return container

    client.containers.run.side_effect = run
    monkeypatch.setattr(module.docker, "from_env", lambda: client)
    for name in (
        "_register_cleanup",
        "_cleanup_orphaned_containers",
        "_start_gc",
        "_ensure_exported_files_table",
        "_ensure_image",
        "_wait_for_kernel",
    ):
        monkeypatch.setattr(SandboxManager, name, lambda *a, **kw: None)
    manager = SandboxManager()
    owner_token = current_user_id.set("alice")
    thread_token = current_thread_id.set("thread-a")
    yield manager, settings, client
    manager.cleanup_all()
    current_user_id.reset(owner_token)
    current_thread_id.reset(thread_token)


def test_same_scope_reuses_and_different_identity_never_reuses(lifecycle_manager):
    manager, _, client = lifecycle_manager
    first = manager.create_session()
    assert manager.create_session().session_id == first.session_id
    assert manager.create_session().reused
    token = current_user_id.set("bob")
    try:
        second = manager.create_session()
        assert second.session_id != first.session_id
        assert second.workspace != first.workspace
        with pytest.raises(PermissionError):
            manager._check_session(first.session_id)
    finally:
        current_user_id.reset(token)
    assert client.containers.run.call_count == 2


def test_mounts_only_managed_workspace_and_readonly_uploads(lifecycle_manager):
    manager, settings, client = lifecycle_manager
    manager.create_session()
    args = client.containers.run.call_args.kwargs
    assert len(args["volumes"]) == 2
    assert all(str(settings.STORAGE_DIR) in path for path in args["volumes"])
    assert {v["bind"]: v["mode"] for v in args["volumes"].values()} == {
        "/workspace": "rw",
        "/mnt/uploads": "ro",
    }
    assert "/workspace" not in args["tmpfs"]
    assert "environment" not in args


def test_turn_lease_prevents_idle_gc_and_defers_age_retirement(lifecycle_manager):
    manager, _, _ = lifecycle_manager
    manager.begin_turn("alice", "thread-a", "run")
    info = manager.create_session()
    info.created_at = datetime.now(timezone.utc) - timedelta(hours=3)
    manager._released_at[info.session_id] = time.monotonic() - 10000
    manager._gc_threads()
    manager._gc_session_hard_cap()
    assert info.session_id in manager.sessions
    assert info.retire_requested
    assert info.status == "active"
    manager.end_turn("alice", "thread-a", "run")
    assert info.status == "warm"
    manager._gc_session_hard_cap()
    assert info.session_id not in manager.sessions


def test_expired_lease_starts_idle_clock_instead_of_instant_destruction(lifecycle_manager):
    manager, settings, _ = lifecycle_manager
    manager.begin_turn("alice", "thread-a", "abandoned")
    info = manager.create_session()
    manager._turn_leases[("alice", "thread-a", "abandoned")] = time.monotonic() - 1
    manager._gc_threads()
    assert info.status == "warm" and info.session_id in manager.sessions
    manager._released_at[info.session_id] -= settings.SESSION_IDLE_TTL_SECONDS + 1
    manager._gc_threads()
    assert info.session_id not in manager.sessions


def test_gc_keeps_pinned_operations(lifecycle_manager):
    manager, _, _ = lifecycle_manager
    info = manager.create_session()
    with manager._operation(info.session_id):
        info.created_at = datetime.now(timezone.utc) - timedelta(hours=3)
        manager._gc_session_hard_cap()
        assert info.session_id in manager.sessions
    manager._gc_session_hard_cap()
    assert info.session_id not in manager.sessions


def test_rebuild_preserves_workspace_but_announces_environment_reset(lifecycle_manager):
    from pathlib import Path

    manager, _, _ = lifecycle_manager
    info = manager.create_session()
    path = Path(info.workspace) / "data.txt"
    path.write_text("keep me")
    manager.stop_session(info.session_id)
    rebuilt = manager.create_session()
    assert rebuilt.session_id != info.session_id
    assert rebuilt.environment_reset and not rebuilt.reused
    assert Path(rebuilt.workspace, "data.txt").read_text() == "keep me"


def test_capacity_evicts_oldest_warm_never_active(lifecycle_manager):
    manager, _, client = lifecycle_manager
    first = manager.create_session()
    manager.begin_turn("alice", "thread-a", "active")
    current_thread_id.set("thread-b")
    second = manager.create_session()
    current_thread_id.set("thread-c")
    third = manager.create_session()
    assert set(manager.sessions) == {first.session_id, third.session_id}
    assert second.session_id not in manager.sessions
    assert client.containers.run.call_count == 3


def test_full_active_capacity_times_out_without_oversubscription(lifecycle_manager):
    manager, settings, client = lifecycle_manager
    settings.CONTAINER_MAX_SESSIONS = 1
    manager.begin_turn("alice", "thread-a", "run")
    manager.create_session()
    current_thread_id.set("thread-b")
    with pytest.raises(TimeoutError, match="active"):
        manager.create_session()
    assert client.containers.run.call_count == 1
    assert manager._waiters == 0


def test_capacity_waiter_wakes_after_release(lifecycle_manager):
    manager, settings, _ = lifecycle_manager
    settings.CONTAINER_MAX_SESSIONS = 1
    settings.SANDBOX_QUEUE_TIMEOUT_SECONDS = 2
    manager.begin_turn("alice", "thread-a", "run")
    first = manager.create_session()
    results = []

    def acquire():
        current_user_id.set("alice")
        current_thread_id.set("thread-b")
        results.append(manager.create_session())

    worker = threading.Thread(target=acquire)
    worker.start()
    time.sleep(0.05)
    assert len(manager.sessions) == 1
    manager.end_turn("alice", "thread-a", "run")
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert results[0].session_id != first.session_id
    assert len(manager.sessions) == 1


def test_kernel_serializes_parallel_operations(lifecycle_manager):
    manager, _, _ = lifecycle_manager
    info = manager.create_session()
    entered = threading.Event()

    def work():
        current_user_id.set("alice")
        current_thread_id.set("thread-a")
        with manager._operation(info.session_id):
            entered.set()

    with manager._operation(info.session_id):
        worker = threading.Thread(target=work)
        worker.start()
        assert not entered.wait(0.05)
    worker.join(timeout=2)
    assert entered.is_set()


def test_retiring_runtime_with_direct_execution_never_creates_second_writer(lifecycle_manager):
    manager, _, client = lifecycle_manager
    info = manager.create_session()
    info.retire_requested = True
    with manager._operation(info.session_id):
        with pytest.raises(TimeoutError, match="Retiring sandbox"):
            manager.create_session()
        assert info.session_id in manager.sessions
        assert client.containers.run.call_count == 1


def test_retiring_runtime_waits_for_execution_before_replacement(lifecycle_manager):
    manager, settings, client = lifecycle_manager
    settings.SANDBOX_QUEUE_TIMEOUT_SECONDS = 2
    first = manager.create_session()
    first.retire_requested = True
    results = []

    def acquire():
        current_user_id.set("alice")
        current_thread_id.set("thread-a")
        results.append(manager.create_session())

    with manager._operation(first.session_id):
        worker = threading.Thread(target=acquire)
        worker.start()
        time.sleep(0.05)
        assert client.containers.run.call_count == 1
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert results[0].session_id != first.session_id
    assert first.session_id not in manager.sessions
    assert len(manager.sessions) == 1


def test_reuse_recovers_if_gc_removes_candidate_before_pin(lifecycle_manager, monkeypatch):
    manager, _, client = lifecycle_manager
    first = manager.create_session()
    original = manager._operation

    @contextmanager
    def removed_before_pin(session_id):
        if session_id == first.session_id:
            manager.stop_session(session_id)
        with original(session_id):
            yield

    monkeypatch.setattr(manager, "_operation", removed_before_pin)
    replacement = manager.create_session()
    assert replacement.session_id != first.session_id
    assert replacement.environment_reset
    assert client.containers.run.call_count == 2


def test_failed_container_removal_preserves_operation_serialization(lifecycle_manager):
    manager, _, _ = lifecycle_manager
    info = manager.create_session()
    container = manager._containers[info.session_id]
    container.remove.side_effect = RuntimeError("Docker temporarily unavailable")
    entered = threading.Event()

    def execute():
        current_user_id.set("alice")
        current_thread_id.set("thread-a")
        with manager._operation(info.session_id):
            entered.set()

    with manager._operation(info.session_id):
        with pytest.raises(RuntimeError, match="temporarily unavailable"):
            manager.stop_session(info.session_id)
        assert info.status == "active"
        assert manager.thread_sessions["thread-a"] == {info.session_id}
        worker = threading.Thread(target=execute)
        worker.start()
        assert not entered.wait(0.05)
    worker.join(timeout=2)
    assert entered.is_set()
    container.remove.side_effect = None


def test_cancel_destroys_execution_but_keeps_persistent_files(lifecycle_manager):
    from pathlib import Path

    manager, _, _ = lifecycle_manager
    info = manager.create_session()
    path = Path(info.workspace, "keep.txt")
    path.write_text("saved")
    assert manager.cancel_execution(info.session_id)
    assert info.session_id not in manager.sessions
    assert path.read_text() == "saved"


def test_cancel_cannot_destroy_another_users_runtime(lifecycle_manager):
    manager, _, _ = lifecycle_manager
    info = manager.create_session()
    token = current_user_id.set("bob")
    try:
        with pytest.raises(PermissionError):
            manager.cancel_execution(info.session_id)
        assert info.session_id in manager.sessions
    finally:
        current_user_id.reset(token)


def test_cancelled_generation_cannot_rebuild_until_host_transfer_finishes(lifecycle_manager):
    manager, settings, client = lifecycle_manager
    settings.SANDBOX_QUEUE_TIMEOUT_SECONDS = 2
    info = manager.create_session()
    results = []

    def acquire():
        current_user_id.set('alice')
        current_thread_id.set('thread-a')
        results.append(manager.create_session())

    with manager._operation(info.session_id):
        manager.stop_session(info.session_id)
        worker = threading.Thread(target=acquire)
        worker.start()
        time.sleep(0.05)
        assert client.containers.run.call_count == 1
        assert not results
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert results[0].session_id != info.session_id
    assert not manager._scope_operations


def test_abandoned_same_scope_container_removed_before_new_mount(lifecycle_manager):
    manager, _, client = lifecycle_manager
    orphan = MagicMock(id='orphan')
    orphan.labels = {'manager-id': 'a' * 32}
    client.containers.list.return_value = [orphan]
    manager.create_session()
    orphan.remove.assert_called_once_with(force=True)


def test_abandoned_writer_removal_failure_prevents_new_mount(lifecycle_manager):
    manager, _, client = lifecycle_manager
    orphan = MagicMock(id='orphan')
    orphan.labels = {'manager-id': 'a' * 32}
    orphan.remove.side_effect = RuntimeError('cannot remove old writer')
    client.containers.list.return_value = [orphan]
    with pytest.raises(RuntimeError, match='old writer'):
        manager.create_session()
    client.containers.run.assert_not_called()
