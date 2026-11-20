"""Opt-in end-to-end Docker acceptance tests for the sandbox refactor.

Run with SANDBOX_DOCKER_TESTS=1 and a local Docker daemon. These tests use
isolated temporary storage and destroy only the containers they create.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from taskharbor.sandbox.manager import SandboxManager, current_thread_id, current_user_id

pytestmark = pytest.mark.skipif(
    os.environ.get("SANDBOX_DOCKER_TESTS") != "1", reason="opt-in local Docker acceptance"
)


@pytest.fixture
def real_sandbox(tmp_path, monkeypatch):
    import taskharbor.sandbox.manager as module

    settings = SimpleNamespace(
        STORAGE_DIR=tmp_path,
        SANDBOX_STANDALONE_OWNER=None,
        SANDBOX_LEASE_TTL_SECONDS=30,
        SANDBOX_QUEUE_TIMEOUT_SECONDS=2,
        SANDBOX_MAX_WAITERS=4,
        SESSION_IDLE_TTL_SECONDS=600,
        SESSION_MAX_LIFETIME_SECONDS=7200,
        SESSION_GC_INTERVAL_SECONDS=60,
        CONTAINER_ORPHAN_MIN_AGE_SECONDS=300,
        CONTAINER_MAX_SESSIONS=2,
        CONTAINER_MAX_SESSIONS_PER_THREAD=3,
        CONTAINER_MEMORY_LIMIT="256m",
        CONTAINER_CPU_QUOTA=100000,
        CONTAINER_PIDS_LIMIT=64,
        CONTAINER_TMPFS_SIZE="32m",
        CONTAINER_NETWORK_ENABLED=False,
        CONTAINER_EXECUTE_AS_ROOT=False,
        CONTAINER_EXECUTION_TIMEOUT_SECONDS=5,
        CONTAINER_TERMINAL_TIMEOUT_SECONDS=10,
        CONTAINER_INSTALL_TIMEOUT_SECONDS=60,
        MAX_STDOUT_CHARS=50000,
        MAX_STDERR_CHARS=50000,
        MAX_RESULT_CHARS=30000,
        IMPORT_ALLOWED_DIRS=[],
    )
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    monkeypatch.setattr(SandboxManager, "_get_db_pool", lambda _: None)
    monkeypatch.setattr(SandboxManager, "_register_cleanup", lambda _: None)
    monkeypatch.setattr(SandboxManager, "_start_gc", lambda _: None)
    manager = SandboxManager()
    owner = current_user_id.set("acceptance-user")
    thread = current_thread_id.set("acceptance-thread")
    try:
        yield manager
    finally:
        manager.cleanup_all()
        current_thread_id.reset(thread)
        current_user_id.reset(owner)


def test_real_warm_reuse_retains_files_and_kernel_variables(real_sandbox):
    manager = real_sandbox
    manager.begin_turn("acceptance-user", "acceptance-thread", "one")
    info = manager.create_session()
    assert manager.execute_code(
        info.session_id,
        "x = 42\nfrom pathlib import Path\nPath('/workspace/data.txt').write_text('persist')",
    ).success
    manager.end_turn("acceptance-user", "acceptance-thread", "one")
    assert manager.sessions[info.session_id].status == "warm"
    manager.begin_turn("acceptance-user", "acceptance-thread", "two")
    reused = manager.create_session()
    assert reused.session_id == info.session_id and reused.reused
    result = manager.execute_code(
        reused.session_id, "print(x, Path('/workspace/data.txt').read_text())"
    )
    assert result.success and "42 persist" in result.stdout
    mounts = manager.client.api.inspect_container(info.container_id)["Mounts"]
    assert {m["Destination"]: m["RW"] for m in mounts}["/mnt/uploads"] is False
    manager.end_turn("acceptance-user", "acceptance-thread", "two")


def test_real_artifact_survives_rebuild_and_is_immutable(real_sandbox):
    manager = real_sandbox
    info = manager.create_session()
    assert (
        manager.execute_terminal(
            info.session_id, "printf original > /workspace/result.txt"
        ).exit_code
        == 0
    )
    exported = manager.export_files(info.session_id, [{"source": "result.txt"}]).files[0]
    assert exported.success and exported.artifact_id
    assert (
        manager.execute_terminal(
            info.session_id, "printf changed > /workspace/result.txt"
        ).exit_code
        == 0
    )
    manager.stop_session(info.session_id)
    assert (
        b"".join(
            manager.stream_exported_file(
                "acceptance-thread",
                info.session_id,
                exported.path,
                user_id="acceptance-user",
                artifact_id=exported.artifact_id,
            )
        )
        == b"original"
    )
    rebuilt = manager.create_session()
    assert rebuilt.environment_reset and rebuilt.session_id != info.session_id
    assert (
        manager.execute_terminal(rebuilt.session_id, "cat /workspace/result.txt").stdout
        == "changed"
    )
    from taskharbor.sandbox.storage import WorkspaceStore

    restarted_store = WorkspaceStore(manager._get_storage().root)
    artifact = restarted_store.find(
        "acceptance-user",
        "acceptance-thread",
        info.session_id,
        exported.path,
        artifact_id=exported.artifact_id,
    )
    assert b"".join(artifact.stream()) == b"original"
    assert (
        restarted_store.find(
            "other-user",
            "acceptance-thread",
            info.session_id,
            exported.path,
            artifact_id=exported.artifact_id,
        )
        is None
    )


def test_real_readonly_upload_and_imported_copy(real_sandbox):
    manager = real_sandbox
    info = manager.create_session()
    scope = manager._storage_scope(info)
    original = scope.uploads / "input.txt"
    original.write_text("input")
    result = manager.execute_terminal(info.session_id, "printf changed > /mnt/uploads/input.txt")
    assert result.exit_code != 0
    imported = manager.import_files(
        info.session_id, [{"source": str(original), "destination": "input.txt"}]
    )
    assert imported.success
    assert (
        manager.execute_terminal(info.session_id, "printf changed > /workspace/input.txt").exit_code
        == 0
    )
    assert original.read_text() == "input"
    assert Path(info.workspace, "input.txt").read_text() == "changed"


def test_real_code_timeout_destroys_execution_but_preserves_files(real_sandbox):
    manager = real_sandbox
    info = manager.create_session()
    result = manager.execute_code(
        info.session_id,
        "from pathlib import Path\nPath('/workspace/keep.txt').write_text('keep')\n"
        "import time\ntime.sleep(60)",
        timeout=1,
    )
    assert not result.success
    assert info.session_id not in manager.sessions
    recreated = manager.create_session()
    assert recreated.environment_reset
    assert (
        manager.execute_terminal(recreated.session_id, "cat /workspace/keep.txt").stdout == "keep"
    )


def test_real_terminal_timeout_keeps_healthy_kernel(real_sandbox):
    manager = real_sandbox
    info = manager.create_session()
    result = manager.execute_terminal(info.session_id, "sleep 60", timeout=1)
    assert result.timed_out and result.exit_code != 0
    assert manager.execute_code(info.session_id, "1 + 1").success
