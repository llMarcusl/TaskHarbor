from contextlib import contextmanager
from pathlib import Path

import pytest

from taskharbor.sandbox.manager import (
    current_execution_scope,
    current_thread_id,
    current_user_id,
)
from tests.test_sandbox_lifecycle import lifecycle_manager as _lifecycle_manager


@pytest.fixture
def lifecycle_manager(tmp_path, monkeypatch):
    yield from _lifecycle_manager.__wrapped__(tmp_path, monkeypatch)


@contextmanager
def scope(value):
    token = current_execution_scope.set(value)
    try:
        yield
    finally:
        current_execution_scope.reset(token)


def freezeable(manager, info):
    container = manager._containers[info.session_id]
    container.pause.side_effect = lambda: setattr(container, "status", "paused")
    container.unpause.side_effect = lambda: setattr(container, "status", "running")


def test_same_thread_workers_have_independent_runtime_workspace_and_uploads(lifecycle_manager):
    manager, settings, _ = lifecycle_manager
    settings.CONTAINER_MAX_SESSIONS = 4
    root = manager.create_session()
    with scope("task-a"):
        a = manager.create_session()
        assert manager.create_session().session_id == a.session_id
    with scope("task-b"):
        b = manager.create_session()
        with pytest.raises(PermissionError, match="another task"):
            manager._check_session(a.session_id)
    assert len({root.session_id, a.session_id, b.session_id}) == 3
    assert len({root.workspace, a.workspace, b.workspace}) == 3
    paths = [manager._storage_scope(info) for info in (root, a, b)]
    assert len({p.uploads for p in paths}) == 3
    assert len({p.execution_root for p in paths}) == 3
    assert len({p.outputs for p in paths}) == 1
    assert manager.client.containers.run.call_args.kwargs["labels"]["execution-scope"] == "task-b"
    with pytest.raises(PermissionError, match="another task"):
        manager._check_session(a.session_id)


def test_leases_and_scoped_cancel_do_not_touch_sibling_worker(lifecycle_manager):
    manager, _, _ = lifecycle_manager
    with scope("task-a"):
        a = manager.create_session()
    with scope("task-b"):
        b = manager.create_session()
    manager.begin_turn("alice", "thread-a", "run", execution_scope="task-a")
    assert manager._has_turn_locked(a)
    assert not manager._has_turn_locked(b)
    manager._active_ops[a.session_id] = manager._active_ops[b.session_id] = 1
    manager.cancel_turn("alice", "thread-a", "run", execution_scope="task-a")
    assert a.session_id not in manager.sessions
    assert b.session_id in manager.sessions
    assert manager._cancel_generations[("alice", "thread-a", "task-a")] == 1
    assert ("alice", "thread-a") not in manager._cancel_generations
    manager._active_ops.clear()


def test_task_output_staged_then_imported_after_compute_and_workspace_cleanup(lifecycle_manager):
    manager, _, _ = lifecycle_manager
    root = manager.create_session()
    freezeable(manager, root)
    with scope("task-a"):
        a = manager.create_session()
        freezeable(manager, a)
        (Path(a.workspace) / "report.txt").write_text("task output", encoding="utf-8")
        manager._save_environment(a)
        paths = manager._storage_scope(a)
        (paths.uploads / "private-input").write_text("input", encoding="utf-8")
        staged = manager.stage_execution_scope(
            "alice",
            "thread-a",
            "task-a",
            [
                {"session_id": a.session_id, "path": "/workspace/report.txt"},
            ],
        )
        manager.cleanup_execution_scope("alice", "thread-a", "task-a")
        assert not paths.execution_root.exists()
        with pytest.raises(PermissionError, match="closed"):
            manager.create_session()
    assert len(staged) == 1
    assert Path(staged[0]["staged_path"]).read_text(encoding="utf-8") == "task output"
    result = manager.import_staged_files(
        "alice",
        "thread-a",
        "task-a",
        root.session_id,
        [
            {"staged_path": staged[0]["staged_path"], "destination": "results/report.txt"},
        ],
    )
    assert result.success
    assert (Path(root.workspace) / "results/report.txt").read_text(
        encoding="utf-8"
    ) == "task output"
    assert root.session_id in manager.sessions
    manager.remove_staged_execution_scope("alice", "thread-a", "task-a")
    assert not Path(staged[0]["staged_path"]).exists()


def test_internal_transfer_checks_scope_and_thread_before_touching_files(lifecycle_manager):
    manager, _, _ = lifecycle_manager
    root = manager.create_session()
    with scope("task-a"):
        a = manager.create_session()
    with pytest.raises(PermissionError, match="scope"):
        manager.copy_scope_files(
            root.session_id, a.session_id, [], source_scope=None, destination_scope="wrong"
        )
    token = current_thread_id.set("another-thread")
    try:
        with pytest.raises(PermissionError, match="conversation"):
            manager.stage_execution_scope("alice", "thread-a", "task-a", [])
    finally:
        current_thread_id.reset(token)
    token = current_user_id.set("bob")
    try:
        with pytest.raises(PermissionError, match="conversation"):
            manager.cleanup_execution_scope("alice", "thread-a", "task-a")
    finally:
        current_user_id.reset(token)
    assert a.session_id in manager.sessions


def test_input_handoff_copies_only_authorized_file_without_publication(lifecycle_manager):
    manager, _, _ = lifecycle_manager
    root = manager.create_session()
    freezeable(manager, root)
    (Path(root.workspace) / "data.csv").write_text("x\n1\n", encoding="utf-8")
    with scope("task-a"):
        child = manager.create_session()
    freezeable(manager, child)
    result = manager.copy_scope_files(
        root.session_id,
        child.session_id,
        [{"path": "/workspace/data.csv", "destination": "inputs/data.csv"}],
        source_scope=None,
        destination_scope="task-a",
    )
    assert result.success
    assert (Path(child.workspace) / "inputs/data.csv").read_text(encoding="utf-8") == "x\n1\n"
    (Path(child.workspace) / "inputs/data.csv").write_text("changed", encoding="utf-8")
    assert (Path(root.workspace) / "data.csv").read_text(encoding="utf-8") == "x\n1\n"
    assert list(manager._storage_scope(root).outputs.iterdir()) == []


def test_staged_import_rejects_host_path_outside_exact_task(lifecycle_manager):
    manager, _, _ = lifecycle_manager
    root = manager.create_session()
    freezeable(manager, root)
    secret = Path(root.workspace) / "secret.txt"
    secret.write_text("root only", encoding="utf-8")
    with pytest.raises(PermissionError, match="another task"):
        manager.import_staged_files(
            "alice",
            "thread-a",
            "task-a",
            root.session_id,
            [{"staged_path": str(secret), "destination": "stolen.txt"}],
        )
    assert not (Path(root.workspace) / "stolen.txt").exists()


def test_staging_failure_removes_partial_snapshots(lifecycle_manager, monkeypatch):
    import taskharbor.sandbox.storage as storage_module

    manager, _, _ = lifecycle_manager
    with scope("task-a"):
        child = manager.create_session()
        freezeable(manager, child)
        (Path(child.workspace) / "first.txt").write_text("first", encoding="utf-8")
        (Path(child.workspace) / "second.txt").write_text("second", encoding="utf-8")
        original = storage_module.safe_copy
        calls = 0

        def fail_second(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                target.write_text("partial", encoding="utf-8")
                raise OSError("copy failed")
            return original(source, target)

        monkeypatch.setattr(storage_module, "safe_copy", fail_second)
        with pytest.raises(OSError, match="copy failed"):
            manager.stage_execution_scope(
                "alice",
                "thread-a",
                "task-a",
                files=[
                    {"session_id": child.session_id, "path": "/workspace/first.txt"},
                    {"session_id": child.session_id, "path": "/workspace/second.txt"},
                ],
            )
    stage = manager._get_storage().staged_scope("alice", "thread-a", "task-a")
    assert list(stage.iterdir()) == []
    assert manager._containers[child.session_id].status == "running"
