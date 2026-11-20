"""Durable sandbox files, immutable publications and identity/path boundaries."""

from __future__ import annotations

import io
import os
import shutil
import threading
import zipfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from taskharbor.sandbox.manager import SandboxManager, current_thread_id, current_user_id
from taskharbor.sandbox.storage import WorkspaceStore, checked_path


def test_scope_uses_opaque_identity_components_and_separate_runtime_workspaces(tmp_path):
    store = WorkspaceStore(tmp_path)
    python = store.scope("alice/../bob", "../../thread")
    node = store.scope("alice/../bob", "../../thread", "node")
    other_user = store.scope("bob", "../../thread")
    other_thread = store.scope("alice/../bob", "thread")
    assert python.root.is_relative_to(tmp_path)
    assert python.uploads == node.uploads
    assert python.workspace != node.workspace
    assert len({python.root, other_user.root, other_thread.root}) == 3


def test_publication_survives_workspace_deletion_and_new_store_instance(tmp_path):
    store = WorkspaceStore(tmp_path)
    scope = store.scope("alice", "thread")
    (scope.workspace / "report.csv").write_bytes(b"a,b\n1,2\n")
    published = store.publish(scope, "session", "report.csv")
    shutil.rmtree(scope.workspace)
    restored = WorkspaceStore(tmp_path).find("alice", "thread", "session", "report.csv")
    assert restored is not None
    assert restored.artifact_id == published.artifact_id
    assert b"".join(restored.stream()) == b"a,b\n1,2\n"
    assert store.find("bob", "thread", "session", "report.csv") is None
    assert store.find("alice", "other-thread", "session", "report.csv") is None
    assert store.find("alice", "thread", "other-session", "report.csv") is None


def test_republishing_retains_immutable_version_and_legacy_lookup_returns_latest(tmp_path):
    store = WorkspaceStore(tmp_path)
    scope = store.scope("alice", "thread")
    report = scope.workspace / "report.txt"
    report.write_text("first", encoding="utf-8")
    first = store.publish(scope, "session", "report.txt")
    report.write_text("second", encoding="utf-8")
    second = store.publish(scope, "session", "report.txt")
    assert first.artifact_id != second.artifact_id
    assert b"".join(first.stream()) == b"first"
    assert b"".join(second.stream()) == b"second"
    exact = store.find("alice", "thread", "session", "report.txt", artifact_id=first.artifact_id)
    assert b"".join(exact.stream()) == b"first"
    assert store.find("alice", "thread", "session", "report.txt").artifact_id == second.artifact_id


def test_directory_publication_is_zip_and_members_are_available_after_source_changes(tmp_path):
    store = WorkspaceStore(tmp_path)
    scope = store.scope("alice", "thread")
    reports = scope.workspace / "reports"
    reports.mkdir()
    (reports / "a.txt").write_text("original", encoding="utf-8")
    publication = store.publish(scope, "session", "reports")
    (reports / "a.txt").write_text("modified", encoding="utf-8")
    archive = zipfile.ZipFile(io.BytesIO(b"".join(publication.stream())))
    assert archive.read("reports/a.txt") == b"original"
    member = store.find("alice", "thread", "session", "/workspace/reports/a.txt")
    assert member is not None and not member.is_directory
    assert b"".join(member.stream()) == b"original"
    assert store.find("alice", "thread", "session", "/workspace/reports-private/a.txt") is None


def test_import_keeps_original_separate_from_editable_workspace(tmp_path):
    store = WorkspaceStore(tmp_path / "storage")
    scope = store.scope("alice", "thread")
    source = tmp_path / "input.txt"
    source.write_text("original", encoding="utf-8")
    assert store.import_path(scope, source, "nested/input.txt") == 8
    (scope.workspace / "nested/input.txt").write_text("changed", encoding="utf-8")
    snapshots = list(scope.uploads.glob("*/input.txt"))
    assert len(snapshots) == 1
    assert snapshots[0].read_text(encoding="utf-8") == "original"
    assert source.read_text(encoding="utf-8") == "original"


def _symlink(target: Path, link: Path, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError:
        pytest.skip("Symlink creation requires Windows developer mode or privilege")


def test_publication_rejects_nested_symlinks_instead_of_following_host_files(tmp_path):
    store = WorkspaceStore(tmp_path / "storage")
    scope = store.scope("alice", "thread")
    secret = tmp_path / "secret.txt"
    secret.write_text("credential", encoding="utf-8")
    directory = scope.workspace / "report"
    directory.mkdir()
    _symlink(secret, directory / "leak.txt")
    with pytest.raises(PermissionError, match="Symlinks"):
        store.publish(scope, "session", "report")
    assert not list(scope.outputs.glob("*/manifest.json"))


def test_import_rejects_destination_symlink_and_source_directory_symlink(tmp_path):
    store = WorkspaceStore(tmp_path / "storage")
    scope = store.scope("alice", "thread")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "untouched.txt").write_text("keep", encoding="utf-8")
    _symlink(outside, scope.workspace / "redirect", directory=True)
    with pytest.raises(PermissionError):
        store.copy_to_workspace(scope, outside / "untouched.txt", "redirect/new.txt")
    assert not (outside / "new.txt").exists()
    source = tmp_path / "input"
    source.mkdir()
    _symlink(outside, source / "nested", directory=True)
    with pytest.raises(PermissionError):
        store.import_path(scope, source, "input")
    assert not (scope.workspace / "input").exists()


def test_publication_rejects_hardlinks_to_files_outside_workspace(tmp_path):
    store = WorkspaceStore(tmp_path / "storage")
    scope = store.scope("alice", "thread")
    secret = tmp_path / "secret.txt"
    secret.write_text("credential", encoding="utf-8")
    os.link(secret, scope.workspace / "link.txt")
    with pytest.raises(PermissionError, match="Hard-linked"):
        store.publish(scope, "session", "link.txt")
    assert not list(scope.outputs.glob("*/manifest.json"))


@pytest.mark.parametrize("path", ["/etc/passwd", "../secret", "/workspace/../../secret",
                                       "C:\\secret", "/workspace/file:secret"])
def test_workspace_paths_cannot_escape_or_use_host_path_syntax(tmp_path, path):
    store = WorkspaceStore(tmp_path)
    with pytest.raises(ValueError):
        store.workspace_path(store.scope("alice", "thread"), path)


def test_checked_path_rejects_storage_root_replaced_with_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    _symlink(real, tmp_path / "redirect", directory=True)
    with pytest.raises(PermissionError):
        checked_path(tmp_path / "redirect", "file")


def test_windows_reparse_attribute_is_rejected_without_is_junction_api(tmp_path, monkeypatch):
    target = tmp_path / "junction"
    target.mkdir()
    original = Path.lstat

    def lstat(path):
        details = original(path)
        if path == target:
            return SimpleNamespace(st_mode=details.st_mode, st_file_attributes=0x400)
        return details

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(PermissionError, match="junctions"):
        checked_path(tmp_path, "junction/file")


@pytest.fixture
def storage_manager(tmp_path, monkeypatch):
    import taskharbor.sandbox.manager as module

    store = WorkspaceStore(tmp_path / "storage")
    manager = object.__new__(SandboxManager)
    manager._manager_id = "test-manager"
    manager._storage = store
    manager.sessions = {}
    manager._containers = {}
    manager._condition = threading.Condition(threading.RLock())
    manager._quiesced = set()
    manager._operation = lambda sid: nullcontext()
    manager._touch_session = lambda sid: None
    manager._assert_container_alive = lambda sid: None
    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(
        STORAGE_DIR=store.root, IMPORT_ALLOWED_DIRS=[tmp_path],
    ))
    owner = current_user_id.set("alice")
    thread = current_thread_id.set("thread")
    yield manager, store
    current_user_id.reset(owner)
    current_thread_id.reset(thread)


class TransferContainer:
    def __init__(self):
        self.status = "running"
        self.attrs = {"State": {"ExitCode": 0}}
        self.events = []

    def pause(self):
        self.events.append("pause")
        self.status = "paused"

    def unpause(self):
        self.events.append("unpause")
        self.status = "running"

    def reload(self):
        pass


def _session(manager, sid, *, user="alice", thread="thread", runtime="python"):
    manager.sessions[sid] = SimpleNamespace(
        session_id=sid, user_id=user, thread_id=thread, runtime=runtime,
    )
    manager._containers[sid] = TransferContainer()
    return manager._storage.scope(user, thread, runtime)


def test_manager_download_and_cross_session_import_do_not_need_source_container(storage_manager):
    manager, store = storage_manager
    original = _session(manager, "source")
    (original.workspace / "report.txt").write_text("result", encoding="utf-8")
    exported = manager.export_files("source", [{"source": "report.txt"}])
    assert exported.success
    artifact_id = exported.files[0].artifact_id
    manager.sessions.clear()
    manager._containers.clear()
    assert manager.is_file_exported("thread", "source", "report.txt", artifact_id=artifact_id)
    assert b"".join(manager.stream_exported_file(
        "thread", "source", "report.txt", artifact_id=artifact_id
    )) == b"result"
    node = _session(manager, "destination", runtime="node")
    imported = manager.import_files("destination", [{
        "session_id": "source", "path": "report.txt", "artifact_id": artifact_id,
    }])
    assert imported.success
    assert (node.workspace / "report.txt").read_text(encoding="utf-8") == "result"
    token = current_user_id.set("bob")
    try:
        assert not manager.is_file_exported("thread", "source", "report.txt")
    finally:
        current_user_id.reset(token)


def test_configured_host_root_cannot_bypass_another_users_managed_storage(storage_manager):
    manager, store = storage_manager
    _session(manager, "destination")
    other = store.scope("bob", "thread")
    private = other.uploads / "private.txt"
    private.write_text("credential", encoding="utf-8")
    result = manager.import_files("destination", [{"source": str(private)}])
    assert not result.success
    assert "another user" in result.files[0].error


def test_import_parent_segments_do_not_bypass_identity_boundary(storage_manager):
    manager, store = storage_manager
    own = _session(manager, "destination")
    other = store.scope("bob", "thread")
    private = other.uploads / "private.txt"
    private.write_text("credential", encoding="utf-8")
    traversal = own.uploads / os.path.relpath(private, own.uploads)
    result = manager.import_files("destination", [{"source": str(traversal)}])
    assert not result.success
    assert "another user" in result.files[0].error


def test_export_freezes_background_writers_until_host_copy_completes(storage_manager, monkeypatch):
    manager, store = storage_manager
    scope = _session(manager, "session")
    (scope.workspace / "report.txt").write_text("data")
    publish = store.publish

    def checked_publish(*args):
        assert manager._containers["session"].status == "paused"
        assert "session" in manager._quiesced
        assert manager._inspect_container_health("session") is None
        return publish(*args)

    monkeypatch.setattr(store, "publish", checked_publish)
    assert manager.export_files("session", [{"source": "report.txt"}]).success
    assert manager._containers["session"].events == ["pause", "unpause"]
    assert manager._containers["session"].status == "running"
    assert not manager._quiesced


def test_pause_failure_denies_transfer_before_any_host_copy(storage_manager, monkeypatch):
    manager, store = storage_manager
    scope = _session(manager, "session")
    (scope.workspace / "report.txt").write_text("data")

    def pause_failed():
        raise RuntimeError("freezer unavailable")

    monkeypatch.setattr(manager._containers["session"], "pause", pause_failed)
    with pytest.raises(RuntimeError, match="Unable to freeze"):
        manager.export_files("session", [{"source": "report.txt"}])
    assert not list(scope.outputs.iterdir())
    assert not manager._quiesced


def test_failed_transfer_resumes_container_in_finally(storage_manager):
    manager, _ = storage_manager
    _session(manager, "session")
    result = manager.export_files("session", [{"source": "missing.txt"}])
    assert not result.success
    assert manager._containers["session"].events == ["pause", "unpause"]
    assert not manager._quiesced


def test_unpause_failure_destroys_uncertain_environment(storage_manager, monkeypatch):
    manager, _ = storage_manager
    scope = _session(manager, "session")
    (scope.workspace / "report.txt").write_text("data")
    destroyed = []

    def unpause_failed():
        raise RuntimeError("resume failure")

    monkeypatch.setattr(manager._containers["session"], "unpause", unpause_failed)
    monkeypatch.setattr(manager, "stop_session", lambda sid: destroyed.append(sid))
    with pytest.raises(RuntimeError, match="resume failed"):
        manager.export_files("session", [{"source": "report.txt"}])
    assert destroyed == ["session"]
    assert not manager._quiesced


def test_other_runtime_mutable_workspace_requires_published_artifact(storage_manager):
    manager, store = storage_manager
    python = _session(manager, "python")
    node = store.scope("alice", "thread", "node")
    data = node.workspace / "data.txt"
    data.write_text("data")
    result = manager.import_files("python", [{"source": str(data)}])
    assert not result.success
    assert "cross-runtime" in result.files[0].error
    assert not (python.workspace / "data.txt").exists()


def test_current_runtime_raw_import_is_allowed_only_while_frozen(storage_manager, monkeypatch):
    manager, store = storage_manager
    scope = _session(manager, "python")
    data = scope.workspace / "data.txt"
    data.write_text("data")
    import_path = store.import_path

    def checked_import(*args):
        assert manager._containers["python"].status == "paused"
        return import_path(*args)

    monkeypatch.setattr(store, "import_path", checked_import)
    result = manager.import_files("python", [{"source": str(data), "destination": "copy.txt"}])
    assert result.success
    assert (scope.workspace / "copy.txt").read_text() == "data"


def test_explicit_thread_delete_leaves_other_identity_data_intact(tmp_path):
    store = WorkspaceStore(tmp_path)
    alice = store.scope("alice", "thread")
    bob = store.scope("bob", "thread")
    (alice.workspace / "file").write_text("alice")
    (bob.workspace / "file").write_text("bob")
    store.delete_thread("alice", "thread")
    assert not alice.root.exists()
    assert (bob.workspace / "file").read_text() == "bob"
