"""Storage ownership is enforced by the operating system, across processes."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from taskharbor.sandbox.manager import SandboxManager
from taskharbor.sandbox.ownership import DeploymentLock
from tests.test_sandbox_lifecycle import lifecycle_manager as lifecycle_manager

__all__ = ["lifecycle_manager"]

CHILD_ACQUIRE = """
import sys
from taskharbor.sandbox.ownership import DeploymentLock
try:
    with DeploymentLock(sys.argv[1]):
        print('acquired')
except RuntimeError:
    raise SystemExit(17)
"""


def child_acquire(root):
    return subprocess.run(
        [sys.executable, "-c", CHILD_ACQUIRE, str(root)],
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_excludes_other_process_then_reacquires_without_unlink(tmp_path):
    owner = DeploymentLock(tmp_path).acquire()
    path = tmp_path / "control" / "manager.lock"
    inode = path.stat().st_ino
    try:
        blocked = child_acquire(tmp_path)
        assert blocked.returncode == 17, blocked.stderr
        with pytest.raises(RuntimeError, match="STORAGE_DIR"):
            DeploymentLock(tmp_path).acquire()
    finally:
        owner.release()
    owner.release()  # Shutdown can safely be called more than once.
    assert path.exists() and path.stat().st_ino == inode
    acquired = child_acquire(tmp_path)
    assert acquired.returncode == 0, acquired.stderr
    assert acquired.stdout.strip() == "acquired"
    assert path.stat().st_ino == inode


def test_process_death_releases_ownership(tmp_path):
    code = """
import os, sys
from taskharbor.sandbox.ownership import DeploymentLock
owner = DeploymentLock(sys.argv[1]).acquire()
os._exit(0)
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], timeout=15)
    assert result.returncode == 0
    with DeploymentLock(tmp_path):
        assert (tmp_path / "control" / "manager.lock").exists()


def test_lock_rejects_redirected_control_directory(tmp_path):
    target = tmp_path / "outside"
    target.mkdir()
    try:
        os.symlink(target, tmp_path / "control", target_is_directory=True)
    except OSError:
        pytest.skip("Host does not permit directory symlink creation")
    with pytest.raises(PermissionError):
        DeploymentLock(tmp_path).acquire()


def test_manager_enforces_single_owner_then_releases_on_shutdown(lifecycle_manager):
    manager, settings, _ = lifecycle_manager
    with pytest.raises(RuntimeError, match="STORAGE_DIR"):
        SandboxManager()
    manager.cleanup_all()
    replacement = SandboxManager()
    try:
        assert replacement._heartbeat_path.exists()
        assert (settings.STORAGE_DIR / "control" / "manager.lock").exists()
    finally:
        replacement.cleanup_all()


def test_failed_startup_releases_lock(lifecycle_manager, monkeypatch):
    manager, settings, _ = lifecycle_manager
    manager.cleanup_all()

    def fail(_):
        raise RuntimeError("startup failed")

    monkeypatch.setattr(SandboxManager, "_start_gc", fail)
    with pytest.raises(RuntimeError, match="startup failed"):
        SandboxManager()
    with DeploymentLock(settings.STORAGE_DIR):
        pass


def test_shutdown_keeps_lock_until_pending_create_finishes(lifecycle_manager):
    manager, settings, _ = lifecycle_manager
    manager._create_lock.acquire()
    manager.cleanup_all()
    try:
        with pytest.raises(RuntimeError, match="STORAGE_DIR"):
            DeploymentLock(settings.STORAGE_DIR).acquire()
    finally:
        manager._create_lock.release()
    with manager._condition:
        manager._release_ownership_if_idle_locked()
    with DeploymentLock(settings.STORAGE_DIR):
        pass


def test_shutdown_keeps_lock_until_removed_session_finishes_io(lifecycle_manager):
    manager, settings, _ = lifecycle_manager
    info = manager.create_session()
    with manager._operation(info.session_id):
        manager.cleanup_all()
        assert info.session_id not in manager.sessions
        with pytest.raises(RuntimeError, match="STORAGE_DIR"):
            DeploymentLock(settings.STORAGE_DIR).acquire()
    with DeploymentLock(settings.STORAGE_DIR):
        pass
