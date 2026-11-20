"""A cancelled acquisition must not publish a container after Stop returns."""

from __future__ import annotations

import contextvars
import threading
from unittest.mock import MagicMock

import pytest

from taskharbor.sandbox.manager import current_user_id
from tests.test_sandbox_lifecycle import lifecycle_manager as lifecycle_manager

__all__ = ["lifecycle_manager"]


def test_cancel_during_image_build_never_starts_container(lifecycle_manager, monkeypatch):
    manager, _, _ = lifecycle_manager
    monkeypatch.setattr(
        manager, "_ensure_image", lambda _: manager.cancel_turn("alice", "thread-a", "run")
    )
    with pytest.raises(RuntimeError, match="acquisition was cancelled"):
        manager.create_session()
    manager.client.containers.run.assert_not_called()
    assert not manager.sessions


def test_cancel_during_docker_create_removes_unregistered_container(lifecycle_manager):
    manager, _, client = lifecycle_manager
    created = MagicMock(id="unregistered-container")

    def create(*args, **kwargs):
        manager.cancel_turn("alice", "thread-a", "run")
        return created

    client.containers.run.side_effect = create
    with pytest.raises(RuntimeError, match="acquisition was cancelled"):
        manager.create_session()
    created.remove.assert_called_once_with(force=True)
    assert not manager.sessions


@pytest.mark.parametrize("phase", ["kernel", "dependencies"])
def test_cancel_during_initialization_cannot_save_or_revive_session(
    lifecycle_manager, monkeypatch, phase
):
    manager, _, client = lifecycle_manager
    save = MagicMock()
    monkeypatch.setattr(manager, "_save_environment", save)

    def cancelled(*args):
        manager.cancel_turn("alice", "thread-a", "run")

    monkeypatch.setattr(
        manager,
        "_wait_for_kernel" if phase == "kernel" else "_install_initial_packages",
        cancelled,
    )
    with pytest.raises(RuntimeError, match="acquisition was cancelled"):
        manager.create_session(dependencies={"example": ""} if phase == "dependencies" else {})
    save.assert_not_called()
    assert not manager.sessions
    assert not manager._active_ops


def test_cancel_queued_acquisition_then_new_turn_can_create(lifecycle_manager, monkeypatch):
    manager, _, client = lifecycle_manager
    queued = threading.Event()
    released = threading.Event()
    result = []
    context = contextvars.copy_context()
    original_lock = manager._create_lock

    class QueuedLock:
        def acquire(self, **kwargs):
            queued.set()
            assert released.wait(2)
            return True

        def release(self):
            pass

    monkeypatch.setattr(manager, "_create_lock", QueuedLock())

    def acquire():
        try:
            manager.create_session()
        except BaseException as exc:
            result.append(exc)

    worker = threading.Thread(target=lambda: context.run(acquire))
    worker.start()
    try:
        assert queued.wait(1)
        manager.cancel_turn("alice", "thread-a", "old-run")
    finally:
        released.set()
        worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(result) == 1 and "acquisition was cancelled" in str(result[0])
    client.containers.run.assert_not_called()

    monkeypatch.setattr(manager, "_create_lock", original_lock)
    manager.begin_turn("alice", "thread-a", "new-run")
    assert manager.create_session().user_id == "alice"
    token = current_user_id.set("bob")
    try:
        assert manager.create_session().user_id == "bob"
    finally:
        current_user_id.reset(token)
