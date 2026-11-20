"""Signed parent/child transfers reuse persistent workspace validation."""

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from taskharbor.agent.subagents import files as file_tools
from taskharbor.agent.subagents.files import TaskFile
from taskharbor.sandbox.manager import current_thread_id, current_user_id
from tests.test_sandbox_lifecycle import lifecycle_manager as _lifecycle_manager
from tests.test_sandbox_task_scopes import freezeable


@pytest.fixture
def transfers(tmp_path, monkeypatch):
    yield from _lifecycle_manager.__wrapped__(tmp_path, monkeypatch)


@contextmanager
def thread(thread_id):
    token = current_thread_id.set(thread_id)
    try:
        yield SimpleNamespace(config={"configurable": {"thread_id": thread_id}})
    finally:
        current_thread_id.reset(token)


def setup_tools(manager, settings, monkeypatch):
    settings.ENCRYPTION_KEY = "test-signing-key"
    monkeypatch.setattr(
        manager,
        "resolve_owner",
        lambda thread: "bob" if thread == "foreign" else "alice",
    )

    async def parent(owner, child):
        return "thread-a" if (owner, child) == ("alice", "child") else None

    monkeypatch.setattr(file_tools, "worker_parent", parent)
    prepare, collect = file_tools.create_parent_file_tools(manager, settings)
    (importing,) = file_tools.create_worker_file_tools(manager, settings)
    return prepare, collect, importing


def test_signed_input_and_selected_output_survive_child_compute_retirement(transfers, monkeypatch):
    manager, settings, _ = transfers
    prepare, collect, importing = setup_tools(manager, settings, monkeypatch)
    root = manager.create_session()
    freezeable(manager, root)
    (Path(root.workspace) / "input.txt").write_text("source", encoding="utf-8")
    with thread("thread-a") as runtime:
        grant = json.loads(
            prepare.func(
                [TaskFile(path="/workspace/input.txt")],
                runtime,
            )
        )
    with thread("child") as runtime:
        child = manager.create_session()
        freezeable(manager, child)
        result = json.loads(asyncio.run(importing.coroutine(grant["manifest"], runtime)))
        assert result["success"]
        assert Path(child.workspace, "input.txt").read_text() == "source"
        Path(child.workspace, "output.txt").write_text("verified result", encoding="utf-8")
        manager.stop_session(child.session_id)
    with thread("thread-a") as runtime:
        result = json.loads(
            asyncio.run(
                collect.coroutine(
                    "child",
                    [TaskFile(path="/workspace/output.txt")],
                    runtime,
                )
            )
        )
        assert result["success"]
        assert Path(root.workspace, "output.txt").read_text() == "verified result"
    assert root.workspace != child.workspace


def test_transfer_rejects_forged_manifest_sibling_and_cross_owner(transfers, monkeypatch):
    manager, settings, _ = transfers
    prepare, collect, importing = setup_tools(manager, settings, monkeypatch)
    root = manager.create_session()
    Path(root.workspace, "input.txt").write_text("private", encoding="utf-8")
    with thread("thread-a") as runtime:
        grant = json.loads(prepare.func([TaskFile(path="input.txt")], runtime))
        with pytest.raises(PermissionError, match="another parent"):
            asyncio.run(collect.coroutine("sibling", [TaskFile(path="input.txt")], runtime))
        with pytest.raises(PermissionError, match="ownership"):
            manager.copy_thread_files("alice", "foreign", "thread-a", "python", [])
    with thread("child") as runtime:
        with pytest.raises(PermissionError, match="capability"):
            asyncio.run(importing.coroutine(grant["manifest"] + "corrupt", runtime))
    with thread("sibling") as runtime:
        with pytest.raises(PermissionError, match="another task owner"):
            asyncio.run(importing.coroutine(grant["manifest"], runtime))


def test_file_manifest_rejects_paths_outside_workspace(transfers, monkeypatch):
    manager, settings, _ = transfers
    prepare, _, _ = setup_tools(manager, settings, monkeypatch)
    with thread("thread-a") as runtime:
        with pytest.raises(ValueError, match="outside"):
            prepare.func([TaskFile(path="/etc/passwd")], runtime)


def in_thread(thread_id, operation):
    owner_token = current_user_id.set("alice")
    try:
        with thread(thread_id):
            return operation()
    finally:
        current_user_id.reset(owner_token)


def test_copy_excludes_container_mounting_until_offline_transfer_finishes(transfers, monkeypatch):
    manager, settings, client = transfers
    settings.SANDBOX_QUEUE_TIMEOUT_SECONDS = 3
    monkeypatch.setattr(manager, "resolve_owner", lambda thread: "alice")
    store = manager._get_storage()
    source = store.scope("alice", "thread-a", "python")
    (source.workspace / "input.txt").write_text("stable input", encoding="utf-8")
    copying = threading.Event()
    release = threading.Event()
    ready_to_create = threading.Event()
    original_copy = store.copy_to_workspace

    def copy(scope, path, destination):
        copying.set()
        assert release.wait(timeout=3)
        return original_copy(scope, path, destination)

    monkeypatch.setattr(store, "copy_to_workspace", copy)
    monkeypatch.setattr(manager, "_ensure_image", lambda runtime: ready_to_create.set())
    with ThreadPoolExecutor(max_workers=2) as workers:
        transfer = workers.submit(
            in_thread,
            "thread-a",
            lambda: manager.copy_thread_files(
                "alice",
                "thread-a",
                "child",
                "python",
                [{"path": "/workspace/input.txt"}],
            ),
        )
        assert copying.wait(timeout=3)
        creating = workers.submit(in_thread, "child", manager.create_session)
        try:
            assert ready_to_create.wait(timeout=3)
            assert not creating.done()
            assert client.containers.run.call_count == 0
        finally:
            release.set()
        assert transfer.result(timeout=3).success
        created = creating.result(timeout=3)
    assert Path(created.workspace, "input.txt").read_text() == "stable input"


def test_copy_waits_for_inflight_container_and_freezes_it_before_reading(transfers, monkeypatch):
    manager, settings, client = transfers
    settings.SANDBOX_QUEUE_TIMEOUT_SECONDS = 3
    monkeypatch.setattr(manager, "resolve_owner", lambda thread: "alice")
    store = manager._get_storage()
    source = store.scope("alice", "thread-a", "python")
    (source.workspace / "input.txt").write_text("stable input", encoding="utf-8")
    mounting = threading.Event()
    copying = threading.Event()
    release = threading.Event()
    original_run = client.containers.run.side_effect
    original_copy = store.copy_to_workspace
    containers = []

    def run(*args, **kwargs):
        container = original_run(*args, **kwargs)
        container.pause.side_effect = lambda: setattr(container, "status", "paused")
        container.unpause.side_effect = lambda: setattr(container, "status", "running")
        containers.append(container)
        mounting.set()
        assert release.wait(timeout=3)
        return container

    def copy(scope, path, destination):
        assert containers[0].status == "paused"
        copying.set()
        return original_copy(scope, path, destination)

    monkeypatch.setattr(client.containers.run, "side_effect", run)
    monkeypatch.setattr(store, "copy_to_workspace", copy)
    with ThreadPoolExecutor(max_workers=2) as workers:
        creating = workers.submit(in_thread, "child", manager.create_session)
        assert mounting.wait(timeout=3)
        transfer = workers.submit(
            in_thread,
            "thread-a",
            lambda: manager.copy_thread_files(
                "alice",
                "thread-a",
                "child",
                "python",
                [{"path": "/workspace/input.txt"}],
            ),
        )
        try:
            assert not copying.wait(timeout=0.05)
            assert not transfer.done()
        finally:
            release.set()
        created = creating.result(timeout=3)
        assert transfer.result(timeout=3).success
    assert copying.is_set()
    assert containers[0].pause.call_count == 1
    assert Path(created.workspace, "input.txt").read_text() == "stable input"


def test_full_capacity_creator_does_not_block_file_handoff(transfers, monkeypatch):
    manager, settings, client = transfers
    settings.CONTAINER_MAX_SESSIONS = 1
    settings.SANDBOX_QUEUE_TIMEOUT_SECONDS = 3
    monkeypatch.setattr(manager, "resolve_owner", lambda thread: "alice")
    manager.begin_turn("alice", "thread-a", "active-root")
    root = manager.create_session()
    freezeable(manager, root)
    Path(root.workspace, "input.txt").write_text("stable input", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=1) as workers:
        creating = workers.submit(in_thread, "unrelated", manager.create_session)
        try:
            deadline = time.monotonic() + 2
            while not manager._create_lock.locked():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            result = manager.copy_thread_files(
                "alice",
                "thread-a",
                "child",
                "python",
                [{"path": "/workspace/input.txt"}],
            )
            assert result.success
            assert not creating.done()
            assert client.containers.run.call_count == 1
        finally:
            manager.cancel_turn("alice", "unrelated", "cancel-waiter")
        with pytest.raises(RuntimeError, match="cancelled"):
            creating.result(timeout=3)
    child = manager._get_storage().scope("alice", "child", "python", create=False)
    assert (child.workspace / "input.txt").read_text() == "stable input"
