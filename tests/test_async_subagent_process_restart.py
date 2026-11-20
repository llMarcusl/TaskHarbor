"""Opt-in process restart over real HTTP, Aegra and PostgreSQL checkpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from taskharbor.agent.subagents.protocol import flatten_interrupts
from tests.test_async_subagent_service_integration import (
    postgres_database as _postgres_database,
)
from tests.test_async_subagent_service_integration import real_service, run_async

pytestmark = pytest.mark.skipif(
    os.environ.get("SANDBOX_SERVICE_TESTS") != "1",
    reason="opt-in local PostgreSQL and HTTP process restart acceptance",
)


@pytest.fixture(scope="module")
def postgres_database():
    yield from _postgres_database.__wrapped__()


async def _serve(dsn, storage, ready_file, port):
    with pytest.MonkeyPatch.context() as monkeypatch:
        async with real_service(
            dsn, storage, monkeypatch, scenario="child-approval", listen_port=port,
        ) as env:
            temporary = Path(ready_file).with_suffix(".tmp")
            temporary.write_text(
                json.dumps({
                    "url": str(env.client.base_url), "pid": os.getpid(), "parent_pid": os.getppid(),
                }),
                encoding="utf-8",
            )
            temporary.replace(ready_file)
            await asyncio.Event().wait()


@contextmanager
def server_process(dsn, storage, attempt, port):
    ready_file = storage / f"ready-{attempt}.json"
    log_file = storage / f"server-{attempt}.log"
    storage.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "-m",
        "tests.test_async_subagent_process_restart",
        "--serve",
        "--dsn",
        dsn,
        "--storage",
        str(storage),
        "--ready-file",
        str(ready_file),
        "--port",
        str(port),
    ]
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        server_pid = None
        try:
            deadline = time.monotonic() + 40
            while not ready_file.exists():
                if process.poll() is not None:
                    raise AssertionError(log_file.read_text(encoding="utf-8")[-5000:])
                if time.monotonic() >= deadline:
                    raise TimeoutError("Restart acceptance HTTP service did not become ready")
                time.sleep(0.05)
            ready = json.loads(ready_file.read_text(encoding="utf-8"))
            server_pid = ready["pid"]
            # Windows venv executables redirect to the base interpreter.
            assert process.pid in {server_pid, ready["parent_pid"]}
            with httpx.Client(base_url=ready["url"], timeout=20) as client:
                yield client, process, server_pid
        finally:
            if process.poll() is None:
                kill_server(process, server_pid)
            process.wait(timeout=15)


def kill_server(process, server_pid):
    if server_pid is not None:
        os.kill(server_pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    elif os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        process.kill()
    process.wait(timeout=15)


def get_json(client, path):
    response = client.get(path)
    response.raise_for_status()
    return response.json()


def wait_until(operation, predicate, *, timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        value = operation()
        if predicate(value):
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"HTTP state did not reach its expected outcome: {value}")
        time.sleep(0.05)


def test_child_approval_and_parent_business_wait_survive_process_kill(postgres_database, tmp_path):
    storage = tmp_path / "service-storage"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with server_process(postgres_database, storage, "before", port) as (
        client, process, server_pid,
    ):
        response = client.post("/threads", json={})
        response.raise_for_status()
        parent = response.json()["thread_id"]
        response = client.post(
            f"/threads/{parent}/runs",
            json={
                "assistant_id": "taskharbor",
                "input": {"messages": [{"role": "user", "content": "execute delegated work"}]},
                "config": {
                    "configurable": {
                        "chat_model": "test-scripted",
                        "chat_model_provider": "openai",
                        "chat_model_api_key": "test-key",
                    }
                },
                "stream_mode": ["values", "updates"],
            },
        )
        response.raise_for_status()
        approvals = wait_until(
            lambda: get_json(client, f"/threads/{parent}/approvals"),
            lambda result: any(row["task_id"] for row in result["approvals"]),
        )
        pending = next(row for row in approvals["approvals"] if row["task_id"])
        child = pending["execution_thread_id"]
        state = wait_until(
            lambda: get_json(client, f"/threads/{parent}/state"),
            lambda result: any(
                item["value"].get("kind") == "awaiting_subtasks"
                for item in flatten_interrupts(result)
            ),
        )
        original_tasks = set(state["values"]["async_tasks"])
        assert len(original_tasks) == 2
        assert pending["status"] == "pending" and pending["interrupt_id"]
        sibling, = original_tasks - {child}
        wait_until(
            lambda: get_json(client, f"/threads/{sibling}"),
            lambda result: result["status"] == "idle",
        )
        first_pid = server_pid
        kill_server(process, server_pid)

    with server_process(postgres_database, storage, "after", port) as (client, process, server_pid):
        assert server_pid != first_pid
        approvals = get_json(client, f"/threads/{parent}/approvals")
        restored = next(row for row in approvals["approvals"] if row["id"] == pending["id"])
        assert restored["status"] == "pending"
        assert restored["execution_thread_id"] == child
        assert restored["interrupt_id"] == pending["interrupt_id"]
        state = get_json(client, f"/threads/{parent}/state")
        assert set(state["values"]["async_tasks"]) == original_tasks
        assert state["values"]["awaiting_subtasks"]
        response = client.post(
            f"/threads/{parent}/approvals/{restored['id']}/decision",
            json={"decision": "allow"},
        )
        response.raise_for_status()
        response = client.post(
            f"/threads/{child}/runs",
            json={
                "assistant_id": restored["assistant_id"],
                "command": {
                    "resume": {
                        restored["interrupt_id"]: {"approval_id": restored["id"]},
                    }
                },
                "stream_mode": ["values", "updates"],
            },
        )
        response.raise_for_status()
        final = wait_until(
            lambda: get_json(client, f"/threads/{parent}"),
            lambda result: result["status"] in {"idle", "error"},
        )
        assert final["status"] == "idle"
        state = get_json(client, f"/threads/{parent}/state")
        assert set(state["values"]["async_tasks"]) == original_tasks
        assert not state["values"]["awaiting_subtasks"]
        assert all(task["status"] == "success" for task in state["values"]["async_tasks"].values())
        assert state["values"]["messages"][-1]["content"] == "root collected all outcomes"
        child_state = get_json(client, f"/threads/{child}/state")
        actions = [
            message
            for message in child_state["values"]["messages"]
            if message.get("type") == "tool" and message.get("name") == "custom_action"
        ]
        assert len(actions) == 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()
    run_async(_serve(arguments.dsn, arguments.storage, arguments.ready_file, arguments.port))
