"""Cancellation must stop the graph and still persist the executor's final result."""

import asyncio
import json
import os
import threading

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from tests import test_host_graph
from tests.test_agent_refactor import call


@pytest.fixture
def build_host_graph(monkeypatch):
    yield from test_host_graph.build_host_graph.__wrapped__(monkeypatch)


@pytest.mark.parametrize("cleanup_failed", [False, True])
def test_graph_cancel_saves_cleanup_result_without_continuing_model(
    build_host_graph, monkeypatch, tmp_path, cleanup_failed
):
    started = threading.Event()
    cleaned = threading.Event()
    attempts = []

    graph, store, model, executor = build_host_graph(
        calls=[call("execute_host_command", {"command": "wait", "cwd": str(tmp_path)})],
    )
    monkeypatch.setattr(
        "taskharbor.security.host.prepare_host_command",
        lambda command, cwd, timeout=None: {
            "command": command, "cwd": cwd, "timeout": timeout or 60, "shell": "powershell",
        },
    )

    def execute(command, cwd, timeout, *, scope, _cancelled):
        attempts.append(command)
        started.set()
        assert _cancelled.wait(5), "The tool must signal its worker when the graph is cancelled"
        cleaned.set()
        return {
            "success": False,
            "executor": "host",
            "stdout": "partial output",
            "exit_code": 130,
            "cancelled": True,
            "cleanup_failed": cleanup_failed,
        }

    monkeypatch.setattr(executor, "run", execute)

    async def scenario():
        config = {"configurable": {"thread_id": "host-graph", "run_id": "origin"}}
        paused = await graph.ainvoke({"messages": [HumanMessage(content="start")]}, config)
        assert paused.get("__interrupt__")
        record = store.list_thread("alice", "host-graph")[0]
        store.decide(record["id"], "alice", "host-graph", "allow")
        task = asyncio.create_task(graph.ainvoke(Command(resume=record["id"]), config))
        assert await asyncio.to_thread(started.wait, 5)
        # Match the HTTP cancellation order: invalidate the round before cancelling its task.
        store.cancel_thread("alice", "host-graph")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set()
        saved = store.get(record["id"], "alice", "host-graph")
        assert saved["status"] == ("unknown" if cleanup_failed else "cancelled")
        message = ToolMessage.model_validate(saved["result"])
        payload = json.loads(message.content)
        assert payload["cancelled"] and payload["cleanup_failed"] == cleanup_failed
        assert payload["stdout"] == "partial output"
        assert message.tool_call_id == record["tool_call_id"]
        assert attempts == ["wait"]
        assert len(model.seen) == 1, "The cancelled round must not call the model again"
        assert not store.blocked_thread("alice", "host-graph")

        # A new message can start normally without replaying the cancelled command.
        next_config = {"configurable": {"thread_id": "host-graph", "run_id": "next-run"}}
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="next question")]}, next_config
        )
        assert any(
            isinstance(message, HumanMessage) and message.content == "next question"
            for message in result["messages"]
        )
        assert result["messages"][-1].content == "done"
        assert attempts == ["wait"]

    asyncio.run(scenario())


@pytest.mark.skipif(os.name != "nt", reason="Real Windows command through the approval graph")
def test_real_host_graph_cancellation_keeps_confirmed_result(build_host_graph, tmp_path):
    marker = tmp_path / "started.txt"
    escaped = str(marker).replace("'", "''")
    command = f"[IO.File]::WriteAllText('{escaped}', 'started'); Start-Sleep -Seconds 60"
    graph, store, model, executor = build_host_graph(
        calls=[
            call("execute_host_command", {"command": command, "cwd": str(tmp_path), "timeout": 90})
        ],
    )

    async def scenario():
        config = {"configurable": {"thread_id": "host-graph", "run_id": "real-cancel"}}
        await graph.ainvoke({"messages": [HumanMessage(content="start")]}, config)
        record = store.list_thread("alice", "host-graph")[0]
        store.decide(record["id"], "alice", "host-graph", "allow")
        task = asyncio.create_task(graph.ainvoke(Command(resume=record["id"]), config))
        try:
            async with asyncio.timeout(10):
                while not marker.exists():
                    await asyncio.sleep(0.05)
            store.cancel_thread("alice", "host-graph")
            executor.cancel_scope("alice:host-graph")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        saved = store.get(record["id"], "alice", "host-graph")
        assert saved["status"] == "cancelled"
        payload = json.loads(saved["result"]["content"])
        assert payload["cancelled"] and not payload["cleanup_failed"]
        assert payload["exit_code"] == 130
        assert not executor._active
        assert len(model.seen) == 1

    asyncio.run(scenario())
