"""Real Windows commands overlap after their checkpoint approvals are resolved."""

import asyncio
import json
import os

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from tests import test_host_graph
from tests.test_agent_refactor import call


@pytest.fixture
def build_host_graph(monkeypatch):
    yield from test_host_graph.build_host_graph.__wrapped__(monkeypatch)


@pytest.mark.skipif(os.name != "nt", reason="Real Windows PowerShell commands")
def test_second_command_starts_while_first_is_running(build_host_graph, tmp_path):
    started_a = tmp_path / "started-a.txt"
    started_b = tmp_path / "started-b.txt"
    release_a = tmp_path / "release-a.txt"

    def quote(path):
        return "'" + str(path).replace("'", "''") + "'"

    command_a = (
        f"[IO.File]::WriteAllText({quote(started_a)}, 'started'); "
        f"while (-not (Test-Path -LiteralPath {quote(release_a)})) "
        "{ Start-Sleep -Milliseconds 50 }; Write-Output 'A done'"
    )
    command_b = f"[IO.File]::WriteAllText({quote(started_b)}, 'started'); Write-Output 'B done'"
    graph, store, model, executor = build_host_graph(
        calls=[
            call(
                "execute_host_command",
                {
                    "command": command_a,
                    "cwd": str(tmp_path),
                    "timeout": 20,
                },
                id="call-a",
            ),
            call(
                "execute_host_command",
                {
                    "command": command_b,
                    "cwd": str(tmp_path),
                    "timeout": 20,
                },
                id="call-b",
            ),
        ]
    )

    async def wait_for_file(path):
        async with asyncio.timeout(8):
            while not path.exists():
                await asyncio.sleep(0.05)

    async def scenario():
        config = {"configurable": {"thread_id": "host-graph", "run_id": "overlap-run"}}
        paused = await graph.ainvoke({"messages": [HumanMessage(content="run both")]}, config)
        assert paused.get("__interrupt__")
        records = {r["tool_call_id"]: r for r in store.list_thread("alice", "host-graph")}
        assert set(records) == {"call-a", "call-b"}
        assert records["call-a"]["batch_index"] == 0
        assert records["call-b"]["batch_index"] == 1
        store.decide(records["call-a"]["id"], "alice", "host-graph", "allow")
        store.decide(records["call-b"]["id"], "alice", "host-graph", "allow")
        task = asyncio.create_task(
            graph.ainvoke(
                Command(
                    resume={
                        item.id: {"approval_id": item.value["approval_id"]}
                        for item in paused["__interrupt__"]
                    }
                ),
                config,
            )
        )
        try:
            await wait_for_file(started_a)
            await wait_for_file(started_b)
            assert not release_a.exists() and not task.done()
            assert len(model.seen) == 1
            release_a.write_text("finish", encoding="utf-8")
            result = await asyncio.wait_for(task, 8)
        finally:
            if not task.done():
                store.cancel_thread("alice", "host-graph")
                executor.cancel_scope("alice:host-graph")
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert [m.tool_call_id for m in messages] == ["call-a", "call-b"]
        assert all(json.loads(m.content)["success"] for m in messages)
        assert result["messages"][-1].content == "done"
        assert len(model.seen) == 2
        assert not executor._active

    asyncio.run(scenario())
