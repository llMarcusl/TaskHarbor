"""Standard per-tool Send scheduling and checkpoint approval boundaries."""

import asyncio
import json
import threading
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from tests.test_agent_refactor import ScriptedModel, call
from tests.test_security_graph import make_graph, run


def confirmed_graph(action, labels, *, max_concurrency=4):
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[call(action.name, {"label": label}, id=label) for label in labels],
            ),
            AIMessage(content="batch complete"),
        ]
    )
    graph, store, _ = make_graph([action], model=model, max_concurrency=max_concurrency)
    return graph, store, model


async def resume_approvals(graph, config, approval_ids):
    snapshot = await graph.aget_state(config)
    return Command(
        resume={
            item.id: {"approval_id": item.value["approval_id"]}
            for item in snapshot.interrupts
            if item.value.get("approval_id") in approval_ids
        }
    )


@pytest.mark.parametrize("async_mode", [False, True])
def test_tool_parallelism_is_real_and_bounded_in_sync_and_async_graphs(async_mode):
    lock = threading.Lock()
    first_wave = threading.Barrier(2)
    active = 0
    peak = 0
    attempts = []

    @tool
    def execute_code(label: str) -> str:
        """An automatic tool whose first two invocations must overlap."""
        nonlocal active, peak
        with lock:
            first = len(attempts) < 2
            attempts.append(label)
            active += 1
            peak = max(peak, active)
        try:
            if first:
                first_wave.wait(timeout=3)
            time.sleep(0.02)
            return label
        finally:
            with lock:
                active -= 1

    labels = list("ABCDE")
    graph, store, _ = make_graph(
        [execute_code],
        calls=[call("execute_code", {"label": label}, id=label) for label in labels],
        max_concurrency=2,
    )
    result = run(graph, {"messages": [HumanMessage(content="execute independently")]}, async_mode)
    assert peak == 2
    assert sorted(attempts) == labels
    assert active == 0
    outputs = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert [(message.tool_call_id, message.content) for message in outputs] == [
        (label, label) for label in labels
    ]
    assert {record["status"] for record in store.list_thread("alice", "t")} == {"succeeded"}


def test_selected_approval_runs_once_and_later_decision_resumes_its_own_checkpoint():
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        attempts = []

        @tool
        async def custom_action(label: str) -> str:
            """Pause A while B can receive a durable decision."""
            attempts.append(label)
            if label == "A":
                started.set()
                await release.wait()
            return label

        graph, store, model = confirmed_graph(custom_action, "AB")
        config = {"configurable": {"thread_id": "t", "run_id": "selected"}}
        paused = await graph.ainvoke({"messages": [HumanMessage(content="do both")]}, config)
        assert len(paused["__interrupt__"]) == 2
        records = {r["arguments"]["label"]: r for r in store.list_thread("alice", "t")}
        store.decide(records["A"]["id"], "alice", "t", "allow")
        command = await resume_approvals(graph, config, {records["A"]["id"]})
        execution = asyncio.create_task(graph.ainvoke(command, config))
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            store.decide(records["B"]["id"], "alice", "t", "allow")
            assert store.get(records["B"]["id"], "alice", "t")["status"] == "approved"
            assert attempts == ["A"]
            assert len(model.seen) == 1
            release.set()
            partial = await asyncio.wait_for(execution, timeout=3)
        finally:
            release.set()
            if not execution.done():
                execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        assert partial.get("__interrupt__")
        command = await resume_approvals(graph, config, {records["B"]["id"]})
        result = await graph.ainvoke(command, config)
        assert attempts == ["A", "B"]
        assert result["messages"][-1].content == "batch complete"
        assert len(model.seen) == 2
        await graph.ainvoke(None, config)
        assert attempts == ["A", "B"]

    asyncio.run(scenario())


def test_multiple_saved_approvals_execute_concurrently_and_denial_skips_only_its_call():
    async def scenario():
        barrier = asyncio.Event()
        started = set()
        attempts = []

        @tool
        async def custom_action(label: str) -> str:
            """Both approved calls must start before either can finish."""
            attempts.append(label)
            started.add(label)
            if started == {"A", "C"}:
                barrier.set()
            await asyncio.wait_for(barrier.wait(), timeout=3)
            return label

        graph, store, model = confirmed_graph(custom_action, "ABC")
        config = {"configurable": {"thread_id": "t", "run_id": "decisions"}}
        await graph.ainvoke({"messages": [HumanMessage(content="do all")]}, config)
        records = {r["arguments"]["label"]: r for r in store.list_thread("alice", "t")}
        for label, record in records.items():
            store.decide(record["id"], "alice", "t", "deny" if label == "B" else "allow")
        command = await resume_approvals(graph, config, {r["id"] for r in records.values()})
        result = await graph.ainvoke(command, config)
        assert sorted(attempts) == ["A", "C"]
        assert len(model.seen) == 2
        outputs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert [m.tool_call_id for m in outputs] == ["A", "B", "C"]
        assert json.loads(outputs[1].content)["error"] == "APPROVAL_DENIED"
        assert store.get(records["B"]["id"], "alice", "t")["status"] == "denied"

    asyncio.run(scenario())


@pytest.mark.parametrize("max_concurrency", [1, 4])
def test_service_run_cancellation_stops_active_and_queued_calls(max_concurrency):
    async def scenario():
        started = asyncio.Event()
        stopped = asyncio.Event()
        attempts = []

        @tool
        async def custom_action(label: str) -> str:
            """Wait until cancellation of the owning service run."""
            attempts.append(label)
            started.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()

        graph, store, model = confirmed_graph(custom_action, "ABC", max_concurrency=max_concurrency)
        config = {"configurable": {"thread_id": "t", "run_id": "cancel"}}
        await graph.ainvoke({"messages": [HumanMessage(content="do all")]}, config)
        records = store.list_thread("alice", "t")
        for record in records:
            store.decide(record["id"], "alice", "t", "allow")
        command = await resume_approvals(graph, config, {r["id"] for r in records})
        execution = asyncio.create_task(graph.ainvoke(command, config))
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            if max_concurrency == 1:
                assert attempts == ["A"]
            store.cancel_thread("alice", "t")
            execution.cancel()
            with pytest.raises(asyncio.CancelledError):
                await execution
        finally:
            if not execution.done():
                execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        assert stopped.is_set()
        assert len(model.seen) == 1
        if max_concurrency == 1:
            assert attempts == ["A"]
        await graph.ainvoke(None, config)
        assert len(model.seen) == 1
        assert not store.blocked_thread("alice", "t")

    asyncio.run(scenario())
