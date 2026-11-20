"""Consecutive approvals through Aegra's actual background run and stream bridge.

Only database persistence, graph lookup and event delivery are replaced. The
installed Aegra resume admission, run execution and LangGraph streaming stay real.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aegra_api.api import runs
from aegra_api.models import User
from aegra_api.services import langgraph_service
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from tests.test_agent_refactor import ScriptedModel, call
from tests.test_security_graph import make_graph


@pytest.mark.parametrize("count", [2, 3])
def test_aegra_keeps_each_approval_resumable_until_the_batch_completes(monkeypatch, count):
    attempts = []
    labels = list("ABC"[:count])

    @tool
    def custom_action(label: str) -> str:
        """An operation requiring a saved user decision before execution."""
        attempts.append(label)
        return label

    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[call("custom_action", {"label": label}, id=label) for label in labels],
            ),
            AIMessage(content="batch complete"),
        ]
    )
    graph, store, _ = make_graph([custom_action], model=model)
    thread = SimpleNamespace(status="idle")
    session = SimpleNamespace(scalar=AsyncMock(return_value=thread))
    statuses = {}
    outputs = {}
    events = {}

    async def update_run_status(run_id, status, output=None, error=None, session=None):
        statuses.setdefault(run_id, []).append(status)
        if output is not None:
            outputs[run_id] = output
        assert error is None, f"The approval stream must not fail: {error}"

    async def set_thread_status(session, thread_id, status):
        assert thread_id == "t"
        thread.status = status

    async def put_event(run_id, event_id, event):
        events.setdefault(run_id, []).append(event)

    @asynccontextmanager
    async def get_graph(graph_id):
        assert graph_id == "security-test"
        yield graph

    transport = SimpleNamespace(
        put_to_broker=put_event,
        store_event_from_raw=AsyncMock(),
        signal_run_error=AsyncMock(),
        signal_run_cancelled=AsyncMock(),
        cleanup_run=AsyncMock(),
    )
    monkeypatch.setattr(runs, "update_run_status", update_run_status)
    monkeypatch.setattr(runs, "set_thread_status", set_thread_status)
    monkeypatch.setattr(runs, "get_langgraph_service", lambda: SimpleNamespace(get_graph=get_graph))
    monkeypatch.setattr(runs, "streaming_service", transport)
    monkeypatch.setattr(runs, "broker_manager", SimpleNamespace(get_broker=lambda _: None))
    monkeypatch.setattr(langgraph_service, "get_tracing_callbacks", lambda: [])
    monkeypatch.setattr(langgraph_service, "get_tracing_metadata", lambda *args: {})

    async def execute(run_id, *, command=None):
        # This is the exact admission check that returned HTTP 400 after the old
        # second-interrupt regression changed the persisted thread to error.
        await runs._validate_resume_command(session, "t", command)
        await runs.execute_run_async(
            run_id=run_id,
            thread_id="t",
            graph_id="security-test",
            input_data={"messages": [HumanMessage(content="execute the batch")]},
            user=User(identity="alice"),
            stream_mode=["values"],  # Same mode used by the current frontend.
            session=session,
            command=command,
        )

    async def scenario():
        await execute("initial")
        assert statuses["initial"] == ["running", "interrupted"]
        assert thread.status == "interrupted"
        assert not attempts
        for index, label in enumerate(labels):
            assert thread.status == "interrupted"
            assert len(model.seen) == 1
            pending = [r for r in store.list_thread("alice", "t") if r["status"] == "pending"]
            assert len(pending) == count - index
            record = next(record for record in pending if record["arguments"]["label"] == label)
            assert record["arguments"] == {"label": label}
            store.decide(record["id"], "alice", "t", "allow")
            run_id = f"resume-{label}"
            snapshot = await graph.aget_state({"configurable": {"thread_id": "t"}})
            selected = next(
                item
                for item in snapshot.interrupts
                if item.value.get("approval_id") == record["id"]
            )
            await execute(
                run_id,
                command={
                    "resume": {
                        selected.id: {"approval_id": record["id"]},
                    }
                },
            )
            assert attempts == labels[: index + 1]
            if index < count - 1:
                assert statuses[run_id] == ["running", "interrupted"]
                assert thread.status == "interrupted"
                assert len(model.seen) == 1
                assert any(
                    isinstance(data, dict) and data.get("__interrupt__")
                    for _, data in events[run_id]
                )
            else:
                assert statuses[run_id] == ["running", "success"]
                assert thread.status == "idle"
                assert outputs[run_id]["messages"][-1].content == "batch complete"

        assert len(model.seen) == 2
        results = [message for message in model.seen[-1] if isinstance(message, ToolMessage)]
        assert [(message.tool_call_id, message.content) for message in results] == [
            (label, label) for label in labels
        ]
        transport.signal_run_error.assert_not_awaited()
        transport.signal_run_cancelled.assert_not_awaited()
        assert transport.cleanup_run.await_count == count + 1

    asyncio.run(scenario())
