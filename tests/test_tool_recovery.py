"""Interruption recovery through real graph execution, without Docker or LLM I/O."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from taskharbor.agent import factory
from taskharbor.agent.tool_recovery import INTERRUPTED_TOOL_RESULT, repair_tool_messages
from taskharbor.sandbox import SandboxManager
from taskharbor.settings import Settings
from tests.test_agent_refactor import ScriptedModel, call


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    settings = Settings(
        _env_file=None,
        SUBAGENTS_ENABLED=False,
        CHAT_MODEL_API_KEY="test-key",
        POSTGRES_USER="test",
        POSTGRES_PASSWORD="test",
        POSTGRES_DB="test",
        POSTGRES_HOST="localhost",
        POSTGRES_PORT="5432",
        SESSION_MEMORY_ENABLED=False,
        LONG_TERM_MEMORY_ENABLED=False,
    )
    monkeypatch.setattr(factory, "get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.agent.configuration.get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.crypto.resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(
        "taskharbor.agent.model_registry.init_chat_model",
        lambda **_: pytest.fail("Tests must never create a network model"),
    )


def test_repair_partial_results_in_order_without_mutation():
    ai = AIMessage(
        content="", tool_calls=[call("create_session", id="a"), call("execute_code", id="b")]
    )
    user = HumanMessage(content="continue")
    real = ToolMessage(content="actual output", tool_call_id="b", status="success")
    messages = [ai, user, real, ToolMessage(content="duplicate", tool_call_id="b")]
    before = [m.model_dump() for m in messages]
    patched = repair_tool_messages(messages)
    assert patched[0] is ai
    assert patched[1].tool_call_id == "a"
    assert patched[1].content == INTERRUPTED_TOOL_RESULT
    assert patched[1].status == "error"
    assert patched[2] is real
    assert patched[3] is user
    assert [m.model_dump() for m in messages] == before
    assert repair_tool_messages(patched) == patched


def test_orphan_and_reused_ids_do_not_attach_result_to_old_call():
    old = AIMessage(content="", tool_calls=[call("create_session", id="same")])
    new = AIMessage(content="", tool_calls=[call("create_session", id="same")])
    real = ToolMessage(content="new result", tool_call_id="same")
    orphan = ToolMessage(content="orphan", tool_call_id="unknown")
    patched = repair_tool_messages([orphan, old, HumanMessage(content="retry"), new, real])
    assert patched[1].content == INTERRUPTED_TOOL_RESULT
    assert patched[-1] is real
    assert orphan not in patched


@pytest.mark.parametrize("async_mode", [False, True])
def test_repaired_input_is_not_checkpointed_and_next_turn_remains_valid(async_mode):
    model = ScriptedModel(responses=[AIMessage(content="first"), AIMessage(content="second")])
    graph = factory.build_agent(
        manager=MagicMock(spec=SandboxManager), llm=model, checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "broken-thread", "run_id": "run-new"}}
    next_config = {"configurable": {"thread_id": "broken-thread", "run_id": "run-next"}}
    old = AIMessage(content="", id="old-ai", tool_calls=[call("create_session", id="lost")])
    initial = [HumanMessage(content="start"), old, HumanMessage(content="continue")]

    async def execute():
        await graph.ainvoke({"messages": initial}, config)
        return await graph.ainvoke({"messages": [HumanMessage(content="again")]}, next_config)

    if async_mode:
        result = asyncio.run(execute())
    else:
        graph.invoke({"messages": initial}, config)
        result = graph.invoke({"messages": [HumanMessage(content="again")]}, next_config)
    assert result["messages"][-1].content == "second"
    assert not any(isinstance(m, ToolMessage) for m in result["messages"])
    stored = graph.get_state(config).values["messages"]
    assert not any(isinstance(m, ToolMessage) for m in stored)
    for request in model.seen:
        index = next(i for i, message in enumerate(request) if message.id == "old-ai")
        assert request[index + 1].tool_call_id == "lost"
        assert request[index + 1].status == "error"


def test_real_async_cancel_then_continue_does_not_reexecute_tool():
    async def execute():
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        manager = MagicMock(spec=SandboxManager)

        def create(**kwargs):
            started.set()
            try:
                assert release.wait(5), "Test failed to release the simulated blocking tool"
                return SimpleNamespace(
                    session_id="s1", runtime="python", status="running", dependencies={}, stderr=""
                )
            finally:
                finished.set()

        manager.create_session.side_effect = create
        model = ScriptedModel(
            responses=[
                AIMessage(content="", tool_calls=[call("create_session", id="create")]),
                AIMessage(content="continued"),
            ]
        )
        graph = factory.build_agent(manager=manager, llm=model, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "t", "run_id": "old-run"}}
        task = asyncio.create_task(
            graph.ainvoke({"messages": [HumanMessage(content="start")]}, config)
        )
        try:
            assert await asyncio.to_thread(started.wait, 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            stored = (await graph.aget_state(config)).values["messages"]
            tool_call = next(m for m in stored if isinstance(m, AIMessage))
            assert tool_call.additional_kwargs["sandbox_run_id"] == "old-run"
            assert not any(isinstance(m, ToolMessage) for m in stored)
            result = await asyncio.wait_for(
                graph.ainvoke(
                    {"messages": [HumanMessage(content="continue")]},
                    {"configurable": {"thread_id": "t", "run_id": "next-run"}},
                ),
                2,
            )
            assert result["messages"][-1].content == "continued"
            assert manager.create_session.call_count == 1
            assert not any(isinstance(m, ToolMessage) for m in result["messages"])
            assert next(m for m in model.seen[-1] if isinstance(m, ToolMessage)).status == "error"
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            assert await asyncio.to_thread(finished.wait, 3)

    asyncio.run(execute())
