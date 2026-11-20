"""Additional real-graph security tests through the production agent factory."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from taskharbor.agent import factory
from taskharbor.security.repository import InMemoryApprovalStore
from taskharbor.settings import Settings
from tests.test_agent_refactor import ScriptedModel, call


@pytest.fixture
def agent_builder(monkeypatch):
    settings = Settings(
        _env_file=None,
        SUBAGENTS_ENABLED=False,
        CHAT_MODEL_API_KEY="test-key",
        POSTGRES_USER="test",
        POSTGRES_PASSWORD="test",
        POSTGRES_DB="test",
        POSTGRES_HOST="localhost",
        POSTGRES_PORT="5432",
    )
    monkeypatch.setattr(factory, "get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.agent.configuration.get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.crypto.resolve_api_key", lambda: "test-key")

    def build(tools, *, rules=(), calls=None):
        settings.TOOL_SECURITY_RULES = list(rules)
        monkeypatch.setattr(factory, "create_tools", lambda _: tools)
        store = InMemoryApprovalStore()
        manager = MagicMock()
        manager.resolve_owner.return_value = "alice"
        model = ScriptedModel(
            responses=[
                AIMessage(content="", tool_calls=calls or [call(tools[0].name)]),
                AIMessage(content="done"),
            ]
        )
        graph = factory.build_agent(
            manager=manager, llm=model, checkpointer=InMemorySaver(), security_store=store
        )
        return graph, store

    return build


def run(graph, value, async_mode):
    config = {"configurable": {"thread_id": "t", "run_id": "run"}}
    if async_mode:
        return asyncio.run(graph.ainvoke(value, config))
    return graph.invoke(value, config)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("denied", [False, True])
def test_factory_rule_allow_or_deny_precedence(agent_builder, async_mode, denied):
    executed = []

    @tool
    def custom_action() -> str:
        """Perform an action covered by explicit server rules."""
        executed.append(True)
        return "ok"

    rules = [{"effect": "allow", "tool": "custom_action"}]
    if denied:
        rules.append({"effect": "ask", "tool": "custom_action"})
        rules.append({"effect": "deny", "group": "integration"})
    graph, store = agent_builder([custom_action], rules=rules)
    result = run(graph, {"messages": [HumanMessage(content="do it")]}, async_mode)
    assert not result.get("__interrupt__")
    assert executed == ([] if denied else [True])
    if denied:
        output = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        assert json.loads(output.content)["error"] == "TOOL_DENIED"
    else:
        assert store.list_thread("alice", "t")[0]["status"] == "succeeded"


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("decision", ["allow", "deny"])
def test_ask_rule_pauses_automatic_tool_before_execution(agent_builder, async_mode, decision):
    executed = []

    @tool
    def execute_terminal() -> str:
        """Record whether a normally automatic sandbox tool actually ran."""
        executed.append(True)
        return "ok"

    graph, store = agent_builder(
        [execute_terminal],
        rules=[
            {"effect": "allow", "group": "sandbox"},
            {"effect": "ask", "tool": "execute_terminal"},
        ],
    )
    paused = run(graph, {"messages": [HumanMessage(content="do it")]}, async_mode)
    assert paused.get("__interrupt__")
    assert executed == []
    record = store.list_thread("alice", "t")[0]
    assert record["status"] == "pending"
    store.decide(record["id"], "alice", "t", decision)
    result = run(
        graph,
        Command(resume={paused["__interrupt__"][0].id: {"approval_id": record["id"]}}),
        async_mode,
    )
    assert not result.get("__interrupt__")
    assert executed == ([True] if decision == "allow" else [])


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("status", ["executing", "unknown"])
def test_started_or_uncertain_execution_never_replays(agent_builder, async_mode, status):
    executed = []

    @tool
    def custom_action() -> str:
        """An external action that must not be executed twice."""
        executed.append(True)
        return "ok"

    graph, store = agent_builder([custom_action])
    run(graph, {"messages": [HumanMessage(content="do it")]}, async_mode)
    record = store.list_thread("alice", "t")[0]
    store.decide(record["id"], "alice", "t", "allow")
    store.claim(record["id"], "alice", "t", record["request_digest"])
    if status == "unknown":
        store.finish(record["id"], "unknown")
    result = run(graph, Command(resume={"approval_id": record["id"]}), async_mode)
    assert not executed
    assert not result.get("__interrupt__")
    assert store.get(record["id"], "alice", "t")["status"] == status
    assert store.blocked_thread("alice", "t")


@pytest.mark.parametrize("async_mode", [False, True])
def test_factory_batch_cancel_keeps_completed_automatic_calls_and_skips_unapproved_call(
    agent_builder, async_mode
):
    executed = []

    @tool
    def execute_code() -> str:
        """Automatic initial side effect."""
        executed.append("A")
        return "A"

    @tool
    def custom_action() -> str:
        """The action awaiting the user decision."""
        executed.append("B")
        return "B"

    @tool
    def execute_terminal() -> str:
        """Automatic later side effect."""
        executed.append("C")
        return "C"

    tools = [execute_code, custom_action, execute_terminal]
    graph, store = agent_builder(tools, calls=[call(t.name, id=t.name) for t in tools])
    first = run(graph, {"messages": [HumanMessage(content="do all")]}, async_mode)
    assert first.get("__interrupt__")
    assert sorted(executed) == ["A", "C"]
    pending = next(r for r in store.list_thread("alice", "t") if r["status"] == "pending")
    store.cancel_thread("alice", "t")
    run(graph, Command(resume={"approval_id": pending["id"]}), async_mode)
    assert sorted(executed) == ["A", "C"]
    assert store.get(pending["id"], "alice", "t")["status"] == "cancelled"
    assert not store.blocked_thread("alice", "t")


@pytest.mark.parametrize("async_mode", [False, True])
def test_expired_approval_returns_error_to_model_without_executing(
    agent_builder, async_mode, monkeypatch
):
    executed = []

    @tool
    def execute_terminal() -> str:
        """This operation must not run after its approval expires."""
        executed.append(True)
        return "unexpected"

    graph, store = agent_builder(
        [execute_terminal], rules=[{"effect": "ask", "tool": "execute_terminal"}]
    )
    paused = run(graph, {"messages": [HumanMessage(content="do it")]}, async_mode)
    record = store.list_thread("alice", "t")[0]
    later = datetime.fromisoformat(record["expires_at"]) + timedelta(seconds=1)
    store._clock = lambda: later

    class LaterDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return later.astimezone(tz)

    monkeypatch.setattr("taskharbor.agent.security_middleware.datetime", LaterDatetime)
    store.list_thread("alice", "t")
    result = run(graph, Command(resume={
        paused["__interrupt__"][0].id: {"approval_id": record["id"]},
    }), async_mode)
    assert executed == []
    output = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert json.loads(output.content)["error"] == "APPROVAL_EXPIRED"
    assert result["messages"][-1].content == "done"
    assert store.get(record["id"], "alice", "t")["result"] is not None
