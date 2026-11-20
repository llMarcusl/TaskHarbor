"""Real LangGraph pause/resume without network, Docker or a model provider."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from taskharbor.agent.middleware import SandboxLifecycleMiddleware, ToolErrorMiddleware
from taskharbor.agent.security_middleware import ToolSecurityMiddleware
from taskharbor.security.policy import PolicyEngine
from taskharbor.security.registry import tool_spec
from taskharbor.security.repository import InMemoryApprovalStore
from tests.test_agent_refactor import ScriptedModel, call


def make_graph(
    tools,
    *,
    calls=None,
    rules=(),
    manager=None,
    store=None,
    responses=None,
    model=None,
    max_concurrency=4,
    worker=False,
    checkpointer=None,
):
    manager = manager or MagicMock()
    manager.resolve_owner.return_value = "alice"
    settings = SimpleNamespace(
        HOST_COMMANDS_ENABLED=False, TOOL_SECURITY_RULES=list(rules), TOOL_APPROVAL_TTL_SECONDS=1800
    )
    store = store or InMemoryApprovalStore()
    responses = responses or [
        AIMessage(content="", tool_calls=calls or [call(tools[0].name)]),
        AIMessage(content="done"),
    ]
    model = model or ScriptedModel(responses=responses)
    graph = create_agent(
        model=model,
        tools=tools,
        middleware=[
            ToolSecurityMiddleware(manager, settings, tools, store, worker=worker),
            SandboxLifecycleMiddleware(manager),
            ToolErrorMiddleware(),
        ],
        checkpointer=checkpointer or InMemorySaver(),
    ).with_config(max_concurrency=max_concurrency)
    return graph, store, manager


def run(graph, value, async_mode=False, *, run_id="origin"):
    config = {"configurable": {"thread_id": "t", "run_id": run_id}}
    if isinstance(value, Command) and value.resume is not None:
        approval_id = (
            value.resume.get("approval_id") if isinstance(value.resume, dict) else value.resume
        )
        pending = graph.get_state(config).interrupts
        selected = next(
            (
                item
                for item in pending
                if isinstance(item.value, dict) and item.value.get("approval_id") == approval_id
            ),
            None,
        )
        if selected is not None:
            value = Command(resume={selected.id: value.resume})
    if async_mode:
        return asyncio.run(graph.ainvoke(value, config))
    return graph.invoke(value, config)


@pytest.mark.parametrize("async_mode", [False, True])
def test_user_approval_resumes_the_saved_call(async_mode):
    events = []

    @tool
    def custom_action() -> str:
        """An operation requiring a user decision."""
        events.append("executed")
        return "ok"

    graph, store, manager = make_graph([custom_action])
    result = run(graph, {"messages": [HumanMessage(content="do it")]}, async_mode)
    assert result.get("__interrupt__")
    assert not events
    manager.end_turn.assert_called()
    record = next(r for r in store.list_thread("alice", "t") if r["status"] == "pending")
    store.decide(record["id"], "alice", "t", "allow")
    result = run(graph, Command(resume={"approval_id": record["id"]}), async_mode, run_id="resume")
    assert events == ["executed"]
    assert result["messages"][-1].content == "done"
    assert store.get(record["id"], "alice", "t")["status"] == "succeeded"


@pytest.mark.parametrize("async_mode", [False, True])
def test_failed_tool_can_be_retried_by_a_new_model_call(async_mode):
    attempts = []

    @tool
    def execute_code(code: str) -> str:
        """Run code and report an error for the first attempt."""
        attempts.append(code)
        if code == "bad code":
            raise ValueError("Fix the code and try again")
        return "fixed"

    # Reusing the tool-call ID in a later model message is still a new attempt.
    graph, store, _ = make_graph(
        [execute_code],
        responses=[
            AIMessage(content="", tool_calls=[call("execute_code", {"code": "bad code"})]),
            AIMessage(content="", tool_calls=[call("execute_code", {"code": "fixed code"})]),
            AIMessage(content="done"),
        ],
    )
    result = run(graph, {"messages": [HumanMessage(content="run and fix it")]}, async_mode)
    assert attempts == ["bad code", "fixed code"]
    assert result["messages"][-1].content == "done"
    assert {record["status"] for record in store.list_thread("alice", "t")} == {
        "failed",
        "succeeded",
    }
    run(graph, None, async_mode)
    assert attempts == ["bad code", "fixed code"]


@pytest.mark.parametrize("async_mode", [False, True])
def test_denial_is_consumed_without_executing(async_mode):
    events = []

    @tool
    def custom_action() -> str:
        """An operation requiring a user decision."""
        events.append("executed")
        return "ok"

    graph, store, _ = make_graph([custom_action])
    run(graph, {"messages": [HumanMessage(content="do it")]}, async_mode)
    record = next(r for r in store.list_thread("alice", "t") if r["status"] == "pending")
    store.decide(record["id"], "alice", "t", "deny")
    result = run(graph, Command(resume=record["id"]), async_mode)
    assert not events
    message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert json.loads(message.content)["error"] == "APPROVAL_DENIED"
    assert store.get(record["id"], "alice", "t")["result"] is not None


@pytest.mark.parametrize("async_mode", [False, True])
def test_pending_approval_does_not_block_automatic_calls_or_replay_them(async_mode):
    events = []

    @tool
    def execute_code() -> str:
        """Automatic operation A."""
        events.append("A")
        return "A"

    @tool
    def custom_action() -> str:
        """Confirmed operation B."""
        events.append("B")
        return "B"

    @tool
    def execute_terminal() -> str:
        """Automatic operation C."""
        events.append("C")
        return "C"

    tools = [execute_code, custom_action, execute_terminal]
    graph, store, _ = make_graph(tools, calls=[call(t.name, id=t.name) for t in tools])
    run(graph, {"messages": [HumanMessage(content="do all")]}, async_mode)
    assert sorted(events) == ["A", "C"]
    record = next(r for r in store.list_thread("alice", "t") if r["status"] == "pending")
    store.decide(record["id"], "alice", "t", "allow")
    run(graph, Command(resume=record["id"]), async_mode)
    assert sorted(events) == ["A", "B", "C"]


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("count", [2, 3])
def test_consecutive_approvals_wait_for_each_call_then_deliver_all_results(async_mode, count):
    events = []
    labels = list("ABC"[:count])

    @tool
    def custom_action(label: str) -> str:
        """An operation requiring a separate decision for each invocation."""
        events.append(label)
        return label

    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[call("custom_action", {"label": label}, id=label) for label in labels],
            ),
            AIMessage(content="all done"),
        ]
    )
    graph, store, _ = make_graph([custom_action], model=model)
    result = run(graph, {"messages": [HumanMessage(content="do the whole batch")]}, async_mode)
    assert result.get("__interrupt__")
    assert events == []
    for index, label in enumerate(labels):
        records = store.list_thread("alice", "t")
        pending = [record for record in records if record["status"] == "pending"]
        assert len(pending) == count - index
        record = next(record for record in pending if record["arguments"]["label"] == label)
        assert record["arguments"] == {"label": label}
        assert len(model.seen) == 1  # Partial results must not reach the model yet.
        store.decide(record["id"], "alice", "t", "allow")
        result = run(graph, Command(resume={"approval_id": record["id"]}), async_mode)
        assert events == labels[: index + 1]
        if index < count - 1:
            assert result.get("__interrupt__")
            assert len(model.seen) == 1
    assert not result.get("__interrupt__")
    assert result["messages"][-1].content == "all done"
    assert len(model.seen) == 2
    outputs = [message for message in model.seen[-1] if isinstance(message, ToolMessage)]
    assert [(message.tool_call_id, message.content) for message in outputs] == [
        (label, label) for label in labels
    ]
    assert {record["status"] for record in store.list_thread("alice", "t")} == {"succeeded"}
    run(graph, None, async_mode)
    assert events == labels


@pytest.mark.parametrize("async_mode", [False, True])
def test_denial_in_confirmed_batch_continues_to_next_approval(async_mode):
    events = []

    @tool
    def custom_action(label: str) -> str:
        """An operation requiring a user decision."""
        events.append(label)
        return label

    graph, store, _ = make_graph(
        [custom_action],
        calls=[call("custom_action", {"label": label}, id=label) for label in "ABC"],
    )
    run(graph, {"messages": [HumanMessage(content="do all")]}, async_mode)
    for label in "ABC":
        pending = next(
            r
            for r in store.list_thread("alice", "t")
            if r["status"] == "pending" and r["arguments"]["label"] == label
        )
        assert pending["arguments"]["label"] == label
        store.decide(pending["id"], "alice", "t", "deny" if label == "B" else "allow")
        result = run(graph, Command(resume=pending["id"]), async_mode)
        assert bool(result.get("__interrupt__")) == (label != "C")
    assert events == ["A", "C"]
    outputs = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in outputs] == ["A", "B", "C"]
    assert json.loads(outputs[1].content)["error"] == "APPROVAL_DENIED"


@pytest.mark.parametrize("async_mode", [False, True])
def test_cancel_between_batch_approvals_keeps_completed_call_and_stops_later_calls(async_mode):
    events = []

    @tool
    def custom_action(label: str) -> str:
        """An operation requiring a user decision."""
        events.append(label)
        return label

    graph, store, _ = make_graph(
        [custom_action],
        calls=[call("custom_action", {"label": label}, id=label) for label in "ABC"],
    )
    run(graph, {"messages": [HumanMessage(content="do all")]}, async_mode)
    first = next(r for r in store.list_thread("alice", "t") if r["arguments"]["label"] == "A")
    store.decide(first["id"], "alice", "t", "allow")
    paused = run(graph, Command(resume=first["id"]), async_mode)
    assert paused.get("__interrupt__")
    assert events == ["A"]
    second = next(r for r in store.list_thread("alice", "t") if r["arguments"]["label"] == "B")
    store.cancel_thread("alice", "t")
    result = run(graph, Command(resume=second["id"]), async_mode)
    assert not result.get("__interrupt__")
    assert events == ["A"]
    records = store.list_thread("alice", "t")
    assert {r["arguments"]["label"]: r["status"] for r in records} == {
        "A": "succeeded",
        "B": "cancelled",
        "C": "cancelled",
    }


@pytest.mark.parametrize("async_mode", [False, True])
def test_resume_notifications_without_a_saved_decision_keep_waiting(async_mode):
    events = []

    @tool
    def custom_action() -> str:
        """An operation requiring a saved approval decision."""
        events.append("executed")
        return "ok"

    graph, store, _ = make_graph([custom_action])
    run(graph, {"messages": [HumanMessage(content="do it")]}, async_mode)
    record = next(r for r in store.list_thread("alice", "t") if r["status"] == "pending")
    for _ in range(2):
        result = run(
            graph, Command(resume={"approval_id": record["id"], "decision": "allow"}), async_mode
        )
        assert result.get("__interrupt__")
        assert store.get(record["id"], "alice", "t")["status"] == "pending"
        assert events == []
    store.decide(record["id"], "alice", "t", "allow")
    result = run(graph, Command(resume=record["id"]), async_mode)
    assert events == ["executed"]
    assert not result.get("__interrupt__")


@pytest.mark.parametrize("async_mode", [False, True])
def test_cancelled_turn_cannot_continue_tools(async_mode):
    events = []

    @tool
    def custom_action() -> str:
        """Confirmed operation."""
        events.append("executed")
        return "ok"

    graph, store, _ = make_graph([custom_action])
    run(graph, {"messages": [HumanMessage(content="do it")]}, async_mode)
    record = next(r for r in store.list_thread("alice", "t") if r["status"] == "pending")
    store.cancel_thread("alice", "t")
    result = run(graph, Command(resume=record["id"]), async_mode)
    assert not events
    assert not result.get("__interrupt__")


def test_deny_precedes_allow_and_cwd_rules_use_path_boundaries(tmp_path):
    spec = tool_spec("execute_host_command")
    policy = PolicyEngine(
        [
            {"effect": "allow", "group": "terminal"},
            {"effect": "deny", "tool": "execute_host_command", "cwd": str(tmp_path)},
        ]
    )
    assert policy.decide(spec, {"cwd": str(tmp_path / "child")}) == "deny"
    assert policy.decide(spec, {"cwd": str(tmp_path) + "-other"}) == "allow"


@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("async_mode", [False, True])
def test_original_container_disappeared_after_approval_never_executes(replacement, async_mode):
    events = []

    @tool
    def execute_code(session_id: str, code: str) -> str:
        """A sandbox operation explicitly requiring confirmation."""
        events.append("executed")
        return "ok"

    execute_code.metadata = {"requires_confirmation": True}
    manager = MagicMock()
    manager.sessions = {
        "s": SimpleNamespace(
            user_id="alice",
            thread_id="t",
            container_id="original-container",
        )
    }

    def check(sid):
        if sid not in manager.sessions:
            raise ValueError("Original sandbox was recycled; files remain, variables do not")

    manager._check_session.side_effect = check
    graph, store, _ = make_graph(
        [execute_code],
        manager=manager,
        calls=[call("execute_code", {"session_id": "s", "code": "x += 1"})],
    )
    run(graph, {"messages": [HumanMessage(content="do it")]}, async_mode)
    record = store.list_thread("alice", "t")[0]
    store.decide(record["id"], "alice", "t", "allow")
    if replacement:
        manager.sessions["s"].container_id = "replacement-container"
    else:
        manager.sessions.clear()
    run(graph, Command(resume=record["id"]), async_mode)
    assert not events
    assert store.get(record["id"], "alice", "t")["status"] == "superseded"


def test_expired_decision_never_executes():
    from datetime import UTC, datetime, timedelta

    now = [datetime.now(UTC)]
    store = InMemoryApprovalStore(clock=lambda: now[0])
    events = []

    @tool
    def custom_action() -> str:
        """An operation needing confirmation."""
        events.append("executed")
        return "ok"

    graph, store, _ = make_graph([custom_action], store=store)
    run(graph, {"messages": [HumanMessage(content="do it")]})
    record = store.list_thread("alice", "t")[0]
    now[0] += timedelta(hours=1)
    run(graph, Command(resume=record["id"]))
    assert not events
    assert store.get(record["id"], "alice", "t")["status"] == "expired"
