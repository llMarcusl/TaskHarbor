"""Windows host commands through the actual factory/approval graph.

No model provider, Docker or database is used. Real commands write only their
own pytest temporary directory, and every execution has a short timeout.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from taskharbor.agent import factory
from taskharbor.security import host
from taskharbor.security.repository import InMemoryApprovalStore
from taskharbor.settings import Settings
from taskharbor.tools import execute_host_command as host_tool
from tests.test_agent_refactor import ScriptedModel, call


@pytest.fixture
def build_host_graph(monkeypatch):
    executors = []

    def build(*, enabled=True, rules=(), calls=None):
        settings = Settings(
            _env_file=None,
            SUBAGENTS_ENABLED=False,
            CHAT_MODEL_API_KEY="test-key",
            CHAT_MODEL_BASE_URL=None,
            POSTGRES_USER="test",
            POSTGRES_PASSWORD="test",
            POSTGRES_DB="test",
            POSTGRES_HOST="localhost",
            POSTGRES_PORT="5432",
            HOST_COMMANDS_ENABLED=enabled,
            TOOL_SECURITY_RULES=list(rules),
            SESSION_MEMORY_ENABLED=False,
            LONG_TERM_MEMORY_ENABLED=False,
            CONTEXT_COMPACTION_ENABLED=False,
        )
        monkeypatch.setattr(factory, "get_settings", lambda: settings)
        monkeypatch.setattr("taskharbor.agent.configuration.get_settings", lambda: settings)
        monkeypatch.setattr("taskharbor.crypto.resolve_api_key", lambda: "test-key")
        monkeypatch.setattr(host, "get_settings", lambda: settings)
        monkeypatch.setattr(host_tool, "get_settings", lambda: settings)
        executor = host.HostCommandExecutor()
        executors.append(executor)
        monkeypatch.setattr(host, "get_host_executor", lambda: executor)
        monkeypatch.setattr(host_tool, "get_host_executor", lambda: executor)
        manager = MagicMock()
        manager.resolve_owner.return_value = "alice"
        store = InMemoryApprovalStore()
        responses = [AIMessage(content="done")]
        if calls:
            responses.insert(0, AIMessage(content="", tool_calls=calls))
        model = ScriptedModel(responses=responses)
        graph = factory.build_agent(
            manager=manager,
            llm=model,
            checkpointer=InMemorySaver(),
            security_store=store,
        )
        return graph, store, model, executor

    yield build
    for executor in executors:
        executor.shutdown()


def invoke(graph, value, async_mode=False):
    config = {"configurable": {"thread_id": "host-graph", "run_id": "host-run"}}
    if async_mode:
        return asyncio.run(graph.ainvoke(value, config))
    return graph.invoke(value, config)


def test_disabled_host_command_is_not_registered(build_host_graph):
    graph, store, model, _ = build_host_graph(enabled=False)
    result = invoke(graph, {"messages": [HumanMessage(content="hello")]})
    assert result["messages"][-1].content == "done"
    assert "execute_host_command" not in model.bound_names
    assert not store.list_thread("alice", "host-graph")


@pytest.mark.parametrize("async_mode", [False, True])
def test_explicit_deny_blocks_host_before_preparation(build_host_graph, monkeypatch, async_mode):
    graph, store, _, executor = build_host_graph(
        rules=[{"effect": "deny", "tool": "execute_host_command"}],
        calls=[call("execute_host_command", {"command": "Get-Location", "cwd": "C:/"})],
    )
    monkeypatch.setattr(
        executor,
        "run",
        lambda *args, **kwargs: pytest.fail("Denied host command must not execute"),
    )
    result = invoke(graph, {"messages": [HumanMessage(content="do it")]}, async_mode)
    message = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert json.loads(message.content)["error"] == "TOOL_DENIED"
    assert not result.get("__interrupt__")
    assert not store.list_thread("alice", "host-graph")


@pytest.mark.skipif(os.name != "nt", reason="Real approved Windows host process")
@pytest.mark.parametrize("async_mode", [False, True])
def test_real_host_executes_exactly_once_only_after_saved_approval(
    build_host_graph, tmp_path, async_mode
):
    output = tmp_path / "批准后的文件.txt"
    escaped = str(output).replace("'", "''")
    command = f"[IO.File]::AppendAllText('{escaped}', \"once`n\"); Write-Output 'approved'"
    args = {"command": command, "cwd": str(tmp_path), "timeout": 10}
    graph, store, _, _ = build_host_graph(
        # A broad allow cannot suppress the mandatory per-command host decision.
        rules=[{"effect": "allow", "group": "terminal"}],
        calls=[call("execute_host_command", args)],
    )
    result = invoke(graph, {"messages": [HumanMessage(content="write test output")]}, async_mode)
    assert result.get("__interrupt__")
    assert not output.exists()
    record = store.list_thread("alice", "host-graph")[0]
    assert record["status"] == "pending"
    assert record["arguments"] == args
    assert record["execution_context"]["powershell_identity"]["sha256"]
    assert record["execution_context"]["powershell_path"]

    store.decide(record["id"], "alice", "host-graph", "allow")
    result = invoke(graph, Command(resume={"approval_id": record["id"]}), async_mode)
    assert not result.get("__interrupt__")
    assert output.read_text(encoding="utf-8") == "once\n"
    message = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    payload = json.loads(message.content)
    assert payload["success"], payload
    assert payload["executor"] == "host" and "approved" in payload["stdout"]
    assert store.get(record["id"], "alice", "host-graph")["status"] == "succeeded"
    # Retrying a completed graph/checkpoint cannot replay the side effect.
    invoke(graph, None, async_mode)
    assert output.read_text(encoding="utf-8") == "once\n"


@pytest.mark.skipif(os.name != "nt", reason="Real approved Windows host process")
@pytest.mark.parametrize("async_mode", [False, True])
def test_failed_host_command_can_be_retried_as_new_approved_call(
    build_host_graph, tmp_path, async_mode
):
    args = {"command": "exit 7", "cwd": str(tmp_path), "timeout": 10}
    graph, store, model, _ = build_host_graph(
        calls=[call("execute_host_command", args, "failed-call")],
    )
    model.responses.insert(
        1,
        AIMessage(
            content="",
            tool_calls=[
                call(
                    "execute_host_command",
                    {**args, "command": "Write-Output 'retry succeeded'"},
                    "retry-call",
                )
            ],
        ),
    )
    result = invoke(
        graph, {"messages": [HumanMessage(content="run and retry if needed")]}, async_mode
    )
    assert result.get("__interrupt__")
    failed = store.list_thread("alice", "host-graph")[0]
    store.decide(failed["id"], "alice", "host-graph", "allow")

    result = invoke(graph, Command(resume={"approval_id": failed["id"]}), async_mode)
    assert result.get("__interrupt__")
    assert store.get(failed["id"], "alice", "host-graph")["status"] == "failed"
    retry = next(
        record
        for record in store.list_thread("alice", "host-graph")
        if record["tool_call_id"] == "retry-call"
    )
    assert retry["id"] != failed["id"] and retry["status"] == "pending"
    store.decide(retry["id"], "alice", "host-graph", "allow")

    result = invoke(graph, Command(resume={"approval_id": retry["id"]}), async_mode)
    assert not result.get("__interrupt__")
    assert result["messages"][-1].content == "done"
    assert store.get(retry["id"], "alice", "host-graph")["status"] == "succeeded"
    messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in messages] == ["failed-call", "retry-call"]
    assert json.loads(messages[0].content)["exit_code"] == 7
    assert "retry succeeded" in json.loads(messages[1].content)["stdout"]
