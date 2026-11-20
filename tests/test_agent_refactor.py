"""Agent contract tests using real LangGraph execution, without external services."""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph
from pydantic import Field, ValidationError

from taskharbor.agent import factory, model_registry
from taskharbor.agent.factory import build_agent
from taskharbor.agent.middleware import ThreadContextMiddleware
from taskharbor.agent.prompts import SYSTEM_PROMPT
from taskharbor.agent.state import AgentState
from taskharbor.sandbox import SandboxManager, current_thread_id
from taskharbor.sandbox.models import ExecutionResult, ExportFileResult, ExportResult
from taskharbor.settings import Settings
from taskharbor.tools import create_tools


class ScriptedModel(BaseChatModel):
    """A model recording actual inputs and returning a fixed sequence of responses."""

    responses: list[Any]
    seen: list[list[Any]] = Field(default_factory=list)
    bound_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "refactor-test"

    def bind_tools(self, tools, **kwargs):
        self.bound_names = [t.name for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return ChatResult(generations=[ChatGeneration(message=response.model_copy(deep=True))])


@pytest.fixture(autouse=True)
def local_settings(monkeypatch):
    settings = Settings(
        _env_file=None,
        SUBAGENTS_ENABLED=False,
        CHAT_MODEL_API_KEY="test-key",
        POSTGRES_USER="test",
        POSTGRES_PASSWORD="test",
        POSTGRES_DB="test",
        POSTGRES_HOST="localhost",
        POSTGRES_PORT="5432",
        TOOL_SECURITY_RULES=[
            {"effect": "allow", "tool": "failing_tool"},
            {"effect": "allow", "tool": "thread_echo"},
        ],
    )
    monkeypatch.setattr(factory, "get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.agent.configuration.get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.tools._core.get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.crypto.resolve_api_key", lambda: "test-key")
    from taskharbor.security.repository import InMemoryApprovalStore

    store = InMemoryApprovalStore()
    monkeypatch.setattr(
        "taskharbor.agent.security_middleware.get_approval_store", lambda: store
    )
    return settings


def call(name, args=None, id="call-1"):
    return {"name": name, "args": args or {}, "id": id, "type": "tool_call"}


def invoke(graph, messages, async_mode, config=None):
    if async_mode:
        return asyncio.run(graph.ainvoke({"messages": messages}, config=config))
    return graph.invoke({"messages": messages}, config=config)


@pytest.mark.parametrize("async_mode", [False, True])
def test_dependent_tools_rich_output_and_thread_context(async_mode):
    manager = MagicMock(spec=SandboxManager)
    events = []
    rich = [{"type": "image/png", "data": "aW1hZ2U="}, {"type": "text/html", "data": "<b>x</b>"}]

    def execute(*args, **kwargs):
        events.append(("execute", current_thread_id.get()))
        return ExecutionResult(success=True, stdout="ready", display_outputs=rich)

    def export(*args, **kwargs):
        assert events == [("execute", "thread-a")]
        events.append(("export", current_thread_id.get()))
        return ExportResult(
            success=True,
            files=[ExportFileResult("s1", "/workspace/data.csv", True)],
        )

    manager.execute_code.side_effect = execute
    manager.export_files.side_effect = export
    llm = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    call("execute_code", {"session_id": "s1", "code": "make_file()"}, "code"),
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    call(
                        "export_files",
                        {"session_id": "s1", "files": [{"source": "data.csv"}]},
                        "export",
                    ),
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    graph = build_agent(manager=manager, llm=llm)
    result = invoke(
        graph,
        [HumanMessage(content="make a file")],
        async_mode,
        {"configurable": {"thread_id": "thread-a"}},
    )
    assert result["messages"][-1].content == "done"
    outputs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in outputs] == ["code", "export"]
    assert json.loads(outputs[0].content)["display_outputs"] == rich
    assert (
        "/threads/thread-a/files/download"
        in json.loads(outputs[1].content)["files"][0]["download_url"]
    )
    projected = [m for m in llm.seen[1] if isinstance(m, ToolMessage)][0]
    assert projected.content[1]["image_url"]["url"] == "data:image/png;base64,aW1hZ2U="
    assert "display_outputs" not in json.loads(projected.content[0]["text"])
    assert "<b>x</b>" not in str(projected.content)
    assert current_thread_id.get() is None
    assert llm.bound_names == [
        "create_session",
        "execute_code",
        "execute_terminal",
        "import_files",
        "export_files",
        "stop_session",
    ]


@pytest.mark.parametrize("async_mode", [False, True])
def test_tool_errors_match_original_and_do_not_abort_batch(monkeypatch, async_mode):
    @tool
    def failing_tool(value: int) -> str:
        """Raise an execution error."""
        raise RuntimeError("execution failed")

    monkeypatch.setattr(factory, "create_tools", lambda _: [failing_tool])
    llm = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    call("failing_tool", {"value": "bad"}, "invalid"),
                    call("failing_tool", {"value": 1}, "execution"),
                ],
            ),
            AIMessage(content="recovered"),
        ]
    )
    graph = build_agent(manager=MagicMock(), llm=llm)
    result = invoke(graph, [HumanMessage(content="run")], async_mode)
    outputs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    with pytest.raises(ValidationError) as original:
        failing_tool.invoke({"value": "bad"})
    assert json.loads(outputs[0].content)["error"] == f"ValidationError: {original.value}"
    assert json.loads(outputs[1].content) == {
        "success": False,
        "error": "RuntimeError: execution failed",
        "hint": "Fix the invalid parameters and try again.",
    }
    assert all(m.status == "success" for m in outputs)  # Original ToolMessage default.
    assert result["messages"][-1].content == "recovered"


@pytest.mark.parametrize("async_mode", [False, True])
def test_vision_fallback_preserves_history_and_shared_probe(async_mode):
    raw = json.dumps({"success": True, "display_outputs": [{"type": "image/png", "data": "YWJj"}]})
    history = [
        HumanMessage(content="look"),
        AIMessage(content="", tool_calls=[call("execute_code")]),
        ToolMessage(content=raw, tool_call_id="call-1", name="execute_code"),
    ]
    llm = ScriptedModel(
        responses=[
            RuntimeError("image rejected"),
            AIMessage(content="ok"),
            AIMessage(content="next"),
        ]
    )
    graph = build_agent(manager=MagicMock(), llm=llm)
    result = invoke(graph, history, async_mode)
    invoke(graph, history, async_mode)
    assert isinstance(llm.seen[0][-1].content, list)
    assert isinstance(llm.seen[1][-1].content, str)
    assert isinstance(llm.seen[2][-1].content, str)
    assert result["messages"][2].content == raw
    assert history[2].content == raw


@pytest.mark.parametrize("async_mode", [False, True])
def test_text_model_failure_is_not_retried(async_mode):
    llm = ScriptedModel(responses=[RuntimeError("unavailable")])
    graph = build_agent(manager=MagicMock(), llm=llm)
    with pytest.raises(RuntimeError, match="unavailable"):
        invoke(graph, [HumanMessage(content="hello")], async_mode)
    assert len(llm.seen) == 1


@pytest.mark.parametrize("async_mode", [False, True])
def test_model_overrides_cached_and_system_prompt_preserved(monkeypatch, async_mode):
    default = ScriptedModel(responses=[AIMessage(content="default")])
    override = ScriptedModel(responses=[AIMessage(content="override"), AIMessage(content="again")])
    initialize = MagicMock(return_value=override)
    monkeypatch.setattr(model_registry, "init_chat_model", initialize)
    graph = build_agent(manager=MagicMock(), llm=default)
    cfg = {
        "configurable": {
            "chat_model": "other",
            "chat_model_provider": "openai",
            "chat_model_api_key": "override-key",
            "chat_model_base_url": "https://example.invalid/v1",
            "chat_model_supports_vision": "false",
        }
    }
    custom = [SystemMessage(content="custom"), HumanMessage(content="hello")]
    invoke(graph, custom, async_mode, cfg)
    invoke(graph, custom, async_mode, cfg)
    invoke(graph, [HumanMessage(content="default")], async_mode)
    initialize.assert_called_once_with(
        model="other",
        model_provider="openai",
        api_key="override-key",
        base_url="https://example.invalid/v1",
    )
    assert [m.content for m in override.seen[0]] == ["custom", "hello"]
    assert default.seen[0][0].content == SYSTEM_PROMPT


@pytest.mark.parametrize("async_mode", [False, True])
def test_values_stream_checkpoint_edit_and_resume(async_mode):
    llm = ScriptedModel(
        responses=[AIMessage(content="first", id="answer"), AIMessage(content="second")]
    )
    saver = InMemorySaver()
    graph = build_agent(manager=MagicMock(), llm=llm, checkpointer=saver)
    cfg = {"configurable": {"thread_id": "persist"}}

    async def run_async():
        values = [
            v
            async for v in graph.astream(
                {"messages": [HumanMessage(content="hello")]}, cfg, stream_mode="values"
            )
        ]
        await graph.aupdate_state(
            cfg, {"messages": [AIMessage(content="edited", id="answer")]}, as_node="model"
        )
        await graph.ainvoke({"messages": [HumanMessage(content="continue")]}, cfg)
        return values, await graph.aget_state(cfg)

    if async_mode:
        values, snapshot = asyncio.run(run_async())
    else:
        values = list(
            graph.stream({"messages": [HumanMessage(content="hello")]}, cfg, stream_mode="values")
        )
        graph.update_state(
            cfg, {"messages": [AIMessage(content="edited", id="answer")]}, as_node="model"
        )
        graph.invoke({"messages": [HumanMessage(content="continue")]}, cfg)
        snapshot = graph.get_state(cfg)
    assert values[-1]["messages"][-1].content == "first"
    assert [m.content for m in snapshot.values["messages"]] == [
        "hello",
        "edited",
        "continue",
        "second",
    ]
    assert "edited" in [m.content for m in llm.seen[1]]


def test_context_restored_even_on_error():
    middleware = ThreadContextMiddleware()
    request = SimpleNamespace(
        runtime=SimpleNamespace(config={"configurable": {"thread_id": "inner"}})
    )

    def fail(_):
        assert current_thread_id.get() == "inner"
        raise RuntimeError("failed")

    async def afail(req):
        return fail(req)

    token = current_thread_id.set("outer")
    try:
        with pytest.raises(RuntimeError):
            middleware.wrap_tool_call(request, fail)
        assert current_thread_id.get() == "outer"

        async def check_async():
            with pytest.raises(RuntimeError):
                await middleware.awrap_tool_call(request, afail)
            assert current_thread_id.get() == "outer"

        asyncio.run(check_async())
    finally:
        current_thread_id.reset(token)


def test_parallel_runs_keep_thread_ids_separate(monkeypatch):
    @tool
    def thread_echo() -> str:
        """Return the current thread."""
        time.sleep(0.02)
        return current_thread_id.get()

    monkeypatch.setattr(factory, "create_tools", lambda _: [thread_echo])

    class EchoModel(ScriptedModel):
        def _generate(self, messages, **kwargs):
            answer = (
                AIMessage(content=messages[-1].content)
                if isinstance(messages[-1], ToolMessage)
                else AIMessage(content="", tool_calls=[call("thread_echo")])
            )
            return ChatResult(generations=[ChatGeneration(message=answer)])

    graph = build_agent(manager=MagicMock(), llm=EchoModel(responses=[]))

    def run(thread_id):
        return graph.invoke(
            {"messages": [HumanMessage(content="hello")]},
            {"configurable": {"thread_id": thread_id}},
        )["messages"][-1].content

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(run, ["one", "two"])) == ["one", "two"]


def test_schema_and_recursion_limit():
    llm = ScriptedModel(responses=[AIMessage(content="", tool_calls=[call("stop_session")])] * 20)
    manager = MagicMock(spec=SandboxManager)
    graph = build_agent(manager=manager, llm=llm)
    assert graph.config["recursion_limit"] == factory.get_settings().AGENT_MAX_STEPS
    assert graph.config["max_concurrency"] == factory.get_settings().TOOL_MAX_CONCURRENCY
    schema = graph.get_context_jsonschema()
    assert "chat_model_api_key" in schema["properties"]
    with pytest.raises(GraphRecursionError):
        graph.invoke({"messages": [HumanMessage(content="loop")]}, {"recursion_limit": 2})


def test_six_tool_schemas_unchanged():
    tools = create_tools(MagicMock(spec=SandboxManager))
    assert {t.name: set(t.args) for t in tools} == {
        "create_session": {"language", "dependencies"},
        "execute_code": {"session_id", "code", "timeout"},
        "execute_terminal": {"session_id", "command", "timeout"},
        "import_files": {"session_id", "files"},
        "export_files": {"session_id", "files"},
        "stop_session": {"session_id"},
    }


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("vision", [False, "false", True, "true"])
def test_explicit_vision_override(async_mode, vision):
    history = [
        AIMessage(content="", tool_calls=[call("execute_code")]),
        ToolMessage(
            content=json.dumps({"display_outputs": [{"type": "image/png", "data": "YWJj"}]}),
            tool_call_id="call-1",
        ),
    ]
    llm = ScriptedModel(responses=[AIMessage(content="ok")])
    graph = build_agent(manager=MagicMock(), llm=llm)
    invoke(graph, history, async_mode, {"configurable": {"chat_model_supports_vision": vision}})
    assert isinstance(llm.seen[0][-1].content, list) == (vision in (True, "true"))


def test_completed_checkpoint_from_old_agent_node_can_continue():
    saver = InMemorySaver()
    old_graph = StateGraph(AgentState)
    old_graph.add_node("agent", lambda _: {"messages": [AIMessage(content="old answer")]})
    old_graph.set_entry_point("agent")
    old_graph.set_finish_point("agent")
    cfg = {"configurable": {"thread_id": "legacy-thread"}}
    old_graph.compile(checkpointer=saver).invoke(
        {"messages": [HumanMessage(content="old question")]}, cfg
    )
    llm = ScriptedModel(responses=[AIMessage(content="new answer")])
    graph = build_agent(manager=MagicMock(), llm=llm, checkpointer=saver)
    result = graph.invoke({"messages": [HumanMessage(content="continue")]}, cfg)
    assert [m.content for m in result["messages"]] == [
        "old question",
        "old answer",
        "continue",
        "new answer",
    ]


def test_platform_entry_keeps_build_agent_export(monkeypatch):
    import importlib
    import sys

    sentinel = object()
    builder = MagicMock(return_value=sentinel)
    monkeypatch.setattr(factory, "build_agent", builder)
    monkeypatch.setattr("taskharbor.sandbox.get_manager", lambda: "test-manager")
    sys.modules.pop("taskharbor.agent.graph", None)
    try:
        entry = importlib.import_module("taskharbor.agent.graph")
        assert entry.graph is sentinel
        assert entry.build_agent is builder
        builder.assert_called_once_with(manager="test-manager", checkpointer=None)
    finally:
        sys.modules.pop("taskharbor.agent.graph", None)
