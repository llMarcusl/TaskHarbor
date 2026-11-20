"""Compaction contracts with real graph checkpoints and deterministic models."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from taskharbor.agent.compaction import (
    CompactionError,
    Compactor,
    Summarizer,
    compact_command,
    prompt_too_long,
)
from taskharbor.agent.context_middleware import ContextMiddleware, ContextRequestGuard
from taskharbor.agent.memory.service import raw_records
from taskharbor.agent.state import AgentState
from taskharbor.settings import Settings
from tests.test_agent_refactor import ScriptedModel


class Repository:
    def __init__(self):
        self.rows = []
        self.saved = []
        self.session = None
        self.fail = False

    def owner_for_thread(self, thread):
        return "owner"

    def capture(self, owner, thread, records):
        if self.fail:
            raise OSError("database unavailable")
        known = {r["payload"]["id"] for r in self.rows}
        for record in records:
            if record["payload"]["id"] not in known:
                self.rows.append({**record, "seq": len(self.rows) + 1})
                known.add(record["payload"]["id"])
        return len(self.rows)

    def message_sequences(self, owner, thread, ids):
        return {r["payload"]["id"]: r["seq"] for r in self.rows if r["payload"]["id"] in ids}

    def history_page(self, owner, thread, after_seq=0, limit=128, until_seq=None):
        return [r for r in self.rows if after_seq < r["seq"] <= (until_seq or len(self.rows))][
            :limit
        ]

    def session_version(self, owner, thread, until):
        if self.session and self.session["metadata"]["covered_range"]["end_seq"] <= until:
            return self.session
        return None

    def save_compaction(self, owner, thread, body, metadata, identity):
        self.saved.append({"body": body, "metadata": dict(metadata)})
        return {"scope": "compaction:" + thread, "revision": len(self.saved)}


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        CHAT_MODEL_API_KEY="test",
        POSTGRES_USER="test",
        POSTGRES_PASSWORD="test",
        POSTGRES_DB="test",
        POSTGRES_HOST="localhost",
        POSTGRES_PORT="5432",
        CONTEXT_COMPACTION_ENABLED=True,
        CONTEXT_MODEL_WINDOWS={"gpt-4o": 30000},
        CONTEXT_OUTPUT_RESERVE=1000,
        CONTEXT_SAFETY_TOKENS=100,
        CONTEXT_KEEP_MESSAGES=2,
        MEMORY_INJECTION_ENABLED=False,
        MEMORY_MODEL="small",
        MEMORY_MODEL_PROVIDER="openai",
    )


@pytest.fixture
def service():
    return SimpleNamespace(repository=Repository(), start=lambda: None)


def history(size=10000):
    return [
        HumanMessage(id="u1", content="Decisions: preserve SQLite. " + "a" * size),
        AIMessage(id="a1", content="Completed migration; pending indexes. " + "b" * size),
        HumanMessage(id="u2", content="continue"),
        AIMessage(id="a2", content="ready"),
    ]


def test_archive_failure_never_calls_model_or_saves(settings, service):
    service.repository.fail = True
    model = MagicMock()
    original = history()
    with pytest.raises(OSError):
        Compactor(service, settings, model).prepare(
            {"messages": original}, "t", 30000, 0, force=True
        )
    model.invoke.assert_not_called()
    assert original[0].content.endswith("a" * 10000)
    assert not service.repository.saved


def test_lightweight_uses_full_document_not_injection_budget(settings, service):
    body = {"decisions": ["d" * 550 for _ in range(12)], "next_steps": ["pending-index"]}
    service.repository.session = {
        "scope": "session:t",
        "revision": 4,
        "body": body,
        "metadata": {"covered_range": {"end_seq": 2}},
    }
    model = MagicMock()
    candidate, meta = Compactor(service, settings, model).prepare(
        {"messages": history()}, "t", 30000, 0, force=True
    )
    assert len(candidate[0].content) > 6000
    assert "pending-index" in candidate[0].content
    assert meta["scope"] == "session:t"
    assert [m.id for m in candidate[1:]] == ["u2", "a2"]
    model.invoke.assert_not_called()
    assert not service.repository.saved


def test_three_oversized_summaries_do_not_commit(settings, service):
    model = ScriptedModel(responses=[AIMessage(content="x" * 20000) for _ in range(3)])
    with pytest.raises(CompactionError, match="after_3_attempts"):
        Compactor(service, settings, model).prepare(
            {"messages": history(2000)}, "t", 8000, 0, force=True, focus="decisions"
        )
    assert len(model.seen) == 3
    assert all("decisions" in m[0].content for m in model.seen)
    assert not service.repository.saved


def test_repeated_summary_saves_only_success(settings, service):
    model = ScriptedModel(responses=[AIMessage(content="x" * 10000), AIMessage(content="done")])
    candidate, meta = Compactor(service, settings, model).prepare(
        {"messages": history(2000)}, "t", 8000, 0, force=True, focus="decisions"
    )
    assert len(model.seen) == 2
    assert len(service.repository.saved) == 1
    assert meta["summary_attempts"] == 2
    assert "done" in candidate[0].content
    assert len(service.repository.rows) == 4
    Compactor(service, settings).archive("t", candidate)
    assert len(service.repository.rows) == 4


def test_prompt_overflow_has_three_total_retries(settings):
    model = ScriptedModel(responses=[ValueError("prompt-too-long") for _ in range(4)])
    summarizer = Summarizer(settings, model)
    units = [{"start_seq": n, "end_seq": n, "text": str(n)} for n in range(1, 6)]
    with pytest.raises(CompactionError, match="retry_exhausted"):
        summarizer.call(units, "decisions", 1000)
    assert len(model.seen) == 4
    assert summarizer.overflow_retries == 3
    assert [r["start_seq"] for r in summarizer.omitted] == [1, 2, 3]
    assert all("decisions" in m[0].content for m in model.seen)


def test_other_errors_do_not_truncate_or_retry(settings):
    model = ScriptedModel(responses=[ValueError("HTTP 413 upload failed")])
    summarizer = Summarizer(settings, model)
    with pytest.raises(ValueError):
        summarizer.call([{"text": "one"}, {"text": "two"}], "", 1000)
    assert not summarizer.omitted
    assert not prompt_too_long(ValueError("401 authentication failed"))


@pytest.mark.parametrize("async_mode", [False, True])
def test_manual_compact_commits_without_main_model(settings, service, async_mode):
    model = ScriptedModel(responses=[])
    small = ScriptedModel(responses=[AIMessage(content="Decision SQLite; pending indexes")])
    context = ContextMiddleware(settings, [], "system", service, small)
    graph = create_agent(
        model,
        tools=[],
        state_schema=AgentState,
        middleware=[context, ContextRequestGuard(context)],
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "t"}}
    messages = [*history(2000), HumanMessage(id="cmd", content="/compact decisions and pending")]
    if async_mode:
        result = asyncio.run(graph.ainvoke({"messages": messages}, cfg))
    else:
        result = graph.invoke({"messages": messages}, cfg)
    assert not model.seen
    assert result["last_compaction"]["scope"] == "compaction:t"
    assert result["messages"][-1].additional_kwargs["context_notice"]
    assert len(service.repository.rows) == 5
    checkpoint = graph.get_state(cfg).values
    assert checkpoint["context_version"] == 1
    assert checkpoint["messages"][0].additional_kwargs["memory_internal"]


@pytest.mark.parametrize("async_mode", [False, True])
def test_provider_overflow_routes_through_persisted_compaction(settings, service, async_mode):
    model = ScriptedModel(
        responses=[ValueError("context_length_exceeded"), AIMessage(content="continued")]
    )
    small = ScriptedModel(responses=[AIMessage(content="Decision SQLite; pending indexes")])
    context = ContextMiddleware(settings, [], "system", service, small)
    graph = create_agent(
        model,
        tools=[],
        state_schema=AgentState,
        middleware=[context, ContextRequestGuard(context)],
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "t"}}
    messages = history(2000)
    result = (
        asyncio.run(graph.ainvoke({"messages": messages}, cfg))
        if async_mode
        else graph.invoke({"messages": messages}, cfg)
    )
    assert result["messages"][-1].content == "continued"
    assert len(model.seen) == 2
    assert result["context_recovery_used"]
    assert graph.get_state(cfg).values["last_compaction"]["scope"] == "compaction:t"
    assert all(
        not r["payload"]["additional_kwargs"].get("memory_internal")
        for r in service.repository.rows
    )


def test_tool_result_archived_before_trimming(settings, service):
    original = "a" * 30000 + "critical error"
    messages = [
        HumanMessage(id="u", content="run"),
        AIMessage(id="call", content="", tool_calls=[{"id": "c", "name": "execute", "args": {}}]),
        ToolMessage(id="tool", content=original, tool_call_id="c"),
    ]
    candidate, meta = Compactor(service, settings).prepare({"messages": messages}, "t", 100000, 0)
    assert candidate[-1].content != original
    assert candidate[-1].id == "tool"
    assert "critical error" in candidate[-1].content
    assert service.repository.rows[-1]["payload"]["content"] == original
    Compactor(service, settings).archive("t", candidate)
    assert len(service.repository.rows) == 3
    assert service.repository.rows[-1]["payload"]["content"] == original


def test_command_and_internal_filter():
    assert compact_command([HumanMessage(content="/compact 决策和待办")]) == "决策和待办"
    assert compact_command([HumanMessage(content="/compactor")]) is None
    assert (
        raw_records([AIMessage(content="summary", additional_kwargs={"memory_internal": True})])
        == []
    )


def test_failed_model_after_recovery_keeps_compacted_checkpoint(settings, service):
    model = ScriptedModel(responses=[ValueError("prompt-too-long"), ValueError("auth failure")])
    small = ScriptedModel(responses=[AIMessage(content="decision preserved")])
    context = ContextMiddleware(settings, [], "system", service, small)
    graph = create_agent(
        model, tools=[], state_schema=AgentState, middleware=[context], checkpointer=InMemorySaver()
    )
    cfg = {"configurable": {"thread_id": "t"}}
    with pytest.raises(ValueError, match="auth failure"):
        graph.invoke({"messages": history(2000)}, cfg)
    snapshot = graph.get_state(cfg).values
    assert snapshot["last_compaction"]["scope"] == "compaction:t"
    assert snapshot["messages"][0].additional_kwargs["context_summary"]
    assert len(model.seen) == 2


def test_recovery_does_not_repeat_completed_tool(settings, service):
    from langchain_core.tools import tool

    calls = []

    @tool
    def perform() -> str:
        """Perform the operation."""
        calls.append(1)
        return "created output.txt"

    model = ScriptedModel(
        responses=[
            AIMessage(content="", tool_calls=[{"id": "operation", "name": "perform", "args": {}}]),
            ValueError("prompt-too-long"),
            AIMessage(content="done"),
        ]
    )
    small = ScriptedModel(responses=[AIMessage(content="decision preserved")])
    context = ContextMiddleware(settings, [perform], "system", service, small)
    graph = create_agent(
        model,
        tools=[perform],
        state_schema=AgentState,
        middleware=[context],
        checkpointer=InMemorySaver(),
    )
    result = graph.invoke({"messages": history(2000)}, {"configurable": {"thread_id": "t"}})
    assert calls == [1]
    assert result["messages"][-1].content == "done"


def test_history_endpoint_ownership_and_paging(settings, service, monkeypatch):
    from unittest.mock import AsyncMock

    from starlette.testclient import TestClient

    from taskharbor.http_app import app

    service.close = lambda: None
    Compactor(service, settings).archive("t", history(10))
    monkeypatch.setattr("taskharbor.settings.get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.agent.memory.service.get_memory_service", lambda: service)
    auth = AsyncMock(return_value=SimpleNamespace(identity="other"))
    monkeypatch.setattr("aegra_api.core.auth_deps.require_auth", auth)
    with TestClient(app) as client:
        assert client.get("/threads/t/history").status_code == 404
        auth.return_value = SimpleNamespace(identity="owner")
        first = client.get("/threads/t/history?limit=2").json()
        assert first["next_seq"] == 2
        assert [m["id"] for m in first["messages"]] == ["u1", "a1"]
        last = client.get("/threads/t/history?after_seq=2").json()
        assert last["next_seq"] is None
        assert [m["id"] for m in last["messages"]] == ["u2", "a2"]
        assert client.get("/threads/t/history?limit=10000").status_code == 422


def test_history_read_tool_cannot_select_other_thread(service, settings):
    from taskharbor.agent.history_tool import create_history_tool

    Compactor(service, settings).archive("t", history(10))
    tool = create_history_tool(service)
    assert "thread_id" not in tool.args
    assert "owner" not in tool.args


def test_interrupted_old_tool_can_be_compacted_without_fabricating_result(settings, service):
    old = [
        HumanMessage(id="u1", content="create file"),
        AIMessage(
            id="call",
            content="",
            tool_calls=[{"id": "interrupted", "name": "write_file", "args": {}}],
        ),
        HumanMessage(id="u2", content="continue after cancel"),
        AIMessage(id="a2", content="check whether the file exists"),
    ]
    small = ScriptedModel(
        responses=[AIMessage(content="File write interrupted; verify before retry")]
    )
    candidate, meta = Compactor(service, settings, small).prepare(
        {"messages": old}, "t", 10000, 0, force=True, focus="decisions"
    )
    assert "missing_tool_results" in small.seen[0][1].content
    assert "interrupted" in small.seen[0][1].content
    assert all(r["payload"]["type"] != "tool" for r in service.repository.rows)
    assert candidate[-2:] == old[-2:]
    assert meta["covered_until_seq"] == 2


def long_task():
    messages = [HumanMessage(id="task", content="Keep decisions and finish all indexes")]
    for n in range(5):
        messages.extend(
            [
                AIMessage(
                    id=f"call-{n}",
                    content="",
                    tool_calls=[{"id": f"c-{n}", "name": "perform", "args": {}}],
                ),
                ToolMessage(id=f"result-{n}", content=f"completed step {n}", tool_call_id=f"c-{n}"),
            ]
        )
    return messages


@pytest.mark.parametrize("lightweight", [False, True])
def test_long_single_user_task_compacts_early_tools_and_preserves_input(
    settings,
    service,
    lightweight,
):
    messages = long_task()
    if lightweight:
        service.repository.session = {
            "scope": "session:t",
            "revision": 2,
            "body": {"completed": ["steps 0..2 done"]},
            "metadata": {"covered_range": {"end_seq": 7}},
        }
    small = ScriptedModel(responses=[AIMessage(content="steps 0..3 done; finish step 4")])
    candidate, meta = Compactor(service, settings, small).prepare(
        {"messages": messages}, "t", 10000, 0, force=True
    )
    assert next(m for m in candidate if isinstance(m, HumanMessage)).id == "task"
    assert len(candidate) < len(messages)
    assert meta["covered_until_seq"] >= 7
    assert candidate[-1].id == "result-4"
    if lightweight:
        assert meta["scope"] == "session:t"
        assert not small.seen
    else:
        assert meta["scope"] == "compaction:t"


def test_result_after_resume_human_stays_with_issuing_call(settings, service):
    from taskharbor.agent.tool_recovery import repair_tool_messages

    messages = [
        HumanMessage(id="old", content="old goal"),
        AIMessage(id="call", content="", tool_calls=[{"id": "c", "name": "perform", "args": {}}]),
        HumanMessage(id="resume", content="continue"),
        ToolMessage(id="actual", content="file successfully created", tool_call_id="c"),
    ]
    small = ScriptedModel(responses=[AIMessage(content="old goal retained")])
    candidate, meta = Compactor(service, settings, small).prepare(
        {"messages": messages}, "t", 10000, 0, force=True, focus="goal"
    )
    repaired = repair_tool_messages(candidate)
    assert any(m.id == "actual" for m in repaired)
    assert any(m.id == "call" for m in repaired)
    assert meta["covered_until_seq"] == 1


def test_compacted_candidate_does_not_reserve_duplicate_session_injection(settings, service):
    small = ScriptedModel(responses=[AIMessage(content="decision retained")])
    candidate, meta = Compactor(service, settings, small).prepare(
        {"messages": history(2000)},
        "t",
        10000,
        7000,
        compacted_overhead=100,
        force=True,
        focus="decision",
    )
    assert meta["after_estimated_tokens"] < 5000
    assert candidate[0].additional_kwargs["context_summary"]
