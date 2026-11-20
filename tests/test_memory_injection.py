"""Request-only memory injection through real sync/async graph execution."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from taskharbor.agent import factory
from taskharbor.agent.memory.injection import MemoryInjectionMiddleware, render_memory
from taskharbor.agent.memory.repository import MemoryRepository
from taskharbor.agent.prompts import SYSTEM_PROMPT
from taskharbor.settings import Settings
from tests.test_agent_refactor import ScriptedModel


@pytest.fixture
def settings(monkeypatch):
    result = Settings(
        _env_file=None,
        SUBAGENTS_ENABLED=False,
        CHAT_MODEL_API_KEY="test",
        POSTGRES_USER="test",
        POSTGRES_PASSWORD="test",
        POSTGRES_DB="test",
        POSTGRES_HOST="localhost",
        POSTGRES_PORT="5432",
        SESSION_MEMORY_ENABLED=True,
        LONG_TERM_MEMORY_ENABLED=True,
        MEMORY_MODEL="test",
        MEMORY_MODEL_PROVIDER="openai",
    )
    monkeypatch.setattr(factory, "get_settings", lambda: result)
    monkeypatch.setattr("taskharbor.agent.configuration.get_settings", lambda: result)
    monkeypatch.setattr("taskharbor.crypto.resolve_api_key", lambda: "test")
    return result


def documents(preference="偏好中文"):
    return {
        "session": {"revision": 3, "body": {"goals": ["完成串匹配分析"]}},
        "long_term": {
            "revision": 7,
            "body": {
                "facts": [
                    {
                        "category": "preference",
                        "content": preference,
                        "source_ids": ["private-evidence-id"],
                    }
                ],
            },
        },
    }


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("history_system", [False, True])
def test_injection_refreshes_without_persisting_or_archiving(settings, async_mode, history_system):
    service = MagicMock()
    service.repository.read_for_thread.side_effect = [documents(), documents("偏好简洁")]
    model = ScriptedModel(responses=[AIMessage(content="first"), AIMessage(content="second")])
    graph = factory.build_agent(
        manager=MagicMock(), llm=model, memory_service=service, checkpointer=InMemorySaver()
    )
    cfg = {"configurable": {"thread_id": "t", "user_id": "forged-owner"}}
    initial = [HumanMessage(content="分析文档")]
    if history_system:
        initial.insert(0, SystemMessage(content="custom system"))

    async def execute():
        await graph.ainvoke({"messages": initial}, cfg)
        return await graph.ainvoke({"messages": [HumanMessage(content="继续")]}, cfg)

    if async_mode:
        result = asyncio.run(execute())
    else:
        graph.invoke({"messages": initial}, cfg)
        result = graph.invoke({"messages": [HumanMessage(content="继续")]}, cfg)
    for index, seen in enumerate(model.seen):
        assert isinstance(seen[0], SystemMessage)
        assert sum(isinstance(m, SystemMessage) for m in seen) == 1
        text = seen[0].content
        assert text.startswith("custom system" if history_system else SYSTEM_PROMPT)
        assert "完成串匹配分析" in text
        assert ("偏好中文" if index == 0 else "偏好简洁") in text
        assert text.count("Historical memory (") == 1
        assert "private-evidence-id" not in text
    assert "偏好中文" not in model.seen[1][0].content
    for messages in [result["messages"], graph.get_state(cfg).values["messages"]]:
        assert all("Historical memory (" not in str(m.content) for m in messages)
    for invocation in service.observe.call_args_list:
        assert all("Historical memory (" not in str(m.content) for m in invocation.args[1])
    assert all(c.args == ("t",) for c in service.repository.read_for_thread.call_args_list)


@pytest.mark.parametrize("mode", ["disabled", "empty", "failure", "no_thread", "zero_budget"])
def test_missing_memory_does_not_break_model_call(settings, mode):
    service = MagicMock()
    service.repository.read_for_thread.return_value = {}
    if mode == "disabled":
        settings.MEMORY_INJECTION_ENABLED = False
    if mode == "zero_budget":
        settings.MEMORY_INJECTION_MAX_CHARS = 0
    if mode == "failure":
        service.repository.read_for_thread.side_effect = RuntimeError("database unavailable")
    model = ScriptedModel(responses=[AIMessage(content="answer")])
    graph = factory.build_agent(manager=MagicMock(), llm=model, memory_service=service)
    config = {} if mode == "no_thread" else {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [HumanMessage(content="hello")]}, config)
    assert model.seen[0][0].content == SYSTEM_PROMPT
    if mode in ("disabled", "no_thread", "zero_budget"):
        service.repository.read_for_thread.assert_not_called()


def test_budgets_skip_whole_entries_and_keep_shorter_ones(settings):
    settings.SESSION_MEMORY_INJECTION_MAX_CHARS = 150
    settings.LONG_TERM_MEMORY_INJECTION_MAX_CHARS = 180
    doc = documents()
    doc["session"]["body"]["goals"] = ["oversized" * 1000, "short goal"]
    text = render_memory(doc, settings)
    assert "oversized" not in text and "short goal" in text
    start = text.index("\n[User long-term memory")
    split = text.index("\n[Current session memory")
    assert len(text[start:split]) <= 180
    assert len(text[split:]) <= 150
    assert len(text) <= settings.MEMORY_INJECTION_MAX_CHARS
    for budget in (0, 100, 450, 500, 600):
        settings.MEMORY_INJECTION_MAX_CHARS = budget
        assert len(render_memory(doc, settings)) <= budget


def test_independent_flags_and_priority(settings):
    doc = documents()
    doc["long_term"]["body"]["facts"].append({"category": "correction", "content": "更正"})
    text = render_memory(doc, settings)
    assert text.index("更正") < text.index("偏好中文")
    settings.LONG_TERM_MEMORY_ENABLED = False
    assert "偏好中文" not in render_memory(doc, settings)
    settings.LONG_TERM_MEMORY_ENABLED = True
    settings.SESSION_MEMORY_ENABLED = False
    assert "完成串匹配分析" not in render_memory(doc, settings)


def test_structured_system_content_is_preserved():
    original = SystemMessage(content=[{"type": "text", "text": "original"}])
    request = MagicMock(system_message=original)
    MemoryInjectionMiddleware._inject(request, "context")
    enriched = request.override.call_args.kwargs["system_message"]
    assert enriched.content == [*original.content, {"type": "text", "text": "context"}]
    assert original.content == [{"type": "text", "text": "original"}]


def test_repository_reads_only_server_owned_scopes():
    conn = MagicMock()
    owner_by_thread = {"a": "alice", "b": "bob", "a2": "alice"}
    reads = []

    def execute(sql, params):
        if 'FROM "thread"' in sql:
            owner = owner_by_thread.get(params[0])
            row = {"user_id": owner} if owner else None
        else:
            reads.append(params)
            row = {"revision": 1, "body": {"owner": params[0], "scope": params[1]}}
        return SimpleNamespace(fetchone=lambda: row)

    conn.execute.side_effect = execute
    pool = MagicMock()
    pool.connection.return_value.__enter__.return_value = conn
    repo = MemoryRepository(pool)
    for thread, owner in owner_by_thread.items():
        result = repo.read_for_thread(thread)
        assert result["session"]["body"] == {"owner": owner, "scope": "session:" + thread}
        assert result["long_term"]["body"] == {"owner": owner, "scope": "user"}
    count = len(reads)
    assert repo.read_for_thread("deleted") == {}
    assert len(reads) == count
    assert set(repo.read_for_thread("a", session=False)) == {"long_term"}
    assert set(repo.read_for_thread("a", long_term=False)) == {"session"}
