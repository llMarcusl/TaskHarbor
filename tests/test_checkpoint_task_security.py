"""Independent checkpoint approvals and restored security policy checks."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aegra_api.core.auth_ctx import with_auth_ctx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from taskharbor.agent.security_middleware import ToolSecurityMiddleware
from taskharbor.security.repository import InMemoryApprovalStore
from tests.test_security_graph import make_graph


def test_client_flags_do_not_bypass_pending_approval_admission():
    store = InMemoryApprovalStore()
    store.prepare(
        {
            "invocation_key": "saved",
            "owner": "alice",
            "thread_id": "parent",
            "tool_call_id": "call",
            "tool_name": "custom_action",
            "group": "integration",
            "executor": "connector",
            "arguments": {},
            "request_digest": "digest",
        }
    )
    manager = MagicMock()
    manager.resolve_owner.return_value = "alice"
    middleware = ToolSecurityMiddleware(
        manager,
        SimpleNamespace(HOST_COMMANDS_ENABLED=False, TOOL_SECURITY_RULES=[]),
        [],
        store,
    )
    runtime = SimpleNamespace(
        config={
            "configurable": {
                "thread_id": "parent",
                "task_id": "forged",
                "is_subagent": True,
            }
        }
    )
    with pytest.raises(RuntimeError, match="waiting for a tool decision"):
        middleware.before_agent({}, runtime)


def test_authenticated_run_cannot_target_another_users_conversation():
    async def scenario():
        manager = MagicMock()
        manager.resolve_owner.return_value = "alice"
        middleware = ToolSecurityMiddleware(
            manager,
            SimpleNamespace(HOST_COMMANDS_ENABLED=False, TOOL_SECURITY_RULES=[]),
            [],
        )
        runtime = SimpleNamespace(config={"configurable": {"thread_id": "alice-thread"}})
        async with with_auth_ctx(SimpleNamespace(identity="bob")):
            with pytest.raises(PermissionError, match="does not own"):
                middleware.before_agent({}, runtime)

    asyncio.run(scenario())


def test_worker_and_root_approvals_resume_independently_after_graph_reconstruction():
    async def scenario():
        attempts = []

        @tool
        async def custom_action() -> str:
            """Record an approved operation in its own execution graph."""
            attempts.append("executed")
            return "ok"

        saver = InMemorySaver()
        store = InMemoryApprovalStore()
        root, _, _ = make_graph([custom_action], store=store, checkpointer=saver)
        worker, _, _ = make_graph([custom_action], store=store, checkpointer=saver, worker=True)
        parent_config = {"configurable": {"thread_id": "parent", "run_id": "parent-run"}}
        child_config = {"configurable": {"thread_id": "child", "run_id": "child-run"}}
        parent = await root.ainvoke({"messages": [HumanMessage(content="parent")]}, parent_config)
        child = await worker.ainvoke({"messages": [HumanMessage(content="child")]}, child_config)
        assert parent.get("__interrupt__") and child.get("__interrupt__")
        parent_record = store.list_thread("alice", "parent")[0]
        child_record = store.list_thread("alice", "child")[0]
        assert parent_record["task_id"] is None
        assert child_record["task_id"] == "child"
        assert parent_record["transport"] == child_record["transport"] == "checkpoint"
        store.decide(child_record["id"], "alice", "child", "allow")
        rebuilt, _, _ = make_graph(
            [custom_action],
            store=store,
            checkpointer=saver,
            worker=True,
            responses=[AIMessage(content="worker done")],
        )
        child_result = await rebuilt.ainvoke(
            Command(resume={child["__interrupt__"][0].id: {"approval_id": child_record["id"]}}),
            child_config,
        )
        assert child_result["messages"][-1].content == "worker done"
        assert attempts == ["executed"]
        assert store.get(parent_record["id"], "alice", "parent")["status"] == "pending"
        store.decide(parent_record["id"], "alice", "parent", "deny")
        result = await root.ainvoke(
            Command(resume={parent["__interrupt__"][0].id: {"approval_id": parent_record["id"]}}),
            parent_config,
        )
        assert not result.get("__interrupt__")
        assert attempts == ["executed"]

    asyncio.run(scenario())


def test_checkpoint_resume_rechecks_policy_before_claiming_approval():
    async def scenario():
        attempts = []

        @tool
        async def custom_action() -> str:
            """An operation whose policy changes while awaiting approval."""
            attempts.append("executed")
            return "ok"

        saver = InMemorySaver()
        store = InMemoryApprovalStore()
        graph, _, _ = make_graph([custom_action], store=store, checkpointer=saver)
        config = {"configurable": {"thread_id": "t", "run_id": "run"}}
        result = await graph.ainvoke({"messages": [HumanMessage(content="do it")]}, config)
        row = store.list_thread("alice", "t")[0]
        store.decide(row["id"], "alice", "t", "allow")
        rebuilt, _, _ = make_graph(
            [custom_action],
            store=store,
            checkpointer=saver,
            rules=[{"effect": "deny", "tool": "custom_action"}],
            responses=[AIMessage(content="blocked")],
        )
        result = await rebuilt.ainvoke(
            Command(resume={result["__interrupt__"][0].id: {"approval_id": row["id"]}}),
            config,
        )
        assert not attempts
        message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        assert json.loads(message.content)["error"] == "TOOL_DENIED"
        assert store.get(row["id"], "alice", "t")["status"] == "superseded"

    asyncio.run(scenario())
