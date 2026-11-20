"""Opt-in checkpointed worker and cross-thread file checks with real Docker."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from docker.errors import NotFound
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from taskharbor.agent.subagents import factory as worker_factory
from taskharbor.agent.subagents import files as file_tools
from taskharbor.agent.subagents.factory import build_worker
from taskharbor.agent.subagents.files import TaskFile
from taskharbor.sandbox.manager import current_thread_id
from taskharbor.security.repository import InMemoryApprovalStore
from taskharbor.settings import Settings
from taskharbor.tools import create_tools
from tests.test_agent_refactor import ScriptedModel, call
from tests.test_async_task_files import thread
from tests.test_sandbox_acceptance import real_sandbox as _real_sandbox

pytestmark = pytest.mark.skipif(
    os.environ.get("SANDBOX_DOCKER_TESTS") != "1",
    reason="opt-in local Docker acceptance",
)


@pytest.fixture
def real_sandbox(tmp_path, monkeypatch):
    yield from _real_sandbox.__wrapped__(tmp_path, monkeypatch)


def test_worker_checkpoint_resume_and_persistent_parent_file_delivery(real_sandbox, monkeypatch):
    async def scenario():
        from taskharbor.sandbox import manager as manager_module

        manager = real_sandbox
        settings = Settings(
            _env_file=None,
            **{
                **vars(manager_module.get_settings()),
                "SANDBOX_STANDALONE_OWNER": "acceptance-user",
                "CHAT_MODEL_API_KEY": "test-key",
                "SUBAGENTS_ENABLED": True,
                "SESSION_MEMORY_ENABLED": False,
                "LONG_TERM_MEMORY_ENABLED": False,
                "CONTEXT_COMPACTION_ENABLED": False,
                "HOST_COMMANDS_ENABLED": False,
                "TOOL_SECURITY_RULES": [],
                "ENCRYPTION_KEY": "docker-file-grant-key",
                "POSTGRES_USER": "test",
                "POSTGRES_PASSWORD": "test",
                "POSTGRES_DB": "test",
                "POSTGRES_HOST": "localhost",
                "POSTGRES_PORT": "5432",
            },
        )
        monkeypatch.setattr(manager_module, "get_settings", lambda: settings)
        monkeypatch.setattr("taskharbor.agent.configuration.get_settings", lambda: settings)
        monkeypatch.setattr("taskharbor.crypto.resolve_api_key", lambda: "test-key")
        monkeypatch.setattr(manager, "resolve_owner", lambda _: "acceptance-user")

        async def parent(owner, child):
            return "acceptance-thread" if child == "worker-thread" else None

        monkeypatch.setattr(file_tools, "worker_parent", parent)
        prepare, collect = file_tools.create_parent_file_tools(manager, settings)
        (importing,) = file_tools.create_worker_file_tools(manager, settings)
        root = await asyncio.to_thread(manager.create_session)
        assert (
            await asyncio.to_thread(
                manager.execute_code,
                root.session_id,
                "root_secret = 42\nfrom pathlib import Path\n"
                "Path('/workspace/input.txt').write_text('source')",
            )
        ).success
        runtime = SimpleNamespace(config={"configurable": {"thread_id": "acceptance-thread"}})
        grant = json.loads(
            await asyncio.to_thread(
                prepare.func,
                [TaskFile(path="input.txt")],
                runtime,
            )
        )
        with thread("worker-thread") as child_runtime:
            await importing.coroutine(grant["manifest"], child_runtime)
            child = await asyncio.to_thread(manager.create_session)
        assert root.container_id != child.container_id
        assert root.workspace != child.workspace
        assert Path(child.workspace, "input.txt").read_text() == "source"

        def confirmed_tools(manager):
            tools = create_tools(manager)
            for item in tools:
                if item.name == "execute_code":
                    item.metadata = {"requires_confirmation": True}
            return tools

        monkeypatch.setattr(worker_factory, "create_tools", confirmed_tools)
        store = InMemoryApprovalStore()
        saver = InMemorySaver()
        script = (
            "from pathlib import Path\n"
            "assert 'root_secret' not in globals()\n"
            "text = Path('/workspace/input.txt').read_text()\n"
            "Path('/workspace/output.txt').write_text(text + '-verified')"
        )
        model = ScriptedModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        call(
                            "execute_code",
                            {
                                "session_id": child.session_id,
                                "code": script,
                            },
                        )
                    ],
                ),
                AIMessage(
                    content='{"summary":"verified","files":[{"runtime":"python","path":"/workspace/output.txt"}]}'
                ),
            ]
        )
        graph = build_worker(
            manager=manager, llm=model, settings=settings, security_store=store, checkpointer=saver
        )
        config = {"configurable": {"thread_id": "worker-thread", "run_id": "worker-first"}}
        paused = await graph.ainvoke({"messages": [HumanMessage(content="verify input")]}, config)
        assert paused.get("__interrupt__"), [message.content for message in paused["messages"]]
        assert not Path(child.workspace, "output.txt").exists()
        (record,) = store.list_thread("acceptance-user", "worker-thread")
        assert record["transport"] == "checkpoint" and record["task_id"] == "worker-thread"
        # The parent kernel stays available while the child's graph is paused.
        assert (
            "42"
            in (
                await asyncio.to_thread(manager.execute_code, root.session_id, "print(root_secret)")
            ).stdout
        )
        store.decide(record["id"], "acceptance-user", "worker-thread", "allow")
        # Rebuilding the worker retains checkpoint position and approval identity.
        graph = build_worker(
            manager=manager, llm=model, settings=settings, security_store=store, checkpointer=saver
        )
        result = await graph.ainvoke(
            Command(
                resume={
                    paused["__interrupt__"][0].id: {
                        "approval_id": record["id"],
                    }
                }
            ),
            config,
        )
        assert json.loads(result["messages"][-1].content)["summary"] == "verified"
        with thread("worker-thread"):
            await asyncio.to_thread(
                manager.cleanup_thread_sessions,
                "worker-thread",
                user_id="acceptance-user",
                delete_files=False,
            )
        assert child.session_id not in manager.sessions
        delivered = json.loads(
            await collect.coroutine(
                "worker-thread",
                [TaskFile(path="output.txt")],
                runtime,
            )
        )
        assert delivered["success"]
        assert Path(root.workspace, "output.txt").read_text() == "source-verified"
        exported = await asyncio.to_thread(
            manager.export_files, root.session_id, [{"source": "output.txt"}]
        )
        assert exported.success and exported.files[0].artifact_id
        assert current_thread_id.get() == "acceptance-thread"

    asyncio.run(scenario())


def test_cancelling_running_worker_retires_its_container_and_preserves_parent(
    real_sandbox, monkeypatch
):
    async def scenario():
        from taskharbor.sandbox import manager as manager_module

        manager = real_sandbox
        settings = Settings(
            _env_file=None,
            **{
                **vars(manager_module.get_settings()),
                "SANDBOX_STANDALONE_OWNER": "acceptance-user",
                "CHAT_MODEL_API_KEY": "test-key",
                "SUBAGENTS_ENABLED": True,
                "SESSION_MEMORY_ENABLED": False,
                "LONG_TERM_MEMORY_ENABLED": False,
                "CONTEXT_COMPACTION_ENABLED": False,
                "HOST_COMMANDS_ENABLED": False,
                "TOOL_SECURITY_RULES": [],
                "POSTGRES_USER": "test",
                "POSTGRES_PASSWORD": "test",
                "POSTGRES_DB": "test",
                "POSTGRES_HOST": "localhost",
                "POSTGRES_PORT": "5432",
            },
        )
        monkeypatch.setattr(manager_module, "get_settings", lambda: settings)
        monkeypatch.setattr("taskharbor.agent.configuration.get_settings", lambda: settings)
        monkeypatch.setattr("taskharbor.crypto.resolve_api_key", lambda: "test-key")
        monkeypatch.setattr(manager, "resolve_owner", lambda _: "acceptance-user")
        root = await asyncio.to_thread(manager.create_session)
        assert (
            await asyncio.to_thread(manager.execute_code, root.session_id, "root_value = 42")
        ).success
        with thread("cancel-worker"):
            child = await asyncio.to_thread(manager.create_session)
        model = ScriptedModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        call(
                            "execute_terminal",
                            {
                                "session_id": child.session_id,
                                "command": "touch /workspace/running; sleep 60",
                            },
                        )
                    ],
                )
            ]
        )
        store = InMemoryApprovalStore()
        graph = build_worker(
            manager=manager,
            llm=model,
            settings=settings,
            security_store=store,
            checkpointer=InMemorySaver(),
        )
        invocation = asyncio.create_task(
            graph.ainvoke(
                {"messages": [HumanMessage(content="run the bounded operation")]},
                {"configurable": {"thread_id": "cancel-worker", "run_id": "cancel-worker-run"}},
            )
        )
        try:
            async with asyncio.timeout(15):
                while not Path(child.workspace, "running").exists():
                    if invocation.done():
                        pytest.fail(
                            f"Worker stopped before execution started: {invocation.result()}"
                        )
                    await asyncio.sleep(0.02)
            invocation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(invocation, 15)
        finally:
            if not invocation.done():
                invocation.cancel()
            await asyncio.gather(invocation, return_exceptions=True)
        assert child.session_id not in manager.sessions
        with pytest.raises(NotFound):
            manager.client.containers.get(child.container_id)
        # The service closes the durable operation ledger after native run
        # cancellation has drained the graph and retired its compute resources.
        from taskharbor.agent.subagents.protocol import _cancel_worker_approvals

        monkeypatch.setattr("taskharbor.security.repository.get_approval_store", lambda: store)
        await asyncio.to_thread(_cancel_worker_approvals, "acceptance-user", "cancel-worker")
        assert (
            "42"
            in (
                await asyncio.to_thread(
                    manager.execute_code,
                    root.session_id,
                    "print(root_value)",
                )
            ).stdout
        )
        assert not store.blocked_thread("acceptance-user", "cancel-worker")

    asyncio.run(scenario())
