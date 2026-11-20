"""Check the assembled async graphs against Deep Agents' actual middleware."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from taskharbor.agent import factory
from taskharbor.agent.factory import build_agent
from taskharbor.agent.subagents import factory as worker_factory
from taskharbor.agent.subagents.factory import build_worker
from taskharbor.security.repository import InMemoryApprovalStore
from taskharbor.settings import Settings
from tests.test_agent_refactor import ScriptedModel, call


class ObservedModel(ScriptedModel):
    observer: Any = None

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        response = self.responses.pop(0)
        if self.observer is not None:
            await self.observer(response)
        return ChatResult(generations=[ChatGeneration(message=response.model_copy(deep=True))])


class GraphProtocol:
    """Small SDK stand-in whose background runs execute real checkpointed graphs.

    The service's HTTP/auth adapter has separate tests. This fixture keeps the
    graph tests focused on upstream tools, independent runs and graph recovery.
    """

    def __init__(self, make_worker):
        self.make_worker = make_worker
        self.records = {}
        self.runs_by_thread = {}
        self.background = {}
        self.launches = 0
        self.client = SimpleNamespace(
            threads=SimpleNamespace(create=self.create_thread, get=self.get_thread),
            runs=SimpleNamespace(create=self.create_run, get=self.get_run, list=self.list_runs),
        )

    async def create_thread(self, *, metadata=None, **kwargs):
        self.launches += 1
        thread_id = f"child-{self.launches}"
        self.records[thread_id] = {"thread_id": thread_id, "metadata": metadata or {}}
        self.runs_by_thread[thread_id] = []
        return self.records[thread_id]

    async def create_run(self, *, thread_id, assistant_id, input=None, command=None, **kwargs):
        assert assistant_id == "taskharbor-worker"
        record = self.records[thread_id]
        if "graph" not in record:
            record["graph"] = self.make_worker(thread_id, input)
        run_id = f"{thread_id}-run-{len(self.runs_by_thread[thread_id]) + 1}"
        run = {"run_id": run_id, "thread_id": thread_id, "status": "running"}
        self.runs_by_thread[thread_id].append(run)
        record["status"] = "busy"

        async def execute():
            payload = Command(resume=command["resume"]) if command else input
            result = await record["graph"].ainvoke(
                payload,
                {
                    "configurable": {"thread_id": thread_id, "run_id": run_id},
                },
            )
            record["status"] = "interrupted" if result.get("__interrupt__") else "idle"
            run["checkpoint_paused"] = bool(result.get("__interrupt__"))
            # The protocol adapter normalizes Aegra's paused run to this status.
            run["status"] = "success"
            return result

        self.background[run_id] = asyncio.create_task(execute())
        return dict(run)

    async def get_run(self, *, thread_id, run_id):
        runs = self.runs_by_thread[thread_id]
        result = dict(next(run for run in runs if run["run_id"] == run_id))
        # A resumed Aegra pause has a successor. The adapter keeps this older
        # run interrupted so upstream follows the thread's newest run.
        if result.get("checkpoint_paused") and run_id != runs[-1]["run_id"]:
            result["status"] = "interrupted"
        return result

    async def list_runs(self, *, thread_id, limit=10):
        return [dict(run) for run in reversed(self.runs_by_thread[thread_id][-limit:])]

    async def get_thread(self, *, thread_id):
        record = self.records[thread_id]
        state = await record["graph"].aget_state({"configurable": {"thread_id": thread_id}})
        interrupts = {
            task.id: [{"id": item.id, "value": item.value} for item in task.interrupts]
            for task in state.tasks
            if task.interrupts
        }
        return {
            "thread_id": thread_id,
            "status": record["status"],
            "values": state.values,
            "interrupts": interrupts,
        }

    async def settle(self, thread_id):
        run_id = self.runs_by_thread[thread_id][-1]["run_id"]
        return await asyncio.wait_for(self.background[run_id], 5)

    async def close(self):
        for task in self.background.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.background.values(), return_exceptions=True)


@pytest.fixture
def dependencies(monkeypatch, tmp_path):
    settings = Settings(
        _env_file=None,
        SUBAGENTS_ENABLED=True,
        CHAT_MODEL_API_KEY="test-key",
        SESSION_MEMORY_ENABLED=False,
        LONG_TERM_MEMORY_ENABLED=False,
        CONTEXT_COMPACTION_ENABLED=False,
        HOST_COMMANDS_ENABLED=False,
        TOOL_SECURITY_RULES=[{"effect": "allow", "tool": "root_work"}],
        STORAGE_DIR=str(tmp_path),
        ENCRYPTION_KEY="test-subagent-graph-key",
        POSTGRES_USER="test",
        POSTGRES_PASSWORD="test",
        POSTGRES_DB="test",
        POSTGRES_HOST="localhost",
        POSTGRES_PORT="5432",
    )
    monkeypatch.setattr(factory, "get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.agent.configuration.get_settings", lambda: settings)
    monkeypatch.setattr("taskharbor.crypto.resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(
        "taskharbor.agent.subagents.protocol.async_subagent_spec",
        lambda *_: {
            "name": "taskharbor-worker",
            "description": "Bounded worker",
            "graph_id": "taskharbor-worker",
            "url": "http://graph-protocol.test",
            "notify_parent": False,
        },
    )
    manager = MagicMock()
    manager.resolve_owner.return_value = "alice"
    manager.sessions = {}
    return manager, InMemoryApprovalStore(), settings


def parent_config(run_id="parent-run"):
    return {"configurable": {"thread_id": "parent", "run_id": run_id}}


def test_sync_parent_without_delegation_preserves_normal_entry_point(dependencies, monkeypatch):
    manager, store, _settings = dependencies
    monkeypatch.setattr(factory, "create_tools", lambda _: [])
    graph = build_agent(
        manager=manager,
        llm=ScriptedModel(responses=[AIMessage(content="completed directly")]),
        security_store=store,
        checkpointer=InMemorySaver(),
    )
    result = graph.invoke({"messages": [HumanMessage(content="answer directly")]}, parent_config())
    assert result["messages"][-1].content == "completed directly"


def start(task_id="start", description="Check data"):
    return call(
        "start_async_task",
        {"description": description, "subagent_type": "taskharbor-worker"},
        id=task_id,
    )


def resume_interrupt(result, kind=None):
    interrupts = result["__interrupt__"]
    item = next(item for item in interrupts if kind is None or item.value.get("kind") == kind)
    return Command(resume={item.id: {"ready": True}})


def install_client(monkeypatch, protocol):
    monkeypatch.setattr(
        "deepagents.middleware.async_subagents.get_client", lambda **_: protocol.client
    )


@pytest.mark.parametrize("work_rounds", [1, 12])
def test_launch_returns_before_worker_finishes_and_parent_collects_after_checkpoint(
    dependencies, monkeypatch, work_rounds
):
    async def scenario():
        manager, store, settings = dependencies
        release = asyncio.Event()
        root_progress = []

        @tool
        async def root_work() -> str:
            """Perform an independent parent operation."""
            assert protocol.launches == 1
            assert not release.is_set()
            root_progress.append("continued")
            return "parent work completed"

        @tool
        async def worker_work() -> str:
            """Wait until the test releases the worker."""
            await release.wait()
            return "verified worker evidence"

        settings.TOOL_SECURITY_RULES.append({"effect": "allow", "tool": "worker_work"})
        monkeypatch.setattr(factory, "create_tools", lambda _: [root_work])
        monkeypatch.setattr(worker_factory, "create_tools", lambda _: [worker_work])

        def make_worker(thread_id, payload):
            model = ScriptedModel(
                responses=[
                    AIMessage(content="", tool_calls=[call("worker_work")]),
                    AIMessage(content='{"summary":"verified worker evidence","files":[]}'),
                ]
            )
            return build_worker(
                manager=manager,
                llm=model,
                settings=settings,
                security_store=store,
                checkpointer=InMemorySaver(),
            )

        protocol = GraphProtocol(make_worker)
        install_client(monkeypatch, protocol)
        model = ScriptedModel(
            responses=[
                AIMessage(content="", tool_calls=[start()]),
                *[
                    AIMessage(content="", tool_calls=[call("root_work", id=f"work-{index}")])
                    for index in range(work_rounds)
                ],
                AIMessage(content="premature final answer"),
                AIMessage(content="Collected and verified worker evidence"),
            ]
        )
        graph = build_agent(
            manager=manager, llm=model, security_store=store, checkpointer=InMemorySaver()
        )
        try:
            paused = await asyncio.wait_for(
                graph.ainvoke({"messages": [HumanMessage(content="delegate")]}, parent_config()), 5
            )
            assert root_progress == ["continued"] * work_rounds
            assert paused["awaiting_subtasks"] == ["child-1"]
            assert paused["__interrupt__"][0].value == {
                "kind": "awaiting_subtasks",
                "task_ids": ["child-1"],
            }
            assert all(
                message.content != "premature final answer" for message in paused["messages"]
            )
            assert not protocol.background["child-1-run-1"].done()
            release.set()
            await protocol.settle("child-1")
            result = await graph.ainvoke(
                resume_interrupt(paused, "awaiting_subtasks"), parent_config("collection-resume")
            )
            assert result["messages"][-1].content == "Collected and verified worker evidence"
            assert result["awaiting_subtasks"] == []
            assert result["async_tasks"]["child-1"]["status"] == "success"
            assert "verified worker evidence" in str(model.seen[-1])
            assert protocol.launches == 1
            expected = {
                "start_async_task",
                "check_async_task",
                "update_async_task",
                "cancel_async_task",
                "list_async_tasks",
            }
            assert expected.issubset(model.bound_names)
        finally:
            await protocol.close()

    asyncio.run(scenario())


def test_completed_start_is_not_replayed_when_sibling_approval_resumes_on_rebuilt_graph(
    dependencies, monkeypatch
):
    async def scenario():
        manager, store, settings = dependencies
        release = asyncio.Event()
        executed = []

        @tool
        async def restricted_action() -> str:
            """An operation requiring a durable user decision."""
            executed.append("root")
            return "approved operation completed"

        @tool
        async def root_work() -> str:
            """Keep this worker alive while its parent is interrupted."""
            await release.wait()
            return "worker evidence"

        monkeypatch.setattr(factory, "create_tools", lambda _: [restricted_action])
        monkeypatch.setattr(worker_factory, "create_tools", lambda _: [root_work])

        def make_worker(thread_id, payload):
            return build_worker(
                manager=manager,
                llm=ScriptedModel(
                    responses=[
                        AIMessage(content="", tool_calls=[call("root_work")]),
                        AIMessage(content="worker evidence"),
                    ]
                ),
                settings=settings,
                security_store=store,
                checkpointer=InMemorySaver(),
            )

        protocol = GraphProtocol(make_worker)
        install_client(monkeypatch, protocol)
        saver = InMemorySaver()
        graph = build_agent(
            manager=manager,
            llm=ScriptedModel(
                responses=[
                    AIMessage(
                        content="", tool_calls=[start(), call("restricted_action", id="approval")]
                    ),
                ]
            ),
            security_store=store,
            checkpointer=saver,
        )
        try:
            paused = await graph.ainvoke(
                {"messages": [HumanMessage(content="do both")]}, parent_config()
            )
            assert paused.get("__interrupt__")
            assert protocol.launches == 1
            assert not protocol.background["child-1-run-1"].done()
            assert any(
                isinstance(message, ToolMessage) and message.tool_call_id == "start"
                for message in paused["messages"]
            )
            approval = store.list_thread("alice", "parent")[0]
            store.decide(approval["id"], "alice", "parent", "allow")
            resumed_graph = build_agent(
                manager=manager,
                llm=ScriptedModel(
                    responses=[
                        AIMessage(content="premature"),
                        AIMessage(content="final with evidence"),
                    ]
                ),
                security_store=store,
                checkpointer=saver,
            )
            paused_again = await resumed_graph.ainvoke(
                Command(
                    resume={
                        paused["__interrupt__"][0].id: {"approval_id": approval["id"]},
                    }
                ),
                parent_config("approval-resume"),
            )
            assert executed == ["root"]
            assert protocol.launches == 1
            assert paused_again["__interrupt__"][0].value["kind"] == "awaiting_subtasks"
            release.set()
            await protocol.settle("child-1")
            result = await resumed_graph.ainvoke(
                resume_interrupt(paused_again), parent_config("collection-resume")
            )
            assert result["messages"][-1].content == "final with evidence"
            assert protocol.launches == 1
            assert executed == ["root"]
        finally:
            await protocol.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("tool_name", ["restricted_action", "execute_terminal"])
def test_child_checkpoint_approval_resumes_while_parent_stays_interrupted(
    dependencies, monkeypatch, tool_name
):
    async def scenario():
        manager, store, settings = dependencies
        executed = []
        settings.TOOL_SECURITY_RULES.append({"effect": "ask", "tool": tool_name})

        @tool(tool_name)
        async def restricted_action(label: str) -> str:
            """Record a separately approved parent or worker operation."""
            executed.append(label)
            return label + " completed"

        monkeypatch.setattr(factory, "create_tools", lambda _: [restricted_action])
        monkeypatch.setattr(worker_factory, "create_tools", lambda _: [restricted_action])

        def make_worker(thread_id, payload):
            return build_worker(
                manager=manager,
                llm=ScriptedModel(
                    responses=[
                        AIMessage(
                            content="", tool_calls=[call(tool_name, {"label": "child"})]
                        ),
                        AIMessage(content="child evidence"),
                    ]
                ),
                settings=settings,
                security_store=store,
                checkpointer=InMemorySaver(),
            )

        protocol = GraphProtocol(make_worker)
        install_client(monkeypatch, protocol)
        graph = build_agent(
            manager=manager,
            llm=ScriptedModel(
                responses=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            start(),
                            call(tool_name, {"label": "parent"}, "parent-call"),
                        ],
                    ),
                    AIMessage(content="combined result"),
                ]
            ),
            security_store=store,
            checkpointer=InMemorySaver(),
        )
        try:
            parent_pause = await graph.ainvoke(
                {"messages": [HumanMessage(content="delegate and operate")]}, parent_config()
            )
            child_pause = await protocol.settle("child-1")
            root_approval = store.list_thread("alice", "parent")[0]
            child_approval = store.list_thread("alice", "child-1")[0]
            assert root_approval["transport"] == child_approval["transport"] == "checkpoint"
            assert child_approval["task_id"] == "child-1"
            assert root_approval["task_id"] is None
            store.decide(child_approval["id"], "alice", "child-1", "allow")
            await protocol.create_run(
                thread_id="child-1",
                assistant_id="taskharbor-worker",
                command={
                    "resume": {
                        child_pause["__interrupt__"][0].id: {"approval_id": child_approval["id"]},
                    }
                },
            )
            child_result = await protocol.settle("child-1")
            assert child_result["messages"][-1].content == "child evidence"
            assert executed == ["child"]
            root_state = await graph.aget_state(parent_config())
            assert any(task.interrupts for task in root_state.tasks)
            assert store.get(root_approval["id"], "alice", "parent")["status"] == "pending"
            store.decide(root_approval["id"], "alice", "parent", "allow")
            result = await graph.ainvoke(
                Command(
                    resume={
                        parent_pause["__interrupt__"][0].id: {"approval_id": root_approval["id"]},
                    }
                ),
                parent_config("root-resume"),
            )
            assert result["messages"][-1].content == "combined result"
            assert executed == ["child", "parent"]
            assert result["async_tasks"]["child-1"]["run_id"] == "child-1-run-2"
            assert result["async_tasks"]["child-1"]["status"] == "success"
        finally:
            await protocol.close()

    asyncio.run(scenario())


def test_worker_reconstruction_uses_its_checkpoint_and_saved_approval(dependencies, monkeypatch):
    async def scenario():
        manager, store, settings = dependencies
        executed = []

        @tool
        async def restricted_action() -> str:
            """Execute after the worker graph has been reconstructed."""
            executed.append("once")
            return "resumed evidence"

        monkeypatch.setattr(worker_factory, "create_tools", lambda _: [restricted_action])
        saver = InMemorySaver()
        config = {"configurable": {"thread_id": "independent-worker", "run_id": "before-restart"}}
        worker = build_worker(
            manager=manager,
            llm=ScriptedModel(
                responses=[
                    AIMessage(content="", tool_calls=[call("restricted_action")]),
                ]
            ),
            settings=settings,
            security_store=store,
            checkpointer=saver,
        )
        paused = await worker.ainvoke(
            {"messages": [HumanMessage(content="perform operation")]}, config
        )
        approval = store.list_thread("alice", "independent-worker")[0]
        assert approval["status"] == "pending"
        store.decide(approval["id"], "alice", "independent-worker", "allow")
        reconstructed = build_worker(
            manager=manager,
            llm=ScriptedModel(
                responses=[
                    AIMessage(content='{"summary":"resumed evidence","files":[]}'),
                ]
            ),
            settings=settings,
            security_store=store,
            checkpointer=saver,
        )
        result = await reconstructed.ainvoke(
            Command(
                resume={
                    paused["__interrupt__"][0].id: {"approval_id": approval["id"]},
                }
            ),
            {"configurable": {"thread_id": "independent-worker", "run_id": "after-restart"}},
        )
        assert executed == ["once"]
        assert not result.get("__interrupt__")
        assert store.get(approval["id"], "alice", "independent-worker")["status"] == "succeeded"

    asyncio.run(scenario())


def test_waiting_child_does_not_block_parent_work_or_sibling_worker(dependencies, monkeypatch):
    async def scenario():
        manager, store, settings = dependencies
        progress = []

        @tool
        async def root_work() -> str:
            """Continue the parent while one child awaits its user."""
            progress.append("parent continued")
            return "independent parent evidence"

        @tool
        async def restricted_action() -> str:
            """Pause the first child for a user decision."""
            progress.append("first child approved")
            return "approved child evidence"

        monkeypatch.setattr(factory, "create_tools", lambda _: [root_work])
        monkeypatch.setattr(worker_factory, "create_tools", lambda _: [restricted_action])

        def make_worker(thread_id, payload):
            responses = [AIMessage(content="sibling evidence")]
            if thread_id == "child-1":
                responses = [
                    AIMessage(content="", tool_calls=[call("restricted_action")]),
                    AIMessage(content="approved child evidence"),
                ]
            return build_worker(
                manager=manager,
                llm=ScriptedModel(responses=responses),
                settings=settings,
                security_store=store,
                checkpointer=InMemorySaver(),
            )

        protocol = GraphProtocol(make_worker)
        install_client(monkeypatch, protocol)
        graph = build_agent(
            manager=manager,
            llm=ScriptedModel(
                responses=[
                    AIMessage(content="", tool_calls=[start("first"), start("second")]),
                    AIMessage(content="", tool_calls=[call("root_work")]),
                    AIMessage(content="premature final"),
                    AIMessage(content="all evidence collected"),
                ]
            ),
            security_store=store,
            checkpointer=InMemorySaver(),
        )
        try:
            paused = await graph.ainvoke(
                {"messages": [HumanMessage(content="parallel work")]}, parent_config()
            )
            first = await protocol.settle("child-1")
            second = await protocol.settle("child-2")
            assert progress == ["parent continued"]
            assert second["messages"][-1].content == "sibling evidence"
            assert first.get("__interrupt__")
            assert paused["__interrupt__"][0].value["kind"] == "awaiting_subtasks"
            assert paused["awaiting_subtasks"] == ["child-1"]
            approval = store.list_thread("alice", "child-1")[0]
            store.decide(approval["id"], "alice", "child-1", "allow")
            await protocol.create_run(
                thread_id="child-1",
                assistant_id="taskharbor-worker",
                command={
                    "resume": {
                        first["__interrupt__"][0].id: {"approval_id": approval["id"]},
                    }
                },
            )
            await protocol.settle("child-1")
            result = await graph.ainvoke(resume_interrupt(paused), parent_config("all-ready"))
            assert result["messages"][-1].content == "all evidence collected"
            assert {task["status"] for task in result["async_tasks"].values()} == {"success"}
            assert progress == ["parent continued", "first child approved"]
            assert protocol.launches == 2
        finally:
            await protocol.close()

    asyncio.run(scenario())


def test_legacy_cancellation_is_applied_before_reporting_unfinished_workers(
    dependencies, monkeypatch
):
    async def scenario():
        manager, store, _settings = dependencies
        monkeypatch.setattr(factory, "create_tools", lambda _: [])
        model = ScriptedModel(responses=[AIMessage(content="The previous task was cancelled")])
        graph = build_agent(
            manager=manager, llm=model, security_store=store, checkpointer=InMemorySaver()
        )
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content="old task"),
                    AIMessage(
                        content="",
                        tool_calls=[call("task", {"instruction": "old work"}, "legacy-start")],
                    ),
                    ToolMessage(
                        content=json.dumps({"task_id": "legacy-1", "status": "running"}),
                        name="task",
                        tool_call_id="legacy-start",
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[call("cancel_task", {"task_id": "legacy-1"}, "legacy-cancel")],
                    ),
                    ToolMessage(
                        content=json.dumps({"task_id": "legacy-1", "status": "cancelled"}),
                        name="cancel_task",
                        tool_call_id="legacy-cancel",
                    ),
                    HumanMessage(content="continue after upgrade"),
                ]
            },
            parent_config(),
        )
        assert not any(
            message.additional_kwargs.get("legacy_subagents_interrupted")
            for message in result["messages"]
        )
        assert sum(isinstance(message, ToolMessage) for message in result["messages"]) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["queued", "running", "waiting_approval", "cancelling"])
def test_legacy_unfinished_workers_receive_an_explicit_recovery_notice(
    dependencies, monkeypatch, status
):
    async def scenario():
        manager, store, _settings = dependencies
        monkeypatch.setattr(factory, "create_tools", lambda _: [])
        model = ScriptedModel(
            responses=[AIMessage(content="Previous work has an unconfirmed result")]
        )
        graph = build_agent(
            manager=manager, llm=model, security_store=store, checkpointer=InMemorySaver()
        )
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content="old task"),
                    AIMessage(
                        content="",
                        tool_calls=[call("task", {"instruction": "old work"}, "legacy-start")],
                    ),
                    ToolMessage(
                        content=json.dumps({"task_id": "legacy-1", "status": status}),
                        name="task",
                        tool_call_id="legacy-start",
                    ),
                    HumanMessage(content="continue after upgrade"),
                ]
            },
            parent_config(),
        )
        notices = [
            message
            for message in result["messages"]
            if message.additional_kwargs.get("legacy_subagents_interrupted")
        ]
        assert len(notices) == 1
        assert "legacy-1" in notices[0].content
        assert "legacy-1" in str(model.seen[0])
        assert sum(isinstance(message, ToolMessage) for message in result["messages"]) == 1

    asyncio.run(scenario())


def test_completion_during_final_model_call_discards_the_superseded_answer(
    dependencies, monkeypatch
):
    async def scenario():
        manager, store, settings = dependencies
        release = asyncio.Event()

        @tool
        async def root_work() -> str:
            """Finish while the parent model is producing a draft answer."""
            await release.wait()
            return "new evidence"

        monkeypatch.setattr(factory, "create_tools", lambda _: [])
        monkeypatch.setattr(worker_factory, "create_tools", lambda _: [root_work])

        def make_worker(thread_id, payload):
            return build_worker(
                manager=manager,
                llm=ScriptedModel(
                    responses=[
                        AIMessage(content="", tool_calls=[call("root_work")]),
                        AIMessage(content="new evidence"),
                    ]
                ),
                settings=settings,
                security_store=store,
                checkpointer=InMemorySaver(),
            )

        protocol = GraphProtocol(make_worker)
        install_client(monkeypatch, protocol)

        async def finish_during_model(response):
            if response.content == "outdated draft":
                release.set()
                await protocol.settle("child-1")

        model = ObservedModel(
            observer=finish_during_model,
            responses=[
                AIMessage(content="", tool_calls=[start()]),
                AIMessage(content="outdated draft"),
                AIMessage(content="final with new evidence"),
            ],
        )
        graph = build_agent(
            manager=manager, llm=model, security_store=store, checkpointer=InMemorySaver()
        )
        try:
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="delegate")]}, parent_config()
            )
            assert result["messages"][-1].content == "final with new evidence"
            assert all(message.content != "outdated draft" for message in result["messages"]), [
                (type(message).__name__, message.content, message.additional_kwargs)
                for message in result["messages"]
            ]
            assert "new evidence" in str(model.seen[-1])
        finally:
            await protocol.close()

    asyncio.run(scenario())
