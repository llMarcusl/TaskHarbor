"""Opt-in real PostgreSQL, Aegra HTTP runs and independent graph recovery."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import docker
import httpx
import pytest
import uvicorn
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from pydantic import Field
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from starlette.authentication import AuthCredentials

from taskharbor.agent import factory
from taskharbor.agent.subagents import factory as worker_factory
from taskharbor.agent.subagents import protocol
from taskharbor.security.repository import ApprovalStore
from taskharbor.settings import Settings
from tests.test_agent_refactor import call

pytestmark = pytest.mark.skipif(
    os.environ.get("SANDBOX_SERVICE_TESTS") != "1",
    reason="opt-in local PostgreSQL and HTTP service acceptance",
)


def run_async(coroutine):
    # Psycopg's Windows async transport requires a selector-based event loop.
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        return runner.run(coroutine)


class RoutingModel(BaseChatModel):
    """Choose deterministic calls from graph history across service reconstructions."""

    respond: Any = Field(exclude=True)

    @property
    def _llm_type(self):
        return "service-acceptance"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.respond(messages))])


@pytest.fixture(scope="module")
def postgres_database():
    client = docker.from_env()
    container = client.containers.run(
        "postgres:16-alpine",
        detach=True,
        environment={
            "POSTGRES_USER": "acceptance",
            "POSTGRES_PASSWORD": "acceptance",
            "POSTGRES_DB": "acceptance",
        },
        ports={"5432/tcp": ("127.0.0.1", None)},
        labels={"taskharbor-test": "async-service"},
    )
    try:
        container.reload()
        port = container.attrs["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostPort"]
        dsn = f"postgresql://acceptance:acceptance@127.0.0.1:{port}/acceptance"
        import psycopg

        deadline = time.monotonic() + 30
        while True:
            try:
                with psycopg.connect(dsn, connect_timeout=1):
                    break
            except psycopg.OperationalError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)
        yield dsn
    finally:
        container.remove(force=True, v=True)
        client.close()


@asynccontextmanager
async def real_service(dsn, tmp_path, monkeypatch, *, scenario, listen_port=0):
    from aegra_api.api import runs, threads
    from aegra_api.core import auth_deps, auth_handlers, database, orm
    from aegra_api.core.auth_middleware import LangGraphUser
    from aegra_api.services import langgraph_service
    from fastapi import FastAPI
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres.aio import AsyncPostgresStore

    from taskharbor import http_app
    from taskharbor.security.api import create_security_router
    from taskharbor.security.http import SecurityRunMiddleware, _has_active_run

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", listen_port))
    port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    engine = create_async_engine(dsn.replace("postgresql://", "postgresql+asyncpg://"))
    sync_pool = ConnectionPool(
        dsn,
        open=True,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    async_pool = AsyncConnectionPool(
        dsn,
        open=False,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
    )
    await async_pool.open()
    try:
        async with engine.begin() as connection:
            await connection.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
            await connection.run_sync(orm.Base.metadata.create_all)
        saver = AsyncPostgresSaver(async_pool)
        await saver.setup()
        graph_store = AsyncPostgresStore(async_pool)
        await graph_store.setup()
        store = ApprovalStore(sync_pool)
        await asyncio.to_thread(store.setup)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        monkeypatch.setattr(orm, "async_session_maker", maker)
        monkeypatch.setattr(database.db_manager, "engine", engine)
        monkeypatch.setattr(database.db_manager, "lg_pool", async_pool)
        monkeypatch.setattr(database.db_manager, "_checkpointer", saver)
        monkeypatch.setattr(database.db_manager, "_store", graph_store)
        settings = Settings(
            _env_file=None,
            STORAGE_DIR=tmp_path,
            ENCRYPTION_KEY="service-acceptance-key",
            API_BASE_URL=url,
            SUBAGENT_PROTOCOL_URL=url + protocol._PREFIX,
            SUBAGENTS_ENABLED=True,
            SUBAGENT_MAX_CONCURRENCY=3,
            SUBAGENT_MAX_TASKS_PER_RUN=6,
            SUBAGENT_TIMEOUT_SECONDS=120,
            SUBAGENT_MAX_STEPS=40,
            CHAT_MODEL="test-scripted",
            CHAT_MODEL_PROVIDER="openai",
            CHAT_MODEL_API_KEY="test-key",
            SESSION_MEMORY_ENABLED=False,
            LONG_TERM_MEMORY_ENABLED=False,
            CONTEXT_COMPACTION_ENABLED=False,
            HOST_COMMANDS_ENABLED=False,
            TOOL_SECURITY_RULES=[],
            POSTGRES_USER="acceptance",
            POSTGRES_PASSWORD="acceptance",
            POSTGRES_DB="acceptance",
            POSTGRES_HOST="127.0.0.1",
            POSTGRES_PORT=dsn.rsplit(":", 1)[1].split("/", 1)[0],
        )
        monkeypatch.setattr("taskharbor.settings.get_settings", lambda: settings)
        monkeypatch.setattr(factory, "get_settings", lambda: settings)
        monkeypatch.setattr("taskharbor.agent.configuration.get_settings", lambda: settings)
        monkeypatch.setattr("taskharbor.crypto.resolve_api_key", lambda: "test-key")
        monkeypatch.setattr("taskharbor.clients.infra.get_db_pool", lambda: sync_pool)
        monkeypatch.setattr("taskharbor.security.repository.get_approval_store", lambda: store)
        monkeypatch.setattr("taskharbor.security.http.get_approval_store", lambda: store)
        monkeypatch.setattr(http_app, "get_approval_store", lambda: store)
        manager = MagicMock()
        manager.sessions = {}

        def resolve_owner(thread_id):
            with sync_pool.connection() as connection:
                row = connection.execute(
                    "SELECT user_id FROM thread WHERE thread_id=%s",
                    (thread_id,),
                ).fetchone()
            if not row:
                raise PermissionError("Thread not found")
            return row["user_id"]

        manager.resolve_owner.side_effect = resolve_owner
        monkeypatch.setattr(http_app, "_get_manager", lambda: manager)
        gate = asyncio.Event()
        progress = asyncio.Event()
        attempted = []
        control = {}

        @tool
        async def execute_code(label: str) -> str:
            """A deterministic automatic action used to observe independent progress."""
            attempted.append(label)
            if label == "parent-gate":
                await gate.wait()
            if label == "root-progress":
                progress.set()
            return label

        @tool
        async def custom_action(label: str) -> str:
            """An action requiring the persistent user approval channel."""
            attempted.append(label)
            return label

        tools = [execute_code, custom_action]
        monkeypatch.setattr(factory, "create_tools", lambda manager: list(tools))
        monkeypatch.setattr(worker_factory, "create_tools", lambda manager: list(tools))

        def root_response(messages):
            results = [message for message in messages if isinstance(message, ToolMessage)]
            if not any(message.name == "start_async_task" for message in results):
                descriptions = (
                    ["parent-gate"]
                    if scenario == "parent-approval"
                    else [
                        "approval-worker",
                        "quick-worker",
                    ]
                )
                calls = [
                    call(
                        "start_async_task",
                        {
                            "description": description,
                            "subagent_type": "taskharbor-worker",
                        },
                        id=f"start-{index}",
                    )
                    for index, description in enumerate(descriptions)
                ]
                calls.append(
                    call(
                        "custom_action" if scenario == "parent-approval" else "execute_code",
                        {
                            "label": "root-approved"
                            if scenario == "parent-approval"
                            else "root-progress"
                        },
                        id="root-work",
                    )
                )
                return AIMessage(content="", tool_calls=calls)
            action = control.get("action")
            if action and not any(message.name == action for message in results):
                arguments = {"task_id": control["task_id"]}
                if action == "update_async_task":
                    arguments["message"] = "Finish the updated task and drop the earlier action."
                return AIMessage(content="", tool_calls=[call(action, arguments, id="manage-task")])
            return AIMessage(content="root collected all outcomes")

        def worker_response(messages):
            if not any(isinstance(message, ToolMessage) for message in messages):
                description = next(
                    str(message.content)
                    for message in messages
                    if isinstance(message, HumanMessage)
                )
                label = next(
                    label
                    for label in [
                        "parent-gate",
                        "approval-worker",
                        "quick-worker",
                    ]
                    if label in description
                )
                return AIMessage(
                    content="",
                    tool_calls=[
                        call(
                            "custom_action" if label == "approval-worker" else "execute_code",
                            {"label": label},
                            id="worker-call",
                        )
                    ],
                )
            return AIMessage(content=json.dumps({"summary": "worker finished", "files": []}))

        service = langgraph_service.LangGraphService()
        service._graph_registry = {
            name: {"file_path": "test-owned-graph", "export_name": "graph"}
            for name in ["taskharbor", "taskharbor-worker"]
        }

        def rebuild_graphs():
            service._base_graph_cache = {
                "taskharbor": factory.build_agent(
                    manager=manager,
                    llm=RoutingModel(respond=root_response),
                    security_store=store,
                ),
                "taskharbor-worker": worker_factory.build_worker(
                    manager=manager,
                    llm=RoutingModel(respond=worker_response),
                    settings=settings,
                    security_store=store,
                ),
            }

        rebuild_graphs()
        monkeypatch.setattr(langgraph_service, "get_langgraph_service", lambda: service)
        monkeypatch.setattr(runs, "get_langgraph_service", lambda: service)
        monkeypatch.setattr(langgraph_service, "get_tracing_callbacks", lambda: [])
        monkeypatch.setattr(langgraph_service, "get_tracing_metadata", lambda *args: {})
        await service._ensure_default_assistants()

        class TestAuthentication:
            async def authenticate(self, request):
                owner = request.headers.get("x-test-user", "alice")
                return AuthCredentials(["authenticated"]), LangGraphUser({"identity": owner})

        monkeypatch.setattr(auth_deps, "get_auth_backend", lambda: TestAuthentication())
        monkeypatch.setattr(auth_handlers, "get_auth_instance", lambda: None)
        app = FastAPI()
        app.include_router(protocol.create_protocol_router())
        app.include_router(
            create_security_router(
                lambda: store,
                http_app._authorized_thread_owner,
                http_app._authenticated_owner,
                is_run_active=_has_active_run,
                related_threads=protocol.related_threads,
                get_interrupts=protocol.thread_interrupts,
            )
        )
        app.include_router(threads.router)
        app.include_router(runs.router)
        app.add_middleware(protocol.WorkerResumeConfigMiddleware)
        app.add_middleware(SecurityRunMiddleware)
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                log_level="error",
                lifespan="off",
                ws="none",
            )
        )
        server_task = asyncio.create_task(server.serve(sockets=[listener]))
        async with asyncio.timeout(10):
            while not server.started:
                await asyncio.sleep(0.01)
        reconciler = asyncio.create_task(protocol.run_reconciler())
        try:
            async with httpx.AsyncClient(base_url=url, timeout=20) as client:
                yield SimpleNamespace(
                    client=client,
                    gate=gate,
                    progress=progress,
                    attempted=attempted,
                    store=store,
                    maker=maker,
                    rebuild_graphs=rebuild_graphs,
                    saver=saver,
                    graph_service=service,
                    async_pool=async_pool,
                    control=control,
                )
        finally:
            gate.set()
            reconciler.cancel()
            await asyncio.gather(reconciler, return_exceptions=True)
            active = list(runs.active_runs.values())
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            server.should_exit = True
            await asyncio.wait_for(server_task, timeout=10)
    finally:
        listener.close()
        await async_pool.close()
        sync_pool.close()
        await engine.dispose()


async def create_root(environment):
    response = await environment.client.post("/threads", json={})
    response.raise_for_status()
    thread = response.json()["thread_id"]
    response = await environment.client.post(
        f"/threads/{thread}/runs",
        json={
            "assistant_id": "taskharbor",
            "input": {"messages": [{"role": "user", "content": "execute delegated work"}]},
            "config": {
                "configurable": {
                    "chat_model": "test-scripted",
                    "chat_model_provider": "openai",
                    "chat_model_api_key": "test-key",
                }
            },
            "stream_mode": ["values", "updates"],
        },
    )
    response.raise_for_status()
    return thread, response.json()["run_id"]


async def wait_until(operation, predicate, *, timeout=20):
    async with asyncio.timeout(timeout):
        while True:
            value = await operation()
            if predicate(value):
                return value
            await asyncio.sleep(0.05)


async def get_json(client, path):
    response = await client.get(path)
    response.raise_for_status()
    return response.json()


async def child_rows(environment, parent):
    from aegra_api.core.orm import Thread

    async with environment.maker() as session:
        return list(
            await session.scalars(
                select(Thread).where(
                    Thread.metadata_json["parent_thread_id"].astext == parent,
                )
            )
        )


async def approve(environment, parent, row):
    response = await environment.client.post(
        f"/threads/{parent}/approvals/{row['id']}/decision",
        json={"decision": "allow"},
    )
    response.raise_for_status()
    approval = response.json()
    assert approval["interrupt_id"]
    response = await environment.client.post(
        f"/threads/{approval['execution_thread_id']}/runs",
        json={
            "assistant_id": approval["assistant_id"],
            "command": {
                "resume": {
                    approval["interrupt_id"]: {"approval_id": approval["id"]},
                }
            },
            "stream_mode": ["values", "updates"],
        },
    )
    response.raise_for_status()
    return response.json()


def test_parent_approval_preserves_independent_worker_execution(
    postgres_database, tmp_path, monkeypatch
):
    async def scenario():
        async with real_service(
            postgres_database, tmp_path, monkeypatch, scenario="parent-approval"
        ) as env:
            parent, root_run = await create_root(env)
            await wait_until(
                lambda: get_json(env.client, f"/threads/{parent}/runs/{root_run}"),
                lambda row: row["status"] in {"interrupted", "error"},
            )
            approvals = await get_json(env.client, f"/threads/{parent}/approvals")
            pending = next(row for row in approvals["approvals"] if row["status"] == "pending")
            assert pending["execution_thread_id"] == parent
            children = await child_rows(env, parent)
            assert len(children) == 1
            env.gate.set()
            child = children[0].thread_id
            await wait_until(
                lambda: get_json(env.client, f"/threads/{child}"),
                lambda row: row["status"] == "idle",
            )
            assert "parent-gate" in env.attempted
            assert (await get_json(env.client, f"/threads/{parent}"))["status"] == "interrupted"
            assert env.store.get(pending["id"], "alice", parent)["status"] == "pending"
            resumed = await approve(env, parent, pending)
            final = await wait_until(
                lambda: get_json(env.client, f"/threads/{parent}/runs/{resumed['run_id']}"),
                lambda row: row["status"] in {"success", "error"},
            )
            assert final["status"] == "success", final
            state = await get_json(env.client, f"/threads/{parent}/state")
            assert all(
                task["status"] == "success" for task in state["values"]["async_tasks"].values()
            )
            assert len(await child_rows(env, parent)) == 1

    run_async(scenario())


def test_child_approval_and_persistent_business_wait_recover_after_reconstruction(
    postgres_database,
    tmp_path,
    monkeypatch,
):
    async def scenario():
        from aegra_api.core import database
        from aegra_api.core.orm import Run, Thread
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with real_service(
            postgres_database, tmp_path, monkeypatch, scenario="child-approval"
        ) as env:
            parent, root_run = await create_root(env)
            await asyncio.wait_for(env.progress.wait(), timeout=20)
            approvals = await wait_until(
                lambda: get_json(env.client, f"/threads/{parent}/approvals"),
                lambda result: any(row["task_id"] for row in result["approvals"]),
            )
            pending = next(row for row in approvals["approvals"] if row["task_id"])
            assert pending["execution_thread_id"] != parent
            state = await wait_until(
                lambda: get_json(env.client, f"/threads/{parent}/state"),
                lambda result: any(
                    item["value"].get("kind") == "awaiting_subtasks"
                    for item in protocol.flatten_interrupts(result)
                ),
            )
            child = pending["execution_thread_id"]
            children = await child_rows(env, parent)
            assert len(children) == 2
            sibling = next(row.thread_id for row in children if row.thread_id != child)
            await wait_until(
                lambda: get_json(env.client, f"/threads/{sibling}"),
                lambda row: row["status"] == "idle",
            )
            assert "quick-worker" in env.attempted and "root-progress" in env.attempted
            assert "approval-worker" not in env.attempted
            assert state["values"]["awaiting_subtasks"]

            # Simulate lost service execution records at restart while preserving
            # real PostgreSQL checkpoints, approvals and signed relationships.
            async with env.maker() as session:
                await session.execute(
                    update(Run)
                    .where(
                        Run.thread_id.in_([parent, child]),
                        Run.status == "interrupted",
                    )
                    .values(status="running")
                )
                await session.execute(
                    update(Thread)
                    .where(
                        Thread.thread_id.in_([parent, child]),
                    )
                    .values(status="busy")
                )
                await session.commit()
            env.rebuild_graphs()
            database.db_manager._checkpointer = AsyncPostgresSaver(env.async_pool)
            protocol._signing_key.cache_clear()
            await protocol.reconcile_orphaned_runs()
            assert (await get_json(env.client, f"/threads/{parent}"))["status"] == "interrupted"
            assert (await get_json(env.client, f"/threads/{child}"))["status"] == "interrupted"
            approvals = await get_json(env.client, f"/threads/{parent}/approvals")
            restored = next(row for row in approvals["approvals"] if row["id"] == pending["id"])
            assert restored["status"] == "pending" and restored["interrupt_id"]
            resumed = await approve(env, parent, restored)
            await wait_until(
                lambda: get_json(env.client, f"/threads/{child}/runs/{resumed['run_id']}"),
                lambda row: row["status"] in {"success", "error"},
            )
            final = await wait_until(
                lambda: get_json(env.client, f"/threads/{parent}"),
                lambda row: row["status"] in {"idle", "error"},
            )
            assert final["status"] == "idle"
            state = await get_json(env.client, f"/threads/{parent}/state")
            assert not protocol.flatten_interrupts(state)
            assert not state["values"]["awaiting_subtasks"]
            assert len(state["values"]["async_tasks"]) == 2
            assert all(
                task["status"] == "success" for task in state["values"]["async_tasks"].values()
            )
            assert env.attempted.count("approval-worker") == 1
            async with env.maker() as session:
                latest = await session.scalar(
                    select(Run)
                    .where(
                        Run.thread_id == child,
                    )
                    .order_by(Run.created_at.desc())
                    .limit(1)
                )
                assert latest.config["configurable"]["chat_model"] == "test-scripted"
                assert latest.config["configurable"]["chat_model_api_key"] == "test-key"
            foreign = await env.client.get(
                f"/threads/{parent}/approvals",
                headers={"x-test-user": "bob"},
            )
            assert foreign.status_code == 404

    run_async(scenario())


@pytest.mark.parametrize("action", ["update_async_task", "cancel_async_task"])
def test_waiting_task_management_retires_earlier_approval(
    postgres_database,
    tmp_path,
    monkeypatch,
    action,
):
    async def scenario():
        async with real_service(
            postgres_database,
            tmp_path,
            monkeypatch,
            scenario="child-approval",
        ) as env:
            parent, _ = await create_root(env)
            approvals = await wait_until(
                lambda: get_json(env.client, f"/threads/{parent}/approvals"),
                lambda result: any(row["task_id"] for row in result["approvals"]),
            )
            earlier = next(row for row in approvals["approvals"] if row["task_id"])
            child = earlier["execution_thread_id"]
            state = await wait_until(
                lambda: get_json(env.client, f"/threads/{parent}/state"),
                lambda result: any(
                    item["value"].get("kind") == "awaiting_subtasks"
                    for item in protocol.flatten_interrupts(result)
                ),
            )
            env.control.update(action=action, task_id=child)
            wait = next(
                item
                for item in protocol.flatten_interrupts(state)
                if item["value"].get("kind") == "awaiting_subtasks"
            )
            response = await env.client.post(
                f"/threads/{parent}/runs",
                json={
                    "assistant_id": "taskharbor",
                    "command": {"resume": {wait["id"]: {"subtasks_ready": False}}},
                    "stream_mode": ["values", "updates"],
                },
            )
            response.raise_for_status()
            await wait_until(
                lambda: get_json(env.client, f"/threads/{parent}"),
                lambda row: row["status"] in {"idle", "error"},
            )
            final = await get_json(env.client, f"/threads/{parent}/state")
            tracked = final["values"]["async_tasks"][child]
            assert tracked["status"] == (
                "success" if action == "update_async_task" else "cancelled"
            ), tracked
            await wait_until(
                lambda: get_json(env.client, f"/threads/{child}"),
                lambda row: row["status"] in {"idle", "error"},
            )
            retired = env.store.get(earlier["id"], "alice", child)
            assert retired["status"] == "cancelled"
            assert retired["result"] is not None
            assert not env.store.blocked_thread("alice", child)
            assert "approval-worker" not in env.attempted

    run_async(scenario())
