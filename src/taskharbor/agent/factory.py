"""Assemble the sandbox agent using LangChain's middleware-based agent loop."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from taskharbor.agent.configuration import Configuration
from taskharbor.agent.memory import LongTermMemoryMiddleware, SessionMemoryMiddleware
from taskharbor.agent.memory.injection import MemoryInjectionMiddleware
from taskharbor.agent.middleware import (
    DynamicModelMiddleware,
    SandboxLifecycleMiddleware,
    ThreadContextMiddleware,
    ToolErrorMiddleware,
    VisionMiddleware,
)
from taskharbor.agent.model_registry import ModelRegistry
from taskharbor.agent.prompts import SYSTEM_PROMPT
from taskharbor.agent.security_middleware import ToolSecurityMiddleware
from taskharbor.agent.state import AgentState
from taskharbor.agent.tool_recovery import RunIdentityMiddleware, ToolHistoryMiddleware
from taskharbor.clients import get_chat_model
from taskharbor.sandbox import SandboxManager, get_manager
from taskharbor.settings import get_settings
from taskharbor.tools import create_tools


def build_agent(
    *,
    manager: SandboxManager | None = None,
    llm: BaseChatModel | None = None,
    checkpointer: Any = None,
    memory_service: Any = None,
    security_store: Any = None,
) -> Any:
    """Build the agent with independently checkpointed tool calls."""
    if manager is None:
        manager = get_manager()
    if llm is None:
        llm = get_chat_model()
    settings = get_settings()
    subagents_enabled = getattr(settings, "SUBAGENTS_ENABLED", False)
    memory_middleware = []
    if settings.SESSION_MEMORY_ENABLED:
        memory_middleware.append(SessionMemoryMiddleware(memory_service))
    if settings.LONG_TERM_MEMORY_ENABLED:
        memory_middleware.append(LongTermMemoryMiddleware(memory_service))
    tools = create_tools(manager)
    delegation_middleware = []
    delegation_tools = []
    if subagents_enabled:
        from deepagents.middleware.async_subagents import AsyncSubAgentMiddleware

        from taskharbor.agent.subagents.factory import CollectTasksMiddleware
        from taskharbor.agent.subagents.files import create_parent_file_tools
        from taskharbor.agent.subagents.protocol import async_subagent_spec

        async_subagents = AsyncSubAgentMiddleware(
            async_subagents=[async_subagent_spec(manager, settings)],
        )
        delegation_tools = list(async_subagents.tools)
        delegation_middleware = [
            async_subagents,
            CollectTasksMiddleware(),
        ]
        tools.extend(create_parent_file_tools(manager, settings))
    host_enabled = getattr(settings, "HOST_COMMANDS_ENABLED", False)
    if host_enabled:
        from taskharbor.tools.execute_host_command import create_execute_host_command_tool

        tools.append(create_execute_host_command_tool())
    system_prompt = SYSTEM_PROMPT
    if subagents_enabled:
        from taskharbor.agent.subagents.factory import DELEGATION_PROMPT

        system_prompt += DELEGATION_PROMPT
    if host_enabled:
        system_prompt += (
            "\nHost operations: execute_host_command runs PowerShell on the real Windows host, "
            "separately from execute_terminal in Docker. Use it only for tasks involving the "
            "user's local machine. Require an explicit absolute Windows working directory; "
            "ask the user if it is unknown. Each host command needs the application's user "
            "confirmation. Do not claim a chat message alone grants that approval. "
            "If the original sandbox was recycled, files remain but variables do not; "
            "prepare a new environment and a new call rather than reusing an old approval."
        )
    context_middleware = []
    request_guard = []
    if settings.CONTEXT_COMPACTION_ENABLED:
        from .context_middleware import ContextMiddleware, ContextRequestGuard
        from .history_tool import create_history_tool

        tools.append(create_history_tool(memory_service))
        context = ContextMiddleware(
            settings, [*tools, *delegation_tools], system_prompt, memory_service,
        )
        context_middleware = [context]
        request_guard = [ContextRequestGuard(context)]
    else:
        from .context_middleware import DisabledCompactCommand

        context_middleware = [DisabledCompactCommand()]
    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        state_schema=AgentState,
        context_schema=Configuration,
        middleware=[
            *context_middleware,
            *memory_middleware,
            *delegation_middleware,
            ThreadContextMiddleware(manager),
            ToolSecurityMiddleware(
                manager, settings, [*tools, *delegation_tools], store=security_store,
            ),
            SandboxLifecycleMiddleware(manager, settings.SANDBOX_LEASE_TTL_SECONDS),
            RunIdentityMiddleware(),
            DynamicModelMiddleware(ModelRegistry(llm, settings)),
            MemoryInjectionMiddleware(settings, memory_service),
            VisionMiddleware(settings.CHAT_MODEL_SUPPORTS_VISION),
            ToolHistoryMiddleware(),
            ToolErrorMiddleware(),
            *request_guard,
        ],
        checkpointer=checkpointer,
    )
    # Middleware hooks consume graph steps too; these are not model iterations.
    return graph.with_config(
        max_concurrency=settings.TOOL_MAX_CONCURRENCY,
        recursion_limit=settings.AGENT_MAX_STEPS,
    )
