"""Graph-owned compaction and bounded provider overflow recovery."""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ExtendedModelResponse, ModelResponse, hook_config
from langchain_core.messages import AIMessage, RemoveMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.config import get_config
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from .compaction import CompactionError, Compactor, compact_command, estimate, prompt_too_long
from .memory.service import get_memory_service


class DisabledCompactCommand(AgentMiddleware):
    """Reserve /compact even when the feature is disabled; never send it as a task."""

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if compact_command(state["messages"]) is not None:
            return {
                "messages": [AIMessage(content="上下文压缩尚未启用，请先配置模型预算和压缩开关。")],
                "jump_to": "end",
            }
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)


class ContextMiddleware(AgentMiddleware):
    def __init__(self, settings, tools, system_prompt, service=None, model=None):
        self.settings, self.tools, self.system_prompt = settings, tools, system_prompt
        self.service, self.model = service, model
        self.tool_cost = len(
            json.dumps(
                [convert_to_openai_tool(t) for t in tools],
                ensure_ascii=False,
            ).encode("utf-8")
        )

    def budget(self):
        config = get_config().get("configurable", {})
        name = config.get("chat_model") or self.settings.CHAT_MODEL
        window = self.settings.CONTEXT_MODEL_WINDOWS.get(name)
        if window is None:
            raise CompactionError("context_window_not_configured_for_selected_model")
        return window - self.settings.CONTEXT_OUTPUT_RESERVE - self.settings.CONTEXT_SAFETY_TOKENS

    def before_agent(self, state, runtime):
        return {"context_recovery_used": False, "context_recovery_pending": False}

    async def abefore_agent(self, state, runtime):
        return self.before_agent(state, runtime)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        settings = self.settings
        focus = compact_command(state["messages"])
        manual = focus is not None
        reactive = state.get("context_recovery_pending", False)
        config = get_config().get("configurable", {})
        thread_id = config.get("thread_id")
        vision = config.get("chat_model_supports_vision", settings.CHAT_MODEL_SUPPORTS_VISION)
        if isinstance(vision, str):
            vision = vision.lower() in ("true", "1", "yes")
        if not thread_id:
            raise CompactionError("compaction_requires_thread_id")
        service = self.service or get_memory_service()
        overhead = self.tool_cost + estimate([SystemMessage(content=self.system_prompt)])
        compacted_overhead = overhead
        if settings.MEMORY_INJECTION_ENABLED and (
            settings.SESSION_MEMORY_ENABLED or settings.LONG_TERM_MEMORY_ENABLED
        ):
            from .memory.injection import MemoryInjectionMiddleware

            injection = MemoryInjectionMiddleware(settings, service)
            current = injection._read(thread_id, bool(state.get("last_compaction")))
            future = injection._read(thread_id, True)
            overhead += len(current.encode("utf-8"))
            compacted_overhead += len(future.encode("utf-8"))
        candidate, metadata = Compactor(service, settings, self.model).prepare(
            state,
            thread_id,
            self.budget(),
            overhead,
            force=manual or reactive,
            manual=manual,
            focus=focus or "",
            reactive=reactive,
            compacted_overhead=compacted_overhead,
            vision=vision is not False,
        )
        update = {"context_recovery_pending": False}
        if candidate != state["messages"]:
            update["messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *candidate]
        if metadata:
            update.update(
                last_compaction=metadata, context_version=state.get("context_version", 0) + 1
            )
        if manual:
            notice = AIMessage(
                id=str(uuid4()),
                content="上下文已压缩。" + (f"保留重点：{focus}" if focus else ""),
                additional_kwargs={"memory_internal": True, "context_notice": True},
            )
            # Keep the command visible, but not in the model's future working context.
            update["messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *candidate, notice]
            update["jump_to"] = "end"
        return update

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime):
        return await asyncio.to_thread(self.before_model, state, runtime)

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        if state.get("context_recovery_pending"):
            return {"jump_to": "model"}
        # Archive final responses as well, even when both extraction switches are off.
        service = self.service or get_memory_service()
        thread_id = get_config().get("configurable", {}).get("thread_id")
        Compactor(service, self.settings).archive(thread_id, state["messages"])
        return None

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state, runtime):
        return await asyncio.to_thread(self.after_model, state, runtime)

    def _recovery(self, request, exc):
        if not prompt_too_long(exc):
            raise exc
        if request.state.get("context_recovery_used"):
            raise CompactionError("context_overflow_after_recovery") from exc
        # Commit this marker, then after_model routes through before_model. No tools run.
        marker = AIMessage(
            content="",
            additional_kwargs={
                "memory_internal": True,
                "context_recovery": True,
            },
        )
        return ExtendedModelResponse(
            model_response=ModelResponse(result=[marker]),
            command=Command(
                update={"context_recovery_pending": True, "context_recovery_used": True}
            ),
        )

    def wrap_model_call(self, request, handler):
        try:
            return handler(request)
        except Exception as exc:
            return self._recovery(request, exc)

    async def awrap_model_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as exc:
            return self._recovery(request, exc)


class ContextRequestGuard(AgentMiddleware):
    """Check the final projected/injected request inside all other wrappers."""

    def __init__(self, context):
        self.context = context

    def _check(self, request):
        messages = [m for m in request.messages if not m.additional_kwargs.get("context_notice")]
        if request.system_message:
            messages = [request.system_message, *messages]
        size = estimate(messages, self.context.settings.CONTEXT_IMAGE_TOKENS)
        if size + self.context.tool_cost > self.context.budget():
            raise CompactionError("prompt-too-long: final request exceeds configured budget")
        return request.override(
            messages=[m for m in request.messages if not m.additional_kwargs.get("context_notice")],
            model_settings={
                **request.model_settings,
                "max_tokens": min(
                    request.model_settings.get("max_tokens")
                    or self.context.settings.CONTEXT_OUTPUT_RESERVE,
                    self.context.settings.CONTEXT_OUTPUT_RESERVE,
                ),
            },
        )

    def wrap_model_call(self, request, handler):
        return handler(self._check(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._check(request))
