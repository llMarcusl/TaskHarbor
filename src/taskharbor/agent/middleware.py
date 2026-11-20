"""Middleware carrying the behavior of the original hand-written agent loop."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from contextlib import contextmanager
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.config import get_config
from langgraph.errors import GraphBubbleUp

from taskharbor.agent.configuration import Configuration
from taskharbor.agent.message_projection import (
    prepare_messages_for_llm,
    strip_images_from_messages,
)
from taskharbor.agent.model_registry import ModelRegistry
from taskharbor.agent.state import AgentState
from taskharbor.sandbox import current_thread_id
from taskharbor.sandbox.manager import current_user_id

logger = logging.getLogger(__name__)


def _runtime_config(runtime):
    # ToolRuntime exposes config; graph Runtime (before/after hooks) does not.
    return getattr(runtime, "config", None) or get_config()


class DynamicModelMiddleware(AgentMiddleware):
    """Select the model without changing the external configurable protocol."""

    def __init__(self, models: ModelRegistry) -> None:
        self.models = models

    def _prepare(self, request: ModelRequest) -> ModelRequest:
        cfg = Configuration.from_runnable_config(get_config())
        # The old loop respects a system message already present in history.
        system_message = request.system_message
        if request.messages and isinstance(request.messages[0], SystemMessage):
            system_message = None
        return request.override(model=self.models.resolve(cfg), system_message=system_message)

    def wrap_model_call(self, request, handler):
        return handler(self._prepare(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._prepare(request))


class VisionMiddleware(AgentMiddleware):
    """Project rich outputs and preserve the existing shared vision probe/fallback."""

    def __init__(self, default_vision: bool | None) -> None:
        self.default_vision = default_vision
        self.supported = default_vision

    def _prepare(self, request: ModelRequest) -> tuple[ModelRequest, bool]:
        configurable = get_config().get("configurable", {})
        override = configurable.get("chat_model_supports_vision", self.default_vision)
        if isinstance(override, str):
            override = override.lower() in ("true", "1", "yes")
        if override is not None:
            self.supported = override
        vision = (
            override if override is not None else self.supported is True or self.supported is None
        )
        messages = prepare_messages_for_llm(request.messages, vision)
        has_images = any(
            isinstance(m, ToolMessage) and isinstance(m.content, list) for m in messages
        )
        return request.override(messages=messages), has_images

    def _without_images(self, request: ModelRequest, exc: Exception) -> ModelRequest:
        logger.info("LLM call failed with images (%s), retrying without.", type(exc).__name__)
        self.supported = False
        return request.override(messages=strip_images_from_messages(request.messages))

    def _record_success(self, has_images: bool) -> None:
        if has_images and self.supported is None:
            self.supported = True
            logger.info("Vision probe succeeded, enabling multimodal display outputs.")

    def wrap_model_call(self, request, handler):
        prepared, has_images = self._prepare(request)
        try:
            response = handler(prepared)
        except Exception as exc:
            if not has_images:
                raise
            response = handler(self._without_images(prepared, exc))
        self._record_success(has_images)
        return response

    async def awrap_model_call(self, request, handler):
        prepared, has_images = self._prepare(request)
        try:
            response = await handler(prepared)
        except Exception as exc:
            if not has_images:
                raise
            response = await handler(self._without_images(prepared, exc))
        self._record_success(has_images)
        return response


class ThreadContextMiddleware(AgentMiddleware):
    """Bind authoritative ownership for exactly one tool invocation."""

    def __init__(self, manager=None) -> None:
        self.manager = manager

    @contextmanager
    def _bind(self, request):
        thread_id = _runtime_config(request.runtime).get("configurable", {}).get("thread_id")
        owner = self.manager.resolve_owner(thread_id) if self.manager and thread_id else None
        thread_token = current_thread_id.set(thread_id)
        owner_token = current_user_id.set(owner)
        try:
            yield
        finally:
            current_user_id.reset(owner_token)
            current_thread_id.reset(thread_token)

    def wrap_tool_call(self, request, handler):
        with self._bind(request):
            return handler(request)

    async def awrap_tool_call(self, request, handler):
        with self._bind(request):
            return await handler(request)


class SandboxTurnState(AgentState, total=False):
    sandbox_turn_id: str


class SandboxLifecycleMiddleware(AgentMiddleware):
    """Lease environments for a whole turn, including time spent awaiting the model.

    Heartbeats exist only during a model/tool call. An interrupted graph cannot
    leave an immortal heartbeat behind: any abandoned lease expires naturally.
    Checkpoints contain only a turn ID, never a client-supplied owner identity.
    """

    state_schema = SandboxTurnState

    def __init__(self, manager, lease_ttl_seconds: float = 120) -> None:
        self.manager = manager
        self.heartbeat_interval = max(0.01, lease_ttl_seconds / 3)

    def _identity(self, state, runtime):
        thread_id = _runtime_config(runtime).get("configurable", {}).get("thread_id")
        run_id = state.get("sandbox_turn_id")
        if not thread_id or not run_id:
            return None
        return self.manager.resolve_owner(thread_id), thread_id, run_id

    def before_agent(self, state, runtime):
        from taskharbor.sandbox.manager import current_execution_scope

        thread_id = _runtime_config(runtime).get("configurable", {}).get("thread_id")
        if not thread_id:
            return None
        run_id = uuid4().hex
        owner = self.manager.resolve_owner(thread_id)
        self.manager.begin_turn(
            owner, thread_id, run_id, execution_scope=current_execution_scope.get(),
        )
        return {"sandbox_turn_id": run_id}

    async def abefore_agent(self, state, runtime):
        return self.before_agent(state, runtime)

    def after_agent(self, state, runtime):
        from taskharbor.sandbox.manager import current_execution_scope

        identity = self._identity(state, runtime)
        if identity:
            self.manager.end_turn(*identity, execution_scope=current_execution_scope.get())

    async def aafter_agent(self, state, runtime):
        return self.after_agent(state, runtime)

    @contextmanager
    def _keep_alive(self, state, runtime):
        from taskharbor.sandbox.manager import current_execution_scope

        scope = current_execution_scope.get()
        identity = self._identity(state, runtime)
        if identity is None:
            yield
            return
        self.manager.renew_turn(*identity, execution_scope=scope)
        stopped = threading.Event()

        def heartbeat():
            while not stopped.wait(self.heartbeat_interval):
                try:
                    self.manager.renew_turn(*identity, execution_scope=scope)
                except Exception:
                    logger.warning("Sandbox lease heartbeat failed", exc_info=True)
                    return

        worker = threading.Thread(target=heartbeat, name="sandbox-turn-lease", daemon=True)
        worker.start()
        failed = False
        cancelled = False
        paused = False
        try:
            yield
        except GraphBubbleUp:
            paused = True
            raise
        except BaseException as exc:
            failed = True
            cancelled = isinstance(exc, (asyncio.CancelledError, GeneratorExit, KeyboardInterrupt))
            raise
        finally:
            stopped.set()
            # renew_turn only updates local manager state. Join before release,
            # otherwise a racing heartbeat could recreate the released lease.
            worker.join()
            if paused:
                self.manager.end_turn(*identity, execution_scope=scope)
            elif failed:
                if cancelled:
                    try:
                        self.manager.cancel_turn(*identity, execution_scope=scope)
                    except Exception:
                        logger.warning("Failed to cancel sandbox turn", exc_info=True)
                self.manager.end_turn(*identity, execution_scope=scope)
            else:
                self.manager.renew_turn(*identity, execution_scope=scope)

    def wrap_model_call(self, request, handler):
        with self._keep_alive(request.state, request.runtime):
            return handler(request)

    async def awrap_model_call(self, request, handler):
        with self._keep_alive(request.state, request.runtime):
            return await handler(request)

    def wrap_tool_call(self, request, handler):
        with self._keep_alive(request.state, request.runtime):
            return handler(request)

    async def awrap_tool_call(self, request, handler):
        with self._keep_alive(request.state, request.runtime):
            return await handler(request)


class ToolErrorMiddleware(AgentMiddleware):
    """Keep the original error JSON, including Pydantic parameter errors."""

    @staticmethod
    def _validate(request) -> None:
        # Injected ToolRuntime is supplied by ToolNode, never by the model.
        # Validate only the public schema before execution/injection.
        schema = request.tool.tool_call_schema
        if hasattr(schema, "model_validate"):
            schema.model_validate(request.tool_call["args"])

    @staticmethod
    def _error(request, exc: Exception) -> ToolMessage:
        return ToolMessage(
            content=json.dumps(
                {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "hint": "Fix the invalid parameters and try again.",
                },
                ensure_ascii=False,
            ),
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )

    def wrap_tool_call(self, request, handler):
        if request.tool is None:
            raise KeyError(request.tool_call["name"])
        try:
            self._validate(request)
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return self._error(request, exc)

    async def awrap_tool_call(self, request, handler):
        if request.tool is None:
            raise KeyError(request.tool_call["name"])
        try:
            self._validate(request)
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return self._error(request, exc)
