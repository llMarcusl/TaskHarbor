"""Agent-only rule checks and durable, single-use user confirmations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command, interrupt

from taskharbor.agent.middleware import ToolErrorMiddleware, _runtime_config
from taskharbor.agent.state import AgentState
from taskharbor.security.host import HostCommandCancelled
from taskharbor.security.policy import PolicyEngine
from taskharbor.security.registry import tool_spec
from taskharbor.security.repository import ApprovalError, get_approval_store

logger = logging.getLogger(__name__)


class SecurityState(AgentState, total=False):
    security_turn_id: str
    security_generation: int
    security_origin_run: str | None


@dataclass
class PreparedToolCall:
    request: Any
    record: dict | None = None
    digest: str | None = None
    identity: tuple | None = None


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _error(request, code, message):
    return ToolMessage(
        content=json.dumps(
            {"success": False, "error": code, "message": message}, ensure_ascii=False
        ),
        tool_call_id=request.tool_call["id"],
        name=request.tool_call["name"],
        status="error",
    )


def _saved_message(record, request):
    result = record.get("result")
    if isinstance(result, dict) and result.get("type") == "tool":
        return ToolMessage.model_validate(result)
    return _error(
        request,
        "EXECUTION_STATE_UNKNOWN",
        "The prior operation may have executed. Check its state before retrying.",
    )


class ToolSecurityMiddleware(AgentMiddleware):
    """Keep approval control flow outside actual tool execution and leases."""

    state_schema = SecurityState

    def __init__(self, manager, settings, tools, store=None, *, worker=False):
        self.manager = manager
        self.settings = settings
        self._registered_tools = {tool.name: tool for tool in tools}
        self.policy = PolicyEngine(getattr(settings, "TOOL_SECURITY_RULES", ()))
        self._store = store
        self.worker = worker
        self.tracked = bool(
            store is not None
            or getattr(settings, "SUBAGENTS_ENABLED", False)
            or getattr(settings, "HOST_COMMANDS_ENABLED", False)
            or self.policy.rules
            or any(not tool_spec(name).automatic for name in self._registered_tools)
            or any(
                (tool.metadata or {}).get("requires_confirmation")
                for tool in self._registered_tools.values()
            )
        )

    @property
    def store(self):
        if self._store is None:
            self._store = get_approval_store()
        return self._store

    def _identity(self, runtime):
        config = _runtime_config(runtime).get("configurable", {})
        thread = config.get("thread_id")
        if not thread:
            return None
        owner = str(self.manager.resolve_owner(thread))
        from aegra_api.core.auth_ctx import get_auth_ctx

        authentication = get_auth_ctx()
        if authentication is not None and authentication.user.identity != owner:
            raise PermissionError("The run does not own this conversation")
        return owner, str(thread), config

    def before_agent(self, state, runtime):
        identity = self._identity(runtime)
        if not identity:
            return None
        owner, thread, config = identity
        generation = 0
        if self.tracked:
            if self.store.blocked_thread(owner, thread):
                raise RuntimeError("This conversation is waiting for a tool decision or execution")
            generation = self.store.generation(owner, thread)
        return {
            "security_turn_id": uuid4().hex,
            "security_generation": generation,
            "security_origin_run": str(config["run_id"]) if config.get("run_id") else None,
        }

    async def abefore_agent(self, state, runtime):
        return await asyncio.to_thread(self.before_agent, state, runtime)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        identity = self._identity(runtime)
        if self.tracked and identity:
            owner, thread, _ = identity
            if self.store.generation(owner, thread) != state.get("security_generation", 0):
                return {"jump_to": "end"}
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime):
        # A cancelled turn must also skip the graph's normal tools-to-model edge.
        return await asyncio.to_thread(self.before_model, state, runtime)

    def _key(self, request, owner, thread):
        state = request.state
        message_id = ""
        for message in reversed(state.get("messages", [])):
            if isinstance(message, AIMessage):
                message_id = message.id or _digest(message.model_dump(mode="json"))
                break
        task_id = thread if self.worker else None
        return _digest(
            [
                owner,
                thread,
                task_id,
                state.get("security_turn_id"),
                message_id,
                request.tool_call["id"],
            ]
        )

    def _existing(self, invocation_key, owner, thread):
        if not self.tracked:
            return None
        return self.store.get_invocation(invocation_key, owner, thread)

    def _prepare(self, request):
        name = request.tool_call["name"]
        if name not in self._registered_tools or request.tool is None:
            return _error(request, "UNKNOWN_TOOL", "This tool is not registered")
        spec = tool_spec(name)
        # Preserve the existing Pydantic/error middleware contract for invalid input.
        schema = request.tool.tool_call_schema
        if hasattr(schema, "model_validate"):
            try:
                args = schema.model_validate(request.tool_call["args"]).model_dump(mode="json")
            except Exception as exc:
                return ToolErrorMiddleware._error(request, exc)
        else:
            args = dict(request.tool_call["args"])
        identity = self._identity(request.runtime)
        task_id = identity[1] if identity and self.worker else None
        existing = None
        if identity:
            owner, thread, _ = identity
            key = self._key(request, owner, thread)
            existing = self._existing(key, owner, thread)
        verdict = self.policy.decide(spec, args)
        if verdict == "deny":
            msg = _error(request, "TOOL_DENIED", "A configured rule forbids this operation")
            if existing and existing["status"] in {"pending", "approved"}:
                self.store.finish(existing["id"], "superseded", msg.model_dump(mode="json"))
            return msg
        if name == "execute_host_command" and not getattr(
            self.settings, "HOST_COMMANDS_ENABLED", False
        ):
            msg = _error(
                request, "HOST_COMMANDS_DISABLED", "Host commands are disabled in server settings"
            )
            if existing and existing["status"] in {"pending", "approved"}:
                self.store.finish(existing["id"], "superseded", msg.model_dump(mode="json"))
            return msg
        ask = (
            name == "execute_host_command"
            or verdict == "ask"
            or bool((request.tool.metadata or {}).get("requires_confirmation"))
            or (verdict != "allow" and not spec.automatic)
        )
        if not identity:
            if ask:
                return _error(
                    request,
                    "APPROVAL_REQUIRED",
                    "This operation needs a conversation and a user decision",
                )
            return PreparedToolCall(request)
        if existing and existing.get("result") is not None:
            return _saved_message(existing, request)
        if self.tracked:
            expected = request.state.get("security_generation", 0)
            if self.store.generation(owner, thread) != expected:
                msg = _error(request, "TURN_CANCELLED", "This turn was cancelled")
                if existing:
                    self.store.finish(existing["id"], "cancelled", msg.model_dump(mode="json"))
                return Command(update={"messages": [msg]}, goto="__end__")
        if spec.group == "orchestration":
            if self.worker:
                return _error(
                    request,
                    "DELEGATION_UNAVAILABLE",
                    "Delegation tools are available only in the root agent",
                )
            # Upstream task tools update graph state with Command. Their external
            # submission and replay semantics belong to the upstream middleware.
            return PreparedToolCall(
                request.override(tool_call={**request.tool_call, "args": args}),
                identity=identity,
            )
        execution_context = {}
        if self.worker:
            execution_context["task_id"] = task_id
        elif existing and "task_id" in existing.get("execution_context", {}):
            # Saved root approvals from the live implementation included null.
            # Preserve their digest when replaying the original invocation.
            execution_context["task_id"] = existing["execution_context"]["task_id"]
        try:
            if name == "execute_host_command":
                from taskharbor.security.host import prepare_host_command

                prepared = prepare_host_command(args["command"], args["cwd"], args.get("timeout"))
                args = {field: prepared[field] for field in ("command", "cwd", "timeout")}
                execution_context = {k: v for k, v in prepared.items() if k not in args}
            elif spec.group == "sandbox" and args.get("session_id") and ask:
                # Only freeze a container when this call will wait for a decision.
                self.manager._check_session(args["session_id"])
                info = self.manager.sessions[args["session_id"]]
                if info.user_id != owner or info.thread_id != thread:
                    raise PermissionError("Sandbox does not belong to this conversation")
                execution_context.update(container_id=info.container_id)
        except (ValueError, KeyError, PermissionError, OSError, RuntimeError) as exc:
            msg = _error(
                request, "ENVIRONMENT_CHANGED" if existing else "INVALID_RESOURCE", str(exc)
            )
            if existing:
                self.store.finish(existing["id"], "superseded", msg.model_dump(mode="json"))
            return msg
        # Sandbox tools are automatic by default; explicit per-tool rules can deny them.
        # Custom tools remain ask unless a trusted allow rule covers them.
        normalized = request.override(tool_call={**request.tool_call, "args": args})
        if not ask and not existing and not self.tracked:
            logger.info("Tool allowed: tool=%s executor=%s", name, spec.executor)
            return PreparedToolCall(normalized, identity=identity)
        request_digest = _digest(
            {
                "tool": name,
                "version": spec.version,
                "executor": spec.executor,
                "arguments": args,
                "execution_context": execution_context,
                "rules": [r.model_dump(mode="json") for r in self.policy.rules],
            }
        )
        if existing and existing["request_digest"] != request_digest:
            msg = _error(
                request,
                "APPROVAL_CHANGED",
                "The command, policy or environment changed; request a new operation",
            )
            self.store.finish(existing["id"], "superseded", msg.model_dump(mode="json"))
            return msg
        now = datetime.now(UTC)
        try:
            record = self.store.prepare(
                {
                    "invocation_key": key,
                    "owner": owner,
                    "thread_id": thread,
                    "run_id": request.state.get("security_origin_run"),
                    "tool_call_id": request.tool_call["id"],
                    "task_id": task_id,
                    "transport": "checkpoint",
                    "batch_index": next(
                        (
                            index
                            for message in reversed(request.state.get("messages", []))
                            if isinstance(message, AIMessage) and message.tool_calls
                            for index, call in enumerate(message.tool_calls)
                            if call["id"] == request.tool_call["id"]
                        ),
                        0,
                    ),
                    "tool_name": name,
                    "group": spec.group,
                    "executor": spec.executor,
                    "arguments": args,
                    "execution_context": execution_context,
                    "request_digest": request_digest,
                    "generation": request.state.get("security_generation", 0),
                    "reason": "本机命令需要确认"
                    if spec.executor == "host"
                    else "没有可直接放行此工具的规则，请确认本次操作",
                    "expires_at": (
                        now
                        + timedelta(
                            seconds=getattr(self.settings, "TOOL_APPROVAL_TTL_SECONDS", 1800)
                        )
                    ).isoformat(),
                    "automatic": not ask,
                }
            )
        except ApprovalError as exc:
            return _error(request, "APPROVAL_UNAVAILABLE", str(exc))
        if not ask and record["status"] == "pending":
            record = self.store.decide(record["id"], owner, thread, "allow")
        return PreparedToolCall(normalized, record, request_digest, identity)

    def _decision(self, prepared):
        record, request = prepared.record, prepared.request
        if not record:
            return None
        owner, thread, _ = prepared.identity
        current = record
        while current is not None and current["status"] == "pending":
            turn = request.state.get("sandbox_turn_id")
            if turn:
                self.manager.end_turn(owner, thread, turn)
            # The resume value selects the graph task. The durable decision is
            # the authority for execution, including after service restarts.
            interrupt(
                {
                    "approval_id": current["id"],
                    "tool_call_id": current["tool_call_id"],
                    "execution_thread_id": thread,
                    "task_id": current.get("task_id"),
                }
            )
            current = self.store.get(record["id"], owner, thread)
        if current is None:
            return _error(request, "APPROVAL_UNAVAILABLE", "Approval record is unavailable")
        if current.get("result") is not None:
            return _saved_message(current, request)
        if current["status"] != "approved":
            msg = _error(
                request,
                "APPROVAL_" + current["status"].upper(),
                "The operation was not approved and was not executed",
            )
            if current["status"] not in {"executing", "unknown"}:
                self.store.finish(current["id"], current["status"], msg.model_dump(mode="json"))
            return msg
        # Resuming re-enters _prepare, so parameters and environment were checked above.
        try:
            self.store.claim(record["id"], owner, thread, prepared.digest)
        except ApprovalError as exc:
            return _error(request, "EXECUTION_STATE_UNKNOWN", str(exc))
        return None

    def _finish(self, prepared, result, *, status="succeeded"):
        record = prepared.record
        if record:
            saved = result.model_dump(mode="json") if isinstance(result, ToolMessage) else None
            if isinstance(result, ToolMessage) and status == "succeeded":
                if result.status == "error":
                    status = "failed"
                try:
                    payload = json.loads(result.content) if isinstance(result.content, str) else {}
                except (ValueError, TypeError):
                    payload = {}
                if isinstance(payload, dict):
                    if payload.get("cleanup_failed"):
                        status = "unknown"
                    elif payload.get("cancelled"):
                        status = "cancelled"
                    elif (
                        payload.get("success") is False
                        or payload.get("timed_out")
                        or payload.get("exit_code", 0) not in (0, None)
                    ):
                        status = "failed"
            self.store.finish(record["id"], status, saved)

    def wrap_tool_call(self, request, handler):
        prepared = self._prepare(request)
        if not isinstance(prepared, PreparedToolCall):
            return prepared
        decision = self._decision(prepared)
        if decision is not None:
            return decision
        try:
            result = handler(prepared.request)
        except GraphBubbleUp:
            raise
        except BaseException:
            self._finish(prepared, None, status="unknown")
            raise
        self._finish(prepared, result)
        return result

    async def awrap_tool_call(self, request, handler):
        prepared = await asyncio.to_thread(self._prepare, request)
        if not isinstance(prepared, PreparedToolCall):
            return prepared
        # interrupt must run in the graph task's context, not an unrelated worker.
        decision = self._decision(prepared)
        if decision is not None:
            return decision
        try:
            result = await handler(prepared.request)
        except GraphBubbleUp:
            raise
        except HostCommandCancelled as exc:
            # Save any cleanup result while keeping the graph cancelled.
            result = None
            if exc.result is not None:
                result = ToolMessage(
                    content=json.dumps(exc.result, ensure_ascii=False),
                    tool_call_id=prepared.request.tool_call["id"],
                    name=prepared.request.tool_call["name"],
                    status="error",
                )
            await asyncio.to_thread(
                self._finish,
                prepared,
                result,
                status="succeeded" if result is not None else "unknown",
            )
            raise
        except BaseException:
            await asyncio.to_thread(self._finish, prepared, None, status="unknown")
            raise
        await asyncio.to_thread(self._finish, prepared, result)
        return result
