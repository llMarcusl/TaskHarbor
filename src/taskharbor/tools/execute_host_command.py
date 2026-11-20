"""Agent-only Windows host command tool; approval is handled by middleware."""

from __future__ import annotations

import asyncio
import json
import logging
import threading

from langchain_core.tools import BaseTool, StructuredTool

from taskharbor.sandbox.manager import current_thread_id, current_user_id
from taskharbor.security.host import (
    HostCommandCancelled,
    HostCommandExecutor,
    get_host_executor,
)
from taskharbor.settings import get_settings

# Job termination (3s), root wait (3s), two output readers (2s each), plus
# scheduling headroom. Healthy cancellation normally finishes within a poll.
CANCEL_CLEANUP_WAIT_SECONDS = 3 + 3 + 2 + 2 + 2
logger = logging.getLogger(__name__)


def create_execute_host_command_tool(executor: HostCommandExecutor | None = None) -> BaseTool:
    controller = executor or get_host_executor()

    def scope() -> str:
        if not getattr(get_settings(), "HOST_COMMANDS_ENABLED", False):
            raise PermissionError("Windows host commands are disabled")
        owner, thread = current_user_id.get(None), current_thread_id.get(None)
        if not owner or not thread:
            raise PermissionError("Host commands require a trusted user and conversation")
        return f"{owner}:{thread}"

    def failure(exc: Exception) -> str:
        return json.dumps(
            {
                "success": False,
                "executor": "host",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            },
            ensure_ascii=False,
        )

    def execute_host_command(command: str, cwd: str, timeout: int | None = None) -> str:
        """Execute an individually approved PowerShell command on the Windows host.

        This operates on the real local machine, not inside Docker. The command
        must first be approved by the user through the Agent security middleware.
        Use the sandbox terminal for temporary code and data work instead.

        Args:
            command: Exact PowerShell script to execute; no interactive input is supported.
            cwd: Explicit existing absolute Windows directory, never /workspace.
            timeout: Execution seconds, default 60 and maximum 300.

        Returns:
            JSON containing execution_id, host/shell identity, cwd, exit_code,
            stdout, stderr, timed_out, cancelled, truncation and cleanup status.
        """
        try:
            execution_scope = scope()
            result = controller.run(
                command,
                cwd,
                timeout,
                scope=execution_scope,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return failure(exc)

    async def aexecute_host_command(command: str, cwd: str, timeout: int | None = None) -> str:
        execution_scope = None
        cancelled = threading.Event()
        worker = None
        try:
            execution_scope = scope()
            worker = asyncio.create_task(
                asyncio.to_thread(
                    controller.run,
                    command,
                    cwd,
                    timeout,
                    scope=execution_scope,
                    _cancelled=cancelled,
                )
            )
            # Cancelling the graph must not discard the process cleanup result.
            result = await asyncio.shield(worker)
            return json.dumps(result, ensure_ascii=False)
        except asyncio.CancelledError as exc:
            cancelled.set()
            if execution_scope is not None:
                controller.cancel_scope(execution_scope)
            result = None
            if worker is not None:
                try:
                    result = await asyncio.wait_for(
                        asyncio.shield(worker), timeout=CANCEL_CLEANUP_WAIT_SECONDS
                    )
                except (Exception, asyncio.CancelledError):
                    # A timeout or interrupted cleanup cannot confirm termination.
                    logger.warning("Host command cleanup result unavailable after cancellation")
            # Keep cancellation in control of the graph; middleware only saves
            # the definitive worker result, without continuing the model turn.
            raise HostCommandCancelled(result) from exc
        except Exception as exc:
            return failure(exc)

    return StructuredTool.from_function(
        func=execute_host_command,
        coroutine=aexecute_host_command,
        name="execute_host_command",
    )
