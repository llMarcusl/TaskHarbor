"""Bounded terminal execution transported through ``docker exec``.

The container helper owns the command's lifetime. Killing the host Docker client
alone does not kill a command inside a container, so failed cleanup is explicit
and the caller must destroy the affected sandbox before reusing it.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

HELPER_PATH = "/client/execute_terminal.py"
EXECUTOR_IMAGE_LABEL = "sandbox-executor-version"
EXECUTOR_VERSION = "2"
MAX_OUTPUT_BYTES = 256 * 1024


@dataclass
class TerminalExecutionResult:
    execution_id: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    cancelled: bool = False
    output_truncated: bool = False
    cleanup_failed: bool = False


@dataclass
class _Execution:
    session_id: str
    container: Any
    cancelled: threading.Event = field(default_factory=threading.Event)


class _BoundedReader:
    def __init__(self, pipe: Any, limit: int):
        self.pipe = pipe
        self.limit = limit
        self.data = bytearray()
        self.truncated = False
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        try:
            while chunk := self.pipe.read(65536):
                remaining = max(0, self.limit - len(self.data))
                self.data.extend(chunk[:remaining])
                self.truncated |= len(chunk) > remaining
        except (OSError, ValueError):
            pass

    def finish(self) -> str:
        self.thread.join(timeout=2)
        return bytes(self.data).decode("utf-8", errors="replace")


class TerminalExecutor:
    """Thread-safe process controller; one instance may manage many sessions."""

    def __init__(self, *, max_output_bytes: int = MAX_OUTPUT_BYTES):
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.max_output_bytes = max_output_bytes
        self._lock = threading.Lock()
        self._active: dict[str, _Execution] = {}

    @staticmethod
    def _command(
        container: Any, args: list[str], *, user: str | None, workdir: str | None
    ) -> list[str]:
        command = ["docker", "exec", "-i"]
        if user:
            command.extend(["--user", user])
        if workdir:
            command.extend(["--workdir", workdir])
        container_id = str(getattr(container, "id", None) or container.name)
        return [*command, container_id, "python3", HELPER_PATH, *args]

    def cancel(self, session_id_or_execution_id: str) -> bool:
        """Request cancellation; ``run`` waits for confirmed container cleanup."""
        matched = False
        with self._lock:
            for execution_id, execution in self._active.items():
                if (
                    execution_id == session_id_or_execution_id
                    or execution.session_id == session_id_or_execution_id
                ):
                    execution.cancelled.set()
                    matched = True
        return matched

    def _request_cancel(
        self, container: Any, execution_id: str, *, user: str | None = None
    ) -> bool:
        try:
            result = subprocess.run(
                self._command(
                    container, ["cancel", execution_id], user=user, workdir=None
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def run(
        self,
        container: Any,
        command: str,
        *,
        session_id: str,
        timeout: float,
        user: str | None = None,
        workdir: str = "/workspace",
    ) -> TerminalExecutionResult:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        execution_id = uuid.uuid4().hex
        execution = _Execution(session_id, container)
        with self._lock:
            self._active[execution_id] = execution
        process = None
        try:
            process = subprocess.Popen(
                self._command(
                    container,
                    ["run", execution_id, str(timeout), str(self.max_output_bytes)],
                    user=user,
                    workdir=workdir,
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # The helper emits one bounded JSON object; reserve room for escaping.
            stdout = _BoundedReader(process.stdout, self.max_output_bytes * 12 + 65536)
            stderr = _BoundedReader(process.stderr, self.max_output_bytes)
            payload = json.dumps({"command": command}, ensure_ascii=False).encode("utf-8")
            process.stdin.write(payload + b"\n")
            process.stdin.close()
            deadline = time.monotonic() + timeout + 5
            cancel_sent = False
            cancel_deadline = None
            while process.poll() is None:
                now = time.monotonic()
                if execution.cancelled.is_set():
                    if cancel_deadline is None:
                        cancel_deadline = now + 5
                    if not cancel_sent:
                        cancel_sent = self._request_cancel(container, execution_id, user=user)
                if now >= deadline or (cancel_deadline is not None and now >= cancel_deadline):
                    self._request_cancel(container, execution_id, user=user)
                    process.kill()
                    process.wait(timeout=2)
                    return TerminalExecutionResult(
                        execution_id=execution_id,
                        exit_code=130 if execution.cancelled.is_set() else 124,
                        stderr="Execution helper did not confirm cleanup; destroy the sandbox.",
                        timed_out=not execution.cancelled.is_set(),
                        cancelled=execution.cancelled.is_set(),
                        cleanup_failed=True,
                    )
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            raw, transport_error = stdout.finish(), stderr.finish()
            try:
                result = json.loads(raw)
                if not isinstance(result, dict) or not isinstance(result.get("exit_code"), int):
                    raise ValueError("invalid execution result")
            except (ValueError, TypeError):
                return TerminalExecutionResult(
                    execution_id=execution_id,
                    exit_code=process.returncode or 1,
                    stderr=(transport_error or "Invalid response from execution helper.")[:4096],
                    cancelled=execution.cancelled.is_set(),
                    output_truncated=stdout.truncated or stderr.truncated,
                    cleanup_failed=True,
                )
            return TerminalExecutionResult(
                execution_id=execution_id,
                exit_code=result["exit_code"],
                stdout=str(result.get("stdout", "")),
                stderr=str(result.get("stderr", "")),
                timed_out=bool(result.get("timed_out", False)),
                cancelled=bool(result.get("cancelled", False)),
                output_truncated=bool(result.get("output_truncated", False)),
                cleanup_failed=bool(result.get("cleanup_failed", False)),
            )
        except (OSError, BrokenPipeError) as exc:
            if process is not None:
                self._request_cancel(container, execution_id, user=user)
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=2)
            return TerminalExecutionResult(
                execution_id=execution_id,
                exit_code=1,
                stderr=f"Terminal execution transport failed: {exc}",
                cleanup_failed=process is not None,
            )
        finally:
            with self._lock:
                self._active.pop(execution_id, None)
            if process is not None:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
