"""Bounded Windows host commands, separate from the Docker sandbox executor.

This module does not authorize commands. The Agent's security middleware must
approve the exact prepared call before invoking it. Host commands retain the
Windows account's filesystem/network permissions; a working directory is not a
security boundary.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import ctypes
import hashlib
import locale
import logging
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from taskharbor.settings import get_settings

MAX_OUTPUT_BYTES = 64 * 1024
DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 300
CREATE_SUSPENDED = 0x00000004
CREATE_NO_WINDOW = 0x08000000
logger = logging.getLogger(__name__)


class HostCommandCancelled(asyncio.CancelledError):
    """Propagate graph cancellation together with the worker's cleanup result."""

    def __init__(self, result: dict[str, Any] | None = None):
        super().__init__("Host command task was cancelled")
        self.result = result


def _file_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(128 * 1024), b""):
            digest.update(chunk)
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


@lru_cache(maxsize=1)
def _powershell_installation() -> tuple[str, dict[str, int | str]]:
    """Pin an installed interpreter without consulting cwd or executable PATH."""
    if os.name != "nt":
        raise RuntimeError("Host commands are supported only on Windows")
    candidates: list[Path] = []
    seen: set[str] = set()
    for variable in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(variable)
        if not root:
            continue
        installation_root = (Path(root) / "PowerShell").resolve()
        if str(installation_root).casefold() in seen:
            continue
        seen.add(str(installation_root).casefold())
        candidates.append(installation_root / "7" / "pwsh.exe")
        if installation_root.is_dir():
            versions = sorted(
                (path for path in installation_root.iterdir() if path.name.isdigit()),
                key=lambda path: int(path.name),
                reverse=True,
            )
            candidates.extend(path / "pwsh.exe" for path in versions)
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve()
    candidates.append(system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve(strict=True)
        # An installation-directory link must not redirect interpreter selection
        # to an unrelated user-writable directory.
        if not resolved.is_relative_to(candidate.parent.parent.resolve()):
            continue
        return str(resolved), _file_identity(resolved)
    raise RuntimeError("No installed PowerShell interpreter was found")


def _encoded_command(command: str) -> str:
    # UTF-16LE is the format required by both Windows PowerShell and pwsh's
    # -EncodedCommand. A single argument avoids cmd.exe and command-line quoting.
    script = (
        "$__hostUtf8 = [System.Text.UTF8Encoding]::new($false)\n"
        "[Console]::InputEncoding = $__hostUtf8\n"
        "[Console]::OutputEncoding = $__hostUtf8\n"
        "$OutputEncoding = $__hostUtf8\n"
        "$env:PYTHONIOENCODING = 'utf-8'\n"
        "$env:PYTHONUTF8 = '1'\n"
        "$ProgressPreference = 'SilentlyContinue'\n"
        "$ErrorActionPreference = 'Stop'\n"
        "try {\n& {\n" + command + "\n}\n"
        "$__hostSucceeded = $?\n"
        "if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n"
        "if (-not $__hostSucceeded) { exit 1 }\n"
        "} catch { [Console]::Error.WriteLine(($_ | Out-String)); exit 1 }\n"
    )
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def _argv(prepared: dict[str, Any]) -> list[str]:
    return [
        prepared["powershell_path"],
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        _encoded_command(prepared["command"]),
    ]


def prepare_host_command(
    command: str, cwd: str, timeout: int | float | None = None
) -> dict[str, Any]:
    """Validate and freeze command parameters for display and approval hashing.

    The middleware stores/digests this complete dictionary, then compares a new
    preparation immediately before execution. The interpreter is pinned for the
    service lifetime; installation changes require restarting the service.
    """
    if not isinstance(command, str) or not command.strip() or "\x00" in command:
        raise ValueError("command must be a nonempty string without NUL characters")
    if not isinstance(cwd, str) or not cwd or "\x00" in cwd:
        raise ValueError("cwd must be an explicit existing absolute Windows directory")
    directory = Path(cwd)
    if not directory.is_absolute():
        raise ValueError("cwd must be an explicit existing absolute Windows directory")
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("cwd must be a directory")
    settings = get_settings()
    default_timeout = getattr(settings, "HOST_COMMAND_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    maximum = min(getattr(settings, "HOST_COMMAND_MAX_TIMEOUT_SECONDS", MAX_TIMEOUT_SECONDS), 300)
    seconds = default_timeout if timeout is None else timeout
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError("timeout must be a number of seconds")
    if not 1 <= seconds <= maximum:
        raise ValueError(f"timeout must be between 1 and {maximum} seconds")
    executable, expected_identity = _powershell_installation()
    if _file_identity(Path(executable)) != expected_identity:
        raise RuntimeError(
            "PowerShell interpreter changed; restart the service before approving calls"
        )
    prepared = {
        "command": command,
        "cwd": str(directory),
        "timeout": seconds,
        "shell": "powershell",
        "powershell_path": executable,
        "powershell_identity": dict(expected_identity),
    }
    # CreateProcess has a 32,767 UTF-16 code-unit command-line limit. Check before
    # asking a user to approve a command which Windows could never launch.
    command_line = subprocess.list2cmdline(_argv(prepared))
    if len(command_line.encode("utf-16-le")) // 2 >= 32767:
        raise ValueError("command is too long for Windows; use an explicitly approved script file")
    return prepared


def _minimal_environment() -> dict[str, str]:
    """Preserve OS/tool lookup paths, never arbitrary backend credentials."""
    allowed = {
        "SYSTEMROOT",
        "WINDIR",
        "SYSTEMDRIVE",
        "COMSPEC",
        "TEMP",
        "TMP",
        "PATH",
        "PATHEXT",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "COMMONPROGRAMFILES",
        "COMMONPROGRAMFILES(X86)",
        "COMMONPROGRAMW6432",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "APPDATA",
        "PROCESSOR_ARCHITECTURE",
        "NUMBER_OF_PROCESSORS",
    }
    result = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    result["PYTHONIOENCODING"] = "utf-8"
    result["PYTHONUTF8"] = "1"
    return result


def _decode_output(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    encodings = ["utf-8-sig", locale.getpreferredencoding(False)]
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        encodings.extend((f"cp{kernel32.GetOEMCP()}", f"cp{kernel32.GetACP()}"))
    for encoding in dict.fromkeys(encodings):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError as exc:
            if encoding == "utf-8-sig" and exc.reason == "unexpected end of data":
                return data.decode("utf-8-sig", errors="replace")
        except (UnicodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


class _BoundedReader:
    def __init__(self, pipe: Any, limit: int):
        self.pipe = pipe
        self.limit = limit
        self.data = bytearray()
        self.truncated = False
        self.thread = threading.Thread(target=self._read, name="host-command-output", daemon=True)
        self.thread.start()

    def _read(self) -> None:
        try:
            while chunk := self.pipe.read(8192):
                remaining = max(0, self.limit - len(self.data))
                self.data.extend(chunk[:remaining])
                self.truncated |= len(chunk) > remaining
        except (OSError, ValueError):
            pass
        finally:
            try:
                self.pipe.close()
            except (OSError, ValueError):
                pass

    def finish(self) -> tuple[str, bool]:
        self.thread.join(timeout=2)
        return _decode_output(bytes(self.data)), not self.thread.is_alive()


class _WindowsJob:
    """Kill-on-close job, assigned before the PowerShell process can run."""

    def __init__(self) -> None:
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class Accounting(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_longlong),
                ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        self._lock = threading.RLock()
        self._accounting = Accounting
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self._kernel32.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self._kernel32.TerminateJobObject.restype = wintypes.BOOL
        self._kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        self._kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._resume = ctypes.WinDLL("ntdll", use_last_error=True).NtResumeProcess
        self._resume.argtypes = [wintypes.HANDLE]
        self._resume.restype = ctypes.c_long
        self.handle = self._kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
        if not self._kernel32.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign_and_resume(self, process: subprocess.Popen[bytes]) -> None:
        with self._lock:
            if not self._kernel32.AssignProcessToJobObject(self.handle, int(process._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
            status = self._resume(int(process._handle))
            if status < 0:
                raise OSError(f"Could not resume job-contained process (NTSTATUS {status:#x})")

    def terminate_and_wait(self, exit_code: int, timeout: float = 3) -> bool:
        with self._lock:
            if not self.handle:
                return False
            if not self._kernel32.TerminateJobObject(self.handle, exit_code):
                return False
            deadline = time.monotonic() + timeout
            while True:
                counters = self._accounting()
                if not self._kernel32.QueryInformationJobObject(
                    self.handle, 1, ctypes.byref(counters), ctypes.sizeof(counters), None
                ):
                    return False
                if counters.ActiveProcesses == 0:
                    return True
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.01)

    def close(self) -> bool:
        with self._lock:
            if not self.handle:
                return True
            if self._kernel32.CloseHandle(self.handle):
                self.handle = None
                return True
            return False


@dataclass
class _Execution:
    scope: str
    cancelled: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    job: _WindowsJob | None = None


class HostCommandExecutor:
    """Thread-safe host process controller; all executions have finite lifetimes."""

    def __init__(self, *, max_output_bytes: int = MAX_OUTPUT_BYTES):
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.max_output_bytes = max_output_bytes
        self._lock = threading.Lock()
        self._active: dict[str, _Execution] = {}
        self._closed = False

    def cancel(self, execution_id_or_scope: str) -> bool:
        with self._lock:
            execution = self._active.get(execution_id_or_scope)
            if execution is not None:
                execution.cancelled.set()
                return True
        return self.cancel_scope(execution_id_or_scope)

    def cancel_scope(self, scope: str) -> bool:
        if not scope:
            raise ValueError("A trusted execution scope is required")
        with self._lock:
            matched = False
            for execution in self._active.values():
                if execution.scope == scope:
                    execution.cancelled.set()
                    matched = True
            return matched

    def shutdown(self) -> None:
        """Cancel in-flight commands during application/process shutdown."""
        with self._lock:
            self._closed = True
            active = list(self._active.values())
            for execution in active:
                execution.cancelled.set()
        deadline = time.monotonic() + 5
        for execution in active:
            execution.done.wait(timeout=max(0, deadline - time.monotonic()))
            if not execution.done.is_set() and execution.job is not None:
                try:
                    if not execution.job.terminate_and_wait(130):
                        logger.error(
                            "Host command process-tree cleanup was not confirmed at shutdown"
                        )
                except Exception:
                    logger.exception("Host command process-tree cleanup failed at shutdown")

    def run(
        self,
        command: str,
        cwd: str,
        timeout: int | float | None = None,
        *,
        scope: str,
        _cancelled: threading.Event | None = None,
    ) -> dict[str, Any]:
        if not scope:
            raise ValueError("A trusted execution scope is required")
        prepared = prepare_host_command(command, cwd, timeout)
        execution_id = uuid.uuid4().hex
        execution = _Execution(scope, cancelled=_cancelled or threading.Event())
        result: dict[str, Any] = {
            "execution_id": execution_id,
            "executor": "host",
            "shell": prepared["shell"],
            "powershell_path": prepared["powershell_path"],
            "cwd": prepared["cwd"],
            "exit_code": 1,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "cancelled": False,
            "output_truncated": False,
            "cleanup_failed": False,
        }
        with self._lock:
            if self._closed:
                raise RuntimeError("Host command executor is shutting down")
            if execution.cancelled.is_set():
                return {
                    **result,
                    "success": False,
                    "cancelled": True,
                    "exit_code": 130,
                    "stderr": "Host command was cancelled before execution started",
                }
            self._active[execution_id] = execution
        process = None
        job = None
        stdout = stderr = None
        assigned = False
        try:
            deadline = time.monotonic() + prepared["timeout"]
            job = _WindowsJob()
            execution.job = job
            if not execution.cancelled.is_set():
                process = subprocess.Popen(
                    _argv(prepared),
                    cwd=prepared["cwd"],
                    env=_minimal_environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                    close_fds=True,
                    creationflags=CREATE_SUSPENDED | CREATE_NO_WINDOW,
                )
                # Fail closed: no user command runs before the job owns the
                # process, so a fast command cannot spawn an untracked child.
                with self._lock:
                    if not execution.cancelled.is_set() and not self._closed:
                        job.assign_and_resume(process)
                        assigned = True
                    else:
                        result["cancelled"] = True
                if assigned:
                    stdout = _BoundedReader(process.stdout, self.max_output_bytes)
                    stderr = _BoundedReader(process.stderr, self.max_output_bytes)
                    while process.poll() is None:
                        if execution.cancelled.is_set():
                            result["cancelled"] = True
                            break
                        if time.monotonic() >= deadline:
                            result["timed_out"] = True
                            break
                        execution.cancelled.wait(timeout=0.05)
                    result["exit_code"] = (
                        process.returncode if process.returncode is not None else 1
                    )
            else:
                result["cancelled"] = True
        except Exception as exc:
            result["stderr"] = f"Host command could not complete: {exc}"
        finally:
            # Background tasks are not a supported API: clean all descendants
            # even when the PowerShell root exits successfully.
            if process is not None:
                cleanup_code = 130 if result["cancelled"] else 124 if result["timed_out"] else 1
                if job is not None and assigned:
                    try:
                        result["cleanup_failed"] |= not job.terminate_and_wait(cleanup_code)
                    except Exception:
                        result["cleanup_failed"] = True
                elif process.poll() is None:
                    # Failed assignment leaves only the suspended root; no user
                    # script has run and it cannot have created descendants.
                    try:
                        process.kill()
                    except OSError:
                        result["cleanup_failed"] = True
                try:
                    process.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    result["cleanup_failed"] = True
            if job is not None:
                try:
                    result["cleanup_failed"] |= not job.close()
                except Exception:
                    result["cleanup_failed"] = True
            for reader, key in ((stdout, "stdout"), (stderr, "stderr")):
                if reader is not None:
                    try:
                        output, finished = reader.finish()
                        result[key] = result[key] + output
                        result["output_truncated"] |= reader.truncated
                        result["cleanup_failed"] |= not finished
                    except Exception:
                        result["cleanup_failed"] = True
            if process is not None:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if any(
                        reader is not None and reader.pipe is stream and reader.thread.is_alive()
                        for reader in (stdout, stderr)
                    ):
                        # A redirected handle unexpectedly held outside the job
                        # must not make stream.close() block the caller forever.
                        continue
                    if stream is not None:
                        try:
                            stream.close()
                        except (OSError, ValueError):
                            result["cleanup_failed"] = True
            result["cancelled"] |= execution.cancelled.is_set()
            if result["cancelled"]:
                result["exit_code"] = 130
            elif result["timed_out"]:
                result["exit_code"] = 124
            if result["cleanup_failed"]:
                result["stderr"] += (
                    "\nProcess-tree cleanup was not confirmed; check local processes."
                )
            result["success"] = (
                result["exit_code"] == 0
                and not result["timed_out"]
                and not result["cancelled"]
                and not result["cleanup_failed"]
            )
            with self._lock:
                self._active.pop(execution_id, None)
            execution.done.set()
        return result


_executor: HostCommandExecutor | None = None
_executor_lock = threading.Lock()


def get_host_executor() -> HostCommandExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = HostCommandExecutor()
            atexit.register(_executor.shutdown)
        return _executor
