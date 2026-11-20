"""Linux container-side terminal supervisor (stdlib only).

Each invocation adopts orphaned descendants, drains bounded output and kills
remaining children before returning. This is process cleanup, not an additional
security boundary against code already running with the same container UID.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import selectors
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
GRACE_SECONDS = 0.5


def _directory(execution_id: str) -> Path:
    if not ID_PATTERN.fullmatch(execution_id):
        raise ValueError("Invalid execution ID")
    return Path("/tmp") / f"sandbox-exec-{execution_id}"


def cancel(execution_id: str) -> int:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(1)
        try:
            client.connect(str(_directory(execution_id) / "control.sock"))
            client.sendall(b"cancel")
            return 0
        except OSError:
            return 1


def _descendants() -> set[int]:
    """Find children including double-forked/setsid processes adopted by us."""
    parents = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # comm can contain spaces and parentheses; fields after its last ')'.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z":
                parents[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    descendants = {os.getpid()}
    while True:
        found = {pid for pid, ppid in parents.items() if ppid in descendants}
        if found <= descendants:
            return descendants - {os.getpid()}
        descendants |= found


def _signal_children(signum: int) -> None:
    for pid in _descendants():
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


def execute(command: str, execution_id: str, timeout: float, limit: int) -> dict:
    if timeout <= 0 or limit <= 0:
        raise ValueError("timeout and output limit must be positive")
    # An orphaned daemon becomes our child rather than escaping to container PID 1.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Unable to enable child subreaper")
    directory = _directory(execution_id)
    directory.mkdir(mode=0o700)
    control_path = directory / "control.sock"
    selector = selectors.DefaultSelector()
    control = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    process = None
    cancelled = False
    timed_out = False
    terminating_at = None
    output = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = False
    old_handlers = {}

    def request_cancel(_signum, _frame):
        nonlocal cancelled
        cancelled = True

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, request_cancel)
        control.bind(str(control_path))
        control.listen(2)
        control.setblocking(False)
        selector.register(control, selectors.EVENT_READ, "control")
        process = subprocess.Popen(
            ["sh", "-c", command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        for name in ("stdout", "stderr"):
            stream = getattr(process, name)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        deadline = time.monotonic() + timeout
        while True:
            now = time.monotonic()
            done = process.poll() is not None
            timed_out |= not done and now >= deadline
            if terminating_at is None and (done or timed_out or cancelled):
                terminating_at = now
                _signal_children(signal.SIGTERM)
            if terminating_at is not None and now - terminating_at >= GRACE_SECONDS:
                _signal_children(signal.SIGKILL)
            for key, _ in selector.select(timeout=0.05):
                if key.data == "control":
                    connection, _ = control.accept()
                    # Any connection on this per-execution socket requests cancel.
                    connection.close()
                    cancelled = True
                    continue
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                remaining = max(0, limit - sum(map(len, output.values())))
                output[key.data].extend(chunk[:remaining])
                truncated |= len(chunk) > remaining
            if terminating_at is not None:
                if not _descendants() and process.poll() is not None:
                    # Drain EOF/data once more; no writer remains to keep pipes open.
                    for key in list(selector.get_map().values()):
                        if key.data == "control":
                            continue
                        while chunk := os.read(key.fileobj.fileno(), 65536):
                            remaining = max(0, limit - sum(map(len, output.values())))
                            output[key.data].extend(chunk[:remaining])
                            truncated |= len(chunk) > remaining
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    break
                if now - terminating_at > GRACE_SECONDS + 2:
                    break
        cleanup_failed = bool(_descendants())
        exit_code = process.poll()
        return {
            "exit_code": 130 if cancelled else 124 if timed_out else (exit_code or 0),
            "stdout": output["stdout"].decode("utf-8", errors="replace"),
            "stderr": output["stderr"].decode("utf-8", errors="replace"),
            "timed_out": timed_out,
            "cancelled": cancelled,
            "output_truncated": truncated,
            "cleanup_failed": cleanup_failed,
        }
    finally:
        if process is not None:
            _signal_children(signal.SIGKILL)
        selector.close()
        control.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        control_path.unlink(missing_ok=True)
        directory.rmdir()


def main() -> int:
    try:
        action, execution_id = sys.argv[1:3]
        if action == "cancel":
            return cancel(execution_id)
        if action != "run":
            raise ValueError("Invalid action")
        request = json.loads(sys.stdin.buffer.readline(4 * 1024 * 1024))
        result = execute(
            request["command"], execution_id, float(sys.argv[3]), int(sys.argv[4])
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"exit_code": 1, "stderr": str(exc), "cleanup_failed": True}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
