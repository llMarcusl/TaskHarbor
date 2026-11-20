"""Exercise host/kernel Unicode transport without Docker."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from taskharbor.sandbox.manager import SandboxManager


@pytest.mark.parametrize(
    "code, response",
    [
        ("# 中文注释\nprint(1)", {"success": True, "stdout": "1\n"}),
        ('print("你好，世界！😊")', {"success": True, "stdout": "你好，世界！😊\n"}),
        (
            'raise ValueError("中文异常")',
            {"success": False, "error": {"type": "ValueError", "message": "中文异常"}},
        ),
    ],
)
def test_kernel_transport_uses_utf8(monkeypatch, code, response):
    manager = object.__new__(SandboxManager)
    manager.sessions = {"session": SimpleNamespace(runtime="python", container_name="sandbox")}
    monkeypatch.setattr(manager, "_assert_container_alive", lambda _: None)
    original_run = subprocess.run
    payload = {"action": "execute", "code": code}

    # Replace only Docker with a local peer. Real subprocess text encoding still
    # handles stdin/stdout; the peer uses the kernel's UTF-8 JSON wire format.
    peer = (
        "import json, sys; "
        "request = json.loads(sys.stdin.buffer.read()); "
        f"assert request == {payload!r}; "
        f"sys.stdout.buffer.write({json.dumps(response, ensure_ascii=False).encode('utf-8')!r})"
    )

    def run_peer(command, **kwargs):
        assert command[:3] == ["docker", "exec", "-i"]
        assert kwargs["encoding"] == "utf-8"
        return original_run([sys.executable, "-c", peer], **kwargs)

    monkeypatch.setattr(subprocess, "run", run_peer)
    assert manager._send_to_kernel("session", payload) == response
