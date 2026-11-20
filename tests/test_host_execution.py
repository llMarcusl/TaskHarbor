"""Host execution safety checks; real commands touch only pytest directories."""

from __future__ import annotations

import asyncio
import base64
import ctypes
import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from taskharbor.sandbox.manager import current_thread_id, current_user_id
from taskharbor.security import host
from taskharbor.tools import execute_host_command as host_tool


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    value = {
        "command": "Write-Output 'ok'",
        "cwd": str(tmp_path),
        "timeout": 2,
        "shell": "powershell",
        "powershell_path": r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        "powershell_identity": {"sha256": "frozen"},
    }
    monkeypatch.setattr(host, "prepare_host_command", lambda *args, **kwargs: dict(value))
    return value


def test_environment_does_not_inherit_backend_credentials(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL_API_KEY", "private-model-key")
    monkeypatch.setenv("DATABASE_URL", "private-database-url")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "private-session-token")
    monkeypatch.setenv("MY_PLUGIN_SECRET", "private-plugin-key")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    environment = host._minimal_environment()
    assert {key.upper(): value for key, value in environment.items()}["SYSTEMROOT"] == r"C:\Windows"
    assert environment["PYTHONUTF8"] == "1"
    assert not {"CHAT_MODEL_API_KEY", "DATABASE_URL", "AWS_SESSION_TOKEN", "MY_PLUGIN_SECRET"} & {
        key.upper() for key in environment
    }


def test_command_is_one_utf16_encoded_argument(prepared):
    command = "Write-Output '中文'; Write-Output '$(`n) & special'"
    prepared["command"] = command
    argv = host._argv(prepared)
    assert argv[:4] == [prepared["powershell_path"], "-NoLogo", "-NoProfile", "-NonInteractive"]
    assert argv[4] == "-EncodedCommand"
    assert len(argv) == 6
    assert command in base64.b64decode(argv[5]).decode("utf-16-le")
    assert "cmd.exe" not in argv


def test_reader_drains_and_bounds_both_memory_and_output():
    reader = host._BoundedReader(io.BytesIO(b"x" * 100_000), 1024)
    output, finished = reader.finish()
    assert finished and reader.truncated
    assert len(output) == 1024


def test_utf8_truncation_does_not_redecode_entire_chinese_output():
    value = "中文输出".encode()
    assert host._decode_output(value[:-1]) == "中文输\ufffd"


def test_failed_job_assignment_never_resumes_user_script(prepared, monkeypatch):
    process = Mock()
    process.poll.return_value = None
    process.stdin = None
    process.stdout = io.BytesIO()
    process.stderr = io.BytesIO()
    process.returncode = 1
    job = Mock()
    job.assign_and_resume.side_effect = OSError("job assignment denied")
    job.close.return_value = True
    monkeypatch.setattr(host, "_WindowsJob", lambda: job)
    popen = Mock(return_value=process)
    monkeypatch.setattr(host.subprocess, "Popen", popen)
    result = host.HostCommandExecutor().run("ignored", prepared["cwd"], scope="user:thread")
    assert not result["success"]
    assert "job assignment denied" in result["stderr"]
    process.kill.assert_called_once()
    job.terminate_and_wait.assert_not_called()
    kwargs = popen.call_args.kwargs
    assert kwargs["shell"] is False
    assert kwargs["creationflags"] & host.CREATE_SUSPENDED
    assert kwargs["creationflags"] & host.CREATE_NO_WINDOW
    assert kwargs["stdin"] == host.subprocess.DEVNULL


def test_cleanup_failure_is_not_reported_as_success(prepared, monkeypatch):
    process = Mock()
    process.poll.return_value = 0
    process.returncode = 0
    process.stdin = None
    process.stdout = io.BytesIO(b"ok")
    process.stderr = io.BytesIO()
    job = Mock()
    job.terminate_and_wait.return_value = False
    job.close.return_value = True
    monkeypatch.setattr(host, "_WindowsJob", lambda: job)
    monkeypatch.setattr(host.subprocess, "Popen", lambda *args, **kwargs: process)
    result = host.HostCommandExecutor().run("ignored", prepared["cwd"], scope="user:thread")
    assert result["cleanup_failed"]
    assert not result["success"]
    assert "not confirmed" in result["stderr"]
    assert result["stdout"] == "ok"


def test_cleanup_exception_still_closes_job_and_releases_tracking(prepared, monkeypatch):
    process = Mock()
    process.poll.return_value = 0
    process.returncode = 0
    process.stdin = None
    process.stdout = io.BytesIO()
    process.stderr = io.BytesIO()
    job = Mock()
    job.terminate_and_wait.side_effect = OSError("job cleanup error")
    job.close.return_value = True
    monkeypatch.setattr(host, "_WindowsJob", lambda: job)
    monkeypatch.setattr(host.subprocess, "Popen", lambda *args, **kwargs: process)
    executor = host.HostCommandExecutor()
    result = executor.run("ignored", prepared["cwd"], scope="user:thread")
    assert result["cleanup_failed"] and not result["success"]
    job.close.assert_called_once()
    assert not executor._active


def test_cancelled_before_worker_registration_does_not_launch(prepared, monkeypatch):
    cancelled = threading.Event()
    cancelled.set()
    job = Mock()
    job.close.return_value = True
    monkeypatch.setattr(host, "_WindowsJob", lambda: job)
    popen = Mock(side_effect=AssertionError("cancelled call must not launch"))
    monkeypatch.setattr(host.subprocess, "Popen", popen)
    result = host.HostCommandExecutor().run(
        "ignored", prepared["cwd"], scope="user:thread", _cancelled=cancelled
    )
    assert result["cancelled"] and result["exit_code"] == 130
    popen.assert_not_called()


def test_tool_rejects_disabled_and_missing_trusted_identity(monkeypatch):
    controller = Mock()
    tool = host_tool.create_execute_host_command_tool(controller)
    monkeypatch.setattr(
        host_tool, "get_settings", lambda: SimpleNamespace(HOST_COMMANDS_ENABLED=False)
    )
    result = json.loads(tool.invoke({"command": "Get-Location", "cwd": "C:/"}))
    assert result["error"]["type"] == "PermissionError"
    monkeypatch.setattr(
        host_tool, "get_settings", lambda: SimpleNamespace(HOST_COMMANDS_ENABLED=True)
    )
    owner_token, thread_token = current_user_id.set(None), current_thread_id.set(None)
    try:
        result = json.loads(tool.invoke({"command": "Get-Location", "cwd": "C:/"}))
    finally:
        current_user_id.reset(owner_token)
        current_thread_id.reset(thread_token)
    assert "trusted user" in result["error"]["message"]
    controller.run.assert_not_called()


def test_tool_uses_owner_and_thread_scope(monkeypatch):
    controller = Mock()
    controller.run.return_value = {"success": True, "stdout": "中文"}
    monkeypatch.setattr(
        host_tool, "get_settings", lambda: SimpleNamespace(HOST_COMMANDS_ENABLED=True)
    )
    tool = host_tool.create_execute_host_command_tool(controller)
    owner_token, thread_token = current_user_id.set("owner"), current_thread_id.set("thread")
    try:
        result = json.loads(tool.invoke({"command": "Get-Location", "cwd": "C:/", "timeout": 4}))
    finally:
        current_user_id.reset(owner_token)
        current_thread_id.reset(thread_token)
    assert result["stdout"] == "中文"
    controller.run.assert_called_once_with(
        "Get-Location", "C:/", 4, scope="owner:thread"
    )


def test_async_tool_cancellation_signals_not_yet_registered_worker(monkeypatch):
    controller = Mock()
    captured = {}
    monkeypatch.setattr(
        host_tool, "get_settings", lambda: SimpleNamespace(HOST_COMMANDS_ENABLED=True)
    )

    async def delayed_thread(func, *args, **kwargs):
        captured.update(kwargs)
        while not kwargs["_cancelled"].is_set():
            await asyncio.sleep(0)
        return {"cancelled": True, "cleanup_failed": False, "exit_code": 130}

    monkeypatch.setattr(host_tool.asyncio, "to_thread", delayed_thread)
    tool = host_tool.create_execute_host_command_tool(controller)

    async def scenario():
        task = asyncio.create_task(tool.ainvoke({"command": "Get-Location", "cwd": "C:/"}))
        for _ in range(20):
            if captured:
                break
            await asyncio.sleep(0)
        assert captured
        task.cancel()
        with pytest.raises(host.HostCommandCancelled) as raised:
            await task
        assert raised.value.result == {
            "cancelled": True, "cleanup_failed": False, "exit_code": 130
        }

    owner_token, thread_token = current_user_id.set("owner"), current_thread_id.set("thread")
    try:
        asyncio.run(scenario())
    finally:
        current_user_id.reset(owner_token)
        current_thread_id.reset(thread_token)
    assert captured["_cancelled"].is_set()
    controller.cancel_scope.assert_called_once_with("owner:thread")


def test_async_tool_cancellation_preserves_cleanup_failure_result(monkeypatch):
    controller = Mock()
    started = threading.Event()
    expected = {"cancelled": True, "cleanup_failed": True, "exit_code": 130}

    def run(*args, **kwargs):
        started.set()
        assert kwargs["_cancelled"].wait(timeout=3)
        return expected

    controller.run.side_effect = run
    monkeypatch.setattr(
        host_tool, "get_settings", lambda: SimpleNamespace(HOST_COMMANDS_ENABLED=True)
    )
    tool = host_tool.create_execute_host_command_tool(controller)

    async def scenario():
        task = asyncio.create_task(tool.ainvoke({"command": "Get-Location", "cwd": "C:/"}))
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        with pytest.raises(host.HostCommandCancelled) as raised:
            await task
        assert raised.value.result is expected

    owner_token, thread_token = current_user_id.set("owner"), current_thread_id.set("thread")
    try:
        asyncio.run(scenario())
    finally:
        current_user_id.reset(owner_token)
        current_thread_id.reset(thread_token)
    assert controller.run.call_count == 1


def test_async_tool_cleanup_wait_is_bounded_and_still_cancels(monkeypatch, caplog):
    controller = Mock()
    captured = {}
    monkeypatch.setattr(
        host_tool, "get_settings", lambda: SimpleNamespace(HOST_COMMANDS_ENABLED=True)
    )
    monkeypatch.setattr(host_tool, "CANCEL_CLEANUP_WAIT_SECONDS", 0.01)

    async def stalled_worker(func, *args, **kwargs):
        captured.update(kwargs)
        await asyncio.Future()

    monkeypatch.setattr(host_tool.asyncio, "to_thread", stalled_worker)
    tool = host_tool.create_execute_host_command_tool(controller)

    async def scenario():
        task = asyncio.create_task(tool.ainvoke({"command": "Get-Location", "cwd": "C:/"}))
        for _ in range(20):
            if captured:
                break
            await asyncio.sleep(0)
        assert captured
        task.cancel()
        with pytest.raises(host.HostCommandCancelled) as raised:
            await asyncio.wait_for(task, timeout=1)
        assert raised.value.result is None

    owner_token, thread_token = current_user_id.set("owner"), current_thread_id.set("thread")
    try:
        asyncio.run(scenario())
    finally:
        current_user_id.reset(owner_token)
        current_thread_id.reset(thread_token)
    assert captured["_cancelled"].is_set()
    assert "cleanup result unavailable" in caplog.text


windows_only = pytest.mark.skipif(os.name != "nt", reason="Windows host commands")


@windows_only
def test_prepare_validates_input_and_pins_interpreter(tmp_path):
    prepared = host.prepare_host_command("Get-Location", str(tmp_path))
    assert Path(prepared["powershell_path"]).is_absolute()
    assert prepared["powershell_identity"]["sha256"]
    assert prepared["shell"] == "powershell"
    for cwd in (".", "/workspace", str(tmp_path / "missing")):
        with pytest.raises((ValueError, FileNotFoundError)):
            host.prepare_host_command("Get-Location", cwd)
    for timeout in (0, 301, True, float("nan")):
        with pytest.raises(ValueError):
            host.prepare_host_command("Get-Location", str(tmp_path), timeout)
    with pytest.raises(ValueError):
        host.prepare_host_command("x" * 20_000, str(tmp_path))


@windows_only
def test_interpreter_replacement_requires_service_restart(tmp_path, monkeypatch):
    executable = tmp_path / "powershell.exe"
    executable.write_bytes(b"original executable identity")
    identity = host._file_identity(executable)
    monkeypatch.setattr(host, "_powershell_installation", lambda: (str(executable), identity))
    executable.write_bytes(b"different executable identity")
    with pytest.raises(RuntimeError, match="interpreter changed"):
        host.prepare_host_command("Get-Location", str(tmp_path))


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


@windows_only
def test_real_windows_chinese_output_cwd_and_child_encoding(tmp_path, monkeypatch):
    directory = tmp_path / "中文 空格"
    directory.mkdir()
    child = directory / "子进程.py"
    child.write_text('print("子进程中文")', encoding="utf-8")
    monkeypatch.setenv("CHAT_MODEL_API_KEY", "must-not-be-inherited")
    script = (
        "Write-Output '中文输出'; (Get-Location).Path; "
        f"& {_ps_quote(sys.executable)} {_ps_quote(str(child))}; "
        "Write-Output ('secret_present=' + [bool]$env:CHAT_MODEL_API_KEY)"
    )
    result = host.HostCommandExecutor().run(script, str(directory), 10, scope="test:unicode")
    assert result["success"], result
    assert "中文输出" in result["stdout"] and "子进程中文" in result["stdout"]
    assert str(directory) in result["stdout"]
    assert "secret_present=False" in result["stdout"]


@windows_only
def test_real_windows_output_bound_and_native_exit_code(tmp_path):
    executor = host.HostCommandExecutor(max_output_bytes=2048)
    result = executor.run("Write-Output ('x' * 100000)", str(tmp_path), 10, scope="test:bound")
    assert result["success"] and result["output_truncated"], result
    assert len(result["stdout"].encode()) == 2048
    result = executor.run("exit 7", str(tmp_path), 10, scope="test:exit")
    assert result["exit_code"] == 7 and not result["success"], result


def _process_is_running(pid: int) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        assert kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        return code.value == 259
    finally:
        kernel32.CloseHandle(handle)


def _child_script(tmp_path: Path, *, keep_parent: bool) -> str:
    executable = host.prepare_host_command("Get-Location", str(tmp_path))["powershell_path"]
    payload = base64.b64encode("Start-Sleep -Seconds 30".encode("utf-16-le")).decode()
    command = (
        f"$child = Start-Process -FilePath {_ps_quote(executable)} -WindowStyle Hidden "
        f"-ArgumentList @('-NoProfile','-NonInteractive','-EncodedCommand','{payload}') -PassThru; "
        f"[System.IO.File]::WriteAllText({_ps_quote(str(tmp_path / 'child.pid'))}, "
        "[string]$child.Id); "
    )
    return command + (
        "Start-Sleep -Seconds 30" if keep_parent else "Write-Output 'parent finished'"
    )


@windows_only
@pytest.mark.parametrize("keep_parent", [False, True])
def test_real_windows_descendants_cleaned_after_exit_or_timeout(tmp_path, keep_parent):
    command = _child_script(tmp_path, keep_parent=keep_parent)
    # Cold PowerShell startup can exceed two seconds on loaded Windows hosts.
    # Leave enough startup headroom while still timing out the 30-second child.
    result = host.HostCommandExecutor().run(command, str(tmp_path), 5, scope="test:tree")
    assert (tmp_path / "child.pid").exists(), result
    pid = int((tmp_path / "child.pid").read_text())
    assert not result["cleanup_failed"], result
    assert result["timed_out"] is keep_parent, result
    assert not _process_is_running(pid)


@windows_only
def test_real_windows_cancel_scope_and_executor_reusable(tmp_path):
    executor = host.HostCommandExecutor()
    result = {}
    command = _child_script(tmp_path, keep_parent=True)
    worker = threading.Thread(
        target=lambda: result.update(executor.run(command, str(tmp_path), 20, scope="test:cancel"))
    )
    worker.start()
    deadline = time.monotonic() + 10
    try:
        while not (tmp_path / "child.pid").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (tmp_path / "child.pid").exists()
        assert not executor.cancel_scope("other:thread")
        assert executor.cancel_scope("test:cancel")
    finally:
        executor.cancel_scope("test:cancel")
        worker.join(timeout=10)
    assert not worker.is_alive()
    assert result["cancelled"] and result["exit_code"] == 130, result
    assert not result["cleanup_failed"], result
    assert not _process_is_running(int((tmp_path / "child.pid").read_text()))
    assert executor.run("Write-Output 'ok'", str(tmp_path), 10, scope="test:again")["success"]


@windows_only
def test_real_windows_async_cancel_preserves_result_and_kills_descendants(tmp_path, monkeypatch):
    executor = host.HostCommandExecutor()
    monkeypatch.setattr(
        host_tool, "get_settings", lambda: SimpleNamespace(HOST_COMMANDS_ENABLED=True)
    )
    tool = host_tool.create_execute_host_command_tool(executor)

    async def scenario():
        task = asyncio.create_task(
            tool.ainvoke({
                "command": _child_script(tmp_path, keep_parent=True),
                "cwd": str(tmp_path),
                "timeout": 20,
            })
        )
        deadline = time.monotonic() + 10
        while not (tmp_path / "child.pid").exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert (tmp_path / "child.pid").exists()
        task.cancel()
        with pytest.raises(host.HostCommandCancelled) as raised:
            await asyncio.wait_for(task, timeout=15)
        return raised.value.result

    owner_token, thread_token = current_user_id.set("owner"), current_thread_id.set("thread")
    try:
        result = asyncio.run(scenario())
    finally:
        current_user_id.reset(owner_token)
        current_thread_id.reset(thread_token)
        executor.shutdown()
    assert result is not None
    assert result["cancelled"] and result["exit_code"] == 130, result
    assert not result["cleanup_failed"], result
    assert not _process_is_running(int((tmp_path / "child.pid").read_text()))
    assert not executor._active


@windows_only
def test_real_windows_shutdown_cancels_commands(tmp_path):
    executor = host.HostCommandExecutor()
    result = {}
    command = (
        f"[IO.File]::WriteAllText({_ps_quote(str(tmp_path / 'started'))}, 'yes'); Start-Sleep 30"
    )
    worker = threading.Thread(
        target=lambda: result.update(
            executor.run(command, str(tmp_path), 20, scope="test:shutdown")
        )
    )
    worker.start()
    deadline = time.monotonic() + 10
    try:
        while not (tmp_path / "started").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (tmp_path / "started").exists()
    finally:
        executor.shutdown()
        worker.join(timeout=10)
    assert result["cancelled"] and not result["cleanup_failed"], result
    with pytest.raises(RuntimeError, match="shutting down"):
        executor.run("Get-Location", str(tmp_path), scope="test:closed")
