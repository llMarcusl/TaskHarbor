"""Terminal controller checks; enable Docker cases with SANDBOX_DOCKER_TESTS=1."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from taskharbor.sandbox.execution import (
    EXECUTOR_VERSION,
    TerminalExecutor,
    _BoundedReader,
    _Execution,
)


def test_output_reader_drains_large_process_output_without_retaining_it():
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 3000000)"],
        stdout=subprocess.PIPE,
    )
    reader = _BoundedReader(process.stdout, 1024)
    process.wait(timeout=10)
    assert reader.finish() == "x" * 1024
    assert reader.truncated
    process.stdout.close()


def test_cancel_selects_execution_or_all_executions_in_session():
    executor = TerminalExecutor()
    first = _Execution("a", None)
    second = _Execution("a", None)
    unrelated = _Execution("b", None)
    executor._active.update(first=first, second=second, unrelated=unrelated)
    assert executor.cancel("first")
    assert first.cancelled.is_set()
    assert not second.cancelled.is_set()
    assert executor.cancel("a")
    assert second.cancelled.is_set()
    assert not unrelated.cancelled.is_set()
    assert not executor.cancel("missing")


@pytest.mark.parametrize("timeout", [0, -1])
def test_run_rejects_invalid_timeout_before_launch(timeout):
    with pytest.raises(ValueError, match="timeout"):
        TerminalExecutor().run(None, "true", session_id="test", timeout=timeout)


def test_missing_docker_client_returns_transport_error(monkeypatch):
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("docker executable unavailable")

    monkeypatch.setattr(subprocess, "Popen", missing)
    executor = TerminalExecutor()
    result = executor.run(SimpleNamespace(id="container"), "true", session_id="a", timeout=1)
    assert result.exit_code == 1
    assert "unavailable" in result.stderr
    assert not result.cleanup_failed  # No command could have started.
    assert not executor.cancel("a")


@pytest.fixture(scope="module", params=["python", "node"])
def execution_container(request):
    if os.environ.get("SANDBOX_DOCKER_TESTS") != "1":
        pytest.skip("Set SANDBOX_DOCKER_TESTS=1 to run real container process checks")
    import docker

    client = docker.from_env()
    client.ping()
    image_name = f"sandbox-{request.param}:latest"
    image = client.images.get(image_name)
    assert image.labels.get("sandbox-executor-version") == EXECUTOR_VERSION, f"Rebuild {image_name}"
    container = client.containers.run(
        image_name,
        command=["sleep", "infinity"],
        detach=True,
        network_disabled=True,
        mem_limit="256m",
        pids_limit=64,
        cap_drop=["ALL"],
        security_opt=["no-new-privileges:true"],
    )
    try:
        yield container
    finally:
        container.remove(force=True)
        client.close()


def _python_command(source: str) -> str:
    return "python3 -c " + shlex.quote(source)


def _read(container, path: str) -> str:
    result = container.exec_run(["cat", path])
    return result.output.decode("utf-8") if result.exit_code == 0 else ""


def _assert_no_live_pid(container, pid: int):
    result = container.exec_run(
        [
            "python3",
            "-c",
            "import pathlib,sys; p=pathlib.Path('/proc')/sys.argv[1]/'stat'; "
            "print(p.read_text().rsplit(')',1)[1].split()[0] if p.exists() else 'gone')",
            str(pid),
        ]
    )
    assert result.output.decode().strip() in {"gone", "Z"}


def test_docker_terminal_unicode_nonzero_and_bounded_output(execution_container):
    executor = TerminalExecutor(max_output_bytes=1024)
    result = executor.run(
        execution_container,
        _python_command(
            "import sys; print('你好'); print('problem', file=sys.stderr); sys.exit(7)"
        ),
        session_id="unicode",
        timeout=5,
    )
    assert result.exit_code == 7
    assert "你好" in result.stdout
    assert "problem" in result.stderr
    assert not result.cleanup_failed
    result = executor.run(
        execution_container,
        _python_command("print('x' * 2000000)"),
        session_id="output",
        timeout=5,
    )
    assert result.exit_code == 0
    assert result.output_truncated
    assert len(result.stdout.encode()) <= 1024


def test_docker_timeout_kills_child_ignoring_sigterm(execution_container):
    command = _python_command(
        "import os,signal,time; "
        "open('/workspace/timeout.pid','w').write(str(os.getpid())); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    )
    started = time.monotonic()
    result = TerminalExecutor().run(
        execution_container, command, session_id="timeout", timeout=0.5
    )
    assert time.monotonic() - started < 8
    assert result.timed_out
    assert result.exit_code == 124
    assert not result.cleanup_failed
    pid = int(_read(execution_container, "/workspace/timeout.pid"))
    _assert_no_live_pid(execution_container, pid)


def test_docker_cancel_waits_for_process_cleanup(execution_container):
    executor = TerminalExecutor()
    results = []
    command = _python_command(
        "import os,time; open('/workspace/cancel.pid','w').write(str(os.getpid())); "
        "time.sleep(60)"
    )
    worker = threading.Thread(
        target=lambda: results.append(
            executor.run(execution_container, command, session_id="cancel-me", timeout=60)
        )
    )
    worker.start()
    deadline = time.monotonic() + 8
    pid_text = ""
    while time.monotonic() < deadline:
        pid_text = _read(execution_container, "/workspace/cancel.pid")
        if pid_text:
            break
        time.sleep(0.05)
    try:
        assert pid_text, "Command did not start"
        assert executor.cancel("cancel-me")
        worker.join(timeout=8)
        assert not worker.is_alive()
        assert results[0].cancelled
        assert results[0].exit_code == 130
        assert not results[0].cleanup_failed
        _assert_no_live_pid(execution_container, int(pid_text))
    finally:
        executor.cancel("cancel-me")
        worker.join(timeout=8)


def test_docker_normal_return_kills_detached_background_writer(execution_container):
    child = (
        "import os,time; "
        "open('/workspace/background.pid','w').write(str(os.getpid())); "
        "exec(\"while True:\\n "
        "open('/workspace/background.txt','a').write('x'); time.sleep(0.02)\")"
    )
    parent = (
        "import pathlib,subprocess,time; "
        f"subprocess.Popen(['python3','-c',{child!r}], start_new_session=True); "
        "exec(\"while not pathlib.Path('/workspace/background.pid').exists():\\n "
        "time.sleep(0.01)\")"
    )
    result = TerminalExecutor().run(
        execution_container, _python_command(parent), session_id="background", timeout=5
    )
    assert result.exit_code == 0
    assert not result.cleanup_failed
    pid = int(_read(execution_container, "/workspace/background.pid"))
    _assert_no_live_pid(execution_container, pid)
    before = _read(execution_container, "/workspace/background.txt")
    time.sleep(0.2)
    assert _read(execution_container, "/workspace/background.txt") == before


def test_docker_helper_launch_failure_is_explicit(execution_container):
    # A missing helper must never result in a successful response or reuse.
    class MissingHelper(TerminalExecutor):
        @staticmethod
        def _command(container, args, *, user, workdir):
            command = TerminalExecutor._command(container, args, user=user, workdir=workdir)
            return [
                part.replace("/client/execute_terminal.py", "/missing-helper.py")
                for part in command
            ]

    result = MissingHelper().run(execution_container, "true", session_id="missing", timeout=5)
    assert result.exit_code != 0
    assert result.cleanup_failed
    assert "missing-helper" in result.stderr
