"""Kernel time limits must trigger container destruction, not leave child jobs."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from taskharbor.sandbox.manager import SandboxManager
from tests import test_sandbox_acceptance as acceptance


@pytest.fixture
def kernel_sandbox(tmp_path, monkeypatch):
    yield from acceptance.real_sandbox.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "error_type,message,should_stop",
    [
        ("TimeoutError", "Execution exceeded deadline", True),
        ("Error", "A business operation timed out", False),
        ("ValueError", "Invalid timeout setting", False),
    ],
)
def test_manager_destroys_on_normalized_timeout_only(
    monkeypatch, error_type, message, should_stop
):
    import taskharbor.sandbox.manager as module

    settings = SimpleNamespace(
        CONTAINER_EXECUTION_TIMEOUT_SECONDS=1,
        MAX_STDOUT_CHARS=1000,
        MAX_STDERR_CHARS=1000,
        MAX_RESULT_CHARS=1000,
    )
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    manager = object.__new__(SandboxManager)
    manager._check_session = lambda _: None
    manager._touch_session = lambda _: None
    manager._send_to_kernel = lambda *_args, **_kwargs: {
        "success": False, "error": {"type": error_type, "message": message}
    }
    manager.stop_session = MagicMock()
    result = SandboxManager.execute_code.__wrapped__(manager, "session", "code", timeout=1)
    assert result.error["type"] == error_type
    assert manager.stop_session.called is should_stop


@pytest.mark.skipif(
    os.environ.get("SANDBOX_DOCKER_TESTS") != "1", reason="opt-in Docker Node timeout check"
)
@pytest.mark.parametrize("completion", ["while (true) {}", "new Promise(() => {})"])
def test_node_timeout_destroys_container_with_detached_child(kernel_sandbox, completion):
    manager = kernel_sandbox
    info = manager.create_session(runtime="node")
    child_command = (
        "trap '' TERM; echo $$ > /workspace/node-timeout.pid; "
        "while :; do sleep 1; done"
    )
    code = (
        "require('child_process').spawn('sh', ['-c', " + repr(child_command) + "], "
        "{detached: true, stdio: 'ignore'}); " + completion
    )
    result = manager.execute_code(info.session_id, code, timeout=1)
    assert not result.success
    assert result.error["type"] == "TimeoutError"
    assert info.session_id not in manager.sessions
    assert Path(info.workspace, "node-timeout.pid").read_text().strip().isdigit()
    # Deleting the execution container also deletes its entire process tree.
    import docker.errors

    with pytest.raises(docker.errors.NotFound):
        manager.client.containers.get(info.container_id)


def test_r_actual_elapsed_limit_is_normalized_when_r_is_available():
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is unavailable; R image build is intentionally excluded")
    kernel = (
        Path(__file__).parents[1]
        / "src" / "taskharbor" / "docker" / "kernel" / "kernel_r.R"
    )
    # Evaluate only the production error-normalizer definition; importing the
    # entire persistent kernel would start its listening loop.
    source = r"""
expressions <- as.list(parse(file=commandArgs(TRUE)[1L]))
definition <- Filter(function(expr) {
  is.call(expr) && identical(expr[[1L]], as.name("<-")) &&
    identical(expr[[2L]], as.name("execution_error_type"))
}, expressions)
stopifnot(length(definition) == 1L)
eval(definition[[1L]])
stopifnot(execution_error_type(simpleError("unrelated timeout")) == "simpleError")
error <- tryCatch({
  setTimeLimit(elapsed=0.01, transient=TRUE)
  repeat { for (i in 1:100000) sqrt(i) }
}, error=identity)
setTimeLimit(elapsed=Inf)
stopifnot(execution_error_type(error) == "TimeoutError")
"""
    result = subprocess.run(
        [rscript, "--vanilla", "-e", source, str(kernel)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
