"""POSIX sandbox references map to scoped durable publications on Windows too."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from taskharbor.sandbox.manager import SandboxManager
from taskharbor.sandbox.storage import WorkspaceStore


@pytest.fixture
def registry_manager(tmp_path):
    manager = object.__new__(SandboxManager)
    manager._storage = WorkspaceStore(tmp_path)
    manager.resolve_owner = lambda thread: "alice"
    scope = manager._storage.scope("alice", "t")
    reports = scope.workspace / "reports"
    reports.mkdir()
    (reports / "a.txt").write_text("file-content", encoding="utf-8")
    manager._storage.publish(scope, "s", "/workspace/reports")
    return manager


@pytest.mark.parametrize(
    "path", ["/workspace/reports", "/workspace/reports/a.txt", "reports/a.txt"]
)
def test_download_permission_uses_container_paths(registry_manager, monkeypatch, path):
    def host_resolution_forbidden(*args, **kwargs):
        raise AssertionError("container paths must never use host Path.resolve")

    monkeypatch.setattr(Path, "resolve", host_resolution_forbidden)
    assert registry_manager.is_file_exported("t", "s", path)


def test_download_permission_retains_scope_and_directory_boundaries(registry_manager):
    manager = registry_manager
    assert not manager.is_file_exported("other-thread", "s", "/workspace/reports/a.txt")
    assert not manager.is_file_exported("t", "other-session", "/workspace/reports/a.txt")
    assert not manager.is_file_exported("t", "s", "/workspace/reports-private/a.txt")
    assert not manager.is_file_exported("t", "s", "/workspace/reports/../private.txt")
    with pytest.raises(ValueError, match="Path outside /workspace"):
        manager.is_file_exported("t", "s", "/workspace/../etc/passwd")


def test_directory_stream_is_persisted_zip_without_docker(registry_manager):
    with patch("taskharbor.sandbox.manager.subprocess.run") as run:
        content = b"".join(registry_manager.stream_exported_file("t", "s", "/workspace/reports"))
    run.assert_not_called()
    archive = zipfile.ZipFile(io.BytesIO(content))
    assert archive.read("reports/a.txt") == b"file-content"


def test_http_download_with_real_publication_permission(registry_manager, monkeypatch):
    from taskharbor.http_app import app

    monkeypatch.setattr(
        "aegra_api.core.auth_deps.require_auth",
        AsyncMock(return_value=SimpleNamespace(identity="alice")),
    )
    monkeypatch.setattr(
        "taskharbor.settings.get_settings",
        lambda: SimpleNamespace(
            SESSION_MEMORY_ENABLED=False,
            LONG_TERM_MEMORY_ENABLED=False,
            CONTEXT_COMPACTION_ENABLED=False,
        ),
    )
    with patch("taskharbor.http_app._get_manager", return_value=registry_manager):
        with TestClient(app) as client:
            response = client.get(
                "/threads/t/files/download",
                params={"session_id": "s", "path": "/workspace/reports/a.txt"},
            )
    assert response.status_code == 200
    assert response.content == b"file-content"
