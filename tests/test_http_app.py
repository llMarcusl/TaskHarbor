"""Tests for the custom FastAPI http_app (middleware + download endpoints).

All tests are mock-based — no Docker or external server required.
Run with: pytest tests/test_http_app.py -v
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient


@pytest.fixture()
def mock_manager():
    manager = MagicMock()
    manager.resolve_owner.return_value = "owner"
    return manager


@pytest.fixture(autouse=True)
def authenticated_requests(monkeypatch):
    monkeypatch.setattr(
        "aegra_api.core.auth_deps.require_auth",
        AsyncMock(return_value=SimpleNamespace(identity="owner")),
    )
    monkeypatch.setattr(
        "taskharbor.settings.get_settings",
        lambda: SimpleNamespace(
            SESSION_MEMORY_ENABLED=False,
            LONG_TERM_MEMORY_ENABLED=False,
            CONTEXT_COMPACTION_ENABLED=False,
        ),
    )


@pytest.fixture()
def client(mock_manager):
    """TestClient for http_app.app with _get_manager patched."""
    with patch("taskharbor.http_app._get_manager", return_value=mock_manager):
        from taskharbor.http_app import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── ThreadDeleteCleanupMiddleware ──────────────────────────────


class TestThreadDeleteMiddleware:
    @staticmethod
    def _make_app(mock_manager, status_code: int = 200):
        """Build a fresh FastAPI app with middleware and a catch-all returning *status_code*."""
        from fastapi import FastAPI
        from starlette.responses import Response as StarletteResponse

        from taskharbor.http_app import ThreadDeleteCleanupMiddleware

        test_app = FastAPI()
        test_app.add_middleware(ThreadDeleteCleanupMiddleware)

        @test_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def _dummy(path: str):
            return StarletteResponse(status_code=status_code)

        return test_app

    def test_delete_thread_triggers_cleanup(self, mock_manager):
        with patch("taskharbor.http_app._get_manager", return_value=mock_manager):
            mock_manager.cleanup_thread_sessions.return_value = 2
            app = self._make_app(mock_manager, 200)

            with TestClient(app, raise_server_exceptions=False) as c:
                r = c.delete("/threads/abc123-def456")

            assert r.status_code == 200
            mock_manager.cleanup_thread_sessions.assert_called_once_with(
                "abc123-def456", user_id="owner", delete_files=True
            )

    def test_delete_non_thread_path_skips_cleanup(self, mock_manager):
        with patch("taskharbor.http_app._get_manager", return_value=mock_manager):
            app = self._make_app(mock_manager, 200)

            with TestClient(app, raise_server_exceptions=False) as c:
                c.delete("/assistants/something")

            mock_manager.cleanup_thread_sessions.assert_not_called()

    def test_non_delete_method_skips_cleanup(self, mock_manager):
        with patch("taskharbor.http_app._get_manager", return_value=mock_manager):
            app = self._make_app(mock_manager, 200)

            with TestClient(app, raise_server_exceptions=False) as c:
                c.get("/threads/abc123-def456")

            mock_manager.cleanup_thread_sessions.assert_not_called()

    def test_delete_with_404_skips_cleanup(self, mock_manager):
        with patch("taskharbor.http_app._get_manager", return_value=mock_manager):
            app = self._make_app(mock_manager, 404)

            with TestClient(app, raise_server_exceptions=False) as c:
                c.delete("/threads/abc123-def456")

            mock_manager.cleanup_thread_sessions.assert_not_called()

    def test_cleanup_exception_does_not_break_response(self, mock_manager):
        with patch("taskharbor.http_app._get_manager", return_value=mock_manager):
            mock_manager.cleanup_thread_sessions.side_effect = RuntimeError("boom")
            app = self._make_app(mock_manager, 200)

            with TestClient(app, raise_server_exceptions=False) as c:
                r = c.delete("/threads/abc123-def456")

            assert r.status_code == 200


# ── download_thread_file endpoint ──────────────────────────────


class TestDownloadThreadFile:
    def test_download_exported_file(self, client, mock_manager):
        mock_manager.is_file_exported.return_value = True
        mock_manager.is_exported_path_file.return_value = True
        mock_manager.stream_exported_file.return_value = iter([b"file content"])

        r = client.get(
            "/threads/t1/files/download",
            params={"session_id": "s1", "path": "/workspace/report.txt"},
        )

        assert r.status_code == 200
        assert r.content == b"file content"
        assert "report.txt" in r.headers["content-disposition"]
        assert r.headers["content-type"] == "application/octet-stream"
        mock_manager.is_file_exported.assert_called_once_with(
            "t1", "s1", "/workspace/report.txt", user_id="owner", artifact_id=None
        )
        mock_manager.stream_exported_file.assert_called_once_with(
            "t1", "s1", "/workspace/report.txt", user_id="owner", artifact_id=None
        )

    def test_download_directory_as_zip(self, client, mock_manager):
        mock_manager.is_file_exported.return_value = True
        mock_manager.is_exported_path_file.return_value = False
        mock_manager.stream_exported_file.return_value = iter([b"zip data"])

        r = client.get(
            "/threads/t1/files/download",
            params={"session_id": "s1", "path": "/workspace/mydir"},
        )

        assert r.status_code == 200
        assert "mydir.zip" in r.headers["content-disposition"]
        assert r.headers["content-type"] == "application/zip"

    def test_file_not_exported_returns_403(self, client, mock_manager):
        mock_manager.is_file_exported.return_value = False

        r = client.get(
            "/threads/t1/files/download",
            params={"session_id": "s1", "path": "/workspace/secret.txt"},
        )

        assert r.status_code == 403

    def test_invalid_path_returns_400(self, client, mock_manager):
        mock_manager.is_file_exported.side_effect = ValueError("Invalid path")

        r = client.get(
            "/threads/t1/files/download",
            params={"session_id": "s1", "path": "/workspace/../etc/passwd"},
        )

        assert r.status_code == 400

    def test_stream_error_returns_403(self, client, mock_manager):
        mock_manager.is_file_exported.return_value = True
        mock_manager.is_exported_path_file.side_effect = RuntimeError("container gone")

        r = client.get(
            "/threads/t1/files/download",
            params={"session_id": "s1", "path": "/workspace/file.txt"},
        )

        assert r.status_code == 403


@pytest.mark.parametrize("filename", ["报告.csv", "a%20b.csv", "line\nbreak.txt"])
def test_download_filename_is_encoded_without_double_decoding(client, mock_manager, filename):
    from urllib.parse import quote

    path = "/workspace/" + filename
    mock_manager.is_file_exported.return_value = True
    mock_manager.is_exported_path_file.return_value = True
    mock_manager.stream_exported_file.return_value = iter([b"content"])
    response = client.get("/threads/t/files/download", params={"session_id": "s", "path": path})
    assert response.status_code == 200
    assert response.headers["content-disposition"] == "attachment; filename*=UTF-8''" + quote(
        filename, safe=""
    )
    assert mock_manager.is_file_exported.call_args.args[2] == path
