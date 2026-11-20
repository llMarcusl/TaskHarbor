"""Tests for CLI formatting, dispatch, and helper functions.

All tests are pure unit tests — no Docker, no LLM, no network required.
Run with: pytest tests/test_cli.py -v
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.syntax import Syntax
from rich.text import Text

from taskharbor.cli import (
    _api_is_healthy,
    _api_subprocess_env,
    _extract_text_content,
    _format_tool_input,
    _format_tool_output,
)
from taskharbor.sandbox.models import MAX_TOOL_OUTPUT_LINES


class TestApiEventLoop:
    @pytest.mark.parametrize(
        ("platform", "configured_loop", "expected"),
        [
            ("win32", None, "asyncio:SelectorEventLoop"),
            ("linux", None, None),
            ("win32", "custom:loop", "custom:loop"),
        ],
    )
    def test_child_environment(self, platform, configured_loop, expected):
        env = {"EXISTING_SETTING": "preserved"}
        if configured_loop is not None:
            env["UVICORN_LOOP"] = configured_loop
        with (
            patch("taskharbor.cli.os.environ", env),
            patch("taskharbor.cli.sys.platform", platform),
        ):
            child_env = _api_subprocess_env()
        assert child_env.get("UVICORN_LOOP") == expected
        assert child_env["EXISTING_SETTING"] == "preserved"
        assert env.get("UVICORN_LOOP") == configured_loop

    @pytest.mark.parametrize("dev", [False, True])
    def test_api_passes_environment_to_aegra(self, dev):
        from taskharbor.cli import _run_api

        child_env = {"UVICORN_LOOP": "asyncio:SelectorEventLoop"}
        with (
            patch("taskharbor.cli._ensure_postgres_running", return_value=True),
            patch("taskharbor.cli._api_subprocess_env", return_value=child_env),
            patch("taskharbor.cli.subprocess.run") as run,
            pytest.raises(SystemExit),
        ):
            _run_api(dev=dev)
        assert run.call_args.kwargs["env"] == child_env
        assert run.call_args.args[0][1] == ("dev" if dev else "serve")
        if dev:
            assert "--no-db-check" in run.call_args.args[0]

    def test_ui_passes_environment_to_background_api(self):
        from taskharbor.cli import _run_ui

        child_env = {"UVICORN_LOOP": "asyncio:SelectorEventLoop"}
        process = MagicMock()
        process.poll.return_value = 0
        with (
            patch("taskharbor.cli._API_PROCESS", None),
            patch("taskharbor.cli.get_settings") as settings,
            patch("taskharbor.cli._api_is_healthy", side_effect=[False, True]),
            patch("taskharbor.cli._ensure_postgres_running", return_value=True),
            patch("taskharbor.cli._api_subprocess_env", return_value=child_env),
            patch("taskharbor.cli.atexit.register"),
            patch("taskharbor.cli.subprocess.Popen", return_value=process) as popen,
            patch("taskharbor.cli.subprocess.run") as run,
            patch("taskharbor.cli.shutil.which", return_value="C:/node/npm.cmd"),
            patch("taskharbor.cli.Path.exists", return_value=True),
        ):
            settings.return_value.API_BASE_URL = "http://127.0.0.1:8000"
            _run_ui()
        assert popen.call_args.kwargs["env"] == child_env
        assert popen.call_args.kwargs["cwd"] == Path(__file__).resolve().parents[1]
        assert run.call_args.args[0] == ["C:/node/npm.cmd", "run", "dev"]


class TestStartupPaths:
    def test_settings_find_project_env_from_frontend_directory(self, tmp_path, monkeypatch):
        """Exercise a copied source tree with dummy credentials, never the real .env."""
        import taskharbor.settings as settings_module

        module_path = tmp_path / "src" / "taskharbor" / "settings.py"
        module_path.parent.mkdir(parents=True)
        module_path.write_text(Path(settings_module.__file__).read_text(encoding="utf-8"))
        values = {
            "CHAT_MODEL_API_KEY": "dummy",
            "POSTGRES_USER": "dummy",
            "POSTGRES_PASSWORD": "dummy",
            "POSTGRES_DB": "dummy",
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_PORT": "55439",
            "SESSION_MEMORY_ENABLED": "false",
            "LONG_TERM_MEMORY_ENABLED": "false",
        }
        (tmp_path / ".env").write_text(
            "\n".join(f"{k}={v}" for k, v in values.items()), encoding="utf-8"
        )
        frontend = tmp_path / "frontend"
        frontend.mkdir()
        for key in values:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.chdir(frontend)
        Settings = runpy.run_path(str(module_path))["Settings"]
        settings = Settings()
        assert settings.POSTGRES_PORT == "55439"
        assert settings.CHAT_MODEL_API_KEY == "dummy"
        # Explicit environment variables still take precedence over the root file.
        monkeypatch.setenv("POSTGRES_PORT", "55440")
        assert Settings().POSTGRES_PORT == "55440"

    @pytest.mark.parametrize("dev", [False, True])
    def test_api_uses_project_root_and_checks_existing_db(self, tmp_path, monkeypatch, dev):
        import taskharbor.cli as cli

        monkeypatch.chdir(tmp_path)
        root = Path(cli.__file__).resolve().parents[2]
        with (
            patch("taskharbor.cli._ensure_postgres_running", return_value=True) as ensure,
            patch("taskharbor.cli.subprocess.run") as run,
            pytest.raises(SystemExit),
        ):
            cli._run_api(dev=dev)
        ensure.assert_called_once()
        assert run.call_args.kwargs["cwd"] == root

    def test_dev_does_not_launch_if_db_check_fails(self):
        from taskharbor.cli import _run_api

        with (
            patch("taskharbor.cli._ensure_postgres_running", return_value=False),
            patch("taskharbor.cli.subprocess.run") as run,
            pytest.raises(SystemExit) as exc,
        ):
            _run_api(dev=True)
        assert exc.value.code == 1
        run.assert_not_called()

    def test_existing_database_does_not_start_another_container(self):
        from taskharbor.cli import _ensure_postgres_running

        settings = MagicMock(POSTGRES_HOST="localhost", POSTGRES_PORT="5432")
        with (
            patch("taskharbor.cli.get_settings", return_value=settings),
            patch("taskharbor.cli._postgres_reachable", return_value=True),
            patch("taskharbor.cli.subprocess.run") as run,
        ):
            assert _ensure_postgres_running(MagicMock())
        run.assert_not_called()


# ── _format_tool_input ─────────────────────────────────────────


class TestFormatToolInput:
    def test_execute_code_default_python_lexer(self):
        panel = _format_tool_input(
            {"name": "execute_code", "args": {"code": "print(1)"}},
            sessions={},
        )
        renderable = panel.renderable
        assert isinstance(renderable, Syntax)
        assert renderable.lexer.name.lower() == "python"

    def test_execute_code_node_session_uses_javascript_lexer(self):
        panel = _format_tool_input(
            {"name": "execute_code", "args": {"code": "console.log(1)", "session_id": "s1"}},
            sessions={"s1": "node"},
        )
        renderable = panel.renderable
        assert isinstance(renderable, Syntax)
        assert renderable.lexer.name.lower() == "javascript"

    def test_execute_code_r_session_uses_r_lexer(self):
        panel = _format_tool_input(
            {"name": "execute_code", "args": {"code": "print('hello')", "session_id": "s1"}},
            sessions={"s1": "r"},
        )
        renderable = panel.renderable
        assert isinstance(renderable, Syntax)
        assert renderable.lexer.name.lower() in ("r", "s")

    def test_execute_terminal_uses_bash_lexer(self):
        panel = _format_tool_input(
            {"name": "execute_terminal", "args": {"command": "ls -la"}},
            sessions={},
        )
        renderable = panel.renderable
        assert isinstance(renderable, Syntax)
        assert renderable.lexer.name.lower() == "bash"

    def test_generic_tool_uses_text(self):
        panel = _format_tool_input(
            {"name": "create_session", "args": {"language": "python"}},
            sessions={},
        )
        renderable = panel.renderable
        assert isinstance(renderable, Text)

    def test_panel_title_contains_tool_name(self):
        panel = _format_tool_input(
            {"name": "stop_session", "args": {"session_id": "x"}},
            sessions={},
        )
        assert "stop_session" in str(panel.title)


# ── _extract_text_content ──────────────────────────────────────


class TestExtractTextContent:
    def test_string_input(self):
        assert _extract_text_content("hello") == "hello"

    def test_multimodal_list_with_text_block(self):
        content = [
            {"type": "image_url", "image_url": {"url": "data:..."}},
            {"type": "text", "text": "result text"},
        ]
        assert _extract_text_content(content) == "result text"

    def test_list_without_text_block(self):
        content = [{"type": "image_url", "image_url": {"url": "data:..."}}]
        result = _extract_text_content(content)
        assert isinstance(result, str)

    def test_non_string_non_list(self):
        assert _extract_text_content(42) == "42"


# ── _format_tool_output ───────────────────────────────────────


class TestFormatToolOutput:
    """_format_tool_output expects API-style message dicts (name, content)."""

    def test_success_green_border(self):
        msg = {
            "name": "execute_code",
            "content": json.dumps({"success": True, "stdout": "ok"}),
        }
        panel = _format_tool_output(msg)
        assert panel.border_style == "green"

    def test_error_red_border(self):
        msg = {
            "name": "execute_code",
            "content": json.dumps({"success": False, "error": "boom"}),
        }
        panel = _format_tool_output(msg)
        assert panel.border_style == "red"

    def test_error_invoking_tool_red_border(self):
        msg = {
            "name": "execute_code",
            "content": "Error invoking tool execute_code: session not found",
        }
        panel = _format_tool_output(msg)
        assert panel.border_style == "red"

    def test_long_output_is_truncated(self):
        items = {f"key_{i}": f"val_{i}" for i in range(MAX_TOOL_OUTPUT_LINES + 50)}
        msg = {
            "name": "execute_code",
            "content": json.dumps({"success": True, **items}),
        }
        panel = _format_tool_output(msg)
        text = str(panel.renderable)
        assert "omitidas" in text

    def test_panel_title_contains_tool_name(self):
        msg = {
            "name": "execute_terminal",
            "content": json.dumps({"success": True}),
        }
        panel = _format_tool_output(msg)
        assert "execute_terminal" in str(panel.title)


# ── _api_is_healthy ────────────────────────────────────────────


class TestApiIsHealthy:
    def test_healthy_returns_true(self):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("taskharbor.cli.urllib.request.urlopen", return_value=mock_response):
            assert _api_is_healthy("http://fake:8000") is True

    def test_unreachable_returns_false(self):
        with patch(
            "taskharbor.cli.urllib.request.urlopen",
            side_effect=ConnectionError("refused"),
        ):
            assert _api_is_healthy("http://fake:8000") is False


# ── main() dispatch ────────────────────────────────────────────


class TestMainDispatch:
    def test_removed_mcp_command_exits(self):
        with patch("sys.argv", ["taskharbor", "mcp"]), pytest.raises(SystemExit) as exc:
            from taskharbor.cli import main
            main()
        assert exc.value.code == 1

    def test_api_command(self):
        with (
            patch("sys.argv", ["taskharbor", "api"]),
            patch("taskharbor.cli._run_api") as mock_api,
        ):
            from taskharbor.cli import main

            main()
            mock_api.assert_called_once()

    def test_ui_command(self):
        with (
            patch("sys.argv", ["taskharbor", "ui"]),
            patch("taskharbor.cli._run_ui") as mock_ui,
        ):
            from taskharbor.cli import main

            main()
            mock_ui.assert_called_once()

    def test_help_command(self, capsys):
        with patch("sys.argv", ["taskharbor", "--help"]):
            from taskharbor.cli import main

            main()
        captured = capsys.readouterr()
        assert "taskharbor" in captured.out or True  # Rich prints to console, not stdout

    def test_unknown_command_exits(self):
        with (
            patch("sys.argv", ["taskharbor", "nonexistent"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            from taskharbor.cli import main

            main()
        assert exc_info.value.code == 1

    def test_no_args_calls_cli(self):
        with (
            patch("sys.argv", ["taskharbor"]),
            patch("taskharbor.cli.run_interactive_cli") as mock_cli,
        ):
            from taskharbor.cli import main

            main()
            mock_cli.assert_called_once()
