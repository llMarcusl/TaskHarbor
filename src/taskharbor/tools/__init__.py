"""LangChain tools for the TaskHarbor.

Tools are created via factory functions that bind to a shared
:class:`~taskharbor.sandbox.SandboxManager` instance.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from taskharbor.sandbox.manager import SandboxManager
from taskharbor.tools.create_session import create_create_session_tool
from taskharbor.tools.execute_code import create_execute_code_tool
from taskharbor.tools.execute_terminal import create_execute_terminal_tool
from taskharbor.tools.export_files import create_export_files_tool
from taskharbor.tools.import_files import create_import_files_tool
from taskharbor.tools.stop_session import create_stop_session_tool


def create_tools(manager: SandboxManager) -> list[BaseTool]:
    """Create all agent tools bound to the given SandboxManager.

    Args:
        manager: The shared SandboxManager instance.

    Returns:
        List of LangChain tool callables.
    """
    return [
        create_create_session_tool(manager),
        create_execute_code_tool(manager),
        create_execute_terminal_tool(manager),
        create_import_files_tool(manager),
        create_export_files_tool(manager),
        create_stop_session_tool(manager),
    ]


__all__ = ["create_tools"]
