"""Trusted metadata for tools assembled by this application."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    group: str
    executor: str
    automatic: bool = False
    version: str = "1"


SANDBOX_TOOLS = frozenset(
    {
        "create_session",
        "execute_code",
        "execute_terminal",
        "import_files",
        "export_files",
        "stop_session",
        "prepare_task_files",
        "import_task_files",
        "collect_task_files",
    }
)

ORCHESTRATION_TOOLS = frozenset(
    {
        "task",
        "wait_tasks",
        "cancel_task",
        "start_async_task",
        "check_async_task",
        "update_async_task",
        "cancel_async_task",
        "list_async_tasks",
    }
)


def tool_spec(name: str) -> ToolSpec:
    if name in ORCHESTRATION_TOOLS:
        return ToolSpec(name, "orchestration", "orchestration", automatic=True)
    if name in SANDBOX_TOOLS:
        return ToolSpec(name, "sandbox", "sandbox", automatic=True)
    if name == "execute_host_command":
        return ToolSpec(name, "terminal", "host")
    if name == "read_history":
        return ToolSpec(name, "history", "host", automatic=True)
    # Assembled custom tools are registered but do not acquire automatic permission.
    return ToolSpec(name, "integration", "connector")
