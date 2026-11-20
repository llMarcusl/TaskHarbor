"""Explicit, signed file grants between independent parent and worker threads."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, ConfigDict, Field

from taskharbor.sandbox.storage import normalize_workspace_path

from .protocol import sign_payload, verify_payload, worker_parent


class TaskFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime: str = Field(default="python", pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    path: str = Field(min_length=1)
    destination: str | None = None


def _identity(manager, runtime):
    thread = str(runtime.config.get("configurable", {}).get("thread_id") or "")
    if not thread:
        raise PermissionError("Task files require a persisted thread")
    return manager.resolve_owner(thread), thread


def _entries(files):
    if not 1 <= len(files) <= 20:
        raise ValueError("Provide between 1 and 20 files")
    entries = []
    for item in files:
        entry = item.model_dump(exclude_none=True)
        entry["path"] = normalize_workspace_path(entry["path"])
        if entry.get("destination"):
            entry["destination"] = normalize_workspace_path(entry["destination"])
        entries.append(entry)
    return entries


async def _copy(manager, owner, source, destination, entries):
    results = []
    for runtime in dict.fromkeys(entry["runtime"] for entry in entries):
        result = await asyncio.to_thread(
            manager.copy_thread_files, owner, source, destination, runtime,
            [entry for entry in entries if entry["runtime"] == runtime],
        )
        results.extend({"runtime": runtime, **asdict(entry)} for entry in result.files)
    return json.dumps({"success": True, "files": results}, ensure_ascii=False)


def create_parent_file_tools(manager, settings):
    @tool
    def prepare_task_files(files: list[TaskFile], runtime: ToolRuntime) -> str:
        """Authorize specific parent workspace files for a worker.

        Include the returned manifest verbatim in start_async_task's description.
        Files are referenced by runtime and /workspace path, and may include a
        destination path inside the child's workspace.
        """
        owner, parent = _identity(manager, runtime)
        entries = _entries(files)
        store = manager._get_storage()
        for entry in entries:
            scope = store.scope(owner, parent, entry["runtime"], create=False)
            store.workspace_path(scope, entry["path"], exists=True)
        manifest = sign_payload(
            {"owner": owner, "parent": parent, "files": entries},
            purpose="file-manifest", settings=settings,
        )
        return json.dumps({"manifest": manifest, "files": entries}, ensure_ascii=False)

    @tool
    async def collect_task_files(
        task_id: str, files: list[TaskFile], runtime: ToolRuntime,
    ) -> str:
        """Copy selected worker result files into the main workspace.

        Use the task ID returned by start_async_task and the runtime/path refs
        in that worker's verified result. Export final files from your own session.
        """
        owner, parent = _identity(manager, runtime)
        if await worker_parent(owner, task_id) != parent:
            raise PermissionError("Worker belongs to another parent")
        return await _copy(manager, owner, task_id, parent, _entries(files))

    return [prepare_task_files, collect_task_files]


def create_worker_file_tools(manager, settings):
    @tool
    async def import_task_files(manifest: str, runtime: ToolRuntime) -> str:
        """Import the exact parent files authorized by a manifest in your task.

        Call before inspecting inputs. The returned runtime/path refs are in your
        own workspace and create_session will mount that persistent workspace.
        """
        owner, child = _identity(manager, runtime)
        grant = verify_payload(manifest, purpose="file-manifest", settings=settings)
        parent = await worker_parent(owner, child)
        if not parent or (grant.get("owner"), grant.get("parent")) != (owner, parent):
            raise PermissionError("File manifest belongs to another task owner")
        entries = _entries([TaskFile.model_validate(entry) for entry in grant.get("files", [])])
        return await _copy(manager, owner, parent, child, entries)

    return [import_task_files]
