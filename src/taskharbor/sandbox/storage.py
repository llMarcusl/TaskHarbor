"""Persistent, identity-scoped files independent of Docker container lifetime.

Only uploads and one runtime's workspace are mounted into a sandbox. Published
artifacts and their manifests stay outside those mounts, so subsequent code
cannot alter a previously published result.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import time
import uuid
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator


def _component(value: str | None) -> str:
    return hashlib.sha256((value or "local").encode("utf-8")).hexdigest()


def _is_link(path: Path) -> bool:
    # Junctions on Windows must be treated the same as symlinks.
    if path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)()):
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    # Path.is_junction is unavailable on Python 3.11. All Windows reparse
    # points fail closed, including junctions and unsupported link types.
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def checked_path(root: Path, relative: str | Path = ".", *, exists: bool = False) -> Path:
    """Join an untrusted relative path without following links or reparse points."""
    rel = Path(relative)
    if rel.is_absolute() or rel.drive or any(part == ".." for part in rel.parts):
        raise ValueError("Path must stay inside its storage directory")
    root = Path(os.path.abspath(root))
    current = root
    # Check ancestors too: a replaced storage root must not redirect reads.
    for ancestor in (*reversed(current.parents), current):
        if _is_link(ancestor):
            raise PermissionError("Symlinks and junctions are not allowed in storage paths")
    for part in rel.parts:
        current = current / part
        if _is_link(current):
            raise PermissionError("Symlinks and junctions are not allowed in storage paths")
    if not current.is_relative_to(root):
        raise ValueError("Path outside storage directory")
    if exists and not current.exists():
        raise FileNotFoundError(f"File or directory not found: {current.name}")
    return current


def normalize_workspace_path(path: str) -> str:
    if not path or "\x00" in path or "\\" in path:
        raise ValueError("Invalid sandbox path")
    path = path if posixpath.isabs(path) else posixpath.join("/workspace", path)
    normalized = posixpath.normpath(path)
    if normalized != "/workspace" and not normalized.startswith("/workspace/"):
        raise ValueError(f"Path outside /workspace: {path}")
    # Reject host-special names as well, including Windows drive/ADS syntax.
    if any(":" in part for part in normalized.split("/")):
        raise ValueError("Invalid sandbox path")
    return normalized


def _copy_open_file(source: Path, destination: Path) -> int:
    checked_path(source.parent, source.name, exists=True)
    descriptor = os.open(
        source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise PermissionError("Only regular files and directories can be copied")
        if details.st_nlink > 1:
            raise PermissionError("Hard-linked files are not allowed in sandbox transfers")
        # Verify again after open, including intermediate directories.
        checked_path(source.parent, source.name, exists=True)
        with os.fdopen(descriptor, "rb", closefd=False) as reader:
            with destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=65536)
        return destination.stat().st_size
    finally:
        os.close(descriptor)


def safe_copy(source: Path, destination: Path) -> int:
    """Copy a regular tree; never traverse source or destination symlinks."""
    source = checked_path(source.parent, source.name, exists=True)
    destination = checked_path(destination.parent, destination.name)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination.name}")
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        return _copy_open_file(source, destination)
    if not source.is_dir():
        raise PermissionError("Only regular files and directories can be copied")
    destination.mkdir(parents=True)
    total = 0
    for child in source.iterdir():
        total += safe_copy(child, destination / child.name)
    return total


@dataclass(frozen=True)
class ScopePaths:
    root: Path
    uploads: Path
    workspace: Path
    outputs: Path
    execution_root: Path


@dataclass(frozen=True)
class PublishedArtifact:
    artifact_id: str
    session_id: str
    export_path: str
    payload: Path
    size: int
    is_directory: bool
    created_ns: int
    archive: Path | None = None

    def stream(self, chunk_size: int = 65536) -> Iterator[bytes]:
        target = self.archive if self.is_directory else self.payload
        if target is None:
            raise ValueError("Directory artifact has no downloadable archive")
        checked_path(target.parent, target.name, exists=True)
        with target.open("rb") as reader:
            while chunk := reader.read(chunk_size):
                yield chunk


class WorkspaceStore:
    """Filesystem-backed storage for a single backend deployment."""

    def __init__(self, root: str | Path):
        self.root = Path(os.path.abspath(root))
        checked_path(self.root)

    def scope(
        self,
        user_id: str | None,
        thread_id: str,
        runtime: str = "python",
        *,
        create: bool = True,
        execution_scope: str | None = None,
    ) -> ScopePaths:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", runtime):
            raise ValueError("Invalid runtime storage name")
        root = checked_path(
            self.root, Path("users") / _component(user_id) / "threads" / _component(thread_id)
        )
        execution_root = (
            checked_path(root, Path("tasks") / _component(execution_scope))
            if execution_scope is not None
            else root
        )
        paths = ScopePaths(
            root=root,
            uploads=checked_path(execution_root, "uploads"),
            workspace=checked_path(execution_root, Path("workspaces") / runtime),
            outputs=checked_path(root, "outputs"),
            execution_root=execution_root,
        )
        if create:
            for directory in (paths.uploads, paths.workspace, paths.outputs):
                directory.mkdir(parents=True, exist_ok=True)
            # Docker runs as 65532. Only task data is writable; no secret-bearing
            # service directory or published artifact is mounted into containers.
            if os.name != "nt":
                paths.workspace.chmod(0o777)
                paths.uploads.chmod(0o755)
        return paths

    def delete_execution_scope(self, user_id: str, thread_id: str, execution_scope: str) -> None:
        """Remove only a task's mutable files; publications remain thread-owned."""
        if not execution_scope:
            raise ValueError("A non-empty task execution scope is required")
        scope = self.scope(user_id, thread_id, create=False, execution_scope=execution_scope)
        target = checked_path(self.root, scope.execution_root.relative_to(self.root))
        if target.exists():
            shutil.rmtree(target)

    def staged_scope(self, user_id: str, thread_id: str, execution_scope: str) -> Path:
        """Internal per-run result staging, never mounted into a child container."""
        if not execution_scope:
            raise ValueError("A non-empty task execution scope is required")
        scope = self.scope(user_id, thread_id, create=False)
        return checked_path(scope.root, Path("task-results") / _component(execution_scope))

    def delete_staged_scope(self, user_id: str, thread_id: str, execution_scope: str) -> None:
        target = self.staged_scope(user_id, thread_id, execution_scope)
        checked_path(self.root, target.relative_to(self.root))
        if target.exists():
            shutil.rmtree(target)

    @staticmethod
    def workspace_path(scope: ScopePaths, path: str, *, exists: bool = False) -> Path:
        normalized = normalize_workspace_path(path)
        relative = normalized.removeprefix("/workspace").lstrip("/")
        return checked_path(scope.workspace, relative or ".", exists=exists)

    def import_path(self, scope: ScopePaths, source: Path, destination: str) -> int:
        """Keep an original snapshot, then copy it into the writable workspace."""
        target = self.workspace_path(scope, destination)
        if target == scope.workspace:
            raise ValueError("Import destination must name a file or directory inside /workspace")
        # Stage outside the sandbox's writable mount before replacing a target.
        staging = checked_path(scope.uploads, uuid.uuid4().hex)
        staging.mkdir()
        original = staging / (source.name or "imported")
        try:
            size = safe_copy(source, original)
            self.copy_to_workspace(scope, original, destination)
            return size
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def copy_to_workspace(self, scope: ScopePaths, source: Path, destination: str) -> int:
        target = self.workspace_path(scope, destination)
        if target == scope.workspace:
            raise ValueError("Import destination must name a file or directory inside /workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        checked_path(scope.workspace, target.relative_to(scope.workspace))
        if target.exists():
            # Preserve the old file if copying the new one fails.
            temporary = target.with_name(f".import-{uuid.uuid4().hex}")
            size = safe_copy(source, temporary)
            if target.is_dir() or temporary.is_dir():
                shutil.rmtree(temporary) if temporary.is_dir() else temporary.unlink()
                raise FileExistsError("Directory imports do not overwrite existing destinations")
            temporary.replace(target)
        else:
            size = safe_copy(source, target)
        # The backend creates files, but the sandbox UID must be able to edit them.
        if os.name != "nt":
            targets = [target, *target.rglob("*")] if target.is_dir() else [target]
            for entry in targets:
                entry.chmod(0o777 if entry.is_dir() else 0o666)
        return size

    def publish(self, scope: ScopePaths, session_id: str, path: str) -> PublishedArtifact:
        normalized = normalize_workspace_path(path)
        source = self.workspace_path(scope, normalized, exists=True)
        artifact_id = uuid.uuid4().hex
        directory = checked_path(scope.outputs, artifact_id)
        directory.mkdir()
        payload = directory / "payload" / (source.name or "workspace")
        try:
            size = safe_copy(source, payload)
            is_directory = payload.is_dir()
            archive = directory / "download.zip" if is_directory else None
            if archive:
                with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
                    zipped.write(payload, payload.name + "/")
                    for entry in sorted(payload.rglob("*")):
                        archive_name = Path(payload.name) / entry.relative_to(payload)
                        zipped.write(entry, archive_name.as_posix())
            artifact = PublishedArtifact(
                artifact_id,
                session_id,
                normalized,
                payload,
                size,
                is_directory,
                time.time_ns(),
                archive,
            )
            manifest = {
                "artifact_id": artifact_id,
                "session_id": session_id,
                "path": normalized,
                "name": payload.name,
                "size": size,
                "is_directory": is_directory,
                "created_ns": artifact.created_ns,
            }
            temporary = directory / "manifest.tmp"
            with temporary.open("x", encoding="utf-8") as writer:
                json.dump(manifest, writer, ensure_ascii=False)
                writer.flush()
                os.fsync(writer.fileno())
            temporary.replace(directory / "manifest.json")
            return artifact
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def find(
        self,
        user_id: str | None,
        thread_id: str,
        session_id: str,
        path: str,
        *,
        artifact_id: str | None = None,
    ) -> PublishedArtifact | None:
        normalized = normalize_workspace_path(path)
        scope = self.scope(user_id, thread_id, create=False)
        if not scope.outputs.exists():
            return None
        if artifact_id is not None:
            if not re.fullmatch(r"[0-9a-f]{32}", artifact_id):
                raise ValueError("Invalid artifact ID")
            directories = [checked_path(scope.outputs, artifact_id)]
        else:
            directories = [
                checked_path(scope.outputs, entry.name)
                for entry in scope.outputs.iterdir()
                if re.fullmatch(r"[0-9a-f]{32}", entry.name)
            ]
        matches: list[PublishedArtifact] = []
        for directory in directories:
            try:
                manifest_path = checked_path(directory, "manifest.json", exists=True)
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                exported = normalize_workspace_path(data["path"])
                if data["session_id"] != session_id:
                    continue
                if normalized != exported and not (
                    data["is_directory"] and normalized.startswith(exported.rstrip("/") + "/")
                ):
                    continue
                payload = checked_path(directory, Path("payload") / data["name"], exists=True)
                artifact = PublishedArtifact(
                    directory.name,
                    session_id,
                    exported,
                    payload,
                    data["size"],
                    data["is_directory"],
                    data["created_ns"],
                    directory / "download.zip" if data["is_directory"] else None,
                )
                if normalized != exported:
                    relative = normalized[len(exported) :].lstrip("/")
                    member = checked_path(payload, relative, exists=True)
                    # Downloading a nested directory would need its own archive.
                    if not member.is_file():
                        continue
                    artifact = replace(
                        artifact,
                        payload=member,
                        size=member.stat().st_size,
                        is_directory=False,
                        archive=None,
                    )
                matches.append(artifact)
            except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError):
                continue  # Ignore interrupted/incomplete publications.
        return max(matches, key=lambda item: item.created_ns) if matches else None

    def delete_thread(self, user_id: str | None, thread_id: str) -> None:
        scope = self.scope(user_id, thread_id, create=False)
        checked_path(self.root, scope.root.relative_to(self.root))
        if scope.root.exists():
            shutil.rmtree(scope.root)
