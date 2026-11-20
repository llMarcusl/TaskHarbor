"""
SandboxManager — orchestrates Docker containers via docker exec.
No exposed ports. Only application-owned scoped data directories are mounted.
Containers have CPU/memory/PID limits and use a non-root user by default.
Writable persistent task data requires separate host disk capacity monitoring.
"""

from __future__ import annotations

import atexit
import contextvars
import json
import logging
import posixpath
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Generator

import docker
import docker.errors

from taskharbor.sandbox.execution import (
    EXECUTOR_IMAGE_LABEL,
    EXECUTOR_VERSION,
    TerminalExecutor,
)
from taskharbor.sandbox.models import (
    ExecutionResult,
    ExportFileResult,
    ExportResult,
    ImportFileResult,
    ImportResult,
    SessionInfo,
    TerminalResult,
    truncate_field,
)
from taskharbor.settings import get_settings

logger = logging.getLogger(__name__)

current_thread_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_thread_id",
    default=None,
)
current_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_user_id",
    default=None,
)
current_execution_scope: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_execution_scope", default=None
)
_ALL_EXECUTION_SCOPES = object()


def _scoped_key(*parts, execution_scope: str | None):
    # Preserve legacy root keys and their storage/recovery behavior.
    return (*parts, execution_scope) if execution_scope is not None else parts


def _session_operation(method):
    @wraps(method)
    def guarded(self, session_id, *args, **kwargs):
        with self._operation(session_id):
            return method(self, session_id, *args, **kwargs)

    return guarded


SANDBOX_UID = 65532
SANDBOX_GID = 65532


RUNTIME_CONFIG: dict[str, dict[str, Any]] = {
    "python": {
        "image": "sandbox-python:latest",
        "dockerfile": "Dockerfile.python",
        "client_cmd": ["python", "/client/client_python.py"],
        "install_cmd": lambda pkgs: ["pip", "install", "--no-cache-dir", *pkgs],
    },
    "node": {
        "image": "sandbox-node:latest",
        "dockerfile": "Dockerfile.node",
        "client_cmd": ["node", "/client/client_node.js"],
        "install_cmd": lambda pkgs: ["npm", "install", "--save", *pkgs],
    },
    "r": {
        "image": "sandbox-r:latest",
        "dockerfile": "Dockerfile.r",
        "client_cmd": ["/client/client_c"],
        "install_cmd": lambda pkgs: [
            "Rscript",
            "-e",
            "install.packages(c(" + ",".join(f"'{p}'" for p in pkgs) + "), Ncpus=2L)",
        ],
        "override_settings": {
            "CONTAINER_EXECUTE_AS_ROOT": True,
        },
    },
}


def _runtime_setting(runtime: str, key: str) -> Any:
    """Return a setting value, respecting per-runtime overrides."""
    overrides = RUNTIME_CONFIG[runtime].get("override_settings", {})
    if key in overrides:
        return overrides[key]
    return getattr(get_settings(), key)


_SAFE_PKG_NAME_RE = re.compile(r"^@?[a-zA-Z0-9]([a-zA-Z0-9._/-]*[a-zA-Z0-9])?$")


def _validate_package_names(packages: dict[str, str]) -> None:
    """Reject package names that could be used for injection attacks (e.g. R string injection)."""
    for name in packages:
        if not _SAFE_PKG_NAME_RE.match(name):
            raise ValueError(
                f"Invalid package name: {name!r}. "
                "Names must contain only letters, digits, dots, hyphens, underscores, and slashes."
            )


class ContainerDiedError(RuntimeError):
    """Raised when a sandbox container has exited unexpectedly."""

    def __init__(self, session_id: str, reason: str, exit_code: int | None = None):
        self.session_id = session_id
        self.reason = reason
        self.exit_code = exit_code
        super().__init__(
            f"Container for session '{session_id}' died: {reason}"
            + (f" (exit code {exit_code})" if exit_code is not None else "")
        )


class SandboxManager:
    """Manages Docker container sandboxes with persistent kernels."""

    def __init__(self, docker_dir: str | Path | None = None) -> None:
        self.client = docker.from_env()
        self.sessions: dict[str, SessionInfo] = {}
        self._containers: dict[str, Any] = {}
        self.thread_sessions: dict[str, set[str]] = {}
        self._exported_files: dict[
            str, dict[str, set[str]]
        ] = {}  # thread_key -> session_id -> set(paths)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._create_lock = threading.Lock()
        self._execution_locks: dict[str, threading.RLock] = {}
        self._workspace_locks: dict[tuple, threading.RLock] = {}
        self._active_ops: dict[str, int] = {}
        self._scope_operations: dict[tuple, int] = {}
        self._turn_leases: dict[tuple, float] = {}
        self._cancel_generations: dict[tuple, int] = {}
        self._deleted_scopes: set[tuple[str, str]] = set()
        self._deleted_execution_scopes: set[tuple[str, str, str]] = set()
        self._scope_acquisitions: dict[tuple, int] = {}
        self._released_at: dict[str, float] = {}
        self._quiesced: set[str] = set()
        self._waiters = 0
        self._closed = False
        self._gc_stop = threading.Event()
        self._manager_id = uuid.uuid4().hex
        self._terminal_executor = TerminalExecutor()
        from taskharbor.sandbox.ownership import DeploymentLock

        self._deployment_lock = DeploymentLock(get_settings().STORAGE_DIR)
        self._heartbeat_path = (
            get_settings().STORAGE_DIR / "control" / (self._manager_id + ".lease")
        )
        self.docker_dir = (
            Path(docker_dir) if docker_dir else Path(__file__).parent.parent / "docker"
        )
        self._cleanup_registered = False
        self._deployment_lock.acquire()
        try:
            self._heartbeat_path.touch()
            self._register_cleanup()
            self._cleanup_orphaned_containers()
            self._start_gc()
            self._ensure_exported_files_table()
        except BaseException:
            self.cleanup_all()
            raise

    def _get_db_pool(self):
        """Return the shared DB connection pool (lazy import to avoid circular deps)."""
        if not hasattr(self, "_db_pool"):
            try:
                from taskharbor.clients import get_db_pool

                self._db_pool = get_db_pool()
            except Exception:
                self._db_pool = None
        return self._db_pool

    def _ensure_exported_files_table(self) -> None:
        """Create exported_files table in PostgreSQL for cross-process sharing."""
        pool = self._get_db_pool()
        if pool is None:
            return
        try:
            with pool.connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS exported_files (
                        thread_key TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        path TEXT NOT NULL,
                        PRIMARY KEY (thread_key, session_id, path)
                    )
                """)
        except Exception:
            logger.debug("Could not create exported_files table", exc_info=True)

    def _db_add_exported(self, thread_key: str, session_id: str, path: str) -> None:
        pool = self._get_db_pool()
        if pool is None:
            return
        try:
            with pool.connection() as conn:
                conn.execute(
                    "INSERT INTO exported_files (thread_key, session_id, path) "
                    "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (thread_key, session_id, path),
                )
        except Exception:
            logger.debug("Failed to add exported file to DB", exc_info=True)

    def _db_remove_exported_session(self, session_id: str) -> None:
        pool = self._get_db_pool()
        if pool is None:
            return
        try:
            with pool.connection() as conn:
                conn.execute(
                    "DELETE FROM exported_files WHERE session_id = %s",
                    (session_id,),
                )
        except Exception:
            logger.debug("Failed to remove exported session from DB", exc_info=True)

    def _db_remove_exported_thread(self, thread_key: str) -> None:
        pool = self._get_db_pool()
        if pool is None:
            return
        try:
            with pool.connection() as conn:
                conn.execute(
                    "DELETE FROM exported_files WHERE thread_key = %s",
                    (thread_key,),
                )
        except Exception:
            logger.debug("Failed to remove exported thread from DB", exc_info=True)

    def _db_is_file_exported(self, thread_key: str, session_id: str, path: str) -> bool:
        pool = self._get_db_pool()
        if pool is None:
            return False
        try:
            norm = self._normalize_container_path(path)
            with pool.connection() as conn:
                rows = conn.execute(
                    "SELECT path FROM exported_files WHERE thread_key = %s AND session_id = %s",
                    (thread_key, session_id),
                ).fetchall()
            for row in rows:
                p = row["path"] if isinstance(row, dict) else row[0]
                if norm == p or norm.startswith(p.rstrip("/") + "/"):
                    return True
            return False
        except Exception:
            return False

    # ── Image Build ────────────────────────────────────

    def _ensure_image(self, runtime: str) -> None:
        config = RUNTIME_CONFIG[runtime]
        try:
            image = self.client.images.get(config["image"])
            if image.labels.get(EXECUTOR_IMAGE_LABEL) != EXECUTOR_VERSION:
                raise docker.errors.ImageNotFound("Sandbox executor image needs an update")
        except docker.errors.ImageNotFound:
            self.client.images.build(
                path=str(self.docker_dir),
                dockerfile=config["dockerfile"],
                tag=config["image"],
                rm=True,
            )

    # ── Orphan Cleanup & GC ──────────────────────────────

    def _db_session_has_exports(self, session_id: str) -> bool:
        """Check if any exported files exist in the DB for this session."""
        pool = self._get_db_pool()
        if pool is None:
            return False
        try:
            with pool.connection() as conn:
                row = conn.execute(
                    "SELECT 1 FROM exported_files WHERE session_id = %s LIMIT 1",
                    (session_id,),
                ).fetchone()
            return row is not None
        except Exception:
            return False

    def _cleanup_orphaned_containers(self) -> None:
        """Remove sandbox containers left over from previous runs.

        Only this deployment's marked containers are eligible. Live manager
        heartbeats and the minimum age protect other running manager instances.
        Legacy containers without ownership markers require explicit cleanup.
        """
        try:
            containers = self.client.containers.list(
                all=True,
                filters={"label": "taskharbor=true"},
            )
        except Exception:
            logger.warning("Failed to list orphaned containers", exc_info=True)
            return

        now = datetime.now(timezone.utc)

        for container in containers:
            cid = container.short_id
            manager_id = container.labels.get("manager-id")
            if not manager_id or not re.fullmatch(r"[0-9a-f]{32}", manager_id):
                # Legacy/foreign containers have no verifiable deployment ownership.
                # Do not destroy them merely because they share a generic label.
                continue
            if manager_id:
                if container.labels.get("storage-id") != str(get_settings().STORAGE_DIR):
                    continue
                heartbeat = get_settings().STORAGE_DIR / "control" / (manager_id + ".lease")
                if heartbeat.exists() and time.time() - heartbeat.stat().st_mtime < max(
                    get_settings().SANDBOX_LEASE_TTL_SECONDS,
                    get_settings().SESSION_GC_INTERVAL_SECONDS * 3,
                ):
                    continue

            # Skip recently created containers
            min_age = get_settings().CONTAINER_ORPHAN_MIN_AGE_SECONDS
            try:
                created_str = container.attrs.get("Created", "")
                if created_str:
                    created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    age = (now - created_at).total_seconds()
                    if age < min_age:
                        logger.info(
                            "Skipping orphan cleanup for %s (age %.0fs < %ds)",
                            cid,
                            age,
                            min_age,
                        )
                        continue
            except Exception:
                logger.debug("Could not parse container creation time", exc_info=True)

            try:
                container.stop(timeout=3)
                container.remove(force=True)
                logger.info("Removed orphaned container %s", cid)
            except Exception:
                logger.warning("Failed to remove orphaned container %s", cid, exc_info=True)

    def resolve_owner(self, thread_id: str | None) -> str:
        """Resolve server-owned identity; never use client configurable.user_id."""
        settings = get_settings()
        if settings.SANDBOX_STANDALONE_OWNER:
            return settings.SANDBOX_STANDALONE_OWNER
        if thread_id:
            pool = self._get_db_pool()
            if pool is not None:
                with pool.connection() as conn:
                    row = conn.execute(
                        'SELECT user_id FROM "thread" WHERE thread_id = %s', (thread_id,)
                    ).fetchone()
                if row:
                    owner = row["user_id"] if isinstance(row, dict) else row[0]
                    if owner:
                        return str(owner)
            raise PermissionError("Thread owner is unavailable; sandbox access denied")
        return "local"  # Direct trusted Python use, with a fresh scope per session.

    def _owner(self) -> str:
        return current_user_id.get(None) or self.resolve_owner(current_thread_id.get(None))

    def begin_turn(
        self, owner: str, thread_id: str, run_id: str, *, execution_scope: str | None = None
    ) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("Sandbox manager is shutting down")
            if (owner, thread_id) in self._deleted_scopes:
                raise PermissionError("Conversation was deleted")
            if (owner, thread_id, execution_scope) in self._deleted_execution_scopes:
                raise PermissionError("Task execution was closed")
            self._turn_leases[
                _scoped_key(owner, thread_id, run_id, execution_scope=execution_scope)
            ] = time.monotonic() + get_settings().SANDBOX_LEASE_TTL_SECONDS
            for sid, info in self.sessions.items():
                if (info.user_id, info.thread_id, info.execution_scope) == (
                    owner,
                    thread_id,
                    execution_scope,
                ):
                    info.status = "active"
                    self._released_at.pop(sid, None)

    def renew_turn(
        self, owner: str, thread_id: str, run_id: str, *, execution_scope: str | None = None
    ) -> None:
        self.begin_turn(owner, thread_id, run_id, execution_scope=execution_scope)

    def end_turn(
        self, owner: str, thread_id: str, run_id: str, *, execution_scope: str | None = None
    ) -> None:
        with self._condition:
            self._turn_leases.pop(
                _scoped_key(owner, thread_id, run_id, execution_scope=execution_scope), None
            )
            self._refresh_warm_locked()
            for sid, info in list(self.sessions.items()):
                if info.retire_requested and info.status == "warm":
                    self.stop_session(sid, force=False)
            self._condition.notify_all()

    def cancel_turn(
        self,
        owner: str,
        thread_id: str,
        run_id: str,
        *,
        execution_scope: str | None | object = _ALL_EXECUTION_SCOPES,
    ) -> None:
        """Cancel one execution scope, or the entire conversation when omitted."""
        with self._condition:
            all_scopes = execution_scope is _ALL_EXECUTION_SCOPES
            scope = (owner, thread_id) if all_scopes else (owner, thread_id, execution_scope)
            self._cancel_generations[scope] = self._cancel_generations.get(scope, 0) + 1
            for key in list(self._turn_leases):
                scope_matches = all_scopes or (key[3] if len(key) == 4 else None) == execution_scope
                if (
                    key[:2] == (owner, thread_id)
                    and scope_matches
                    and (all_scopes or key[2] == run_id)
                ):
                    self._turn_leases.pop(key, None)
            sids = [
                sid
                for sid, info in self.sessions.items()
                if (info.user_id, info.thread_id) == (owner, thread_id)
                and (all_scopes or info.execution_scope == execution_scope)
                and self._active_ops.get(sid, 0)
            ]
            for sid in sids:
                self.stop_session(sid)
            self._refresh_warm_locked()
            self._condition.notify_all()

    def _has_turn_locked(self, info: SessionInfo) -> bool:
        now = time.monotonic()
        return any(
            key[:2] == (info.user_id, info.thread_id)
            and (key[3] if len(key) == 4 else None) == info.execution_scope
            and until > now
            for key, until in self._turn_leases.items()
        )

    def _refresh_warm_locked(self) -> None:
        now = time.monotonic()
        self._turn_leases = {k: v for k, v in self._turn_leases.items() if v > now}
        for sid, info in self.sessions.items():
            if info.status == "starting":
                continue
            if self._active_ops.get(sid, 0) or self._has_turn_locked(info):
                info.status = "active"
                self._released_at.pop(sid, None)
            else:
                info.status = "warm"
                self._released_at.setdefault(sid, now)

    @contextmanager
    def _operation(self, session_id: str):
        """Atomically pin a session and serialize its persistent kernel/files."""
        with self._condition:
            self._check_session(session_id)
            if self._closed:
                raise RuntimeError("Sandbox manager is shutting down")
            info = self.sessions[session_id]
            if (info.user_id, info.thread_id) in self._deleted_scopes:
                raise PermissionError("Conversation was deleted")
            if (
                info.user_id,
                info.thread_id,
                info.execution_scope,
            ) in self._deleted_execution_scopes:
                raise PermissionError("Task execution was closed")
            scope_key = _scoped_key(
                info.user_id, info.thread_id, info.runtime, execution_scope=info.execution_scope
            )
            self._scope_operations[scope_key] = self._scope_operations.get(scope_key, 0) + 1
            self._active_ops[session_id] = self._active_ops.get(session_id, 0) + 1
            self.sessions[session_id].status = "active"
            self._released_at.pop(session_id, None)
            execution_lock = self._execution_locks.setdefault(session_id, threading.RLock())
        try:
            with execution_lock:
                self._check_session(session_id)
                yield
        finally:
            with self._condition:
                scope_remaining = self._scope_operations.get(scope_key, 1) - 1
                if scope_remaining > 0:
                    self._scope_operations[scope_key] = scope_remaining
                else:
                    self._scope_operations.pop(scope_key, None)
                remaining = self._active_ops.get(session_id, 1) - 1
                if remaining > 0:
                    self._active_ops[session_id] = remaining
                else:
                    self._active_ops.pop(session_id, None)
                if session_id in self.sessions:
                    self.sessions[session_id].last_activity = datetime.now(timezone.utc)
                self._refresh_warm_locked()
                self._release_ownership_if_idle_locked()
                self._condition.notify_all()

    def _start_gc(self) -> None:
        self._gc_thread = threading.Thread(target=self._gc_loop, daemon=True, name="sandbox-gc")
        self._gc_thread.start()

    def _gc_loop(self) -> None:
        while not self._gc_stop.wait(get_settings().SESSION_GC_INTERVAL_SECONDS):
            try:
                self._heartbeat_path.touch()
                self._gc_threads()
                self._gc_session_hard_cap()
            except Exception:
                logger.exception("Sandbox lifecycle check failed")

    def _gc_threads(self) -> None:
        """Expire warm entries only; persistent files are never idle-GC'd."""
        with self._condition:
            self._refresh_warm_locked()
            timeout = get_settings().SESSION_IDLE_TTL_SECONDS
            now = time.monotonic()
            expired = [
                sid
                for sid, released in self._released_at.items()
                if timeout > 0 and now - released >= timeout
            ]
            for sid in expired:
                self.stop_session(sid, force=False)

    def _gc_session_hard_cap(self) -> None:
        """Retire old environments at a safe turn boundary instead of killing active work."""
        with self._condition:
            self._refresh_warm_locked()
            now = datetime.now(timezone.utc)
            maximum = get_settings().SESSION_MAX_LIFETIME_SECONDS
            for sid, info in list(self.sessions.items()):
                if maximum > 0 and (now - info.created_at).total_seconds() >= maximum:
                    info.retire_requested = True
                if info.retire_requested and info.status == "warm":
                    self.stop_session(sid, force=False)

    def _touch_session(self, session_id: str) -> None:
        """Update the last_activity timestamp for a session."""
        info = self.sessions.get(session_id)
        if info:
            info.last_activity = datetime.now(timezone.utc)

    # ── Container Health ───────────────────────────────

    def _inspect_container_health(self, session_id: str) -> str | None:
        """Check if the container is still running. Returns a human-readable
        death reason or None if the container is healthy."""
        container = self._containers.get(session_id)
        if container is None:
            return "container not found (already removed)"

        try:
            container.reload()
        except docker.errors.NotFound:
            self._mark_session_dead(session_id)
            return "container no longer exists (removed externally)"
        except Exception as exc:
            return f"unable to inspect container: {exc}"

        status = container.status
        if status == "running" or (status == "paused" and session_id in self._quiesced):
            return None

        state = container.attrs.get("State", {})
        oom = state.get("OOMKilled", False)
        exit_code = state.get("ExitCode")

        if oom:
            reason = "OOM-killed — the code exhausted the container memory limit"
        elif exit_code == 137:
            reason = "killed by signal 9 (SIGKILL) — likely OOM or external kill"
        elif exit_code == 139:
            reason = "segmentation fault (SIGSEGV)"
        elif exit_code == 143:
            reason = "terminated by SIGTERM"
        else:
            reason = f"exited with status {exit_code} (container status: {status})"

        self._mark_session_dead(session_id)
        return reason

    def _mark_session_dead(self, session_id: str) -> None:
        if session_id in self.sessions:
            self.sessions[session_id].status = "dead"

    def _assert_container_alive(self, session_id: str) -> None:
        """Raise ContainerDiedError if the container is not running."""
        reason = self._inspect_container_health(session_id)
        if reason is not None:
            state = self._containers.get(session_id)
            exit_code = None
            if state is not None:
                try:
                    exit_code = state.attrs.get("State", {}).get("ExitCode")
                except Exception:
                    logger.debug("Could not read container exit code", exc_info=True)
            raise ContainerDiedError(session_id, reason, exit_code)

    # ── Kernel Communication ───────────────────────────

    def _send_to_kernel(self, session_id: str, payload: dict, timeout: int = 35) -> dict:
        self._assert_container_alive(session_id)

        info = self.sessions[session_id]
        config = RUNTIME_CONFIG[info.runtime]
        container_name = info.container_name

        try:
            result = subprocess.run(
                ["docker", "exec", "-i", container_name, *config["client_cmd"]],
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Execution in session '{session_id}' timed out after {timeout}s")

        if result.returncode != 0:
            health = self._inspect_container_health(session_id)
            if health is not None:
                raise ContainerDiedError(session_id, health)
            raise RuntimeError(
                f"docker exec failed (rc={result.returncode}): {result.stderr[:500]}"
            )

        raw = (result.stdout or "").strip()
        if not raw:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "result": None,
                "error": {
                    "type": "RuntimeError",
                    "message": "Kernel returned empty response",
                },
                "display_outputs": [],
            }
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "success": False,
                "stdout": "",
                "stderr": (result.stderr or "")[:500],
                "result": None,
                "error": {
                    "type": "RuntimeError",
                    "message": f"Kernel returned invalid JSON: {raw[:200]!r}",
                },
                "display_outputs": [],
            }

    def _wait_for_kernel(self, session_id: str, timeout: int = 60) -> None:
        deadline = time.time() + timeout
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                resp = self._send_to_kernel(session_id, {"action": "ping"}, timeout=5)
                if resp.get("success"):
                    return
            except ContainerDiedError:
                raise
            except Exception as e:
                last_err = e
            time.sleep(0.5)
        raise TimeoutError(f"Kernel did not start in {timeout}s: {last_err}")

    # ── Public API (Tools) ─────────────────────────────

    def _workspace_lock(self, owner, thread, runtime, execution_scope=None):
        key = _scoped_key(owner, thread, runtime, execution_scope=execution_scope)
        with self._condition:
            return self._workspace_locks.setdefault(key, threading.RLock())

    def create_session(
        self,
        runtime: str = "python",
        dependencies: dict[str, str] | None = None,
        mem_limit: str | None = None,
        cpu_quota: int | None = None,
    ) -> SessionInfo:
        """Acquire a healthy scoped runtime, or recreate it over durable files."""
        if runtime not in RUNTIME_CONFIG:
            raise ValueError(f"Runtime '{runtime}' not supported. Use: {list(RUNTIME_CONFIG)}")
        _validate_package_names(dependencies or {})
        settings = get_settings()
        owner, thread = self._owner(), current_thread_id.get(None)
        execution_scope = current_execution_scope.get()
        generation_key = (owner, thread, execution_scope)
        deadline = time.monotonic() + settings.SANDBOX_QUEUE_TIMEOUT_SECONDS
        with self._condition:
            generation = (
                self._cancel_generations.get((owner, thread), 0),
                self._cancel_generations.get(generation_key, 0),
            )
            if self._waiters >= settings.SANDBOX_MAX_WAITERS + 1:
                raise RuntimeError("Sandbox acquisition queue is full")
            self._waiters += 1
            self._scope_acquisitions[generation_key] = (
                self._scope_acquisitions.get(generation_key, 0) + 1
            )

        def check_acquisition():
            with self._condition:
                if self._closed:
                    raise RuntimeError("Sandbox manager is shutting down")
                if thread and (owner, thread) in self._deleted_scopes:
                    raise PermissionError("Conversation was deleted")
                if generation_key in self._deleted_execution_scopes:
                    raise PermissionError("Task execution was closed")
                if (
                    self._cancel_generations.get((owner, thread), 0),
                    self._cancel_generations.get(generation_key, 0),
                ) != generation:
                    raise RuntimeError("Sandbox acquisition was cancelled")

        def reuse_existing():
            if thread:
                with self._condition:
                    self._refresh_warm_locked()
                    candidates = [
                        i
                        for i in self.sessions.values()
                        if i.status != "starting"
                        and (i.user_id, i.thread_id, i.runtime, i.execution_scope)
                        == (owner, thread, runtime, execution_scope)
                    ]
                for info in candidates:
                    with self._condition:
                        # A direct tool operation can be active without a turn lease.
                        # Never replace it with a second writer on the same workspace.
                        while (
                            info.session_id in self.sessions
                            and info.retire_requested
                            and not self._has_turn_locked(info)
                            and self._active_ops.get(info.session_id, 0)
                        ):
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError("Retiring sandbox is still active")
                            self._condition.wait(timeout=min(remaining, 0.25))
                            if self._closed:
                                raise RuntimeError("Sandbox manager is shutting down")
                            check_acquisition()
                            if (owner, thread) in self._deleted_scopes:
                                raise PermissionError("Conversation was deleted")
                        if info.session_id not in self.sessions:
                            continue
                        if self._inspect_container_health(info.session_id) is not None:
                            self.stop_session(info.session_id)
                            continue
                        if info.retire_requested and not self._has_turn_locked(info):
                            if not self.stop_session(info.session_id, force=False):
                                raise RuntimeError("Retiring sandbox could not be released")
                            continue
                    try:
                        with self._operation(info.session_id):
                            missing = {
                                n: v
                                for n, v in (dependencies or {}).items()
                                if n not in info.dependencies or (v and info.dependencies[n] != v)
                            }
                            if missing:
                                self._install_initial_packages(info.session_id, missing)
                                check_acquisition()
                                info.dependencies.update(missing)
                                self._save_environment(info)
                            check_acquisition()
                    except ValueError:
                        # Idle GC may remove the candidate between inspection and
                        # operation pinning. Recreate it instead of failing acquisition.
                        with self._condition:
                            if info.session_id in self.sessions:
                                raise
                        continue
                    return replace(info, reused=True, environment_reset=False)
            return None

        acquired = False
        workspace_lock = None
        try:
            # Reuse must not wait behind a creator blocked on full capacity.
            check_acquisition()
            info = reuse_existing()
            if info is not None:
                return info
            acquired = self._create_lock.acquire(timeout=max(0, deadline - time.monotonic()))
            if not acquired:
                raise TimeoutError("Timed out waiting for sandbox creation")
            check_acquisition()
            info = reuse_existing()
            if info is not None:
                return info
            with self._condition:
                while True:
                    check_acquisition()
                    if self._closed:
                        raise RuntimeError("Sandbox manager is shutting down")
                    self._refresh_warm_locked()
                    scoped = (
                        [
                            i
                            for i in self.sessions.values()
                            if (i.user_id, i.thread_id) == (owner, thread)
                        ]
                        if thread
                        else []
                    )
                    full = len(self.sessions) >= settings.CONTAINER_MAX_SESSIONS
                    scope_full = len(scoped) >= settings.CONTAINER_MAX_SESSIONS_PER_THREAD
                    if not full and not scope_full:
                        break
                    warm = [
                        i
                        for i in (scoped if scope_full else self.sessions.values())
                        if i.status == "warm" and not self._active_ops.get(i.session_id)
                    ]
                    if warm:
                        oldest = min(warm, key=lambda i: self._released_at.get(i.session_id, 0))
                        self.stop_session(oldest.session_id, force=False)
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("All sandbox slots are active; acquisition timed out")
                    self._condition.wait(timeout=min(remaining, 0.25))
            self._ensure_image(runtime)
            check_acquisition()
            candidate_lock = self._workspace_lock(owner, thread, runtime, execution_scope)
            if not candidate_lock.acquire(timeout=max(0, deadline - time.monotonic())):
                raise TimeoutError("Workspace file transfer is still finishing")
            workspace_lock = candidate_lock
            check_acquisition()
            # A cancelled generation may still be finishing a host-side transfer
            # after its container was removed. Do not introduce a new writer yet.
            with self._condition:
                while self._scope_operations.get(
                    _scoped_key(owner, thread, runtime, execution_scope=execution_scope), 0
                ):
                    check_acquisition()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Previous sandbox operation is still finishing")
                    self._condition.wait(timeout=min(remaining, 0.25))
            self._remove_stale_scope_containers(owner, thread, runtime, execution_scope)
            check_acquisition()
            sid = uuid.uuid4().hex
            scope = self._get_storage().scope(
                owner, thread or sid, runtime, execution_scope=execution_scope
            )
            environment = scope.execution_root / ("environment-" + runtime + ".json")
            reset = environment.exists() or any(scope.workspace.iterdir())
            previous = (
                json.loads(environment.read_text(encoding="utf-8")) if environment.exists() else {}
            )
            packages = {**previous.get("dependencies", {}), **(dependencies or {})}
            _validate_package_names(packages)
            check_acquisition()
            name = f"sandbox-{sid}"
            size = settings.CONTAINER_TMPFS_SIZE
            container = self.client.containers.run(
                RUNTIME_CONFIG[runtime]["image"],
                name=name,
                detach=True,
                mem_limit=mem_limit or settings.CONTAINER_MEMORY_LIMIT,
                memswap_limit=mem_limit or settings.CONTAINER_MEMORY_LIMIT,
                cpu_period=100_000,
                cpu_quota=cpu_quota or settings.CONTAINER_CPU_QUOTA,
                pids_limit=settings.CONTAINER_PIDS_LIMIT,
                network_disabled=not settings.CONTAINER_NETWORK_ENABLED,
                volumes={
                    str(scope.workspace): {"bind": "/workspace", "mode": "rw"},
                    str(scope.uploads): {"bind": "/mnt/uploads", "mode": "ro"},
                },
                tmpfs={
                    "/tmp": f"size={size},nosuid,uid={SANDBOX_UID},gid={SANDBOX_GID}",
                    "/home/sandbox": f"size={size},uid={SANDBOX_UID},gid={SANDBOX_GID}",
                },
                cap_drop=["ALL"],
                cap_add=["CHOWN", "DAC_OVERRIDE", "FOWNER", "KILL", "SETGID", "SETUID"],
                security_opt=["no-new-privileges"],
                labels={
                    "taskharbor": "true",
                    "session-id": sid,
                    "manager-id": self._manager_id,
                    "user-id": owner,
                    "thread-id": thread or "",
                    "runtime": runtime,
                    "execution-scope": execution_scope or "",
                    "storage-id": str(settings.STORAGE_DIR),
                },
            )
            info = SessionInfo(
                session_id=sid,
                container_id=container.id,
                container_name=name,
                runtime=runtime,
                status="starting",
                thread_id=thread,
                user_id=owner,
                execution_scope=execution_scope,
                workspace=str(scope.workspace),
                dependencies=packages,
                environment_reset=reset,
            )
            with self._condition:
                if (
                    self._closed
                    or (owner, thread) in self._deleted_scopes
                    or generation_key in self._deleted_execution_scopes
                    or (
                        self._cancel_generations.get((owner, thread), 0),
                        self._cancel_generations.get(generation_key, 0),
                    )
                    != generation
                ):
                    container.remove(force=True)
                    raise RuntimeError("Sandbox acquisition was cancelled")
                self._containers[sid] = container
                self.sessions[sid] = info
                self._execution_locks[sid] = threading.RLock()
                self._active_ops[sid] = 1
                if thread:
                    self.thread_sessions.setdefault(thread, set()).add(sid)
            try:
                self._wait_for_kernel(sid)
                check_acquisition()
                if packages:
                    self._install_initial_packages(sid, packages)
                check_acquisition()
                self._save_environment(info)
                with self._condition:
                    check_acquisition()
                    info.status = "running"
                    self._active_ops.pop(sid, None)
                    self._refresh_warm_locked()
                return info
            except BaseException:
                try:
                    self.stop_session(sid)
                finally:
                    with self._condition:
                        remaining = self._active_ops.get(sid, 1) - 1
                        if remaining > 0:
                            self._active_ops[sid] = remaining
                        else:
                            self._active_ops.pop(sid, None)
                        self._refresh_warm_locked()
                        self._condition.notify_all()
                raise
        finally:
            if workspace_lock is not None:
                workspace_lock.release()
            if acquired:
                self._create_lock.release()
            with self._condition:
                self._waiters -= 1
                remaining = self._scope_acquisitions[generation_key] - 1
                if remaining:
                    self._scope_acquisitions[generation_key] = remaining
                else:
                    self._scope_acquisitions.pop(generation_key, None)
                self._release_ownership_if_idle_locked()
                self._condition.notify_all()

    def _save_environment(self, info: SessionInfo) -> None:
        scope = self._storage_scope(info)
        target = scope.execution_root / ("environment-" + info.runtime + ".json")
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps({"dependencies": info.dependencies}), encoding="utf-8")
        temporary.replace(target)

    def _remove_stale_scope_containers(
        self, owner: str, thread: str | None, runtime: str, execution_scope: str | None = None
    ) -> None:
        """Exclusive deployment ownership permits clearing abandoned same-scope writers."""
        if thread is None:
            return  # Unscoped direct calls always receive a fresh workspace.
        containers = self.client.containers.list(
            all=True,
            filters={
                "label": [
                    "taskharbor=true",
                    f"storage-id={get_settings().STORAGE_DIR}",
                    f"user-id={owner}",
                    f"thread-id={thread}",
                    f"runtime={runtime}",
                ]
            },
        )
        for container in containers:
            # Legacy root containers omit the label; task writers must never
            # be removed while recovering a different execution's workspace.
            if (container.labels.get("execution-scope") or None) != execution_scope:
                continue
            if container.id in {c.id for c in self._containers.values()}:
                raise RuntimeError("A runtime for this workspace is already managed")
            # Without a verified marker, refuse a second mount instead of guessing ownership.
            if not re.fullmatch(r"[0-9a-f]{32}", container.labels.get("manager-id", "")):
                raise RuntimeError("An unowned sandbox already uses this workspace")
            container.remove(force=True)

    @_session_operation
    def execute_code(
        self, session_id: str, code: str, timeout: int | None = None
    ) -> ExecutionResult:
        self._check_session(session_id)
        self._touch_session(session_id)
        settings = get_settings()
        timeout = timeout or settings.CONTAINER_EXECUTION_TIMEOUT_SECONDS

        try:
            resp = self._send_to_kernel(
                session_id,
                {"action": "execute", "code": code, "timeout": timeout},
                timeout=timeout + 10,
            )
        except TimeoutError:
            self.stop_session(session_id)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                result=None,
                error={
                    "type": "TimeoutError",
                    "message": f"Execution timed out after {timeout}s",
                },
                display_outputs=[],
            )

        if "stdout" in resp:
            resp["stdout"] = truncate_field(resp["stdout"], settings.MAX_STDOUT_CHARS)
        if "stderr" in resp:
            resp["stderr"] = truncate_field(resp["stderr"], settings.MAX_STDERR_CHARS)
        if "result" in resp and isinstance(resp["result"], dict):
            for key in list(resp["result"]):
                val = resp["result"][key]
                if isinstance(val, str):
                    resp["result"][key] = truncate_field(val, settings.MAX_RESULT_CHARS)

        error_type = str((resp.get("error") or {}).get("type", ""))
        if "timeout" in error_type.lower():
            # Kernel alarms do not guarantee termination of spawned children.
            self.stop_session(session_id)
        return ExecutionResult(**resp)

    @_session_operation
    def execute_terminal(
        self, session_id: str, command: str, timeout: int | None = None
    ) -> TerminalResult:
        self._assert_container_alive(session_id)
        settings = get_settings()
        runtime = self.sessions[session_id].runtime
        user = "root" if _runtime_setting(runtime, "CONTAINER_EXECUTE_AS_ROOT") else None
        result = self._terminal_executor.run(
            self._containers[session_id],
            command,
            session_id=session_id,
            timeout=timeout or settings.CONTAINER_TERMINAL_TIMEOUT_SECONDS,
            user=user,
        )
        if result.cleanup_failed:
            self.stop_session(session_id)
        return TerminalResult(
            execution_id=result.execution_id,
            exit_code=result.exit_code,
            stdout=truncate_field(result.stdout, settings.MAX_STDOUT_CHARS),
            stderr=truncate_field(result.stderr, settings.MAX_STDERR_CHARS),
            timed_out=result.timed_out,
            cancelled=result.cancelled,
            output_truncated=result.output_truncated,
        )

    def cancel_execution(self, session_id: str) -> bool:
        """Cancel all work in one runtime, retaining persistent files."""
        self._check_session(session_id)
        self._terminal_executor.cancel(session_id)
        return self.stop_session(session_id)

    def _install_initial_packages(self, session_id: str, packages: dict[str, str]) -> None:
        """Install packages during session creation.

        Respects per-runtime ``override_settings`` and the global
        ``CONTAINER_EXECUTE_AS_ROOT`` to decide which user runs the install."""
        _validate_package_names(packages)

        info = self.sessions[session_id]
        config = RUNTIME_CONFIG[info.runtime]
        container = self._containers[session_id]

        if info.runtime == "python":
            specs = [f"{n}=={v}" if v else n for n, v in packages.items()]
        elif info.runtime == "r":
            specs = list(packages.keys())
        else:
            specs = [f"{n}@{v}" if v else n for n, v in packages.items()]

        run_as_root = _runtime_setting(info.runtime, "CONTAINER_EXECUTE_AS_ROOT")
        command = shlex.join(config["install_cmd"](specs))
        result = self._terminal_executor.run(
            container,
            command,
            session_id=session_id,
            timeout=get_settings().CONTAINER_INSTALL_TIMEOUT_SECONDS,
            user="root" if run_as_root else None,
        )
        info.stderr = result.stderr
        if result.exit_code != 0 or result.cleanup_failed:
            if result.cleanup_failed:
                self.stop_session(session_id)
            raise RuntimeError(
                f"Dependency installation failed (exit {result.exit_code}): "
                + result.stderr[-2000:]
            )

    @contextmanager
    def _quiesce(self, session_id: str):
        """Freeze all sandbox processes while trusted host code touches its mount.

        A per-session operation lock alone cannot stop background processes from
        swapping paths between validation and copying. Docker pause freezes the
        entire container, including detached children, before any host file IO.
        """
        container = self._containers[session_id]
        with self._condition:
            try:
                container.pause()
            except Exception as exc:
                # No host copy is permitted when the freezer cannot be engaged.
                raise RuntimeError("Unable to freeze sandbox for safe file transfer") from exc
            self._quiesced.add(session_id)
        try:
            container.reload()
            if container.status != "paused":
                raise RuntimeError("Sandbox pause was not confirmed; file transfer denied")
            yield
        finally:
            try:
                container.unpause()
                container.reload()
                if container.status != "running":
                    raise RuntimeError("Sandbox did not resume after file transfer")
            except Exception as exc:
                # Do not put an uncertain or permanently frozen environment in
                # the warm pool. Persistent files and publications are retained.
                self.stop_session(session_id)
                raise RuntimeError("Sandbox resume failed; compute environment destroyed") from exc
            finally:
                with self._condition:
                    self._quiesced.discard(session_id)

    def _get_storage(self):
        """Storage outlives all containers, including across backend restarts."""
        from taskharbor.sandbox.storage import WorkspaceStore

        if not hasattr(self, "_storage"):
            self._storage = WorkspaceStore(get_settings().STORAGE_DIR)
        return self._storage

    def _storage_scope(self, info: SessionInfo):
        return self._get_storage().scope(
            info.user_id or self._owner(),
            info.thread_id or info.session_id,
            info.runtime,
            execution_scope=getattr(info, "execution_scope", None),
        )

    def _published_artifact(
        self,
        thread_id: str | None,
        session_id: str,
        path: str,
        *,
        user_id: str | None = None,
        artifact_id: str | None = None,
    ):
        owner = user_id or current_user_id.get(None) or self.resolve_owner(thread_id)
        return self._get_storage().find(
            owner, thread_id or session_id, session_id, path, artifact_id=artifact_id
        )

    @_session_operation
    def import_files(
        self,
        session_id: str,
        files: list[dict[str, str]],
    ) -> ImportResult:
        """Snapshot allowed originals or published artifacts into the workspace.

        Cross-session imports read immutable publications; their source container
        need not still exist. Destinations are always inside /workspace.
        """
        self._check_session(session_id)
        self._touch_session(session_id)
        self._assert_container_alive(session_id)
        with self._quiesce(session_id):
            info = self.sessions[session_id]
            scope = self._storage_scope(info)
            store = self._get_storage()
            results: list[ImportFileResult] = []
            for entry in files:
                source_session = entry.get("session_id")
                source_path = entry.get("path", "")
                source = entry.get("source", "")
                label = f"{source_session}:{source_path}" if source_session else source
                destination = entry.get("destination", "")
                try:
                    if source_session:
                        normalized = self._normalize_container_path(source_path)
                        publication = store.find(
                            info.user_id or self._owner(),
                            info.thread_id or session_id,
                            source_session,
                            normalized,
                            artifact_id=entry.get("artifact_id"),
                        )
                        if publication is None:
                            raise PermissionError(
                                "File not exported in this conversation (use export_files first)"
                            )
                        local_source = publication.payload
                        destination = destination or posixpath.basename(normalized) or "imported"
                        destination = self._normalize_container_path(destination)
                        size = store.copy_to_workspace(scope, local_source, destination)
                    else:
                        if not source:
                            raise ValueError("source or (session_id, path) is required")
                        self._validate_import_source(source, session_id=session_id)
                        local_source = Path(source).absolute()
                        destination = destination or local_source.name
                        destination = self._normalize_container_path(destination)
                        size = store.import_path(scope, local_source, destination)
                    results.append(
                        ImportFileResult(
                            source=label,
                            destination=destination,
                            success=True,
                            size=size,
                        )
                    )
                except Exception as exc:
                    results.append(
                        ImportFileResult(
                            source=label,
                            destination=destination,
                            success=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            return ImportResult(success=all(item.success for item in results), files=results)

    @_session_operation
    def export_files(
        self,
        session_id: str,
        files: list[dict[str, str]],
    ) -> ExportResult:
        """Publish immutable snapshots with durable, container-independent downloads."""
        self._check_session(session_id)
        self._touch_session(session_id)
        self._assert_container_alive(session_id)
        with self._quiesce(session_id):
            info = self.sessions[session_id]
            scope = self._storage_scope(info)
            results: list[ExportFileResult] = []
            for entry in files:
                source = entry.get("source", "")
                try:
                    if not source:
                        raise ValueError("source is required")
                    source = self._normalize_container_path(source)
                    publication = self._get_storage().publish(scope, session_id, source)
                    results.append(
                        ExportFileResult(
                            session_id=session_id,
                            path=source,
                            success=True,
                            size=publication.size,
                            artifact_id=publication.artifact_id,
                        )
                    )
                except Exception as exc:
                    results.append(
                        ExportFileResult(
                            session_id=session_id,
                            path=source,
                            success=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            return ExportResult(success=all(item.success for item in results), files=results)

    def is_file_exported(
        self,
        thread_key: str,
        session_id: str,
        path: str,
        *,
        user_id: str | None = None,
        artifact_id: str | None = None,
    ) -> bool:
        norm = self._normalize_container_path(path)
        if (
            self._published_artifact(
                thread_key, session_id, norm, user_id=user_id, artifact_id=artifact_id
            )
            is not None
        ):
            return True
        return False

    def is_exported_path_file(
        self,
        thread_id: str | None,
        session_id: str,
        path: str,
        *,
        user_id: str | None = None,
        artifact_id: str | None = None,
    ) -> bool:
        publication = self._published_artifact(
            thread_id, session_id, path, user_id=user_id, artifact_id=artifact_id
        )
        if publication is not None:
            return not publication.is_directory
        return False

    def stream_exported_file(
        self,
        thread_id: str | None,
        session_id: str,
        path: str,
        *,
        user_id: str | None = None,
        artifact_id: str | None = None,
    ) -> Generator[bytes, None, None]:
        publication = self._published_artifact(
            thread_id, session_id, path, user_id=user_id, artifact_id=artifact_id
        )
        if publication is not None:
            yield from publication.stream()
            return
        raise ValueError(f"File not exported: {session_id}:{path}")

    def stop_session(self, session_id: str, *, force: bool = True) -> bool:
        """Destroy computation only. Published artifacts and workspaces survive."""
        with self._condition:
            info = self.sessions.get(session_id)
            if info is None:
                return False
            if not force and (
                self._active_ops.get(session_id, 0)
                or self._has_turn_locked(info)
                or info.status == "starting"
            ):
                return False
            container = self._containers.get(session_id)
            # Hold admission until removal completes: the hard cap counts physical containers.
            try:
                executor = getattr(self, "_terminal_executor", None)
                if executor:
                    executor.cancel(session_id)
                if container is not None:
                    container.remove(force=True)
            except docker.errors.NotFound:
                pass
            except Exception:
                # Preserve all tracking, including the execution lock, so existing
                # and future operations remain serialized if removal fails.
                info.retire_requested = True
                if self._active_ops.get(session_id) or self._has_turn_locked(info):
                    info.status = "active"
                    self._released_at.pop(session_id, None)
                else:
                    info.status = "warm"
                    self._released_at[session_id] = time.monotonic()
                logger.exception("Unable to destroy sandbox %s", session_id)
                raise
            finally:
                self._condition.notify_all()
            self._containers.pop(session_id, None)
            self.sessions.pop(session_id, None)
            self._released_at.pop(session_id, None)
            self._execution_locks.pop(session_id, None)
            self._quiesced.discard(session_id)
            if info.thread_id:
                members = self.thread_sessions.get(info.thread_id, set())
                members.discard(session_id)
                if not members:
                    self.thread_sessions.pop(info.thread_id, None)
            return True

    def cleanup_thread_sessions(
        self, thread_id: str, *, user_id: str | None = None, delete_files: bool = False
    ) -> int:
        owner = user_id or current_user_id.get(None) or self.resolve_owner(thread_id)
        with self._condition:
            if delete_files:
                self._deleted_scopes.add((owner, thread_id))
            sids = [
                sid
                for sid, info in self.sessions.items()
                if (info.user_id, info.thread_id) == (owner, thread_id)
            ]
        count = sum(self.stop_session(sid) for sid in sids)
        if delete_files:
            self._get_storage().delete_thread(owner, thread_id)
            self._exported_files.pop(thread_id, None)
            self._db_remove_exported_thread(thread_id)
        return count

    def _check_internal_owner(self, owner: str, thread_id: str) -> None:
        """Internal transfers still require a bound, authoritative conversation."""
        if current_thread_id.get(None) != thread_id or self._owner() != owner:
            raise PermissionError("Task files belong to another conversation")

    @contextmanager
    def _bound_session_scope(self, info: SessionInfo):
        token = current_execution_scope.set(info.execution_scope)
        try:
            yield
        finally:
            current_execution_scope.reset(token)

    def copy_scope_files(
        self,
        source_session_id: str,
        destination_session_id: str,
        files: list[dict[str, str]],
        *,
        source_scope: str | None,
        destination_scope: str | None,
    ) -> ImportResult:
        """Trusted task handoff; never exposes arbitrary host paths to a tool."""
        with self._condition:
            source = self.sessions.get(source_session_id)
            destination = self.sessions.get(destination_session_id)
            if source is None or destination is None:
                raise ValueError("Task transfer requires two live sandbox sessions")
            owner, thread = source.user_id, source.thread_id
            if not owner or not thread:
                raise PermissionError("Task transfer requires conversation ownership")
            self._check_internal_owner(owner, thread)
            if (destination.user_id, destination.thread_id) != (owner, thread):
                raise PermissionError("Task transfer crosses conversation ownership")
            if (
                source.execution_scope != source_scope
                or destination.execution_scope != destination_scope
            ):
                raise PermissionError("Task transfer execution scope does not match")
        with ExitStack() as stack:
            # Stable order prevents concurrent opposite-direction handoffs deadlocking.
            for info in sorted(
                {source_session_id: source, destination_session_id: destination}.values(),
                key=lambda i: i.session_id,
            ):
                stack.enter_context(self._bound_session_scope(info))
                stack.enter_context(self._operation(info.session_id))
                stack.enter_context(self._quiesce(info.session_id))
            store = self._get_storage()
            source_paths = self._storage_scope(source)
            destination_paths = self._storage_scope(destination)
            results = []
            for entry in files:
                path = entry.get("path") or entry.get("source", "")
                target = entry.get("destination") or posixpath.basename(path)
                local_source = store.workspace_path(source_paths, path, exists=True)
                size = store.copy_to_workspace(destination_paths, local_source, target)
                results.append(
                    ImportFileResult(path, self._normalize_container_path(target), True, size)
                )
            return ImportResult(success=True, files=results)

    def copy_thread_files(
        self,
        owner: str,
        source_thread: str,
        destination_thread: str,
        runtime: str,
        files: list[dict[str, str]],
    ) -> ImportResult:
        """Copy a service-authorized parent/child handoff using persistent paths.

        The caller validates the signed relationship and input manifest. This
        layer checks ownership and freezes any live containers touching the paths.
        A completed child's files remain readable after its container is retired.
        """
        caller = current_thread_id.get(None)
        if caller not in {source_thread, destination_thread} or self._owner() != owner:
            raise PermissionError("File transfer requires a bound participant")
        if any(
            self.resolve_owner(thread) != owner for thread in (source_thread, destination_thread)
        ):
            raise PermissionError("File transfer crosses user ownership")
        with ExitStack() as stack:
            # Container creation holds the same workspace lock from mount setup
            # through registration. Snapshot only after all prior creators have
            # finished, and prevent a new writer until the handoff is complete.
            # Capacity waiters hold no workspace lock, so copying needs no slot.
            for thread_id in sorted({source_thread, destination_thread}):
                stack.enter_context(self._workspace_lock(owner, thread_id, runtime))
            with self._condition:
                if any(
                    (owner, thread) in self._deleted_scopes
                    for thread in (
                        source_thread,
                        destination_thread,
                    )
                ):
                    raise PermissionError("Conversation was deleted")
                sessions = sorted(
                    (
                        info
                        for info in self.sessions.values()
                        if info.user_id == owner
                        and info.thread_id in {source_thread, destination_thread}
                        and info.runtime == runtime
                        and info.execution_scope is None
                    ),
                    key=lambda info: info.session_id,
                )
            for info in sessions:
                thread_token = current_thread_id.set(info.thread_id)
                scope_token = current_execution_scope.set(None)
                try:
                    stack.enter_context(self._operation(info.session_id))
                    stack.enter_context(self._quiesce(info.session_id))
                finally:
                    current_execution_scope.reset(scope_token)
                    current_thread_id.reset(thread_token)
            store = self._get_storage()
            source = store.scope(owner, source_thread, runtime, create=False)
            destination = store.scope(owner, destination_thread, runtime)
            results = []
            for entry in files:
                path = entry["path"]
                target = entry.get("destination") or posixpath.basename(path)
                local_source = store.workspace_path(source, path, exists=True)
                size = store.copy_to_workspace(destination, local_source, target)
                results.append(
                    ImportFileResult(
                        path,
                        self._normalize_container_path(target),
                        True,
                        size,
                    )
                )
            return ImportResult(success=True, files=results)

    def stage_execution_scope(
        self,
        owner: str,
        thread_id: str,
        execution_scope: str,
        files: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Snapshot declared task files outside the compute mounts before retirement."""
        from taskharbor.sandbox.storage import checked_path, safe_copy

        self._check_internal_owner(owner, thread_id)
        if not execution_scope:
            raise ValueError("A task execution scope is required")
        store = self._get_storage()
        stage_root = store.staged_scope(owner, thread_id, execution_scope)
        with self._condition:
            infos = {
                sid: info
                for sid, info in self.sessions.items()
                if (info.user_id, info.thread_id, info.execution_scope)
                == (owner, thread_id, execution_scope)
            }
        if files is None:
            files = [
                {"session_id": sid, "path": "/workspace/" + path.name}
                for sid, info in infos.items()
                for path in self._storage_scope(info).workspace.iterdir()
            ]
        selected = {entry.get("session_id") for entry in files}
        if not selected.issubset(infos):
            raise PermissionError("Task output session is unavailable or belongs to another task")
        snapshots = []
        created_directories = []
        try:
            with ExitStack() as stack:
                for sid in sorted(selected):
                    info = infos[sid]
                    stack.enter_context(self._bound_session_scope(info))
                    stack.enter_context(self._operation(sid))
                    stack.enter_context(self._quiesce(sid))
                stage_root.mkdir(parents=True, exist_ok=True)
                for entry in files:
                    sid, path = entry["session_id"], entry["path"]
                    info = infos[sid]
                    source = store.workspace_path(self._storage_scope(info), path, exists=True)
                    # A fresh opaque target makes repeated filenames and output dirs safe.
                    relative = Path(uuid.uuid4().hex) / source.name
                    target = checked_path(stage_root, relative)
                    target.parent.mkdir(parents=True)
                    created_directories.append(target.parent)
                    size = safe_copy(source, target)
                    snapshots.append(
                        {
                            "session_id": sid,
                            "path": self._normalize_container_path(path),
                            "runtime": info.runtime,
                            "staged_path": str(target),
                            "size": size,
                        }
                    )
        except BaseException:
            import shutil

            # Roll back only directories created by this staging attempt.
            for directory in created_directories:
                target = checked_path(stage_root, directory.relative_to(stage_root))
                if target.exists():
                    shutil.rmtree(target)
            raise
        return snapshots

    def import_staged_files(
        self,
        owner: str,
        thread_id: str,
        execution_scope: str,
        destination_session_id: str,
        files: list[dict[str, str]],
    ) -> ImportResult:
        """Copy an authorized task's staged outputs into the caller's sandbox."""
        from taskharbor.sandbox.storage import checked_path

        self._check_internal_owner(owner, thread_id)
        store = self._get_storage()
        stage_root = store.staged_scope(owner, thread_id, execution_scope)
        with self._operation(destination_session_id), self._quiesce(destination_session_id):
            destination = self.sessions[destination_session_id]
            results = []
            for entry in files:
                source = Path(entry["staged_path"])
                try:
                    relative = source.relative_to(stage_root)
                except ValueError as exc:
                    raise PermissionError("Staged file belongs to another task") from exc
                source = checked_path(stage_root, relative, exists=True)
                target = entry.get("destination") or source.name
                size = store.copy_to_workspace(self._storage_scope(destination), source, target)
                results.append(
                    ImportFileResult(
                        str(source), self._normalize_container_path(target), True, size
                    )
                )
            return ImportResult(success=True, files=results)

    def cleanup_execution_scope(self, owner: str, thread_id: str, execution_scope: str) -> None:
        """Retire one task and drain its host I/O before removing mutable files."""
        self._check_internal_owner(owner, thread_id)
        if not execution_scope:
            raise ValueError("Only task execution scopes may be deleted")
        key = (owner, thread_id, execution_scope)
        with self._condition:
            self._deleted_execution_scopes.add(key)
        self.cancel_turn(owner, thread_id, "task-cleanup", execution_scope=execution_scope)
        deadline = time.monotonic() + get_settings().SANDBOX_QUEUE_TIMEOUT_SECONDS
        with self._condition:
            sids = [
                sid
                for sid, info in self.sessions.items()
                if (info.user_id, info.thread_id, info.execution_scope) == key
            ]
            for sid in sids:
                self.stop_session(sid)
            for lease in list(self._turn_leases):
                if (
                    lease[:2] == (owner, thread_id)
                    and len(lease) == 4
                    and lease[3] == execution_scope
                ):
                    self._turn_leases.pop(lease, None)
            while self._scope_acquisitions.get(key) or any(
                op[:2] == (owner, thread_id) and len(op) == 4 and op[3] == execution_scope
                for op in self._scope_operations
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Task sandbox cleanup is still waiting for active I/O")
                self._condition.wait(timeout=min(remaining, 0.1))
            self._get_storage().delete_execution_scope(owner, thread_id, execution_scope)

    def remove_staged_execution_scope(
        self, owner: str, thread_id: str, execution_scope: str
    ) -> None:
        self._check_internal_owner(owner, thread_id)
        self._get_storage().delete_staged_scope(owner, thread_id, execution_scope)

    def cleanup_all(self) -> None:
        """Shutdown this manager; leave durable data and other managers untouched."""
        with self._condition:
            self._closed = True
            self._gc_stop.set()
            sids = list(self.sessions)
            self._condition.notify_all()
        for sid in sids:
            try:
                self.stop_session(sid)
            except Exception:
                logger.exception("Shutdown cleanup failed for %s", sid)
        self._heartbeat_path.unlink(missing_ok=True)
        with self._condition:
            self._release_ownership_if_idle_locked()

    def _release_ownership_if_idle_locked(self) -> None:
        """Keep exclusive storage ownership until all old workers finish I/O."""
        if (
            self._closed
            and not self.sessions
            and not self._active_ops
            and not self._scope_operations
            and not self._create_lock.locked()
        ):
            self._deployment_lock.release()

    # ── Internal ───────────────────────────────────────

    def _normalize_container_path(self, path: str) -> str:
        """Normalize container path and ensure it is inside /workspace (prevents path traversal).

        Uses posixpath instead of Path.resolve() to avoid resolving against the
        host filesystem — these paths only exist inside the container.
        """
        if not posixpath.isabs(path):
            path = posixpath.join("/workspace", path)
        normalized = posixpath.normpath(path)
        if not (normalized == "/workspace" or normalized.startswith("/workspace/")):
            raise ValueError(f"Path outside /workspace: {path}")
        return normalized

    def _validate_import_source(self, src: str, session_id: str | None = None) -> None:
        """Only the frozen runtime's workspace may be read as mutable host data."""
        from os.path import abspath

        from taskharbor.sandbox.storage import checked_path

        settings = get_settings()
        source = Path(abspath(src))
        checked_path(source.parent, source.name, exists=True)
        owner = self._owner()
        thread = current_thread_id.get(None)
        scope = None
        if session_id is not None:
            info = self.sessions[session_id]
            scope = self._storage_scope(info)
        elif thread:
            scope = self._get_storage().scope(
                owner, thread, create=False, execution_scope=current_execution_scope.get()
            )
        allowed = [Path(abspath(path)) for path in settings.IMPORT_ALLOWED_DIRS]
        if scope is not None:
            allowed.append(scope.uploads)
            if session_id is not None:
                allowed.append(scope.workspace)
        managed = Path(abspath(settings.STORAGE_DIR)) / "users"
        if source.is_relative_to(managed):
            permitted = scope is not None and (
                source.is_relative_to(scope.uploads)
                or (session_id is not None and source.is_relative_to(scope.workspace))
            )
            if not permitted:
                raise PermissionError(
                    "Import belongs to another user, conversation or runtime; "
                    "use a published artifact for cross-runtime transfers"
                )
        elif managed.is_relative_to(source):
            raise PermissionError("Import cannot include managed data for other users or runtimes")
        if any(source == root or source.is_relative_to(root) for root in allowed):
            return
        raise PermissionError(
            "Import denied: source is outside this conversation's uploads/current workspace "
            "and configured IMPORT_ALLOWED_DIRS"
        )

    def _check_session(self, session_id: str) -> None:
        if session_id not in self._containers:
            raise ValueError(
                f"Session '{session_id}' not found. Call create_session to rebuild the runtime; "
                "persistent files remain available, but in-memory variables must be recreated."
            )
        info = self.sessions[session_id]
        thread = current_thread_id.get(None)
        owner = current_user_id.get(None)
        if thread is not None and info.thread_id != thread:
            raise PermissionError("Session belongs to another conversation")
        if owner is not None and info.user_id != owner:
            raise PermissionError("Session belongs to another user")
        if getattr(info, "execution_scope", None) != current_execution_scope.get():
            raise PermissionError("Session belongs to another task execution")

    def _register_cleanup(self) -> None:
        if self._cleanup_registered:
            return
        self._cleanup_registered = True

        atexit.register(self.cleanup_all)

        original_sigterm = signal.getsignal(signal.SIGTERM)
        original_sigint = signal.getsignal(signal.SIGINT)

        def _handle_signal(signum: int, frame: Any) -> None:
            self.cleanup_all()
            handler = original_sigterm if signum == signal.SIGTERM else original_sigint
            if callable(handler):
                handler(signum, frame)
            else:
                raise SystemExit(128 + signum)

        try:
            signal.signal(signal.SIGTERM, _handle_signal)
            signal.signal(signal.SIGINT, _handle_signal)
        except ValueError:
            pass
