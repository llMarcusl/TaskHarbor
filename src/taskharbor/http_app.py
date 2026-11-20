"""Custom FastAPI app for Aegra — file downloads, uploads, and thread cleanup."""

from __future__ import annotations

import asyncio
import logging
import re
from contextlib import asynccontextmanager, suppress
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, UploadFile
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from taskharbor.crypto import (
    load_encrypted_settings,
    mask_api_key,
    save_encrypted_settings,
)

logger = logging.getLogger(__name__)

_THREAD_DELETE_PATTERN = re.compile(r"^/threads/([a-fA-F0-9\-]+)$")


def _get_manager():
    """Lazy import to avoid circular deps and ensure graph is loaded first."""
    from taskharbor.sandbox import get_manager

    return get_manager()


async def _authenticated_owner(request: Request) -> str:
    from aegra_api.core.auth_deps import require_auth

    user = await require_auth(request)
    return user.identity


async def _authorized_thread_owner(request: Request, thread_id: str) -> str:
    owner = await _authenticated_owner(request)
    try:
        actual_owner = await asyncio.to_thread(_get_manager().resolve_owner, thread_id)
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Thread not found") from exc
    if owner != actual_owner:
        raise HTTPException(status_code=404, detail="Thread not found")
    return owner


def _cleanup_thread_background(thread_id: str, owner: str) -> None:
    try:
        manager = _get_manager()
        count = manager.cleanup_thread_sessions(thread_id, user_id=owner, delete_files=True)
        logger.info(
            "Thread delete cleanup: removed %d sessions for %s",
            count,
            thread_id[:12],
        )
    except Exception:
        logger.warning(
            "Thread delete cleanup failed for %s",
            thread_id[:12],
            exc_info=True,
        )


class ThreadDeleteCleanupMiddleware(BaseHTTPMiddleware):
    """On DELETE /threads/{id}, clean up Docker sessions and storage in background."""

    async def dispatch(self, request: Request, call_next) -> Response:
        match = (
            _THREAD_DELETE_PATTERN.match(request.url.path) if request.method == "DELETE" else None
        )
        owner = None
        children = []
        subagents_enabled = False
        if match:
            # Capture ownership before Aegra removes its authoritative thread row.
            # Aegra still owns the DELETE authorization and response itself.
            try:
                owner = await _authorized_thread_owner(request, match.group(1))
                from taskharbor.settings import get_settings

                subagents_enabled = getattr(get_settings(), "SUBAGENTS_ENABLED", False)
                if subagents_enabled:
                    from taskharbor.agent.subagents.protocol import related_threads

                    children = await related_threads(owner, match.group(1))
            except HTTPException:
                pass
        response = await call_next(request)
        if owner and match and response.status_code in (200, 204):
            try:
                from taskharbor.security.repository import get_approval_store
                from taskharbor.settings import get_settings

                settings = get_settings()
                if getattr(settings, "HOST_COMMANDS_ENABLED", False) or getattr(
                    settings, "TOOL_SECURITY_RULES", ()
                ):
                    await asyncio.to_thread(
                        get_approval_store().revoke_thread, owner, match.group(1)
                    )
                    await asyncio.to_thread(_cancel_security_work, owner, match.group(1))
            except Exception:
                logger.warning("Tool decision cleanup failed after thread deletion", exc_info=True)
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, _cleanup_thread_background, match.group(1), owner)
            from taskharbor.agent.subagents.protocol import schedule_deleted_children

            if subagents_enabled:
                schedule_deleted_children(owner, match.group(1), children)
        return response


@asynccontextmanager
async def lifespan(app):
    from taskharbor.agent.memory.service import get_memory_service
    from taskharbor.settings import get_settings

    settings = get_settings()
    service = None
    reconciler = None
    if (
        settings.SESSION_MEMORY_ENABLED
        or settings.LONG_TERM_MEMORY_ENABLED
        or settings.CONTEXT_COMPACTION_ENABLED
    ):
        service = get_memory_service()
        try:
            await asyncio.to_thread(service.start)
        except Exception as exc:
            logger.warning("Memory workers unavailable at startup (%s)", type(exc).__name__)
    if (
        getattr(settings, "SUBAGENTS_ENABLED", False)
        or getattr(settings, "HOST_COMMANDS_ENABLED", False)
        or getattr(settings, "TOOL_SECURITY_RULES", ())
    ):
        from taskharbor.agent.subagents.protocol import run_reconciler

        reconciler = asyncio.create_task(run_reconciler())
    try:
        yield
    finally:
        if reconciler is not None:
            reconciler.cancel()
            with suppress(asyncio.CancelledError):
                await reconciler
            from taskharbor.agent.subagents.protocol import close_protocol

            await close_protocol()
        if service is not None:
            await asyncio.to_thread(service.close)


app = FastAPI(lifespan=lifespan)
app.add_middleware(ThreadDeleteCleanupMiddleware)


def _cancel_security_work(owner: str, thread_id: str) -> None:
    from taskharbor.security.host import get_host_executor

    get_host_executor().cancel_scope(f"{owner}:{thread_id}")
    _get_manager().cancel_turn(owner, thread_id, "security-cancel")
    from taskharbor.agent.subagents.protocol import schedule_cancel_children

    schedule_cancel_children(owner, thread_id)


from taskharbor.agent.subagents.protocol import (  # noqa: E402
    WorkerResumeConfigMiddleware,
    create_protocol_router,
    related_threads,
    thread_interrupts,
)
from taskharbor.security.api import create_security_router  # noqa: E402
from taskharbor.security.http import SecurityRunMiddleware, _has_active_run  # noqa: E402
from taskharbor.security.repository import get_approval_store  # noqa: E402

app.include_router(create_protocol_router())

app.include_router(
    create_security_router(
        get_approval_store,
        _authorized_thread_owner,
        _authenticated_owner,
        on_cancel=_cancel_security_work,
        is_run_active=_has_active_run,
        related_threads=related_threads,
        get_interrupts=thread_interrupts,
    )
)
app.add_middleware(SecurityRunMiddleware)
app.add_middleware(WorkerResumeConfigMiddleware)


@app.get("/security/config")
async def security_config(request: Request):
    import os

    from taskharbor.settings import get_settings

    await _authenticated_owner(request)
    settings = get_settings()
    return {
        "host_commands_enabled": getattr(settings, "HOST_COMMANDS_ENABLED", False),
        "host_shell": "PowerShell",
        "host_supported": os.name == "nt",
        "default_host_timeout": getattr(settings, "HOST_COMMAND_TIMEOUT_SECONDS", 60),
        "approval_ttl_seconds": getattr(settings, "TOOL_APPROVAL_TTL_SECONDS", 1800),
    }


@app.get("/threads/{thread_id}/history")
async def get_archived_history(
    thread_id: str,
    request: Request,
    after_seq: int = Query(default=0, ge=0),
    limit: int = Query(default=128, ge=1, le=128),
):
    """Raw archive access uses the same authenticated identity as Aegra threads."""
    from aegra_api.core.auth_deps import require_auth

    from taskharbor.agent.memory.service import get_memory_service
    from taskharbor.settings import get_settings

    settings = get_settings()
    if not (
        settings.CONTEXT_COMPACTION_ENABLED
        or settings.SESSION_MEMORY_ENABLED
        or settings.LONG_TERM_MEMORY_ENABLED
    ):
        return {"enabled": False, "messages": [], "next_seq": None}
    user = await require_auth(request)
    service = get_memory_service()
    await asyncio.to_thread(service.start)
    owner = await asyncio.to_thread(service.repository.owner_for_thread, thread_id)
    if not owner or owner != user.identity:
        return JSONResponse(status_code=404, content={"detail": "Thread not found"})
    rows = await asyncio.to_thread(
        service.repository.history_page,
        owner,
        thread_id,
        after_seq,
        limit,
    )
    return {
        "enabled": True,
        "messages": [r["payload"] for r in rows],
        "next_seq": rows[-1]["seq"] if len(rows) == limit else None,
    }


@app.get("/threads/{thread_id}/files/download")
async def download_thread_file(
    thread_id: str,
    request: Request,
    session_id: str = Query(..., description="Session that exported the file"),
    path: str = Query(..., description="Container path (e.g. /workspace/report.pdf)"),
    artifact_id: str | None = Query(default=None, description="Published immutable artifact"),
):
    """Download an owned published artifact, even after its container was removed."""
    owner = await _authorized_thread_owner(request, thread_id)
    manager = _get_manager()
    # FastAPI has already URL-decoded the query; a second decode corrupts '%' names.
    path_decoded = path

    try:
        if not manager.is_file_exported(
            thread_id, session_id, path_decoded, user_id=owner, artifact_id=artifact_id
        ):
            return Response(
                status_code=403,
                content="File not exported or access denied",
            )
    except ValueError:
        return Response(status_code=400, content="Invalid path")

    try:

        def stream():
            yield from manager.stream_exported_file(
                thread_id, session_id, path_decoded, user_id=owner, artifact_id=artifact_id
            )

        basename = path_decoded.rsplit("/", 1)[-1] if "/" in path_decoded else path_decoded
        is_file = manager.is_exported_path_file(
            thread_id, session_id, path_decoded, user_id=owner, artifact_id=artifact_id
        )
        filename = basename if is_file else f"{basename}.zip"
        media_type = "application/octet-stream" if is_file else "application/zip"
        return StreamingResponse(
            stream(),
            media_type=media_type,
            headers={
                "Content-Disposition": "attachment; filename*=UTF-8''" + quote(filename, safe="")
            },
        )
    except (ValueError, RuntimeError) as exc:
        return Response(status_code=403, content=str(exc))


def _get_storage_dir() -> Path:
    from taskharbor.settings import get_settings

    return get_settings().STORAGE_DIR


# ── Frontend settings persistence ──────────────────────────────────────────


class FrontendSettings(BaseModel):
    chatModel: str | None = None
    chatModelProvider: str | None = None
    chatModelApiKey: str | None = None
    chatModelBaseUrl: str | None = None
    supportsVision: bool | None = None


def _load_frontend_settings() -> dict:
    return load_encrypted_settings()


def _backend_defaults() -> dict:
    from taskharbor.settings import get_settings

    s = get_settings()
    return {
        "chatModel": s.CHAT_MODEL,
        "chatModelProvider": s.CHAT_MODEL_PROVIDER,
        "chatModelApiKey": "",
        "chatModelApiKeyHint": mask_api_key(s.CHAT_MODEL_API_KEY),
        "chatModelBaseUrl": s.CHAT_MODEL_BASE_URL or "",
        "supportsVision": s.CHAT_MODEL_SUPPORTS_VISION
        if s.CHAT_MODEL_SUPPORTS_VISION is not None
        else True,
    }


@app.get("/settings")
async def get_frontend_settings():
    """Return persisted frontend settings merged over backend defaults."""
    defaults = _backend_defaults()
    saved = _load_frontend_settings()

    stored_key = saved.get("chatModelApiKey", "")
    hint = mask_api_key(stored_key) if stored_key else defaults.get("chatModelApiKeyHint", "")

    merged = {**defaults, **saved}
    merged["chatModelApiKey"] = ""
    merged["chatModelApiKeyHint"] = hint
    return JSONResponse(merged)


@app.put("/settings")
async def save_frontend_settings(body: FrontendSettings):
    """Persist frontend settings to database (encrypted)."""
    data = body.model_dump(exclude_none=True)

    new_key = data.get("chatModelApiKey", "")
    if not new_key:
        existing = _load_frontend_settings()
        stored = existing.get("chatModelApiKey", "")
        if stored:
            data["chatModelApiKey"] = stored
        else:
            data.pop("chatModelApiKey", None)

    save_encrypted_settings(data)

    plain_key = data.get("chatModelApiKey", "")
    defaults = _backend_defaults()
    result = {**defaults, **data}
    result["chatModelApiKey"] = ""
    result["chatModelApiKeyHint"] = (
        mask_api_key(plain_key) if plain_key else defaults.get("chatModelApiKeyHint", "")
    )
    return JSONResponse(result)


# ── Active sandbox sessions management ─────────────────────────────────────


def _runtime_from_image(image_tag: str) -> str:
    """Extract runtime name from Docker image tag like 'sandbox-python:latest'."""
    name = image_tag.split(":")[0]
    return name.removeprefix("sandbox-") if name.startswith("sandbox-") else name


@app.get("/sessions")
async def list_sessions(request: Request):
    """Return only the authenticated user's tracked sandbox containers."""
    import docker

    owner = await _authenticated_owner(request)
    manager = _get_manager()
    try:
        containers = await asyncio.to_thread(
            manager.client.containers.list,
            filters={"label": ["taskharbor=true", f"user-id={owner}"]},
        )
    except docker.errors.DockerException:
        logger.warning("Failed to list Docker containers", exc_info=True)
        return JSONResponse([])

    result = []
    for container in containers:
        sid = container.labels.get("session-id", container.short_id)
        image_tag = ",".join(container.image.tags) if container.image.tags else ""
        mem_info = manager.sessions.get(sid)
        if mem_info is None or mem_info.user_id != owner:
            continue
        result.append(
            {
                "session_id": sid,
                "container_id": container.short_id,
                "container_name": container.name,
                "runtime": mem_info.runtime if mem_info else _runtime_from_image(image_tag),
                "status": mem_info.status,
                "thread_id": mem_info.thread_id if mem_info else None,
                "created_at": (
                    mem_info.created_at.isoformat()
                    if mem_info
                    else container.attrs.get("Created", "")
                ),
                "last_activity": (mem_info.last_activity.isoformat() if mem_info else None),
            }
        )
    return JSONResponse(result)


@app.delete("/sessions/{session_id}")
async def kill_session(session_id: str, request: Request):
    """Explicit cancellation may remove an owned session, including a busy one."""
    owner = await _authenticated_owner(request)
    manager = _get_manager()
    info = manager.sessions.get(session_id)
    if info is None or info.user_id != owner:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    if info.thread_id:
        await _authorized_thread_owner(request, info.thread_id)
    if await asyncio.to_thread(manager.stop_session, session_id):
        return JSONResponse({"ok": True, "session_id": session_id})
    return JSONResponse({"error": "Session not found"}, status_code=404)


@app.post("/threads/{thread_id}/files/upload")
async def upload_thread_files(thread_id: str, request: Request, files: list[UploadFile]):
    """Upload files to be available for import into sandbox sessions."""
    from taskharbor.sandbox.storage import checked_path

    owner = await _authorized_thread_owner(request, thread_id)
    dest_dir = _get_manager()._get_storage().scope(owner, thread_id, "python").uploads
    dest_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for f in files:
        if not f.filename:
            continue
        relative = PurePosixPath(f.filename.replace("\\", "/"))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or any(":" in part or "\x00" in part for part in relative.parts)
            or not relative.name
        ):
            raise HTTPException(status_code=400, detail="Invalid upload filename")
        safe_relative = Path(*relative.parts)
        file_path = checked_path(dest_dir, safe_relative)
        if file_path.exists():
            file_path = file_path.with_name(
                f"{file_path.stem}-{uuid4().hex[:12]}{file_path.suffix}"
            )
            safe_relative = file_path.relative_to(dest_dir)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        checked_path(dest_dir, safe_relative)
        size = 0
        try:
            with file_path.open("xb") as output:
                while chunk := await f.read(1024 * 1024):
                    output.write(chunk)
                    size += len(chunk)
        except BaseException:
            file_path.unlink(missing_ok=True)
            raise
        results.append(
            {
                "name": str(safe_relative),
                "path": str(file_path.resolve()),
                "size": size,
            }
        )

    return JSONResponse(results)
