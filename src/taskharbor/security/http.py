"""Keep new runs from replacing a conversation waiting for a user decision."""

import asyncio
import re
from weakref import WeakValueDictionary

from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from taskharbor.security.repository import get_approval_store

_RUN = re.compile(r"^/threads/([^/]+)/runs(?:/(?:stream|wait))?$")
_CANCEL = re.compile(r"^/threads/([^/]+)/runs/([^/]+)/cancel$")
_run_locks = WeakValueDictionary()


def thread_run_lock(owner: str, thread: str) -> asyncio.Lock:
    """Serialize public and internal run submissions on the service loop."""
    key = (asyncio.get_running_loop(), owner, thread)
    return _run_locks.setdefault(key, asyncio.Lock())


class RunBusy(RuntimeError):
    pass


def _has_active_run(owner, thread, run_id=None):
    """Read run status; the worker owns and returns its database connection."""
    from taskharbor.clients.infra import get_db_pool

    query = (
        "SELECT 1 FROM runs WHERE user_id=%s AND thread_id=%s AND status IN ('pending','running')"
    )
    parameters = (owner, thread)
    if run_id is not None:
        query += " AND run_id=%s"
        parameters += (run_id,)
    with get_db_pool().connection() as connection:
        return connection.execute(query + " LIMIT 1", parameters).fetchone() is not None


class SecurityRunMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        run = _RUN.match(request.url.path) if request.method == "POST" else None
        cancel = _CANCEL.match(request.url.path) if request.method == "POST" else None
        if not run and not cancel:
            return await call_next(request)
        from taskharbor.http_app import _authorized_thread_owner, _cancel_security_work
        from taskharbor.settings import get_settings

        settings = get_settings()
        security_enabled = bool(
            getattr(settings, "HOST_COMMANDS_ENABLED", False)
            or getattr(settings, "TOOL_SECURITY_RULES", ())
            or getattr(settings, "SUBAGENTS_ENABLED", False)
        )
        try:
            thread = (run or cancel).group(1)
            owner = await _authorized_thread_owner(request, thread)
            body = await request.json() if run else {}
            config = body.get("config") or {}
            for identity in (config.get("configurable") or {}, body.get("context") or {}):
                if identity.get("thread_id", thread) != thread:
                    raise HTTPException(
                        status_code=400, detail="Run conversation does not match its URL"
                    )
                if identity.get("user_id", owner) != owner or "langgraph_auth_user" in identity:
                    raise HTTPException(
                        status_code=400, detail="Run identity is assigned by the server"
                    )
            if not security_enabled:
                return await call_next(request)
            lock = thread_run_lock(owner, thread)
            async with lock:
                if cancel:
                    if await asyncio.to_thread(
                        _has_active_run, owner, thread, run_id=cancel.group(2)
                    ):
                        await asyncio.to_thread(get_approval_store().cancel_thread, owner, thread)
                        await asyncio.to_thread(_cancel_security_work, owner, thread)
                else:
                    if await asyncio.to_thread(_has_active_run, owner, thread):
                        raise RunBusy("This conversation already has an active run")
                    command = body.get("command") or {}
                    if not command.get("resume") and await asyncio.to_thread(
                        get_approval_store().blocked_thread, owner, thread
                    ):
                        raise RunBusy("Resolve or cancel the pending tool decision first")
                return await call_next(request)
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        except RunBusy as exc:
            return JSONResponse({"detail": str(exc)}, status_code=409)
        except Exception:
            return JSONResponse({"detail": "Tool decision storage is unavailable"}, status_code=503)
