"""Authenticated approval endpoints. Recording a decision never runs a tool."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from .repository import (
    ApprovalConflict,
    ApprovalExpired,
    ApprovalNotFound,
    ApprovalUnavailable,
)

logger = logging.getLogger(__name__)


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["allow", "deny"]


def public_approval(record):
    """Do not expose internal identities or executable authorization digests."""
    fields = (
        "id",
        "thread_id",
        "run_id",
        "tool_call_id",
        "batch_index",
        "transport",
        "task_id",
        "tool_name",
        "group",
        "executor",
        "arguments",
        "execution_context",
        "reason",
        "status",
        "created_at",
        "expires_at",
        "result",
    )
    return {key: record[key] for key in fields if key in record}


async def _call_store(get_store, operation, *args, **kwargs):
    try:

        def call():
            return getattr(get_store(), operation)(*args, **kwargs)

        return await asyncio.to_thread(call)
    except ApprovalNotFound as exc:
        raise HTTPException(status_code=404, detail="Approval not found") from exc
    except ApprovalExpired as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except ApprovalConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ApprovalUnavailable as exc:
        raise HTTPException(status_code=503, detail="Approval storage unavailable") from exc
    except Exception as exc:
        # Do not include SQL, credentials, or tool arguments in the HTTP error.
        logger.warning("Approval storage operation failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Approval storage unavailable") from exc


def create_security_router(
    get_store,
    authorized_thread_owner,
    authenticated_owner,
    *,
    on_cancel=None,
    is_run_active=None,
    related_threads=None,
    get_interrupts=None,
):
    """Inject the application's existing authentication and resource ownership checks."""
    router = APIRouter()

    async def execution_threads(owner, parent_thread):
        if related_threads is None:
            return [parent_thread]
        try:
            children = await related_threads(owner, parent_thread)
            return list(dict.fromkeys([parent_thread, *children]))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Task status unavailable") from exc

    async def execution_active(owner, thread_id):
        if is_run_active is None:
            return False
        try:
            return bool(await asyncio.to_thread(is_run_active, owner, thread_id))
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Run status unavailable") from exc

    async def display_records(owner, parent_thread, target_thread, records):
        visible = [row for row in records if not row.get("automatic", False)]
        if not visible:
            return []
        active = await execution_active(owner, target_thread)
        interrupts = []
        if get_interrupts is not None:
            try:
                interrupts = await get_interrupts(owner, target_thread)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail="Checkpoint status unavailable"
                ) from exc
        interrupt_ids = {
            item["value"]["approval_id"]: item["id"]
            for item in interrupts
            if isinstance(item, dict)
            and item.get("id")
            and isinstance(item.get("value"), dict)
            and item["value"].get("approval_id")
        }
        return [
            {
                **public_approval(record),
                "thread_id": parent_thread,
                "transport": "checkpoint",
                "execution_thread_id": target_thread,
                "assistant_id": "taskharbor-worker"
                if target_thread != parent_thread or record.get("task_id")
                else "taskharbor",
                "interrupt_id": interrupt_ids.get(record["id"]),
                "execution_active": active,
            }
            for record in visible
        ]

    @router.get("/threads/{thread_id}/approvals")
    async def list_approvals(thread_id: str, request: Request):
        owner = await authorized_thread_owner(request, thread_id)
        approvals = []
        for target in await execution_threads(owner, thread_id):
            records = await _call_store(get_store, "list_thread", owner, target)
            approvals.extend(await display_records(owner, thread_id, target, records))
        return {"approvals": approvals}

    @router.post("/threads/{thread_id}/approvals/{approval_id}/decision")
    async def decide_approval(
        thread_id: str, approval_id: str, body: ApprovalDecision, request: Request
    ):
        # Recheck the authoritative thread row as well as the stored record.
        # A deleted thread must not be approved while background revoke waits.
        owner = await authorized_thread_owner(request, thread_id)
        for target in await execution_threads(owner, thread_id):
            record = await _call_store(get_store, "get", approval_id, owner, target)
            if record is not None:
                record = await _call_store(
                    get_store, "decide", approval_id, owner, target, body.decision
                )
                return (await display_records(owner, thread_id, target, [record]))[0]
        raise HTTPException(status_code=404, detail="Approval not found")

    @router.post("/threads/{thread_id}/approvals/cancel")
    async def cancel_approvals(thread_id: str, request: Request):
        owner = await authorized_thread_owner(request, thread_id)
        targets = await execution_threads(owner, thread_id)
        for target in targets:
            await _call_store(get_store, "cancel_thread", owner, target)
        if on_cancel is not None:
            for target in targets:
                try:
                    await asyncio.to_thread(on_cancel, owner, target)
                except Exception as exc:
                    logger.warning("Approval cancellation cleanup failed (%s)", type(exc).__name__)
                    raise HTTPException(
                        status_code=503,
                        detail="Approval cancelled; execution cleanup could not be confirmed",
                    ) from exc
        return {"status": "cancelled"}

    return router
