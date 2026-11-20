"""Durable approval decisions and single-consumer execution claims.

Approval decisions are server-side facts, never inferred from a graph resume
payload. PostgreSQL errors propagate so storage outages cannot permit execution.
"""

from __future__ import annotations

import copy
import json
import threading
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from uuid import uuid4

from psycopg.types.json import Jsonb

ACTIVE_STATUSES = {"pending", "approved", "executing"}
UNCONSUMED_STATUSES = {"denied", "expired", "cancelled", "superseded", "unknown"}
FINAL_STATUSES = {"succeeded", "failed", "unknown", "superseded", "cancelled", "denied", "expired"}
CANCELLED_EXECUTION_RESULT = {
    "success": False,
    "error": "EXECUTION_STATE_UNKNOWN",
    "message": "Cancelled waiting; prior action may have executed",
}


def _cancelled_task_result(row):
    """Live child calls have no later graph resume to consume cancellation."""
    return {
        "type": "tool",
        "name": row["tool_name"],
        "tool_call_id": row["tool_call_id"],
        "status": "error",
        "content": json.dumps({
            "success": False, "error": "TASK_CANCELLED",
            "message": "The task was cancelled before this operation executed",
        }),
    }


class ApprovalError(RuntimeError):
    """A decision or execution claim cannot be accepted."""


class ApprovalNotFound(ApprovalError):
    pass


class ApprovalConflict(ApprovalError):
    pass


class ApprovalExpired(ApprovalError):
    pass


class ApprovalUnavailable(ApprovalError):
    pass


def _utc(value=None):
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Approval timestamps must include a timezone")
    return value.astimezone(UTC)


def _record(record, now):
    data = copy.deepcopy(record)
    for key in (
        "invocation_key",
        "owner",
        "thread_id",
        "tool_call_id",
        "tool_name",
        "group",
        "executor",
        "request_digest",
    ):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"Missing approval field: {key}")
    data.setdefault("run_id", None)
    if data["run_id"] is not None and not isinstance(data["run_id"], str):
        raise ValueError("Approval run_id must be a string or null")
    if data.get("transport", "checkpoint") not in {"checkpoint", "live"}:
        raise ValueError("Invalid approval transport")
    if data.get("task_id") is not None and not isinstance(data["task_id"], str):
        raise ValueError("Approval task_id must be a string or null")
    if not isinstance(data.get("arguments"), dict):
        raise ValueError("Approval arguments must be an object")
    if not isinstance(data.get("execution_context", {}), dict):
        raise ValueError("Approval execution_context must be an object")
    # JSON round-trip freezes nested values and refuses arbitrary Python objects.
    data = json.loads(json.dumps(data, allow_nan=False))
    data.setdefault("id", uuid4().hex)
    data.setdefault("reason", "This operation requires confirmation")
    data.setdefault("execution_context", {})
    data.setdefault("generation", 0)
    if not isinstance(data["generation"], int) or data["generation"] < 0:
        raise ValueError("Invalid cancellation generation")
    data["status"] = "pending"
    data["created_at"] = _utc(data.get("created_at") or now).isoformat()
    data["expires_at"] = _utc(data.get("expires_at") or now + timedelta(minutes=30)).isoformat()
    if _utc(data["expires_at"]) <= now:
        raise ApprovalExpired("Approval request has expired")
    data["result"] = None
    return data


def _check_identity(record, owner, thread_id):
    return bool(record and record["owner"] == owner and record["thread_id"] == thread_id)


def _check_existing(existing, requested):
    if (
        existing["owner"] != requested["owner"]
        or existing["thread_id"] != requested["thread_id"]
        or existing["request_digest"] != requested["request_digest"]
        or existing["generation"] != requested["generation"]
    ):
        raise ApprovalConflict("Approval invocation does not match its original request")


def _check_decision(record, decision):
    if decision not in {"allow", "deny"}:
        raise ValueError("Decision must be allow or deny")
    if record is None:
        raise ApprovalNotFound("Approval not found")
    if record["status"] == "expired":
        raise ApprovalExpired("Approval request has expired")
    target = "approved" if decision == "allow" else "denied"
    if record["status"] not in {"pending", target}:
        raise ApprovalConflict("Approval is no longer waiting for this decision")
    return target


def _check_claim(record, digest, scope):
    if record is None:
        raise ApprovalNotFound("Approval not found")
    if record["status"] == "expired":
        raise ApprovalExpired("Approval request has expired")
    if record["request_digest"] != digest:
        raise ApprovalConflict("Approval parameters changed")
    if scope["revoked"] or record["generation"] != scope["generation"]:
        raise ApprovalConflict("Approval was cancelled")
    if record["status"] != "approved":
        raise ApprovalConflict("Approval has not been granted or execution already started")


def _can_finish(row, status):
    if row["status"] == status:
        return True
    if row["status"] in {"pending", "approved"}:
        return status in {"cancelled", "superseded", "expired"}
    if row["status"] in {"executing", "unknown"}:
        return status in {"succeeded", "failed", "unknown", "cancelled"}
    return False


class InMemoryApprovalStore:
    """Thread-safe test implementation; production always uses PostgreSQL."""

    def __init__(self, *, clock=None):
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._records = {}
        self._keys = {}
        self._scopes = {}

    def setup(self):
        pass

    def _scope(self, owner, thread_id):
        return self._scopes.setdefault((owner, thread_id), {"generation": 0, "revoked": False})

    def _expire(self):
        now = _utc(self._clock())
        for row in self._records.values():
            if row["status"] in {"pending", "approved"} and _utc(row["expires_at"]) <= now:
                row["status"] = "expired"

    def generation(self, owner, thread_id):
        with self._lock:
            scope = self._scope(owner, thread_id)
            if scope["revoked"]:
                raise ApprovalConflict("Conversation was deleted")
            return scope["generation"]

    def prepare(self, record):
        with self._lock:
            requested = _record(record, _utc(self._clock()))
            scope = self._scope(requested["owner"], requested["thread_id"])
            if scope["revoked"] or requested["generation"] != scope["generation"]:
                raise ApprovalConflict("Conversation was cancelled or deleted")
            self._expire()
            existing = self._records.get(self._keys.get(requested["invocation_key"]))
            if existing:
                _check_existing(existing, requested)
                return copy.deepcopy(existing)
            if requested["id"] in self._records:
                raise ApprovalConflict("Approval ID already exists")
            self._records[requested["id"]] = requested
            self._keys[requested["invocation_key"]] = requested["id"]
            return copy.deepcopy(requested)

    def get(self, approval_id, owner, thread_id):
        with self._lock:
            self._expire()
            row = self._records.get(approval_id)
            return copy.deepcopy(row) if _check_identity(row, owner, thread_id) else None

    def list_thread(self, owner, thread_id):
        with self._lock:
            self._expire()
            rows = [r for r in self._records.values() if _check_identity(r, owner, thread_id)]
            return copy.deepcopy(sorted(rows, key=lambda r: (
                r["status"] in ACTIVE_STATUSES
                or (r["status"] in UNCONSUMED_STATUSES and r.get("result") is None),
                r["created_at"],
            ), reverse=True)[:50])

    def get_invocation(self, invocation_key, owner, thread_id):
        with self._lock:
            self._expire()
            row = self._records.get(self._keys.get(invocation_key))
            return copy.deepcopy(row) if _check_identity(row, owner, thread_id) else None

    def decide(self, approval_id, owner, thread_id, decision):
        with self._lock:
            row = self.get(approval_id, owner, thread_id)
            status = _check_decision(row, decision)
            self._records[approval_id]["status"] = status
            return copy.deepcopy(self._records[approval_id])

    def claim(self, approval_id, owner, thread_id, request_digest):
        with self._lock:
            row = self.get(approval_id, owner, thread_id)
            _check_claim(row, request_digest, self._scope(owner, thread_id))
            self._records[approval_id]["status"] = "executing"
            return copy.deepcopy(self._records[approval_id])

    def finish(self, approval_id, status, result=None):
        if status not in FINAL_STATUSES:
            raise ValueError("Invalid execution completion status")
        with self._lock:
            row = self._records.get(approval_id)
            if row is None:
                raise ApprovalNotFound("Approval not found")
            if not _can_finish(row, status):
                raise ApprovalConflict("Approval cannot be completed from its current state")
            if (
                row["status"] == status
                and row.get("result") is not None
                and (status != "unknown" or result is None)
            ):
                return
            row["status"] = status
            row["result"] = copy.deepcopy(result)

    def cancel_thread(self, owner, thread_id, *, revoke=False):
        with self._lock:
            scope = self._scope(owner, thread_id)
            scope["generation"] += 1
            scope["revoked"] = scope["revoked"] or revoke
            for row in self._records.values():
                if _check_identity(row, owner, thread_id):
                    if row["status"] in {"pending", "approved"}:
                        row["status"] = "cancelled"
                    elif row["status"] == "executing" or (
                        row["status"] == "unknown" and row.get("result") is None
                    ):
                        row["status"] = "unknown"
                        row["result"] = copy.deepcopy(CANCELLED_EXECUTION_RESULT)

    def revoke_thread(self, owner, thread_id):
        self.cancel_thread(owner, thread_id, revoke=True)

    def cancel_task(self, owner, thread_id, run_id, task_id, *, transport=None):
        """Revoke one run's task without cancelling its siblings or conversation."""
        with self._lock:
            for row in self._records.values():
                if (
                    _check_identity(row, owner, thread_id)
                    and row.get("run_id") == run_id
                    and row.get("task_id") == task_id
                    and (transport is None or row.get("transport", "checkpoint") == transport)
                ):
                    if row["status"] in {"pending", "approved"}:
                        row["status"] = "cancelled"
                        row["result"] = _cancelled_task_result(row)
                    elif row["status"] in {"cancelled", "expired", "denied", "superseded"} and (
                        row.get("result") is None
                    ):
                        row["result"] = _cancelled_task_result(row)
                    elif row["status"] == "executing" or (
                        row["status"] == "unknown" and row.get("result") is None
                    ):
                        row["status"] = "unknown"
                        row["result"] = copy.deepcopy(CANCELLED_EXECUTION_RESULT)

    def blocked_thread(self, owner, thread_id):
        with self._lock:
            self._expire()
            return any(
                _check_identity(row, owner, thread_id)
                and (
                    row["status"] in ACTIVE_STATUSES
                    or (row["status"] in UNCONSUMED_STATUSES and row.get("result") is None)
                )
                for row in self._records.values()
            )


DDL = (
    """CREATE TABLE IF NOT EXISTS sandbox_tool_security_scopes (
        owner_id text NOT NULL, thread_id text NOT NULL,
        generation bigint NOT NULL DEFAULT 0, revoked boolean NOT NULL DEFAULT false,
        PRIMARY KEY (owner_id, thread_id))""",
    """CREATE TABLE IF NOT EXISTS sandbox_tool_approvals (
        id text PRIMARY KEY, invocation_key text NOT NULL UNIQUE,
        owner_id text NOT NULL, thread_id text NOT NULL,
        status text NOT NULL, generation bigint NOT NULL,
        request_digest text NOT NULL, created_at timestamptz NOT NULL,
        expires_at timestamptz NOT NULL, payload jsonb NOT NULL)""",
    """CREATE INDEX IF NOT EXISTS sandbox_tool_approvals_thread
        ON sandbox_tool_approvals (owner_id, thread_id, created_at DESC)""",
)


class ApprovalStore:
    """Synchronous PostgreSQL store; each decision/claim is transactionally serialized."""

    def __init__(self, pool):
        self.pool = pool
        self._setup_lock = threading.Lock()
        self._ready = False

    def setup(self):
        if self._ready:
            return
        with self._setup_lock:
            if self._ready:
                return
            with self.pool.connection() as conn, conn.transaction():
                conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(current_schema()), "
                    "hashtext('sandbox_tool_approvals_v1'))"
                )
                for statement in DDL:
                    conn.execute(statement)
            self._ready = True

    @staticmethod
    def _scope(conn, owner, thread_id):
        conn.execute(
            "INSERT INTO sandbox_tool_security_scopes (owner_id, thread_id) VALUES (%s,%s) "
            "ON CONFLICT DO NOTHING",
            (owner, thread_id),
        )
        return conn.execute(
            "SELECT generation,revoked FROM sandbox_tool_security_scopes "
            "WHERE owner_id=%s AND thread_id=%s FOR UPDATE",
            (owner, thread_id),
        ).fetchone()

    @staticmethod
    def _expire(conn, owner, thread_id):
        conn.execute(
            "UPDATE sandbox_tool_approvals SET status='expired' "
            "WHERE owner_id=%s AND thread_id=%s AND status IN ('pending','approved') "
            "AND expires_at<=now()",
            (owner, thread_id),
        )

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        data = copy.deepcopy(row["payload"])
        data["status"] = row["status"]
        return data

    @staticmethod
    def _find(conn, approval_id, owner, thread_id):
        return conn.execute(
            "SELECT * FROM sandbox_tool_approvals WHERE id=%s AND owner_id=%s "
            "AND thread_id=%s FOR UPDATE",
            (approval_id, owner, thread_id),
        ).fetchone()

    def generation(self, owner, thread_id):
        self.setup()
        with self.pool.connection() as conn, conn.transaction():
            scope = self._scope(conn, owner, thread_id)
        if scope["revoked"]:
            raise ApprovalConflict("Conversation was deleted")
        return scope["generation"]

    def prepare(self, record):
        requested = _record(record, _utc())
        owner, thread_id = requested["owner"], requested["thread_id"]
        self.setup()
        with self.pool.connection() as conn, conn.transaction():
            scope = self._scope(conn, owner, thread_id)
            if scope["revoked"] or requested["generation"] != scope["generation"]:
                raise ApprovalConflict("Conversation was cancelled or deleted")
            self._expire(conn, owner, thread_id)
            inserted = conn.execute(
                "INSERT INTO sandbox_tool_approvals (id,invocation_key,owner_id,thread_id,status,"
                "generation,request_digest,created_at,expires_at,payload) "
                "VALUES (%s,%s,%s,%s,'pending',%s,%s,%s,%s,%s) "
                "ON CONFLICT (invocation_key) DO NOTHING RETURNING *",
                (
                    requested["id"],
                    requested["invocation_key"],
                    owner,
                    thread_id,
                    requested["generation"],
                    requested["request_digest"],
                    requested["created_at"],
                    requested["expires_at"],
                    Jsonb(requested),
                ),
            ).fetchone()
            row = (
                inserted
                or conn.execute(
                    "SELECT * FROM sandbox_tool_approvals WHERE invocation_key=%s FOR UPDATE",
                    (requested["invocation_key"],),
                ).fetchone()
            )
            data = self._decode(row)
            _check_existing(data, requested)
        return data

    def get(self, approval_id, owner, thread_id):
        self.setup()
        with self.pool.connection() as conn, conn.transaction():
            self._expire(conn, owner, thread_id)
            return self._decode(self._find(conn, approval_id, owner, thread_id))

    def list_thread(self, owner, thread_id):
        self.setup()
        with self.pool.connection() as conn, conn.transaction():
            self._expire(conn, owner, thread_id)
            rows = conn.execute(
                "SELECT * FROM sandbox_tool_approvals WHERE owner_id=%s AND thread_id=%s "
                "ORDER BY (status IN ('pending','approved','executing') OR "
                "(status IN ('unknown','denied','expired','cancelled','superseded') AND "
                "(payload->'result' IS NULL OR payload->'result'='null'))) DESC, "
                "created_at DESC LIMIT 50",
                (owner, thread_id),
            ).fetchall()
        return [self._decode(row) for row in rows]

    def get_invocation(self, invocation_key, owner, thread_id):
        self.setup()
        with self.pool.connection() as conn, conn.transaction():
            self._expire(conn, owner, thread_id)
            row = conn.execute(
                "SELECT * FROM sandbox_tool_approvals WHERE invocation_key=%s "
                "AND owner_id=%s AND thread_id=%s",
                (invocation_key, owner, thread_id),
            ).fetchone()
        return self._decode(row)

    def decide(self, approval_id, owner, thread_id, decision):
        self.setup()
        with self.pool.connection() as conn, conn.transaction():
            self._scope(conn, owner, thread_id)
            self._expire(conn, owner, thread_id)
            data = self._decode(self._find(conn, approval_id, owner, thread_id))
            target = _check_decision(data, decision)
            conn.execute(
                "UPDATE sandbox_tool_approvals SET status=%s WHERE id=%s",
                (target, approval_id),
            )
            data["status"] = target
        return data

    def claim(self, approval_id, owner, thread_id, request_digest):
        self.setup()
        with self.pool.connection() as conn, conn.transaction():
            scope = self._scope(conn, owner, thread_id)
            self._expire(conn, owner, thread_id)
            data = self._decode(self._find(conn, approval_id, owner, thread_id))
            _check_claim(data, request_digest, scope)
            conn.execute(
                "UPDATE sandbox_tool_approvals SET status='executing' WHERE id=%s",
                (approval_id,),
            )
            data["status"] = "executing"
        return data

    def finish(self, approval_id, status, result=None):
        if status not in FINAL_STATUSES:
            raise ValueError("Invalid execution completion status")
        self.setup()
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                "SELECT * FROM sandbox_tool_approvals WHERE id=%s FOR UPDATE",
                (approval_id,),
            ).fetchone()
            if row is None:
                raise ApprovalNotFound("Approval not found")
            current = self._decode(row)
            if not _can_finish(current, status):
                raise ApprovalConflict("Approval cannot be completed from its current state")
            # An executor may provide cleanup details after an unknown placeholder.
            if (
                row["status"] == status
                and current.get("result") is not None
                and (status != "unknown" or result is None)
            ):
                return
            payload = row["payload"]
            payload["result"] = result
            conn.execute(
                "UPDATE sandbox_tool_approvals SET status=%s,payload=%s WHERE id=%s",
                (status, Jsonb(payload), approval_id),
            )

    def cancel_thread(self, owner, thread_id, *, revoke=False):
        self.setup()
        with self.pool.connection() as conn, conn.transaction():
            self._scope(conn, owner, thread_id)
            conn.execute(
                "UPDATE sandbox_tool_security_scopes SET generation=generation+1,"
                "revoked=revoked OR %s WHERE owner_id=%s AND thread_id=%s",
                (revoke, owner, thread_id),
            )
            conn.execute(
                "UPDATE sandbox_tool_approvals SET status='cancelled' WHERE owner_id=%s "
                "AND thread_id=%s AND status IN ('pending','approved')",
                (owner, thread_id),
            )
            conn.execute(
                "UPDATE sandbox_tool_approvals SET status='unknown', "
                "payload=jsonb_set(payload,'{result}',%s::jsonb) "
                "WHERE owner_id=%s AND thread_id=%s AND (status='executing' OR "
                "(status='unknown' AND (payload->'result' IS NULL OR payload->'result'='null'))) ",
                (Jsonb(CANCELLED_EXECUTION_RESULT), owner, thread_id),
            )

    def revoke_thread(self, owner, thread_id):
        self.cancel_thread(owner, thread_id, revoke=True)

    def cancel_task(self, owner, thread_id, run_id, task_id, *, transport=None):
        """Cancel only approvals attached to this exact run/task identity."""
        self.setup()
        with self.pool.connection() as conn, conn.transaction():
            self._scope(conn, owner, thread_id)
            conditions = (
                "owner_id=%s AND thread_id=%s "
                "AND payload->>'run_id' IS NOT DISTINCT FROM %s "
                "AND payload->>'task_id' IS NOT DISTINCT FROM %s"
            )
            identity = (owner, thread_id, run_id, task_id)
            if transport is not None:
                conditions += " AND COALESCE(payload->>'transport','checkpoint')=%s"
                identity += (transport,)
            rows = conn.execute(
                "SELECT * FROM sandbox_tool_approvals WHERE " + conditions
                + " AND (status IN ('pending','approved') OR "
                "(status IN ('cancelled','expired','denied','superseded') AND "
                "(payload->'result' IS NULL OR payload->'result'='null'))) FOR UPDATE", identity,
            ).fetchall()
            for row in rows:
                payload = copy.deepcopy(row["payload"])
                payload["result"] = _cancelled_task_result(payload)
                conn.execute(
                    "UPDATE sandbox_tool_approvals SET status=%s,payload=%s WHERE id=%s",
                    (
                        "cancelled" if row["status"] in {"pending", "approved"} else row["status"],
                        Jsonb(payload), row["id"],
                    ),
                )
            conn.execute(
                "UPDATE sandbox_tool_approvals SET status='unknown', "
                "payload=jsonb_set(payload,'{result}',%s::jsonb) WHERE "
                + conditions + " AND (status='executing' OR "
                "(status='unknown' AND (payload->'result' IS NULL OR payload->'result'='null')))",
                (Jsonb(CANCELLED_EXECUTION_RESULT), *identity),
            )

    def blocked_thread(self, owner, thread_id):
        self.setup()
        with self.pool.connection() as conn, conn.transaction():
            self._expire(conn, owner, thread_id)
            row = conn.execute(
                "SELECT 1 FROM sandbox_tool_approvals WHERE owner_id=%s AND thread_id=%s "
                "AND (status IN ('pending','approved','executing') OR "
                "(status IN ('unknown','denied','expired','cancelled','superseded') "
                "AND (payload->'result' IS NULL OR payload->'result'='null'))) LIMIT 1",
                (owner, thread_id),
            ).fetchone()
        return row is not None


@lru_cache(maxsize=1)
def get_approval_store():
    from taskharbor.clients.infra import get_db_pool

    return ApprovalStore(get_db_pool())
