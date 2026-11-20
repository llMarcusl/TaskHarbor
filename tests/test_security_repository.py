"""Approval records, decisions, single execution claims, and saved results."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from taskharbor.security.repository import (
    ApprovalConflict,
    ApprovalExpired,
    ApprovalNotFound,
    ApprovalStore,
    InMemoryApprovalStore,
)


@pytest.fixture
def clock():
    return [datetime(2026, 11, 10, tzinfo=UTC)]


@pytest.fixture
def store(clock):
    return InMemoryApprovalStore(clock=lambda: clock[0])


def request(**overrides):
    record = {
        "invocation_key": "invocation-1",
        "owner": "alice",
        "thread_id": "thread-1",
        "run_id": "run-1",
        "tool_call_id": "call-1",
        "tool_name": "execute_host_command",
        "group": "terminal",
        "executor": "host",
        "arguments": {"command": "Get-Date"},
        "request_digest": "digest-1",
        "reason": "Host command requires confirmation",
        "execution_context": {"interpreter": "C:/Windows/PowerShell/powershell.exe"},
    }
    return record | overrides


def prepare(store, **overrides):
    return store.prepare(request(**overrides))


def approve(store, record):
    return store.decide(record["id"], "alice", "thread-1", "allow")


def test_prepare_freezes_parameters_and_reuses_exact_invocation(store):
    original = request()
    row = store.prepare(original)
    original["arguments"]["command"] = "Remove-Item something"
    assert row["arguments"]["command"] == "Get-Date"
    assert row["status"] == "pending"
    assert store.prepare(request())["id"] == row["id"]
    with pytest.raises(ApprovalConflict):
        store.prepare(request(request_digest="changed"))
    with pytest.raises(ApprovalConflict):
        store.prepare(request(owner="bob"))


@pytest.mark.parametrize("run_id", [None, "run-1"])
def test_source_run_id_is_optional(store, run_id):
    assert prepare(store, run_id=run_id)["run_id"] == run_id


def test_missing_run_id_stays_null_instead_of_inventing_an_execution_id(store):
    original = request()
    del original["run_id"]
    assert store.prepare(original)["run_id"] is None
    with pytest.raises(ValueError, match="run_id must be a string or null"):
        prepare(store, run_id=123)


def test_owner_and_thread_isolation(store):
    row = prepare(store)
    assert store.get(row["id"], "bob", "thread-1") is None
    assert store.get(row["id"], "alice", "other-thread") is None
    assert store.list_thread("bob", "thread-1") == []
    with pytest.raises(ApprovalNotFound):
        store.decide(row["id"], "bob", "thread-1", "allow")
    with pytest.raises(ApprovalNotFound):
        store.claim(row["id"], "alice", "other-thread", "digest-1")


def test_decision_is_idempotent_but_cannot_be_reversed(store):
    row = prepare(store)
    assert approve(store, row)["status"] == "approved"
    assert approve(store, row)["status"] == "approved"
    with pytest.raises(ApprovalConflict):
        store.decide(row["id"], "alice", "thread-1", "deny")
    denied = prepare(store, invocation_key="denied")
    store.decide(denied["id"], "alice", "thread-1", "deny")
    assert store.decide(denied["id"], "alice", "thread-1", "deny")["status"] == "denied"
    with pytest.raises(ApprovalConflict):
        store.claim(denied["id"], "alice", "thread-1", "digest-1")


@pytest.mark.parametrize("approved", [False, True])
def test_expiration_is_checked_again_at_decision_and_execution(store, clock, approved):
    row = prepare(store, expires_at=(clock[0] + timedelta(seconds=1)).isoformat())
    if approved:
        approve(store, row)
    clock[0] += timedelta(seconds=1)
    with pytest.raises(ApprovalExpired):
        if approved:
            store.claim(row["id"], "alice", "thread-1", "digest-1")
        else:
            approve(store, row)
    assert store.get(row["id"], "alice", "thread-1")["status"] == "expired"
    assert store.blocked_thread("alice", "thread-1")
    store.finish(row["id"], "expired", {"success": False, "error": "expired"})
    assert not store.blocked_thread("alice", "thread-1")


def test_claim_requires_approval_and_exact_parameters(store):
    row = prepare(store)
    with pytest.raises(ApprovalConflict):
        store.claim(row["id"], "alice", "thread-1", "digest-1")
    approve(store, row)
    with pytest.raises(ApprovalConflict):
        store.claim(row["id"], "alice", "thread-1", "different")
    assert store.claim(row["id"], "alice", "thread-1", "digest-1")["status"] == "executing"
    with pytest.raises(ApprovalConflict):
        store.claim(row["id"], "alice", "thread-1", "digest-1")


def test_concurrent_resumes_only_one_claims_execution(store):
    row = prepare(store)
    approve(store, row)

    def claim_once(_):
        try:
            store.claim(row["id"], "alice", "thread-1", "digest-1")
            return True
        except ApprovalConflict:
            return False

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert sum(executor.map(claim_once, range(16))) == 1


def test_cancel_invalidates_delayed_prepare_and_approved_claim(store):
    row = prepare(store)
    approve(store, row)
    assert store.generation("alice", "thread-1") == 0
    store.cancel_thread("alice", "thread-1")
    assert store.generation("alice", "thread-1") == 1
    assert store.get(row["id"], "alice", "thread-1")["status"] == "cancelled"
    with pytest.raises(ApprovalConflict):
        store.claim(row["id"], "alice", "thread-1", "digest-1")
    with pytest.raises(ApprovalConflict):
        prepare(store, invocation_key="delayed")
    assert prepare(store, invocation_key="new-turn", generation=1)["status"] == "pending"


def test_delete_permanently_revokes_thread_without_affecting_another(store):
    row = prepare(store)
    store.revoke_thread("alice", "thread-1")
    assert store.get(row["id"], "alice", "thread-1")["status"] == "cancelled"
    with pytest.raises(ApprovalConflict):
        store.generation("alice", "thread-1")
    with pytest.raises(ApprovalConflict):
        prepare(store, invocation_key="new", generation=1)
    assert prepare(store, invocation_key="another", thread_id="thread-2")["status"] == "pending"


def test_started_execution_is_not_expired_or_automatically_reclaimed(store, clock):
    row = prepare(store)
    approve(store, row)
    store.claim(row["id"], "alice", "thread-1", "digest-1")
    clock[0] += timedelta(days=1)
    assert store.get(row["id"], "alice", "thread-1")["status"] == "executing"
    assert store.blocked_thread("alice", "thread-1")
    with pytest.raises(ApprovalConflict):
        store.claim(row["id"], "alice", "thread-1", "digest-1")


def test_explicit_cancel_acknowledges_unknown_execution_without_reexecution(store):
    row = prepare(store)
    approve(store, row)
    store.claim(row["id"], "alice", "thread-1", "digest-1")
    store.cancel_thread("alice", "thread-1")
    result = store.get(row["id"], "alice", "thread-1")
    assert result["status"] == "unknown"
    assert result["result"]["error"] == "EXECUTION_STATE_UNKNOWN"
    assert not store.blocked_thread("alice", "thread-1")
    with pytest.raises(ApprovalConflict):
        store.claim(row["id"], "alice", "thread-1", "digest-1")
    store.finish(row["id"], "cancelled", {"cancelled": True})
    assert store.get(row["id"], "alice", "thread-1")["status"] == "cancelled"


@pytest.mark.parametrize("status", ["succeeded", "failed"])
def test_completed_execution_caches_result_and_cannot_be_reclaimed(store, status):
    row = prepare(store)
    approve(store, row)
    store.claim(row["id"], "alice", "thread-1", "digest-1")
    result = {"success": status == "succeeded", "stdout": "中文"}
    store.finish(row["id"], status, result)
    result["stdout"] = "changed"
    assert store.get(row["id"], "alice", "thread-1")["result"]["stdout"] == "中文"
    assert not store.blocked_thread("alice", "thread-1")
    store.finish(row["id"], status, {"different": True})
    assert store.get(row["id"], "alice", "thread-1")["result"]["stdout"] == "中文"
    with pytest.raises(ApprovalConflict):
        store.claim(row["id"], "alice", "thread-1", "digest-1")


def test_original_execution_can_supply_a_late_result_after_unknown(store):
    row = prepare(store)
    approve(store, row)
    store.claim(row["id"], "alice", "thread-1", "digest-1")
    store.finish(row["id"], "unknown", {"message": "The result has not arrived"})
    with pytest.raises(ApprovalConflict):
        store.claim(row["id"], "alice", "thread-1", "digest-1")
    result = {"success": True, "stdout": "Finished by the original execution"}
    store.finish(row["id"], "succeeded", result)
    saved = store.get(row["id"], "alice", "thread-1")
    assert saved["status"] == "succeeded"
    assert saved["result"] == result


def test_failed_call_can_be_retried_as_a_new_invocation(store):
    original = prepare(store)
    approve(store, original)
    store.claim(original["id"], "alice", "thread-1", "digest-1")
    store.finish(original["id"], "failed", {"success": False})
    retry = prepare(store, invocation_key="retry", tool_call_id="call-2")
    assert retry["id"] != original["id"]
    approve(store, retry)
    assert store.claim(retry["id"], "alice", "thread-1", "digest-1")["status"] == "executing"


def test_executor_cleanup_details_replace_unknown_cancellation_placeholder(store):
    row = prepare(store)
    approve(store, row)
    store.claim(row["id"], "alice", "thread-1", "digest-1")
    store.cancel_thread("alice", "thread-1")
    result = {"cancelled": True, "cleanup_failed": True, "stdout": "partial output"}
    store.finish(row["id"], "unknown", result)
    assert store.get(row["id"], "alice", "thread-1")["result"] == result
    store.finish(row["id"], "unknown", None)
    assert store.get(row["id"], "alice", "thread-1")["result"] == result
    with pytest.raises(ApprovalConflict):
        store.claim(row["id"], "alice", "thread-1", "digest-1")


@pytest.mark.parametrize("status", ["denied", "expired", "cancelled", "superseded"])
def test_consumed_nonexecuted_result_preserves_final_status(store, clock, status):
    row = prepare(store)
    if status == "denied":
        store.decide(row["id"], "alice", "thread-1", "deny")
    elif status == "expired":
        clock[0] += timedelta(hours=1)
        store.get(row["id"], "alice", "thread-1")
    elif status == "cancelled":
        store.cancel_thread("alice", "thread-1")
    else:
        store.finish(row["id"], "superseded")
    assert store.blocked_thread("alice", "thread-1")
    store.finish(row["id"], status, {"success": False, "error": status})
    consumed = store.get(row["id"], "alice", "thread-1")
    assert consumed["status"] == status
    assert consumed["result"] == {"success": False, "error": status}
    assert not store.blocked_thread("alice", "thread-1")


def test_approval_cannot_be_marked_successful_without_claim(store):
    row = prepare(store)
    approve(store, row)
    with pytest.raises(ApprovalConflict):
        store.finish(row["id"], "succeeded", {"success": True})


def test_list_is_bounded_to_recent_fifty(store, clock):
    for i in range(55):
        prepare(store, invocation_key=f"key-{i}")
        clock[0] += timedelta(seconds=1)
    rows = store.list_thread("alice", "thread-1")
    assert len(rows) == 50
    assert rows[0]["invocation_key"] == "key-54"


def test_postgres_transactions_survive_reinitialization_without_reclaim():
    dsn = os.getenv("SECURITY_TEST_DSN")
    if not dsn:
        pytest.skip("Set SECURITY_TEST_DSN for isolated PostgreSQL approval acceptance")
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    schema = "security_test_" + uuid4().hex
    try:
        with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    except psycopg.Error as exc:
        pytest.fail(f"PostgreSQL acceptance unavailable ({type(exc).__name__})", pytrace=False)
    try:
        with ConnectionPool(
            dsn,
            kwargs={
                "autocommit": True,
                "row_factory": dict_row,
                "options": f"-c search_path={schema}",
                "connect_timeout": 5,
            },
            min_size=1,
            max_size=4,
            timeout=5,
        ) as pool:
            first = ApprovalStore(pool)
            row = first.prepare(request())
            assert first.get_invocation("invocation-1", "alice", "thread-1")["id"] == row["id"]
            assert first.get_invocation("invocation-1", "bob", "thread-1") is None
            approve(first, row)
            first.claim(row["id"], "alice", "thread-1", "digest-1")
            second = ApprovalStore(pool)
            second.setup()
            assert second.get(row["id"], "alice", "thread-1")["status"] == "executing"
            with pytest.raises(ApprovalConflict):
                second.claim(row["id"], "alice", "thread-1", "digest-1")
            second.cancel_thread("alice", "thread-1")
            assert first.generation("alice", "thread-1") == 1
            assert not second.blocked_thread("alice", "thread-1")
            assert second.get(row["id"], "alice", "thread-1")["status"] == "unknown"

            cleanup_result = {"cancelled": True, "cleanup_failed": True, "stdout": "partial output"}
            first.finish(row["id"], "unknown", cleanup_result)
            first.finish(row["id"], "unknown", None)
            assert second.get(row["id"], "alice", "thread-1")["result"] == cleanup_result
            first.finish(row["id"], "cancelled", {"cancelled": True, "cleanup_failed": False})
            assert second.get(row["id"], "alice", "thread-1")["status"] == "cancelled"

            pending = first.prepare(request(invocation_key="decision-2", generation=1))
            first.decide(pending["id"], "alice", "thread-1", "deny")
            assert first.blocked_thread("alice", "thread-1")
            first.finish(pending["id"], "denied", {"success": False})
            assert not first.blocked_thread("alice", "thread-1")

            expiring = first.prepare(request(invocation_key="expired", generation=1))
            with pool.connection() as conn:
                conn.execute(
                    "UPDATE sandbox_tool_approvals SET expires_at=now()-interval '1 second' "
                    "WHERE id=%s",
                    (expiring["id"],),
                )
            with pytest.raises(ApprovalExpired):
                first.decide(expiring["id"], "alice", "thread-1", "allow")
            assert second.get(expiring["id"], "alice", "thread-1")["status"] == "expired"
            first.finish(expiring["id"], "expired", {"success": False})

            concurrent = first.prepare(request(invocation_key="race", generation=1))
            approve(first, concurrent)

            def claim_once(_):
                try:
                    first.claim(concurrent["id"], "alice", "thread-1", "digest-1")
                    return True
                except ApprovalConflict:
                    return False

            with ThreadPoolExecutor(max_workers=8) as executor:
                assert sum(executor.map(claim_once, range(16))) == 1
            first.finish(concurrent["id"], "succeeded", {"success": True})

            live_pending = first.prepare(request(
                invocation_key="live-pending", generation=1, transport="live",
            ))
            live_executing = first.prepare(request(
                invocation_key="live-executing", generation=1, transport="live",
            ))
            approve(first, live_executing)
            first.claim(live_executing["id"], "alice", "thread-1", "digest-1")
            checkpoint = first.prepare(request(
                invocation_key="checkpoint", generation=1, transport="checkpoint",
            ))
            legacy = first.prepare(request(invocation_key="legacy", generation=1))
            second.cancel_task("alice", "thread-1", "run-1", None, transport="live")
            cancelled = first.get(live_pending["id"], "alice", "thread-1")
            assert cancelled["status"] == "cancelled" and cancelled["result"] is not None
            unknown = first.get(live_executing["id"], "alice", "thread-1")
            assert unknown["status"] == "unknown"
            assert unknown["result"]["error"] == "EXECUTION_STATE_UNKNOWN"
            for saved in (checkpoint, legacy):
                preserved = first.get(saved["id"], "alice", "thread-1")
                assert preserved["status"] == "pending" and preserved["result"] is None
            assert first.generation("alice", "thread-1") == 1
            second.cancel_task("alice", "thread-1", "run-1", None)
            assert not first.blocked_thread("alice", "thread-1")

    finally:
        with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
