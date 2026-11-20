"""One-time, transactional conversion of earlier archives to append-only sequences.

Old source tables are retained under *_pre_append_v4, including inactive branches.
Stop old workers before setup. Unsettled edit jobs fail closed: finish their old
rebuild first instead of silently treating stale memories as current.
"""

from psycopg import sql
from psycopg.types.json import Jsonb

SUFFIX = "_pre_append_v4"


def message_key(record):
    """Prefer stable message IDs; legacy/id-less evidence uses its content identity."""
    message_id = record["payload"].get("id")
    return "id:" + message_id if message_id else "hash:" + record["event_id"]


def prepare_legacy(conn):
    columns = {
        r["column_name"]
        for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema=current_schema() AND table_name='sam_memory_events'"
        ).fetchall()
    }
    if not columns or "message_key" in columns:
        return None
    head_columns = {
        r["column_name"]
        for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema=current_schema() AND table_name='sam_memory_heads'"
        ).fetchall()
    }
    mode = (
        "refs" if "refs" in head_columns else "append" if "last_seq" in head_columns else "ranges"
    )
    for name in ("events", "heads", "jobs", "positions", "snapshots"):
        table = "sam_memory_" + name
        if conn.execute("SELECT to_regclass(%s) AS t", (table,)).fetchone()["t"]:
            conn.execute(
                sql.SQL("ALTER TABLE {} RENAME TO {}").format(
                    sql.Identifier(table), sql.Identifier(table + SUFFIX)
                )
            )
    conn.execute(
        "ALTER INDEX IF EXISTS sam_memory_jobs_ready RENAME TO sam_memory_jobs_ready_pre_append_v4"
    )
    return mode


def _boundaries(conn, mode, head):
    key = head["owner_id"], head["thread_id"]
    jobs = conn.execute(
        "SELECT * FROM sam_memory_jobs_pre_append_v4 WHERE owner_id=%s AND thread_id=%s",
        key,
    ).fetchall()
    for job in jobs:
        if mode == "append":
            target, cursor = job["target_seq"], job["cursor_seq"]
        else:
            snapshot = conn.execute(
                "SELECT * FROM sam_memory_snapshots_pre_append_v4 "
                "WHERE owner_id=%s AND thread_id=%s AND revision=%s",
                (*key, job["desired_revision"]),
            ).fetchone()
            if snapshot is None:
                raise ValueError("legacy_memory_snapshot_missing")
            if mode == "refs":
                target, cursor = len(snapshot["refs"]), len(job["cursor_refs"])
                if (
                    snapshot["refs"] != head["refs"][:target]
                    or job["cursor_refs"] != head["refs"][:cursor]
                ):
                    raise ValueError("legacy_memory_unsettled_edit")
            else:
                target, cursor = snapshot["end_seq"], job["cursor_seq"]
                if snapshot["generation"] != head["generation"] or (
                    cursor and job["cursor_generation"] != head["generation"]
                ):
                    raise ValueError("legacy_memory_unsettled_edit")
        job.update(old_target=target, old_cursor=cursor)
    return jobs


def migrate_legacy(conn, mode):
    # Setup owns one transaction/advisory lock: any invalid range rolls back DDL too.
    with conn.cursor(name="append_migration_heads") as heads:
        heads.execute("SELECT * FROM sam_memory_heads_pre_append_v4")
        for head in heads:
            key = head["owner_id"], head["thread_id"]
            jobs = _boundaries(conn, mode, head)
            end = (
                len(head["refs"])
                if mode == "refs"
                else head["last_seq"]
                if mode == "append"
                else head["end_seq"]
            )
            boundaries = {0: 0, end: None}
            for job in jobs:
                if not 0 <= job["old_cursor"] <= job["old_target"] <= end:
                    raise ValueError("legacy_memory_cursor_invalid")
                boundaries[job["old_cursor"]] = None
                boundaries[job["old_target"]] = None
            boundaries[0] = 0
            if mode == "refs":
                query = """SELECT p.seq,e.event_id,e.payload,e.created_at
                    FROM sam_memory_heads_pre_append_v4 h,
                    jsonb_array_elements_text(h.refs) WITH ORDINALITY AS p(event_id,seq)
                    LEFT JOIN sam_memory_events_pre_append_v4 e
                    ON e.owner_id=%s AND e.thread_id=%s AND e.event_id=p.event_id
                    WHERE h.owner_id=%s AND h.thread_id=%s ORDER BY p.seq"""
                params = (*key, *key)
            elif mode == "ranges":
                query = """SELECT p.seq,e.event_id,e.payload,e.created_at
                    FROM sam_memory_positions_pre_append_v4 p
                    LEFT JOIN sam_memory_events_pre_append_v4 e USING(owner_id,thread_id,event_id)
                    WHERE p.owner_id=%s AND p.thread_id=%s AND p.generation=%s AND p.seq<=%s
                    ORDER BY p.seq"""
                params = (*key, head["generation"], end)
            else:
                query = """SELECT seq,event_id,payload,created_at
                    FROM sam_memory_events_pre_append_v4
                    WHERE owner_id=%s AND thread_id=%s ORDER BY seq"""
                params = key
            count, new_end = 0, 0
            with conn.cursor(name="append_migration_rows") as rows:
                rows.execute(query, params)
                for record in rows:
                    count += 1
                    if record["seq"] != count or record["payload"] is None:
                        raise ValueError("legacy_memory_events_missing")
                    identity = message_key(record)
                    existing = conn.execute(
                        "SELECT event_id FROM sam_memory_events WHERE owner_id=%s AND thread_id=%s "
                        "AND message_key=%s",
                        (*key, identity),
                    ).fetchone()
                    if existing and existing["event_id"] != record["event_id"]:
                        raise ValueError("legacy_memory_conflicting_message_id")
                    if not existing:
                        new_end += 1
                        conn.execute(
                            "INSERT INTO sam_memory_events "
                            "(owner_id,thread_id,seq,message_key,event_id,payload,created_at) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                            (
                                *key,
                                new_end,
                                identity,
                                record["event_id"],
                                Jsonb(record["payload"]),
                                record["created_at"],
                            ),
                        )
                    if count in boundaries:
                        boundaries[count] = new_end
            if count != end:
                raise ValueError("legacy_memory_events_missing")
            conn.execute("INSERT INTO sam_memory_heads VALUES (%s,%s,%s)", (*key, new_end))
            for job in jobs:
                conn.execute(
                    """INSERT INTO sam_memory_jobs
                    (owner_id,thread_id,kind,target_seq,cursor_seq,status,due_at,queued_at,attempts,last_error)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        *key,
                        job["kind"],
                        boundaries[job["old_target"]],
                        boundaries[job["old_cursor"]],
                        "pending" if job["status"] == "running" else job["status"],
                        job["due_at"],
                        job["queued_at"],
                        job["attempts"],
                        job["last_error"],
                    ),
                )
    orphan = conn.execute("""SELECT 1 FROM sam_memory_jobs_pre_append_v4 j
        LEFT JOIN sam_memory_heads h USING(owner_id,thread_id)
        WHERE h.thread_id IS NULL LIMIT 1""").fetchone()
    if orphan:
        raise ValueError("legacy_memory_head_missing")
