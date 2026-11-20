"""PostgreSQL evidence archive, leased work queue and optimistic memory commits.

All writes use application-owned tables. Aegra's thread table is read only to
resolve the server-side owner, never to persist memory or change checkpoints.
"""

from __future__ import annotations

from uuid import uuid4

from psycopg.types.json import Jsonb

from .migration import message_key, migrate_legacy, prepare_legacy
from .schemas import digest

DDL = [
    """CREATE TABLE IF NOT EXISTS sam_memory_events (
        owner_id text NOT NULL, thread_id text NOT NULL, seq bigint NOT NULL,
        message_key text NOT NULL, event_id text NOT NULL, payload jsonb NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (owner_id,thread_id,seq), UNIQUE (owner_id,thread_id,message_key))""",
    """CREATE TABLE IF NOT EXISTS sam_memory_heads (
        owner_id text NOT NULL, thread_id text NOT NULL, end_seq bigint NOT NULL DEFAULT 0,
        PRIMARY KEY (owner_id,thread_id))""",
    """CREATE TABLE IF NOT EXISTS sam_memory_jobs (
        owner_id text NOT NULL, thread_id text NOT NULL, kind text NOT NULL,
        target_seq bigint NOT NULL, cursor_seq bigint NOT NULL DEFAULT 0,
        status text NOT NULL DEFAULT 'pending',
        due_at timestamptz NOT NULL DEFAULT now(), queued_at timestamptz NOT NULL DEFAULT now(),
        lease_token text, lease_until timestamptz, attempts integer NOT NULL DEFAULT 0,
        last_error text, PRIMARY KEY (owner_id,thread_id,kind))""",
    """CREATE INDEX IF NOT EXISTS sam_memory_jobs_ready
        ON sam_memory_jobs (due_at) WHERE status IN ('pending', 'running')""",
    """CREATE TABLE IF NOT EXISTS sam_memory_documents (
        owner_id text NOT NULL, scope text NOT NULL, revision bigint NOT NULL DEFAULT 0,
        body jsonb NOT NULL DEFAULT '{}', updated_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (owner_id, scope))""",
    """CREATE TABLE IF NOT EXISTS sam_memory_versions (
        owner_id text NOT NULL, scope text NOT NULL, revision bigint NOT NULL,
        extraction_id text NOT NULL UNIQUE, body jsonb NOT NULL, metadata jsonb NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (owner_id, scope, revision))""",
]


class LostLease(RuntimeError):
    pass


def scope_for(kind, thread_id):
    return "user" if kind == "long_term" else "session:" + thread_id


class MemoryRepository:
    PAGE_SIZE = 128

    def __init__(self, pool, *, standalone_owner=None):
        self.pool = pool
        self.standalone_owner = standalone_owner

    def setup(self):
        with self.pool.connection() as conn, conn.transaction():
            # IF NOT EXISTS alone can race during concurrent first-time startup.
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext(current_schema()),"
                "hashtext('sam_memory_setup_v1'))"
            )
            legacy = prepare_legacy(conn)
            for statement in DDL:
                conn.execute(statement)
            if legacy:
                migrate_legacy(conn, legacy)

    def owner_for_thread(self, thread_id):
        if self.standalone_owner:
            # Explicit deployment option for standalone trusted, single-user graphs.
            return self.standalone_owner
        with self.pool.connection() as conn:
            row = conn.execute(
                'SELECT user_id FROM "thread" WHERE thread_id = %s', (thread_id,)
            ).fetchone()
        return row["user_id"] if row else None

    def capture(self, owner, thread_id, records):
        """Append unseen raw messages. A shortened state never removes archived rows.

        Lock only this thread's head to allocate contiguous, committed sequence numbers.
        Message IDs are immutable archive identities; no history editing is supported.
        Evidence event_ids remain payload fingerprints for existing memory provenance.
        """
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO sam_memory_heads (owner_id,thread_id) VALUES (%s,%s) "
                "ON CONFLICT DO NOTHING",
                (owner, thread_id),
            )
            end_seq = conn.execute(
                "SELECT end_seq FROM sam_memory_heads WHERE owner_id=%s AND thread_id=%s "
                "FOR UPDATE",
                (owner, thread_id),
            ).fetchone()["end_seq"]
            # Bound lookup parameters even when state still carries a long transcript.
            for start in range(0, len(records), self.PAGE_SIZE):
                batch = records[start : start + self.PAGE_SIZE]
                keys = [message_key(r) for r in batch]
                existing = {
                    r["message_key"]
                    for r in conn.execute(
                        "SELECT message_key FROM sam_memory_events WHERE owner_id=%s "
                        "AND thread_id=%s AND message_key=ANY(%s)",
                        (owner, thread_id, keys),
                    ).fetchall()
                }
                inserts = []
                for record, key in zip(batch, keys, strict=True):
                    if key in existing:
                        continue
                    existing.add(key)
                    end_seq += 1
                    inserts.append(
                        (
                            owner,
                            thread_id,
                            end_seq,
                            key,
                            record["event_id"],
                            Jsonb(record["payload"]),
                        )
                    )
                if inserts:
                    with conn.cursor() as cursor:
                        cursor.executemany(
                            "INSERT INTO sam_memory_events "
                            "(owner_id,thread_id,seq,message_key,event_id,payload) "
                            "VALUES (%s,%s,%s,%s,%s,%s)",
                            inserts,
                        )
            conn.execute(
                "UPDATE sam_memory_heads SET end_seq=%s WHERE owner_id=%s AND thread_id=%s",
                (end_seq, owner, thread_id),
            )
            return end_seq

    def enqueue(self, owner, thread_id, kind, target_seq, debounce, max_delay):
        if target_seq <= 0:
            return
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                """INSERT INTO sam_memory_jobs
                    (owner_id,thread_id,kind,target_seq,due_at)
                VALUES (%s,%s,%s,%s,now() + %s * interval '1 second')
                ON CONFLICT (owner_id,thread_id,kind) DO UPDATE SET
                    target_seq=EXCLUDED.target_seq,
                    status=CASE WHEN sam_memory_jobs.status='running' THEN 'running'
                                ELSE 'pending' END,
                    attempts=CASE WHEN sam_memory_jobs.target_seq < EXCLUDED.target_seq
                                  THEN 0 ELSE sam_memory_jobs.attempts END,
                    queued_at=CASE WHEN sam_memory_jobs.status IN ('done','failed')
                                   THEN now() ELSE sam_memory_jobs.queued_at END,
                    due_at=LEAST(now() + %s * interval '1 second',
                        CASE WHEN sam_memory_jobs.status IN ('done','failed') THEN now()
                             ELSE sam_memory_jobs.queued_at END + %s * interval '1 second')
                WHERE sam_memory_jobs.target_seq < EXCLUDED.target_seq
                """,
                (owner, thread_id, kind, target_seq, debounce, debounce, max_delay),
            )

    def history_page(self, owner, thread_id, after_seq=0, limit=128, until_seq=None):
        """Bounded raw history; caller must authorize the owner/thread pair."""
        with self.pool.connection() as conn:
            return conn.execute(
                "SELECT seq,message_key,event_id,payload,created_at FROM sam_memory_events "
                "WHERE owner_id=%s AND thread_id=%s AND seq>%s "
                "AND (%s::bigint IS NULL OR seq<=%s) ORDER BY seq LIMIT %s",
                (owner, thread_id, after_seq, until_seq, until_seq, min(128, max(1, limit))),
            ).fetchall()

    def message_sequences(self, owner, thread_id, ids):
        result = {}
        with self.pool.connection() as conn:
            for start in range(0, len(ids), 128):
                rows = conn.execute(
                    "SELECT message_key,seq FROM sam_memory_events "
                    "WHERE owner_id=%s AND thread_id=%s AND message_key=ANY(%s)",
                    (owner, thread_id, ["id:" + i for i in ids[start : start + 128]]),
                ).fetchall()
                result.update({r["message_key"][3:]: r["seq"] for r in rows})
        return result

    def session_version(self, owner, thread_id, until_seq):
        with self.pool.connection() as conn:
            return conn.execute(
                "SELECT scope,revision,body,metadata FROM sam_memory_versions "
                "WHERE owner_id=%s AND scope=%s "
                "AND (metadata->'covered_range'->>'end_seq')::bigint<=%s "
                "ORDER BY revision DESC LIMIT 1",
                (owner, "session:" + thread_id, until_seq),
            ).fetchone()

    def save_compaction(self, owner, thread_id, body, metadata, compaction_id):
        """Independent scope; never advances extraction jobs or session memory cursors."""
        scope = "compaction:" + thread_id
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO sam_memory_documents (owner_id,scope) VALUES (%s,%s) "
                "ON CONFLICT DO NOTHING",
                (owner, scope),
            )
            current = conn.execute(
                "SELECT revision FROM sam_memory_documents WHERE owner_id=%s AND scope=%s "
                "FOR UPDATE",
                (owner, scope),
            ).fetchone()
            existing = conn.execute(
                "SELECT revision FROM sam_memory_versions WHERE extraction_id=%s",
                (compaction_id,),
            ).fetchone()
            if existing:
                return {"scope": scope, "revision": existing["revision"]}
            revision = current["revision"] + 1
            conn.execute(
                "INSERT INTO sam_memory_versions "
                "(owner_id,scope,revision,extraction_id,body,metadata) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (owner, scope, revision, compaction_id, Jsonb(body), Jsonb(metadata)),
            )
            conn.execute(
                "UPDATE sam_memory_documents SET body=%s,revision=%s,updated_at=now() "
                "WHERE owner_id=%s AND scope=%s",
                (Jsonb(body), revision, owner, scope),
            )
            return {"scope": scope, "revision": revision}

    def claim(self, lease_seconds, kinds=("session", "long_term")):
        token = str(uuid4())
        with self.pool.connection() as conn, conn.transaction():
            job = conn.execute(
                """SELECT * FROM sam_memory_jobs WHERE kind=ANY(%s) AND (
                    (status='pending' AND due_at<=now()) OR
                    (status='running' AND lease_until<now()))
                ORDER BY due_at FOR UPDATE SKIP LOCKED LIMIT 1""",
                (list(kinds),),
            ).fetchone()
            if job is None:
                return None
            conn.execute(
                """UPDATE sam_memory_jobs SET status='running',lease_token=%s,
                    lease_until=now() + %s * interval '1 second',attempts=attempts+1
                    WHERE owner_id=%s AND thread_id=%s AND kind=%s""",
                (token, lease_seconds, job["owner_id"], job["thread_id"], job["kind"]),
            )
            job.update(status="running", lease_token=token, attempts=job["attempts"] + 1)
            return job

    @staticmethod
    def _key(job):
        return job["owner_id"], job["thread_id"], job["kind"]

    def heartbeat(self, job, lease_seconds):
        with self.pool.connection() as conn:
            result = conn.execute(
                """UPDATE sam_memory_jobs SET lease_until=now() + %s * interval '1 second'
                    WHERE owner_id=%s AND thread_id=%s AND kind=%s AND lease_token=%s
                    AND status='running' AND lease_until>now()""",
                (lease_seconds, *self._key(job), job["lease_token"]),
            )
            if result.rowcount != 1:
                raise LostLease("memory_job_lease_lost")

    def load(self, job):
        """Read the claimed fixed target and bounded memory, never the transcript."""
        return {"end_seq": job["target_seq"]}, self.read(
            job["owner_id"], scope_for(job["kind"], job["thread_id"])
        )

    def pages(self, job, target, after_seq=0):
        """Read a fixed sequence range; release connections before model calls."""
        while after_seq < target["end_seq"]:
            with self.pool.connection() as conn:
                rows = conn.execute(
                    "SELECT seq,event_id,payload,created_at FROM sam_memory_events "
                    "WHERE owner_id=%s AND thread_id=%s AND seq>%s AND seq<=%s "
                    "ORDER BY seq LIMIT %s",
                    (*self._key(job)[:2], after_seq, target["end_seq"], self.PAGE_SIZE),
                ).fetchall()
            if not rows or any(r["seq"] != after_seq + i for i, r in enumerate(rows, 1)):
                raise ValueError("memory_history_gap")
            after_seq = rows[-1]["seq"]
            yield rows

    def read(self, owner, scope, conn=None):
        if conn is None:
            with self.pool.connection() as connection:
                return self.read(owner, scope, connection)
        return conn.execute(
            "SELECT * FROM sam_memory_documents WHERE owner_id=%s AND scope=%s", (owner, scope)
        ).fetchone() or {"revision": 0, "body": {}}

    def read_for_thread(self, thread_id, *, session=True, long_term=True):
        """Resolve ownership server-side; read only committed documents in that scope."""
        with self.pool.connection() as conn:
            owner = self.standalone_owner
            if not owner:
                row = conn.execute(
                    'SELECT user_id FROM "thread" WHERE thread_id = %s', (thread_id,)
                ).fetchone()
                owner = row["user_id"] if row else None
            if not owner:
                return {}
            documents = {}
            if long_term:
                documents["long_term"] = self.read(owner, "user", conn)
            if session:
                documents["session"] = self.read(owner, "session:" + thread_id, conn)
            return documents

    def finish(self, job, body, expected_revision, metadata):
        """Claim and memory document revision checked in the same short transaction.

        A stale document is recomputed against the latest version. No model calls
        run under a database lock. A completed extraction and its cursor are atomic.
        body=None acknowledges a below-threshold observation without advancing coverage.
        """
        scope = scope_for(job["kind"], job["thread_id"])
        with self.pool.connection() as conn, conn.transaction():
            if not self.standalone_owner:
                owner = conn.execute(
                    'SELECT user_id FROM "thread" WHERE thread_id=%s AND user_id=%s FOR KEY SHARE',
                    (job["thread_id"], job["owner_id"]),
                ).fetchone()
                if owner is None:
                    conn.execute(
                        "UPDATE sam_memory_jobs SET status='cancelled',lease_token=NULL,"
                        "lease_until=NULL WHERE owner_id=%s AND thread_id=%s AND kind=%s "
                        "AND lease_token=%s",
                        (*self._key(job), job["lease_token"]),
                    )
                    return False
            current = conn.execute(
                "SELECT * FROM sam_memory_jobs WHERE owner_id=%s AND thread_id=%s AND kind=%s "
                "FOR UPDATE",
                self._key(job),
            ).fetchone()
            if (
                current is None
                or current["status"] != "running"
                or current["lease_token"] != job["lease_token"]
            ):
                return False
            valid_lease = conn.execute(
                "SELECT lease_until>now() AS valid FROM sam_memory_jobs "
                "WHERE owner_id=%s AND thread_id=%s AND kind=%s",
                self._key(job),
            ).fetchone()["valid"]
            if not valid_lease:
                return False
            if body is not None:
                conn.execute(
                    "INSERT INTO sam_memory_documents (owner_id,scope) VALUES (%s,%s) "
                    "ON CONFLICT DO NOTHING",
                    (job["owner_id"], scope),
                )
                doc = conn.execute(
                    "SELECT revision FROM sam_memory_documents WHERE owner_id=%s AND scope=%s "
                    "FOR UPDATE",
                    (job["owner_id"], scope),
                ).fetchone()
                if doc["revision"] != expected_revision:
                    self._release(conn, job)
                    return False
                revision = expected_revision + 1
                covered = {"end_seq": job["target_seq"]}
                extraction_id = digest([*self._key(job), covered, expected_revision])
                metadata = {**metadata, "schema_version": 4, "covered_range": covered}
                conn.execute(
                    "INSERT INTO sam_memory_versions "
                    "(owner_id,scope,revision,extraction_id,body,metadata) "
                    "VALUES (%s,%s,%s,%s,%s,%s)",
                    (job["owner_id"], scope, revision, extraction_id, Jsonb(body), Jsonb(metadata)),
                )
                conn.execute(
                    "UPDATE sam_memory_documents SET revision=%s,body=%s,updated_at=now() "
                    "WHERE owner_id=%s AND scope=%s",
                    (revision, Jsonb(body), job["owner_id"], scope),
                )
            conn.execute(
                """UPDATE sam_memory_jobs SET
                    status=CASE WHEN target_seq>%s THEN 'pending' ELSE 'done' END,
                    cursor_seq=%s,lease_token=NULL,lease_until=NULL,last_error=NULL,attempts=0
                    WHERE owner_id=%s AND thread_id=%s AND kind=%s""",
                (
                    job["target_seq"],
                    job["target_seq"] if body is not None else job["cursor_seq"],
                    *self._key(job),
                ),
            )
            return True

    def _release(self, conn, job):
        conn.execute(
            "UPDATE sam_memory_jobs SET status='pending',lease_token=NULL,lease_until=NULL,"
            "due_at=now(),attempts=0 WHERE owner_id=%s AND thread_id=%s AND kind=%s",
            self._key(job),
        )

    def fail(self, job, error_code, max_attempts):
        with self.pool.connection() as conn:
            conn.execute(
                """UPDATE sam_memory_jobs SET
                    status=CASE WHEN target_seq>%s OR attempts<%s THEN 'pending'
                                ELSE 'failed' END,
                    due_at=now() + %s * interval '1 second',
                    lease_token=NULL,lease_until=NULL,last_error=%s
                    WHERE owner_id=%s AND thread_id=%s AND kind=%s AND lease_token=%s""",
                (
                    job["target_seq"],
                    max_attempts,
                    min(60, 2 ** min(job["attempts"], 6)),
                    error_code,
                    *self._key(job),
                    job["lease_token"],
                ),
            )
