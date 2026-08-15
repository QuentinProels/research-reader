"""Paper metadata store and job queue, on Postgres + pgvector.

The queue is Postgres rather than Redis/Celery: jobs are minutes long and rare, the
database is already here, and `FOR UPDATE SKIP LOCKED` gives safe multi-worker claiming
in one statement. Durability across restarts comes free, which matters -- the previous
in-process thread died with the API process and left rows frozen mid-pipeline forever.

Crash recovery is lease-based. A worker stamps `heartbeat_at` as it goes; anything
non-terminal whose heartbeat has gone stale is assumed dead and requeued, up to
MAX_ATTEMPTS. That covers a killed worker, an OOM, and a machine reboot alike.

All SQL in the project lives in this module.
"""

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.config import settings

TERMINAL_STATUSES = ("ready", "failed")
MAX_ATTEMPTS = 3
LEASE_SECONDS = 300  # a single vision call or reflow block can legitimately take minutes

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    source       TEXT NOT NULL,          -- 'upload' or the original URL
    status       TEXT NOT NULL,          -- queued|parsing|captioning|reflowing|synthesizing|ready|failed
    progress     REAL NOT NULL DEFAULT 0,
    detail       TEXT NOT NULL DEFAULT '',
    error        TEXT,
    pdf_path     TEXT,
    audio_path   TEXT,
    duration_s   DOUBLE PRECISION,
    chapters     JSONB NOT NULL DEFAULT '[]'::jsonb,
    figures      JSONB NOT NULL DEFAULT '[]'::jsonb,
    resume_s     DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL
);

ALTER TABLE papers ADD COLUMN IF NOT EXISTS claimed_by   TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS claimed_at   TIMESTAMPTZ;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS attempts     INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS papers_created_at_idx ON papers (created_at DESC);
-- The claim query's hot path: oldest queued row.
CREATE INDEX IF NOT EXISTS papers_queue_idx ON papers (created_at) WHERE status = 'queued';

CREATE TABLE IF NOT EXISTS workers (
    id        TEXT PRIMARY KEY,
    last_seen TIMESTAMPTZ NOT NULL
);

-- Step 7 lands here: chunk text + embedding, one row per retrievable passage.
CREATE TABLE IF NOT EXISTS chunks (
    id         BIGSERIAL PRIMARY KEY,
    paper_id   TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    text       TEXT NOT NULL,
    embedding  vector(768),
    UNIQUE (paper_id, ordinal)
);
"""

_JSON_COLUMNS = ("chapters", "figures")
_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=8,
            open=True,
            kwargs={"row_factory": dict_row},
        )
    return _pool


@contextmanager
def _conn():
    with _get_pool().connection() as connection:
        yield connection


def init_db() -> None:
    with _conn() as connection:
        connection.execute(SCHEMA)


# --------------------------------------------------------------------------- papers


def create_paper(title: str, source: str) -> str:
    """Insert a paper in 'queued'. A worker picks it up; nothing runs inline."""
    paper_id = uuid.uuid4().hex[:12]
    with _conn() as connection:
        connection.execute(
            "INSERT INTO papers (id, title, source, status, created_at) VALUES (%s, %s, %s, %s, %s)",
            (paper_id, title, source, "queued", datetime.now(timezone.utc)),
        )
    return paper_id


def update_paper(paper_id: str, **fields: Any) -> None:
    if not fields:
        return
    for column in _JSON_COLUMNS:
        if column in fields:
            fields[column] = Jsonb(fields[column])
    assignments = ", ".join(f"{key} = %s" for key in fields)
    with _conn() as connection:
        connection.execute(
            f"UPDATE papers SET {assignments} WHERE id = %s", (*fields.values(), paper_id)
        )


def set_status(paper_id: str, status: str, progress: float, detail: str = "") -> None:
    """Progress update that doubles as the worker's liveness signal."""
    with _conn() as connection:
        connection.execute(
            "UPDATE papers SET status = %s, progress = %s, detail = %s, heartbeat_at = now() "
            "WHERE id = %s",
            (status, progress, detail, paper_id),
        )


def get_paper(paper_id: str) -> dict[str, Any] | None:
    with _conn() as connection:
        return connection.execute("SELECT * FROM papers WHERE id = %s", (paper_id,)).fetchone()


def list_papers() -> list[dict[str, Any]]:
    """Newest first, with a 1-based queue position on anything still waiting."""
    with _conn() as connection:
        papers = connection.execute(
            "SELECT * FROM papers ORDER BY created_at DESC"
        ).fetchall()
    waiting = sorted(
        (p for p in papers if p["status"] == "queued"), key=lambda p: p["created_at"]
    )
    positions = {p["id"]: index + 1 for index, p in enumerate(waiting)}
    for paper in papers:
        paper["queue_position"] = positions.get(paper["id"])
    return papers


def delete_paper(paper_id: str) -> bool:
    with _conn() as connection:
        result = connection.execute("DELETE FROM papers WHERE id = %s", (paper_id,))
        return result.rowcount > 0


def cancel_paper(paper_id: str) -> bool:
    """Only meaningful while queued -- a claimed job is mid-pipeline and left alone."""
    with _conn() as connection:
        result = connection.execute(
            "UPDATE papers SET status = 'failed', error = 'Cancelled', detail = '' "
            "WHERE id = %s AND status = 'queued'",
            (paper_id,),
        )
        return result.rowcount > 0


# ---------------------------------------------------------------------------- queue


def claim_next_job(worker_id: str) -> dict[str, Any] | None:
    """Atomically take the oldest queued paper, or return None.

    SKIP LOCKED means concurrent workers never block each other and never hand the
    same paper to two of them.
    """
    with _conn() as connection:
        return connection.execute(
            """
            UPDATE papers SET
                status       = 'parsing',
                claimed_by   = %s,
                claimed_at   = now(),
                heartbeat_at = now(),
                attempts     = attempts + 1,
                detail       = 'Starting',
                error        = NULL
            WHERE id = (
                SELECT id FROM papers
                WHERE status = 'queued'
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING *
            """,
            (worker_id,),
        ).fetchone()


def heartbeat(paper_id: str) -> None:
    with _conn() as connection:
        connection.execute("UPDATE papers SET heartbeat_at = now() WHERE id = %s", (paper_id,))


def release_job(paper_id: str) -> None:
    """Hand a job back on graceful shutdown, so it restarts now, not after the lease."""
    with _conn() as connection:
        connection.execute(
            "UPDATE papers SET status = 'queued', claimed_by = NULL, claimed_at = NULL, "
            "heartbeat_at = NULL, progress = 0, detail = 'Requeued after worker shutdown' "
            "WHERE id = %s AND status NOT IN ('ready', 'failed')",
            (paper_id,),
        )


def reap_stale_jobs() -> list[str]:
    """Requeue jobs whose worker stopped heartbeating; fail those out of attempts.

    Returns the ids that were requeued.
    """
    with _conn() as connection:
        connection.execute(
            """
            UPDATE papers SET
                status = 'failed',
                error  = 'Gave up after ' || attempts || ' attempts (worker kept dying)',
                detail = ''
            WHERE status NOT IN ('ready', 'failed')
              AND heartbeat_at < now() - make_interval(secs => %s)
              AND attempts >= %s
            """,
            (LEASE_SECONDS, MAX_ATTEMPTS),
        )
        revived = connection.execute(
            """
            UPDATE papers SET
                status       = 'queued',
                claimed_by   = NULL,
                claimed_at   = NULL,
                heartbeat_at = NULL,
                progress     = 0,
                detail       = 'Requeued after interrupted run'
            WHERE status NOT IN ('ready', 'failed')
              AND heartbeat_at < now() - make_interval(secs => %s)
            RETURNING id
            """,
            (LEASE_SECONDS,),
        ).fetchall()
    return [row["id"] for row in revived]


# --------------------------------------------------------------------------- workers


def worker_seen(worker_id: str) -> None:
    with _conn() as connection:
        connection.execute(
            "INSERT INTO workers (id, last_seen) VALUES (%s, now()) "
            "ON CONFLICT (id) DO UPDATE SET last_seen = now()",
            (worker_id,),
        )


def worker_gone(worker_id: str) -> None:
    with _conn() as connection:
        connection.execute("DELETE FROM workers WHERE id = %s", (worker_id,))


def live_workers(within_seconds: int = 60) -> int:
    """Zero means uploads will sit in 'queued' forever -- the UI says so out loud."""
    with _conn() as connection:
        row = connection.execute(
            "SELECT count(*) AS n FROM workers WHERE last_seen > now() - make_interval(secs => %s)",
            (within_seconds,),
        ).fetchone()
    return row["n"]
