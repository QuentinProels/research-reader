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
from datetime import UTC, datetime
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.config import settings

TERMINAL_STATUSES = ("ready", "failed")
MAX_ATTEMPTS = 3
LEASE_SECONDS = 300  # a single vision call or reflow block can legitimately take minutes
MAX_OUTAGE_RETRIES = 8  # roughly two hours of backoff before giving up on a dead server
OUTAGE_BACKOFF_SECONDS = (30, 60, 120, 300, 600, 900, 1800, 1800)

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
ALTER TABLE papers ADD COLUMN IF NOT EXISTS outage_retries  INTEGER NOT NULL DEFAULT 0;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS papers_created_at_idx ON papers (created_at DESC);
-- The claim query's hot path: oldest queued row.
CREATE INDEX IF NOT EXISTS papers_queue_idx ON papers (created_at) WHERE status = 'queued';

CREATE TABLE IF NOT EXISTS workers (
    id        TEXT PRIMARY KEY,
    last_seen TIMESTAMPTZ NOT NULL
);

-- One row per narrated segment: the transcript of the audio, with the timestamp it is
-- spoken at. This is what "what did that mean" retrieves against, and what "repeat that
-- section" seeks by. The embedding column stays null until step 7 fills it.
CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    paper_id      TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    text          TEXT NOT NULL,
    start_s       DOUBLE PRECISION NOT NULL DEFAULT 0,
    chapter_title TEXT NOT NULL DEFAULT '',
    figure_id     TEXT,
    embedding     vector(768),
    UNIQUE (paper_id, ordinal)
);

-- CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so columns
-- added after the first deployment need their own statements.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS start_s       DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chapter_title TEXT NOT NULL DEFAULT '';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS figure_id     TEXT;

CREATE INDEX IF NOT EXISTS chunks_position_idx ON chunks (paper_id, start_s);

-- Single row. Settings live server-side rather than in browser storage so that a choice
-- made at a desk is already in force on the phone in the car, which is the device least
-- convenient to configure.
CREATE TABLE IF NOT EXISTS settings (
    id      BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
    values  JSONB NOT NULL DEFAULT '{}'::jsonb
);
"""

# Every default here is a measurement or a stated preference, not a guess. Changing one
# must change behaviour: a toggle that does nothing is worse than no toggle.
DEFAULT_SETTINGS = {
    "activation": "both",          # tap | handsfree | both
    "vad_sensitivity": "medium",   # low | medium | high
    "echo_rejection": True,        # discard triggers that match the narration transcript
    "answer_depth": "fast",        # fast | thorough  (thorough measured 13x slower, no better)
    "stt_model": "tiny.en",        # tiny.en | base.en | small.en
    "keep_screen_awake": True,     # Wake Lock, so a mounted phone does not sleep mid-drive
    "speak_answers": True,         # read answers aloud as well as showing them
    "auto_resume": True,           # resume narration once an answer finishes
    "offline_download": False,     # cache the audio for dead spots
}

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
            (paper_id, title, source, "queued", datetime.now(UTC)),
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
                  AND (next_attempt_at IS NULL OR next_attempt_at <= now())
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


def requeue_after_outage(paper_id: str, reason: str) -> bool:
    """Put a job back because the model server was unreachable, not because the paper
    is bad. Returns False once the backoff schedule is exhausted, at which point the
    caller should fail it for real.

    The attempt is not counted against MAX_ATTEMPTS: that budget exists for jobs that
    kill their worker, and an outage is nobody's fault but the model server's.
    """
    with _conn() as connection:
        row = connection.execute(
            "SELECT outage_retries FROM papers WHERE id = %s", (paper_id,)
        ).fetchone()
        if row is None:
            return False
        retries = row["outage_retries"]
        if retries >= MAX_OUTAGE_RETRIES:
            return False
        delay = OUTAGE_BACKOFF_SECONDS[min(retries, len(OUTAGE_BACKOFF_SECONDS) - 1)]
        connection.execute(
            """
            UPDATE papers SET
                status          = 'queued',
                claimed_by      = NULL,
                claimed_at      = NULL,
                heartbeat_at    = NULL,
                progress        = 0,
                attempts        = GREATEST(attempts - 1, 0),
                outage_retries  = outage_retries + 1,
                next_attempt_at = now() + make_interval(secs => %s),
                detail          = %s
            WHERE id = %s
            """,
            (delay, f"{reason}; retrying in {delay // 60 or 1} min", paper_id),
        )
    return True


def clear_outage_backoff(paper_id: str) -> None:
    """A job that got going again should not carry its outage history forward."""
    with _conn() as connection:
        connection.execute(
            "UPDATE papers SET outage_retries = 0, next_attempt_at = NULL WHERE id = %s",
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


# ---------------------------------------------------------------------------- chunks


def save_chunks(paper_id: str, chunks: list[dict]) -> None:
    """Record the narration transcript with timestamps, replacing any earlier run."""
    with _conn() as connection:
        connection.execute("DELETE FROM chunks WHERE paper_id = %s", (paper_id,))
        if not chunks:
            return
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO chunks (paper_id, ordinal, text, start_s, chapter_title, figure_id)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (
                        paper_id,
                        index,
                        chunk["text"],
                        chunk["start_s"],
                        chunk.get("chapter_title", ""),
                        chunk.get("figure_id"),
                    )
                    for index, chunk in enumerate(chunks)
                ],
            )


def chunks_around(paper_id: str, position_s: float, window_s: float = 180.0) -> list[dict]:
    """The narration either side of where the listener is.

    Reaches backwards further than forwards: a question is nearly always about something
    just heard, and the text ahead has not been heard yet.
    """
    with _conn() as connection:
        return connection.execute(
            """
            SELECT text, start_s, chapter_title, figure_id FROM chunks
            WHERE paper_id = %s AND start_s BETWEEN %s AND %s
            ORDER BY start_s
            """,
            (paper_id, position_s - window_s, position_s + window_s / 3),
        ).fetchall()


def chapter_at(paper_id: str, position_s: float) -> dict | None:
    """Which chapter is playing, and where it starts -- for "repeat that section"."""
    paper = get_paper(paper_id)
    if not paper:
        return None
    current = None
    for chapter in paper["chapters"]:
        if chapter.get("start_s", 0) <= position_s:
            current = chapter
        else:
            break
    return current


def adjacent_chapter(paper_id: str, position_s: float, direction: int) -> dict | None:
    """The next or previous chapter, for section navigation."""
    paper = get_paper(paper_id)
    if not paper or not paper["chapters"]:
        return None
    chapters = sorted(paper["chapters"], key=lambda c: c.get("start_s", 0))
    index = 0
    for position, chapter in enumerate(chapters):
        if chapter.get("start_s", 0) <= position_s:
            index = position
    target = index + direction
    if target < 0 or target >= len(chapters):
        return None
    return chapters[target]


# -------------------------------------------------------------------------- settings


def get_settings() -> dict:
    """Stored settings merged over the defaults, so a new key needs no migration."""
    with _conn() as connection:
        row = connection.execute("SELECT values FROM settings WHERE id IS TRUE").fetchone()
    return {**DEFAULT_SETTINGS, **((row or {}).get("values") or {})}


def save_settings(values: dict) -> dict:
    """Persist only keys that exist, so an unknown field cannot smuggle itself in."""
    clean = {k: v for k, v in values.items() if k in DEFAULT_SETTINGS}
    merged = {**get_settings(), **clean}
    with _conn() as connection:
        connection.execute(
            "INSERT INTO settings (id, values) VALUES (TRUE, %s) "
            "ON CONFLICT (id) DO UPDATE SET values = EXCLUDED.values",
            (Jsonb(merged),),
        )
    return merged
