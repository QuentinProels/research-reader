"""Paper metadata store, on Postgres + pgvector.

The vector extension is created up front even though nothing writes embeddings
yet: step 7 (Q&A fallback search over the full paper text) is the consumer, and
having it present means that step is a migration rather than a re-provision.

All SQL in the project lives in this module.
"""

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.config import settings

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
    chapters     JSONB NOT NULL DEFAULT '[]'::jsonb,   -- [{title, start_s}]
    figures      JSONB NOT NULL DEFAULT '[]'::jsonb,   -- [{id, label, image_path, caption, start_s}]
    resume_s     DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS papers_created_at_idx ON papers (created_at DESC);

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
            settings.database_url, min_size=1, max_size=8, kwargs={"row_factory": dict_row}
        )
    return _pool


@contextmanager
def _conn():
    with _get_pool().connection() as connection:
        yield connection


def init_db() -> None:
    with _conn() as connection:
        connection.execute(SCHEMA)


def create_paper(title: str, source: str) -> str:
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
    update_paper(paper_id, status=status, progress=progress, detail=detail)


def get_paper(paper_id: str) -> dict[str, Any] | None:
    with _conn() as connection:
        return connection.execute("SELECT * FROM papers WHERE id = %s", (paper_id,)).fetchone()


def list_papers() -> list[dict[str, Any]]:
    with _conn() as connection:
        return connection.execute("SELECT * FROM papers ORDER BY created_at DESC").fetchall()
