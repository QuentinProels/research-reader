"""Paper metadata store.

v0 uses SQLite: the schema is small, single-user, and has no vector column yet.
Postgres + pgvector arrives with build-order step 7 (Q&A fallback search), which
is the first step that actually needs embeddings. Keep all SQL behind this module
so that swap stays a one-file change.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from app.config import settings

DB_PATH = settings.data_dir / "reader.sqlite3"

SCHEMA = """
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
    duration_s   REAL,
    chapters     TEXT NOT NULL DEFAULT '[]',   -- json: [{title, start_s, figure_id}]
    figures      TEXT NOT NULL DEFAULT '[]',   -- json: [{id, label, image_path, caption, start_s}]
    resume_s     REAL NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);
"""

_JSON_COLUMNS = ("chapters", "figures")


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.executescript(SCHEMA)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    paper = dict(row)
    for column in _JSON_COLUMNS:
        paper[column] = json.loads(paper[column])
    return paper


def create_paper(title: str, source: str) -> str:
    paper_id = uuid.uuid4().hex[:12]
    with _conn() as con:
        con.execute(
            "INSERT INTO papers (id, title, source, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (paper_id, title, source, "queued", datetime.now(timezone.utc).isoformat()),
        )
    return paper_id


def update_paper(paper_id: str, **fields: Any) -> None:
    if not fields:
        return
    for column in _JSON_COLUMNS:
        if column in fields:
            fields[column] = json.dumps(fields[column])
    assignments = ", ".join(f"{key} = ?" for key in fields)
    with _conn() as con:
        con.execute(
            f"UPDATE papers SET {assignments} WHERE id = ?", (*fields.values(), paper_id)
        )


def set_status(paper_id: str, status: str, progress: float, detail: str = "") -> None:
    update_paper(paper_id, status=status, progress=progress, detail=detail)


def get_paper(paper_id: str) -> dict[str, Any] | None:
    with _conn() as con:
        row = con.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_papers() -> list[dict[str, Any]]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM papers ORDER BY created_at DESC").fetchall()
    return [_row_to_dict(row) for row in rows]
