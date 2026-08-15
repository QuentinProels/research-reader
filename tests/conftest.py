"""Point the test suite at a throwaway database before anything imports app.config.

The queue tests need a real Postgres -- the behaviour under test is SQL. But they also
truncate tables between tests, and the first version of them ran against DATABASE_URL
straight from .env, which is the live database. Running pytest deleted queued papers out
from under a running worker.

pytest imports conftest before test modules, and pydantic-settings gives os.environ
priority over the .env file, so setting the variable here redirects every connection the
suite makes. The test database is created on demand and never touched by the app.
"""

import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse

TEST_DB_SUFFIX = "_test"


def _live_url() -> str:
    if "DATABASE_URL" in os.environ:
        return os.environ["DATABASE_URL"]
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip()
    return "postgresql://reader:reader@127.0.0.1:5433/reader"


def _redirect_to_test_database() -> None:
    parsed = urlparse(_live_url())
    live_name = parsed.path.lstrip("/")
    if live_name.endswith(TEST_DB_SUFFIX):
        return  # already pointed somewhere safe
    test_name = f"{live_name}{TEST_DB_SUFFIX}"

    try:
        import psycopg
    except ImportError:  # the suite will skip on its own
        return

    admin = urlunparse(parsed._replace(path=f"/{live_name}"))
    try:
        with psycopg.connect(admin, autocommit=True, connect_timeout=5) as connection:
            exists = connection.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (test_name,)
            ).fetchone()
            if not exists:
                connection.execute(f'CREATE DATABASE "{test_name}"')
    except psycopg.Error:
        return  # postgres is down; the db-backed tests skip themselves

    os.environ["DATABASE_URL"] = urlunparse(parsed._replace(path=f"/{test_name}"))


_redirect_to_test_database()
