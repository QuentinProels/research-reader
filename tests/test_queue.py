"""Queue tests. These hit a real Postgres -- the behaviour under test is SQL
(SKIP LOCKED claiming, lease expiry), so mocking it would test nothing.

Skipped when the database is not up.
"""

import pytest

from app import store


@pytest.fixture(autouse=True)
def _clean_db():
    # These tests truncate tables. Refuse to touch anything that is not obviously a
    # throwaway database -- an earlier version of this file ran against DATABASE_URL
    # from .env and deleted queued papers out from under a running worker.
    from app.config import settings

    if not settings.database_url.rstrip("/").endswith("_test"):
        pytest.fail(
            f"refusing to run destructive tests against {settings.database_url!r}; "
            "tests/conftest.py should have redirected this to a _test database"
        )
    try:
        store.init_db()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"postgres unavailable: {exc}")
    with store._conn() as c:
        c.execute("DELETE FROM papers")
        c.execute("DELETE FROM workers")
    yield
    with store._conn() as c:
        c.execute("DELETE FROM papers")
        c.execute("DELETE FROM workers")


def _age_heartbeat(paper_id: str, seconds: int) -> None:
    with store._conn() as c:
        c.execute(
            "UPDATE papers SET heartbeat_at = now() - make_interval(secs => %s) WHERE id = %s",
            (seconds, paper_id),
        )


class TestClaiming:
    def test_claims_the_oldest_queued_paper_first(self):
        first = store.create_paper("First", "upload")
        store.create_paper("Second", "upload")
        assert store.claim_next_job("w1")["id"] == first

    def test_a_paper_is_never_handed_to_two_workers(self):
        store.create_paper("Only one", "upload")
        assert store.claim_next_job("w1") is not None
        assert store.claim_next_job("w2") is None

    def test_returns_none_when_the_queue_is_empty(self):
        assert store.claim_next_job("w1") is None

    def test_claiming_marks_the_worker_and_counts_the_attempt(self):
        store.create_paper("A paper", "upload")
        job = store.claim_next_job("worker-7")
        assert job["claimed_by"] == "worker-7"
        assert job["attempts"] == 1
        assert job["status"] == "parsing"


class TestQueuePosition:
    def test_positions_are_one_based_and_in_arrival_order(self):
        ids = [store.create_paper(f"Paper {i}", "upload") for i in range(3)]
        positions = {p["id"]: p["queue_position"] for p in store.list_papers()}
        assert [positions[i] for i in ids] == [1, 2, 3]

    def test_a_claimed_paper_has_no_position(self):
        store.create_paper("Being worked on", "upload")
        claimed = store.claim_next_job("w1")["id"]
        positions = {p["id"]: p["queue_position"] for p in store.list_papers()}
        assert positions[claimed] is None

    def test_positions_close_up_after_one_is_claimed(self):
        store.create_paper("First", "upload")
        second = store.create_paper("Second", "upload")
        store.claim_next_job("w1")
        positions = {p["id"]: p["queue_position"] for p in store.list_papers()}
        assert positions[second] == 1


class TestCrashRecovery:
    def test_a_job_whose_worker_died_is_requeued(self):
        """The exact failure that motivated the queue: the process holding a job
        disappears and the row sits at its last progress value forever."""
        paper_id = store.create_paper("Interrupted", "upload")
        store.claim_next_job("doomed-worker")
        store.set_status(paper_id, "reflowing", 0.45, "Cleaning section 4 of 27")
        _age_heartbeat(paper_id, store.LEASE_SECONDS + 30)

        assert store.reap_stale_jobs() == [paper_id]
        revived = store.get_paper(paper_id)
        assert revived["status"] == "queued"
        assert revived["claimed_by"] is None
        assert revived["progress"] == 0

    def test_a_live_job_is_left_alone(self):
        paper_id = store.create_paper("Working fine", "upload")
        store.claim_next_job("w1")
        store.set_status(paper_id, "reflowing", 0.5, "busy")
        assert store.reap_stale_jobs() == []
        assert store.get_paper(paper_id)["status"] == "reflowing"

    def test_it_gives_up_after_max_attempts(self):
        paper_id = store.create_paper("Poison pill", "upload")
        for _ in range(store.MAX_ATTEMPTS):
            store.claim_next_job("w1")
            _age_heartbeat(paper_id, store.LEASE_SECONDS + 30)
            store.reap_stale_jobs()
        paper = store.get_paper(paper_id)
        assert paper["status"] == "failed"
        assert "Gave up" in paper["error"]

    def test_graceful_shutdown_requeues_immediately(self):
        paper_id = store.create_paper("Shutting down", "upload")
        store.claim_next_job("w1")
        store.release_job(paper_id)
        assert store.get_paper(paper_id)["status"] == "queued"

    def test_release_does_not_resurrect_a_finished_paper(self):
        paper_id = store.create_paper("Done", "upload")
        store.claim_next_job("w1")
        store.update_paper(paper_id, status="ready")
        store.release_job(paper_id)
        assert store.get_paper(paper_id)["status"] == "ready"


class TestCancel:
    def test_a_queued_paper_can_be_cancelled(self):
        paper_id = store.create_paper("Not wanted", "upload")
        assert store.cancel_paper(paper_id) is True
        assert store.get_paper(paper_id)["error"] == "Cancelled"

    def test_a_running_paper_cannot_be_cancelled(self):
        paper_id = store.create_paper("Already going", "upload")
        store.claim_next_job("w1")
        assert store.cancel_paper(paper_id) is False

    def test_a_cancelled_paper_is_not_claimed(self):
        store.create_paper("Not wanted", "upload")
        store.cancel_paper(store.list_papers()[0]["id"])
        assert store.claim_next_job("w1") is None


class TestWorkerLiveness:
    def test_no_workers_reported_when_none_have_checked_in(self):
        assert store.live_workers() == 0

    def test_a_checked_in_worker_is_counted(self):
        store.worker_seen("w1")
        assert store.live_workers() == 1

    def test_a_stale_worker_is_not_counted(self):
        store.worker_seen("w1")
        with store._conn() as c:
            c.execute("UPDATE workers SET last_seen = now() - interval '10 minutes'")
        assert store.live_workers() == 0

    def test_a_departed_worker_is_removed(self):
        store.worker_seen("w1")
        store.worker_gone("w1")
        assert store.live_workers() == 0
