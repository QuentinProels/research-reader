"""Queue worker. Run alongside the API:

    uv run python -m app.worker

Deliberately a separate process from the API. When the pipeline ran in an API thread,
restarting the API killed whatever was mid-flight and left the row frozen at its last
progress value with nobody working on it. Now the API only enqueues, and it can be
restarted, redeployed, or crash without touching a running job.

Run more than one for parallelism -- claiming is atomic, so they will not collide. On
this box one is usually right: llama-server serves a single model and Kokoro is on CPU,
so two workers mostly contend rather than go faster.
"""

import logging
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

from app import pipeline, store
from app.config import settings

log = logging.getLogger("worker")

POLL_SECONDS = 3
REAP_EVERY_SECONDS = 60
HEARTBEAT_SECONDS = 15

_stopping = False
_current_paper: str | None = None


def _handle_signal(signum, _frame) -> None:
    """First signal: stop after the current job. Second: give the job back and exit."""
    global _stopping
    if _stopping and _current_paper:
        log.warning("second signal -- releasing %s back to the queue", _current_paper)
        store.release_job(_current_paper)
        sys.exit(130)
    _stopping = True
    log.info("signal %s received, finishing current job then stopping", signum)


def _pdf_path_for(job: dict) -> Path:
    if job.get("pdf_path"):
        return settings.data_dir / job["pdf_path"]
    return settings.papers_dir / job["id"] / "source.pdf"


def _heartbeat_loop(worker_id: str, stop: threading.Event) -> None:
    """Announce liveness on its own thread.

    The poll loop cannot do this: it disappears into pipeline.run for the length of a
    whole paper, so a worker that was busy rendering looked dead to the UI, which then
    warned that the queue would not drain -- exactly backwards.
    """
    while not stop.wait(HEARTBEAT_SECONDS):
        try:
            store.worker_seen(worker_id)
        except Exception:  # noqa: BLE001 -- a blip here must not take the worker down
            log.warning("heartbeat failed", exc_info=True)


def run_forever() -> None:
    global _current_paper
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    store.init_db()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    log.info("worker %s up, polling every %ss", worker_id, POLL_SECONDS)

    store.worker_seen(worker_id)
    stop_heartbeat = threading.Event()
    threading.Thread(
        target=_heartbeat_loop, args=(worker_id, stop_heartbeat), daemon=True, name="heartbeat"
    ).start()

    last_reap = 0.0
    try:
        while not _stopping:
            store.worker_seen(worker_id)

            if time.monotonic() - last_reap > REAP_EVERY_SECONDS:
                for paper_id in store.reap_stale_jobs():
                    log.warning("requeued %s: previous run stopped heartbeating", paper_id)
                last_reap = time.monotonic()

            job = store.claim_next_job(worker_id)
            if job is None:
                time.sleep(POLL_SECONDS)
                continue

            _current_paper = job["id"]
            pdf_path = _pdf_path_for(job)
            log.info("claimed %s (attempt %s): %s", job["id"], job["attempts"], job["title"][:60])
            if not pdf_path.exists():
                store.update_paper(
                    job["id"], status="failed", error=f"Source PDF is missing at {pdf_path}"
                )
            else:
                pipeline.run(job["id"], pdf_path)  # records its own failure in the row
                log.info("finished %s", job["id"])
            _current_paper = None
    finally:
        _current_paper = None
        stop_heartbeat.set()
        store.worker_gone(worker_id)
        log.info("worker %s stopped", worker_id)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    run_forever()
