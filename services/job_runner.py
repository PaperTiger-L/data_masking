import threading
import uuid
from datetime import datetime, timedelta
from typing import Callable, Optional

from loguru import logger

from config import JOB_LEASE_SECONDS, JOB_RECOVERY_GRACE_SECONDS, RUNNER_POLL_INTERVAL_SECONDS
from repositories.jobs import claim_next_queued_job, mark_job_error, release_job_claim, requeue_stale_jobs, update_upload_session
from services.staging import get_input_dir


class JobRunner:
    def __init__(self, process_job: Callable[[str], None]):
        self.process_job = process_job
        self.worker_id = f"runner-{uuid.uuid4().hex[:8]}"
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._recover_stale_jobs()
        self._thread = threading.Thread(target=self._run_loop, name="job-runner", daemon=True)
        self._thread.start()
        logger.info(f"Job runner started: {self.worker_id}")

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Job runner stopped")

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self._recover_stale_jobs()
                job = claim_next_queued_job(self.worker_id, JOB_LEASE_SECONDS)
                if not job:
                    self._stop_event.wait(RUNNER_POLL_INTERVAL_SECONDS)
                    continue

                session_id = job["upload_session_id"]
                if not get_input_dir(session_id).exists():
                    logger.warning(f"Job {session_id} missing staging input directory, marking as error")
                    mark_job_error(session_id, "Staging directory missing; task cannot be resumed")
                    release_job_claim(job["id"])
                    continue

                update_upload_session(session_id, status="running", phase="queued", error=None)
                try:
                    self.process_job(session_id)
                finally:
                    release_job_claim(job["id"])
            except Exception as exc:
                logger.exception(f"Job runner loop error: {exc}")
                self._stop_event.wait(RUNNER_POLL_INTERVAL_SECONDS)

    def _recover_stale_jobs(self):
        stale_before = (datetime.utcnow() - timedelta(seconds=JOB_RECOVERY_GRACE_SECONDS)).isoformat()
        requeue_stale_jobs(stale_before)
