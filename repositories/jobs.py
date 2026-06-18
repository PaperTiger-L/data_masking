import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from db import get_connection


TERMINAL_JOB_STATUSES = {"completed", "error"}
ACTIVE_JOB_STATUSES = {"queued", "running", "delivering"}


def _now() -> str:
    return datetime.utcnow().isoformat()


def _future_timestamp(seconds: int) -> str:
    return (datetime.utcnow() + timedelta(seconds=seconds)).isoformat()


def create_upload_session(session_id: str, server_region: str, staging_root: str, request_file_count: int,
                          status: str, phase: str, progress: dict[str, Any], summary: dict[str, Any],
                          results: dict[str, Any]) -> None:
    now = _now()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO upload_sessions (
                id, created_at, updated_at, status, phase, error, server_region,
                storage_mode, output_path, staging_root, request_file_count,
                processed_files, progress_json, summary_json, results_json
            ) VALUES (?, ?, ?, ?, ?, NULL, ?, NULL, NULL, ?, ?, 0, ?, ?, ?)
            """,
            (
                session_id,
                now,
                now,
                status,
                phase,
                server_region,
                staging_root,
                request_file_count,
                json.dumps(progress, ensure_ascii=False),
                json.dumps(summary, ensure_ascii=False),
                json.dumps(results, ensure_ascii=False),
            ),
        )


def create_job(upload_session_id: str, blur_face_plates: bool, blur_texts: bool,
               blur_method: str, remote_dir_name: str, anonymization_enabled: bool) -> str:
    job_id = uuid.uuid4().hex
    now = _now()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id, upload_session_id, created_at, updated_at, status, phase, error,
                blur_face_plates, blur_texts, blur_method, anonymization_enabled, remote_dir_name,
                started_at, completed_at, attempt_count, last_heartbeat_at, claimed_by, lease_expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, NULL, NULL, 0, NULL, NULL, NULL)
            """,
            (
                job_id,
                upload_session_id,
                now,
                now,
                "queued",
                "queued",
                int(blur_face_plates),
                int(blur_texts),
                blur_method,
                int(anonymization_enabled),
                remote_dir_name,
            ),
        )
    return job_id


def create_job_file(job_id: str, upload_session_id: str, original_name: str, relative_path: str,
                    local_staged_path: str, media_kind: str, ingest_status: str,
                    size_bytes: Optional[int] = None, error: Optional[str] = None) -> str:
    file_id = uuid.uuid4().hex
    now = _now()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO job_files (
                id, job_id, upload_session_id, original_name, relative_path, local_staged_path,
                media_kind, ingest_status, size_bytes, discovered_from_archive, output_relative_path,
                error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
            """,
            (
                file_id,
                job_id,
                upload_session_id,
                original_name,
                relative_path,
                local_staged_path,
                media_kind,
                ingest_status,
                size_bytes,
                error,
                now,
                now,
            ),
        )
    return file_id


def list_job_files(upload_session_id: str):
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM job_files WHERE upload_session_id = ? ORDER BY created_at ASC",
            (upload_session_id,),
        ).fetchall()


def get_upload_session(session_id: str):
    with get_connection() as conn:
        return conn.execute("SELECT * FROM upload_sessions WHERE id = ?", (session_id,)).fetchone()


def get_job_by_session_id(session_id: str):
    with get_connection() as conn:
        return conn.execute("SELECT * FROM jobs WHERE upload_session_id = ?", (session_id,)).fetchone()


def get_job_by_id(job_id: str):
    with get_connection() as conn:
        return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def list_jobs_by_status(statuses: list[str]):
    if not statuses:
        return []
    placeholders = ", ".join("?" for _ in statuses)
    with get_connection() as conn:
        return conn.execute(
            f"SELECT * FROM jobs WHERE status IN ({placeholders}) ORDER BY created_at ASC",
            statuses,
        ).fetchall()


def claim_next_queued_job(worker_id: str, lease_seconds: int):
    now = _now()
    lease_expires_at = _future_timestamp(lease_seconds)
    with get_connection() as conn:
        candidate = conn.execute(
            """
            SELECT id, upload_session_id, attempt_count
            FROM jobs
            WHERE status = 'queued'
              AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (now,),
        ).fetchone()
        if not candidate:
            return None

        updated = conn.execute(
            """
            UPDATE jobs
            SET status = 'running',
                phase = 'running',
                started_at = COALESCE(started_at, ?),
                updated_at = ?,
                last_heartbeat_at = ?,
                claimed_by = ?,
                lease_expires_at = ?,
                attempt_count = attempt_count + 1,
                error = NULL
            WHERE id = ?
              AND status = 'queued'
              AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
            """,
            (now, now, now, worker_id, lease_expires_at, candidate["id"], now),
        )
        if updated.rowcount != 1:
            return None

    return get_job_by_id(candidate["id"])


def heartbeat_job(job_id: str, worker_id: str, lease_seconds: int):
    now = _now()
    lease_expires_at = _future_timestamp(lease_seconds)
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET updated_at = ?,
                last_heartbeat_at = ?,
                claimed_by = ?,
                lease_expires_at = ?
            WHERE id = ?
            """,
            (now, now, worker_id, lease_expires_at, job_id),
        )


def release_job_claim(job_id: str):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET claimed_by = NULL,
                lease_expires_at = NULL,
                last_heartbeat_at = NULL,
                updated_at = ?
            WHERE id = ?
            """,
            (_now(), job_id),
        )


def requeue_stale_jobs(stale_before: str):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = 'queued',
                phase = 'queued',
                claimed_by = NULL,
                lease_expires_at = NULL,
                updated_at = ?,
                error = CASE
                    WHEN error IS NULL OR error = '' THEN 'Processing interrupted by server restart'
                    ELSE error
                END
            WHERE status IN ('running', 'delivering')
              AND (
                    lease_expires_at IS NOT NULL AND lease_expires_at <= ?
                 OR last_heartbeat_at IS NOT NULL AND last_heartbeat_at <= ?
              )
            """,
            (_now(), stale_before, stale_before),
        )


def mark_job_error(upload_session_id: str, message: str):
    update_job(upload_session_id, status="error", phase="error", error=message)
    update_upload_session(upload_session_id, status="error", phase="error", error=message)


def get_recoverable_sessions(retention_cutoff: str):
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT * FROM upload_sessions
            WHERE status IN ('completed', 'error')
              AND updated_at <= ?
            ORDER BY updated_at ASC
            """,
            (retention_cutoff,),
        ).fetchall()


def get_app_setting(key: str) -> Optional[str]:
    with get_connection() as conn:
        row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_app_setting(key: str, value: str) -> None:
    now = _now()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, now),
        )


def update_upload_session(session_id: str, **fields):
    if not fields:
        return
    fields["updated_at"] = _now()
    columns = ", ".join(f"{key} = ?" for key in fields.keys())
    values = list(fields.values()) + [session_id]
    with get_connection() as conn:
        conn.execute(f"UPDATE upload_sessions SET {columns} WHERE id = ?", values)


def update_job(upload_session_id: str, **fields):
    if not fields:
        return
    terminal_status = fields.get("status") in TERMINAL_JOB_STATUSES
    if terminal_status and "completed_at" not in fields:
        fields["completed_at"] = _now()
    fields["updated_at"] = _now()
    columns = ", ".join(f"{key} = ?" for key in fields.keys())
    values = list(fields.values()) + [upload_session_id]
    with get_connection() as conn:
        conn.execute(f"UPDATE jobs SET {columns} WHERE upload_session_id = ?", values)


def update_job_file_status(file_id: str, ingest_status: str, error: Optional[str] = None,
                           output_relative_path: Optional[str] = None):
    fields: dict[str, Any] = {"ingest_status": ingest_status, "updated_at": _now()}
    if error is not None:
        fields["error"] = error
    if output_relative_path is not None:
        fields["output_relative_path"] = output_relative_path
    columns = ", ".join(f"{key} = ?" for key in fields.keys())
    values = list(fields.values()) + [file_id]
    with get_connection() as conn:
        conn.execute(f"UPDATE job_files SET {columns} WHERE id = ?", values)
