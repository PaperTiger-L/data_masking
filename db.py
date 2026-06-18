import sqlite3
from datetime import datetime

from config import ADMIN_CONFIG, SQLITE_DB_PATH


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, ddl: str):
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {ddl}")


def init_db():
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS upload_sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL,
                phase TEXT NOT NULL,
                error TEXT,
                server_region TEXT NOT NULL,
                storage_mode TEXT,
                output_path TEXT,
                staging_root TEXT NOT NULL,
                request_file_count INTEGER NOT NULL DEFAULT 0,
                processed_files INTEGER NOT NULL DEFAULT 0,
                progress_json TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                results_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                upload_session_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL,
                phase TEXT NOT NULL,
                error TEXT,
                blur_face_plates INTEGER NOT NULL,
                blur_texts INTEGER NOT NULL,
                blur_method TEXT NOT NULL,
                anonymization_enabled INTEGER NOT NULL DEFAULT 1,
                remote_dir_name TEXT,
                output_artifact_name TEXT,
                output_artifact_path TEXT,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY(upload_session_id) REFERENCES upload_sessions(id)
            );

            CREATE TABLE IF NOT EXISTS job_files (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                upload_session_id TEXT NOT NULL,
                original_name TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                local_staged_path TEXT NOT NULL,
                media_kind TEXT NOT NULL,
                ingest_status TEXT NOT NULL,
                size_bytes INTEGER,
                discovered_from_archive TEXT,
                output_relative_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id),
                FOREIGN KEY(upload_session_id) REFERENCES upload_sessions(id)
            );
            """
        )

        _ensure_column(conn, "jobs", "attempt_count", "attempt_count INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "jobs", "last_heartbeat_at", "last_heartbeat_at TEXT")
        _ensure_column(conn, "jobs", "claimed_by", "claimed_by TEXT")
        _ensure_column(conn, "jobs", "lease_expires_at", "lease_expires_at TEXT")
        _ensure_column(conn, "jobs", "anonymization_enabled", "anonymization_enabled INTEGER NOT NULL DEFAULT 1")
        _ensure_column(conn, "jobs", "output_artifact_name", "output_artifact_name TEXT")
        _ensure_column(conn, "jobs", "output_artifact_path", "output_artifact_path TEXT")

        conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO NOTHING
            """,
            (
                "anonymization_enabled",
                "1" if ADMIN_CONFIG["anonymization_enabled"] else "0",
                datetime.utcnow().isoformat(),
            ),
        )

        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_upload_sessions_status ON upload_sessions(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_upload_session_id ON jobs(upload_session_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_lease_expires_at ON jobs(lease_expires_at);
            CREATE INDEX IF NOT EXISTS idx_job_files_job_id ON job_files(job_id);
            CREATE INDEX IF NOT EXISTS idx_job_files_upload_session_id ON job_files(upload_session_id);
            """
        )
