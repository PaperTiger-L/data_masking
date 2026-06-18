from pathlib import Path
from typing import List

from fastapi import HTTPException, UploadFile

from config import ALLOWED_ARCHIVE_EXTENSIONS, ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from repositories.jobs import create_job_file


def _classify_extension(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in ALLOWED_EXTENSIONS:
        return "image"

    lower_name = path.name.lower()
    if any(lower_name.endswith(ext) for ext in ALLOWED_ARCHIVE_EXTENSIONS):
        return "archive"

    return "other"


def ingest_upload_files(files: List[UploadFile], session_id: str, job_id: str, input_dir: Path,
                        sanitize_path, classify_media_kind=None) -> list[str]:
    staged_paths: list[str] = []

    for file in files:
        if not file.filename:
            continue

        safe_relative_path = sanitize_path(file.filename)
        file_path = input_dir / safe_relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        media_kind = _classify_extension(file_path)

        total_written = 0
        with open(file_path, "wb") as buffer:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > MAX_FILE_SIZE:
                    raise HTTPException(status_code=400, detail=f"文件过大: {file.filename}")
                buffer.write(chunk)

        create_job_file(
            job_id=job_id,
            upload_session_id=session_id,
            original_name=file.filename,
            relative_path=str(safe_relative_path),
            local_staged_path=str(file_path),
            media_kind=media_kind,
            ingest_status="staged",
            size_bytes=total_written,
        )
        staged_paths.append(str(file_path))

    return staged_paths
