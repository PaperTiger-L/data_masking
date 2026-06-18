import json
from typing import Any

from repositories.jobs import get_job_by_session_id, get_upload_session, update_job, update_upload_session


def default_progress_payload(total_uploaded_files: int = 0) -> dict[str, Any]:
    return {
        "total_uploaded_files": total_uploaded_files,
        "archives_detected": 0,
        "archives_processed": 0,
        "images_discovered": 0,
        "processable_images": 0,
        "processed_images": 0,
        "skipped_files": 0,
        "error_files": 0,
        "uploaded_outputs": 0,
        "current_item": None,
        "percent": 0,
    }


def default_summary_payload(total_uploaded_files: int = 0) -> dict[str, Any]:
    return {
        "uploaded_files": total_uploaded_files,
        "archives": 0,
        "direct_images": 0,
        "extracted_images": 0,
        "non_images_preserved": 0,
        "non_images_skipped": 0,
        "empty_archives": 0,
        "corrupt_archives": 0,
        "duplicate_output_paths_resolved": 0,
        "oversized_files_skipped": 0,
    }


def default_results_payload() -> dict[str, Any]:
    return {"warnings": [], "sample_skipped": [], "sample_errors": []}


def update_session_progress(session_id: str, **updates):
    session = get_upload_session(session_id)
    if not session:
        return

    progress = json.loads(session["progress_json"])
    summary = json.loads(session["summary_json"])
    results = json.loads(session["results_json"])

    fields = {}
    for key, value in updates.items():
        if key == "progress" and isinstance(value, dict):
            progress.update(value)
        elif key == "summary" and isinstance(value, dict):
            summary.update(value)
        elif key == "results" and isinstance(value, dict):
            for result_key, result_value in value.items():
                if isinstance(result_value, list):
                    results.setdefault(result_key, []).extend(result_value)
                else:
                    results[result_key] = result_value
        else:
            fields[key] = value

    phase = fields.get("phase", session["phase"])
    if phase == "extracting":
        total_archives = max(progress.get("archives_detected", 0), 1)
        percent = int((progress.get("archives_processed", 0) / total_archives) * 20)
    elif phase == "processing":
        total_images = max(progress.get("processable_images", 0), 1)
        percent = 20 + int((progress.get("processed_images", 0) / total_images) * 70)
    elif phase == "uploading":
        total_outputs = max(progress.get("processable_images", 0), 1)
        percent = 90 + int((progress.get("uploaded_outputs", 0) / total_outputs) * 10)
    elif phase == "completed":
        percent = 100
    else:
        percent = progress.get("percent", 0)
    progress["percent"] = min(percent, 100)

    update_fields = {
        **fields,
        "progress_json": json.dumps(progress, ensure_ascii=False),
        "summary_json": json.dumps(summary, ensure_ascii=False),
        "results_json": json.dumps(results, ensure_ascii=False),
    }
    update_upload_session(session_id, **update_fields)

    mirrored_fields = {key: value for key, value in fields.items() if key in {"status", "phase", "error"}}
    if mirrored_fields:
        update_job(session_id, **mirrored_fields)


def build_status_response(session_id: str) -> dict[str, Any] | None:
    session = get_upload_session(session_id)
    if not session:
        return None
    job = get_job_by_session_id(session_id)
    return {
        "status": session["status"],
        "phase": session["phase"],
        "error": session["error"],
        "processed_files": session["processed_files"],
        "storage_mode": session["storage_mode"],
        "output_path": session["output_path"],
        "anonymization_enabled": bool(job["anonymization_enabled"]) if job else None,
        "output_artifact_name": job["output_artifact_name"] if job else None,
        "output_artifact_path": job["output_artifact_path"] if job else None,
        "progress": json.loads(session["progress_json"]),
        "summary": json.loads(session["summary_json"]),
        "results": json.loads(session["results_json"]),
    }
