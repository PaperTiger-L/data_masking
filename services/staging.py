from pathlib import Path
import shutil

from config import STAGING_ROOT


def get_session_staging_root(session_id: str) -> Path:
    return STAGING_ROOT / session_id


def get_input_dir(session_id: str) -> Path:
    return get_session_staging_root(session_id) / "input"


def get_extracted_dir(session_id: str) -> Path:
    return get_session_staging_root(session_id) / "extracted"


def get_work_dir(session_id: str) -> Path:
    return get_session_staging_root(session_id) / "work"


def get_output_dir(session_id: str) -> Path:
    return get_session_staging_root(session_id) / "output"


def ensure_session_dirs(session_id: str) -> Path:
    root = get_session_staging_root(session_id)
    for path in (root, get_input_dir(session_id), get_extracted_dir(session_id), get_work_dir(session_id), get_output_dir(session_id)):
        path.mkdir(parents=True, exist_ok=True)
    return root


def remove_session_staging(session_id: str):
    shutil.rmtree(get_session_staging_root(session_id), ignore_errors=True)
