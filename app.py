#!/usr/bin/env python3
"""
数据脱敏系统主应用
支持人脸、车牌、街牌检测与模糊处理
"""

import os
import sys
import shutil
import tarfile
import tempfile
import zipfile
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from loguru import logger
import paramiko

# 导入我们的脱敏管道
from src.pipeline.unified_blurrer import UnifiedBlurrer
from src.pipeline.texts import TextBlurrer
from config import (
    ADMIN_CONFIG,
    ANONYMIZATION,
    SERVER_REGIONS,
    LOG_CONFIG,
    MAX_FILE_SIZE,
    ALLOWED_EXTENSIONS,
    ALLOWED_ARCHIVE_EXTENSIONS,
    MAX_FILES_PER_REQUEST,
    PROCESSING_TIMEOUT,
    MAX_ARCHIVE_TOTAL_UNCOMPRESSED_SIZE,
    MAX_EXTRACTED_FILES_PER_ARCHIVE,
    MAX_PROCESSABLE_IMAGES_PER_REQUEST,
    UPLOAD_DIR,
    OUTPUT_DIR,
    TEMP_DIR,
)
from db import init_db
from repositories.jobs import create_job, create_upload_session, get_app_setting, get_job_by_session_id, get_upload_session, list_job_files, set_app_setting, update_job, update_upload_session
from services.job_runner import JobRunner
from services.job_status import build_status_response, default_progress_payload, default_results_payload, default_summary_payload, update_session_progress
from services.staging import ensure_session_dirs, get_extracted_dir, get_input_dir, get_output_dir, get_session_staging_root, get_work_dir, remove_session_staging
from services.upload_ingest import ingest_upload_files
from translations import TRANSLATIONS

app = FastAPI(title="数据脱敏系统", description="支持人脸、车牌、街牌检测与脱敏")

# 静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def translate(key: str, lang: str = "en") -> str:
    """翻译函数"""
    entry = TRANSLATIONS.get(key, {})
    return entry.get(lang, entry.get("en", key))


job_runner = JobRunner(process_job=lambda session_id: process_and_upload_in_background(session_id))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    job_runner.start()
    try:
        yield
    finally:
        job_runner.stop()


app.router.lifespan_context = lifespan

def get_lang_from_request(request: Request) -> str:
    lang = request.query_params.get("lang")
    if lang in ("zh", "en"):
        return lang
    cookie_lang = request.cookies.get("lang")
    if cookie_lang in ("zh", "en"):
        return cookie_lang
    return "en"

# 默认远程 SFTP 目录
SFTP_REMOTE_DIR = "/mnt/data"

# 区域映射：将前端选项映射到文件夹标识
REGION_MAP = {
    "america": "US",
    "europe": "EU",
    "asia": "AS",
}

def _normalize_sftp_value(value: Optional[str]) -> str:
    return value.strip() if isinstance(value, str) else ""


def _is_sftp_configured(host: Optional[str], user: Optional[str], a_pass: Optional[str]) -> bool:
    return all(_normalize_sftp_value(value) for value in (host, user, a_pass))


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTENSIONS


def _is_archive_path(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in ALLOWED_ARCHIVE_EXTENSIONS)


def _sanitize_relative_upload_path(filename: str, lang: str = "en") -> Path:
    normalized = filename.replace("\\", "/").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail=translate("error_invalid_filename", lang))

    pure_path = PurePosixPath(normalized)
    if pure_path.is_absolute() or any(part in ("", ".", "..") for part in pure_path.parts):
        raise HTTPException(status_code=400, detail=translate("error_invalid_filepath", lang).replace("{name}", filename))

    return Path(*pure_path.parts)


def _sanitize_archive_member_path(name: str) -> Optional[Path]:
    normalized = name.replace("\\", "/").strip()
    if not normalized:
        return None

    pure_path = PurePosixPath(normalized)
    if pure_path.is_absolute() or any(part in ("", ".", "..") for part in pure_path.parts):
        return None

    return Path(*pure_path.parts)


def _ensure_within_root(destination: Path, root: Path) -> bool:
    try:
        destination.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_extract_zip(archive_path: Path, extract_root: Path):
    extracted_files = []
    extracted_count = 0
    total_size = 0

    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            rel_path = _sanitize_archive_member_path(info.filename)
            if rel_path is None:
                continue

            if info.file_size > MAX_FILE_SIZE:
                raise ValueError(f"压缩包成员过大: {info.filename}")

            extracted_count += 1
            total_size += info.file_size
            if extracted_count > MAX_EXTRACTED_FILES_PER_ARCHIVE:
                raise ValueError("压缩包文件数量超限")
            if total_size > MAX_ARCHIVE_TOTAL_UNCOMPRESSED_SIZE:
                raise ValueError("压缩包解压总大小超限")

            destination = extract_root / rel_path
            if not _ensure_within_root(destination, extract_root):
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(destination, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_files.append(destination)

    return extracted_files


def _safe_extract_tar(archive_path: Path, extract_root: Path):
    extracted_files = []
    extracted_count = 0
    total_size = 0

    with tarfile.open(archive_path) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue

            rel_path = _sanitize_archive_member_path(member.name)
            if rel_path is None:
                continue

            if member.size > MAX_FILE_SIZE:
                raise ValueError(f"压缩包成员过大: {member.name}")

            extracted_count += 1
            total_size += member.size
            if extracted_count > MAX_EXTRACTED_FILES_PER_ARCHIVE:
                raise ValueError("压缩包文件数量超限")
            if total_size > MAX_ARCHIVE_TOTAL_UNCOMPRESSED_SIZE:
                raise ValueError("压缩包解压总大小超限")

            destination = extract_root / rel_path
            if not _ensure_within_root(destination, extract_root):
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with src, open(destination, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_files.append(destination)

    return extracted_files


def _discover_images_recursively(root: Path) -> List[Path]:
    return [path for path in root.rglob("*") if path.is_file() and _is_image_path(path)]


def _resolve_output_path(output_dir: Path, logical_rel_path: Path, used_paths: set[Path]) -> Path:
    candidate = logical_rel_path
    counter = 2
    while candidate in used_paths:
        candidate = logical_rel_path.with_name(f"{logical_rel_path.stem}__{counter}{logical_rel_path.suffix}")
        counter += 1
    used_paths.add(candidate)
    return output_dir / candidate

def _classify_media_kind(path: Path) -> str:
    if _is_image_path(path):
        return "image"
    if _is_archive_path(path):
        return "archive"
    return "other"


# 管理员会话管理
_SESSIONS: dict[str, dict] = {}


def _new_admin_session() -> str:
    token = uuid.uuid4().hex
    _SESSIONS[token] = {"role": "admin"}
    return token


def _get_session_from_cookie(request: Request) -> Optional[dict]:
    token = request.cookies.get("remote_session")
    if token and token in _SESSIONS:
        return _SESSIONS[token]
    return None


def _require_admin_session(request: Request) -> dict:
    session = _get_session_from_cookie(request)
    if not session or session.get("role") != "admin":
        lang = get_lang_from_request(request)
        raise HTTPException(status_code=401, detail=translate("error_unauthorized", lang))
    return session


def _get_admin_anonymization_enabled() -> bool:
    value = get_app_setting("anonymization_enabled")
    if value is None:
        return bool(ADMIN_CONFIG["anonymization_enabled"])
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_artifact_name(name: str, lang: str = "en") -> str:
    candidate = Path(name).name.strip()
    if not candidate or candidate in {".", ".."} or any(sep in candidate for sep in ("/", "\\")):
        raise HTTPException(status_code=400, detail=translate("error_invalid_zip_name", lang))
    if not candidate.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail=translate("error_zip_only", lang))
    return candidate


def _create_zip_from_tree(source_dir: Path, zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(source_dir))


def _deliver_output_zip_local(zip_path: Path, artifact_name: str) -> Path:
    destination = Path(SFTP_REMOTE_DIR) / artifact_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, destination)
    return destination


def _open_region_sftp(region_name: str, lang: str = "en"):
    region_config = SERVER_REGIONS.get(region_name, SERVER_REGIONS["europe"])
    sftp_config = region_config.get("sftp", {})
    host = _normalize_sftp_value(sftp_config.get("host"))
    user = _normalize_sftp_value(sftp_config.get("user"))
    password = _normalize_sftp_value(sftp_config.get("password"))
    if not _is_sftp_configured(host, user, password):
        raise HTTPException(status_code=400, detail=translate("error_region_not_configured", lang))
    transport = paramiko.Transport((host, 22))
    transport.connect(username=user, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return transport, sftp, host, user, password


def _list_local_zip_outputs():
    base_dir = Path(SFTP_REMOTE_DIR)
    if not base_dir.exists():
        return []
    files = [
        {"name": path.name, "size": path.stat().st_size, "modified_at": path.stat().st_mtime}
        for path in base_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".zip"
    ]
    files.sort(key=lambda item: item["name"], reverse=True)
    return files


def _list_remote_zip_outputs(region_name: str, lang: str = "en"):
    transport = None
    sftp = None
    try:
        transport, sftp, *_ = _open_region_sftp(region_name, lang)
        files = []
        for attr in sftp.listdir_attr(SFTP_REMOTE_DIR):
            if not paramiko.sftp_client.stat.S_ISDIR(attr.st_mode) and attr.filename.lower().endswith(".zip"):
                files.append({"name": attr.filename, "size": attr.st_size, "modified_at": attr.st_mtime})
        files.sort(key=lambda item: item["name"], reverse=True)
        return files
    finally:
        if sftp:
            sftp.close()
        if transport:
            transport.close()

# 确保目录存在
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# 初始化脱敏器（延迟加载以避免启动时间过长）
unified_blurrer = None
text_blurrer = None

def get_unified_blurrer():
    global unified_blurrer
    if unified_blurrer is None:
        try:
            unified_blurrer = UnifiedBlurrer(config_path='config/unified_blur_config.yaml', warmup=True)
            logger.info("统一检测器初始化完成")
        except Exception as e:
            logger.error(f"统一检测器初始化失败: {e}")
            raise HTTPException(status_code=500, detail=translate("error_unified_init", "en"))
    return unified_blurrer

def get_text_blurrer():
    global text_blurrer
    if text_blurrer is None:
        try:
            dilate_px = int(ANONYMIZATION.get("text_dilate_px", 8))
            pad_ratio = float(ANONYMIZATION.get("text_pad_ratio", 0.0))
            use_padded_rect = bool(ANONYMIZATION.get("text_use_padded_rect", False))
            text_blurrer = TextBlurrer(
                warmup=True,
                dilate_px=dilate_px,
                pad_ratio=pad_ratio,
                use_padded_rect=use_padded_rect,
            )
            logger.info("文本检测器初始化完成")
        except Exception as e:
            logger.error(f"文本检测器初始化失败: {e}")
            raise HTTPException(status_code=500, detail=translate("error_text_init", "en"))
    return text_blurrer

def upload_file_sftp(host: str, user: str, a_pass: str, local_path: str, remote_dir: str, remote_filename: str):
    """使用 Paramiko 通过 SFTP 上传单个文件，并确保远程目录存在"""
    remote_path = os.path.join(remote_dir, remote_filename)
    transport = None
    sftp = None
    try:
        transport = paramiko.Transport((host, 22))
        transport.connect(username=user, password=a_pass)
        sftp = paramiko.SFTPClient.from_transport(transport)

        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            logger.info(f"远程目录 {remote_dir} 不存在，正在创建...")
            sftp.mkdir(remote_dir)

        logger.info(f"正在上传 {local_path} 到 {host}:{remote_path}")
        sftp.put(local_path, remote_path)
        logger.info(f"上传成功: {remote_path}")
        return True
    except Exception as e:
        logger.error(f"SFTP 上传失败 to {host}: {e}")
        return False
    finally:
        if sftp: sftp.close()
        if transport: transport.close()

def process_and_upload_in_background(session_id: str):
    """在后台处理所有文件并上传"""
    session = build_status_response(session_id)
    if not session:
        logger.error(f"后台任务启动失败：未找到会话 {session_id}")
        return

    job = get_job_by_session_id(session_id)
    if not job:
        logger.error(f"后台任务启动失败：未找到任务 {session_id}")
        update_session_progress(session_id, status='error', phase='error', error='任务不存在')
        return

    blur_face_plates = bool(job["blur_face_plates"])
    blur_texts = bool(job["blur_texts"])
    blur_method = job["blur_method"]
    anonymization_enabled = bool(job["anonymization_enabled"])
    remote_dir_name = job["remote_dir_name"] or session_id
    artifact_name = f"{remote_dir_name}.zip"
    storage_mode = session.get("storage_mode") or "local"

    upload_session_row = get_upload_session(session_id)
    region_name = upload_session_row["server_region"] if upload_session_row else "europe"
    region_config = SERVER_REGIONS.get(region_name, SERVER_REGIONS["europe"])
    sftp_config = region_config.get("sftp", {})
    sftp_host = _normalize_sftp_value(sftp_config.get("host"))
    sftp_user = _normalize_sftp_value(sftp_config.get("user"))
    sftp_pass = _normalize_sftp_value(sftp_config.get("password"))

    update_job(session_id, status='running', phase='extracting', error=None)
    update_session_progress(session_id, status='running', phase='extracting', error=None)

    input_dir = get_input_dir(session_id)
    extracted_dir = get_extracted_dir(session_id)
    output_dir = get_output_dir(session_id)
    work_dir = get_work_dir(session_id)
    input_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    processed_files_count = 0
    final_output_path = Path(SFTP_REMOTE_DIR) / artifact_name
    final_output_path_str = str(final_output_path)
    start_time = datetime.now()
    job_files = list_job_files(session_id)
    uploaded_files_paths = [row["local_staged_path"] for row in job_files]

    update_upload_session(session_id, storage_mode=storage_mode, output_path=final_output_path_str)
    update_job(session_id, output_artifact_name=artifact_name, output_artifact_path=final_output_path_str)

    progress_state = default_progress_payload(len(uploaded_files_paths))
    summary_state = default_summary_payload(len(uploaded_files_paths))
    update_session_progress(
        session_id,
        progress=progress_state,
        summary=summary_state,
        results=default_results_payload(),
    )

    logger.info(f"后台任务 {session_id} 开始：收到 {len(uploaded_files_paths)} 个顶层上传项")

    try:
        manifest = []
        used_rel_paths = set()
        archives_detected = 0

        for index, file_path_str in enumerate(uploaded_files_paths):
            if (datetime.now() - start_time).total_seconds() > PROCESSING_TIMEOUT:
                raise TimeoutError("处理超时")

            file_path = Path(file_path_str)
            rel_input_path = file_path.relative_to(input_dir)

            if _is_archive_path(file_path):
                archives_detected += 1
                summary_state["archives"] = archives_detected
                progress_state["archives_detected"] = archives_detected
                update_session_progress(session_id, summary={"archives": archives_detected}, progress={"archives_detected": archives_detected, "current_item": str(rel_input_path)})

                archive_stem = file_path.name.lower()
                for suffix in sorted(ALLOWED_ARCHIVE_EXTENSIONS, key=len, reverse=True):
                    if archive_stem.endswith(suffix):
                        archive_stem = file_path.name[:-len(suffix)] or file_path.stem
                        break
                archive_extract_root = extracted_dir / f"{index}_{archive_stem}"
                archive_extract_root.mkdir(parents=True, exist_ok=True)

                try:
                    if file_path.name.lower().endswith('.zip'):
                        _safe_extract_zip(file_path, archive_extract_root)
                    else:
                        _safe_extract_tar(file_path, archive_extract_root)
                except Exception as e:
                    summary_state["corrupt_archives"] += 1
                    progress_state["error_files"] += 1
                    update_session_progress(
                        session_id,
                        summary={"corrupt_archives": summary_state["corrupt_archives"]},
                        progress={"error_files": progress_state["error_files"]},
                        results={"sample_errors": [f"{rel_input_path}: {e}"]},
                    )
                    continue

                archive_entries = [path for path in archive_extract_root.rglob('*') if path.is_file()]
                if not archive_entries:
                    summary_state["empty_archives"] += 1
                    update_session_progress(
                        session_id,
                        summary={"empty_archives": summary_state["empty_archives"]},
                        results={"warnings": [f"{rel_input_path}: 压缩包内未发现文件"]},
                    )
                for archive_entry in archive_entries:
                    rel_archive_path = archive_entry.relative_to(archive_extract_root)
                    logical_rel_path = Path(archive_stem) / rel_archive_path
                    output_file = _resolve_output_path(output_dir, logical_rel_path, used_rel_paths)
                    if output_file.relative_to(output_dir) != logical_rel_path:
                        summary_state["duplicate_output_paths_resolved"] += 1
                        update_session_progress(session_id, summary={"duplicate_output_paths_resolved": summary_state["duplicate_output_paths_resolved"]})
                    manifest.append({
                        "source_kind": "archive_entry",
                        "source_path": archive_entry,
                        "logical_rel_path": output_file.relative_to(output_dir),
                        "display_name": f"{rel_input_path}!/{rel_archive_path}",
                        "is_image": _is_image_path(archive_entry),
                    })
                    if _is_image_path(archive_entry):
                        summary_state["extracted_images"] += 1

                progress_state["archives_processed"] += 1
                progress_state["images_discovered"] = len([item for item in manifest if item["is_image"]])
                update_session_progress(
                    session_id,
                    summary={"extracted_images": summary_state["extracted_images"]},
                    progress={
                        "archives_processed": progress_state["archives_processed"],
                        "images_discovered": progress_state["images_discovered"],
                    },
                )
                continue

            output_file = _resolve_output_path(output_dir, rel_input_path, used_rel_paths)
            manifest.append({
                "source_kind": "direct_upload",
                "source_path": file_path,
                "logical_rel_path": output_file.relative_to(output_dir),
                "display_name": str(rel_input_path),
                "is_image": _is_image_path(file_path),
            })
            if _is_image_path(file_path):
                summary_state["direct_images"] += 1
                update_session_progress(session_id, summary={"direct_images": summary_state["direct_images"]})
            else:
                summary_state["non_images_preserved"] = summary_state.get("non_images_preserved", 0) + 1
                update_session_progress(session_id, summary={"non_images_preserved": summary_state["non_images_preserved"]})

        if not manifest:
            update_upload_session(session_id, processed_files=0)
            update_job(session_id, status='completed', phase='completed')
            update_session_progress(session_id, status='completed', phase='completed', results={"warnings": ["未发现可输出文件"]})
            return

        image_items = [item for item in manifest if item["is_image"]]
        progress_state["processable_images"] = len(image_items)
        progress_state["images_discovered"] = len(image_items)
        logger.info(f"任务 {session_id}: 实际发现 {len(image_items)} 张可处理图片，最终输出条目 {len(manifest)} 个")
        update_job(session_id, phase='processing')
        update_session_progress(session_id, phase='processing', progress={"processable_images": len(image_items), "images_discovered": len(image_items)})

        for idx, item in enumerate(manifest, 1):
            if (datetime.now() - start_time).total_seconds() > PROCESSING_TIMEOUT:
                raise TimeoutError("处理超时")

            source_path = item["source_path"]
            output_file = output_dir / item["logical_rel_path"]
            output_file.parent.mkdir(parents=True, exist_ok=True)
            processed_path = str(source_path)
            current_label = item["display_name"]
            update_session_progress(session_id, progress={"current_item": current_label})

            try:
                if anonymization_enabled and item["is_image"]:
                    if blur_face_plates:
                        blurrer = get_unified_blurrer()
                        temp_path = str(work_dir / f"{idx}_unified{source_path.suffix.lower()}")
                        blurrer.process_image(processed_path, temp_path, method=blur_method)
                        processed_path = temp_path

                    if blur_texts:
                        blurrer = get_text_blurrer()
                        temp_path = str(work_dir / f"{idx}_text{source_path.suffix.lower()}")
                        blurrer.process_image(processed_path, temp_path, method=blur_method)
                        processed_path = temp_path

                    processed_files_count += 1
                    progress_state["processed_images"] = processed_files_count
                    update_session_progress(session_id, progress={"processed_images": processed_files_count})

                shutil.copy2(processed_path, output_file)
            except Exception as e:
                progress_state["error_files"] += 1
                update_session_progress(
                    session_id,
                    progress={"error_files": progress_state["error_files"]},
                    results={"sample_errors": [f"{current_label}: {e}"]},
                )

        update_job(session_id, phase='delivering', status='delivering')
        update_session_progress(session_id, status='running', phase='uploading')

        zip_path = work_dir / artifact_name
        _create_zip_from_tree(output_dir, zip_path)

        uploaded_outputs = 0
        if storage_mode == 'sftp' and _is_sftp_configured(sftp_host, sftp_user, sftp_pass):
            logger.info(f"任务 {session_id}: 检测到 SFTP 登录信息，开始上传 zip...")
            upload_success = upload_file_sftp(sftp_host, sftp_user, sftp_pass, str(zip_path), SFTP_REMOTE_DIR, artifact_name)
            if upload_success:
                uploaded_outputs = 1
                update_session_progress(session_id, progress={"uploaded_outputs": uploaded_outputs})
            else:
                progress_state["error_files"] += 1
                update_session_progress(
                    session_id,
                    progress={"error_files": progress_state["error_files"]},
                    results={"sample_errors": [f"{artifact_name}: SFTP 上传失败"]},
                )
        else:
            logger.warning(f"任务 {session_id}: 未配置 SFTP，结果将保存到本机目录 {final_output_path_str}")
            _deliver_output_zip_local(zip_path, artifact_name)
            uploaded_outputs = 1
            update_session_progress(session_id, progress={"uploaded_outputs": uploaded_outputs})

        update_upload_session(session_id, processed_files=processed_files_count, output_path=final_output_path_str)
        update_job(session_id, output_artifact_name=artifact_name, output_artifact_path=final_output_path_str)
        logger.info(f"任务 {session_id}: 已完成处理 {processed_files_count} 张图片，正在结束任务")
        update_job(session_id, status='completed', phase='completed')
        update_session_progress(session_id, status='completed', phase='completed')
        remove_session_staging(session_id)

    except Exception as e:
        logger.error(f"后台任务 {session_id} 发生致命错误: {e}")
        update_job(session_id, status='error', phase='error', error=str(e))
        update_session_progress(session_id, status='error', phase='error', error=str(e))

@app.get("/remote/local/list")
async def remote_local_list(request: Request):
    _require_admin_session(request)
    return {"files": _list_local_zip_outputs()}


@app.get("/remote/local/download")
async def remote_local_download(request: Request, name: str):
    lang = get_lang_from_request(request)
    _require_admin_session(request)
    artifact_name = _sanitize_artifact_name(name, lang)
    file_path = (Path(SFTP_REMOTE_DIR) / artifact_name).resolve()
    base_dir = Path(SFTP_REMOTE_DIR).resolve()
    try:
        file_path.relative_to(base_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail=translate("error_invalid_zip_name", lang))
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=translate("error_file_not_found", lang))
    return FileResponse(path=str(file_path), filename=artifact_name, media_type="application/zip")


@app.get("/remote/remote/list")
async def remote_remote_list(request: Request, region: str):
    _require_admin_session(request)
    lang = get_lang_from_request(request)
    return {"files": _list_remote_zip_outputs(region, lang)}


@app.get("/remote/remote/download")
async def remote_remote_download(request: Request, region: str, name: str, background_tasks: BackgroundTasks):
    _require_admin_session(request)
    lang = get_lang_from_request(request)
    artifact_name = _sanitize_artifact_name(name, lang)
    transport = None
    sftp = None
    try:
        transport, sftp, *_ = _open_region_sftp(region, lang)
        remote_file_path = f"{SFTP_REMOTE_DIR.rstrip('/')}/{artifact_name}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            sftp.get(remote_file_path, tmp_file.name)
            tmp_file_path = tmp_file.name
        background_tasks.add_task(os.remove, tmp_file_path)
        return FileResponse(path=tmp_file_path, filename=artifact_name, media_type="application/zip", background=background_tasks)
    finally:
        if sftp:
            sftp.close()
        if transport:
            transport.close()


@app.post("/remote/logout")
async def remote_logout(request: Request):
    lang = get_lang_from_request(request)
    token = request.cookies.get("remote_session")
    if token and token in _SESSIONS:
        _SESSIONS.pop(token, None)
    resp = RedirectResponse(url=f"/remote?lang={lang}", status_code=302)
    resp.delete_cookie("remote_session")
    return resp


@app.get("/remote", response_class=HTMLResponse)
async def remote_page(request: Request, region: str = "europe"):
    lang = get_lang_from_request(request)
    def t(key: str):
        return translate(key, lang)

    session = _get_session_from_cookie(request)
    context = {
        "request": request,
        "lang": lang,
        "t": t,
        "authed": bool(session and session.get("role") == "admin"),
        "login_error": None,
        "anonymization_enabled": _get_admin_anonymization_enabled(),
        "local_files": [],
        "remote_files": [],
        "remote_error": None,
        "selected_region": region,
        "regions": [
            {"key": "europe", "label": translate("server_europe", lang)},
            {"key": "america", "label": translate("server_america", lang)},
            {"key": "asia", "label": translate("server_asia", lang)},
        ],
    }
    if context["authed"]:
        context["local_files"] = _list_local_zip_outputs()
        try:
            context["remote_files"] = _list_remote_zip_outputs(region, lang)
        except Exception as exc:
            logger.error(f"无法列出远程 zip 文件: {exc}")
            context["remote_error"] = translate("error_remote_list_failed", lang)
    response = templates.TemplateResponse("remote.html", request=request, context=context)
    response.set_cookie("lang", lang, max_age=3600*24*365)
    return response


@app.post("/remote/login")
async def remote_login(request: Request, password: str = Form(...)):
    lang = get_lang_from_request(request)
    region = request.query_params.get("region", "europe")
    if password == ADMIN_CONFIG["password"] and ADMIN_CONFIG["password"]:
        token = _new_admin_session()
        resp = RedirectResponse(url=f"/remote?lang={lang}&region={region}", status_code=302)
        resp.set_cookie("remote_session", token, httponly=True, max_age=3600*8)
        resp.set_cookie("lang", lang, max_age=3600*24*365)
        return resp

    def t(key: str):
        return translate(key, lang)
    context = {
        "request": request,
        "lang": lang,
        "t": t,
        "authed": False,
        "login_error": translate("remote_invalid", lang),
        "anonymization_enabled": _get_admin_anonymization_enabled(),
        "local_files": [],
        "remote_files": [],
        "remote_error": None,
        "selected_region": "europe",
        "regions": [
            {"key": "europe", "label": translate("server_europe", lang)},
            {"key": "america", "label": translate("server_america", lang)},
            {"key": "asia", "label": translate("server_asia", lang)},
        ],
    }
    return templates.TemplateResponse("remote.html", request=request, context=context)


@app.post("/remote/settings/anonymization")
async def remote_update_anonymization(request: Request, enabled: str = Form(...), region: str = Form("europe")):
    _require_admin_session(request)
    lang = get_lang_from_request(request)
    set_app_setting("anonymization_enabled", "1" if enabled.lower() in {"1", "true", "yes", "on"} else "0")
    return RedirectResponse(url=f"/remote?lang={lang}&region={region}", status_code=302)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """主页"""
    lang = get_lang_from_request(request)
    def t(key: str):
        return translate(key, lang)
    context = {"request": request, "lang": lang, "t": t, "sftp_remote_dir": SFTP_REMOTE_DIR}
    response = templates.TemplateResponse("index.html", request=request, context=context)
    response.set_cookie("lang", lang, max_age=3600*24*365)
    return response

@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    """隐私协议页面"""
    lang = get_lang_from_request(request)
    def t(key: str):
        return translate(key, lang)
    context = {"request": request, "lang": lang, "t": t}
    response = templates.TemplateResponse("privacy.html", request=request, context=context)
    response.set_cookie("lang", lang, max_age=3600*24*365)
    return response

@app.post("/api/upload")
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    enable_anonymization: bool = Form(True),
    server_region: str = Form("europe"),
    blur_face_plates: bool = Form(True),
    blur_texts: bool = Form(True),
    blur_method: str = Form("gaussian")
):
    """接收文件，启动后台处理任务，并立即返回响应"""

    lang = get_lang_from_request(request)

    if not files:
        raise HTTPException(status_code=400, detail=translate("error_no_files_selected", lang))
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=translate("error_file_count_limit", lang).replace("{limit}", str(MAX_FILES_PER_REQUEST)))
    if blur_method not in {"gaussian", "pixelate", "solid"}:
        raise HTTPException(status_code=400, detail=translate("error_unsupported_blur_method", lang))

    session_id = str(uuid.uuid4())
    staging_root = ensure_session_dirs(session_id)
    input_dir = get_input_dir(session_id)
    enable_anonymization = _get_admin_anonymization_enabled()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    region_code = REGION_MAP.get(server_region, "XX")
    remote_dir_name = f"{ts}_{region_code}"

    progress = default_progress_payload(len(files))
    summary = default_summary_payload(len(files))
    results = default_results_payload()

    create_upload_session(
        session_id=session_id,
        server_region=server_region,
        staging_root=str(staging_root),
        request_file_count=len(files),
        status="uploading",
        phase="uploading",
        progress=progress,
        summary=summary,
        results=results,
    )
    job_id = create_job(
        upload_session_id=session_id,
        blur_face_plates=blur_face_plates,
        blur_texts=blur_texts,
        blur_method=blur_method,
        remote_dir_name=remote_dir_name,
        anonymization_enabled=enable_anonymization,
    )

    try:
        staged_paths = ingest_upload_files(
            files=files,
            session_id=session_id,
            job_id=job_id,
            input_dir=input_dir,
            sanitize_path=lambda filename: _sanitize_relative_upload_path(filename, lang),
        )
    except HTTPException as exc:
        update_session_progress(session_id, status="error", phase="error", error=exc.detail)
        raise
    except Exception:
        update_session_progress(session_id, status="error", phase="error", error=translate("error_upload_save_failed", lang))
        raise

    if not staged_paths:
        update_session_progress(session_id, status="error", phase="error", error=translate("error_no_valid_files", lang))
        raise HTTPException(status_code=400, detail=translate("error_no_valid_files", lang))

    # 根据服务器区域自动获取SFTP凭据
    region_config = SERVER_REGIONS.get(server_region, SERVER_REGIONS["europe"])
    sftp_config = region_config.get("sftp", {})
    sftp_host = _normalize_sftp_value(sftp_config.get("host"))
    sftp_user = _normalize_sftp_value(sftp_config.get("user"))
    sftp_pass = _normalize_sftp_value(sftp_config.get("password"))
    storage_mode = "sftp" if _is_sftp_configured(sftp_host, sftp_user, sftp_pass) else "local"

    update_upload_session(session_id, storage_mode=storage_mode, output_path=str(Path(SFTP_REMOTE_DIR) / remote_dir_name))
    update_session_progress(session_id, status="queued", phase="queued")

    logger.info(f"会话 {session_id}: 已接收 {len(staged_paths)} 个文件，即将启动后台处理。")
    logger.info(f"会话 {session_id}: 使用服务器区域 {server_region}，存储模式: {storage_mode}，SFTP主机: {sftp_host or 'local'}")

    return JSONResponse({"success": True, "session_id": session_id})

@app.post("/api/test_sftp")
async def test_sftp_connection(
    request: Request,
    host: str = Form(...),
    user: str = Form(...),
    a_pass: str = Form(...)
):
    """测试 SFTP 连接"""
    lang = get_lang_from_request(request)
    transport = None
    try:
        transport = paramiko.Transport((host, 22))
        transport.set_keepalive(5)
        transport.connect(username=user, password=a_pass)
        return {"status": "ok", "message": translate("sftp_success", lang)}
    except paramiko.AuthenticationException:
        raise HTTPException(status_code=401, detail=translate("sftp_auth_fail", lang))
    except Exception as e:
        logger.error(f"SFTP 连接测试失败: {e}")
        raise HTTPException(status_code=500, detail=translate("sftp_conn_fail", lang))
    finally:
        if transport and transport.is_active():
            transport.close()

@app.get("/api/status/{session_id}")
async def check_processing_status(request: Request, session_id: str):
    """检查处理状态"""
    status = build_status_response(session_id)
    if not status:
        lang = get_lang_from_request(request)
        raise HTTPException(status_code=404, detail=translate("error_session_not_found", lang))
    return status


if __name__ == "__main__":
    # 配置日志：移除默认处理器，添加控制台和文件输出
    logger.remove()

    # 控制台输出（带颜色）
    logger.add(
        sys.stdout,
        colorize=True,
        level=LOG_CONFIG["level"],
        format=LOG_CONFIG["format"]
    )

    # 文件输出（带轮转和保留策略）
    logger.add(
        str(LOG_CONFIG["file"]),
        rotation=LOG_CONFIG["rotation"],
        retention=LOG_CONFIG["retention"],
        level=LOG_CONFIG["level"],
        format=LOG_CONFIG["format"],
        encoding="utf-8"
    )

    logger.info(f"日志系统已初始化，日志文件: {LOG_CONFIG['file']}")

    server_port = int(os.getenv("PORT", os.getenv("SERVER_PORT", "8000")))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=server_port,
        reload=False,
        log_level="info"
    )
