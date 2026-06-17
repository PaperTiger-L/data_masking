#!/usr/bin/env python3
"""
数据脱敏系统主应用
支持人脸、车牌、街牌检测与模糊处理
"""

import os
import sys
import shutil
import tempfile
import zipfile
import mimetypes
from pathlib import Path
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
from config import ANONYMIZATION, PARALLEL, SERVER_REGIONS, LOG_CONFIG
from translations import TRANSLATIONS

app = FastAPI(title="数据脱敏系统", description="支持人脸、车牌、街牌检测与脱敏")

# 静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def translate(key: str, lang: str = "zh") -> str:
    """翻译函数"""
    entry = TRANSLATIONS.get(key, {})
    return entry.get(lang, entry.get("zh", key))

def get_lang_from_request(request: Request) -> str:
    lang = request.query_params.get("lang")
    if lang in ("zh", "en"):
        return lang
    cookie_lang = request.cookies.get("lang")
    if cookie_lang in ("zh", "en"):
        return cookie_lang
    return "zh"

# 全局配置
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")

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

# 处理状态跟踪
PROCESSING_SESSIONS = {}

# 远程访问会话管理
_SESSIONS: dict[str, dict] = {}

def _new_session(host: str, user: str, a_pass: str) -> str:
    """创建一个新的远程会话，并存储连接凭据"""
    token = uuid.uuid4().hex
    # 警告：在生产环境中，不应明文存储密码。应使用加密或更安全的令牌机制。
    _SESSIONS[token] = {"host": host, "user": user, "pass": a_pass}
    return token

def _get_session_from_cookie(request: Request) -> Optional[dict]:
    """从 cookie 中获取远程会话信息"""
    token = request.cookies.get("remote_session")
    if token and token in _SESSIONS:
        return _SESSIONS[token]
    return None

def _safe_region_to_dir(region: str) -> Path:
    if region not in REGION_MAP:
        raise HTTPException(status_code=400, detail="Invalid region")
    return UPLOAD_DIR / REGION_MAP[region]

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
            raise HTTPException(status_code=500, detail="统一检测器初始化失败")
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
            raise HTTPException(status_code=500, detail="文本检测器初始化失败")
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

def process_and_upload_in_background(
    session_id: str,
    session_dir_str: str,
    uploaded_files_paths: List[str],
    blur_face_plates: bool,
    blur_texts: bool,
    blur_method: str,
    sftp_host: Optional[str],
    sftp_user: Optional[str],
    sftp_pass: Optional[str],
    remote_dir_name: str,
    storage_mode: str,
):
    """在后台处理所有文件并上传"""
    if session_id not in PROCESSING_SESSIONS:
        logger.error(f"后台任务启动失败：未找到会话 {session_id}")
        return

    PROCESSING_SESSIONS[session_id]['status'] = 'processing'
    session_dir = Path(session_dir_str)
    output_dir = session_dir / "output"
    output_dir.mkdir(exist_ok=True)
    processed_files_count = 0
    final_output_dir = Path(SFTP_REMOTE_DIR) / remote_dir_name
    final_output_dir_str = str(final_output_dir)

    PROCESSING_SESSIONS[session_id]['storage_mode'] = storage_mode
    PROCESSING_SESSIONS[session_id]['output_path'] = final_output_dir_str

    def _is_image(p: Path) -> bool:
        return p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    logger.info(f"后台任务 {session_id} 开始：处理 {len(uploaded_files_paths)} 个文件")

    try:
        for file_path_str in uploaded_files_paths:
            file_path = Path(file_path_str)
            processed_path = str(file_path)
            
            if _is_image(file_path):
                if blur_face_plates:
                    try:
                        blurrer = get_unified_blurrer()
                        temp_path = str(session_dir / f"temp_unified_{file_path.name}")
                        blurrer.process_image(processed_path, temp_path, method=blur_method)
                        processed_path = temp_path
                    except Exception as e:
                        logger.error(f"人脸/车牌模糊处理失败: {e}")

                if blur_texts:
                    try:
                        blurrer = get_text_blurrer()
                        temp_path = str(session_dir / f"temp_text_{file_path.name}")
                        blurrer.process_image(processed_path, temp_path, method=blur_method)
                        processed_path = temp_path
                    except Exception as e:
                        logger.error(f"文本模糊处理失败: {e}")

            shutil.copy2(processed_path, output_dir / file_path.name)
            processed_files_count += 1

        logger.info(f"任务 {session_id}: 所有文件处理完成。共处理 {processed_files_count} 个文件。")

        if storage_mode == 'sftp' and _is_sftp_configured(sftp_host, sftp_user, sftp_pass):
            logger.info(f"任务 {session_id}: 检测到 SFTP 登录信息，开始上传...")
            remote_target_dir = str(final_output_dir)

            for f in output_dir.iterdir():
                if f.is_file():
                    upload_file_sftp(sftp_host, sftp_user, sftp_pass, str(f), remote_target_dir, f.name)
        else:
            logger.warning(f"任务 {session_id}: 未配置 SFTP，结果将保存到本机目录 {final_output_dir_str}")
            final_output_dir.mkdir(parents=True, exist_ok=True)
            for f in output_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, final_output_dir / f.name)

        PROCESSING_SESSIONS[session_id]['status'] = 'completed'
        PROCESSING_SESSIONS[session_id]['processed_files'] = processed_files_count

    except Exception as e:
        logger.error(f"后台任务 {session_id} 发生致命错误: {e}")
        PROCESSING_SESSIONS[session_id]['status'] = 'error'
        PROCESSING_SESSIONS[session_id]['error'] = str(e)
    finally:
        try:
            shutil.rmtree(session_dir)
            logger.info(f"已清理临时目录: {session_dir}")
        except Exception as e:
            logger.error(f"清理临时目录失败: {e}")

@app.get("/remote/list")
async def remote_list(request: Request):
    session = _get_session_from_cookie(request)
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")

    host, user, a_pass = session["host"], session["user"], session["pass"]
    remote_base_dir = "/mnt/data"
    folders = []
    transport = None
    sftp = None
    try:
        transport = paramiko.Transport((host, 22))
        transport.connect(username=user, password=a_pass)
        sftp = paramiko.SFTPClient.from_transport(transport)
        for attr in sftp.listdir_attr(remote_base_dir):
            if paramiko.sftp_client.stat.S_ISDIR(attr.st_mode):
                folders.append(attr.filename)
        folders.sort(reverse=True)
        return {"folders": folders}
    except Exception as e:
        logger.error(f"无法列出远程目录 {remote_base_dir}: {e}")
        raise HTTPException(status_code=500, detail="无法列出远程目录")
    finally:
        if sftp: sftp.close()
        if transport: transport.close()

@app.get("/remote/list")
async def remote_list(request: Request):
    session = _get_session_from_cookie(request)
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")

    host, user, a_pass = session["host"], session["user"], session["pass"]
    remote_base_dir = "/mnt/data"
    folders = []
    transport = None
    sftp = None
    try:
        transport = paramiko.Transport((host, 22))
        transport.connect(username=user, password=a_pass)
        sftp = paramiko.SFTPClient.from_transport(transport)
        for attr in sftp.listdir_attr(remote_base_dir):
            if paramiko.sftp_client.stat.S_ISDIR(attr.st_mode):
                folders.append(attr.filename)
        folders.sort(reverse=True)
        return {"folders": folders}
    except Exception as e:
        logger.error(f"无法列出远程目录 {remote_base_dir}: {e}")
        raise HTTPException(status_code=500, detail="无法列出远程目录")
    finally:
        if sftp: sftp.close()
        if transport: transport.close()

@app.get("/remote/files")
async def remote_files(request: Request, folder: str):
    session = _get_session_from_cookie(request)
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")

    host, user, a_pass = session["host"], session["user"], session["pass"]
    remote_folder_path = os.path.join("/mnt/data", folder)
    files = []
    transport = None
    sftp = None
    try:
        transport = paramiko.Transport((host, 22))
        transport.connect(username=user, password=a_pass)
        sftp = paramiko.SFTPClient.from_transport(transport)
        for attr in sftp.listdir_attr(remote_folder_path):
            if not paramiko.sftp_client.stat.S_ISDIR(attr.st_mode):
                files.append({"name": attr.filename, "size": attr.st_size})
        files.sort(key=lambda x: x["name"])
        return {"folder": folder, "files": files}
    except Exception as e:
        logger.error(f"无法列出远程文件 {remote_folder_path}: {e}")
        raise HTTPException(status_code=500, detail="无法列出远程文件")
    finally:
        if sftp: sftp.close()
        if transport: transport.close()

@app.get("/remote/file")
async def remote_file(request: Request, folder: str, name: str, background_tasks: BackgroundTasks):
    session = _get_session_from_cookie(request)
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")

    host, user, a_pass = session["host"], session["user"], session["pass"]
    remote_file_path = os.path.join("/mnt/data", folder, name)
    
    transport = None
    sftp = None
    try:
        transport = paramiko.Transport((host, 22))
        transport.connect(username=user, password=a_pass)
        sftp = paramiko.SFTPClient.from_transport(transport)

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            sftp.get(remote_file_path, tmp_file.name)
            tmp_file_path = tmp_file.name

        media_type, _ = mimetypes.guess_type(str(remote_file_path))
        
        background_tasks.add_task(os.remove, tmp_file_path)
        
        return FileResponse(path=tmp_file_path, media_type=media_type or "application/octet-stream", background=background_tasks)

    except Exception as e:
        logger.error(f"无法下载远程文件 {remote_file_path}: {e}")
        raise HTTPException(status_code=500, detail="无法下载远程文件")
    finally:
        if sftp: sftp.close()
        if transport: transport.close()


@app.get("/remote/download")
async def remote_download(request: Request, folder: str, background_tasks: BackgroundTasks):
    session = _get_session_from_cookie(request)
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")

    host, user, a_pass = session["host"], session["user"], session["pass"]
    remote_folder_path = os.path.join("/mnt/data", folder)
    
    tmp_dir = Path(tempfile.gettempdir())
    zip_path = tmp_dir / f"{folder}.zip"
    if zip_path.exists():
        zip_path.unlink()

    transport = None
    sftp = None
    try:
        transport = paramiko.Transport((host, 22))
        transport.connect(username=user, password=a_pass)
        sftp = paramiko.SFTPClient.from_transport(transport)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for attr in sftp.listdir_attr(remote_folder_path):
                if not paramiko.sftp_client.stat.S_ISDIR(attr.st_mode):
                    remote_file_path = os.path.join(remote_folder_path, attr.filename)
                    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                        sftp.get(remote_file_path, tmp_file.name)
                        zf.write(tmp_file.name, attr.filename)
        
        def _cleanup_zip(p: str):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        background_tasks.add_task(_cleanup_zip, str(zip_path))
        return FileResponse(path=str(zip_path), filename=f"{folder}.zip")

    except Exception as e:
        logger.error(f"无法打包下载远程目录 {remote_folder_path}: {e}")
        raise HTTPException(status_code=500, detail="无法打包下载远程目录")
    finally:
        if sftp: sftp.close()
        if transport: transport.close()

@app.get("/remote/files")
async def remote_files(request: Request, folder: str):
    session = _get_session_from_cookie(request)
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")

    host, user, a_pass = session["host"], session["user"], session["pass"]
    remote_folder_path = os.path.join("/mnt/data", folder)
    files = []
    transport = None
    sftp = None
    try:
        transport = paramiko.Transport((host, 22))
        transport.connect(username=user, password=a_pass)
        sftp = paramiko.SFTPClient.from_transport(transport)
        for attr in sftp.listdir_attr(remote_folder_path):
            if not paramiko.sftp_client.stat.S_ISDIR(attr.st_mode):
                files.append({"name": attr.filename, "size": attr.st_size})
        files.sort(key=lambda x: x["name"])
        return {"folder": folder, "files": files}
    except Exception as e:
        logger.error(f"无法列出远程文件 {remote_folder_path}: {e}")
        raise HTTPException(status_code=500, detail="无法列出远程文件")
    finally:
        if sftp: sftp.close()
        if transport: transport.close()

@app.get("/remote/file")
async def remote_file(request: Request, folder: str, name: str, background_tasks: BackgroundTasks):
    session = _get_session_from_cookie(request)
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")

    host, user, a_pass = session["host"], session["user"], session["pass"]
    remote_file_path = os.path.join("/mnt/data", folder, name)
    
    transport = None
    sftp = None
    try:
        transport = paramiko.Transport((host, 22))
        transport.connect(username=user, password=a_pass)
        sftp = paramiko.SFTPClient.from_transport(transport)

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            sftp.get(remote_file_path, tmp_file.name)
            tmp_file_path = tmp_file.name

        media_type, _ = mimetypes.guess_type(str(remote_file_path))
        
        # Use BackgroundTasks to ensure cleanup happens
        background_tasks.add_task(os.remove, tmp_file_path)
        return FileResponse(path=tmp_file_path, media_type=media_type or "application/octet-stream")

    except Exception as e:
        logger.error(f"无法下载远程文件 {remote_file_path}: {e}")
        raise HTTPException(status_code=500, detail="无法下载远程文件")
    finally:
        if sftp: sftp.close()
        if transport: transport.close()


@app.get("/remote/download")
async def remote_download(request: Request, folder: str, background_tasks: BackgroundTasks):
    session = _get_session_from_cookie(request)
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")

    host, user, a_pass = session["host"], session["user"], session["pass"]
    remote_folder_path = os.path.join("/mnt/data", folder)
    
    tmp_dir = Path(tempfile.gettempdir())
    zip_path = tmp_dir / f"{folder}.zip"
    if zip_path.exists():
        zip_path.unlink()

    transport = None
    sftp = None
    try:
        transport = paramiko.Transport((host, 22))
        transport.connect(username=user, password=a_pass)
        sftp = paramiko.SFTPClient.from_transport(transport)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for attr in sftp.listdir_attr(remote_folder_path):
                if not paramiko.sftp_client.stat.S_ISDIR(attr.st_mode):
                    remote_file_path = os.path.join(remote_folder_path, attr.filename)
                    # Download to a temporary in-memory buffer or a temp file on disk
                    with sftp.open(remote_file_path, 'rb') as remote_f:
                        zf.writestr(attr.filename, remote_f.read())
        
        def _cleanup_zip(p: str):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        background_tasks.add_task(_cleanup_zip, str(zip_path))
        return FileResponse(path=str(zip_path), filename=f"{folder}.zip")

    except Exception as e:
        logger.error(f"无法打包下载远程目录 {remote_folder_path}: {e}")
        raise HTTPException(status_code=500, detail="无法打包下载远程目录")
    finally:
        if sftp: sftp.close()
        if transport: transport.close()

@app.post("/remote/logout")
async def remote_logout(request: Request):
    token = request.cookies.get("remote_session")
    if token and token in _SESSIONS:
        _SESSIONS.pop(token, None)
    resp = RedirectResponse(url="/remote", status_code=302)
    resp.delete_cookie("remote_session")
    return resp

@app.get("/remote", response_class=HTMLResponse)
async def remote_page(request: Request):
    lang = get_lang_from_request(request)
    def t(key: str):
        return translate(key, lang)
    session = _get_session_from_cookie(request)
    authed_user = session.get("user") if session else None
    context = {"request": request, "lang": lang, "t": t, "authed": bool(session), "username": authed_user or ""}
    response = templates.TemplateResponse("remote.html", context)
    response.set_cookie("lang", lang, max_age=3600*24*365)
    return response

@app.post("/remote/login")
async def remote_login(request: Request, host: str = Form(...), username: str = Form(...), password: str = Form(...)):
    lang = get_lang_from_request(request)
    transport = None
    try:
        transport = paramiko.Transport((host, 22))
        transport.set_keepalive(5)
        transport.connect(username=username, password=password)
        # 验证成功，创建会话
        token = _new_session(host, username, password)
        resp = RedirectResponse(url="/remote", status_code=302)
        resp.set_cookie("remote_session", token, httponly=True, max_age=3600*8)
        resp.set_cookie("lang", lang, max_age=3600*24*365)
        return resp
    except paramiko.AuthenticationException:
        error_message = translate("sftp_auth_fail", lang)
    except Exception as e:
        logger.error(f"远程登录失败: {e}")
        error_message = translate("sftp_conn_fail", lang)
    finally:
        if transport and transport.is_active():
            transport.close()
    
    # 登录失败，返回错误信息
    def t(key: str):
        return translate(key, lang)
    context = {"request": request, "lang": lang, "t": t, "authed": False, "login_error": error_message}
    return templates.TemplateResponse("remote.html", context)

@app.get("/remote", response_class=HTMLResponse)
async def remote_page(request: Request):
    lang = get_lang_from_request(request)
    def t(key: str):
        return translate(key, lang)
    session = _get_session_from_cookie(request)
    authed_user = session.get("user") if session else None
    context = {"request": request, "lang": lang, "t": t, "authed": bool(session), "username": authed_user or ""}
    response = templates.TemplateResponse("remote.html", context)
    response.set_cookie("lang", lang, max_age=3600*24*365)
    return response

@app.post("/remote/login")
async def remote_login(request: Request, host: str = Form(...), username: str = Form(...), password: str = Form(...)):
    lang = get_lang_from_request(request)
    transport = None
    try:
        transport = paramiko.Transport((host, 22))
        transport.set_keepalive(5)
        transport.connect(username=username, password=password)
        # 验证成功，创建会话
        token = _new_session(host, username, password)
        resp = RedirectResponse(url="/remote", status_code=302)
        resp.set_cookie("remote_session", token, httponly=True, max_age=3600*8)
        resp.set_cookie("lang", lang, max_age=3600*24*365)
        return resp
    except paramiko.AuthenticationException:
        error_message = translate("sftp_auth_fail", lang)
    except Exception as e:
        logger.error(f"远程登录失败: {e}")
        error_message = translate("sftp_conn_fail", lang)
    finally:
        if transport and transport.is_active():
            transport.close()
    
    # 登录失败，返回错误信息
    def t(key: str):
        return translate(key, lang)
    context = {"request": request, "lang": lang, "t": t, "authed": False, "login_error": error_message}
    return templates.TemplateResponse("remote.html", context)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """主页"""
    lang = get_lang_from_request(request)
    def t(key: str):
        return translate(key, lang)
    context = {"request": request, "lang": lang, "t": t, "sftp_remote_dir": SFTP_REMOTE_DIR}
    response = templates.TemplateResponse("index.html", context)
    response.set_cookie("lang", lang, max_age=3600*24*365)
    return response

@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    """隐私协议页面"""
    lang = get_lang_from_request(request)
    def t(key: str):
        return translate(key, lang)
    context = {"request": request, "lang": lang, "t": t}
    response = templates.TemplateResponse("privacy.html", context)
    response.set_cookie("lang", lang, max_age=3600*24*365)
    return response

@app.post("/api/upload")
async def upload_files(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    enable_anonymization: bool = Form(True),
    server_region: str = Form("europe"),
    blur_face_plates: bool = Form(True),
    blur_texts: bool = Form(True),
    blur_method: str = Form("gaussian")
):
    """接收文件，启动后台处理任务，并立即返回响应"""

    if not files:
        raise HTTPException(status_code=400, detail="未选择文件")

    session_id = str(uuid.uuid4())
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    # 关键修复：立即为新任务创建初始状态
    PROCESSING_SESSIONS[session_id] = {
        "status": "queued",
        "error": None,
        "processed_files": 0,
        "storage_mode": None,
        "output_path": None,
    }

    input_dir = session_dir / "input"
    input_dir.mkdir(exist_ok=True)

    uploaded_files_paths = []
    for file in files:
        if file.filename:
            file_path = input_dir / file.filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            uploaded_files_paths.append(str(file_path))

    logger.info(f"会话 {session_id}: 已接收 {len(uploaded_files_paths)} 个文件，即将启动后台处理。")

    # 根据服务器区域自动获取SFTP凭据
    region_config = SERVER_REGIONS.get(server_region, SERVER_REGIONS["europe"])
    sftp_config = region_config.get("sftp", {})
    sftp_host = _normalize_sftp_value(sftp_config.get("host"))
    sftp_user = _normalize_sftp_value(sftp_config.get("user"))
    sftp_pass = _normalize_sftp_value(sftp_config.get("password"))
    storage_mode = "sftp" if _is_sftp_configured(sftp_host, sftp_user, sftp_pass) else "local"

    logger.info(f"会话 {session_id}: 使用服务器区域 {server_region}，存储模式: {storage_mode}，SFTP主机: {sftp_host or 'local'}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    region_code = REGION_MAP.get(server_region, "XX")
    remote_dir_name = f"{ts}_{region_code}"

    background_tasks.add_task(
        process_and_upload_in_background,
        session_id=session_id, # 传递 session_id
        session_dir_str=str(session_dir),
        uploaded_files_paths=uploaded_files_paths,
        blur_face_plates=blur_face_plates,
        blur_texts=blur_texts,
        blur_method=blur_method,
        sftp_host=sftp_host,
        sftp_user=sftp_user,
        sftp_pass=sftp_pass,
        remote_dir_name=remote_dir_name,
        storage_mode=storage_mode,
    )

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
async def check_processing_status(session_id: str):
    """检查处理状态"""
    if session_id not in PROCESSING_SESSIONS:
        raise HTTPException(status_code=404, detail="会话不存在")
    return PROCESSING_SESSIONS[session_id]


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
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
