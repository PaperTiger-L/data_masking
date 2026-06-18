"""
数据脱敏系统配置文件
"""

import os
from pathlib import Path

import yaml

# 基础路径
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"
LOGS_DIR = BASE_DIR / "logs"
STAGING_ROOT = Path(os.getenv("STAGING_ROOT", str(BASE_DIR / "staging")))
SQLITE_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "data_masking.db")))
STAGING_RETENTION_HOURS = int(os.getenv("STAGING_RETENTION_HOURS", 72))
RUNNER_POLL_INTERVAL_SECONDS = int(os.getenv("RUNNER_POLL_INTERVAL_SECONDS", 5))
JOB_LEASE_SECONDS = int(os.getenv("JOB_LEASE_SECONDS", 120))
JOB_RECOVERY_GRACE_SECONDS = int(os.getenv("JOB_RECOVERY_GRACE_SECONDS", 300))

# 文件上传限制
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 8 * 1024 * 1024 * 1024))  # 8GB，单文件/单压缩包上限
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
ALLOWED_ARCHIVE_EXTENSIONS = {'.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz'}
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", 30000))
MAX_ARCHIVE_TOTAL_UNCOMPRESSED_SIZE = int(os.getenv("MAX_ARCHIVE_TOTAL_UNCOMPRESSED_SIZE", 20 * 1024 * 1024 * 1024))  # 20GB
MAX_EXTRACTED_FILES_PER_ARCHIVE = int(os.getenv("MAX_EXTRACTED_FILES_PER_ARCHIVE", 50000))
MAX_PROCESSABLE_IMAGES_PER_REQUEST = int(os.getenv("MAX_PROCESSABLE_IMAGES_PER_REQUEST", 30000))

# 处理配置
PROCESSING_TIMEOUT = int(os.getenv("PROCESSING_TIMEOUT", 24 * 60 * 60))  # 24小时超时

# 日志配置
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "file": LOGS_DIR / "app.log",
    "rotation": "1 day",
    "retention": "7 days",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
}

# 脱敏参数（统一从配置读取，可被环境变量覆盖）
ANONYMIZATION = {
    # 文本区域膨胀与额外padding
    "text_dilate_px": int(os.getenv("TEXT_DILATE_PX", "8")),
    "text_pad_ratio": float(os.getenv("TEXT_PAD_RATIO", "0.0")),
    "text_use_padded_rect": os.getenv("TEXT_USE_PADDED_RECT", "false").lower() in ("1", "true", "yes"),
}

SERVER_REGIONS_CONFIG_PATH = BASE_DIR / "config" / "server_regions.yaml"


def _load_server_regions_config():
    with open(SERVER_REGIONS_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_server_regions(config_data: dict):
    regions = {
        key: value for key, value in config_data.items()
        if key in {"europe", "america", "asia"}
    }

    env_overrides = {
        "europe": {
            "host": os.getenv("EU_SFTP_HOST", "").strip(),
            "user": os.getenv("EU_SFTP_USER", "").strip(),
            "password": os.getenv("EU_SFTP_PASSWORD", "").strip(),
        },
        "america": {
            "host": os.getenv("US_SFTP_HOST", "").strip(),
            "user": os.getenv("US_SFTP_USER", "").strip(),
            "password": os.getenv("US_SFTP_PASSWORD", "").strip(),
        },
        "asia": {
            "host": os.getenv("AS_SFTP_HOST", "").strip(),
            "user": os.getenv("AS_SFTP_USER", "").strip(),
            "password": os.getenv("AS_SFTP_PASSWORD", "").strip(),
        },
    }

    for region_name, overrides in env_overrides.items():
        region_config = regions.setdefault(region_name, {})
        sftp_config = region_config.setdefault("sftp", {})
        for key, value in overrides.items():
            if value:
                sftp_config[key] = value
            else:
                sftp_config[key] = str(sftp_config.get(key, "")).strip()

    return regions


_SERVER_REGIONS_CONFIG = _load_server_regions_config()
ADMIN_CONFIG = {
    "password": str(_SERVER_REGIONS_CONFIG.get("admin", {}).get("password", "")).strip(),
    "anonymization_enabled": bool(_SERVER_REGIONS_CONFIG.get("admin", {}).get("anonymization_enabled", True)),
}

# 服务器区域配置
SERVER_REGIONS = _load_server_regions(_SERVER_REGIONS_CONFIG)
