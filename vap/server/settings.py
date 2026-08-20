from __future__ import annotations

import secrets
import threading
import uuid
from typing import Any

from vap.agent.runtime import VAPAgentRuntime
from vap.runtime_paths import (
    APP_DIR,
    VAP_CONFIG_PATH,
    VAP_LOGS_DIR,
    VAP_TEMP_CONFIG_DIR,
)
from vap.validation import PERFETTO_PORT

STATIC_DIR = APP_DIR / "public"
DEFAULT_CONFIG_PATH = APP_DIR / "example-config.json"
CONFIG_PATH = VAP_CONFIG_PATH
LOGS_DIR = VAP_LOGS_DIR
TEMP_CONFIG_DIR = VAP_TEMP_CONFIG_DIR
DEFAULT_SERVER_HOST = "0.0.0.0"
SERVER_BIND_HOST = DEFAULT_SERVER_HOST
SERVER_SESSION_ID = uuid.uuid4().hex
SERVER_AUTH_TOKEN = secrets.token_urlsafe(32)
SERVER_COOKIE_NAME = f"vap_session_{SERVER_SESSION_ID[:12]}"
MAX_JSON_BODY_BYTES = 2 * 1024 * 1024
MAX_PROFILE_ARCHIVE_FILES = 4096
MAX_PROFILE_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
TEMP_CONFIG_MAX_AGE_SEC = 7 * 24 * 60 * 60
RUN_LOCK = threading.Lock()
RUN_START_LOCK = threading.Lock()
RUN_STATE: dict[str, Any] = {
    "process": None,
    "pid": None,
    "running": False,
    "exit_code": None,
    "started_at": None,
    "ended_at": None,
    "run_dir": None,
    "config_path": None,
    "output": "",
    "stop_requested": False,
}
AGENT_RUNTIME = VAPAgentRuntime()
AGENT_TOOLS_REGISTERED = False
AGENT_TOOLS_LOCK = threading.Lock()
SHUTDOWN_CLEANUP_LOCK = threading.Lock()
SHUTDOWN_CLEANUP_DONE = False
TORCHPROFILER_SKILL_DIR = APP_DIR / "skills" / "TorchProfilerTraceSkill"

__all__ = [
    "AGENT_RUNTIME",
    "AGENT_TOOLS_LOCK",
    "AGENT_TOOLS_REGISTERED",
    "CONFIG_PATH",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_SERVER_HOST",
    "LOGS_DIR",
    "MAX_JSON_BODY_BYTES",
    "MAX_PROFILE_ARCHIVE_BYTES",
    "MAX_PROFILE_ARCHIVE_FILES",
    "PERFETTO_PORT",
    "RUN_LOCK",
    "RUN_START_LOCK",
    "RUN_STATE",
    "SERVER_AUTH_TOKEN",
    "SERVER_BIND_HOST",
    "SERVER_COOKIE_NAME",
    "SERVER_SESSION_ID",
    "SHUTDOWN_CLEANUP_DONE",
    "SHUTDOWN_CLEANUP_LOCK",
    "STATIC_DIR",
    "TEMP_CONFIG_DIR",
    "TEMP_CONFIG_MAX_AGE_SEC",
    "TORCHPROFILER_SKILL_DIR",
]
