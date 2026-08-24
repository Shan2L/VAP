from __future__ import annotations

import secrets
import threading
import uuid
from typing import Any

from vap.agent.runtime import VAPAgentRuntime
from vap.runtime_paths import (
    ASSET_DIR,
    VAP_ACTIVE_RUN_PATH,
    VAP_CONFIG_PATH,
    VAP_LOGS_DIR,
    VAP_TEMP_CONFIG_DIR,
)
from vap.validation import PERFETTO_PORT

STATIC_DIR = ASSET_DIR / "public"
DEFAULT_CONFIG_PATH = ASSET_DIR / "example-config.json"
CONFIG_PATH = VAP_CONFIG_PATH
ACTIVE_RUN_PATH = VAP_ACTIVE_RUN_PATH
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
MAX_STATUS_LOG_BYTES = 256 * 1024
MAX_RUN_OUTPUT_CHARS = 256 * 1024
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
TORCHPROFILER_SKILL_DIR = ASSET_DIR / "skills" / "TorchProfilerTraceSkill"
