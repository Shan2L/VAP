from __future__ import annotations

import io
import json
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from vap.runtime_paths import ensure_vap_home, resolve_under_vap_home
from vap.server import settings
from vap.server.state import get_run_state_snapshot


def cleanup_old_temp_configs(now: float | None = None) -> None:
    if not settings.TEMP_CONFIG_DIR.is_dir():
        return
    cutoff = (now or time.time()) - settings.TEMP_CONFIG_MAX_AGE_SEC
    for path in settings.TEMP_CONFIG_DIR.glob("vap-config-*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def save_temp_config(payload: dict[str, Any]) -> Path:
    ensure_vap_home()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    settings.TEMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    settings.TEMP_CONFIG_DIR.chmod(0o700)
    cleanup_old_temp_configs()
    temp_path = (
        settings.TEMP_CONFIG_DIR / f"vap-config-{timestamp}-{uuid.uuid4().hex[:8]}.json"
    )
    with temp_path.open("w", encoding="utf-8") as config_file:
        json.dump(payload, config_file, indent=4, ensure_ascii=False)
        config_file.write("\n")
    temp_path.chmod(0o600)
    return temp_path


def resolve_config_path(raw_path: str | None) -> Path:
    if not raw_path:
        return (
            settings.CONFIG_PATH
            if settings.CONFIG_PATH.is_file()
            else settings.DEFAULT_CONFIG_PATH
        )
    return resolve_under_vap_home(raw_path)


def current_config_payload() -> dict[str, Any]:
    config_path = resolve_config_path(None)
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def latest_log_run_dir() -> Path | None:
    if not settings.LOGS_DIR.is_dir():
        return None
    candidates = [path for path in settings.LOGS_DIR.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def read_current_log_file(file_name: str) -> dict[str, Any]:
    allowed_names = {"vap_log.txt", "vllm_deploy.log", "vllm_bench.log"}
    if file_name not in allowed_names:
        raise ValueError("Unsupported log file")

    snapshot = get_run_state_snapshot()
    run_dir = Path(snapshot["run_dir"]) if snapshot["run_dir"] else None
    if run_dir is None:
        return {
            "exists": False,
            "name": file_name,
            "path": None,
            "run_dir": None,
            "content": "",
            "message": "There is no active run or this run directory has not been created yet",
        }

    log_path = (run_dir / file_name).resolve()
    if log_path.is_file() and log_path.is_relative_to(settings.LOGS_DIR.resolve()):
        return {
            "exists": True,
            "name": file_name,
            "path": str(log_path),
            "run_dir": str(run_dir),
            "content": log_path.read_text(encoding="utf-8", errors="replace"),
        }

    return {
        "exists": False,
        "name": file_name,
        "path": str(log_path),
        "run_dir": str(run_dir),
        "content": "",
        "message": f"This run has not generated {file_name} yet",
    }


def build_log_download(file_name: str) -> tuple[str, bytes]:
    log_info = read_current_log_file(file_name)
    if not log_info.get("exists"):
        raise ValueError(str(log_info.get("message") or f"{file_name} does not exist"))
    return file_name, str(log_info["content"]).encode("utf-8")


def resolve_profile_archive_run_dir(raw_run_dir: str | None = None) -> Path:
    if raw_run_dir:
        run_dir = Path(raw_run_dir).resolve()
        if not run_dir.is_dir() or not run_dir.is_relative_to(
            settings.LOGS_DIR.resolve()
        ):
            raise ValueError("Invalid run directory for profile archive")
        return run_dir

    snapshot = get_run_state_snapshot()
    run_dir = Path(snapshot["run_dir"]).resolve() if snapshot["run_dir"] else None
    if run_dir is None:
        raise ValueError("There is no active or completed run directory yet")
    if not run_dir.is_dir() or not run_dir.is_relative_to(settings.LOGS_DIR.resolve()):
        raise ValueError("Invalid current run directory for profile archive")
    return run_dir


def build_current_profile_archive(raw_run_dir: str | None = None) -> tuple[str, bytes]:
    run_dir = resolve_profile_archive_run_dir(raw_run_dir)
    profile_dir = (run_dir / "vllm-profile").resolve()
    if not profile_dir.is_dir() or not profile_dir.is_relative_to(
        settings.LOGS_DIR.resolve()
    ):
        raise ValueError("This run has not generated a vllm-profile directory yet")

    files: list[Path] = []
    total_size = 0
    for path in profile_dir.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(profile_dir):
            continue
        size = resolved.stat().st_size
        total_size += size
        if len(files) >= settings.MAX_PROFILE_ARCHIVE_FILES:
            raise ValueError("The vllm-profile directory has too many files to archive")
        if total_size > settings.MAX_PROFILE_ARCHIVE_BYTES:
            raise ValueError("The vllm-profile archive would exceed the size limit")
        files.append(resolved)
    if not files:
        raise ValueError("The vllm-profile directory is empty")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(run_dir))

    archive_name = f"{run_dir.name}-vllm-profile.zip"
    return archive_name, buffer.getvalue()
