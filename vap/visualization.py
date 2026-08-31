from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterable, Mapping

from vap.runtime_paths import APP_DIR, VAP_BIN_DIR, VAP_PERFETTO_HOME, VAP_VENV_DIR
from vap.validation import PERFETTO_PORT

logger = logging.getLogger("VAP")


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)
        try:
            sock.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def find_tensorboard_command() -> list[str] | None:
    env_tensorboard = os.getenv("VAP_TENSORBOARD")
    if env_tensorboard:
        return [env_tensorboard]

    for candidate in (
        VAP_VENV_DIR / "bin" / "tensorboard",
        APP_DIR / ".venv" / "bin" / "tensorboard",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return [str(candidate)]

    for candidate in (
        VAP_VENV_DIR / "bin" / "python",
        APP_DIR / ".venv" / "bin" / "python",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return [str(candidate), "-m", "tensorboard.main"]

    path_tensorboard = shutil.which("tensorboard")
    return [path_tensorboard] if path_tensorboard else None


def find_trace_processor() -> str | None:
    for candidate in (
        str(VAP_BIN_DIR / "trace_processor"),
        str(APP_DIR / "bin" / "trace_processor"),
        str(APP_DIR / "trace_processor"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def start_tensorboard(
    profile_dir: str,
    host: str,
    port: int,
) -> subprocess.Popen | None:
    base_command = find_tensorboard_command()
    if base_command is None:
        logger.warning(
            "TensorBoard is not available; checked VAP_TENSORBOARD, %s, %s, and PATH",
            VAP_VENV_DIR / "bin" / "tensorboard",
            APP_DIR / ".venv" / "bin" / "tensorboard",
        )
        return None
    if not is_port_available(port):
        logger.warning("TensorBoard port %s is unavailable; skip visualization", port)
        return None

    command = [
        *base_command,
        "--logdir",
        profile_dir,
        "--host",
        host,
        "--port",
        str(port),
        "--path_prefix",
        "/tensorboard",
    ]
    try:
        process = subprocess.Popen(command)
    except FileNotFoundError:
        logger.warning("TensorBoard command is not available; skip visualization")
        return None
    logger.info("TensorBoard started with pid %s on port %s", process.pid, port)
    warn_if_process_exited(process, "TensorBoard")
    return process


def start_perfetto(
    trace_path: str | None,
    host: str,
) -> subprocess.Popen | None:
    if trace_path is None:
        logger.warning("No Perfetto-compatible trace found")
        return None

    trace_processor = find_trace_processor()
    if trace_processor is None:
        logger.warning("trace_processor is not available; skip Perfetto visualization")
        return None
    if not is_port_available(PERFETTO_PORT):
        logger.warning(
            "Perfetto Trace Processor port %s is not available; "
            "skip Perfetto visualization",
            PERFETTO_PORT,
        )
        return None

    VAP_PERFETTO_HOME.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["HOME"] = str(VAP_PERFETTO_HOME)
    command = [
        trace_processor,
        "--httpd",
        "--http-ip-address",
        host,
        "--http-port",
        str(PERFETTO_PORT),
        trace_path,
    ]
    try:
        process = subprocess.Popen(command, env=environment)
    except FileNotFoundError:
        logger.warning(
            "%s is not available; skip Perfetto visualization",
            trace_processor,
        )
        return None
    logger.info(
        "Perfetto Trace Processor started with pid %s on port %s for %s",
        process.pid,
        PERFETTO_PORT,
        trace_path,
    )
    warn_if_process_exited(process, "Perfetto Trace Processor")
    return process


def write_visualization_pids(
    log_dir: str,
    processes: Mapping[str, subprocess.Popen | None],
) -> None:
    payload = {
        name: process.pid
        for name, process in processes.items()
        if process is not None and process.poll() is None
    }
    path = os.path.join(log_dir, "visualization_pids.json")
    with open(path, "w", encoding="utf-8") as pid_file:
        json.dump(payload, pid_file, indent=2)
        pid_file.write("\n")


def wait_for_visualizations(
    processes: Iterable[subprocess.Popen | None],
) -> None:
    active = [process for process in processes if process is not None]
    try:
        while any(process.poll() is None for process in active):
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping visualization services...")
        raise


def stop_visualizations(
    processes: Iterable[subprocess.Popen | None],
) -> None:
    for process in processes:
        terminate_process(process)


def warn_if_process_exited(
    process: subprocess.Popen,
    name: str,
    delay_sec: float = 0.8,
) -> None:
    time.sleep(delay_sec)
    exit_code = process.poll()
    if exit_code is not None:
        logger.warning("%s exited immediately with code %s", name, exit_code)


def terminate_process(
    process: subprocess.Popen | None,
    timeout_sec: float = 5.0,
) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
