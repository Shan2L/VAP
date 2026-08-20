from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from vap.runtime_paths import APP_DIR
from vap.server import settings


def get_run_state_snapshot() -> dict[str, Any]:
    with settings.RUN_LOCK:
        process = settings.RUN_STATE["process"]
        run_dir = settings.RUN_STATE["run_dir"]
        return {
            "pid": settings.RUN_STATE["pid"],
            "running": settings.RUN_STATE["running"],
            "exit_code": settings.RUN_STATE["exit_code"],
            "started_at": settings.RUN_STATE["started_at"],
            "ended_at": settings.RUN_STATE["ended_at"],
            "run_dir": str(run_dir) if run_dir else None,
            "config_path": (
                str(settings.RUN_STATE["config_path"])
                if settings.RUN_STATE["config_path"]
                else None
            ),
            "output": settings.RUN_STATE["output"],
            "stop_requested": settings.RUN_STATE["stop_requested"],
            "has_process": process is not None,
        }


def discover_run_dir(existing_dirs: set[str], started_at: float) -> Path | None:
    if not settings.LOGS_DIR.is_dir():
        return None
    candidates = []
    for path in settings.LOGS_DIR.iterdir():
        if not path.is_dir() or path.name in existing_dirs:
            continue
        try:
            if path.stat().st_mtime >= started_at - 1:
                candidates.append(path)
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def monitor_run_process(
    process: subprocess.Popen[str], existing_dirs: set[str], started_at: float
) -> None:
    while True:
        with settings.RUN_LOCK:
            if settings.RUN_STATE["process"] is process and settings.RUN_STATE["run_dir"] is None:
                settings.RUN_STATE["run_dir"] = discover_run_dir(existing_dirs, started_at)
        line = process.stdout.readline() if process.stdout else ""
        if line:
            with settings.RUN_LOCK:
                if settings.RUN_STATE["process"] is process:
                    settings.RUN_STATE["output"] += line
        elif process.poll() is not None:
            break
        else:
            time.sleep(0.2)

    remaining = process.stdout.read() if process.stdout else ""
    exit_code = process.wait()
    with settings.RUN_LOCK:
        if settings.RUN_STATE["process"] is not process:
            return
        if remaining:
            settings.RUN_STATE["output"] += remaining
        if settings.RUN_STATE["run_dir"] is None:
            settings.RUN_STATE["run_dir"] = discover_run_dir(existing_dirs, started_at)
        settings.RUN_STATE["running"] = False
        settings.RUN_STATE["exit_code"] = exit_code
        settings.RUN_STATE["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def current_run_is_tensorboard_phase() -> bool:
    snapshot = get_run_state_snapshot()
    tensorboard_marker = "TensorBoard started with pid"
    if tensorboard_marker in (snapshot.get("output") or ""):
        return True

    run_dir = Path(snapshot["run_dir"]) if snapshot.get("run_dir") else None
    if run_dir is None:
        return False
    log_path = (run_dir / "vap_log.txt").resolve()
    if not log_path.is_file() or not log_path.is_relative_to(settings.LOGS_DIR.resolve()):
        return False
    return tensorboard_marker in log_path.read_text(encoding="utf-8", errors="replace")


def terminate_run_process(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()


def stop_process_group_sync(
    process: subprocess.Popen[str], timeout_sec: float = 5.0
) -> bool:
    terminate_run_process(process)
    try:
        process.wait(timeout=timeout_sec)
        return True
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError:
        process.kill()
    try:
        process.wait(timeout=timeout_sec)
        return True
    except subprocess.TimeoutExpired:
        return False


def force_kill_process_group_later(
    process: subprocess.Popen[str], timeout_sec: float = 5.0
) -> None:
    def worker() -> None:
        try:
            process.wait(timeout=timeout_sec)
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            process.kill()

    threading.Thread(target=worker, daemon=True).start()


def start_vap_run(config_path: Path | None = None) -> dict[str, Any]:
    if not settings.RUN_START_LOCK.acquire(blocking=False):
        raise RuntimeError("VAP is already starting")
    try:
        return _start_vap_run(config_path)
    finally:
        settings.RUN_START_LOCK.release()


def _start_vap_run(config_path: Path | None = None) -> dict[str, Any]:
    with settings.RUN_LOCK:
        process = settings.RUN_STATE["process"]
        is_running = (
            settings.RUN_STATE["running"]
            and process is not None
            and process.poll() is None
        )
    if is_running:
        if not current_run_is_tensorboard_phase():
            raise RuntimeError("VAP is already running")
        with settings.RUN_LOCK:
            settings.RUN_STATE[
                "output"
            ] += "\n--- Previous TensorBoard is still running; stopping it before new run ---\n"
        if not stop_process_group_sync(process):
            raise RuntimeError(
                "The previous TensorBoard process did not stop in time. Try again later."
            )

    from vap.server.artifacts import resolve_config_path

    run_config_path = (config_path or resolve_config_path(None)).resolve()
    settings.LOGS_DIR.mkdir(exist_ok=True)
    existing_dirs = {path.name for path in settings.LOGS_DIR.iterdir() if path.is_dir()}
    started_at = time.time()
    with settings.RUN_LOCK:
        process = settings.RUN_STATE["process"]
        if settings.RUN_STATE["running"]:
            if process is None or process.poll() is None:
                raise RuntimeError("VAP is already starting or running")
            settings.RUN_STATE["running"] = False
        settings.RUN_STATE.update(
            {
                "process": None,
                "pid": None,
                "running": True,
                "exit_code": None,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ended_at": None,
                "run_dir": None,
                "config_path": run_config_path,
                "output": "--- VAP is starting ---\n",
                "stop_requested": False,
            }
        )
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vap.main",
                "run",
                "--config",
                str(run_config_path),
                "--visualization-host",
                settings.SERVER_BIND_HOST,
            ],
            cwd=str(APP_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except Exception:
        with settings.RUN_LOCK:
            settings.RUN_STATE.update(
                {
                    "process": None,
                    "pid": None,
                    "running": False,
                    "exit_code": None,
                    "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "output": settings.RUN_STATE["output"]
                    + "--- VAP failed to start ---\n",
                }
            )
        raise
    with settings.RUN_LOCK:
        settings.RUN_STATE.update(
            {
                "process": process,
                "pid": process.pid,
                "output": settings.RUN_STATE["output"]
                + f"--- VAP started (pid {process.pid}) ---\n",
            }
        )
        stop_requested = settings.RUN_STATE["stop_requested"]
    thread = threading.Thread(
        target=monitor_run_process,
        args=(process, existing_dirs, started_at),
        daemon=True,
    )
    thread.start()
    if stop_requested:
        terminate_run_process(process)
        force_kill_process_group_later(process)
    return get_run_state_snapshot()


def process_cmdline(pid: int) -> str:
    try:
        return (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
        )
    except OSError:
        return ""


def terminate_recorded_visualization_pids(
    run_dir: Path | None, timeout_sec: float = 3.0
) -> None:
    if run_dir is None:
        return
    pid_file = (run_dir / "visualization_pids.json").resolve()
    if not pid_file.is_file() or not pid_file.is_relative_to(settings.LOGS_DIR.resolve()):
        return
    try:
        payload = json.loads(pid_file.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Failed to read visualization pid file {pid_file}: {exc}")
        return
    if not isinstance(payload, dict):
        return

    for name, raw_pid in payload.items():
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            continue
        cmdline = process_cmdline(pid)
        if not cmdline:
            continue
        if "tensorboard" not in cmdline and "trace_processor" not in cmdline:
            print(
                f"Skip cleanup for pid {pid}; command is not a VAP visualization process"
            )
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError as exc:
            print(f"Failed to terminate {name} pid {pid}: {exc}")
            continue

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                print(f"Failed to force kill {name} pid {pid}: {exc}")

    try:
        pid_file.write_text("{}\n", encoding="utf-8")
    except OSError:
        pass


def cleanup_active_run_on_server_exit(timeout_sec: float = 8.0) -> None:
    with settings.SHUTDOWN_CLEANUP_LOCK:
        if settings.SHUTDOWN_CLEANUP_DONE:
            return
        settings.SHUTDOWN_CLEANUP_DONE = True

    with settings.RUN_LOCK:
        process = settings.RUN_STATE["process"]
        run_dir = (
            Path(settings.RUN_STATE["run_dir"]).resolve()
            if settings.RUN_STATE["run_dir"]
            else None
        )
        is_running = (
            settings.RUN_STATE["running"]
            and process is not None
            and process.poll() is None
        )
        is_starting = settings.RUN_STATE["running"] and process is None
        if is_running or is_starting:
            settings.RUN_STATE["stop_requested"] = True
            settings.RUN_STATE[
                "output"
            ] += "\n--- Server is exiting; stopping active VAP run ---\n"

    if is_starting:
        return
    if not is_running or process is None:
        terminate_recorded_visualization_pids(run_dir)
        return

    print("Stopping active VAP run before server exits...")
    if not stop_process_group_sync(process, timeout_sec=timeout_sec):
        print("Active VAP run did not stop cleanly before server exit.")
    terminate_recorded_visualization_pids(run_dir)


def stop_vap_run() -> dict[str, Any]:
    with settings.RUN_LOCK:
        process = settings.RUN_STATE["process"]
        run_dir = (
            Path(settings.RUN_STATE["run_dir"]).resolve()
            if settings.RUN_STATE["run_dir"]
            else None
        )
        is_starting = settings.RUN_STATE["running"] and process is None
        if is_starting:
            settings.RUN_STATE["stop_requested"] = True
            settings.RUN_STATE["output"] += "\n--- Stop requested while VAP is starting ---\n"
    if is_starting:
        return {
            "message": "Stop requested. VAP will be terminated as soon as it starts.",
            **get_run_state_snapshot(),
        }
    if process is None or process.poll() is not None:
        terminate_recorded_visualization_pids(run_dir)
        return {"message": "There is no active VAP run", **get_run_state_snapshot()}
    terminate_run_process(process)
    force_kill_process_group_later(process)
    terminate_recorded_visualization_pids(run_dir)
    with settings.RUN_LOCK:
        settings.RUN_STATE["stop_requested"] = True
        settings.RUN_STATE["output"] += "\n--- Stop requested from UI ---\n"
    return {
        "message": "Stop signal sent. The VAP workflow and its child processes will be stopped.",
        **get_run_state_snapshot(),
    }
