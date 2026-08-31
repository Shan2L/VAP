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

# Docker stop defaults to 10s per container; a distributed run may have
# master + workers + ptp4l sidecars. SIGKILL before that skips container cleanup.
STOP_CLEANUP_GRACE_SEC = 60.0


def process_start_ticks(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields_after_command = stat.rpartition(") ")[2].split()
        return int(fields_after_command[19])
    except (OSError, IndexError, ValueError):
        return None


def persist_active_run_locked() -> None:
    process = settings.RUN_STATE["process"]
    if process is None or process.poll() is not None:
        settings.ACTIVE_RUN_PATH.unlink(missing_ok=True)
        return
    payload = {
        "pid": process.pid,
        "pgid": process.pid,
        "start_ticks": process_start_ticks(process.pid),
        "run_dir": (
            str(settings.RUN_STATE["run_dir"])
            if settings.RUN_STATE["run_dir"]
            else None
        ),
        "config_path": (
            str(settings.RUN_STATE["config_path"])
            if settings.RUN_STATE["config_path"]
            else None
        ),
        "started_at": settings.RUN_STATE["started_at"],
    }
    temporary_path = settings.ACTIVE_RUN_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.chmod(0o600)
    temporary_path.replace(settings.ACTIVE_RUN_PATH)


def clear_active_run_record() -> None:
    try:
        settings.ACTIVE_RUN_PATH.unlink(missing_ok=True)
    except OSError as exc:
        print(f"Failed to remove active-run record: {exc}")


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
            if (
                settings.RUN_STATE["process"] is process
                and settings.RUN_STATE["run_dir"] is None
            ):
                discovered = discover_run_dir(existing_dirs, started_at)
                if discovered is not None:
                    settings.RUN_STATE["run_dir"] = discovered
                    try:
                        persist_active_run_locked()
                    except OSError as exc:
                        settings.RUN_STATE[
                            "output"
                        ] += f"\n--- Failed to update active-run record: {exc} ---\n"
        line = process.stdout.readline() if process.stdout else ""
        if line:
            with settings.RUN_LOCK:
                if settings.RUN_STATE["process"] is process:
                    settings.RUN_STATE["output"] = (
                        settings.RUN_STATE["output"] + line
                    )[-settings.MAX_RUN_OUTPUT_CHARS :]
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
        clear_active_run_record()


def current_run_is_tensorboard_phase() -> bool:
    snapshot = get_run_state_snapshot()
    tensorboard_marker = "TensorBoard started with pid"
    if tensorboard_marker in (snapshot.get("output") or ""):
        return True

    run_dir = Path(snapshot["run_dir"]) if snapshot.get("run_dir") else None
    if run_dir is None:
        return False
    log_path = (run_dir / "vap_log.txt").resolve()
    if not log_path.is_file() or not log_path.is_relative_to(
        settings.LOGS_DIR.resolve()
    ):
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
        clear_active_run_record()
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
    try:
        with settings.RUN_LOCK:
            settings.RUN_STATE.update(
                {
                    "process": process,
                    "pid": process.pid,
                    "output": settings.RUN_STATE["output"]
                    + f"--- VAP started (pid {process.pid}) ---\n",
                }
            )
            persist_active_run_locked()
            stop_requested = settings.RUN_STATE["stop_requested"]
    except Exception:
        stop_process_group_sync(process)
        clear_active_run_record()
        with settings.RUN_LOCK:
            settings.RUN_STATE.update(
                {
                    "process": None,
                    "pid": None,
                    "running": False,
                    "exit_code": process.returncode,
                    "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "output": settings.RUN_STATE["output"]
                    + "--- VAP could not persist active-run state and was stopped ---\n",
                }
            )
        raise
    thread = threading.Thread(
        target=monitor_run_process,
        args=(process, existing_dirs, started_at),
        daemon=True,
    )
    thread.start()
    if stop_requested:
        terminate_run_process(process)
        force_kill_process_group_later(process, timeout_sec=STOP_CLEANUP_GRACE_SEC)
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


def recover_orphaned_run(timeout_sec: float = 5.0) -> bool:
    """Stop a VAP process left behind by an unclean control-server exit."""
    if not settings.ACTIVE_RUN_PATH.is_file():
        return False
    try:
        payload = json.loads(settings.ACTIVE_RUN_PATH.read_text(encoding="utf-8"))
        pid = int(payload["pid"])
        pgid = int(payload["pgid"])
        recorded_start_ticks = int(payload["start_ticks"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Discarding invalid active-run record: {exc}")
        clear_active_run_record()
        return False

    cmdline = process_cmdline(pid)
    if (
        process_start_ticks(pid) != recorded_start_ticks
        or pgid != pid
        or "-m vap.main run" not in cmdline
    ):
        clear_active_run_record()
        return False

    print(f"Recovering orphaned VAP run (pid {pid}); stopping its process group...")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError as exc:
        print(f"Failed to stop orphaned VAP process group {pgid}: {exc}")
    else:
        deadline = time.time() + timeout_sec
        while (
            time.time() < deadline and process_start_ticks(pid) == recorded_start_ticks
        ):
            time.sleep(0.1)
        if process_start_ticks(pid) == recorded_start_ticks:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                print(f"Failed to force kill orphaned VAP process group {pgid}: {exc}")

    raw_run_dir = payload.get("run_dir")
    run_dir = Path(raw_run_dir).resolve() if isinstance(raw_run_dir, str) else None
    terminate_recorded_visualization_pids(run_dir)
    clear_active_run_record()
    return True


def terminate_recorded_visualization_pids(
    run_dir: Path | None, timeout_sec: float = 3.0
) -> None:
    if run_dir is None:
        return
    pid_file = (run_dir / "visualization_pids.json").resolve()
    if not pid_file.is_file() or not pid_file.is_relative_to(
        settings.LOGS_DIR.resolve()
    ):
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
        clear_active_run_record()
        return

    print("Stopping active VAP run before server exits...")
    if not stop_process_group_sync(process, timeout_sec=timeout_sec):
        print("Active VAP run did not stop cleanly before server exit.")
    terminate_recorded_visualization_pids(run_dir)
    clear_active_run_record()


def stop_vap_run() -> dict[str, Any]:
    send_signal = False
    already_stopping = False
    is_starting = False
    process: subprocess.Popen[str] | None = None
    run_dir: Path | None = None
    with settings.RUN_LOCK:
        process = settings.RUN_STATE["process"]
        run_dir = (
            Path(settings.RUN_STATE["run_dir"]).resolve()
            if settings.RUN_STATE["run_dir"]
            else None
        )
        already_stopping = bool(settings.RUN_STATE["stop_requested"])
        is_starting = settings.RUN_STATE["running"] and process is None
        if is_starting:
            settings.RUN_STATE["stop_requested"] = True
            if not already_stopping:
                settings.RUN_STATE[
                    "output"
                ] += "\n--- Stop requested while VAP is starting ---\n"
        elif process is not None and process.poll() is None and not already_stopping:
            settings.RUN_STATE["stop_requested"] = True
            settings.RUN_STATE["output"] += "\n--- Stop requested from UI ---\n"
            send_signal = True
    if is_starting:
        return {
            "message": "Stop requested. VAP will be terminated as soon as it starts.",
            **get_run_state_snapshot(),
        }
    if send_signal and process is not None:
        terminate_run_process(process)
        force_kill_process_group_later(process, timeout_sec=STOP_CLEANUP_GRACE_SEC)
        terminate_recorded_visualization_pids(run_dir)
        return {
            "message": (
                "Stop signal sent. The VAP workflow and its child processes "
                "will be stopped."
            ),
            **get_run_state_snapshot(),
        }
    if already_stopping and process is not None and process.poll() is None:
        return {
            "message": "Stop already in progress. Waiting for container cleanup to finish.",
            **get_run_state_snapshot(),
        }
    terminate_recorded_visualization_pids(run_dir)
    return {"message": "There is no active VAP run", **get_run_state_snapshot()}
