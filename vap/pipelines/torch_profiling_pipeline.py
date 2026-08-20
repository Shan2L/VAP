import socket
import logging
import os
import time
import json
import subprocess
import shutil
import glob

from vap.pipelines.pipeline import Pipeline
from vap.config import VAPConfig
from vap.pipelines.containerRunner import ContainerRunner
from vap.validation import PERFETTO_PORT
from vap.runtime_paths import APP_DIR, VAP_BIN_DIR, VAP_PERFETTO_HOME, VAP_VENV_DIR
from vap.trace_fusion import fuse_traces


logger = logging.getLogger("VAP")

def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False

def check_port_availability(config: VAPConfig):
    required_ports = {
        "vLLM": config.vllm_port,
        "TensorBoard": config.profiler_cfg.tensorboard_port,
    }
    for name, port in required_ports.items():
        if is_port_available(port):
            logger.info("%s port %s is available", name, port)
            continue
        logger.error("%s port %s is not available", name, port)
        raise RuntimeError(f"{name} port {port} is not available")

    if is_port_available(PERFETTO_PORT):
        logger.info("Perfetto Trace Processor port %s is available", PERFETTO_PORT)
    else:
        logger.warning(
            "Perfetto Trace Processor port %s is not available; "
            "profiling will continue and Perfetto visualization will be skipped",
            PERFETTO_PORT,
        )

def merged_trace_output_file(profile_dir: str, config: VAPConfig) -> str:
    run_stamp = os.path.basename(os.path.dirname(profile_dir.rstrip(os.sep)))
    model_name = safe_filename_part(config.model_cfg.model_name.replace("/", "_"))
    prefix = "-".join(part for part in (run_stamp, model_name) if part)
    return os.path.join(profile_dir, f"{prefix}-merged_trace.json")

def safe_filename_part(value: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value
    ).strip("_")

def write_visualization_pids(
    log_dir: str, processes: dict[str, subprocess.Popen | None]
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

def find_tensorboard_command(app_dir: str) -> list[str] | None:
    env_tensorboard = os.getenv("VAP_TENSORBOARD")
    if env_tensorboard:
        return [env_tensorboard]

    candidate_bins = [
        VAP_VENV_DIR / "bin" / "tensorboard",
        APP_DIR / ".venv" / "bin" / "tensorboard",
    ]
    for candidate in candidate_bins:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return [str(candidate)]

    candidate_pythons = [
        VAP_VENV_DIR / "bin" / "python",
        APP_DIR / ".venv" / "bin" / "python",
    ]
    for candidate in candidate_pythons:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return [str(candidate), "-m", "tensorboard.main"]

    path_tensorboard = shutil.which("tensorboard")
    if path_tensorboard:
        return [path_tensorboard]

    return None

def collect_pytorch_trace_files(profile_dir: str) -> list[str]:
    patterns = ("*.pt.trace.json.gz", "*.trace.json.gz", "*.trace.json")
    traces: list[str] = []
    for pattern in patterns:
        traces.extend(glob.glob(os.path.join(profile_dir, pattern)))
    traces = [
        trace for trace in traces if "merged_trace" not in os.path.basename(trace)
    ]
    return sorted(dict.fromkeys(traces))

def warn_if_process_exited(
    process: subprocess.Popen, name: str, delay_sec: float = 0.8
):
    time.sleep(delay_sec)
    exit_code = process.poll()
    if exit_code is not None:
        logger.warning("%s exited immediately with code %s", name, exit_code)

def merge_pytorch_traces_for_perfetto(
    profile_dir: str, config: VAPConfig
) -> str | None:
    trace_files = collect_pytorch_trace_files(profile_dir)
    if not trace_files:
        return None
    if len(trace_files) == 1:
        logger.info(
            "Only one PyTorch trace found; Perfetto will load %s", trace_files[0]
        )
        return trace_files[0]

    output_file = merged_trace_output_file(profile_dir, config)
    try:
        logger.info("Merging %d PyTorch traces", len(trace_files))
        merged_trace = fuse_traces(trace_files, output_file)
        logger.info("Merged Perfetto trace has been saved to: %s", merged_trace)
        return merged_trace
    except Exception as exc:
        logger.warning("Trace fusion failed: %s", exc)
        logger.warning("Perfetto will fall back to the first trace: %s", trace_files[0])
        return trace_files[0]

def find_perfetto_trace(profile_dir: str, config: VAPConfig) -> str | None:
    merged_or_single_trace = merge_pytorch_traces_for_perfetto(profile_dir, config)
    if merged_or_single_trace:
        return merged_or_single_trace

    pftrace_files = sorted(glob.glob(os.path.join(profile_dir, "*.pftrace")))
    if pftrace_files:
        return pftrace_files[0]
    return None

def find_trace_processor(app_dir: str) -> str | None:
    candidates = [
        str(VAP_BIN_DIR / "trace_processor"),
        os.path.join(app_dir, "bin", "trace_processor"),
        os.path.join(app_dir, "trace_processor"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None

def terminate_process(process: subprocess.Popen | None, timeout_sec: float = 5.0):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()

def visualize_profile(config: VAPConfig, log_dir: str, visualization_host: str):
    app_dir = str(APP_DIR)
    profile_dir = os.path.join(log_dir, "vllm-profile")
    tensorboard_base_cmd = find_tensorboard_command(app_dir)
    tensorboard_cmd = (
        [
            *tensorboard_base_cmd,
            "--logdir",
            profile_dir,
            "--host",
            visualization_host,
            "--port",
            str(config.profiler_cfg.tensorboard_port),
            "--path_prefix",
            "/tensorboard",
        ]
        if tensorboard_base_cmd
        else None
    )

    tensorboard_process = None
    perfetto_process = None
    try:
        if tensorboard_cmd is None:
            logger.warning(
                "TensorBoard is not available; checked VAP_TENSORBOARD, %s, %s, and PATH",
                VAP_VENV_DIR / "bin" / "tensorboard",
                APP_DIR / ".venv" / "bin" / "tensorboard",
            )
        else:
            if not is_port_available(config.profiler_cfg.tensorboard_port):
                raise RuntimeError(
                    f"TensorBoard port {config.profiler_cfg.tensorboard_port} is not available"
                )
            tensorboard_process = subprocess.Popen(tensorboard_cmd)
    except FileNotFoundError:
        logger.warning("TensorBoard command is not available; skip visualization")
    else:
        if tensorboard_process is not None:
            logger.info(
                "TensorBoard started with pid %s on port %s",
                tensorboard_process.pid,
                config.profiler_cfg.tensorboard_port,
            )
            warn_if_process_exited(tensorboard_process, "TensorBoard")

    trace_path = find_perfetto_trace(profile_dir, config)
    trace_processor = find_trace_processor(app_dir)
    if trace_path is None:
        logger.warning("No Perfetto-compatible trace found under %s", profile_dir)
    elif trace_processor is None:
        logger.warning("trace_processor is not available; skip Perfetto visualization")
    elif not is_port_available(PERFETTO_PORT):
        logger.warning(
            "Perfetto Trace Processor port %s is not available; "
            "skip Perfetto visualization",
            PERFETTO_PORT,
        )
    else:
        perfetto_home = str(VAP_PERFETTO_HOME)
        os.makedirs(perfetto_home, exist_ok=True)
        perfetto_env = os.environ.copy()
        perfetto_env["HOME"] = perfetto_home
        perfetto_cmd = [
            trace_processor,
            "--httpd",
            "--http-ip-address",
            visualization_host,
            "--http-port",
            str(PERFETTO_PORT),
            trace_path,
        ]
        try:
            perfetto_process = subprocess.Popen(perfetto_cmd, env=perfetto_env)
        except FileNotFoundError:
            logger.warning(
                "%s is not available; skip Perfetto visualization", trace_processor
            )
        else:
            logger.info(
                "Perfetto Trace Processor started with pid %s on port %s for %s",
                perfetto_process.pid,
                PERFETTO_PORT,
                trace_path,
            )
            warn_if_process_exited(perfetto_process, "Perfetto Trace Processor")

    processes = [
        process for process in (tensorboard_process, perfetto_process) if process
    ]
    write_visualization_pids(
        log_dir,
        {"tensorboard": tensorboard_process, "perfetto": perfetto_process},
    )
    if not processes:
        return

    try:
        while any(process.poll() is None for process in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping visualization services...")
        raise
    finally:
        for process in processes:
            terminate_process(process)
        write_visualization_pids(log_dir, {})

class TorchProfilingPipeline(Pipeline):
    def __init__(
        self,
        config: VAPConfig,
        log_path: str,
        date_str: str,
        visualization_host: str = "127.0.0.1",
    ):
        self.runner = ContainerRunner(config, log_path, date_str)
        self.config = config
        self.log_path = log_path
        self.date_str = date_str
        self.visualization_host = visualization_host

    def run_pipeline(self):
        check_port_availability(self.config)
        if not os.path.exists(self.config.model_path):
            logger.error("Model weight %s is not available", self.config.model_path)
            raise FileNotFoundError(
                f"Model weight {self.config.model_path} is not available"
            )
        logger.info("Model weight %s is available", self.config.model_path)

        try:
            self.runner.deploy_model()
            self.runner.wait_for_vllm_ready()
            self.runner.bench_and_profile()
        except Exception as exc:
            logger.error("Error: %s", exc)
            raise
        finally:
            self.runner.remove_container()

        logger.info(
            "Profile archive has been saved to: %s",
            os.path.join(self.log_path, "vllm-profile"),
        )
        visualize_profile(self.config, self.log_path, self.visualization_host)

  