import logging
import os
import shlex
import socket
from typing import List

from vap.config import VAPConfig
from vap.pipelines.pipeline import Pipeline
from vap.pipelines.services.clock_probe import (
    CLOCK_PROBE_SOURCE_DIR,
    ClockProbeLifecycle,
    clock_probe_enabled,
)
from vap.pipelines.services.ray_cluster import RayClusterLifecycle
from vap.pipelines.services.trace_postprocessor import (
    CONTAINER_PROFILE_DIR,
    TracePostprocessor,
)
from vap.runners import (
    ContainerSpec,
    DockerRunner,
    DockerTarget,
    MountSpec,
    ProcessHandle,
    UlimitSpec,
)
from vap.runtime_paths import APP_DIR
from vap.validation import PERFETTO_PORT
from vap.visualization import (
    is_port_available,
    start_perfetto,
    start_tensorboard,
    stop_visualizations,
    wait_for_visualizations,
    write_visualization_pids,
)

logger = logging.getLogger("VAP")

CONTAINER_MODEL_ROOT = "/tmp/vap/models"


def check_port_availability(config: VAPConfig):
    required_ports = {
        "vLLM": config.vllm_port,
        "TensorBoard": config.profiler_cfg.tensorboard_port,
    }
    distributed = config.distributed_cfg
    if distributed is not None and distributed.enable:
        required_ports["Ray"] = distributed.ray_port
    if clock_probe_enabled(config):
        required_ports["Clock probe"] = config.clock_probe_cfg.port
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


class TorchProfilingPipeline(Pipeline):
    def __init__(
        self,
        config: VAPConfig,
        log_path: str,
        date_str: str,
        visualization_host: str = "127.0.0.1",
    ):
        self.master_runner: DockerRunner
        self.worker_runners: List[DockerRunner] = []
        self._vllm_process: ProcessHandle | None = None
        self.config = config
        self.log_path = log_path
        self.date_str = date_str
        self.visualization_host = visualization_host
        self.create_runners()
        self.ray_cluster = RayClusterLifecycle(
            self.config,
            self.master_runner,
            self.worker_runners,
        )
        self.clock_probe = ClockProbeLifecycle(
            self.config,
            self.master_runner,
            self.log_path,
            self.date_str,
        )
        self.trace_postprocessor = TracePostprocessor(
            self.config,
            self.log_path,
            self.master_runner,
            self.worker_runners,
        )

    def create_runners(self) -> None:
        workers: List[str] = []
        distributed = self.config.distributed_cfg
        if distributed is not None and distributed.enable:
            workers = list(distributed.worker_nodes)

        master_host = socket.getfqdn()
        self.master_peer = workers[0] if workers else None
        self.worker_peer = master_host
        self.master_runner = DockerRunner()
        for node in workers:
            self.worker_runners.append(DockerRunner(DockerTarget(hostname=node)))

    def run_pipeline(self):
        check_port_availability(self.config)
        if not os.path.exists(self.config.model_path):
            logger.error("Model weight %s is not available", self.config.model_path)
            raise FileNotFoundError(
                f"Model weight {self.config.model_path} is not available"
            )
        logger.info("Model weight %s is available", self.config.model_path)

        distributed = bool(
            self.config.distributed_cfg and self.config.distributed_cfg.enable
        )

        try:
            # 1. Start the local runner container.
            self.master_runner.start(self._container_spec())

            # 2. Distributed runs additionally start worker containers and Ray.
            if distributed:
                for worker in self.worker_runners:
                    worker.start(self._container_spec(include_clock_source=False))
            for runner in [self.master_runner, *self.worker_runners]:
                runner.files.ensure_directory(CONTAINER_PROFILE_DIR)
            self._check_model_weights()
            if distributed:
                self.ray_cluster.start()

            # 3. Deploy vLLM and wait for its HTTP API.
            self._deploy_model()
            self._wait_for_vllm_ready()

            # 4. Clock probing is only part of enabled distributed runs.
            if distributed and self.clock_probe.enabled:
                self.clock_probe.start()
                profiling_error: BaseException | None = None
                try:
                    self._bench_and_profile()
                except BaseException as exc:
                    profiling_error = exc
                    raise
                finally:
                    try:
                        self.clock_probe.stop()
                    except Exception as exc:
                        if profiling_error is None:
                            raise
                        logger.error(
                            "Clock probe stop also failed after profiling " "error: %s",
                            exc,
                        )
            else:
                self._bench_and_profile()

            # 5. Wait for finalized traces and collect worker artifacts.
            self.trace_postprocessor.wait_for_trace_files()
            if distributed:
                self.trace_postprocessor.collect_worker_traces()
        except Exception as exc:
            logger.error("Error: %s", exc)
            raise
        finally:
            self.cleanup()

        # 6. Build rank inputs, optionally align, then fuse when needed.
        profile_dir = self.trace_postprocessor.profile_dir
        if distributed:
            trace_inputs = self.trace_postprocessor.distributed_trace_inputs(
                profile_dir,
                self.ray_cluster.runner_node_ids,
            )
            traces_to_fuse = {
                rank: str(trace.path) for rank, trace in trace_inputs.items()
            }
            aligned = False
            if self.clock_probe.enabled:
                try:
                    traces_to_fuse = self.trace_postprocessor.align_trace_files(
                        trace_inputs,
                        profile_dir,
                        self.clock_probe.local_session_path,
                    )
                    aligned = True
                except Exception as exc:
                    if self.clock_probe.required:
                        raise RuntimeError(f"Trace alignment failed: {exc}") from exc
                    logger.warning(
                        "Trace alignment failed; falling back to raw "
                        "distributed traces: %s",
                        exc,
                    )
            trace_path = self.trace_postprocessor.fuse_trace_files(
                traces_to_fuse,
                profile_dir,
                aligned=aligned,
            )
        else:
            trace_path = self.trace_postprocessor.prepare_single_node_trace(profile_dir)

        if trace_path is not None:
            logger.info("Selected Perfetto trace: %s", trace_path)
            logger.info(
                "Profile archive has been saved to: %s",
                profile_dir,
            )
        else:
            logger.warning(
                "Trace post-processing completed without a " "Perfetto-compatible trace"
            )

        # 7. Start visualization services with the selected final trace.
        visualization_processes = {
            "tensorboard": None,
            "perfetto": None,
        }
        try:
            visualization_processes["tensorboard"] = start_tensorboard(
                profile_dir,
                self.visualization_host,
                self.config.profiler_cfg.tensorboard_port,
            )
            visualization_processes["perfetto"] = start_perfetto(
                trace_path,
                self.visualization_host,
            )
            write_visualization_pids(self.log_path, visualization_processes)
            wait_for_visualizations(visualization_processes.values())
        finally:
            stop_visualizations(visualization_processes.values())
            write_visualization_pids(self.log_path, {})

    def cleanup(self) -> None:
        if self.clock_probe.active:
            self.clock_probe.cleanup()
        if self._vllm_process is not None and self.master_runner.is_started:
            try:
                self.master_runner.process.terminate(self._vllm_process)
            except Exception as exc:
                logger.warning("Failed to terminate vLLM process cleanly: %s", exc)
            self._vllm_process = None
        if self.ray_cluster.processes:
            self.ray_cluster.cleanup()
        for runner in reversed(self.worker_runners):
            runner.cleanup()
        self.master_runner.cleanup()

    def _container_spec(
        self,
        *,
        include_clock_source: bool = True,
    ) -> ContainerSpec:
        container_cfg = self.config.container_cfg
        mounts = [
            MountSpec(
                target=mount.target,
                source=mount.source,
                type=mount.type or "bind",
            )
            for mount in (container_cfg.mounts or [])
        ]
        mounts.extend(
            [
                MountSpec(
                    target=CONTAINER_MODEL_ROOT,
                    source=self.config.model_cfg.model_path,
                ),
                MountSpec(
                    target="/app/VAP/log",
                    source=self.log_path,
                    create_source=True,
                ),
            ]
        )
        clock_probe = self.config.clock_probe_cfg
        if self.clock_probe.enabled and include_clock_source:
            mounts.append(
                MountSpec(
                    target=CLOCK_PROBE_SOURCE_DIR,
                    source=str(APP_DIR),
                    read_only=True,
                )
            )
        os.makedirs(os.path.join(self.log_path, "vllm-profile"), exist_ok=True)
        devices = ["/dev/kfd", "/dev/mem"]
        devices.extend(container_cfg.devices or ["/dev/dri/"])
        if self.clock_probe.enabled and clock_probe.hardware_phc_device:
            devices.append(clock_probe.hardware_phc_device)
        safe_model_name = self.config.model_cfg.model_name.replace("/", "_")
        return ContainerSpec(
            image=self.config.docker_image,
            name=f"vap_{safe_model_name}_{self.date_str}",
            mounts=tuple(mounts),
            devices=tuple(dict.fromkeys(devices)),
            environment=dict(container_cfg.env_vars or {}),
            cap_add=("SYS_ADMIN", "SYS_PTRACE"),
            group_add=("video",),
            security_opt=("seccomp=unconfined",),
            ulimits=(
                UlimitSpec(name="memlock", soft=-1, hard=-1),
                UlimitSpec(name="nofile", soft=65535, hard=65535),
            ),
            shm_size="128G",
        )

    def _check_model_weights(self) -> None:
        model_path = os.path.join(
            CONTAINER_MODEL_ROOT,
            self.config.model_cfg.model_name,
        )
        missing = [
            runner.target.label
            for runner in [self.master_runner, *self.worker_runners]
            if not runner.files.exists(model_path)
        ]
        if missing:
            raise FileNotFoundError(
                f"Model weight {model_path} is unavailable on: {', '.join(missing)}"
            )
        logger.info(
            "Model weight %s is available on all %d nodes",
            model_path,
            1 + len(self.worker_runners),
        )

    def _deploy_model(self) -> None:
        container_model_path = os.path.join(
            CONTAINER_MODEL_ROOT,
            self.config.model_cfg.model_name,
        )
        command = [
            "vllm",
            "serve",
            container_model_path,
            *self.config.vllm_deploy_args(),
        ]
        distributed = self.config.distributed_cfg
        if (
            distributed is not None
            and distributed.enable
            and "--distributed-executor-backend" not in self.config.vllm_deploy_cfg
        ):
            command.extend(["--distributed-executor-backend", "ray"])
        shell_command = (
            f"exec {shlex.join(command)} > /app/VAP/log/vllm_deploy.log 2>&1"
        )
        environment = dict(self.config.container_cfg.env_vars or {})
        if not environment.get("VLLM_HOST_IP") and self.ray_cluster.master_ip:
            environment["VLLM_HOST_IP"] = self.ray_cluster.master_ip
        logger.debug("vLLM deploy command: %s", shell_command)
        self._vllm_process = self.master_runner.process.start_shell(
            shell_command,
            environment=environment,
        )

    def _wait_for_vllm_ready(
        self,
        timeout_sec: float = 1800,
        poll_interval_sec: float = 5,
    ) -> None:
        url = f"http://127.0.0.1:{self.config.vllm_port}/health"
        self.master_runner.network.wait_http(
            url,
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
        )

    def _bench_and_profile(self) -> None:
        port = self.config.vllm_port
        start_url = f"http://127.0.0.1:{port}/start_profile"
        stop_url = f"http://127.0.0.1:{port}/stop_profile"
        start_status = self.master_runner.network.http_status(
            start_url,
            method="POST",
            timeout_sec=10,
        )
        if start_status is None or not 200 <= start_status < 300:
            raise RuntimeError(f"Failed to start vLLM profiler: HTTP {start_status}")

        bench_command = shlex.join(
            ["vllm", "bench", "serve", *self.config.vllm_bench_args()]
        )
        shell_command = (
            "set -o pipefail; "
            f"{bench_command} 2>&1 | tee /app/VAP/log/vllm_bench.log"
        )
        logger.debug("Benchmark command: %s", shell_command)
        benchmark_error: BaseException | None = None
        try:
            result = self.master_runner.process.run_shell(
                shell_command,
                demux=True,
            )
            if result.exit_code != 0:
                logger.error(
                    "Benchmark failed (exit %s): %s",
                    result.exit_code,
                    result.combined_text,
                )
                raise RuntimeError(
                    f"vllm bench failed with exit code {result.exit_code}"
                )
            logger.info("Benchmark finished successfully")
        except BaseException as exc:
            benchmark_error = exc
            raise
        finally:
            stop_status = self.master_runner.network.http_status(
                stop_url,
                method="POST",
                timeout_sec=10,
            )
            if stop_status is None or not 200 <= stop_status < 300:
                message = f"Failed to stop vLLM profiler cleanly: HTTP {stop_status}"
                if benchmark_error is None:
                    raise RuntimeError(message)
                logger.error("%s; benchmark also failed", message)
