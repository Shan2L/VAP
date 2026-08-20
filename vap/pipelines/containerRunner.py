from __future__ import annotations

import logging
import os
import shlex
import time
from typing import Any

import docker
from docker.types import Mount, Ulimit

from vap.config import VAPConfig

logger = logging.getLogger("VAP")


def get_docker_client() -> Any:
    return docker.from_env()


class ContainerRunner:
    def __init__(self, config: VAPConfig, log_path: str, date_str: str):
        self.config = config
        self._docker_client: Any = None
        self.log_path = log_path
        self.date_str = date_str
        self.container: Any = None

    @property
    def docker_client(self) -> Any:
        if self._docker_client is None:
            self._docker_client = get_docker_client()
        return self._docker_client

    def build_container_mounts(self, log_path: str) -> list[Mount]:
        mounts: list[Mount] = []
        container_cfg = self.config.container_cfg
        if container_cfg.mounts:
            for mount in container_cfg.mounts:
                mounts.append(
                    Mount(
                        target=mount.target,
                        source=mount.source,
                        type=mount.type or "bind",
                    )
                )
        mounts.extend(
            [
                Mount(
                    target="/tmp/vap/models",
                    source=self.config.model_cfg.model_path,
                    type="bind",
                ),
                Mount(target="/app/VAP/log", source=log_path, type="bind"),
                Mount(
                    target="/app/VAP/log/vllm-profile",
                    source=os.path.join(log_path, "vllm-profile"),
                    type="bind",
                ),
            ]
        )
        return mounts

    def deploy_model(self) -> None:
        if self.container is None:
            self.container = self.build_container()

        container_model_path = os.path.join(
            "/tmp/vap/models", self.config.model_cfg.model_name
        )
        vllm_serve_argv = [
            "vllm",
            "serve",
            container_model_path,
            *self.config.vllm_deploy_args(),
        ]
        vllm_serve_cmd = (
            f"{shlex.join(vllm_serve_argv)} > /app/VAP/log/vllm_deploy.log 2>&1"
        )
        logger.debug("VLLM deploy command: %s", vllm_serve_cmd)
        self.container.exec_run(["/bin/bash", "-c", vllm_serve_cmd], detach=True)

    def wait_for_vllm_ready(
        self,
        timeout_sec: float = 1800,
        poll_interval_sec: float = 5,
    ) -> None:
        """Wait until vLLM /health returns 200 from inside the runner container."""
        if self.container is None:
            raise RuntimeError("container is required to wait for vLLM")
        vllm_port = self.config.vllm_port
        url = f"http://127.0.0.1:{vllm_port}/health"
        health_cmd = (
            "curl -sS -o /dev/null -w '%{http_code}' "
            f"--connect-timeout 2 --max-time 5 {shlex.quote(url)}"
        )
        deadline = time.monotonic() + timeout_sec
        attempt = 0

        while time.monotonic() < deadline:
            attempt += 1
            exit_code, output = self.container.exec_run(
                ["/bin/bash", "-c", health_cmd]
            )
            status = (output or b"").decode(errors="replace").strip()
            if exit_code == 0 and status == "200":
                logger.info("vLLM ready at %s (attempt %d)", url, attempt)
                return
            if status in {"", "000"} or exit_code != 0:
                logger.info(
                    "Waiting for vLLM: port %d not accepting connections yet (attempt %d)",
                    vllm_port,
                    attempt,
                )
            else:
                logger.info(
                    "Waiting for vLLM: %s returned HTTP %s (attempt %d)",
                    url,
                    status,
                    attempt,
                )
            time.sleep(poll_interval_sec)

        raise TimeoutError(
            f"vLLM did not become ready at {url} within {timeout_sec:.0f}s"
        )

    def bench_and_profile(self) -> None:
        if self.container is None:
            raise RuntimeError("container is required to run vllm bench inside docker")

        port = self.config.vllm_port
        start_url = f"http://127.0.0.1:{port}/start_profile"
        stop_url = f"http://127.0.0.1:{port}/stop_profile"
        start_cmd = f"curl -sS -X POST --max-time 10 {shlex.quote(start_url)}"
        stop_cmd = f"curl -sS -X POST --max-time 10 {shlex.quote(stop_url)}"
        bench_cmd = (
            f"{shlex.join(['vllm', 'bench', 'serve', *self.config.vllm_bench_args()])} "
            "2>&1 | tee /app/VAP/log/vllm_bench.log"
        )
        logger.debug("Benchmark and profile command: %s", bench_cmd)

        started = False
        try:
            exit_code, output = self.container.exec_run(
                ["/bin/bash", "-c", start_cmd]
            )
            if exit_code != 0:
                msg = (output or b"").decode(errors="replace")
                raise RuntimeError(f"Failed to start vLLM profiler: {msg}")
            started = True
            exit_code, output = self.container.exec_run(
                ["/bin/bash", "-c", bench_cmd],
                demux=True,
            )
            stdout, stderr = output if output else (b"", b"")
            if exit_code != 0:
                msg = (stderr or stdout or b"").decode(errors="replace")
                logger.error("Benchmark failed (exit %s): %s", exit_code, msg)
                raise RuntimeError(f"vllm bench failed with exit code {exit_code}")
            logger.info("Benchmark finished successfully")
        finally:
            if started:
                try:
                    self.container.exec_run(["/bin/bash", "-c", stop_cmd])
                except Exception as exc:
                    logger.warning("Failed to stop vLLM profiler cleanly: %s", exc)

    def build_container(self) -> Any:
        try:
            self.docker_client.images.get(self.config.docker_image)
        except docker.errors.ImageNotFound:
            logger.error("Docker image %s is not available", self.config.docker_image)
            raise
        logger.info("Docker image %s is available", self.config.docker_image)

        mounts = self.build_container_mounts(self.log_path)
        safe_model_name = self.config.model_cfg.model_name.replace("/", "_")
        container_name = f"vap_{safe_model_name}_{self.date_str}"
        container_cfg = self.config.container_cfg
        rocm_devices = ["/dev/kfd", "/dev/mem"]
        devices = list(
            dict.fromkeys(rocm_devices + (container_cfg.devices or ["/dev/dri/"]))
        )
        os.makedirs(os.path.join(self.log_path, "vllm-profile"), exist_ok=True)

        container = self.docker_client.containers.run(
            image=self.config.docker_image,
            name=container_name,
            ipc_mode="host",
            network_mode="host",
            cap_add=["SYS_ADMIN", "SYS_PTRACE"],
            devices=devices,
            ulimits=[
                Ulimit(name="memlock", soft=-1, hard=-1),
                Ulimit(name="nofile", soft=65535, hard=65535),
            ],
            shm_size="128G",
            group_add=["video"],
            security_opt=["seccomp=unconfined"],
            mounts=mounts,
            environment=container_cfg.env_vars or {},
            entrypoint=[],
            command=["/bin/bash", "-c", "sleep infinity"],
            detach=True,
            remove=False,
        )
        logger.info("Started runner container %s (%s)", container_name, container.id)
        return container

    def remove_container(self) -> None:
        if self.container is None:
            return
        try:
            self.container.stop()
        except Exception as exc:
            logger.warning("Failed to stop vLLM container: %s", exc)
        try:
            self.container.remove()
        except Exception as exc:
            logger.warning("Failed to remove vLLM container: %s", exc)
        self.container = None
