from __future__ import annotations

import logging
from collections.abc import Sequence

from vap.config import VAPConfig
from vap.runners import DockerRunner, ProcessHandle

logger = logging.getLogger("VAP")

WAIT_FOR_TCP_SCRIPT = """
import socket
import sys
import time

host, port, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1):
            raise SystemExit(0)
    except OSError:
        time.sleep(0.5)
raise SystemExit(f"timed out waiting for {host}:{port}")
"""

WAIT_FOR_RAY_NODES_SCRIPT = """
import ray
import sys
import time

address, expected, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
ray.init(address=address, ignore_reinit_error=True)
try:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [node for node in ray.nodes() if node.get("Alive")]
        if len(alive) >= expected:
            raise SystemExit(0)
        time.sleep(1)
    raise SystemExit(f"expected {expected} Ray nodes, found {len(alive)}")
finally:
    ray.shutdown()
"""


class RayClusterLifecycle:
    """Own the Ray processes and node identities for one profiling run."""

    def __init__(
        self,
        config: VAPConfig,
        master_runner: DockerRunner,
        worker_runners: Sequence[DockerRunner],
    ):
        self.config = config
        self.master_runner = master_runner
        self.worker_runners = list(worker_runners)
        self._processes: list[tuple[DockerRunner, ProcessHandle]] = []
        self._master_ip: str | None = None
        self._runner_node_ids: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        distributed = self.config.distributed_cfg
        return bool(distributed and distributed.enable)

    @property
    def master_ip(self) -> str | None:
        return self._master_ip

    @property
    def runner_node_ids(self) -> dict[str, str]:
        return dict(self._runner_node_ids)

    @property
    def processes(self) -> tuple[tuple[DockerRunner, ProcessHandle], ...]:
        return tuple(self._processes)

    def start(self) -> None:
        distributed = self.config.distributed_cfg
        if distributed is None or not distributed.enable:
            return
        if not self.worker_runners:
            raise RuntimeError("Distributed mode requires at least one worker")

        self._master_ip = self.master_runner.network.select_vllm_host_ip(
            self.worker_runners[0].target.hostname
        )
        self._runner_node_ids = {
            self.master_runner.target.label: self._master_ip,
        }
        ray_environment = {"VLLM_HOST_IP": self._master_ip}
        head_process = self.master_runner.process.start(
            [
                "ray",
                "start",
                "--head",
                f"--node-ip-address={self._master_ip}",
                f"--port={distributed.ray_port}",
                "--block",
            ],
            environment=ray_environment,
        )
        self._processes.append((self.master_runner, head_process))

        ray_address = f"{self._master_ip}:{distributed.ray_port}"
        for worker in self.worker_runners:
            self._wait_for_tcp(worker, self._master_ip, distributed.ray_port)
            worker_ip = worker.network.select_vllm_host_ip(self._master_ip)
            self._runner_node_ids[worker.target.label] = worker_ip
            worker_process = worker.process.start(
                [
                    "ray",
                    "start",
                    f"--address={ray_address}",
                    f"--node-ip-address={worker_ip}",
                    "--block",
                ],
                environment={"VLLM_HOST_IP": worker_ip},
            )
            self._processes.append((worker, worker_process))

        self.wait()

    def wait(self) -> None:
        distributed = self.config.distributed_cfg
        if distributed is None or not distributed.enable:
            return
        if self._master_ip is None:
            raise RuntimeError("Ray master node identity is unavailable")

        expected_nodes = 1 + len(self.worker_runners)
        ray_address = f"{self._master_ip}:{distributed.ray_port}"
        result = self.master_runner.process.run(
            [
                "python3",
                "-c",
                WAIT_FOR_RAY_NODES_SCRIPT,
                ray_address,
                str(expected_nodes),
                "120",
            ],
            environment={"VLLM_HOST_IP": self._master_ip},
            demux=True,
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Ray cluster did not become ready: {result.combined_text.strip()}"
            )
        logger.info("Ray cluster is ready with %d nodes", expected_nodes)

    def cleanup(self) -> None:
        for runner, process in reversed(self._processes):
            if runner.is_started:
                try:
                    runner.process.terminate(process)
                except Exception as exc:
                    logger.warning(
                        "Failed to terminate Ray on %s: %s",
                        runner.target.label,
                        exc,
                    )
        self._processes.clear()

    @staticmethod
    def _wait_for_tcp(
        runner: DockerRunner,
        host: str,
        port: int,
    ) -> None:
        result = runner.process.run(
            [
                "python3",
                "-c",
                WAIT_FOR_TCP_SCRIPT,
                host,
                str(port),
                "30",
            ],
            demux=True,
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"{runner.target.label} cannot reach {host}:{port}: "
                f"{result.combined_text.strip()}"
            )
