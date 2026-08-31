from __future__ import annotations

import logging
import shlex

from vap.config import VAPConfig
from vap.runners import DockerRunner

logger = logging.getLogger("VAP")

BENCH_LOG_PATH = "/app/VAP/log/vllm_bench.log"


class BenchmarkLifecycle:
    """Own `vllm bench serve` independently of Torch Profiler."""

    def __init__(self, config: VAPConfig, master_runner: DockerRunner):
        self.config = config
        self.master_runner = master_runner

    def run(self) -> None:
        bench_command = shlex.join(
            ["vllm", "bench", "serve", *self.config.vllm_bench_args()]
        )
        shell_command = f"set -o pipefail; {bench_command} 2>&1 | tee {BENCH_LOG_PATH}"
        logger.debug("Benchmark command: %s", shell_command)
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
            raise RuntimeError(f"vllm bench failed with exit code {result.exit_code}")
        logger.info("Benchmark finished successfully")
