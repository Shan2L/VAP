from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test_security_and_runtime import PROJECT_ROOT  # installs the docker stub

import main
import validation
from config import VAPConfig


def example_config(**profiler: int) -> dict:
    config = json.loads(
        (PROJECT_ROOT / "example-config.json").read_text(encoding="utf-8")
    )
    config["vllm_bench_cfg"].update(
        {"--num-prompts": 32, "--max-concurrency": 32, "--random-output-len": 512}
    )
    config["profiler_cfg"].update(profiler)
    return config


def profiler_warnings(config: dict) -> list[str]:
    return [
        warning["path"]
        for warning in validation.validate_config_payload(config)["warnings"]
        if warning["path"].startswith("profiler_cfg")
    ]


class RunCheckTests(unittest.TestCase):
    def test_warns_when_the_profiler_window_misses_the_benchmark(self) -> None:
        self.assertEqual(
            profiler_warnings(example_config(delay_iterations=640)),
            ["profiler_cfg.delay_iterations"],
        )
        self.assertEqual(
            profiler_warnings(example_config(delay_iterations=500, max_iterations=24)),
            ["profiler_cfg.max_iterations"],
        )
        self.assertEqual(
            profiler_warnings(example_config(delay_iterations=200, max_iterations=24)),
            [],
        )

    def test_expected_workers_multiply_parallel_sizes(self) -> None:
        config = example_config()
        config["vllm_deploy_cfg"].update({"-tp": 4, "--pipeline-parallel-size": "2"})
        self.assertEqual(
            validation.expected_gpu_workers(VAPConfig.model_validate(config)), 8
        )

    def test_counts_distinct_rank_traces_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "rank0.17.pt.trace.json.gz",
                "rank1.17.pt.trace.json.gz",
                "dp0_pp0_tp1_dcp0_ep0_rank1.16.pt.trace.json.gz",
                "host_7.async_llm.17.pt.trace.json.gz",
                "run-merged_trace.json.gz",
            ):
                (root / name).write_text("{}")
            self.assertEqual(main.count_rank_traces(tmp), 2)
        self.assertEqual(main.count_rank_traces("/nonexistent"), 0)

    def test_container_does_not_map_physical_memory(self) -> None:
        source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertNotIn('"/dev/mem"', source)


if __name__ == "__main__":
    unittest.main()
