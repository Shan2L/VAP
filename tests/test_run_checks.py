from __future__ import annotations

import json
import unittest

from test_security_and_runtime import PROJECT_ROOT  # installs the docker stub
from vap import validation
from vap.config import VAPConfig


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

    def test_benchmark_only_runs_have_no_profiler_window(self) -> None:
        config = example_config(delay_iterations=640)
        config["profiler_cfg"]["enable"] = False
        self.assertEqual(profiler_warnings(config), [])

    def test_configs_written_before_distributed_runs_still_load(self) -> None:
        config = example_config()
        config["clock_probe_cfg"]["enabled"] = False
        config["distributed_cfg"] = {
            "num_nodes": 2,
            "ray_port": 6380,
            "head_node": "localhost",
            "worker_nodes": ["worker-a"],
        }
        result = validation.validate_config_payload(config)
        self.assertTrue(result["valid"], result.get("errors"))
        self.assertIn("distributed_cfg", [w["path"] for w in result["warnings"]])
        distributed = VAPConfig.model_validate(config).distributed_cfg
        self.assertEqual(
            (distributed.enable, distributed.ray_port, distributed.worker_nodes),
            (False, 6380, ["worker-a"]),
        )

    def test_containers_do_not_map_physical_memory(self) -> None:
        for path in (PROJECT_ROOT / "vap").rglob("*.py"):
            self.assertNotIn('"/dev/mem"', path.read_text(encoding="utf-8"), path)


if __name__ == "__main__":
    unittest.main()
