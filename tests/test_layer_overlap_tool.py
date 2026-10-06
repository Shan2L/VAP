from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_trace_attribution import rank_trace, step_kernels, write_traces

import server
from agent_runtime import VAPAgentRuntime


class LayerOverlapToolTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.logs = root / "logs"
        self.logs.mkdir()
        self.model_root = root / "models"
        model_dir = self.model_root / "org" / "tiny"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps({"num_hidden_layers": 1}))
        patcher = patch.object(server, "LOGS_DIR", self.logs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_run(self, name: str, ranks: int, late_attention: float = 0.0) -> Path:
        run_dir = self.logs / name
        (run_dir / "vllm-profile").mkdir(parents=True)
        config = {
            "model_cfg": {"model_name": "org/tiny", "model_path": str(self.model_root)},
            "vllm_deploy_cfg": {"-tp": ranks},
        }
        (run_dir / "config.json").write_text(json.dumps(config))
        traces = [
            rank_trace(rank, step_kernels(late_attention=late_attention))
            for rank in range(ranks)
        ]
        write_traces(run_dir / "vllm-profile", traces)
        return run_dir

    def test_reports_every_layer_with_layer_count_from_model_config(self) -> None:
        self.make_run("run_tp2", 2)

        data = server.analyze_layer_overlap({"run_dir": "run_tp2"})

        self.assertFalse(data["cached"])
        self.assertEqual(data["model"], "org/tiny")
        self.assertEqual(data["layout"]["num_layers"], 1)
        decode = data["phases"]["decode"]
        self.assertEqual(data["parallel"], "TP2")
        self.assertEqual(
            [row[0] for row in decode["layers"]["rows"]], ["pre", "L0", "post"]
        )
        self.assertEqual(
            decode["layers"]["columns"][:3], ["Segment", "Wall µs", "Compute µs"]
        )
        self.assertTrue(data["findings"])
        report = self.logs / "run_tp2" / "attribution" / "attribution_report.md"
        self.assertIn(
            "unknown model, TP2",
            report.read_text(encoding="utf-8").replace("org/tiny", "unknown model"),
        )
        self.assertEqual(
            data["downloads"][0]["download_url"],
            "/api/attribution/file?run_dir=run_tp2&name=attribution_report.md",
        )
        self.assertTrue(server.analyze_layer_overlap({"run_dir": "run_tp2"})["cached"])

    def test_compares_tp_sizes_against_linear_scaling(self) -> None:
        self.make_run("run_tp2", 2)
        self.make_run("run_tp4", 4)

        data = server.compare_runs({"base_run": "run_tp2", "target_run": "run_tp4"})

        self.assertEqual((data["base_run"], data["target_run"]), ("run_tp2", "run_tp4"))
        self.assertEqual(data["comparison"]["mode"], "scaling")
        self.assertEqual(data["comparison"]["phases"]["decode"]["ideal_speedup"], 2.0)
        self.assertEqual(
            (data["comparison"]["a"]["parallel"], data["comparison"]["b"]["parallel"]),
            ("TP2", "TP4"),
        )
        self.assertNotIn("layers", data["comparison"]["phases"]["decode"])
        for suffix in (".md", ".html", "_layers.csv", ".json"):
            self.assertTrue(
                (
                    self.logs
                    / "run_tp4"
                    / "attribution"
                    / f"compare_vs_run_tp2{suffix}"
                ).is_file()
            )
        self.assertEqual(len(data["downloads"]), 5)

    def test_compares_two_runs_of_the_same_tp_as_an_ab_change(self) -> None:
        self.make_run("run_a", 2)
        self.make_run("run_b", 2, late_attention=5.0)

        payload, report = server.compare_two_runs(
            {"base_run": "run_a", "target_run": "run_b"}
        )

        comparison = payload["comparison"]
        self.assertEqual(comparison["mode"], "same_gpus")
        decode = comparison["phases"]["decode"]
        self.assertAlmostEqual(decode["change"]["attention"], 5.0)
        self.assertEqual([row["label"] for row in decode["layers"]], ["L0"])
        self.assertEqual(report.name, "compare_vs_run_a.md")
        text = report.read_text(encoding="utf-8")
        self.assertIn("# TP2 (A) vs TP2 (B)", text)
        self.assertIn("#### Time per layer (µs per token round)", text)
        self.assertIn("### Kernels with the largest change", text)
        self.assertNotIn("layers", payload["runs"][0]["phases"]["decode"])

    def test_rejects_runs_outside_logs_and_bad_arguments(self) -> None:
        self.make_run("run_tp2", 2)
        for run_dir in ("../", "/etc", str(self.logs), "missing"):
            with self.assertRaises(ValueError):
                server.analyze_layer_overlap({"run_dir": run_dir})
        for args in ({"num_layers": 0}, {"num_layers": True}, {"refresh": "yes"}):
            with self.assertRaises(ValueError):
                server.analyze_layer_overlap({"run_dir": "run_tp2", **args})
        for args in (
            {"base_run": "run_tp2", "target_run": "run_tp2"},
            {"base_run": "run_tp2"},
        ):
            with self.assertRaises(ValueError):
                server.compare_runs(args)

    def test_log_reads_are_bounded(self) -> None:
        run_dir = self.make_run("run_tp2", 2)
        lines = [
            f"line {index} {'ERROR boom' if index % 100 == 0 else 'ok'}"
            for index in range(5000)
        ]
        (run_dir / "vllm_deploy.log").write_text("\n".join(lines))
        with patch.object(
            server, "get_run_state_snapshot", return_value={"run_dir": str(run_dir)}
        ):
            tail = server.read_log_tail(
                {"file_name": "vllm_deploy.log", "max_chars": 1000}
            )
            errors = server.read_log_tail(
                {"file_name": "vllm_deploy.log", "contains": "error"}
            )
            viewed = server.read_current_log_file("vllm_deploy.log", 2000)
        self.assertTrue(tail["truncated"])
        self.assertTrue(tail["content"].endswith("line 4999 ok"))
        self.assertLessEqual(len(tail["content"]), 1000)
        self.assertEqual(errors["matched_lines"], 50)
        self.assertNotIn(" ok", errors["content"])
        self.assertTrue(viewed["truncated"])
        self.assertLessEqual(len(viewed["content"]), 2000)
        self.assertTrue(viewed["content"].startswith("line "))

    def test_attribution_file_only_serves_regular_report_files(self) -> None:
        run_dir = self.make_run("run_tp2", 2)
        server.analyze_layer_overlap({"run_dir": "run_tp2"})
        report = server.attribution_file("run_tp2", "attribution_report.md")
        self.assertEqual(
            report, (run_dir / "attribution" / "attribution_report.md").resolve()
        )
        (run_dir / "attribution" / "link.md").symlink_to(run_dir / "config.json")
        for name in (
            "../config.json",
            "notes.py",
            ".hidden.md",
            "link.md",
            "missing.md",
            None,
        ):
            with self.assertRaises(ValueError):
                server.attribution_file("run_tp2", name)
        with self.assertRaises(ValueError):
            server.attribution_file(None, "attribution_report.md")

    def test_lists_runs_with_model_and_rank_traces(self) -> None:
        self.make_run("run_tp2", 2)
        runs = server.list_profile_runs({})["runs"]
        self.assertEqual(
            runs,
            [
                {
                    "run_dir": "run_tp2",
                    "model": "org/tiny",
                    "tensor_parallel": 2,
                    "parallel": "TP2",
                    "gpus": 2,
                    "concurrency": None,
                    "tpot_ms": None,
                    "output_tok_s": None,
                    "rank_traces": 2,
                    "has_layer_report": False,
                }
            ],
        )

    def test_report_falls_back_to_rules_when_the_agent_is_locked(self) -> None:
        self.make_run("run_a", 2)
        self.make_run("run_b", 2, late_attention=5.0)
        runtime = VAPAgentRuntime()
        with (
            patch.object(server, "get_agent_runtime", return_value=runtime),
            patch.dict("os.environ", {"VAP_LLM_SUBSCRIPTION_KEY": ""}),
        ):
            events = list(
                server.report_stream(
                    {"base_run": "run_a", "target_run": "run_b", "language": "zh"}
                )
            )
            again = list(
                server.report_stream(
                    {"base_run": "run_a", "target_run": "run_b", "language": "zh"}
                )
            )
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds[:2], ["status", "data"])
        self.assertIn("notice", kinds)
        report = events[-1]
        self.assertEqual(
            (report["type"], report["source"], report["cached"]),
            ("report", "rules", False),
        )
        self.assertIn("B（TP2）每个 token 轮次需要", report["conclusion"])
        self.assertTrue(again[-1]["cached"])
        names = [item["download_url"].rsplit("=", 1)[1] for item in report["downloads"]]
        self.assertEqual(
            names[:2],
            ["report_zh_compare_vs_run_a.md", "report_zh_compare_vs_run_a.html"],
        )
        text = (self.logs / "run_b" / "attribution" / names[0]).read_text(
            encoding="utf-8"
        )
        for heading in (
            "## 结论",
            "## 关键指标",
            "## 优化建议",
            "# 附录：完整数据",
            "#### Time per layer",
        ):
            self.assertIn(heading, text)

    def test_report_streams_the_agent_narrative_and_keeps_the_template(self) -> None:
        self.make_run("run_a", 2)
        self.make_run("run_b", 2, late_attention=5.0)
        reply = "## Conclusion\n\n- B is slower: attention +5 µs per step.\n\n## Key metrics\n\n| x |\n"

        class FakeRuntime:
            prompts: list[tuple[str, str]] = []

            def status(self) -> dict:
                return {"unlocked": True}

            def stream_completion(self, system, user, purpose, max_tokens=None):
                self.prompts.append((system, user))
                yield from (reply[:20], reply[20:])

        fake = FakeRuntime()
        with patch.object(server, "get_agent_runtime", return_value=fake):
            events = list(
                server.report_stream(
                    {"base_run": "run_a", "target_run": "run_b", "language": "en"}
                )
            )
        deltas = "".join(
            event["content"] for event in events if event["type"] == "delta"
        )
        self.assertEqual(deltas, reply)
        report = events[-1]
        self.assertEqual(report["source"], "agent")
        self.assertEqual(
            report["conclusion"], "- B is slower: attention +5 µs per step."
        )
        system, user = fake.prompts[0]
        self.assertIn("## Recommendations", system)
        self.assertIn('"mode":"same_gpus"', user)

        reply = "Sorry, I cannot help."
        with patch.object(server, "get_agent_runtime", return_value=fake):
            events = list(
                server.report_stream(
                    {
                        "base_run": "run_a",
                        "target_run": "run_b",
                        "language": "en",
                        "refresh": True,
                    }
                )
            )
        self.assertEqual(events[-1]["source"], "rules")
        self.assertIn(
            "did not follow the template",
            " ".join(e.get("message", "") for e in events),
        )

    def test_report_rejects_unknown_languages(self) -> None:
        with self.assertRaises(ValueError):
            next(
                server.report_stream(
                    {"base_run": "a", "target_run": "b", "language": "fr"}
                )
            )

    def test_agent_exposes_the_tools_and_prompt_rule(self) -> None:
        runtime = VAPAgentRuntime()
        server.register_vap_agent_tools(runtime)
        safety = {tool["name"]: tool["safety"] for tool in runtime.status()["tools"]}
        self.assertEqual(safety["analyze_layer_overlap"], "safe")
        self.assertEqual(safety["list_profile_runs"], "read_only")
        self.assertEqual(safety["compare_runs"], "safe")
        self.assertIn("call analyze_layer_overlap", runtime._system_prompt())


if __name__ == "__main__":
    unittest.main()
