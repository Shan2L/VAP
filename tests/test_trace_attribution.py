from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from vap.analysis.attribution import (
    CATEGORIES,
    STEP_KEY,
    Layout,
    Topology,
    analysis_findings,
    analyze,
    classify_kernel,
    compare,
    find_trace_files,
    pp_partition,
    render_compare_markdown,
    select_trace_files,
    step_phase,
    steps_per_round,
    write_analysis,
)

BASE_NS = 1_700_000_000_000_000_000
STEP_US = 120.0
STEP_NAME = "execute_context_0(0)_generation_4(4)"


def step_kernels(late_attention: float = 0.0, post_comm=(110.0, 115.0)):
    return [
        (0.0, 10.0, "embedding_kernel"),
        (10.0, 20.0, "ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)"),
        (20.0, 40.0, "Cijk_Alik_Bljk_BBS_BH_MT64x64"),
        (40.0, 60.0 + late_attention, "kernel_unified_attention_2d"),
        (60.0 + late_attention, 70.0, "ncclDevKernel_Generic_4"),
        (70.0, 90.0, "Cijk_Alik_Bljk_BBS_BH_MT128x64"),
        (90.0, 100.0, "ncclDevKernel_Generic_4"),
        (100.0, 110.0, "rms_norm_kernel"),
        (*post_comm, "ncclDevKernel_Generic_4"),
    ]


def rank_trace(
    rank: int, kernels, extra_stream=(), steps: int = 3, gpu_skew: float = 0.0
) -> dict:
    """Absolute GPU time of step s is 1000 + 120 s us; each rank has its own
    baseTimeNanoseconds, so its relative ts differ by the base offset."""
    base_ns = BASE_NS + rank * 2_000_000
    shift = -rank * 2000.0
    events = []
    for step in range(steps):
        gpu_t0 = 1000.0 + STEP_US * step + shift
        cpu_t0 = gpu_t0 - 500.0
        events.append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": STEP_NAME,
                "pid": 1,
                "tid": 7,
                "ts": cpu_t0,
                "dur": 100.0,
            }
        )
        launches = [(start, end, name, 1) for start, end, name in kernels]
        launches += [(start, end, name, 2) for start, end, name in extra_stream]
        for index, (start, end, name, stream) in enumerate(launches):
            correlation = step * 100 + index
            events.append(
                {
                    "ph": "X",
                    "cat": "cuda_runtime",
                    "name": "hipLaunchKernel",
                    "pid": 1,
                    "tid": 7,
                    "ts": cpu_t0 + 1 + index,
                    "dur": 0.5,
                    "args": {"correlation": correlation},
                }
            )
            events.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": name,
                    "pid": 0,
                    "tid": stream,
                    "ts": gpu_t0 + gpu_skew + start,
                    "dur": end - start,
                    "args": {"correlation": correlation, "stream": stream},
                }
            )
    return {"traceEvents": events, "baseTimeNanoseconds": base_ns}


def write_traces(root: Path, traces) -> list[Path]:
    paths = []
    for rank, trace in enumerate(traces):
        path = root / f"dp0_pp0_tp{rank}_dcp0_ep0_rank{rank}.1700.pt.trace.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(trace, handle)
        paths.append(path)
    return paths


def scaling_run(world_size: int, layer: dict) -> dict:
    values = {category: float(layer.get(category, 0.0)) for category in CATEGORIES}
    return {
        "meta": {"world_size": world_size, "topology": {"tp": world_size}},
        "phases": {
            "decode": {
                "steps": 2,
                "sequences_per_step": 32,
                "segments": ["L0"],
                "segment_stage": {"L0": 0},
                "step_names": {STEP_NAME: 2},
                "mean": {"L0": values, STEP_KEY: values},
                "per_rank": {"L0": {"0": values}, STEP_KEY: {"0": values}},
            }
        },
    }


def pipeline_run() -> dict:
    """TP1 x PP2, one layer per stage: per step each rank spends 60 us in its
    layer and 40 us outside it (stage 0 waits in a bubble, stage 1 samples)."""

    def values(**parts: float) -> dict[str, float]:
        return {category: float(parts.get(category, 0.0)) for category in CATEGORIES}

    layer0, layer1 = values(gemm=50, comm_xfer=10), values(gemm=40, attention=20)
    step0 = values(gemm=50, comm_xfer=10, bubble=30, idle=10)
    step1 = values(gemm=40, attention=20, other=40)
    return {
        "meta": {"world_size": 2, "topology": {"tp": 1, "pp": 2}, "concurrency": 32},
        "phases": {
            "decode": {
                "steps": 2,
                "sequences_per_step": 16,
                "segments": ["L0", "L1"],
                "segment_stage": {"L0": 0, "L1": 1},
                "step_names": {STEP_NAME: 2},
                "mean": {
                    "L0": layer0,
                    "L1": layer1,
                    STEP_KEY: {c: (step0[c] + step1[c]) / 2 for c in CATEGORIES},
                },
                "per_rank": {
                    "L0": {"0": layer0},
                    "L1": {"1": layer1},
                    STEP_KEY: {"0": step0, "1": step1},
                },
                "stages": [
                    {"stage": 0, "ranks": [0], "layers": [0, 1], "step": step0},
                    {"stage": 1, "ranks": [1], "layers": [1, 2], "step": step1},
                ],
            }
        },
    }


def pp_trace(rank: int, stage: int, steps: int = 3) -> dict:
    """TP1 x PP2: stage 0 computes 0-60 us then waits for stage 1's sampled
    tokens on a receive (stream 3) until 100; stage 1 receives (2 us), computes
    10-100 and broadcasts the tokens (2 us)."""
    group = "[0, 1]"
    if stage == 0:
        kernels = [
            (0.0, 30.0, "Cijk_gemm_s0", 1, None),
            (30.0, 60.0, "kernel_unified_attention_2d", 1, None),
            (60.0, 62.0, "ncclDevKernel_SendRecv", 2, "send"),
            (2.0, 98.0, "ncclDevKernel_Generic_4", 3, "broadcast"),
        ]
    else:
        kernels = [
            (0.0, 2.0, "ncclDevKernel_SendRecv", 2, "recv"),
            (10.0, 70.0, "Cijk_gemm_s1", 1, None),
            (70.0, 96.0, "sampler_kernel", 1, None),
            (96.0, 98.0, "ncclDevKernel_Generic_4", 3, "broadcast"),
        ]
    trace = rank_trace(rank, [])
    events = [
        event for event in trace["traceEvents"] if event["cat"] == "user_annotation"
    ]
    for step in range(steps):
        gpu_t0 = 1000.0 + 100.0 * step - rank * 2000.0
        cpu_t0 = gpu_t0 - 500.0
        events[step]["name"] = "execute_context_0(0)_generation_16(16)"
        events[step]["ts"] = cpu_t0
        for index, (start, end, name, stream, collective) in enumerate(kernels):
            correlation = step * 100 + index
            events.append(
                {
                    "ph": "X",
                    "cat": "cuda_runtime",
                    "name": "hipLaunchKernel",
                    "pid": 1,
                    "tid": 7,
                    "ts": cpu_t0 + 1 + index,
                    "dur": 0.5,
                    "args": {"correlation": correlation},
                }
            )
            args = {"correlation": correlation, "stream": stream}
            if collective:
                args.update(
                    {"Collective name": collective, "Process Group Ranks": group}
                )
            events.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": name,
                    "pid": 0,
                    "tid": stream,
                    "ts": gpu_t0 + start,
                    "dur": end - start,
                    "args": args,
                }
            )
    return {"traceEvents": events, "baseTimeNanoseconds": trace["baseTimeNanoseconds"]}


class TraceAttributionTests(unittest.TestCase):
    def analyze_traces(self, traces, tolerance: float = 1.0) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            write_traces(Path(tmp), traces)
            files = find_trace_files(Path(tmp))
            return analyze(
                files,
                Layout(num_layers=1),
                Topology(tp=2),
                violation_tolerance_us=tolerance,
            )

    def assertCategories(self, values: dict, expected: dict) -> None:
        for category in CATEGORIES:
            self.assertAlmostEqual(
                values[category], expected.get(category, 0.0), places=6, msg=category
            )

    def test_classifies_rocm_kernel_names(self) -> None:
        self.assertEqual(
            classify_kernel(
                "ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)"
            ),
            "comm",
        )
        self.assertEqual(classify_kernel("Cijk_Alik_Bljk_BBS_BH_MT64x64x64"), "gemm")
        self.assertEqual(classify_kernel("kernel_unified_attention_2d"), "attention")
        self.assertEqual(classify_kernel("reshape_and_cache_flash_kernel"), "attention")
        self.assertEqual(classify_kernel("__amd_rocclr_copyBuffer"), "memcpy")
        self.assertEqual(classify_kernel("rms_norm_kernel"), "other")
        self.assertEqual(
            step_phase("execute_context_2(2048)_generation_30(30)"), "mixed"
        )
        self.assertEqual(step_phase("execute_context_0(0)_generation_32(32)"), "decode")

    def test_splits_exposed_comm_into_wait_and_transfer_per_layer(self) -> None:
        result = self.analyze_traces(
            [
                rank_trace(0, step_kernels()),
                rank_trace(1, step_kernels(late_attention=5.0)),
            ]
        )
        quality = result["quality"]
        self.assertEqual(quality["alignment"]["method"], "baseTimeNanoseconds")
        self.assertEqual(quality["causality_violation_us_max"], 0.0)
        decode = result["phases"]["decode"]
        self.assertEqual(decode["steps"], 2)
        self.assertEqual(decode["segments"], ["pre", "L0", "post"])
        per_rank = decode["per_rank"]
        self.assertCategories(
            per_rank["L0"]["0"],
            {"gemm": 40, "attention": 20, "comm_wait": 5, "comm_xfer": 15},
        )
        self.assertCategories(
            per_rank["L0"]["1"], {"gemm": 40, "attention": 25, "comm_xfer": 15}
        )
        self.assertCategories(per_rank["pre"]["0"], {"other": 10, "comm_xfer": 10})
        self.assertCategories(
            per_rank["post"]["1"], {"other": 10, "comm_xfer": 5, "idle": 5}
        )
        for rank in ("0", "1"):
            self.assertAlmostEqual(sum(per_rank[STEP_KEY][rank].values()), STEP_US)
        attn = next(row for row in decode["collectives"] if row["role"] == "attention")
        self.assertEqual(
            [(row["dim"], row["op"], row["role"]) for row in decode["collectives"]],
            [
                ("tp", "all_reduce", "embedding"),
                ("tp", "all_reduce", "attention"),
                ("tp", "all_reduce", "mlp"),
                ("tp", "all_reduce", "post"),
            ],
        )
        self.assertAlmostEqual(attn["count_per_step"], 1.0)
        self.assertAlmostEqual(attn["wait_us_max_p50"], 5.0)
        self.assertAlmostEqual(attn["wait_us_mean"], 2.5)
        self.assertAlmostEqual(attn["xfer_us_mean"], 5.0)

    def test_counts_comm_hidden_by_compute_on_another_stream_as_overlap(self) -> None:
        result = self.analyze_traces(
            [
                rank_trace(
                    0, step_kernels(), extra_stream=[(62.0, 68.0, "Cijk_side_gemm")]
                ),
                rank_trace(1, step_kernels(late_attention=5.0)),
            ]
        )
        decode = result["phases"]["decode"]
        self.assertCategories(
            decode["per_rank"]["L0"]["0"],
            {
                "gemm": 40,
                "attention": 20,
                "overlap": 6,
                "comm_wait": 2,
                "comm_xfer": 12,
            },
        )
        derived = decode["derived"]["L0"]
        self.assertAlmostEqual(derived["overlap_ratio"], 3.0 / (3.0 + 1.0 + 13.5))

    def test_reports_collectives_that_end_before_a_peer_starts(self) -> None:
        result = self.analyze_traces(
            [
                rank_trace(0, step_kernels()),
                rank_trace(1, step_kernels(post_comm=(116.0, 118.0))),
            ],
            tolerance=0.5,
        )
        quality = result["quality"]
        self.assertAlmostEqual(quality["causality_violation_us_max"], 1.0)
        self.assertEqual(quality["causality_violations_over_tolerance"], 2)

    def test_scaling_loss_columns_sum_to_segment_loss(self) -> None:
        result = compare(
            scaling_run(4, {"gemm": 100, "attention": 40, "comm_xfer": 20}),
            scaling_run(8, {"gemm": 50, "attention": 20, "comm_xfer": 30}),
        )
        self.assertEqual(result["mode"], "scaling")
        decode = result["phases"]["decode"]
        self.assertAlmostEqual(decode["change"]["comm_xfer"], 20.0)
        self.assertAlmostEqual(decode["change"]["gemm"], 0.0)
        self.assertAlmostEqual(sum(decode["change"].values()), 100.0 - 160.0 / 2)
        self.assertAlmostEqual(decode["layers"][0]["change_us"], 100.0 - 160.0 / 2)
        self.assertAlmostEqual(decode["scaling_efficiency"], (160.0 / 100.0) / 2)

    def test_wait_and_transfer_do_not_depend_on_clock_offset(self) -> None:
        result = self.analyze_traces(
            [
                rank_trace(0, step_kernels()),
                rank_trace(1, step_kernels(late_attention=5.0), gpu_skew=30.0),
            ]
        )
        per_rank = result["phases"]["decode"]["per_rank"]
        self.assertCategories(
            per_rank["L0"]["0"],
            {"gemm": 40, "attention": 20, "comm_wait": 5, "comm_xfer": 15},
        )
        self.assertCategories(
            per_rank["L0"]["1"], {"gemm": 40, "attention": 25, "comm_xfer": 15}
        )
        quality = result["quality"]
        self.assertAlmostEqual(quality["residual_clock_offset_us"]["1"], 30.0)
        self.assertGreater(quality["causality_violations_over_tolerance"], 0)
        texts = [item["text"] for item in analysis_findings(result)]
        self.assertTrue(any("still differ by up to 30.0" in text for text in texts))

    def test_markdown_report_has_findings_quality_and_every_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_traces(
                root,
                [
                    rank_trace(0, step_kernels()),
                    rank_trace(1, step_kernels(late_attention=5.0)),
                ],
            )
            result = analyze(
                find_trace_files(root), Layout(num_layers=1), Topology(tp=2)
            )
            outputs = write_analysis(result, root / "attribution")
            report = Path(outputs["markdown"]).read_text(encoding="utf-8")
        for heading in (
            "## Findings",
            "## Data quality",
            "### Time breakdown",
            "### Per segment",
            "### Communication",
            "## Method",
        ):
            self.assertIn(heading, report)
        for segment in ("pre", "L0", "post"):
            self.assertRegex(report, rf"\n\| {segment} \| ")
        self.assertIn("they run on the compute stream", report)
        self.assertIn("| Comm wait (exposed) | 2.5 |", report)

    def test_scaling_report_names_the_largest_loss(self) -> None:
        result = compare(
            scaling_run(4, {"gemm": 100, "attention": 40, "comm_xfer": 20}),
            scaling_run(8, {"gemm": 50, "attention": 20, "comm_xfer": 30}),
        )
        report = render_compare_markdown(result)
        self.assertIn("# TP4 (A) vs TP8 (B)", report)
        self.assertIn("splits into Exposed collective comm +0.02 ms", report)
        self.assertRegex(report, r"\n\| L0 \| S0/S0 \| 160.0 \| 100.0 \| 20.0 \| ")

    def test_topology_maps_communicators_to_parallel_dimensions(self) -> None:
        topology = Topology(tp=2, pp=2)
        self.assertEqual(topology.label, "TP2×PP2")
        self.assertEqual(topology.stage(2), 1)
        self.assertEqual(topology.comm_dim((2, 3)), "tp")
        self.assertEqual(topology.comm_dim((0, 2)), "pp")
        self.assertEqual(topology.comm_dim(None), "tp")
        self.assertEqual(Topology(tp=2, dp=2).comm_dim((0, 2)), "dp")
        self.assertEqual(Topology(tp=2, dp=2, ep=4).comm_dim((0, 1, 2, 3)), "ep")
        self.assertEqual(Topology(tp=2, dp=2, ep=4).label, "TP2×DP2 EP4")
        self.assertEqual(pp_partition(40, 2), [20, 20])
        self.assertEqual(pp_partition(41, 3), [14, 14, 13])

    def test_pipeline_wait_outside_compute_is_a_bubble_and_steps_form_a_round(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_traces(Path(tmp), [pp_trace(0, 0), pp_trace(1, 1)])
            result = analyze(
                find_trace_files(Path(tmp)), Layout(num_layers=2), Topology(tp=1, pp=2)
            )
        result["meta"]["concurrency"] = 32
        decode = result["phases"]["decode"]
        self.assertEqual(steps_per_round(result, "decode"), 2.0)
        self.assertEqual([stage["ranks"] for stage in decode["stages"]], [[0], [1]])
        stage0 = decode["per_rank"][STEP_KEY]["0"]
        stage1 = decode["per_rank"][STEP_KEY]["1"]
        # Stage 0: 60 us compute, 2 us exposed send, receive spins 62-96, 2 us transfer.
        self.assertCategories(
            stage0, {"gemm": 30, "attention": 30, "pp_xfer": 4, "bubble": 34, "idle": 2}
        )
        self.assertCategories(
            stage1, {"gemm": 60, "other": 26, "pp_xfer": 4, "idle": 10}
        )
        for values in (stage0, stage1):
            self.assertAlmostEqual(sum(values.values()), 100.0)
        texts = [item["text"] for item in analysis_findings(result)]
        self.assertTrue(any("stage 1 is the bottleneck" in text for text in texts))

    def test_layers_count_with_their_share_of_the_token(self) -> None:
        comparison = compare(pipeline_run(), pipeline_run())
        decode = comparison["phases"]["decode"]
        # Per token (two micro-batch steps) a layer runs 120 us on its stage's
        # GPU, which is half of the GPUs; outside the layers takes 80 us.
        self.assertEqual(
            [
                (row["label"], sum(row["a"].values()), row["wall_us"]["a"])
                for row in decode["layers"]
            ],
            [("L0", 60.0, 120.0), ("L1", 60.0, 120.0)],
        )
        for side in ("a", "b"):
            layers = sum(decode["layer_sum"][side].values())
            outside = sum(decode["non_layer"][side].values())
            self.assertEqual(
                (layers, outside, decode["round_us"][side]), (120.0, 80.0, 200.0)
            )
        self.assertIn(
            "by location, the 2 decoder layers +0.00 ms",
            render_compare_markdown(comparison),
        )


class TraceSelectionTests(unittest.TestCase):
    def test_prefers_the_aligned_timeline_and_falls_back_to_raw_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "vllm-profile"
            worker = profile / "workers" / "worker-1"
            worker.mkdir(parents=True)
            master = write_traces(profile, [{}])
            remote = worker / "dp0_pp0_tp1_dcp0_ep0_rank1.1700.pt.trace.json.gz"
            remote.write_bytes(b"")
            self.assertEqual(select_trace_files(profile), (master + [remote], "raw"))

            aligned = profile / "aligned"
            aligned.mkdir()
            for rank in (0, 1):
                (aligned / f"rank-{rank}.aligned.json").write_text("{}")
            manifest = {
                "primary_timeline": "aligned",
                # Absolute paths from another host: re-anchored to this folder.
                "aligned": {
                    "1": "/elsewhere/aligned/rank-1.aligned.json",
                    "0": "/elsewhere/aligned/rank-0.aligned.json",
                },
                "clc": {},
            }
            (aligned / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(
                select_trace_files(profile),
                (
                    [aligned / "rank-0.aligned.json", aligned / "rank-1.aligned.json"],
                    "aligned",
                ),
            )

            manifest["primary_timeline"] = "clc"
            (aligned / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(select_trace_files(profile)[1], "raw")


if __name__ == "__main__":
    unittest.main()
