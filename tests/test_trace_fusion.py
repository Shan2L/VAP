from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from vap.clock_probe.postprocess.align import align_trace_file
from vap.clock_probe.postprocess.nccl import check_nccl_traces
from vap.postprocess.align import TraceInput, align_traces
from vap.postprocess.fuse import fuse_traces, load_trace
from vap.postprocess.trace import (
    find_perfetto_trace,
    profile_trace_candidates,
    trace_rank,
)


def rank_trace() -> dict:
    return {
        "traceEvents": [
            {
                "name": "process_name",
                "ph": "M",
                "pid": 1,
                "tid": 0,
                "args": {"name": "ORIGINAL"},
            },
            {
                "name": "hipLaunchKernel",
                "cat": "cuda_runtime",
                "ph": "X",
                "pid": 1,
                "tid": 10,
                "id": 7,
                "args": {"correlation": 5},
            },
            {
                "name": "vector_kernel",
                "cat": "kernel",
                "ph": "X",
                "pid": 2,
                "tid": 20,
                "id": 8,
                "args": {"correlation": 5},
            },
            {
                "name": "trace_internal",
                "cat": "Trace",
                "ph": "X",
                "pid": 1,
                "args": {},
            },
            {
                "name": "python_frame",
                "cat": "python_function",
                "ph": "X",
                "pid": 1,
                "args": {},
            },
        ]
    }


class TraceFusionTests(unittest.TestCase):
    def test_alignment_accepts_compact_json_and_preserves_nested_ts(self) -> None:
        session = {
            "target_base_time_ns": 0,
            "models": [
                {
                    "model_type": "identity",
                    "status": "PASS",
                    "source": {"hostname": "head"},
                    "segments": [],
                }
            ],
        }
        trace = {
            "baseTimeNanoseconds": 10_000,
            "traceEvents": [
                {
                    "name": "op",
                    "ph": "X",
                    "ts": 2.123,
                    "args": {"ts": 999},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_path = root / "compact.json"
            session_path = root / "session.json"
            output_path = root / "aligned.json"
            trace_path.write_text(
                json.dumps(trace, separators=(",", ":")),
                encoding="utf-8",
            )
            session_path.write_text(json.dumps(session), encoding="utf-8")

            align_trace_file(
                trace_path=trace_path,
                session_path=session_path,
                source_node="head",
                output_path=output_path,
            )
            aligned = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(aligned["baseTimeNanoseconds"], 0)
        self.assertEqual(aligned["traceEvents"][0]["ts"], 12.123)
        self.assertEqual(aligned["traceEvents"][0]["args"]["ts"], 999)

    def test_nccl_check_fails_when_an_input_rank_has_no_kernels(self) -> None:
        kernel = {
            "ph": "X",
            "cat": "kernel",
            "name": "ncclDevKernel_Generic",
            "pid": 1,
            "tid": 0,
            "ts": 1.0,
            "dur": 2.0,
            "args": {
                "stream": 0,
                "correlation": 1,
                "Collective name": "all_reduce",
                "Process Group Name": "pg",
                "Group size": 2,
                "In msg nelems": 1024,
                "Out msg nelems": 1024,
                "dtype": "float16",
                "Seq": 1,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rank_zero = root / "rank0.json"
            rank_one = root / "rank1.json"
            rank_zero.write_text(
                json.dumps(
                    {"baseTimeNanoseconds": 0, "traceEvents": [kernel]},
                    indent=2,
                ),
                encoding="utf-8",
            )
            rank_one.write_text(
                json.dumps(
                    {"baseTimeNanoseconds": 0, "traceEvents": []},
                    indent=2,
                ),
                encoding="utf-8",
            )

            report = check_nccl_traces(
                {0: rank_zero, 1: rank_one},
                uncertainty_us=5.0,
            )

        self.assertEqual(report.status, "FAIL")
        self.assertTrue(any("input ranks [1]" in note for note in report.notes))

    def test_alignment_materializes_gzip_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compressed = root / "worker_rank1.pt.trace.json.gz"
            with gzip.open(compressed, "wt", encoding="utf-8") as trace_file:
                json.dump(rank_trace(), trace_file)
            manifest = Mock()
            with patch(
                "vap.postprocess.align.base.process_traces",
                return_value=manifest,
            ) as process:
                result = align_traces(
                    {
                        1: TraceInput(
                            path=compressed,
                            source_node="192.168.0.10",
                        )
                    },
                    root / "clock-session.json",
                    root / "aligned",
                )

            materialized = process.call_args.args[0][1]
            self.assertEqual(result, manifest)
            self.assertEqual(materialized.source_node, "192.168.0.10")
            self.assertEqual(
                json.loads(materialized.path.read_text(encoding="utf-8")),
                rank_trace(),
            )

    def test_trace_rank_uses_vllm_worker_suffix(self) -> None:
        self.assertEqual(
            trace_rank("dp0_pp0_tp1_dcp0_ep0_rank7.123.pt.trace.json.gz"),
            7,
        )
        self.assertEqual(trace_rank("rank-8.aligned.json"), 8)

    def test_trace_candidates_prefer_aligned_merged_and_exclude_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligned_merged = root / "run-aligned-merged_trace.json.gz"
            raw_merged = root / "run-merged_trace.json.gz"
            raw = root / "rank0.pt.trace.json.gz"
            metadata = root / "aligned" / "manifest.json"
            intermediate = root / "aligned" / "rank-0.aligned.json"
            for path in (aligned_merged, raw_merged, raw, metadata, intermediate):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"{}")

            candidates = profile_trace_candidates(root)

        self.assertEqual(candidates, [aligned_merged, raw_merged, raw])

    def test_single_trace_is_selected_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "rank0.1.pt.trace.json"
            trace.write_text("{}", encoding="utf-8")

            selected = find_perfetto_trace(str(root))

        self.assertEqual(selected, str(trace))

    def test_fuses_json_and_gzip_ranks_with_unique_flow_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rank_zero = root / "rank0.trace.json"
            rank_one = root / "rank1.trace.json.gz"
            rank_zero.write_text(json.dumps(rank_trace()), encoding="utf-8")
            with gzip.open(rank_one, "wt", encoding="utf-8") as trace_file:
                json.dump(rank_trace(), trace_file)

            output = fuse_traces(
                [rank_zero, rank_one],
                root / "merged_trace.json",
            )
            merged = load_trace(output)["traceEvents"]

            self.assertEqual(output, str(root / "merged_trace.json.gz"))
            rank_one_launch = next(
                event
                for event in merged
                if event.get("name") == "hipLaunchKernel"
                and event.get("args", {}).get("rank") == 1
            )
            self.assertEqual(rank_one_launch["pid"], 101)
            self.assertEqual(rank_one_launch["id"], 107)
            self.assertEqual(rank_one_launch["args"]["correlation"], 105)
            self.assertEqual(rank_one_launch["args"]["pid_raw"], 1)
            self.assertEqual(rank_one_launch["args"]["id_raw"], 7)
            self.assertEqual(rank_one_launch["args"]["correlation_raw"], 5)

            names = [event.get("name") for event in merged]
            self.assertNotIn("trace_internal", names)
            self.assertNotIn("python_frame", names)
            self.assertFalse(
                any(event.get("args", {}).get("name") == "ORIGINAL" for event in merged)
            )

            labels = {
                event["args"]["name"]
                for event in merged
                if event.get("name") == "process_name"
            }
            self.assertEqual(
                labels,
                {
                    "RANK 0 - CPU",
                    "RANK 0 - GPU",
                    "RANK 1 - CPU",
                    "RANK 1 - GPU",
                },
            )

    def test_fuse_offsets_use_maximums_from_every_rank(self) -> None:
        def trace(pid: int, event_id: int, correlation: int) -> dict:
            return {
                "traceEvents": [
                    {
                        "name": "hipLaunchKernel",
                        "cat": "cuda_runtime",
                        "ph": "X",
                        "pid": pid,
                        "id": event_id,
                        "args": {"correlation": correlation},
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = [(9, 9, 9), (150, 150, 150), (50, 50, 50)]
            paths = []
            for rank, value in enumerate(values):
                path = root / f"rank{rank}.trace.json"
                path.write_text(json.dumps(trace(*value)), encoding="utf-8")
                paths.append(path)

            output = fuse_traces(paths, root / "merged.json")
            events = [
                event
                for event in load_trace(output)["traceEvents"]
                if event.get("ph") != "M"
            ]

        self.assertEqual(len({event["pid"] for event in events}), 3)
        self.assertEqual(len({event["id"] for event in events}), 3)
        self.assertEqual(
            len({event["args"]["correlation"] for event in events}),
            3,
        )

    def test_uses_external_id_when_no_runtime_launch_is_present(self) -> None:
        trace = {
            "traceEvents": [
                {
                    "name": "op",
                    "cat": "cpu_op",
                    "ph": "X",
                    "pid": 1,
                    "id": 1,
                    "args": {"External id": 9},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "rank0.json", root / "rank1.json"]
            for path in paths:
                path.write_text(json.dumps(trace), encoding="utf-8")

            output = fuse_traces(paths, root / "merged.json")
            events = [
                event
                for event in load_trace(output)["traceEvents"]
                if event.get("ph") != "M"
            ]

            self.assertEqual(events[1]["args"]["External id"], 109)
            self.assertEqual(events[1]["args"]["External id_raw"], 9)

    def test_rejects_trace_without_event_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text('{"notTraceEvents": []}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "traceEvents"):
                load_trace(path)


if __name__ == "__main__":
    unittest.main()
