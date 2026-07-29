from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from trace_fusion import fuse_traces, load_trace


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
