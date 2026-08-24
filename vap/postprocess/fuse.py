"""Fuse per-rank PyTorch traces for Perfetto visualization.

This focused implementation is adapted from AMD-AGI/TraceLens TraceFuse:
https://github.com/AMD-AGI/TraceLens/blob/2eae9b056b3db46656bda030499ec6d4e1310ea4/TraceLens/TraceFusion/trace_fuse.py

Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc.
Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vap.postprocess.json_trace import iter_trace_events

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
REMOVED_METADATA_NAMES = {
    "process_name",
    "process_sort_index",
    "process_labels",
}


def load_trace(path: str | Path) -> dict[str, Any]:
    trace_path = Path(path)
    if trace_path.name.endswith(".json.gz"):
        with gzip.open(trace_path, "rt", encoding="utf-8") as trace_file:
            data = json.load(trace_file)
    elif trace_path.suffix == ".json":
        with trace_path.open("r", encoding="utf-8") as trace_file:
            data = json.load(trace_file)
    else:
        raise ValueError(f"Unsupported trace format: {trace_path}")

    if not isinstance(data, dict) or not isinstance(data.get("traceEvents"), list):
        raise ValueError(f"Trace must contain a traceEvents array: {trace_path}")
    return data


def fuse_traces(
    trace_files: Sequence[str | Path] | Mapping[int, str | Path],
    output_file: str | Path,
) -> str:
    """Stream rank traces into one gzip-compressed Perfetto JSON trace."""
    rank_to_path = _normalize_trace_files(trace_files)
    linking_key, offset_multipliers = _scan_offset_fields(rank_to_path)
    destination = Path(output_file)
    if destination.suffix != ".gz":
        destination = Path(f"{destination}.gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as output:
            output.write('{"traceEvents":[\n')
            first_event = True
            pid_to_rank: dict[int, int] = {}
            gpu_pids: set[int] = set()
            for rank, path in rank_to_path.items():
                for raw_event in iter_trace_events(path):
                    event = _process_rank_event(
                        rank,
                        raw_event,
                        linking_key=linking_key,
                        offset_multipliers=offset_multipliers,
                    )
                    if event is None:
                        continue
                    pid = event.get("pid")
                    if type(pid) is int:
                        pid_to_rank.setdefault(pid, rank)
                        if event.get("cat") in GPU_CATEGORIES:
                            gpu_pids.add(pid)
                    if not first_event:
                        output.write(",\n")
                    json.dump(event, output, separators=(",", ":"))
                    first_event = False

            for event in _generate_rank_metadata(pid_to_rank, gpu_pids):
                if not first_event:
                    output.write(",\n")
                json.dump(event, output, separators=(",", ":"))
                first_event = False
            output.write("\n]}\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination)


def _normalize_trace_files(
    trace_files: Sequence[str | Path] | Mapping[int, str | Path],
) -> dict[int, Path]:
    if isinstance(trace_files, Mapping):
        rank_to_path = {
            int(rank): Path(path) for rank, path in sorted(trace_files.items())
        }
    else:
        rank_to_path = {rank: Path(path) for rank, path in enumerate(trace_files)}
    if not rank_to_path:
        raise ValueError("At least one trace file is required")
    return rank_to_path


def _scan_offset_fields(
    rank_to_path: Mapping[int, Path],
) -> tuple[str, dict[str, int]]:
    maximums: defaultdict[str, int] = defaultdict(int)
    has_correlation_launch = False
    for path in rank_to_path.values():
        for event in iter_trace_events(path):
            args = _event_args(event)
            if (
                event.get("cat") in {"cuda_runtime", "cuda_driver"}
                and "launch" in str(event.get("name", "")).lower()
                and "correlation" in args
            ):
                has_correlation_launch = True
            for field in ("id", "pid"):
                value = event.get(field)
                if type(value) is int:
                    maximums[field] = max(maximums[field], value)
            for field in ("correlation", "External id"):
                value = args.get(field)
                if type(value) is int:
                    maximums[field] = max(maximums[field], value)
    linking_key = "correlation" if has_correlation_launch else "External id"
    selected_maximums = {
        field: maximums[field]
        for field in ("id", "pid", linking_key)
        if field in maximums
    }
    multipliers = {
        field: 10 ** (math.ceil(math.log10(maximum + 1)) + 1)
        for field, maximum in selected_maximums.items()
    }
    return linking_key, multipliers


def _process_rank_event(
    rank: int,
    raw_event: dict[str, Any],
    *,
    linking_key: str,
    offset_multipliers: Mapping[str, int],
) -> dict[str, Any] | None:
    if raw_event.get("ph") == "M" and raw_event.get("name") in REMOVED_METADATA_NAMES:
        return None
    if raw_event.get("cat") in {"Trace", "python_function"}:
        return None

    event = dict(raw_event)
    event["args"] = dict(_event_args(raw_event))
    event["args"]["rank"] = rank
    for field, multiplier in offset_multipliers.items():
        if field == linking_key:
            value = event["args"].get(field)
            if type(value) is int:
                event["args"][f"{field}_raw"] = value
                event["args"][field] = value + rank * multiplier
        else:
            value = event.get(field)
            if type(value) is int:
                event["args"][f"{field}_raw"] = value
                event[field] = value + rank * multiplier
    return event


def _generate_rank_metadata(
    pid_to_rank: Mapping[int, int],
    gpu_pids: set[int],
) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for pid, rank in sorted(pid_to_rank.items(), key=lambda item: (item[1], item[0])):
        label = "GPU" if pid in gpu_pids else "CPU"
        sort_index = rank * 2 + (1 if pid in gpu_pids else 0)
        metadata.extend(
            [
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": 0,
                    "args": {"name": f"RANK {rank} - {label}"},
                },
                {
                    "name": "process_sort_index",
                    "ph": "M",
                    "pid": pid,
                    "tid": 0,
                    "args": {"sort_index": sort_index},
                },
            ]
        )
    return metadata


def _event_args(event: Mapping[str, Any]) -> dict[str, Any]:
    args = event.get("args")
    return args if isinstance(args, dict) else {}
