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
    """Merge rank traces and save a gzip-compressed Perfetto JSON trace."""
    rank_to_path = _normalize_trace_files(trace_files)
    first_events = load_trace(next(iter(rank_to_path.values())))["traceEvents"]
    linking_key = _detect_linking_key(first_events)
    offset_multipliers = _calculate_offset_multipliers(
        first_events, ("id", "pid", linking_key)
    )

    merged_events: list[dict[str, Any]] = []
    for rank, path in rank_to_path.items():
        events = load_trace(path)["traceEvents"]
        merged_events.extend(
            _process_rank_events(
                rank,
                events,
                linking_key=linking_key,
                offset_multipliers=offset_multipliers,
            )
        )

    merged_events.extend(_generate_rank_metadata(merged_events))
    destination = Path(f"{output_file}.gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as output:
            json.dump({"traceEvents": merged_events}, output, indent=4)
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


def _detect_linking_key(events: list[dict[str, Any]]) -> str:
    launch_event = next(
        (
            event
            for event in events
            if event.get("cat") in {"cuda_runtime", "cuda_driver"}
            and "launch" in str(event.get("name", "")).lower()
        ),
        None,
    )
    if launch_event and "correlation" in _event_args(launch_event):
        return "correlation"
    return "External id"


def _calculate_offset_multipliers(
    events: list[dict[str, Any]], fields: Sequence[str]
) -> dict[str, int]:
    maximums: defaultdict[str, int] = defaultdict(int)
    linking_key = fields[-1]
    for event in events:
        for field in fields:
            value = (
                _event_args(event).get(field)
                if field == linking_key
                else event.get(field)
            )
            if type(value) is int:
                maximums[field] = max(maximums[field], value)
    return {
        field: 10 ** (math.ceil(math.log10(maximum + 1)) + 1)
        for field, maximum in maximums.items()
    }


def _process_rank_events(
    rank: int,
    events: list[dict[str, Any]],
    *,
    linking_key: str,
    offset_multipliers: Mapping[str, int],
) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = []
    for raw_event in events:
        if not isinstance(raw_event, dict):
            continue
        if (
            raw_event.get("ph") == "M"
            and raw_event.get("name") in REMOVED_METADATA_NAMES
        ):
            continue
        if raw_event.get("cat") in {"Trace", "python_function"}:
            continue

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
        processed.append(event)
    return processed


def _generate_rank_metadata(
    merged_events: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    pid_to_rank: dict[int, int] = {}
    gpu_pids: set[int] = set()
    for event in merged_events:
        if event.get("ph") == "M":
            continue
        pid = event.get("pid")
        rank = _event_args(event).get("rank")
        if type(pid) is not int or type(rank) is not int:
            continue
        pid_to_rank.setdefault(pid, rank)
        if event.get("cat") in GPU_CATEGORIES:
            gpu_pids.add(pid)

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
