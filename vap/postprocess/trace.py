from __future__ import annotations

import glob
import logging
import os
import re
from pathlib import Path

from vap.config import VAPConfig

logger = logging.getLogger("VAP")
RANK_PATTERN = re.compile(r"(?:^|_)rank-?(?P<rank>\d+)(?=[._-]|$)")


def safe_filename_part(value: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value
    ).strip("_")


def trace_rank(path: str | Path) -> int:
    matches = RANK_PATTERN.findall(Path(path).name)
    if len(matches) != 1:
        raise ValueError(f"Trace filename must contain exactly one rank: {path}")
    return int(matches[0])


def merged_trace_output_file(profile_dir: str, config: VAPConfig) -> str:
    run_stamp = os.path.basename(os.path.dirname(profile_dir.rstrip(os.sep)))
    model_name = safe_filename_part(config.model_cfg.model_name.replace("/", "_"))
    prefix = "-".join(part for part in (run_stamp, model_name) if part)
    return os.path.join(profile_dir, f"{prefix}-merged_trace.json")


def collect_pytorch_trace_files(profile_dir: str) -> list[str]:
    patterns = ("*.pt.trace.json.gz", "*.trace.json.gz", "*.trace.json")
    traces: list[str] = []
    for pattern in patterns:
        traces.extend(glob.glob(os.path.join(profile_dir, pattern)))
    return sorted(
        trace
        for trace in dict.fromkeys(traces)
        if "merged_trace" not in os.path.basename(trace)
    )


def profile_trace_candidates(
    profile_dir: str | Path,
    preferred_name: str = "merged_trace",
) -> list[Path]:
    root = Path(profile_dir).resolve()
    preferred = preferred_name.lower()
    candidates: list[tuple[int, float, str, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        is_root_file = path.parent == root
        is_json_trace = name.endswith(
            (".pt.trace.json.gz", ".trace.json.gz", ".trace.json")
        )
        is_perfetto_trace = name.endswith(".pftrace")
        is_merged = ("merged_trace" in name or "merge_trace" in name) and name.endswith(
            (".json", ".json.gz")
        )
        is_aligned_merged = is_merged and "aligned" in name

        if is_root_file and is_aligned_merged:
            priority = 0
        elif is_root_file and is_merged:
            priority = 1
        elif (
            is_root_file
            and preferred
            and preferred in name
            and (is_json_trace or is_perfetto_trace)
        ):
            priority = 2
        elif is_root_file and is_json_trace:
            priority = 3
        elif is_root_file and is_perfetto_trace:
            priority = 4
        else:
            continue
        candidates.append((priority, -path.stat().st_mtime, name, path))
    candidates.sort(key=lambda item: item[:3])
    return [item[3] for item in candidates]


def select_profile_trace(
    profile_dir: str | Path,
    preferred_name: str = "merged_trace",
) -> Path:
    candidates = profile_trace_candidates(profile_dir, preferred_name)
    if not candidates:
        raise ValueError(f"No supported trace file found under {profile_dir}")
    return candidates[0]


def find_perfetto_trace(profile_dir: str) -> str | None:
    try:
        selected = select_profile_trace(profile_dir)
    except ValueError:
        return None
    logger.info("Perfetto will load %s", selected)
    return str(selected)
