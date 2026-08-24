from __future__ import annotations

import heapq
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vap.postprocess.json_trace import iter_trace_events
from vap.postprocess.trace import profile_trace_candidates
from vap.runtime_paths import APP_DIR, VAP_BIN_DIR, VAP_PERFETTO_HOME
from vap.server import settings
from vap.server.artifacts import (
    profile_archive_info,
    read_current_log_file,
    resolve_profile_archive_run_dir,
)
from vap.server.state import get_run_state_snapshot


def object_schema(
    properties: dict[str, Any] | None = None, required: list[str] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
        "additionalProperties": False,
    }


def resolve_latest_trace(args: dict[str, Any]) -> dict[str, Any]:
    preferred_name = str(args.get("preferred_name") or "merged_trace")
    raw_run_dir = args.get("run_dir")
    if isinstance(raw_run_dir, str) and raw_run_dir.strip():
        run_dir = resolve_profile_archive_run_dir(raw_run_dir)
    else:
        run_dir = resolve_profile_archive_run_dir()

    profile_dir = (run_dir / "vllm-profile").resolve()
    if not profile_dir.is_dir() or not profile_dir.is_relative_to(
        settings.LOGS_DIR.resolve()
    ):
        raise ValueError(
            "The latest run has not generated a vllm-profile directory yet"
        )

    files = profile_trace_candidates(profile_dir, preferred_name)
    if not files:
        raise ValueError("No supported trace is available yet")

    trace_path = files[0]
    stat = trace_path.stat()
    trace_name = trace_path.name.lower()
    return {
        "run_dir": str(run_dir),
        "profile_dir": str(profile_dir),
        "trace_path": str(trace_path),
        "trace_file": trace_path.name,
        "size_bytes": stat.st_size,
        "modified_at": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
        ),
        "candidate_count": len(files),
        "candidates": [path.name for path in files[:10]],
        "looks_merged": "merged_trace" in trace_name or "merge_trace" in trace_name,
    }


def inspect_latest_trace(args: dict[str, Any]) -> dict[str, Any]:
    trace_info = resolve_latest_trace(args)
    summary = summarize_trace_file(Path(trace_info["trace_path"]))
    return {
        **trace_info,
        "summary": summary,
        "diagnosis": build_trace_diagnosis(summary),
    }


PERFETTO_SQL_QUERIES: dict[str, str] = {
    "trace_overview": """
SELECT
  (SELECT COUNT(*) FROM slice) AS slice_count,
  (SELECT COUNT(*) FROM thread_track) AS thread_track_count,
  (SELECT COUNT(*) FROM process) AS process_count,
  (SELECT COUNT(*) FROM sched) AS sched_count;
""",
    "top_slices": """
SELECT
  name,
  dur / 1000000.0 AS dur_ms,
  ts / 1000000.0 AS ts_ms
FROM slice
WHERE dur > 0
ORDER BY dur DESC
LIMIT {limit};
""",
    "category_duration": """
SELECT
  COALESCE(category, 'uncategorized') AS category,
  COUNT(*) AS event_count,
  SUM(dur) / 1000000.0 AS total_dur_ms,
  AVG(dur) / 1000000.0 AS avg_dur_ms,
  MAX(dur) / 1000000.0 AS max_dur_ms
FROM slice
WHERE dur > 0
GROUP BY category
ORDER BY total_dur_ms DESC
LIMIT {limit};
""",
    "sync_events": """
SELECT
  name,
  dur / 1000000.0 AS dur_ms,
  ts / 1000000.0 AS ts_ms
FROM slice
WHERE dur > 0
  AND (
    name LIKE '%Synchronize%'
    OR name LIKE '%sync%'
    OR name LIKE '%Wait%'
    OR name LIKE '%wait%'
    OR name LIKE '%barrier%'
  )
ORDER BY dur DESC
LIMIT {limit};
""",
    "gpu_kernels": """
SELECT
  name,
  dur / 1000000.0 AS dur_ms,
  ts / 1000000.0 AS ts_ms
FROM slice
WHERE dur > 0
  AND (
    category LIKE '%kernel%'
    OR name LIKE '%Kernel%'
    OR name LIKE '%hipLaunchKernel%'
  )
ORDER BY dur DESC
LIMIT {limit};
""",
    "operator_hotspots": """
SELECT
  name,
  COUNT(*) AS calls,
  SUM(dur) / 1000000.0 AS total_dur_ms,
  AVG(dur) / 1000000.0 AS avg_dur_ms,
  MAX(dur) / 1000000.0 AS max_dur_ms
FROM slice
WHERE dur > 0
  AND (
    name LIKE 'aten::%'
    OR name LIKE 'vllm%'
    OR name LIKE '%attention%'
    OR name LIKE '%Attention%'
  )
GROUP BY name
ORDER BY total_dur_ms DESC
LIMIT {limit};
""",
    "timeline_gaps": """
SELECT
  name,
  dur / 1000000.0 AS dur_ms,
  ts / 1000000.0 AS ts_ms
FROM slice
WHERE dur > 0
  AND name LIKE '%idle%'
ORDER BY dur DESC
LIMIT {limit};
""",
}


def load_skill_queries(
    skill_dir: Path = settings.TORCHPROFILER_SKILL_DIR,
) -> dict[str, str]:
    query_file = skill_dir / "queries.yaml"
    if not query_file.is_file():
        return PERFETTO_SQL_QUERIES
    queries: dict[str, list[str]] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for raw_line in query_file.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if not raw_line.startswith(" ") and raw_line.endswith(": |"):
            if current_name is not None:
                queries[current_name] = current_lines
            current_name = raw_line.split(":", 1)[0].strip()
            current_lines = []
            continue
        if current_name is not None:
            current_lines.append(
                raw_line[2:] if raw_line.startswith("  ") else raw_line
            )
    if current_name is not None:
        queries[current_name] = current_lines
    parsed = {name: "\n".join(lines).strip() for name, lines in queries.items()}
    return parsed or PERFETTO_SQL_QUERIES


def torchprofiler_skill_workflows() -> dict[str, list[str]]:
    return {
        "overview": ["trace_overview", "category_duration", "rank_activity"],
        "sync_waits": ["sync_waits", "top_slices"],
        "operator_hotspots": ["operator_hotspots", "aten_hotspots"],
        "gpu_kernels": ["gpu_kernels", "category_duration"],
        "rank_imbalance": ["rank_activity", "rank_longest_slices"],
        "memory_copy": ["memory_copies", "category_duration"],
        "full_report": [
            "trace_overview",
            "category_duration",
            "sync_waits",
            "gpu_kernels",
            "operator_hotspots",
            "rank_activity",
            "memory_copies",
        ],
    }


def trace_processor_path() -> Path:
    candidates = [
        VAP_BIN_DIR / "trace_processor",
        APP_DIR / "bin" / "trace_processor",
        APP_DIR / "trace_processor",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise ValueError("trace_processor is not installed. Run install.sh first.")


def run_perfetto_sql(args: dict[str, Any]) -> dict[str, Any]:
    queries = load_skill_queries()
    query_name = str(args.get("query_name") or "top_slices")
    if query_name not in queries:
        raise ValueError(
            "Unsupported query_name. Use one of: " + ", ".join(sorted(queries))
        )
    limit = args.get("limit", 20)
    if not isinstance(limit, int) or limit < 1 or limit > 100:
        raise ValueError("limit must be an integer from 1 to 100")

    trace_info = inspect_latest_trace(
        {
            "preferred_name": args.get("preferred_name") or "merged_trace",
            "run_dir": args.get("run_dir"),
        }
    )
    trace_path = Path(trace_info["trace_path"])
    sql = queries[query_name].format(limit=limit)
    perfetto_home = VAP_PERFETTO_HOME
    perfetto_home.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HOME"] = str(perfetto_home)
    command = [str(trace_processor_path()), "query", str(trace_path), sql]
    try:
        completed = subprocess.run(
            command,
            cwd=str(APP_DIR),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("Perfetto SQL query timed out after 120s") from exc
    if completed.returncode != 0:
        raise ValueError(
            "Perfetto SQL query failed: "
            + (
                completed.stderr.strip()
                or completed.stdout.strip()
                or f"exit {completed.returncode}"
            )
        )
    return {
        "query_name": query_name,
        "sql": sql.strip(),
        "trace_file": trace_info["trace_file"],
        "trace_path": trace_info["trace_path"],
        "looks_merged": trace_info.get("looks_merged"),
        "stdout": completed.stdout.strip()[:12000],
        "stderr": completed.stderr.strip()[:4000],
    }


def run_torchprofiler_skill(args: dict[str, Any]) -> dict[str, Any]:
    workflow = str(args.get("workflow") or "full_report")
    workflows = torchprofiler_skill_workflows()
    if workflow not in workflows:
        raise ValueError(
            "Unsupported workflow. Use one of: " + ", ".join(sorted(workflows))
        )
    limit = args.get("limit", 20)
    results = []
    for query_name in workflows[workflow]:
        try:
            results.append(
                {
                    "query_name": query_name,
                    "ok": True,
                    "result": run_perfetto_sql(
                        {
                            "query_name": query_name,
                            "preferred_name": args.get("preferred_name")
                            or "merged_trace",
                            "run_dir": args.get("run_dir"),
                            "limit": limit,
                        }
                    ),
                }
            )
        except Exception as exc:
            results.append({"query_name": query_name, "ok": False, "message": str(exc)})
    return {
        "skill": "TorchProfilerTraceSkill",
        "workflow": workflow,
        "attribution": "Inspired by Gracker/Perfetto-Skills evidence-driven workflow design; original VAP SQL presets.",
        "results": results,
    }


def build_trace_diagnosis(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary.get("available"):
        return {
            "available": False,
            "findings": [summary.get("message", "Trace summary is unavailable")],
        }

    findings: list[str] = []
    next_steps: list[str] = []
    hypotheses: list[str] = []
    duration_categories = {
        item["category"]: item["duration_us"]
        for item in summary.get("top_categories_by_duration_us", [])
        if isinstance(item, dict)
    }
    longest = summary.get("longest_events", [])
    longest_names = [
        str(item.get("name", "")) for item in longest if isinstance(item, dict)
    ]

    if (
        duration_categories.get("cuda_runtime", 0)
        > duration_categories.get("kernel", 0) * 10
    ):
        findings.append(
            "cuda_runtime duration is much larger than kernel duration; synchronization or launch overhead may dominate the trace."
        )
        hypotheses.append(
            "Inspect hipEventSynchronize and hipLaunchKernel spans for blocking waits, serialized work, or host-side scheduling gaps."
        )
    if any("hipEventSynchronize" in name for name in longest_names):
        findings.append(
            "The longest events include hipEventSynchronize, which often points to host waiting for GPU completion or synchronization barriers."
        )
        next_steps.append(
            "In Perfetto, focus on hipEventSynchronize regions and check what GPU work precedes each wait."
        )
    if any("aten::sort" in name for name in longest_names):
        findings.append(
            "The longest CPU ops include aten::sort across ranks; sampling/top-k/sorting work may be a decode bottleneck."
        )
        hypotheses.append(
            "Review sampling settings and decode path; compare whether aten::sort aligns with token generation steps."
        )

    ranks = [
        (rank, count)
        for rank, count in summary.get("events_by_rank", [])
        if rank != "unknown"
    ]
    if ranks:
        counts = [count for _, count in ranks]
        if max(counts) - min(counts) > max(counts) * 0.10:
            findings.append(
                "Event counts differ noticeably across ranks; possible rank imbalance."
            )
        else:
            findings.append("Rank event counts look roughly balanced.")
        next_steps.append(
            "In Perfetto, compare rank lanes for idle gaps, long waits, and whether decode steps align across ranks."
        )

    next_steps.extend(
        [
            "Inspect user_annotation events to separate prefill/decode phases and request-level spans.",
            "Use TensorBoard profiler views to cross-check operator time, kernel time, memory copies, and trace step boundaries.",
            "Compare GPU kernel lanes with CPU op lanes to identify host gaps before GPU work launches.",
        ]
    )

    return {
        "available": True,
        "findings": findings[:8],
        "next_steps": next_steps[:8],
        "optimization_hypotheses": hypotheses[:8],
    }


def summarize_trace_file(trace_path: Path) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    rank_counts: Counter[str] = Counter()
    duration_by_category: defaultdict[str, float] = defaultdict(float)
    longest_heap: list[tuple[float, int, dict[str, Any]]] = []
    min_ts: float | None = None
    max_ts: float | None = None
    event_count = 0
    sequence = 0

    try:
        for event in iter_trace_events(trace_path):
            event_count += 1
            category = str(event.get("cat") or "uncategorized")
            category_counts[category] += 1
            args = event.get("args") if isinstance(event.get("args"), dict) else {}
            rank = args.get("rank", args.get("args.rank", "unknown"))
            rank_counts[str(rank)] += 1

            ts = event.get("ts")
            dur = event.get("dur")
            if isinstance(ts, (int, float)):
                min_ts = ts if min_ts is None else min(min_ts, ts)
                if isinstance(dur, (int, float)):
                    max_ts = ts + dur if max_ts is None else max(max_ts, ts + dur)
                else:
                    max_ts = ts if max_ts is None else max(max_ts, ts)
            if isinstance(dur, (int, float)):
                duration = float(dur)
                duration_by_category[category] += duration
                item = {
                    "name": str(event.get("name") or ""),
                    "category": category,
                    "duration_us": round(duration, 3),
                    "rank": str(rank),
                    "pid": event.get("pid"),
                    "tid": event.get("tid"),
                }
                entry = (duration, sequence, item)
                sequence += 1
                if len(longest_heap) < 20:
                    heapq.heappush(longest_heap, entry)
                elif duration > longest_heap[0][0]:
                    heapq.heapreplace(longest_heap, entry)
    except Exception as exc:
        return {"available": False, "message": f"Failed to parse trace JSON: {exc}"}

    longest_events = [
        item
        for _duration, _sequence, item in sorted(
            longest_heap,
            reverse=True,
        )
    ]

    return {
        "available": True,
        "event_count": event_count,
        "time_span_us": (
            round(max_ts - min_ts, 3)
            if min_ts is not None and max_ts is not None
            else None
        ),
        "top_categories_by_count": category_counts.most_common(12),
        "top_categories_by_duration_us": sorted(
            (
                {"category": category, "duration_us": round(duration, 3)}
                for category, duration in duration_by_category.items()
            ),
            key=lambda item: item["duration_us"],
            reverse=True,
        )[:12],
        "events_by_rank": rank_counts.most_common(),
        "longest_events": longest_events,
    }


def prepare_download_artifact(args: dict[str, Any]) -> dict[str, Any]:
    artifact = str(args.get("artifact") or "")
    log_names = {
        "vap_log": "vap_log.txt",
        "vllm_deploy_log": "vllm_deploy.log",
        "vllm_bench_log": "vllm_bench.log",
    }
    if artifact in log_names:
        file_name = log_names[artifact]
        log_info = read_current_log_file(file_name)
        if not log_info.get("exists"):
            raise ValueError(
                str(log_info.get("message") or f"{file_name} does not exist")
            )
        return {
            "artifact": artifact,
            "label": file_name,
            "download_url": f"/api/log-file/download?name={file_name}",
            "content_type": "text/plain",
        }
    if artifact == "trace_archive":
        run_dir = get_run_state_snapshot().get("run_dir")
        info = profile_archive_info(str(run_dir) if run_dir else None)
        query = f"?run_dir={str(run_dir)}" if run_dir else ""
        return {
            "artifact": artifact,
            "label": info["file_name"],
            "download_url": f"/api/profile/archive{query}",
            "content_type": "application/zip",
        }
    raise ValueError(
        "Unsupported artifact. Use vap_log, vllm_deploy_log, vllm_bench_log, or trace_archive."
    )
