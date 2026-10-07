"""Profile-run analysis services shared by the HTTP API and the agent:
per-layer attribution of one run, A/B comparison of two runs and the
templated, agent-written comparison report."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode

from vap.agent.runtime import AGENT_MODEL
from vap.analysis import attribution as ta
from vap.analysis import report as ar
from vap.server import settings
from vap.server.artifacts import latest_log_run_dir, read_current_log_file
from vap.server.state import get_run_state_snapshot


def get_agent_runtime():
    from vap.agent.tools import get_agent_runtime as runtime

    return runtime()


ATTRIBUTION_FILE_TYPES = {
    ".md": "text/markdown; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".json": "application/json",
    ".csv": "text/csv; charset=utf-8",
}


ATTRIBUTION_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}")


def resolve_run_dir(raw_run_dir: Any) -> Path:
    """A run directory name or path under LOGS_DIR; defaults to the current
    or latest run."""
    if isinstance(raw_run_dir, str) and raw_run_dir.strip():
        candidate = Path(raw_run_dir.strip())
        if not candidate.is_absolute():
            candidate = settings.LOGS_DIR / candidate
        run_dir = candidate.resolve()
        logs_dir = settings.LOGS_DIR.resolve()
        if (
            run_dir == logs_dir
            or not run_dir.is_relative_to(logs_dir)
            or not run_dir.is_dir()
        ):
            raise ValueError(f"Unknown run directory: {raw_run_dir}")
        return run_dir
    snapshot = get_run_state_snapshot()
    run_dir = (
        Path(snapshot["run_dir"]).resolve()
        if snapshot["run_dir"]
        else latest_log_run_dir()
    )
    if run_dir is None:
        raise ValueError("No run directory is available yet")
    return run_dir


def bounded_int_arg(
    args: dict[str, Any], name: str, default: int | None, low: int, high: int
) -> int | None:
    value = args.get(name, default)
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return value


def load_cached_attribution(
    path: Path,
    layout: ta.Layout,
    trace_files: list[Path],
    topology: ta.Topology | None = None,
) -> dict[str, Any] | None:
    try:
        if path.is_symlink() or path.stat().st_mtime < max(
            trace.stat().st_mtime for trace in trace_files
        ):
            return None
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    meta = cached.get("meta", {}) if isinstance(cached, dict) else {}
    cached_layout = {key: meta.get("layout", {}).get(key) for key in asdict(layout)}
    if (
        not isinstance(cached, dict)
        or cached.get("schema_version") != ta.SCHEMA_VERSION
        or cached_layout != asdict(layout)
        or (topology is not None and meta.get("topology") != asdict(topology))
    ):
        return None
    return cached


def run_attribution(
    run_dir: Path, args: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str], bool]:
    profile_dir = (run_dir / "vllm-profile").resolve()
    if not profile_dir.is_dir() or not profile_dir.is_relative_to(
        settings.LOGS_DIR.resolve()
    ):
        raise ValueError(f"Run {run_dir.name} has no vllm-profile directory")
    trace_files, timeline = ta.select_trace_files(profile_dir)
    if not trace_files:
        raise ValueError(f"Run {run_dir.name} has no PyTorch trace files")
    info = ta.run_model_info(run_dir)
    num_layers = bounded_int_arg(args, "num_layers", info["num_layers"], 1, 1024)
    if num_layers is None:
        raise ValueError(
            f"Cannot read num_hidden_layers for {info['model'] or run_dir.name}; pass num_layers"
        )
    layout = ta.Layout(
        num_layers,
        bounded_int_arg(args, "comms_per_layer", 2, 1, 8),
        bounded_int_arg(args, "pre_comms", 1, 0, 8),
    )
    topology = ta.Topology(**info["topology"]) if info["topology"] else None
    cached = None
    if not args.get("refresh"):
        cached = load_cached_attribution(
            run_dir / "attribution" / "attribution.json",
            layout,
            trace_files,
            topology,
        )
    if cached is not None and cached.get("meta", {}).get("timeline") != timeline:
        cached = None
    result = cached or ta.analyze(trace_files, layout, topology)
    result["meta"].update(
        timeline=timeline,
        model=info["model"],
        run_dir=run_dir.name,
        trace_dir=str(profile_dir),
        concurrency=info["concurrency"],
        bench=info["bench"],
        model_shape=info["model_shape"],
    )
    outputs = ta.write_analysis(result, run_dir / "attribution")
    return result, outputs, cached is not None


def attribution_download(run_dir: Path, file_path: str, label: str) -> dict[str, str]:
    query = urlencode({"run_dir": run_dir.name, "name": Path(file_path).name})
    return {
        "label": f"{run_dir.name}: {label}",
        "download_url": f"/api/attribution/file?{query}",
    }


def analyze_run(args: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    """Per-layer attribution of one run; returns the payload and report path."""
    validate_refresh_arg(args)
    run_dir = resolve_run_dir(args.get("run_dir"))
    result, outputs, cached = run_attribution(run_dir, args)
    payload = {
        "run": run_dir.name,
        "cached": cached,
        **ta.agent_summary(result),
        "downloads": [
            attribution_download(
                run_dir, outputs["markdown"], "layer report (Markdown)"
            ),
            attribution_download(run_dir, outputs["html"], "layer heatmap (HTML)"),
        ],
    }
    return payload, Path(outputs["markdown"])


def _compare_pair(args: dict[str, Any]) -> dict[str, Any]:
    validate_refresh_arg(args)
    base_raw, target_raw = args.get("base_run"), args.get("target_run")
    if not all(
        isinstance(value, str) and value.strip() for value in (base_raw, target_raw)
    ):
        raise ValueError("base_run and target_run are required")
    base_dir, target_dir = resolve_run_dir(base_raw), resolve_run_dir(target_raw)
    if base_dir == target_dir:
        raise ValueError("base_run and target_run must name different runs")
    base, base_outputs, base_cached = run_attribution(base_dir, args)
    target, target_outputs, target_cached = run_attribution(target_dir, args)
    comparison = ta.compare(base, target)
    stem = "compare_vs_" + re.sub(r"[^A-Za-z0-9_.-]", "_", base_dir.name)
    outputs = ta.write_compare(comparison, target_dir / "attribution", stem)
    return {
        "base_dir": base_dir,
        "target_dir": target_dir,
        "base": base,
        "target": target,
        "cached": (base_cached, target_cached),
        "comparison": comparison,
        "stem": stem,
        "outputs": outputs,
        "run_outputs": (base_outputs, target_outputs),
    }


def compare_two_runs(
    args: dict[str, Any], include_layers: bool = True
) -> tuple[dict[str, Any], Path]:
    """Compare run B (target) against run A (base) per generated token, averaged over the GPUs:
    a plain difference when both use the same GPU count, otherwise the loss
    against linear scaling."""
    pair = _compare_pair(args)
    base_dir, target_dir = pair["base_dir"], pair["target_dir"]
    outputs = pair["outputs"]
    base_outputs, target_outputs = pair["run_outputs"]
    payload = {
        "base_run": base_dir.name,
        "target_run": target_dir.name,
        "comparison": ta.compare_summary(pair["comparison"], include_layers),
        "runs": [
            {
                "run": run_dir.name,
                "cached": cached,
                **ta.agent_summary(result, include_layers=False),
            }
            for run_dir, result, cached in (
                (base_dir, pair["base"], pair["cached"][0]),
                (target_dir, pair["target"], pair["cached"][1]),
            )
        ],
        "downloads": [
            attribution_download(
                target_dir, outputs["markdown"], "comparison data (Markdown)"
            ),
            attribution_download(target_dir, outputs["csv"], "per-layer A/B (CSV)"),
            attribution_download(target_dir, outputs["json"], "comparison (JSON)"),
            attribution_download(
                base_dir, base_outputs["markdown"], "A layer report (Markdown)"
            ),
            attribution_download(
                target_dir, target_outputs["markdown"], "B layer report (Markdown)"
            ),
        ],
    }
    return payload, Path(outputs["markdown"])


def report_stream(args: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Templated A/B report: compute the comparison, let the agent explain it
    (streamed), fall back to the rule-based narrative, then write the report
    with the full data appended."""
    language = args.get("language", "zh")
    if language not in ar.LANGUAGES:
        raise ValueError("language must be one of " + ", ".join(ar.LANGUAGES))
    question = args.get("question")
    if question is not None and (not isinstance(question, str) or len(question) > 2000):
        raise ValueError("question must be a string of at most 2000 characters")
    yield {"type": "status", "message": "Analyzing traces"}
    pair = _compare_pair(args)
    comparison, target_dir = pair["comparison"], pair["target_dir"]
    data = ar.report_data(comparison, pair["base"], pair["target"])
    digest = ar.data_digest({"data": data, "question": question}, language)
    folder = target_dir / "attribution"
    stem = f"report_{language}_" + pair["stem"]
    md_path, html_path = folder / f"{stem}.md", folder / f"{stem}.html"
    sidecar = folder / f"{stem}.meta.json"
    downloads = [
        attribution_download(target_dir, str(md_path), "analysis report (Markdown)"),
        attribution_download(target_dir, str(html_path), "analysis report (HTML)"),
        attribution_download(target_dir, pair["outputs"]["csv"], "per-layer A/B (CSV)"),
        attribution_download(target_dir, pair["outputs"]["json"], "comparison (JSON)"),
    ]
    page = ar.report_data(
        comparison, pair["base"], pair["target"], include_timeline=True
    )
    yield {
        "type": "data",
        "base_run": pair["base_dir"].name,
        "target_run": target_dir.name,
        "comparison": ta.compare_summary(comparison, include_timeline=True),
        "notes": ar.chart_notes(page, language),
    }
    runtime = get_agent_runtime()
    agent_ready = runtime.status()["unlocked"]
    if not args.get("refresh"):
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = None
        # A rule-based report is only reused while the agent is still locked.
        if (
            isinstance(meta, dict)
            and meta.get("digest") == digest
            and md_path.is_file()
            and (meta.get("source") == "agent" or not agent_ready)
        ):
            yield {"type": "report", **meta, "cached": True, "downloads": downloads}
            return
    narrative, source, model = "", "rules", None
    if agent_ready:
        yield {"type": "status", "message": "Agent is writing the report"}
        parts: list[str] = []
        try:
            for chunk in runtime.stream_completion(
                ar.system_prompt(language),
                ar.user_prompt(data, question),
                purpose="analysis_report",
                max_tokens=6000,
            ):
                parts.append(chunk)
                yield {"type": "delta", "content": chunk}
            narrative = "".join(parts).strip()
            source, model = "agent", AGENT_MODEL
        except Exception as exc:
            yield {
                "type": "notice",
                "message": f"Agent failed ({exc}); using the rule-based report.",
            }
        if source == "agent" and not ar.split_sections(narrative):
            yield {
                "type": "notice",
                "message": "Agent reply did not follow the template; using the rule-based report.",
            }
            narrative, source, model = "", "rules", None
    else:
        yield {
            "type": "notice",
            "message": "Agent is locked; showing the rule-based report.",
        }
    if source == "rules":
        narrative = ar.rule_narrative(data, language)
    report = ar.assemble(comparison, narrative, language, source, model)
    folder.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report, encoding="utf-8")
    html_path.write_text(
        ta.markdown_to_html(report, ta.comparison_title(comparison)),
        encoding="utf-8",
    )
    meta = {
        "digest": digest,
        "source": source,
        "model": model,
        "language": language,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conclusion": ar.conclusion_of(narrative, language),
        "narrative": narrative,
    }
    sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    yield {"type": "report", **meta, "cached": False, "downloads": downloads}


def validate_refresh_arg(args: dict[str, Any]) -> None:
    if not isinstance(args.get("refresh", False), bool):
        raise ValueError("refresh must be a boolean")


def analyze_layer_overlap(args: dict[str, Any]) -> dict[str, Any]:
    return analyze_run(args)[0]


def compare_runs(args: dict[str, Any]) -> dict[str, Any]:
    payload = compare_two_runs(args, include_layers=False)[0]
    for phase in payload["comparison"]["phases"].values():
        phase["kernels"] = phase["kernels"][:6]
        for key in ("round", "layer_sum", "non_layer", "parts", "groups"):
            phase.pop(key, None)
        phase["stages"] = {
            side: [
                {
                    "stage": stage["stage"],
                    "ranks": stage["ranks"],
                    "layers": stage["layers"],
                    **ta._rounded(ta._pillars(stage["round"])),
                }
                for stage in stages
            ]
            for side, stages in phase["stages"].items()
        }
        phase["ranks"] = {
            side: [
                {
                    "rank": row["rank"],
                    "stage": row["stage"],
                    **ta._rounded(ta._pillars(row["values"])),
                }
                for row in rows
            ]
            for side, rows in phase["ranks"].items()
        }
    payload["runs"] = [
        {key: run[key] for key in ("run", "parallel", "quality", "findings")}
        for run in payload["runs"]
    ]
    return payload


def read_log_tail(args: dict[str, Any]) -> dict[str, Any]:
    """Agent view of a run log: the tail, optionally only matching lines."""
    max_chars = bounded_int_arg(args, "max_chars", 12000, 500, 60000)
    contains = args.get("contains")
    if contains is not None and (
        not isinstance(contains, str) or not 0 < len(contains) <= 200
    ):
        raise ValueError(
            "contains must be a non-empty string of at most 200 characters"
        )
    log = read_current_log_file(
        str(args.get("file_name") or ""), max_bytes=settings.MAX_STATUS_LOG_BYTES
    )
    if not log.get("exists"):
        return log
    text = log.pop("content")
    if contains:
        needle = contains.lower()
        matches = [line for line in text.splitlines() if needle in line.lower()]
        log["matched_lines"] = len(matches)
        text = "\n".join(matches)
    log["truncated"] = len(text) > max_chars
    if log["truncated"]:
        text = text[-max_chars:]
        text = text[text.find("\n") + 1 :]
    log["content"] = text
    return log


def attribution_file(raw_run_dir: str | None, raw_name: str | None) -> Path:
    if not raw_run_dir:
        raise ValueError("run_dir is required")
    folder = (resolve_run_dir(raw_run_dir) / "attribution").resolve()
    name = raw_name or ""
    if (
        not ATTRIBUTION_FILE_NAME.fullmatch(name)
        or Path(name).suffix not in ATTRIBUTION_FILE_TYPES
    ):
        raise ValueError("Unsupported attribution file name")
    path = folder / name
    if (
        not folder.is_relative_to(settings.LOGS_DIR.resolve())
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError("Attribution file does not exist")
    return path


def list_profile_runs(args: dict[str, Any]) -> dict[str, Any]:
    limit = bounded_int_arg(args, "limit", 20, 1, 100)
    if not settings.LOGS_DIR.is_dir():
        return {"runs": []}
    run_dirs = sorted(
        (
            path
            for path in settings.LOGS_DIR.iterdir()
            if path.is_dir() and not path.is_symlink()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    runs = []
    for run_dir in run_dirs[:limit]:
        info = ta.run_model_info(run_dir)
        profile_dir = run_dir / "vllm-profile"
        traces = ta.select_trace_files(profile_dir)[0] if profile_dir.is_dir() else []
        topology = info["topology"] or {}
        runs.append(
            {
                "run_dir": run_dir.name,
                "model": info["model"],
                "tensor_parallel": info["tensor_parallel"],
                "parallel": info["parallel"],
                "gpus": (ta.Topology(**topology).gpus if topology else None),
                "concurrency": info["concurrency"],
                "tpot_ms": info["bench"].get("tpot_ms_mean"),
                "output_tok_s": info["bench"].get("output_tok_s"),
                "rank_traces": sum(
                    1 for trace in traces if ta.rank_from_name(trace.name) is not None
                ),
                "has_layer_report": (
                    run_dir / "attribution" / "attribution_report.md"
                ).is_file(),
            }
        )
    return {"runs": runs}


LAYOUT_TOOL_PARAMETERS: dict[str, Any] = {
    "num_layers": {
        "type": "integer",
        "minimum": 1,
        "maximum": 1024,
        "description": "Decoder layers. Defaults to num_hidden_layers in the model's config.json.",
    },
    "comms_per_layer": {
        "type": "integer",
        "minimum": 1,
        "maximum": 8,
        "description": "TP collectives per decoder layer; 2 for dense attention + MLP layers.",
    },
    "pre_comms": {
        "type": "integer",
        "minimum": 0,
        "maximum": 8,
        "description": "Collectives before the first layer; 1 for a vocab-parallel embedding.",
    },
    "refresh": {
        "type": "boolean",
        "description": "Recompute instead of reusing a cached analysis.",
    },
}
