"""Human-readable clock calibration summaries for VAP logs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def clock_summary_rows(session: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected_mode = str(
        session.get("execution", {}).get("selected_mode")
        or ("hardware" if session.get("clock_source") == "ptp_hardware" else "software")
    )
    rows = [
        (
            _hardware_row(model, selected_mode)
            if selected_mode == "hardware"
            else _software_row(model, selected_mode)
        )
        for model in session.get("models", [])
    ]
    measured = [
        row
        for row in rows
        if row["model"] != "identity"
        and any(
            row[key] is not None for key in ("bridge_us", "network_us", "end_to_end_us")
        )
    ]
    if measured:
        rows.append(
            {
                "node": "MAX",
                "mode": selected_mode.upper(),
                "model": "-",
                "parameters": "-",
                "bridge_us": _max_value(measured, "bridge_us"),
                "network_us": _max_value(measured, "network_us"),
                "end_to_end_us": _max_value(measured, "end_to_end_us"),
                "status": str(session.get("status") or "-"),
            }
        )
    return rows


def format_clock_summary(session: Mapping[str, Any]) -> str:
    execution = session.get("execution") or {}
    selected = str(
        execution.get("selected_mode")
        or ("hardware" if session.get("clock_source") == "ptp_hardware" else "software")
    )
    requested = str(execution.get("requested_mode") or selected)
    lines = [
        (
            f"Clock probe requested={requested}, selected={selected}, "
            f"session_status={session.get('status', '-')}"
        ),
        "| Node | Mode | Model | Selected parameters | Bridge us | Network us | End-to-end us | Status |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in clock_summary_rows(session):
        lines.append(
            "| {node} | {mode} | {model} | {parameters} | {bridge} | "
            "{network} | {end_to_end} | {status} |".format(
                node=_escape(row["node"]),
                mode=_escape(row["mode"]),
                model=_escape(row["model"]),
                parameters=_escape(row["parameters"]),
                bridge=_format_us(row["bridge_us"]),
                network=_format_us(row["network_us"]),
                end_to_end=_format_us(row["end_to_end_us"]),
                status=_escape(row["status"]),
            )
        )
    failures = clock_failure_reasons(session)
    if failures:
        lines.append("Clock probe FAIL reasons:")
        lines.extend(
            f"- {_one_line(node)}: {_one_line(reason)}" for node, reason in failures
        )
    elif str(session.get("status") or "").upper() == "FAIL":
        lines.append(
            "Clock probe FAIL reasons:\n"
            "- cluster: session failed but recorded no detailed reason"
        )
    return "\n".join(lines)


def clock_failure_reasons(
    session: Mapping[str, Any],
) -> list[tuple[str, str]]:
    """Return de-duplicated per-node reasons suitable for operator logs."""
    failures: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(node: Any, reasons: Any) -> None:
        node_text = str(node or "cluster")
        values = (
            reasons
            if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes))
            else [reasons]
        )
        for reason in values:
            if reason in (None, "", []):
                continue
            item = (node_text, str(reason))
            if item not in seen:
                seen.add(item)
                failures.append(item)

    for failure in session.get("failures", []) or []:
        if not isinstance(failure, Mapping):
            add("cluster", failure)
            continue
        node = failure.get("hostname")
        node_payload = failure.get("node")
        if node is None and isinstance(node_payload, Mapping):
            node = (
                node_payload.get("hostname")
                or node_payload.get("name")
                or node_payload.get("address")
                or node_payload.get("node_id")
            )
        add(node, failure.get("reasons") or failure.get("error"))

    for model in session.get("models", []) or []:
        if not isinstance(model, Mapping) or str(model.get("status")).upper() != "FAIL":
            continue
        node = _node_name(model)
        reasons = model.get("fail_reasons")
        if not reasons:
            ptp = model.get("ptp")
            if isinstance(ptp, Mapping):
                reasons = ptp.get("reasons")
        add(node, reasons or "model status is FAIL")

    add("cluster", session.get("failure"))
    return failures


def _software_row(
    model: Mapping[str, Any],
    selected_mode: str,
) -> dict[str, Any]:
    model_type = str(model.get("model_type") or "-")
    if model_type == "identity":
        parameters = "identity reference"
        bridge_us = network_us = end_to_end_us = 0.0
    else:
        selection = model.get("model_selection", {})
        config = selection.get("selected_config") or model.get("config", {})
        parameters = _parameter_text(
            config,
            (
                ("method", "model_method"),
                ("window", "window_seconds"),
                ("samples/window", "samples_per_window"),
                ("RTT slack us", "rtt_slack_us"),
                ("segment", "segment_seconds"),
            ),
        )
        bridge_us = _bridge_uncertainty(model.get("realtime_monotonic_bridge", {}))
        network_us = _segment_uncertainty(model.get("segments", []))
        end_to_end_us = _float_or_none(
            selection.get("final_score", {}).get("max_total_uncertainty_us")
        )
        if end_to_end_us is None and (bridge_us is not None or network_us is not None):
            end_to_end_us = (bridge_us or 0.0) + (network_us or 0.0)
    return {
        "node": _node_name(model),
        "mode": selected_mode.upper(),
        "model": model_type,
        "parameters": parameters,
        "bridge_us": bridge_us,
        "network_us": network_us,
        "end_to_end_us": end_to_end_us,
        "status": str(model.get("status") or "-"),
    }


def _hardware_row(
    model: Mapping[str, Any],
    selected_mode: str,
) -> dict[str, Any]:
    bridge = model.get("realtime_phc_bridge", {})
    selection = bridge.get("model_selection", {})
    selected_method = selection.get("selected_method")
    selected_parameter = selection.get("selected_parameter")
    if selected_method is not None:
        parameter_name = (
            "sample_stride" if selected_method == "interpolation" else "segment_seconds"
        )
        parameters = f"method={selected_method}, {parameter_name}={selected_parameter}"
    else:
        parameters = _parameter_text(
            model.get("config", {}),
            (
                ("method", "phc_bridge_method"),
                ("segment", "segment_seconds"),
                ("capture attempts", "capture_attempts"),
            ),
        )
    return {
        "node": _node_name(model),
        "mode": selected_mode.upper(),
        "model": str(model.get("model_type") or "-"),
        "parameters": parameters,
        "bridge_us": _float_or_none(model.get("bridge_uncertainty_us")),
        "network_us": _float_or_none(model.get("ptp_uncertainty_us")),
        "end_to_end_us": _float_or_none(model.get("uncertainty_us")),
        "status": str(model.get("status") or "-"),
    }


def _node_name(model: Mapping[str, Any]) -> str:
    source = model.get("source", {})
    return str(
        source.get("hostname")
        or source.get("ray_node_name")
        or source.get("ray_node_address")
        or source.get("ray_node_id")
        or "unknown"
    )


def _bridge_uncertainty(bridge: Mapping[str, Any]) -> float | None:
    direct = _float_or_none(bridge.get("uncertainty_us"))
    if direct is not None:
        return direct
    return _segment_uncertainty(bridge.get("segments", []))


def _segment_uncertainty(segments: Sequence[Mapping[str, Any]]) -> float | None:
    values = [
        float(segment["uncertainty_us"])
        for segment in segments
        if segment.get("status") == "PASS" and segment.get("uncertainty_us") is not None
    ]
    return max(values) if values else None


def _parameter_text(
    config: Mapping[str, Any],
    fields: Sequence[tuple[str, str]],
) -> str:
    values = [
        f"{label}={config[key]}" for label, key in fields if config.get(key) is not None
    ]
    return ", ".join(values) if values else "-"


def _max_value(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return max(values) if values else None


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _format_us(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _one_line(value: Any, limit: int = 800) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[: limit - 3]}..."
