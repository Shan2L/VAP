"""VAP adapter for clock-model trace alignment."""

from __future__ import annotations

import gzip
import shutil
from collections.abc import Mapping
from pathlib import Path

from vap.clock_probe import ProcessManifest, TraceInput, process_traces


def align_traces(
    trace_files: Mapping[int, TraceInput],
    session: str | Path,
    output_dir: str | Path,
    *,
    apply_clc_on_warning: bool = False,
) -> ProcessManifest:
    """Materialize gzip traces and run the clock-probe alignment pipeline."""
    root = Path(output_dir)
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    materialized: dict[int, TraceInput] = {}
    for rank, trace in sorted(trace_files.items()):
        source = trace.path.expanduser().resolve()
        if source.name.endswith(".gz"):
            destination = raw_dir / f"rank-{rank}.trace.json"
            temporary = destination.with_name(f".{destination.name}.tmp")
            try:
                with (
                    gzip.open(source, "rb") as compressed,
                    temporary.open("wb") as output,
                ):
                    shutil.copyfileobj(compressed, output)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            source = destination
        materialized[rank] = TraceInput(
            path=source,
            source_node=trace.source_node,
            boot_id=trace.boot_id,
        )

    return process_traces(
        materialized,
        session,
        root,
        apply_clc_on_warning=apply_clc_on_warning,
    )
