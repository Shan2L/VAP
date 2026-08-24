"""Streaming Chrome Trace JSON readers."""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

TRACE_EVENTS_START_PATTERN = re.compile(r'"traceEvents"\s*:\s*\[')
STREAM_CHUNK_SIZE = 1024 * 1024


@contextmanager
def open_trace_text(path: str | Path) -> Iterator[TextIO]:
    trace_path = Path(path)
    if trace_path.name.endswith(".json.gz"):
        with gzip.open(trace_path, "rt", encoding="utf-8") as trace_file:
            yield trace_file
    elif trace_path.suffix == ".json":
        with trace_path.open(
            "r",
            encoding="utf-8",
            buffering=STREAM_CHUNK_SIZE,
        ) as trace_file:
            yield trace_file
    else:
        raise ValueError(f"Unsupported trace format: {trace_path}")


def iter_trace_events(path: str | Path) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    with open_trace_text(path) as source:
        prefix = ""
        while True:
            chunk = source.read(STREAM_CHUNK_SIZE)
            if not chunk:
                raise ValueError(f"Trace has no traceEvents array: {path}")
            prefix += chunk
            match = TRACE_EVENTS_START_PATTERN.search(prefix)
            if match is not None:
                buffer = prefix[match.end() :]
                break

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            try:
                event, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = source.read(STREAM_CHUNK_SIZE)
                if not chunk:
                    raise ValueError(f"Invalid traceEvents array: {path}")
                buffer += chunk
                continue
            buffer = buffer[end:]
            if isinstance(event, dict):
                yield event
