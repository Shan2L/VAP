from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from vap.config import VAPConfig
from vap.postprocess.align import TraceInput, align_traces
from vap.postprocess.fuse import fuse_traces
from vap.postprocess.trace import (
    collect_pytorch_trace_files,
    find_perfetto_trace,
    merged_trace_output_file,
    safe_filename_part,
    trace_rank,
)
from vap.runners import DockerRunner

logger = logging.getLogger("VAP")

CONTAINER_PROFILE_DIR = "/app/VAP/log/vllm-profile"
RAW_TRACE_SUFFIXES = (
    ".pt.trace.json.gz",
    ".trace.json.gz",
    ".trace.json",
)


class TracePostprocessor:
    """Collect, validate, align, and fuse traces from a profiling run."""

    def __init__(
        self,
        config: VAPConfig,
        log_path: str,
        master_runner: DockerRunner,
        worker_runners: Sequence[DockerRunner],
    ):
        self.config = config
        self.log_path = log_path
        self.master_runner = master_runner
        self.worker_runners = list(worker_runners)
        self._worker_trace_files: dict[str, list[str]] = {}

    @property
    def profile_dir(self) -> str:
        return os.path.join(self.log_path, "vllm-profile")

    @property
    def worker_trace_files(self) -> dict[str, list[str]]:
        return {
            runner: list(paths) for runner, paths in self._worker_trace_files.items()
        }

    def wait_for_trace_files(
        self,
        *,
        timeout_sec: float = 60,
        poll_interval_sec: float = 1,
        stable_checks: int = 2,
    ) -> None:
        runners = [self.master_runner, *self.worker_runners]
        expected_ranks = self.expected_trace_ranks()
        previous: dict[str, dict[str, tuple[int, int]]] | None = None
        stable_count = 0
        last_rank_counts: dict[int, int] = {}
        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:
            snapshots = {
                runner.target.label: runner.files.directory_snapshot(
                    CONTAINER_PROFILE_DIR,
                    RAW_TRACE_SUFFIXES,
                )
                for runner in runners
            }
            rank_counts: dict[int, int] = {}
            all_nonempty = True
            for snapshot in snapshots.values():
                for name, (size, _mtime_ns) in snapshot.items():
                    if "merged_trace" in name or "merge_trace" in name:
                        continue
                    if size <= 0:
                        all_nonempty = False
                    try:
                        rank = trace_rank(name)
                    except ValueError:
                        continue
                    rank_counts[rank] = rank_counts.get(rank, 0) + 1

            complete = (
                set(rank_counts) == expected_ranks
                and all(count == 1 for count in rank_counts.values())
                and all_nonempty
            )
            if complete and snapshots == previous:
                stable_count += 1
                if stable_count >= stable_checks:
                    logger.info(
                        "All %d rank traces are finalized and stable",
                        len(expected_ranks),
                    )
                    return
            else:
                stable_count = 0
            previous = snapshots
            last_rank_counts = rank_counts
            time.sleep(poll_interval_sec)

        missing = sorted(expected_ranks - set(last_rank_counts))
        duplicates = sorted(
            rank for rank, count in last_rank_counts.items() if count > 1
        )
        raise TimeoutError(
            "Trace files did not finalize before timeout: "
            f"missing_ranks={missing}, duplicate_ranks={duplicates}, "
            f"rank_counts={last_rank_counts}"
        )

    def collect_worker_traces(self) -> dict[str, list[str]]:
        collected: dict[str, list[str]] = {}
        for worker in self.worker_runners:
            label = safe_filename_part(worker.target.label)
            destination = os.path.join(
                self.profile_dir,
                "workers",
                label,
            )
            if not worker.files.is_directory(CONTAINER_PROFILE_DIR):
                raise FileNotFoundError(
                    f"Profile directory is unavailable on {worker.target.label}: "
                    f"{CONTAINER_PROFILE_DIR}"
                )
            snapshot = worker.files.directory_snapshot(
                CONTAINER_PROFILE_DIR,
                RAW_TRACE_SUFFIXES,
            )
            trace_names = sorted(
                name
                for name in snapshot
                if "merged_trace" not in name and "merge_trace" not in name
            )
            downloaded = worker.files.download_files(
                CONTAINER_PROFILE_DIR,
                trace_names,
                destination,
            )
            traces = [str(path) for path in downloaded]
            if not traces:
                raise FileNotFoundError(
                    f"No profiling traces were produced on {worker.target.label}"
                )
            collected[worker.target.label] = traces
            logger.info(
                "Collected %d trace files from %s into %s",
                len(traces),
                worker.target.label,
                destination,
            )
        self._worker_trace_files = collected
        return self.worker_trace_files

    def align_trace_files(
        self,
        traces: dict[int, TraceInput],
        profile_dir: str,
        session_path: str | os.PathLike[str],
    ) -> dict[int, str]:
        alignment_dir = os.path.join(profile_dir, "aligned")
        manifest = align_traces(
            traces,
            str(session_path),
            alignment_dir,
            apply_clc_on_warning=(self.config.clock_probe_cfg.apply_clc_on_warning),
        )
        if manifest.primary_timeline == "clc":
            return manifest.clc
        if manifest.primary_timeline == "aligned":
            return manifest.aligned
        raise RuntimeError(
            "Clock validation did not approve an aligned primary timeline"
        )

    def prepare_single_node_trace(self, profile_dir: str) -> str | None:
        local_traces = collect_pytorch_trace_files(profile_dir)
        if not local_traces:
            return find_perfetto_trace(profile_dir)

        ranked_traces: dict[int, str] = {}
        for path in local_traces:
            rank = trace_rank(path)
            if rank in ranked_traces:
                raise RuntimeError(
                    f"Multiple trace files were produced for rank {rank}"
                )
            ranked_traces[rank] = path
        self.validate_trace_ranks(ranked_traces)
        if len(ranked_traces) == 1:
            return next(iter(ranked_traces.values()))
        return self.fuse_trace_files(
            ranked_traces,
            profile_dir,
            aligned=False,
        )

    def fuse_trace_files(
        self,
        traces: dict[int, str],
        profile_dir: str,
        *,
        aligned: bool,
    ) -> str:
        output_file = merged_trace_output_file(profile_dir, self.config)
        if aligned:
            output_file = output_file.replace(
                "-merged_trace.json",
                "-aligned-merged_trace.json",
            )
        logger.info(
            "Fusing %d %s traces",
            len(traces),
            "aligned" if aligned else "raw",
        )
        return fuse_traces(traces, output_file)

    def distributed_trace_inputs(
        self,
        profile_dir: str,
        runner_node_ids: Mapping[str, str],
        worker_trace_files: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[int, TraceInput]:
        master_node = runner_node_ids.get(self.master_runner.target.label)
        if not master_node:
            raise RuntimeError("Ray master node identity is unavailable")

        located: list[tuple[str, str]] = [
            (path, master_node) for path in collect_pytorch_trace_files(profile_dir)
        ]
        worker_traces = (
            self._worker_trace_files
            if worker_trace_files is None
            else worker_trace_files
        )
        for worker_label, paths in worker_traces.items():
            source_node = runner_node_ids.get(worker_label)
            if not source_node:
                raise RuntimeError(
                    f"Ray node identity is unavailable for {worker_label}"
                )
            located.extend((path, source_node) for path in paths)

        traces: dict[int, TraceInput] = {}
        for path, source_node in located:
            rank = trace_rank(path)
            if rank in traces:
                raise RuntimeError(
                    f"Multiple trace files were produced for rank {rank}"
                )
            traces[rank] = TraceInput(
                path=Path(path),
                source_node=source_node,
            )
        self.validate_trace_ranks(traces)
        return traces

    def validate_trace_ranks(self, traces: Mapping[int, object]) -> None:
        expected = self.expected_trace_ranks()
        actual = set(traces)
        if actual == expected:
            return
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            "Trace rank set does not match configured parallel world size: "
            f"missing={missing}, unexpected={unexpected}, "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )

    def expected_trace_ranks(self) -> set[int]:
        return set(range(self.config.parallel_world_size))
