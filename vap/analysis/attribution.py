"""Per-layer compute vs communication attribution of vLLM torch-profiler
traces for tensor- and pipeline-parallel runs.

Each rank's GPU timeline is cut into engine steps (vLLM ``execute_context_*``
annotations) and, per pipeline stage, into decoder layers (the stage's TP
all-reduce sequence). Every microsecond of a segment is assigned to exactly
one category: compute (gemm, attention, other, memcpy), communication hidden
by compute (overlap), exposed collective communication split into waiting for
peer ranks (comm_wait) and transfer (comm_xfer), exposed pipeline transfer
(pp_xfer), pipeline bubble (a stage waits for another stage) or idle.
Communicators are mapped to parallel dimensions (TP, PP, DP, EP) by the ranks
they span, so other layouts only need their collectives classified.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from vap.postprocess.fuse import load_trace

SCHEMA_VERSION = 4
HIDING = ("gemm", "attention", "other")
COMPUTE = (*HIDING, "memcpy")
COMM = ("comm_wait", "comm_xfer", "comm_unmatched")
EXPOSED = (*COMM, "pp_xfer")
CATEGORIES = (*COMPUTE, "overlap", *COMM, "pp_xfer", "bubble", "idle")
STEP_KEY = "__step__"

KERNEL_CLASSES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "comm",
        re.compile(
            r"nccl|rccl|msccl|cross_device_reduce|quickreduce"
            r"|all_?reduce|all_?gather|reduce_?scatter",
            re.IGNORECASE,
        ),
    ),
    ("memcpy", re.compile(r"memcpy|memset|copybuffer|fillbuffer", re.IGNORECASE)),
    (
        "attention",
        re.compile(r"attention|attn|fmha|flash|paged|reshape_and_cache", re.IGNORECASE),
    ),
    (
        "gemm",
        re.compile(
            r"cijk_|gemm|gemv|matmul|wvsplitk|llmm|_mm_|bmm|tensile|hipblaslt"
            r"|wmma|mfma",
            re.IGNORECASE,
        ),
    ),
)
GPU_EVENT_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_CATEGORIES = {"cuda_runtime", "cuda_driver"}
TRACE_PATTERNS = (
    "*.pt.trace.json.gz",
    "*.pt.trace.json",
    "*.trace.json.gz",
    "*.trace.json",
)
# Kineto writes ts relative to baseTimeNanoseconds; absolute epoch-us ts are ~1e15.
RELATIVE_TS_LIMIT_US = 1e12
DIM_LABELS = {
    "tp": "TP",
    "pp": "PP",
    "dp": "DP",
    "ep": "EP",
    "world": "World",
    "local": "Local",
}

_STEP_RE = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
_RANK_RE = re.compile(r"(?:^|[_.-])rank-?(\d+)(?=[_.-]|$)")


def classify_kernel(name: str) -> str:
    for label, pattern in KERNEL_CLASSES:
        if pattern.search(name):
            return label
    return "other"


def step_phase(name: str) -> str:
    match = _STEP_RE.match(name)
    if match is None:
        return "unknown"
    contexts, generations = int(match.group(1)), int(match.group(3))
    if contexts and generations:
        return "mixed"
    if contexts:
        return "prefill"
    return "decode" if generations else "empty"


def step_sequences(name: str) -> int:
    """Requests in a step: context plus generation requests."""
    match = _STEP_RE.match(name)
    return int(match.group(1)) + int(match.group(3)) if match else 0


def collective_op(name: str | None) -> str:
    """Normalized c10d collective name: _allgather_base -> all_gather."""
    if not name:
        return "all_reduce"
    text = name.lower().strip("_")
    for op in (
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "all_to_all",
        "broadcast",
        "send",
        "recv",
    ):
        if op.replace("_", "") in text.replace("_", ""):
            return op
    return text


@dataclass
class Kernel:
    start: float
    end: float
    kind: str
    name: str
    correlation: int | None
    stream: Any = None
    # c10d collectives carry their name and process group; pynccl ones do not.
    collective: str | None = None
    group: tuple[int, ...] | None = None
    nelems: int | None = None
    dim: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def op(self) -> str:
        return collective_op(self.collective)


@dataclass
class Step:
    name: str
    start: float
    end: float
    complete: bool
    kernels: list[Kernel] = field(default_factory=list)

    @property
    def phase(self) -> str:
        return step_phase(self.name)

    @property
    def comms(self) -> list[Kernel]:
        return [kernel for kernel in self.kernels if kernel.kind == "comm"]


@dataclass
class RankTrace:
    rank: int
    path: Path
    base_ns: int | None
    kernels: list[Kernel]
    steps: list[Step]
    unanchored_steps: int
    world_size: int | None = None
    compute_stream: Any = None

    def shift(self, offset_us: float) -> None:
        for kernel in self.kernels:
            kernel.start += offset_us
            kernel.end += offset_us
        for step in self.steps:
            step.start += offset_us
            step.end += offset_us


@dataclass(frozen=True)
class Topology:
    """Parallel layout of one vLLM engine; ranks are numbered as in vLLM,
    rank = (dp_index * pp + pp_index) * tp + tp_index. ep is the expert
    parallel group size (tp * dp when expert parallelism is enabled)."""

    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: int = 1

    @property
    def gpus(self) -> int:
        return self.tp * self.pp * self.dp

    @property
    def label(self) -> str:
        parts = [
            f"{name}{size}"
            for name, size in (("TP", self.tp), ("PP", self.pp), ("DP", self.dp))
            if size > 1
        ]
        text = "×".join(parts) or "TP1"
        return f"{text} EP{self.ep}" if self.ep > 1 else text

    def coords(self, rank: int) -> tuple[int, int, int]:
        return rank // (self.tp * self.pp), (rank // self.tp) % self.pp, rank % self.tp

    def stage(self, rank: int) -> int:
        return self.coords(rank)[1]

    def tp_group(self, rank: int) -> tuple[int, ...]:
        first = rank - rank % self.tp
        return tuple(range(first, first + self.tp))

    def comm_dim(self, group: tuple[int, ...] | None) -> str:
        """Parallel dimension of a communicator from the ranks it spans;
        collectives without a recorded group are vLLM's TP communicator."""
        if not group:
            return "tp"
        if len(group) == 1:
            return "local"
        coords = [self.coords(rank) for rank in group]
        varies = tuple(len({c[axis] for c in coords}) > 1 for axis in range(3))
        dims = {
            (False, False, True): "tp",
            (False, True, False): "pp",
            (True, False, False): "dp",
        }
        if varies in dims:
            return dims[varies]
        if self.ep > 1 and not varies[1]:
            return "ep"
        return "world"


@dataclass(frozen=True)
class Layout:
    """Collective sequence of one forward pass: pre_comms on the first
    pipeline stage, then comms_per_layer per decoder layer, then any trailing
    (post) collectives."""

    num_layers: int
    comms_per_layer: int = 2
    pre_comms: int = 1


def pp_partition(num_layers: int, pp: int) -> list[int]:
    """Decoder layers per pipeline stage, split as vLLM's get_pp_indices does."""
    counts = [num_layers // pp] * pp
    for index in range(2, num_layers % pp + 2):
        counts[-index] += 1
    return counts


@dataclass(frozen=True)
class StagePlan:
    stage: int
    first_layer: int
    num_layers: int
    pre_comms: int
    comms_per_layer: int
    prefix: str = ""

    @property
    def required_comms(self) -> int:
        return self.pre_comms + self.comms_per_layer * self.num_layers

    @property
    def layer_labels(self) -> list[str]:
        return [f"L{self.first_layer + i}" for i in range(self.num_layers)]

    def labels(self) -> list[str]:
        return [f"{self.prefix}pre", *self.layer_labels, f"{self.prefix}post"]

    def segments(
        self, step: Step, markers: Sequence[Kernel], compute_stream: Any
    ) -> tuple[list[str], list[float]]:
        if self.num_layers <= 0 or len(markers) < self.required_comms:
            return ["unsegmented"], [step.start, step.end]
        if self.pre_comms:
            pre_end = markers[self.pre_comms - 1].end
        else:
            # Later stages receive the previous stage's activations before layer 1.
            first = markers[0].start
            pre_end = max(
                (
                    kernel.end
                    for kernel in step.kernels
                    if kernel.kind == "comm"
                    and kernel.stream == compute_stream
                    and kernel.end <= first
                ),
                default=step.start,
            )
        bounds = [step.start, pre_end]
        for layer in range(self.num_layers):
            bounds.append(
                markers[self.pre_comms + self.comms_per_layer * (layer + 1) - 1].end
            )
        bounds.append(step.end)
        for index in range(1, len(bounds)):
            bounds[index] = max(bounds[index], bounds[index - 1])
        return self.labels(), bounds

    def marker_role(self, index: int) -> str:
        if index < self.pre_comms:
            return "embedding"
        if index >= self.required_comms:
            return "post"
        slot = (index - self.pre_comms) % self.comms_per_layer
        if self.comms_per_layer == 2:
            return ("attention", "mlp")[slot]
        return f"layer{slot}"


def stage_plans(
    layout: Layout, topology: Topology, typical_markers: dict[int, int]
) -> list[StagePlan]:
    counts = pp_partition(layout.num_layers, topology.pp)
    pre = [layout.pre_comms if stage == 0 else 0 for stage in range(topology.pp)]
    # The trace wins over vLLM's default split (VLLM_PP_LAYER_PARTITION).
    derived = []
    for stage in range(topology.pp):
        markers = typical_markers.get(stage)
        if markers is None:
            break
        layers, rest = divmod(markers - pre[stage], layout.comms_per_layer)
        derived.append(layers if rest in (0, 1) and layers > 0 else -1)
    if len(derived) == topology.pp and sum(derived) == layout.num_layers:
        counts = derived
    prefix = "" if topology.pp == 1 else "S{} "
    plans, first = [], 0
    for stage, count in enumerate(counts):
        plans.append(
            StagePlan(
                stage,
                first,
                count,
                pre[stage],
                layout.comms_per_layer,
                prefix.format(stage),
            )
        )
        first += count
    return plans


def is_marker(kernel: Kernel) -> bool:
    """vLLM's TP all-reduces (pynccl / custom all-reduce) cut the layers."""
    return kernel.kind == "comm" and kernel.dim == "tp" and kernel.op == "all_reduce"


def find_trace_files(trace_dir: Path) -> list[Path]:
    found: dict[Path, None] = {}
    for pattern in TRACE_PATTERNS:
        for path in sorted(trace_dir.glob(pattern)):
            if "merged_trace" not in path.name:
                found[path] = None
    return list(found)


def select_trace_files(profile_dir: Path) -> tuple[list[Path], str]:
    """Rank traces of a run: the clock-aligned set when the distributed
    alignment approved one (vllm-profile/aligned/manifest.json), otherwise
    the raw per-rank traces of the master and every worker node."""
    manifest_path = profile_dir / "aligned" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = None
    if isinstance(manifest, dict):
        timeline = manifest.get("primary_timeline")
        paths = manifest.get(timeline) if timeline in ("aligned", "clc") else None
        if (
            isinstance(paths, dict)
            and paths
            and all(str(rank).isdigit() for rank in paths)
        ):
            # The manifest stores absolute paths; re-anchor them to this folder so a
            # moved run still resolves and no path outside the run is ever read.
            files = [
                manifest_path.parent / Path(str(path)).name
                for _rank, path in sorted(paths.items(), key=lambda item: int(item[0]))
            ]
            if all(path.is_file() for path in files):
                return files, timeline
    files = find_trace_files(profile_dir)
    workers = profile_dir / "workers"
    if workers.is_dir():
        for node in sorted(path for path in workers.iterdir() if path.is_dir()):
            files += find_trace_files(node)
    return files, "raw"


def _parse_ranks(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, list):
        ranks = [item for item in value if isinstance(item, int)]
    elif isinstance(value, str):
        ranks = [int(item) for item in re.findall(r"\d+", value)]
    else:
        return None
    return tuple(ranks) or None


def read_rank_trace(path: Path, fallback_rank: int) -> RankTrace:
    data = load_trace(path)
    match = _RANK_RE.search(path.name)
    distributed = data.get("distributedInfo")
    distributed = distributed if isinstance(distributed, dict) else {}
    if match:
        rank = int(match.group(1))
    elif isinstance(distributed.get("rank"), int):
        rank = distributed["rank"]
    else:
        rank = fallback_rank

    kernels: list[Kernel] = []
    spans: list[tuple[float, float, str, Any]] = []
    launches: list[tuple[float, Any, int]] = []
    for event in data["traceEvents"]:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        ts, dur = event.get("ts"), event.get("dur")
        if not isinstance(ts, (int, float)) or not isinstance(dur, (int, float)):
            continue
        category = event.get("cat")
        name = str(event.get("name", ""))
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        correlation = args.get("correlation")
        correlation = correlation if isinstance(correlation, int) else None
        start, end = float(ts), float(ts) + float(dur)
        if category in GPU_EVENT_CATEGORIES:
            kind = classify_kernel(name) if category == "kernel" else "memcpy"
            collective = args.get("Collective name")
            nelems = args.get("In msg nelems")
            kernels.append(
                Kernel(
                    start,
                    end,
                    kind,
                    name,
                    correlation,
                    args.get("stream", event.get("tid")),
                    collective if isinstance(collective, str) and collective else None,
                    _parse_ranks(args.get("Process Group Ranks")),
                    nelems if isinstance(nelems, int) else None,
                )
            )
        elif category == "user_annotation" and _STEP_RE.match(name):
            spans.append((start, end, name, event.get("tid")))
        elif category in LAUNCH_CATEGORIES and correlation is not None:
            launches.append((start, event.get("tid"), correlation))

    kernels.sort(key=lambda kernel: kernel.start)
    steps, unanchored = _build_steps(kernels, spans, launches)
    base_ns = data.get("baseTimeNanoseconds")
    world = distributed.get("world_size")
    return RankTrace(
        rank,
        path,
        base_ns if isinstance(base_ns, int) else None,
        kernels,
        steps,
        unanchored,
        world if isinstance(world, int) else None,
    )


def _build_steps(
    kernels: Sequence[Kernel],
    spans: list[tuple[float, float, str, Any]],
    launches: Sequence[tuple[float, Any, int]],
) -> tuple[list[Step], int]:
    """A step starts at the first GPU kernel launched inside its annotation and
    ends where the next step starts, so post-forward work (sampling) and the
    inter-step GPU gap belong to the step that precedes them."""
    spans.sort(key=lambda span: span[0])
    starts_by_tid: dict[Any, list[float]] = defaultdict(list)
    index_by_tid: dict[Any, list[int]] = defaultdict(list)
    for index, (start, _end, _name, tid) in enumerate(spans):
        starts_by_tid[tid].append(start)
        index_by_tid[tid].append(index)

    span_of_correlation: dict[int, int] = {}
    for ts, tid, correlation in launches:
        starts = starts_by_tid.get(tid)
        if not starts:
            continue
        position = bisect.bisect_right(starts, ts) - 1
        if position >= 0:
            index = index_by_tid[tid][position]
            if ts <= spans[index][1]:
                span_of_correlation[correlation] = index

    anchors: dict[int, float] = {}
    for kernel in kernels:
        if kernel.correlation is None:
            continue
        index = span_of_correlation.get(kernel.correlation)
        if index is not None and kernel.start < anchors.get(index, float("inf")):
            anchors[index] = kernel.start

    ordered = sorted((anchor, index) for index, anchor in anchors.items())
    anchor_times = [anchor for anchor, _index in ordered]
    steps = [
        Step(
            spans[index][2],
            anchor,
            anchor_times[position + 1] if position + 1 < len(ordered) else anchor,
            position + 1 < len(ordered),
        )
        for position, (anchor, index) in enumerate(ordered)
    ]
    for kernel in kernels:
        position = bisect.bisect_right(anchor_times, kernel.start) - 1
        if position >= 0:
            steps[position].kernels.append(kernel)
    if steps and steps[-1].kernels:
        steps[-1].end = max(kernel.end for kernel in steps[-1].kernels)
    return steps, len(spans) - len(ordered)


def align_clocks(ranks: Sequence[RankTrace]) -> dict[str, Any]:
    first_ts = [rank.kernels[0].start for rank in ranks if rank.kernels]
    relative = bool(first_ts) and max(first_ts) < RELATIVE_TS_LIMIT_US
    bases = [rank.base_ns for rank in ranks]
    if relative and all(base is not None for base in bases):
        reference = min(base for base in bases if base is not None)
        offsets = {
            rank.rank: ((rank.base_ns or 0) - reference) / 1000.0 for rank in ranks
        }
        method = "baseTimeNanoseconds"
    else:
        offsets = {rank.rank: 0.0 for rank in ranks}
        method = "unaligned_relative_ts" if relative else "absolute_ts"
    for rank in ranks:
        if offsets[rank.rank]:
            rank.shift(offsets[rank.rank])
    return {
        "method": method,
        "offsets_us": {str(rank): offset for rank, offset in offsets.items()},
    }


def match_steps(
    ranks: Sequence[RankTrace], max_shift: int = 8
) -> tuple[list[list[Step]], dict[int, int]]:
    """Pair the same engine step across ranks (same annotation, nearest start)."""
    reference = ranks[0].steps
    shifts: dict[int, int] = {}
    for rank in ranks:
        best: tuple[float, int] | None = None
        for shift in range(-max_shift, max_shift + 1):
            diffs = [
                abs(step.start - rank.steps[index + shift].start)
                for index, step in enumerate(reference)
                if 0 <= index + shift < len(rank.steps)
                and rank.steps[index + shift].name == step.name
            ]
            if len(diffs) < max(1, len(reference) // 2):
                continue
            cost = statistics.median(diffs)
            if best is None or (cost, abs(shift)) < (best[0], abs(best[1])):
                best = (cost, shift)
        shifts[rank.rank] = best[1] if best else 0

    groups: list[list[Step]] = []
    for index, step in enumerate(reference):
        group: list[Step] = []
        for rank in ranks:
            other = index + shifts[rank.rank]
            if not 0 <= other < len(rank.steps) or rank.steps[other].name != step.name:
                break
            group.append(rank.steps[other])
        else:
            groups.append(group)
    return groups, shifts


def pp_key(kernel: Kernel) -> tuple[str, tuple[int, ...] | None]:
    op = kernel.op
    return ("p2p" if op in ("send", "recv") else op), kernel.group


def match_collectives(
    group: Sequence[Step], ranks: Sequence[RankTrace], topology: Topology
) -> tuple[dict[int, float], list[dict[str, Any]], dict[int, list[float]]]:
    """Within each communicator the k-th collective of a step is the same
    collective on every member rank. The fastest member's kernel time is the
    transfer; the extra kernel time of any other member is spent waiting for
    its peers. Durations need no cross-rank clock alignment. Returns the wait
    of every matched kernel (by id), one record per collective and every
    rank's end-time offset from the communicator's first rank."""
    queues: dict[tuple[str, tuple[int, ...]], dict[int, list[Kernel]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rank, step in zip(ranks, group):
        for kernel in step.comms:
            if kernel.dim in ("pp", "local"):
                continue
            members = kernel.group or topology.tp_group(rank.rank)
            queues[(kernel.dim, tuple(members))][rank.rank].append(kernel)
    present = {rank.rank for rank in ranks}
    waits: dict[int, float] = {}
    records: list[dict[str, Any]] = []
    end_offsets: dict[int, list[float]] = defaultdict(list)
    for (dim, members), by_rank in queues.items():
        lists = [by_rank.get(member, []) for member in members if member in present]
        if len(lists) < 2 or len({len(items) for items in lists}) != 1:
            continue
        for index in range(len(lists[0])):
            kernels = [items[index] for items in lists]
            durations = [kernel.duration for kernel in kernels]
            fastest = min(durations)
            for kernel in kernels:
                waits[id(kernel)] = kernel.duration - fastest
            first_end = kernels[0].end
            for member, kernel in zip(
                [m for m in members if m in present], kernels, strict=True
            ):
                end_offsets[member].append(kernel.end - first_end)
            records.append(
                {
                    "dim": dim,
                    "op": kernels[0].op,
                    "kernel": kernels[0],
                    "xfer_us": fastest,
                    "wait_us_mean": statistics.fmean(durations) - fastest,
                    "wait_us_max": max(durations) - fastest,
                    # No rank can finish a collective before every rank has started it.
                    "violation_us": max(
                        0.0,
                        max(k.start for k in kernels) - min(k.end for k in kernels),
                    ),
                }
            )
    return waits, records, end_offsets


def step_intervals(
    step: Step,
    waits: dict[int, float],
    pp_floor: dict[tuple[str, tuple[int, ...] | None], float],
) -> list[tuple[float, float, str]]:
    intervals: list[tuple[float, float, str]] = []
    computing = [kernel for kernel in step.kernels if kernel.kind in HIDING]
    span = (
        (min(k.start for k in computing), max(k.end for k in computing))
        if computing
        else (step.end, step.end)
    )
    for kernel in step.kernels:
        if kernel.kind != "comm":
            intervals.append((kernel.start, kernel.end, kernel.kind))
            continue
        if kernel.dim == "pp":
            # A receive spins until the peer stage sends; the fastest instance
            # of the same transfer is the transfer itself. Gaps inside the
            # stage's own forward pass are launch gaps, not a bubble.
            xfer = min(kernel.duration, pp_floor.get(pp_key(kernel), kernel.duration))
            split = kernel.end - xfer
            for low, high in (
                (kernel.start, min(split, span[0])),
                (max(kernel.start, span[1]), split),
            ):
                if high > low:
                    intervals.append((low, high, "pp_wait"))
            intervals.append((split, kernel.end, "pp_xfer"))
            continue
        if kernel.dim == "local":
            intervals.append((kernel.start, kernel.end, "comm_xfer"))
            continue
        wait = waits.get(id(kernel))
        if wait is None:
            intervals.append((kernel.start, kernel.end, "comm_unmatched"))
            continue
        # An early rank spins at the start of the kernel until its peers arrive.
        split = kernel.start + wait
        if wait > 0:
            intervals.append((kernel.start, split, "comm_wait"))
        if kernel.end > split:
            intervals.append((split, kernel.end, "comm_xfer"))
    return intervals


def attribute_interval(
    intervals: Sequence[tuple[float, float, str]], low: float, high: float
) -> dict[str, float]:
    """Sweep [low, high) and assign each instant to exactly one category."""
    events: list[tuple[float, int, str]] = []
    for start, end, kind in intervals:
        start, end = max(start, low), min(end, high)
        if end > start:
            events.append((start, 1, kind))
            events.append((end, -1, kind))
    events.sort(key=lambda event: (event[0], event[1]))
    totals = dict.fromkeys(CATEGORIES, 0.0)
    active: Counter[str] = Counter()
    cursor = low
    for time, delta, kind in events:
        if time > cursor:
            _accumulate(totals, active, time - cursor)
            cursor = time
        active[kind] += delta
    if high > cursor:
        _accumulate(totals, active, high - cursor)
    return totals


def _accumulate(totals: dict[str, float], active: Counter[str], span: float) -> None:
    comm = [kind for kind in COMM if active[kind] > 0]
    hiding = [kind for kind in HIDING if active[kind] > 0]
    if hiding:
        # A stage waiting for its peer while it computes costs nothing.
        if comm or active["pp_xfer"] > 0:
            totals["overlap"] += span
        else:
            for kind in hiding:
                totals[kind] += span / len(hiding)
        return
    if comm:
        for kind in comm:
            totals[kind] += span / len(comm)
    elif active["pp_xfer"] > 0:
        totals["pp_xfer"] += span
    elif active["memcpy"] > 0:
        totals["memcpy"] += span
    elif active["pp_wait"] > 0:
        totals["bubble"] += span
    else:
        totals["idle"] += span


def derived_metrics(values: dict[str, float]) -> dict[str, float | None]:
    total = sum(values[category] for category in CATEGORIES)
    exposed = sum(values[category] for category in EXPOSED)
    comm_busy = exposed + values["overlap"]
    compute_busy = sum(values[category] for category in COMPUTE) + values["overlap"]

    def share(value: float) -> float | None:
        return value / total if total else None

    return {
        "total_us": total,
        "compute_us": compute_busy,
        "comm_us": comm_busy,
        "overlap_us": values["overlap"],
        "exposed_comm_us": exposed,
        "bubble_us": values["bubble"],
        "idle_us": values["idle"],
        "compute_share": share(compute_busy),
        "comm_share": share(comm_busy),
        "overlap_ratio": values["overlap"] / comm_busy if comm_busy else None,
        "exposed_comm_share": share(exposed),
        "bubble_share": share(values["bubble"]),
        "idle_share": share(values["idle"]),
        "wait_fraction_of_exposed": values["comm_wait"] / exposed if exposed else None,
    }


def _mean(samples: Sequence[dict[str, float]]) -> dict[str, float]:
    return {
        category: statistics.fmean(sample[category] for sample in samples)
        for category in CATEGORIES
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _collective_summary(
    records: Sequence[dict[str, Any]], rank_steps: int
) -> list[dict[str, Any]]:
    """Per communicator dimension, operation and role; counts are per step
    and rank."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["dim"], record["op"], record["role"])].append(record)
    rows = []
    for (dim, op, role), items in grouped.items():
        xfer = [item["xfer_us"] for item in items]
        wait = [item["wait_us_mean"] for item in items]
        wait_max = [item.get("wait_us_max", 0.0) for item in items]
        members = statistics.fmean(item.get("members", 1) for item in items)
        rows.append(
            {
                "dim": dim,
                "op": op,
                "role": role,
                "count_per_step": len(items) * members / rank_steps,
                "duration_us_mean": statistics.fmean(
                    item["duration_us"] for item in items
                ),
                "xfer_us_mean": statistics.fmean(xfer),
                "xfer_us_p50": _percentile(xfer, 0.5),
                "xfer_us_p90": _percentile(xfer, 0.9),
                "wait_us_mean": statistics.fmean(wait),
                "wait_us_max_p50": _percentile(wait_max, 0.5),
                "wait_us_max_p90": _percentile(wait_max, 0.9),
                "violation_us_max": max(
                    item.get("violation_us", 0.0) for item in items
                ),
            }
        )
    rows.sort(key=_collective_order)
    return rows


ROLE_ORDER = (
    "embedding",
    "pre",
    "attention",
    "mlp",
    "layers",
    "post",
    "stage transfer",
)
OP_ORDER = (
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "send",
    "recv",
    "broadcast",
)


def _collective_order(row: dict[str, Any]) -> tuple[int, int, int, str]:
    dims = ("tp", "ep", "dp", "pp")

    def rank(items: Sequence[str], value: str) -> int:
        return items.index(value) if value in items else len(items)

    return (
        rank(dims, row["dim"]),
        rank(OP_ORDER, row["op"]),
        rank(ROLE_ORDER, row["role"]),
        row["role"],
    )


def _segment_kind(kernel: Kernel, markers: Sequence[Kernel]) -> str:
    if not markers or kernel.start < markers[0].start:
        return "pre"
    if kernel.start > markers[-1].start:
        return "post"
    return "layers"


def _replicas(ranks: Sequence[RankTrace]) -> list[list[RankTrace]]:
    """vLLM data-parallel engines each number their ranks from 0, so the
    i-th trace of a rank id belongs to replica i."""
    by_rank: dict[int, list[RankTrace]] = defaultdict(list)
    for rank in sorted(ranks, key=lambda item: item.path.name):
        by_rank[rank.rank].append(rank)
    count = max(len(items) for items in by_rank.values())
    return [
        [items[index] for _rank, items in sorted(by_rank.items()) if index < len(items)]
        for index in range(count)
    ]


def analyze(
    trace_files: Sequence[Path],
    layout: Layout,
    topology: Topology | None = None,
    violation_tolerance_us: float = 1.0,
) -> dict[str, Any]:
    loaded: list[RankTrace] = []
    skipped: list[str] = []
    for fallback, path in enumerate(trace_files):
        rank = read_rank_trace(Path(path), fallback)
        if not rank.kernels or not rank.steps:
            skipped.append(rank.path.name)
            continue
        loaded.append(rank)
    if not loaded:
        raise ValueError(
            "No trace with GPU kernels and vLLM step annotations was found"
        )
    replicas = _replicas(loaded)
    ranks = replicas[0]
    topology = topology or Topology(tp=len(ranks))
    # Each data-parallel engine is its own TP x PP world.
    engine = Topology(topology.tp, topology.pp, 1, topology.ep)
    if engine.tp * engine.pp != len(ranks):
        engine = Topology(tp=len(ranks))
    for rank in ranks:
        streams = Counter(
            kernel.stream for kernel in rank.kernels if kernel.kind in HIDING
        )
        rank.compute_stream = streams.most_common(1)[0][0] if streams else None
        for kernel in rank.kernels:
            if kernel.kind == "comm":
                kernel.dim = engine.comm_dim(kernel.group)

    alignment = align_clocks(ranks)
    groups, shifts = match_steps(ranks)
    complete = [group for group in groups if all(step.complete for step in group)]

    typical: dict[int, int] = {}
    for stage in range(engine.pp):
        counts = Counter(
            sum(1 for kernel in step.kernels if is_marker(kernel))
            for group in complete
            for rank, step in zip(ranks, group)
            if engine.stage(rank.rank) == stage
        )
        if counts:
            typical[stage] = counts.most_common(1)[0][0]
    plans = stage_plans(layout, engine, typical)

    pp_floor: dict[tuple[str, tuple[int, ...] | None], float] = {}
    for group in complete:
        for step in group:
            for kernel in step.comms:
                if kernel.dim == "pp":
                    key = pp_key(kernel)
                    pp_floor[key] = min(pp_floor.get(key, math.inf), kernel.duration)

    samples: dict[str, dict[str, dict[int, list[dict[str, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    collectives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    step_names: dict[str, Counter[str]] = defaultdict(Counter)
    sequences: dict[str, list[int]] = defaultdict(list)
    end_offsets: dict[int, list[float]] = defaultdict(list)
    markers_per_step: dict[str, Counter[int]] = defaultdict(Counter)
    phase_kernels: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0])
    )
    step_series: dict[str, list[float]] = defaultdict(list)
    counts = Counter(incomplete_steps=len(groups) - len(complete))
    for group in complete:
        phase = group[0].phase
        step_names[phase][group[0].name] += 1
        sequences[phase].append(step_sequences(group[0].name))
        waits, records, offsets = match_collectives(group, ranks, engine)
        for member, values in offsets.items():
            end_offsets[member].extend(values)
        rank_totals: list[float] = []
        step_markers: list[list[Kernel]] = []
        for rank, step in zip(ranks, group):
            stage = engine.stage(rank.rank)
            plan = plans[stage]
            markers = [kernel for kernel in step.kernels if is_marker(kernel)]
            step_markers.append(markers)
            markers_per_step[f"S{stage}"][len(markers)] += 1
            labels, bounds = plan.segments(step, markers, rank.compute_stream)
            counts[
                (
                    "unsegmented_rank_steps"
                    if labels == ["unsegmented"]
                    else "segmented_rank_steps"
                )
            ] += 1
            if any(
                kernel.dim not in ("pp", "local") and id(kernel) not in waits
                for kernel in step.comms
            ):
                counts["comm_count_mismatch_rank_steps"] += 1
            intervals = step_intervals(step, waits, pp_floor)
            step_total = dict.fromkeys(CATEGORIES, 0.0)
            for label, low, high in zip(labels, bounds, bounds[1:]):
                values = attribute_interval(intervals, low, high)
                samples[phase][label][rank.rank].append(values)
                for category in CATEGORIES:
                    step_total[category] += values[category]
            samples[phase][STEP_KEY][rank.rank].append(step_total)
            rank_totals.append(sum(step_total.values()))
            for kernel in step.kernels:
                stats = phase_kernels[phase][(kernel.kind, kernel.name)]
                stats[0] += kernel.duration
                stats[1] += 1
            for kernel in step.comms:
                if kernel.dim == "pp":
                    xfer = min(
                        kernel.duration, pp_floor.get(pp_key(kernel), kernel.duration)
                    )
                    collectives[phase].append(
                        {
                            "dim": "pp",
                            "op": kernel.op,
                            "role": "stage transfer",
                            "duration_us": kernel.duration,
                            "xfer_us": xfer,
                            "wait_us_mean": kernel.duration - xfer,
                            "wait_us_max": kernel.duration - xfer,
                        }
                    )
        position_of = {
            id(kernel): position
            for position, step in enumerate(group)
            for kernel in step.comms
        }
        marker_index = {
            id(kernel): index
            for markers in step_markers
            for index, kernel in enumerate(markers)
        }
        for record in records:
            kernel = record.pop("kernel")
            position = position_of[id(kernel)]
            owner, markers = ranks[position], step_markers[position]
            if id(kernel) in marker_index:
                role = plans[engine.stage(owner.rank)].marker_role(
                    marker_index[id(kernel)]
                )
            else:
                role = _segment_kind(kernel, markers)
            members = sum(
                1
                for rank in ranks
                if rank.rank in (kernel.group or engine.tp_group(owner.rank))
            )
            collectives[phase].append(
                {
                    **record,
                    "role": role,
                    "members": members,
                    "duration_us": record["xfer_us"] + record["wait_us_mean"],
                }
            )
        step_series[phase].append(statistics.fmean(rank_totals))

    stage_of = {rank.rank: engine.stage(rank.rank) for rank in ranks}
    order = [label for plan in plans for label in plan.labels()]
    phases: dict[str, Any] = {}
    for phase, by_label in samples.items():
        labels = [label for label in [*order, "unsegmented"] if label in by_label]
        per_rank = {
            label: {
                str(rank): _mean(values)
                for rank, values in sorted(by_label[label].items())
            }
            for label in [*labels, STEP_KEY]
        }
        mean = {
            label: _mean(list(values.values())) for label, values in per_rank.items()
        }
        stages = []
        for plan in plans:
            members = [rank for rank in ranks if stage_of[rank.rank] == plan.stage]
            step_values = _mean(
                [per_rank[STEP_KEY][str(rank.rank)] for rank in members]
            )
            stages.append(
                {
                    "stage": plan.stage,
                    "ranks": [rank.rank for rank in members],
                    "layers": [plan.first_layer, plan.first_layer + plan.num_layers],
                    "segments": [label for label in plan.labels() if label in mean],
                    "step": step_values,
                    "derived": derived_metrics(step_values),
                }
            )
        rank_steps = len(step_series[phase]) * len(ranks)
        phases[phase] = {
            "steps": len(step_series[phase]),
            "step_names": dict(step_names[phase].most_common()),
            "sequences_per_step": statistics.fmean(sequences[phase]),
            "segments": labels,
            "segment_stage": {
                label: plan.stage for plan in plans for label in plan.labels()
            },
            "mean": mean,
            "per_rank": per_rank,
            "derived": {
                label: derived_metrics(values) for label, values in mean.items()
            },
            "stages": stages,
            "collectives": _collective_summary(collectives[phase], rank_steps),
            "kernels": _phase_kernels(phase_kernels[phase], rank_steps),
            "step_us_series": [round(value, 1) for value in step_series[phase]],
        }

    all_records = [record for records in collectives.values() for record in records]
    violations = [
        record.get("violation_us", 0.0)
        for record in all_records
        if record["dim"] != "pp"
    ]
    kernel_time: dict[str, Counter[str]] = defaultdict(Counter)
    kernel_calls: dict[str, Counter[str]] = defaultdict(Counter)
    comm_streams: set[str] = set()
    compute_streams: set[str] = set()
    separate_comm_stream = False
    for rank in ranks:
        rank_comm: set[str] = set()
        rank_compute: set[str] = set()
        for kernel in rank.kernels:
            kernel_time[kernel.kind][kernel.name] += kernel.duration
            kernel_calls[kernel.kind][kernel.name] += 1
            if kernel.kind == "comm" and kernel.dim not in ("pp", "local"):
                rank_comm.add(str(kernel.stream))
            elif kernel.kind in HIDING:
                rank_compute.add(str(kernel.stream))
        separate_comm_stream |= bool(rank_comm - rank_compute)
        comm_streams |= rank_comm
        compute_streams |= rank_compute
    # Collectives end almost together on all members, so a steady end-time
    # difference is clock offset that the alignment did not remove.
    residual = {
        str(rank): statistics.median(values)
        for rank, values in sorted(end_offsets.items())
        if values
    }
    expected = {f"S{plan.stage}": plan.required_comms for plan in plans}
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "world_size": len(ranks),
            "topology": asdict(topology),
            "parallel": topology.label,
            "gpus": topology.gpus,
            "ranks": {
                str(rank.rank): {
                    "file": rank.path.name,
                    "stage": stage_of[rank.rank],
                    "tp_index": engine.coords(rank.rank)[2],
                }
                for rank in ranks
            },
            "replicas": len(replicas),
            "skipped_files": skipped,
            "layout": {
                **asdict(layout),
                "stage_layers": [
                    [plan.first_layer, plan.first_layer + plan.num_layers]
                    for plan in plans
                ],
            },
            "categories": list(CATEGORIES),
            "units": "microseconds per engine step",
        },
        "quality": {
            "alignment": alignment,
            "residual_clock_offset_us": residual,
            "residual_clock_offset_us_max_abs": max(
                (abs(value) for value in residual.values()), default=0.0
            ),
            "streams": {
                "comm": sorted(comm_streams),
                "compute": sorted(compute_streams),
                "comm_on_separate_stream": separate_comm_stream,
            },
            "expected_markers_per_step": expected,
            "markers_per_step": {
                stage: {str(count): steps for count, steps in sorted(counter.items())}
                for stage, counter in sorted(markers_per_step.items())
            },
            "step_shift_vs_rank0": {str(rank): shift for rank, shift in shifts.items()},
            "matched_steps": len(groups),
            "unanchored_steps": {
                str(rank.rank): rank.unanchored_steps for rank in ranks
            },
            **counts,
            "collectives_checked": len(violations),
            "causality_violation_us_max": max(violations, default=0.0),
            "causality_violations_over_tolerance": sum(
                1 for value in violations if value > violation_tolerance_us
            ),
            "violation_tolerance_us": violation_tolerance_us,
        },
        "kernels": {
            kind: [
                {
                    "name": name,
                    "total_us": round(total, 3),
                    "calls": kernel_calls[kind][name],
                }
                for name, total in kernel_time[kind].most_common(15)
            ]
            for kind in ("comm", "gemm", "attention", "other", "memcpy")
        },
        "phases": phases,
    }


def _phase_kernels(
    stats: dict[tuple[str, str], list[float]], rank_steps: int
) -> list[dict[str, Any]]:
    """Kernel time and calls per step and rank, largest first."""
    rows = [
        {
            "kind": kind,
            "name": name,
            "us_per_step": total / rank_steps,
            "calls_per_step": calls / rank_steps,
        }
        for (kind, name), (total, calls) in stats.items()
    ]
    return sorted(rows, key=lambda row: -row["us_per_step"])


LABELS = {
    "gemm": "GEMM",
    "attention": "Attention",
    "other": "Other compute",
    "memcpy": "Memcpy",
    "overlap": "Comm hidden by compute",
    "comm_wait": "Comm wait (exposed)",
    "comm_xfer": "Comm transfer (exposed)",
    "comm_unmatched": "Comm unmatched (exposed)",
    "pp_xfer": "PP transfer (exposed)",
    "bubble": "Pipeline bubble",
    "idle": "Idle",
}
GROUPS = (
    ("Compute", COMPUTE),
    ("Comm hidden by compute", ("overlap",)),
    ("Exposed collective comm", COMM),
    ("PP transfer", ("pp_xfer",)),
    ("Pipeline bubble", ("bubble",)),
    ("Idle", ("idle",)),
)
METHOD_LINES = (
    "Each rank's GPU timeline is cut into engine steps (vLLM `execute_context_*` "
    "annotations; a step runs from its first kernel to the next step's first kernel) "
    "and, per pipeline stage, into decoder layers by the stage's TP all-reduce "
    "sequence. `pre` ends at the embedding all-reduce (first stage) or after the "
    "received activations are gathered (later stages); `post` is the rest of the step.",
    "Every microsecond is assigned to exactly one category. Comm hidden by compute: "
    "a communication kernel transfers while a compute kernel runs. Exposed comm: "
    "communication runs and no compute does. Pipeline bubble: the only running kernel "
    "is a pipeline receive waiting for another stage. Idle: no kernel runs.",
    "Exposed collective comm is split per collective: transfer is the fastest member "
    "rank's kernel time; wait is the rest of this rank's kernel time (an early rank "
    "spins until its peers arrive). Pipeline sends and receives count their fastest "
    "instance as transfer and the rest as waiting. Neither uses cross-rank clocks.",
    "Communicators are mapped to parallel dimensions by the ranks they span "
    "(TP, PP, DP, EP). Values are per token round: the time in which every running "
    "sequence decodes one token (pipeline-parallel decode needs one step per "
    "micro-batch), averaged over the GPUs of the run.",
)


def run_header(result: dict[str, Any]) -> dict[str, Any]:
    meta = result["meta"]
    topology = meta.get("topology") or {"tp": meta["world_size"]}
    return {
        "run": meta.get("run_dir") or meta.get("trace_dir"),
        "model": meta.get("model"),
        "parallel": meta.get("parallel") or f"TP{meta['world_size']}",
        "topology": topology,
        "gpus": meta.get("gpus") or meta["world_size"],
        "concurrency": meta.get("concurrency"),
        "bench": meta.get("bench") or {},
    }


def steps_per_round(result: dict[str, Any], phase: str) -> float:
    """Engine steps in which every running sequence decodes one token: one
    step per pipeline micro-batch in flight."""
    data = result["phases"][phase]
    pp = (result["meta"].get("topology") or {}).get("pp", 1)
    sequences = data.get("sequences_per_step") or 0
    if phase != "decode" or pp <= 1 or not sequences:
        return 1.0
    concurrency = result["meta"].get("concurrency")
    if not concurrency:
        return float(pp)
    return min(float(pp), max(1.0, concurrency / sequences))


def _scaled(values: dict[str, float], factor: float) -> dict[str, float]:
    return {category: values[category] * factor for category in CATEGORIES}


def _non_layer(data: dict[str, Any]) -> dict[str, float]:
    """Per rank: the step minus the decoder layers of the rank's stage."""
    samples = []
    for rank, step in data["per_rank"][STEP_KEY].items():
        values = dict(step)
        for label in _decoder_layers(data["segments"]):
            layer = data["per_rank"].get(label, {}).get(rank)
            if layer:
                for category in CATEGORIES:
                    values[category] -= layer[category]
        samples.append(values)
    return _mean(samples)


def compare(base: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Changes from base (A) to target (B) per token round and GPU. With the
    same GPU count the change is target - base (mode "same_gpus"); otherwise it
    is measured against linear scaling, target - (gpus_A / gpus_B) x base (mode
    "scaling"). Per segment, the category changes sum to the segment change."""
    head_a, head_b = run_header(base), run_header(target)
    factor = head_a["gpus"] / head_b["gpus"]
    phases: dict[str, Any] = {}
    for phase in sorted(base["phases"].keys() & target["phases"].keys()):
        left, right = base["phases"][phase], target["phases"][phase]
        spr_a, spr_b = steps_per_round(base, phase), steps_per_round(target, phase)
        round_a = _scaled(left["mean"][STEP_KEY], spr_a)
        round_b = _scaled(right["mean"][STEP_KEY], spr_b)
        total_a, total_b = sum(round_a.values()), sum(round_b.values())
        speedup = total_a / total_b if total_b else None
        series = [
            [value * spr_a for value in left.get("step_us_series") or []],
            [value * spr_b for value in right.get("step_us_series") or []],
        ]
        noise = (
            math.sqrt(
                sum(statistics.variance(values) / len(values) for values in series)
            )
            if all(len(values) > 1 for values in series)
            else None
        )
        layers = []
        labels_a, labels_b = _decoder_layers(left["segments"]), _decoder_layers(
            right["segments"]
        )
        for label in sorted(
            set(labels_a) | set(labels_b), key=lambda text: int(text[1:])
        ):
            side_a = _scaled(left["mean"][label], spr_a) if label in labels_a else None
            side_b = _scaled(right["mean"][label], spr_b) if label in labels_b else None
            layers.append(
                {
                    "label": label,
                    "stage_a": left.get("segment_stage", {}).get(label, 0),
                    "stage_b": right.get("segment_stage", {}).get(label, 0),
                    "a": side_a,
                    "b": side_b,
                    "a_metrics": derived_metrics(side_a) if side_a else None,
                    "b_metrics": derived_metrics(side_b) if side_b else None,
                    "change_us": (
                        sum(side_b.values()) - factor * sum(side_a.values())
                        if side_a and side_b
                        else None
                    ),
                }
            )
        stages = {
            side: [
                {
                    "stage": stage["stage"],
                    "ranks": stage["ranks"],
                    "layers": stage["layers"],
                    "round": _scaled(stage["step"], spr),
                    "metrics": derived_metrics(_scaled(stage["step"], spr)),
                }
                for stage in data.get("stages", [])
            ]
            for side, data, spr in (("a", left, spr_a), ("b", right, spr_b))
        }
        phases[phase] = {
            "unit": "µs per token round per GPU",
            "steps_per_round": {"a": spr_a, "b": spr_b},
            "sequences_per_step": {
                "a": left.get("sequences_per_step"),
                "b": right.get("sequences_per_step"),
            },
            "steps": {"a": left["steps"], "b": right["steps"]},
            "round_us": {"a": total_a, "b": total_b},
            "speedup": speedup,
            "ideal_speedup": 1 / factor,
            "scaling_efficiency": speedup * factor if speedup else None,
            "round_delta_se_us": noise,
            "round": {"a": round_a, "b": round_b},
            "change": {c: round_b[c] - factor * round_a[c] for c in CATEGORIES},
            "metrics": {"a": derived_metrics(round_a), "b": derived_metrics(round_b)},
            "non_layer": {
                "a": _scaled(_non_layer(left), spr_a),
                "b": _scaled(_non_layer(right), spr_b),
            },
            "stages": stages,
            "layers": layers,
            "collectives": _collective_changes(
                left.get("collectives", []), right.get("collectives", []), spr_a, spr_b
            ),
            "kernels": _kernel_changes(
                left.get("kernels", []), right.get("kernels", []), spr_a, spr_b, factor
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "same_gpus" if head_a["gpus"] == head_b["gpus"] else "scaling",
        "factor": factor,
        "a": head_a,
        "b": head_b,
        "phases": phases,
    }


def _collective_changes(
    rows_a: Sequence[dict[str, Any]],
    rows_b: Sequence[dict[str, Any]],
    spr_a: float,
    spr_b: float,
) -> list[dict[str, Any]]:
    def per_round(row: dict[str, Any] | None, spr: float) -> dict[str, Any] | None:
        if row is None:
            return None
        count = row["count_per_step"] * spr
        return {
            "count": count,
            "duration_us": row["duration_us_mean"],
            "xfer_us": row["xfer_us_mean"],
            "wait_us": row["wait_us_mean"],
            "total_us": count * row["duration_us_mean"],
        }

    index_a = {(row["dim"], row["op"], row["role"]): row for row in rows_a}
    index_b = {(row["dim"], row["op"], row["role"]): row for row in rows_b}
    keys = sorted(
        index_a.keys() | index_b.keys(),
        key=lambda key: _collective_order(
            {"dim": key[0], "op": key[1], "role": key[2]}
        ),
    )
    return [
        {
            "dim": dim,
            "op": op,
            "role": role,
            "a": per_round(index_a.get((dim, op, role)), spr_a),
            "b": per_round(index_b.get((dim, op, role)), spr_b),
        }
        for dim, op, role in keys
    ]


def _kernel_changes(
    rows_a: Sequence[dict[str, Any]],
    rows_b: Sequence[dict[str, Any]],
    spr_a: float,
    spr_b: float,
    factor: float,
    limit: int = 30,
) -> list[dict[str, Any]]:
    index_a = {(row["kind"], row["name"]): row for row in rows_a}
    index_b = {(row["kind"], row["name"]): row for row in rows_b}
    rows = []
    for key in index_a.keys() | index_b.keys():
        before, after = index_a.get(key), index_b.get(key)
        us_a = before["us_per_step"] * spr_a if before else 0.0
        us_b = after["us_per_step"] * spr_b if after else 0.0
        rows.append(
            {
                "kind": key[0],
                "name": key[1],
                "a_us": us_a,
                "b_us": us_b,
                "change_us": us_b - factor * us_a,
                "a_calls": before["calls_per_step"] * spr_a if before else 0.0,
                "b_calls": after["calls_per_step"] * spr_b if after else 0.0,
                "status": (
                    "B only"
                    if before is None
                    else "A only" if after is None else "both"
                ),
            }
        )
    rows.sort(key=lambda row: -abs(row["change_us"]))
    return rows[:limit]


def _decoder_layers(segments: Sequence[str]) -> list[str]:
    return [label for label in segments if re.fullmatch(r"L\d+", label)]


def _ms(us: float) -> str:
    return f"{us / 1000:.2f} ms"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _group_values(values: dict[str, float]) -> dict[str, float]:
    return {name: sum(values[c] for c in members) for name, members in GROUPS}


def analysis_findings(result: dict[str, Any]) -> list[dict[str, str]]:
    """Rule-based findings of one run; every number comes from the result."""
    quality, meta = result["quality"], result["meta"]
    findings: list[dict[str, str]] = []

    def add(severity: str, text: str) -> None:
        findings.append({"severity": severity, "text": text})

    if meta["world_size"] < 2:
        add(
            "warning",
            "Only one rank has GPU kernels and step annotations; no communication can be attributed.",
        )
    if meta.get("replicas", 1) > 1:
        add(
            "warning",
            f"{meta['replicas']} data-parallel engines found; only engine 0 is analyzed (DP attribution is not validated yet).",
        )
    for phase, data in result["phases"].items():
        spr = steps_per_round(result, phase)
        step = data["mean"][STEP_KEY]
        derived = derived_metrics(_scaled(step, spr))
        total = derived["total_us"]
        if not total:
            continue
        unit = "token round" if spr != 1 else "step"
        add(
            "finding",
            f"{phase}: {_ms(total)} per {unit} ({data['steps']} steps of "
            f"{data['sequences_per_step']:.0f} sequences"
            + (f", {spr:.0f} steps per token round" if spr != 1 else "")
            + f"): compute {_pct(derived['compute_share'])}, communication {_pct(derived['comm_share'])} "
            f"of which {_pct(derived['overlap_ratio'])} hidden by compute, exposed communication "
            f"{_pct(derived['exposed_comm_share'])}, pipeline bubble {_pct(derived['bubble_share'])}, "
            f"idle {_pct(derived['idle_share'])}.",
        )
        stages = data.get("stages") or []
        if len(stages) > 1:
            busy = [
                (
                    stage["stage"],
                    (
                        stage["derived"]["total_us"]
                        - stage["step"]["bubble"]
                        - stage["step"]["idle"]
                    )
                    * spr,
                )
                for stage in stages
            ]
            slow = max(busy, key=lambda item: item[1])
            add(
                "finding",
                f"{phase}: pipeline stages are busy "
                + ", ".join(f"S{stage} {_ms(value)}" for stage, value in busy)
                + f" per token round; stage {slow[0]} is the bottleneck and the other stages wait for it "
                f"(bubble {_ms(step['bubble'] * spr)} per GPU and token round).",
            )
        tp_rows = [
            row
            for row in data.get("collectives", [])
            if row["dim"] == "tp" and row["op"] == "all_reduce"
        ]
        if tp_rows:
            count = sum(row["count_per_step"] for row in tp_rows) * spr
            mean = sum(
                row["count_per_step"] * row["duration_us_mean"] for row in tp_rows
            ) / sum(row["count_per_step"] for row in tp_rows)
            add(
                "finding",
                f"{phase}: {count:.0f} TP all-reduces per token round and GPU, {mean:.1f} µs each on average "
                f"({_ms(count * mean)} per token round)"
                + (
                    "; they run on the compute stream, so their time adds directly to the step."
                    if not quality.get("streams", {}).get("comm_on_separate_stream")
                    else "."
                ),
            )
        layers = _decoder_layers(data["segments"])
        if len(layers) >= 2:
            totals = {
                label: sum(data["mean"][label].values()) * spr for label in layers
            }
            median = statistics.median(totals.values())
            outliers = [
                label
                for label, value in totals.items()
                if abs(value - median) > 0.1 * median
            ]
            text = (
                f"{phase}: decoder layers take {min(totals.values()):.0f}-{max(totals.values()):.0f} µs "
                f"per token round (median {median:.0f} µs)"
            )
            if outliers:
                text += "; >10% from median: " + ", ".join(
                    f"{label} {totals[label]:.0f} µs" for label in outliers[:6]
                )
            add("finding", text + ".")
    if quality.get("causality_violations_over_tolerance"):
        add(
            "warning",
            f"{quality['causality_violations_over_tolerance']} of {quality['collectives_checked']} collectives end on one "
            f"rank before another rank starts them (max {quality['causality_violation_us_max']:.1f} µs): rank clocks are "
            "misaligned by at least that much.",
        )
    offset = quality.get("residual_clock_offset_us_max_abs") or 0.0
    if offset > 10:
        add(
            "warning",
            f"After {quality['alignment']['method']} alignment, rank clocks still differ by up to {offset:.1f} µs. "
            "Wait and transfer use kernel durations and are unaffected; cross-rank start times are off by that much.",
        )
    if quality.get("comm_count_mismatch_rank_steps"):
        add(
            "warning",
            f"{quality['comm_count_mismatch_rank_steps']} rank-steps have collectives that could not be matched across ranks; their time is reported as unmatched.",
        )
    if quality.get("unsegmented_rank_steps"):
        add(
            "warning",
            f"{quality['unsegmented_rank_steps']} rank-steps have fewer TP all-reduces than their stage's layers need "
            f"(expected {quality.get('expected_markers_per_step')}) and are reported as `unsegmented`; check num_layers.",
        )
    return findings


def compare_findings(result: dict[str, Any]) -> list[dict[str, str]]:
    """Rule-based explanation of a comparison; every number comes from it."""
    a, b = result["a"], result["b"]
    same = result["mode"] == "same_gpus"
    findings: list[dict[str, str]] = []

    def add(kind: str, text: str) -> None:
        findings.append({"severity": "finding", "kind": kind, "text": text})

    for phase, data in result["phases"].items():
        total_a, total_b = data["round_us"]["a"], data["round_us"]["b"]
        if not total_a or not total_b:
            continue
        change = data["change"]
        if same:
            delta = total_b - total_a
            text = (
                f"{phase}: B ({b['parallel']}) needs {_ms(total_b)} per token round vs A ({a['parallel']}) "
                f"{_ms(total_a)}: {delta / total_a:+.1%}, B is {'slower' if delta > 0 else 'faster'}"
            )
        else:
            text = (
                f"{phase}: {a['parallel']} ({a['gpus']} GPUs) → {b['parallel']} ({b['gpus']} GPUs) changes the token "
                f"round from {_ms(total_a)} to {_ms(total_b)}: speedup {data['speedup']:.2f}x vs ideal "
                f"{data['ideal_speedup']:.2f}x, scaling efficiency {_pct(data['scaling_efficiency'])}"
            )
        noise = data.get("round_delta_se_us")
        if noise:
            gap = abs(total_b - result["factor"] * total_a)
            text += (
                f"; the gap is {gap / noise:.1f}x the step-to-step standard error"
                + ("" if gap >= 2 * noise else " (within noise)")
            )
        add("headline", text + ".")
        grouped = _group_values(change)
        net = sum(grouped.values())
        ranked = sorted(grouped.items(), key=lambda item: -abs(item[1]))
        reference = (
            "per token round and GPU"
            if same
            else "against linear scaling, per token round and GPU"
        )
        add(
            "decomposition",
            f"{phase}: the {_ms(net)} change {reference} splits into "
            + ", ".join(
                f"{name} {value / 1000:+.2f} ms"
                for name, value in ranked
                if abs(value) >= 1
            )
            + ".",
        )
        spr = data["steps_per_round"]
        seqs = data["sequences_per_step"]
        if spr["a"] != spr["b"]:
            add(
                "pipeline",
                f"{phase}: A runs {spr['a']:.0f} step(s) of {seqs['a']:.0f} sequences per token round, B runs "
                f"{spr['b']:.0f} step(s) of {seqs['b']:.0f}: pipeline micro-batches make every GPU run its layers once "
                "per micro-batch, so weights are streamed from memory that many times per token.",
            )
        for side, label in (("a", "A"), ("b", "B")):
            stages = data["stages"][side]
            if len(stages) > 1:
                busy = [
                    (
                        stage["stage"],
                        stage["metrics"]["total_us"]
                        - stage["round"]["bubble"]
                        - stage["round"]["idle"],
                    )
                    for stage in stages
                ]
                slow = max(busy, key=lambda item: item[1])
                add(
                    "pipeline",
                    f"{phase}: in {label} the stages are busy "
                    + ", ".join(f"S{stage} {_ms(value)}" for stage, value in busy)
                    + f" per token round; S{slow[0]} is the bottleneck (it also runs the "
                    + (
                        "LM head, logits gather and sampling"
                        if slow[0] == len(stages) - 1
                        else "embedding"
                    )
                    + f"), so the other stages idle in a pipeline bubble of "
                    f"{_ms(data['round'][side]['bubble'])} per GPU and token round.",
                )
        compute = sum(change[c] for c in COMPUTE)
        if abs(compute) >= 0.05 * max(total_a, total_b):
            add(
                "compute",
                f"{phase}: compute per token round and GPU changes by {compute / 1000:+.2f} ms "
                f"(GEMM {change['gemm'] / 1000:+.2f}, attention {change['attention'] / 1000:+.2f}, "
                f"other {change['other'] / 1000:+.2f} ms"
                + ("" if same else ", against linear scaling")
                + ").",
            )
        reduce_rows = [
            row
            for row in data["collectives"]
            if row["dim"] == "tp"
            and row["op"] == "all_reduce"
            and row["a"]
            and row["b"]
        ]
        if reduce_rows:

            def stats(side: str) -> tuple[float, float, float]:
                count = sum(row[side]["count"] for row in reduce_rows)
                mean = (
                    sum(
                        row[side]["count"] * row[side]["duration_us"]
                        for row in reduce_rows
                    )
                    / count
                )
                wait = (
                    sum(
                        row[side]["count"] * row[side]["wait_us"] for row in reduce_rows
                    )
                    / count
                )
                return count, mean, wait

            count_a, mean_a, wait_a = stats("a")
            count_b, mean_b, wait_b = stats("b")
            add(
                "communication",
                f"{phase}: TP all-reduce per token round and GPU: A {count_a:.0f} × {mean_a:.1f} µs "
                f"({_ms(count_a * mean_a)}, wait {wait_a:.1f} µs each), B {count_b:.0f} × {mean_b:.1f} µs "
                f"({_ms(count_b * mean_b)}, wait {wait_b:.1f} µs each). Exposed collective communication changes by "
                f"{sum(change[c] for c in COMM) / 1000:+.2f} ms, hidden communication by {change['overlap'] / 1000:+.2f} ms.",
            )
        if data["round"]["b"]["pp_xfer"] or data["round"]["a"]["pp_xfer"]:
            add(
                "communication",
                f"{phase}: exposed pipeline transfer is {data['round']['a']['pp_xfer']:.0f} µs (A) vs "
                f"{data['round']['b']['pp_xfer']:.0f} µs (B) per token round; the stage hand-off itself is cheap, "
                "the cost of pipelining is the bubble.",
            )
        kernels = data.get("kernels") or []
        if kernels:
            add(
                "kernels",
                f"{phase}: kernels with the largest change per token round"
                + ("" if same else " against linear scaling")
                + ": "
                + ", ".join(
                    f"{_kernel_name(row['name'], 60)} {row['change_us']:+.0f} µs"
                    for row in kernels[:3]
                )
                + ".",
            )
        layers = [row for row in data["layers"] if row["change_us"] is not None]
        if layers:
            changes = [row["change_us"] for row in layers]
            add(
                "layers",
                f"{phase}: per decoder layer the token-round time changes by {min(changes):+.0f} to "
                f"{max(changes):+.0f} µs (median {statistics.median(changes):+.0f} µs)"
                + ("" if same else " against linear scaling")
                + ".",
            )
    return findings


def _kernel_name(name: str, limit: int = 90) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


def _int_arg(cfg: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


_BENCH_FIELDS = {
    "requests": r"Successful requests:\s+([\d.]+)",
    "output_tok_s": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "request_s": r"Request throughput \(req/s\):\s+([\d.]+)",
    "ttft_ms_mean": r"Mean TTFT \(ms\):\s+([\d.]+)",
    "tpot_ms_mean": r"Mean TPOT \(ms\):\s+([\d.]+)",
    "tpot_ms_median": r"Median TPOT \(ms\):\s+([\d.]+)",
    "itl_ms_mean": r"Mean ITL \(ms\):\s+([\d.]+)",
}


def bench_metrics(run_dir: Path) -> dict[str, float]:
    try:
        text = (run_dir / "vllm_bench.log").read_text(
            encoding="utf-8", errors="replace"
        )[-200_000:]
    except OSError:
        return {}
    metrics = {}
    for key, pattern in _BENCH_FIELDS.items():
        matches = re.findall(pattern, text)
        if matches:
            metrics[key] = float(matches[-1])
    return metrics


def run_model_info(run_dir: Path) -> dict[str, Any]:
    """Model, parallel layout, decode concurrency and decoder layer count from
    a VAP run's config.json, the model's Hugging Face config.json and the
    benchmark log."""
    info: dict[str, Any] = {
        "model": None,
        "tensor_parallel": None,
        "topology": None,
        "parallel": None,
        "num_layers": None,
        "concurrency": None,
        "bench": bench_metrics(run_dir),
    }
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        model_cfg = config["model_cfg"]
        info["model"] = model_cfg["model_name"]
        deploy = config.get("vllm_deploy_cfg") or {}
        bench = config.get("vllm_bench_cfg") or {}
        tp = _int_arg(deploy, "-tp", "--tensor-parallel-size") or 1
        pp = _int_arg(deploy, "-pp", "--pipeline-parallel-size") or 1
        dp = _int_arg(deploy, "-dp", "--data-parallel-size") or 1
        ep = tp * dp if "--enable-expert-parallel" in deploy or "-ep" in deploy else 1
        topology = Topology(tp, pp, dp, ep)
        info["tensor_parallel"] = tp
        info["topology"] = asdict(topology)
        info["parallel"] = topology.label
        limits = [
            _int_arg(bench, "--max-concurrency"),
            _int_arg(bench, "--num-prompts"),
            _int_arg(deploy, "--max-num-seqs"),
        ]
        known = [value for value in limits if value]
        info["concurrency"] = min(known) if known else None
        model_dir = Path(model_cfg["model_path"]) / model_cfg["model_name"]
        hf_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        return info
    for section in (
        hf_config,
        hf_config.get("text_config"),
        hf_config.get("llm_config"),
    ):
        layers = section.get("num_hidden_layers") if isinstance(section, dict) else None
        if isinstance(layers, int) and layers > 0:
            info["num_layers"] = layers
            break
    return info


def rank_from_name(name: str) -> int | None:
    match = _RANK_RE.search(name)
    return int(match.group(1)) if match else None


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value).replace("|", "\\|")


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|"
        + "|".join("---" if index == 0 else "---:" for index in range(len(headers)))
        + "|",
    ]
    lines += ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return [*lines, ""]


def _share(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}"


def layer_row(label: str, values: dict[str, float]) -> list[Any]:
    """Wall, compute, comm, compute %, comm %, overlap %, overlap µs, exposed µs,
    exposed %, bubble, idle of one segment."""
    m = derived_metrics(values)
    return [
        label,
        m["total_us"],
        m["compute_us"],
        m["comm_us"],
        _share(m["compute_share"]),
        _share(m["comm_share"]),
        _share(m["overlap_ratio"]),
        m["overlap_us"],
        m["exposed_comm_us"],
        _share(m["exposed_comm_share"]),
        m["bubble_us"],
        m["idle_us"],
    ]


LAYER_HEADERS = [
    "Segment",
    "Wall µs",
    "Compute µs",
    "Comm µs",
    "Compute %",
    "Comm %",
    "Overlap %",
    "Overlap µs",
    "Exposed µs",
    "Exposed %",
    "Bubble µs",
    "Idle µs",
]


def render_analysis_markdown(result: dict[str, Any]) -> str:
    meta, quality = result["meta"], result["quality"]
    lines = [
        f"# Compute vs communication: {meta.get('model') or 'unknown model'}, {meta.get('parallel') or 'TP' + str(meta['world_size'])}",
        "",
    ]
    source = meta.get("run_dir") or meta.get("trace_dir")
    if source:
        lines += [f"Source: `{source}`", ""]
    lines += ["## Findings", ""]
    for item in analysis_findings(result):
        prefix = "**Warning:** " if item["severity"] == "warning" else ""
        lines.append(f"- {prefix}{item['text']}")
    for phase, data in result["phases"].items():
        spr = steps_per_round(result, phase)
        unit = "per token round" if spr != 1 else "per step"
        step = _scaled(data["mean"][STEP_KEY], spr)
        total = sum(step.values())
        lines += [
            "",
            f"## Phase `{phase}`: {data['steps']} steps, {data['sequences_per_step']:.0f} sequences each",
            "",
        ]
        lines += [f"### Time breakdown (µs {unit}, mean over GPUs)", ""]
        lines += _md_table(
            ["Category", "µs", "Share %"],
            [
                (LABELS[c], step[c], _share(step[c] / total if total else None))
                for c in CATEGORIES
            ],
        )
        stages = data.get("stages") or []
        if len(stages) > 1:
            lines += [
                f"### Pipeline stages (µs {unit}, mean over the stage's GPUs)",
                "",
            ]
            lines += _md_table(
                ["Stage", "Ranks", "Layers", *LAYER_HEADERS[1:]],
                [
                    [
                        f"S{stage['stage']}",
                        ",".join(map(str, stage["ranks"])),
                        f"L{stage['layers'][0]}-L{stage['layers'][1] - 1}",
                        *layer_row("", _scaled(stage["step"], spr))[1:],
                    ]
                    for stage in stages
                ],
            )
        lines += [f"### Per segment (µs {unit}, mean over the GPUs that run it)", ""]
        lines += _md_table(
            LAYER_HEADERS,
            [
                layer_row(label, _scaled(data["mean"][label], spr))
                for label in data["segments"]
            ],
        )
        rows = data.get("collectives") or []
        if rows:
            lines += [f"### Communication (per GPU and {unit.split(' ', 1)[1]})", ""]
            lines += _md_table(
                ["Dim", "Op", "Role", "Count", "Mean µs", "Transfer µs", "Wait µs"],
                [
                    [
                        DIM_LABELS.get(row["dim"], row["dim"]),
                        row["op"],
                        row["role"],
                        f"{row['count_per_step'] * spr:.1f}",
                        row["duration_us_mean"],
                        row["xfer_us_mean"],
                        row["wait_us_mean"],
                    ]
                    for row in rows
                ],
            )
        kernels = data.get("kernels") or []
        if kernels:
            lines += [f"### Top kernels (µs per GPU and {unit.split(' ', 1)[1]})", ""]
            lines += _md_table(
                ["Kind", "Kernel", "µs", "Calls"],
                [
                    [
                        row["kind"],
                        f"`{_kernel_name(row['name'], 80)}`",
                        row["us_per_step"] * spr,
                        f"{row['calls_per_step'] * spr:.1f}",
                    ]
                    for row in kernels[:15]
                ],
            )
    lines += ["## Data quality", ""]
    lines += _md_table(["Check", "Value"], _quality_rows(result))
    lines += ["## Method", "", *(f"- {line}" for line in METHOD_LINES), ""]
    return "\n".join(lines)


def _quality_rows(result: dict[str, Any]) -> list[tuple[str, Any]]:
    meta, quality = result["meta"], result["quality"]
    return [
        ("Parallel layout", meta.get("parallel")),
        (
            "Ranks",
            ", ".join(
                f"{rank} (S{info['stage']})" for rank, info in meta["ranks"].items()
            ),
        ),
        ("Steps matched across ranks", quality["matched_steps"]),
        ("Incomplete steps dropped", quality.get("incomplete_steps", 0)),
        ("Clock alignment", quality["alignment"]["method"]),
        (
            "Residual clock offset (max)",
            f"{quality['residual_clock_offset_us_max_abs']:.1f} µs",
        ),
        (
            "TP all-reduces per step (expected)",
            json.dumps(quality.get("expected_markers_per_step")),
        ),
        ("TP all-reduces per step (seen)", json.dumps(quality.get("markers_per_step"))),
        ("Unsegmented rank-steps", quality.get("unsegmented_rank_steps", 0)),
        (
            "Rank-steps with unmatched collectives",
            quality.get("comm_count_mismatch_rank_steps", 0),
        ),
        (
            "Causality violations over tolerance",
            quality.get("causality_violations_over_tolerance", 0),
        ),
        (
            "Comm on its own stream",
            quality.get("streams", {}).get("comm_on_separate_stream"),
        ),
    ]


def comparison_layer_tables(data: dict[str, Any]) -> list[str]:
    time_rows, share_rows = [], []
    for row in data["layers"]:
        a, b = row["a_metrics"], row["b_metrics"]

        def pick(metrics: dict[str, Any] | None, key: str) -> Any:
            return None if metrics is None else metrics[key]

        stage = f"S{row['stage_a']}/S{row['stage_b']}"
        time_rows.append(
            [
                row["label"],
                stage,
                pick(a, "total_us"),
                pick(b, "total_us"),
                row["change_us"],
                pick(a, "compute_us"),
                pick(b, "compute_us"),
                pick(a, "comm_us"),
                pick(b, "comm_us"),
                pick(a, "overlap_us"),
                pick(b, "overlap_us"),
                pick(a, "exposed_comm_us"),
                pick(b, "exposed_comm_us"),
            ]
        )
        share_rows.append(
            [
                row["label"],
                *(
                    _share(pick(metrics, key))
                    for key in (
                        "compute_share",
                        "comm_share",
                        "overlap_ratio",
                        "exposed_comm_share",
                    )
                    for metrics in (a, b)
                ),
            ]
        )
    lines = ["#### Time per layer (µs per token round)", ""]
    lines += _md_table(
        [
            "Layer",
            "Stage A/B",
            "Wall A",
            "Wall B",
            "Δ",
            "Compute A",
            "Compute B",
            "Comm A",
            "Comm B",
            "Overlap A",
            "Overlap B",
            "Exposed A",
            "Exposed B",
        ],
        time_rows,
    )
    lines += ["#### Shares per layer (%)", ""]
    lines += _md_table(
        [
            "Layer",
            "Compute % A",
            "Compute % B",
            "Comm % A",
            "Comm % B",
            "Overlap % A",
            "Overlap % B",
            "Exposed % A",
            "Exposed % B",
        ],
        share_rows,
    )
    return lines


def comparison_title(result: dict[str, Any]) -> str:
    a, b = result["a"], result["b"]
    model = b.get("model") or a.get("model") or "unknown model"
    return f"{a['parallel']} (A) vs {b['parallel']} (B), {model}"


def comparison_appendix(result: dict[str, Any]) -> list[str]:
    """Every number of a comparison as Markdown tables."""
    a, b = result["a"], result["b"]
    same = result["mode"] == "same_gpus"
    lines = ["## Runs", ""]
    bench_keys = (
        ("tpot_ms_mean", "Bench mean TPOT ms"),
        ("ttft_ms_mean", "Bench mean TTFT ms"),
        ("output_tok_s", "Bench output tok/s"),
    )
    lines += _md_table(
        ["", "A", "B"],
        [
            ("Run", f"`{a['run']}`", f"`{b['run']}`"),
            ("Parallel layout", a["parallel"], b["parallel"]),
            ("GPUs", a["gpus"], b["gpus"]),
            ("Concurrent sequences", a.get("concurrency"), b.get("concurrency")),
            *(
                (label, a["bench"].get(key), b["bench"].get(key))
                for key, label in bench_keys
            ),
        ],
    )
    for phase, data in result["phases"].items():
        change_label = "Δ (B − A)" if same else f"Loss (B − {result['factor']:.3g}·A)"
        lines += [f"## Phase `{phase}` (µs per token round, mean over GPUs)", ""]
        lines += _md_table(
            ["", "A", "B", change_label],
            [
                (
                    "Steps per token round",
                    f"{data['steps_per_round']['a']:.2f}",
                    f"{data['steps_per_round']['b']:.2f}",
                    "",
                ),
                (
                    "Sequences per step",
                    f"{data['sequences_per_step']['a']:.0f}",
                    f"{data['sequences_per_step']['b']:.0f}",
                    "",
                ),
                (
                    "Token round µs",
                    data["round_us"]["a"],
                    data["round_us"]["b"],
                    data["round_us"]["b"] - result["factor"] * data["round_us"]["a"],
                ),
                *(
                    (
                        LABELS[c],
                        data["round"]["a"][c],
                        data["round"]["b"][c],
                        data["change"][c],
                    )
                    for c in CATEGORIES
                ),
            ],
        )
        ma, mb = data["metrics"]["a"], data["metrics"]["b"]
        lines += ["### Shares", ""]
        lines += _md_table(
            ["Metric", "A", "B"],
            [
                (
                    "Compute share %",
                    _share(ma["compute_share"]),
                    _share(mb["compute_share"]),
                ),
                (
                    "Communication share %",
                    _share(ma["comm_share"]),
                    _share(mb["comm_share"]),
                ),
                (
                    "Communication hidden by compute %",
                    _share(ma["overlap_ratio"]),
                    _share(mb["overlap_ratio"]),
                ),
                (
                    "Communication hidden by compute µs",
                    ma["overlap_us"],
                    mb["overlap_us"],
                ),
                (
                    "Exposed communication share %",
                    _share(ma["exposed_comm_share"]),
                    _share(mb["exposed_comm_share"]),
                ),
                (
                    "Pipeline bubble share %",
                    _share(ma["bubble_share"]),
                    _share(mb["bubble_share"]),
                ),
                ("Idle share %", _share(ma["idle_share"]), _share(mb["idle_share"])),
            ],
        )
        if any(len(data["stages"][side]) > 1 for side in ("a", "b")):
            lines += ["### Pipeline stages (µs per token round)", ""]
            lines += _md_table(
                ["Run", "Stage", "Layers", *LAYER_HEADERS[1:]],
                [
                    [
                        side.upper(),
                        f"S{stage['stage']}",
                        f"L{stage['layers'][0]}-L{stage['layers'][1] - 1}",
                        *layer_row("", stage["round"])[1:],
                    ]
                    for side in ("a", "b")
                    for stage in data["stages"][side]
                ],
            )
        lines += ["### Decoder layers", ""]
        lines += comparison_layer_tables(data)
        lines += ["### Outside decoder layers (µs per token round, per GPU)", ""]
        lines += _md_table(
            LAYER_HEADERS,
            [
                layer_row("A pre/post", data["non_layer"]["a"]),
                layer_row("B pre/post", data["non_layer"]["b"]),
            ],
        )
        rows = data.get("collectives") or []
        if rows:
            lines += ["### Communication per token round and GPU", ""]

            def side(stats: dict[str, Any] | None, key: str) -> Any:
                return None if stats is None else stats[key]

            lines += _md_table(
                [
                    "Dim",
                    "Op",
                    "Role",
                    "Count A",
                    "Count B",
                    "Mean µs A",
                    "Mean µs B",
                    "Transfer µs A",
                    "Transfer µs B",
                    "Wait µs A",
                    "Wait µs B",
                    "Total µs A",
                    "Total µs B",
                ],
                [
                    [
                        DIM_LABELS.get(row["dim"], row["dim"]),
                        row["op"],
                        row["role"],
                        *(
                            (
                                side(row[s], key)
                                if key != "count" or row[s] is None
                                else f"{row[s]['count']:.1f}"
                            )
                            for key in (
                                "count",
                                "duration_us",
                                "xfer_us",
                                "wait_us",
                                "total_us",
                            )
                            for s in ("a", "b")
                        ),
                    ]
                    for row in rows
                ],
            )
        kernels = data.get("kernels") or []
        if kernels:
            lines += [
                "### Kernels with the largest change (µs per token round and GPU)",
                "",
            ]
            lines += _md_table(
                [
                    "Kind",
                    "Kernel",
                    "A µs",
                    "B µs",
                    change_label,
                    "A calls",
                    "B calls",
                    "Status",
                ],
                [
                    [
                        row["kind"],
                        f"`{_kernel_name(row['name'], 80)}`",
                        row["a_us"],
                        row["b_us"],
                        row["change_us"],
                        f"{row['a_calls']:.1f}",
                        f"{row['b_calls']:.1f}",
                        row["status"],
                    ]
                    for row in kernels[:20]
                ],
            )
    lines += ["## Method", "", *(f"- {line}" for line in METHOD_LINES), ""]
    return lines


def render_compare_markdown(result: dict[str, Any]) -> str:
    lines = [f"# {comparison_title(result)}", "", "## Findings", ""]
    lines += [f"- {item['text']}" for item in compare_findings(result)]
    lines += ["", *comparison_appendix(result)]
    return "\n".join(lines)


def _inline_html(text: str) -> str:
    text = html_escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?!\w)", r"<em>\1</em>", text)
    return re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        r'<a href="\2" rel="noopener noreferrer">\1</a>',
        text,
    )


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_HTML_STYLE = """
body{font:14px/1.55 system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;color:#18181b;background:#f6f6f7;margin:0}
main{max-width:1180px;margin:24px auto;padding:28px 32px;background:#fff;border:1px solid #e5e5e8;border-radius:8px}
h1{font-size:22px;margin:0 0 12px}h2{font-size:18px;margin:28px 0 10px;padding-top:8px;border-top:1px solid #e5e5e8}
h3{font-size:15px;margin:20px 0 8px}h4{font-size:13px;margin:16px 0 6px;color:#52525b}
table{border-collapse:collapse;margin:6px 0 14px;font-size:12px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}
th,td{border-bottom:1px solid #e5e5e8;padding:4px 10px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{background:#fafafa;color:#52525b;font-weight:600}
code{font:12px ui-monospace,Menlo,Consolas,monospace;background:#f3f3f5;border:1px solid #e5e5e8;border-radius:4px;padding:0 4px}
blockquote{margin:0 0 14px;padding:6px 12px;border-left:3px solid #d4d4d8;color:#52525b;font-size:12px}
pre{background:#f8f8f9;border:1px solid #e5e5e8;border-radius:6px;padding:10px 12px;overflow:auto}pre code{border:0;background:none;padding:0}
"""


def markdown_to_html(markdown: str, title: str) -> str:
    """Self-contained HTML of the Markdown subset used by the reports."""
    out: list[str] = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("```"):
            block = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            out.append(f"<pre><code>{html_escape(chr(10).join(block))}</code></pre>")
            index += 1
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline_html(heading.group(2))}</h{level}>")
            index += 1
            continue
        if stripped.startswith(">"):
            quote = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip()[1:].strip())
                index += 1
            out.append(f"<blockquote>{_inline_html(' '.join(quote))}</blockquote>")
            continue
        if stripped.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [
                    cell.strip().replace("\\|", "|")
                    for cell in re.split(r"(?<!\\)\|", lines[index].strip())[1:-1]
                ]
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    rows.append(cells)
                index += 1
            if rows:
                head = "".join(f"<th>{_inline_html(cell)}</th>" for cell in rows[0])
                body = "".join(
                    "<tr>"
                    + "".join(f"<td>{_inline_html(cell)}</td>" for cell in row)
                    + "</tr>"
                    for row in rows[1:]
                )
                out.append(
                    f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
                )
            continue
        bullet = re.match(r"^(?:[-*]|\d+\.)\s+(.*)$", stripped)
        if bullet:
            ordered = stripped[0].isdigit()
            items = []
            while index < len(lines):
                match = re.match(r"^(?:[-*]|\d+\.)\s+(.*)$", lines[index].strip())
                if not match:
                    break
                items.append(f"<li>{_inline_html(match.group(1))}</li>")
                index += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>{''.join(items)}</{tag}>")
            continue
        if stripped:
            paragraph = [stripped]
            index += 1
            while (
                index < len(lines)
                and lines[index].strip()
                and not re.match(
                    r"^(#{1,4}\s|\||```|>|[-*]\s|\d+\.\s)", lines[index].strip()
                )
            ):
                paragraph.append(lines[index].strip())
                index += 1
            out.append(f"<p>{_inline_html(' '.join(paragraph))}</p>")
            continue
        index += 1
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html_escape(title)}</title>"
        f"<style>{_HTML_STYLE}</style></head><body><main>{''.join(out)}</main></body></html>"
    )


def write_analysis(result: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "attribution.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_path = out_dir / "attribution_layers.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "segment", "stage", "rank", *CATEGORIES, "total_us"])
        for phase, data in result["phases"].items():
            for label in [*data["segments"], STEP_KEY]:
                rows = [("mean", data["mean"][label]), *data["per_rank"][label].items()]
                for rank, values in rows:
                    writer.writerow(
                        [phase, label, data["segment_stage"].get(label, ""), rank]
                        + [round(values[category], 3) for category in CATEGORIES]
                        + [round(sum(values.values()), 3)]
                    )
    markdown = render_analysis_markdown(result)
    markdown_path = out_dir / "attribution_report.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path = out_dir / "attribution.html"
    html_path.write_text(
        markdown_to_html(markdown, "Compute vs communication"), encoding="utf-8"
    )
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "html": str(html_path),
        "markdown": str(markdown_path),
    }


def write_comparison_csv(result: dict[str, Any], path: Path) -> None:
    keys = (
        "total_us",
        "compute_us",
        "comm_us",
        "compute_share",
        "comm_share",
        "overlap_ratio",
        "overlap_us",
        "exposed_comm_us",
        "exposed_comm_share",
        "bubble_us",
        "idle_us",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "phase",
                "layer",
                "stage_a",
                "stage_b",
                *(f"{key}_{side}" for key in keys for side in ("a", "b")),
                "change_us",
            ]
        )
        for phase, data in result["phases"].items():
            for row in data["layers"]:
                values = []
                for key in keys:
                    for side in ("a_metrics", "b_metrics"):
                        value = row[side][key] if row[side] else None
                        values.append("" if value is None else round(value, 4))
                writer.writerow(
                    [
                        phase,
                        row["label"],
                        row["stage_a"],
                        row["stage_b"],
                        *values,
                        "" if row["change_us"] is None else round(row["change_us"], 3),
                    ]
                )


def write_compare(
    result: dict[str, Any], out_dir: Path, stem: str = "compare"
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_path = out_dir / f"{stem}_layers.csv"
    write_comparison_csv(result, csv_path)
    markdown = render_compare_markdown(result)
    markdown_path = out_dir / f"{stem}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path = out_dir / f"{stem}.html"
    html_path.write_text(
        markdown_to_html(markdown, comparison_title(result)), encoding="utf-8"
    )
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "html": str(html_path),
        "markdown": str(markdown_path),
    }


def _rounded(value: Any, digits: int = 1) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _rounded(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item, digits) for item in value]
    return value


def _layer_matrix(
    segments: Sequence[str], mean: dict[str, Any], spr: float
) -> dict[str, Any]:
    return {
        "columns": LAYER_HEADERS,
        "rows": [
            _rounded(layer_row(label, _scaled(mean[label], spr))) for label in segments
        ],
    }


def agent_summary(
    result: dict[str, Any], include_layers: bool = True
) -> dict[str, Any]:
    """Compact, rounded view of one run for an LLM."""
    quality, meta = result["quality"], result["meta"]
    phases: dict[str, Any] = {}
    for phase, data in result["phases"].items():
        spr = steps_per_round(result, phase)
        step = _scaled(data["mean"][STEP_KEY], spr)
        entry: dict[str, Any] = {
            "unit": "µs per token round per GPU" if spr != 1 else "µs per step per GPU",
            "steps": data["steps"],
            "sequences_per_step": data["sequences_per_step"],
            "steps_per_round": spr,
            "breakdown_us": _rounded(step),
            "metrics": _rounded(derived_metrics(step), 4),
            "stages": [
                {
                    "stage": stage["stage"],
                    "ranks": stage["ranks"],
                    "layers": stage["layers"],
                    "metrics": _rounded(
                        derived_metrics(_scaled(stage["step"], spr)), 3
                    ),
                }
                for stage in data.get("stages", [])
            ],
            "collectives": _rounded(data.get("collectives", [])),
        }
        if include_layers:
            entry["layers"] = _layer_matrix(data["segments"], data["mean"], spr)
        phases[phase] = entry
    return {
        "model": meta.get("model"),
        "parallel": meta.get("parallel"),
        "topology": meta.get("topology"),
        "world_size": meta["world_size"],
        "layout": meta["layout"],
        "concurrency": meta.get("concurrency"),
        "bench": meta.get("bench"),
        "quality": {
            key: quality.get(key, 0)
            for key in (
                "matched_steps",
                "incomplete_steps",
                "expected_markers_per_step",
                "markers_per_step",
                "residual_clock_offset_us_max_abs",
                "causality_violations_over_tolerance",
                "comm_count_mismatch_rank_steps",
                "unsegmented_rank_steps",
            )
        }
        | {"alignment": quality["alignment"]["method"]},
        "findings": analysis_findings(result),
        "phases": phases,
    }


def compare_summary(
    result: dict[str, Any], include_layers: bool = True
) -> dict[str, Any]:
    """Compact, rounded view of a comparison for an LLM and the UI."""
    phases: dict[str, Any] = {}
    for phase, data in result["phases"].items():
        entry = {
            key: _rounded(data[key], 3)
            for key in (
                "unit",
                "steps_per_round",
                "sequences_per_step",
                "steps",
                "round_us",
                "speedup",
                "ideal_speedup",
                "scaling_efficiency",
                "round_delta_se_us",
            )
        }
        entry["round"] = _rounded(data["round"])
        entry["change"] = _rounded(data["change"])
        entry["groups"] = {
            side: _rounded(_group_values(data["round"][side])) for side in ("a", "b")
        } | {"change": _rounded(_group_values(data["change"]))}
        entry["metrics"] = _rounded(data["metrics"], 4)
        entry["non_layer"] = _rounded(
            {side: derived_metrics(data["non_layer"][side]) for side in ("a", "b")}
        )
        entry["stages"] = {
            side: [
                {
                    "stage": stage["stage"],
                    "ranks": stage["ranks"],
                    "layers": stage["layers"],
                    "round": _rounded(stage["round"]),
                    "metrics": _rounded(stage["metrics"], 4),
                }
                for stage in stages
            ]
            for side, stages in data["stages"].items()
        }
        entry["collectives"] = _rounded(data["collectives"])
        entry["kernels"] = [
            _rounded({**row, "name": _kernel_name(row["name"], 90)})
            for row in data["kernels"][:12]
        ]
        if include_layers:
            entry["layers"] = [
                {
                    "label": row["label"],
                    "stage_a": row["stage_a"],
                    "stage_b": row["stage_b"],
                    "a": _rounded(row["a_metrics"], 4),
                    "b": _rounded(row["b_metrics"], 4),
                    "change_us": _rounded(row["change_us"]),
                }
                for row in data["layers"]
            ]
        phases[phase] = entry
    return {
        "mode": result["mode"],
        "change_definition": (
            "B - A"
            if result["mode"] == "same_gpus"
            else f"B - {result['factor']:.4g} x A (linear scaling)"
        ),
        "factor": result["factor"],
        "a": result["a"],
        "b": result["b"],
        "findings": compare_findings(result),
        "phases": phases,
    }


def brief(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "parallel": result["meta"].get("parallel"),
        "world_size": result["meta"]["world_size"],
        "alignment": result["quality"]["alignment"]["method"],
        "phases": {
            phase: {
                "steps": data["steps"],
                "segments": len(data["segments"]),
                **_rounded(
                    derived_metrics(
                        _scaled(data["mean"][STEP_KEY], steps_per_round(result, phase))
                    ),
                    4,
                ),
            }
            for phase, data in result["phases"].items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vap attribute",
        description="Per-layer compute vs communication attribution of vLLM rank traces.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze", help="Attribute one run")
    analyze_parser.add_argument(
        "trace_dir", type=Path, help="Run directory or its vllm-profile directory"
    )
    analyze_parser.add_argument(
        "--num-layers",
        type=int,
        help="Decoder layers (default: the run's model config)",
    )
    analyze_parser.add_argument(
        "--tp", type=int, help="Tensor parallel size (default: the run's config)"
    )
    analyze_parser.add_argument(
        "--pp", type=int, help="Pipeline parallel size (default: the run's config)"
    )
    analyze_parser.add_argument("--comms-per-layer", type=int, default=2)
    analyze_parser.add_argument("--pre-comms", type=int, default=1)
    analyze_parser.add_argument("--violation-tolerance-us", type=float, default=1.0)
    analyze_parser.add_argument("--out", type=Path)
    compare_parser = subparsers.add_parser("compare", help="Compare two analyzed runs")
    compare_parser.add_argument("base", type=Path, help="attribution.json of run A")
    compare_parser.add_argument("target", type=Path, help="attribution.json of run B")
    compare_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "analyze":
        profile_dir = args.trace_dir
        if (profile_dir / "vllm-profile").is_dir():
            profile_dir = profile_dir / "vllm-profile"
        files, timeline = select_trace_files(profile_dir)
        if not files:
            parser.error(f"No PyTorch trace files under {args.trace_dir}")
        info = run_model_info(profile_dir.resolve().parent)
        num_layers = args.num_layers or info["num_layers"]
        if not num_layers:
            parser.error(
                "--num-layers is required: the run's model config could not be read"
            )
        topology = Topology(**(info["topology"] or {}))
        if args.tp or args.pp:
            topology = Topology(
                args.tp or topology.tp, args.pp or topology.pp, topology.dp, topology.ep
            )
        layout = Layout(num_layers, args.comms_per_layer, args.pre_comms)
        result = analyze(files, layout, topology, args.violation_tolerance_us)
        result["meta"].update(
            trace_dir=str(profile_dir),
            timeline=timeline,
            model=info["model"],
            concurrency=info["concurrency"],
            bench=info["bench"],
        )
        outputs = write_analysis(result, args.out or profile_dir / "attribution")
        print(json.dumps({"outputs": outputs, "summary": brief(result)}, indent=2))
    else:
        base = json.loads(args.base.read_text(encoding="utf-8"))
        target = json.loads(args.target.read_text(encoding="utf-8"))
        result = compare(base, target)
        outputs = write_compare(result, args.out)
        print(
            json.dumps(
                {"outputs": outputs, "findings": compare_findings(result)},
                indent=2,
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
