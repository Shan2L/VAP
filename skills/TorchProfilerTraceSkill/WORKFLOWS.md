# Workflows

## 1. Trace Overview

Goal: establish whether the trace is usable and what evidence is available.

Run:

- `trace_overview`
- `category_duration`
- `rank_activity`

Look for:

- total slice/event counts
- trace time span
- available ranks
- category dominance, especially `cuda_runtime`, `cpu_op`, `kernel`, `gpu_memcpy`

## 2. Synchronization Waits

Goal: identify host/GPU waits and barriers.

Run:

- `sync_waits`
- `top_slices`

Look for:

- `hipEventSynchronize`
- `cudaDeviceSynchronize`
- wait/barrier/sync slices
- long waits that precede or follow GPU kernels

Interpretation:

- Long synchronization events can mean host-side blocking, serialized GPU work, or cross-rank synchronization pressure.

## 3. Operator Hotspots

Goal: identify high-cost PyTorch/vLLM CPU-side operators.

Run:

- `operator_hotspots`
- `aten_hotspots`

Look for:

- repeated high-duration `aten::` ops
- `aten::sort`, top-k, sampling, attention-related ops
- many calls with large total duration

## 4. GPU Kernel Hotspots

Goal: identify high-duration GPU kernel slices.

Run:

- `gpu_kernels`
- `category_duration`

Look for:

- very long kernels
- high launch overhead relative to kernel time
- repeated kernels aligned with decode steps

## 5. Rank Imbalance

Goal: find whether tensor-parallel ranks behave similarly.

Run:

- `rank_activity`
- `rank_longest_slices`

Look for:

- uneven event counts
- one rank with much longer top slices
- long idle gaps on some ranks while others work

## 6. Memory Copy Overhead

Goal: detect expensive host/device or device/device copies.

Run:

- `memory_copies`
- `category_duration`

Look for:

- `gpu_memcpy`
- copy-like slice names
- memory movement near decode or synchronization spans

## 7. Report

A good report should include:

- trace metadata
- ranked findings by confidence
- SQL evidence
- Perfetto tracks/events to inspect
- optimization hypotheses
- evidence gaps

Avoid claiming root cause when evidence only suggests correlation.
