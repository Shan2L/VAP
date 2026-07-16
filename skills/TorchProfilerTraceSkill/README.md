# TorchProfilerTraceSkill

TorchProfilerTraceSkill is a standalone Agent Skill project for evidence-based analysis of PyTorch profiler traces produced by vLLM/VAP profiling runs.

It focuses on traces viewable in Perfetto, especially merged multi-rank files such as:

```text
20260716_121022-Qwen_Qwen3-32B-merged_trace.json.gz
```

## Why This Exists

Raw Chrome trace JSON is too large and noisy for an LLM to inspect directly. This project uses a smaller and more reliable workflow:

1. Identify the trace.
2. Run deterministic Perfetto SQL presets.
3. Collect structured evidence.
4. Let the Agent explain bottlenecks and optimization hypotheses from that evidence.

## Attribution

This project is inspired by:

- [Gracker/Perfetto-Skills](https://github.com/Gracker/Perfetto-Skills)
- [Gracker/SmartPerfetto](https://github.com/Gracker/SmartPerfetto)

The borrowed idea is the evidence-driven workflow pattern: SQL-backed findings, capability checks, reports, and clear evidence boundaries.

No code or SQL is copied from those projects. The SQL presets here are original and tailored to PyTorch profiler / vLLM traces.

## Files

- `SKILL.md`: standard Agent Skill entry point.
- `WORKFLOWS.md`: analysis workflows and report guidance.
- `queries.yaml`: named Perfetto SQL presets.
- `runner.py`: local helper for running presets with `trace_processor`.

## Quick Usage

```bash
python runner.py --trace /path/to/merged_trace.json.gz --query trace_overview
python runner.py --trace /path/to/merged_trace.json.gz --query sync_waits --limit 20
python runner.py --trace /path/to/merged_trace.json.gz --all
```

Set a custom trace processor path if needed:

```bash
export TRACE_PROCESSOR=/path/to/trace_processor
```

## Intended Integration With VAP

VAP can use this project in two ways:

1. Read the skill instructions as Agent context.
2. Import/copy the SQL presets into VAP's `run_perfetto_sql` whitelist.

The recommended production path is to expose these presets as VAP Agent tools, not to let the Agent run arbitrary SQL.
