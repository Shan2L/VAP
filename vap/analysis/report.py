"""Templated A/B analysis report: the VAP agent's LLM explains a comparison
from a compact data package; a rule-based narrative with the same sections
stands in when the LLM is unavailable. Full data tables are appended by code,
never by the LLM."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from datetime import datetime
from typing import Any

from vap.analysis import attribution as ta

SECTIONS = {
    "zh": [
        "结论",
        "关键指标",
        "时间去哪了",
        "并行方式分析",
        "逐层分析",
        "优化建议",
        "数据可信度",
    ],
    "en": [
        "Conclusion",
        "Key metrics",
        "Where the time goes",
        "Parallelism analysis",
        "Per-layer analysis",
        "Recommendations",
        "Data quality",
    ],
}
GROUP_NAMES = {
    "zh": {
        "Compute": "计算",
        "Comm hidden by compute": "被计算遮盖的通信",
        "Exposed collective comm": "暴露的集合通信",
        "PP transfer": "PP 传输",
        "Pipeline bubble": "流水线气泡",
        "Idle": "空闲",
    },
    "en": {},
}
APPENDIX = {"zh": "附录：完整数据", "en": "Appendix: full data"}
LANGUAGES = tuple(SECTIONS)


def report_data(
    comparison: dict[str, Any], base: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    """Everything the LLM may cite, rounded."""
    summary = ta.compare_summary(comparison)
    for phase in summary["phases"].values():
        phase["layers"] = [
            {
                "label": row["label"],
                "stage": [row["stage_a"], row["stage_b"]],
                "change_us": row["change_us"],
                **{
                    f"{key}_{side}": (
                        None if row[side] is None else ta._rounded(row[side][key], 3)
                    )
                    for side in ("a", "b")
                    for key in (
                        "total_us",
                        "compute_us",
                        "comm_us",
                        "overlap_us",
                        "overlap_ratio",
                        "exposed_comm_share",
                        "bubble_us",
                        "idle_us",
                    )
                },
            }
            for row in phase.get("layers", [])
        ]
    summary["quality"] = {
        side: ta.agent_summary(result, include_layers=False)["quality"]
        for side, result in (("a", base), ("b", target))
    }
    return summary


def data_digest(data: dict[str, Any], language: str) -> str:
    text = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(f"{language}\n{text}".encode()).hexdigest()[:16]


def system_prompt(language: str) -> str:
    sections = SECTIONS[language]
    headings = "\n".join(f"## {name}" for name in sections)
    lang = "Simplified Chinese" if language == "zh" else "English"
    return (
        "You are the VAP performance analyst. You explain why vLLM inference with "
        "one parallel configuration (run A) is faster or slower than another (run B) "
        "on AMD GPUs, using only the measured data you are given.\n"
        "Units: microseconds per token round per GPU. A token round is the time in "
        "which every running sequence decodes one token; with pipeline parallelism a "
        "token round needs one engine step per micro-batch. Values are means over the "
        "GPUs of a run. With the same GPU count, change = B - A; with different GPU "
        "counts, change = B - (gpus_A / gpus_B) x A, i.e. the loss against linear "
        "scaling. Category changes add up to the token-round change. Per-layer values "
        "are the wall time of a layer on the GPUs that run it; with pipeline "
        "parallelism layers of different stages run concurrently, so per-layer times "
        "do not add up to the token round.\n"
        "Rules: cite numbers exactly as given (you may compute simple ratios and "
        "differences, showing the inputs); never invent measurements; label anything "
        "not directly measured as a hypothesis; name the mechanism behind each effect "
        "(for example TP all-reduce cost per instance and count, pipeline micro-batches "
        "re-reading weights, pipeline bubbles from stage imbalance, host launch gaps "
        "showing up as idle time); be specific about layers, stages, kernels and "
        "collectives. Do not repeat the data tables: the full tables are appended to "
        "the report automatically.\n"
        f"Write in {lang}. Output exactly these Markdown sections in this order, each "
        "starting with its heading line, and nothing before the first heading:\n"
        f"{headings}\n"
        f"Section guidance: '{sections[0]}': 3 to 5 bullets; the first bullet answers "
        "directly how much faster or slower B is and why, with numbers. "
        f"'{sections[1]}': one compact table with A, B and change for token-round time, "
        "compute share, communication share, communication hidden by compute, exposed "
        "communication share, pipeline bubble share and idle share. "
        f"'{sections[2]}': rank the category changes and explain each. "
        f"'{sections[3]}': TP communication (count, time per instance, wait vs "
        "transfer), pipeline behaviour (micro-batches, stage balance, bubble, stage "
        "hand-off) and scaling efficiency where relevant. "
        f"'{sections[4]}': how layers differ, outliers and the part outside decoder "
        f"layers. '{sections[5]}': 2 to 4 actionable items ordered by expected gain, "
        "each with the evidence and an expected effect; mark estimates as estimates. "
        f"'{sections[6]}': clock alignment, unmatched or unsegmented data and noise, "
        "and how far the conclusions can be trusted. Keep the whole report under 900 "
        "words."
    )


def user_prompt(data: dict[str, Any], question: str | None) -> str:
    ask = question or "Explain why run B performs differently from run A."
    return (
        f"Question: {ask}\n\n"
        "Measured comparison data (JSON):\n"
        f"{json.dumps(data, ensure_ascii=False, separators=(',', ':'), default=str)}"
    )


def split_sections(markdown: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = None
    lines: list[str] = []
    for line in markdown.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current, lines = heading.group(1), []
        elif current is not None:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    return sections


def conclusion_of(markdown: str, language: str) -> str:
    sections = split_sections(markdown)
    for name, body in sections.items():
        if any(key in name for key in ("结论", "Conclusion")):
            return body
    if sections:
        return next(iter(sections.values()))
    return markdown.strip()[:2000]


def _ms(us: float | None) -> str:
    return "n/a" if us is None else f"{us / 1000:.2f} ms"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _signed_ms(us: float) -> str:
    return f"{us / 1000:+.2f} ms"


def rule_narrative(data: dict[str, Any], language: str) -> str:
    """Report body with the template's sections, built from the numbers alone."""
    zh = language == "zh"
    names = SECTIONS[language]
    a, b = data["a"], data["b"]
    same = data["mode"] == "same_gpus"
    phase_name = (
        "decode" if "decode" in data["phases"] else next(iter(data["phases"]), None)
    )
    if phase_name is None:
        return "\n\n".join(f"## {name}\n\n-" for name in names)
    p = data["phases"][phase_name]
    ta_us, tb_us = p["round_us"]["a"], p["round_us"]["b"]
    groups = p["groups"]["change"]
    ranked = sorted(groups.items(), key=lambda item: -abs(item[1]))
    gname = GROUP_NAMES[language]

    def g(name: str) -> str:
        return gname.get(name, name)

    ma, mb = p["metrics"]["a"], p["metrics"]["b"]
    tp_rows = [
        row
        for row in p["collectives"]
        if row["dim"] == "tp" and row["op"] == "all_reduce"
    ]

    def reduce_stats(side: str) -> tuple[float, float] | None:
        rows = [row[side] for row in tp_rows if row[side]]
        count = sum(row["count"] for row in rows)
        if not count:
            return None
        return count, sum(row["count"] * row["duration_us"] for row in rows) / count

    ra, rb = reduce_stats("a"), reduce_stats("b")
    stages = {side: p["stages"][side] for side in ("a", "b")}

    def busy(stage: dict[str, Any]) -> float:
        return (
            stage["metrics"]["total_us"]
            - stage["round"]["bubble"]
            - stage["round"]["idle"]
        )

    conclusion: list[str] = []
    if same:
        delta = tb_us - ta_us
        if zh:
            conclusion.append(
                f"B（{b['parallel']}）每个 token 轮次需要 {_ms(tb_us)}，A（{a['parallel']}）为 {_ms(ta_us)}，"
                f"B {'慢' if delta > 0 else '快'} {abs(delta) / ta_us:.1%}（{_signed_ms(delta)}）。"
            )
        else:
            conclusion.append(
                f"B ({b['parallel']}) needs {_ms(tb_us)} per token round vs {_ms(ta_us)} for A ({a['parallel']}): "
                f"{abs(delta) / ta_us:.1%} {'slower' if delta > 0 else 'faster'} ({_signed_ms(delta)})."
            )
    else:
        if zh:
            conclusion.append(
                f"{a['parallel']}（{a['gpus']} 卡）→ {b['parallel']}（{b['gpus']} 卡）：每 token 轮次 {_ms(ta_us)} → {_ms(tb_us)}，"
                f"加速比 {p['speedup']:.2f}×，理想 {p['ideal_speedup']:.2f}×，扩展效率 {_pct(p['scaling_efficiency'])}。"
            )
        else:
            conclusion.append(
                f"{a['parallel']} ({a['gpus']} GPUs) → {b['parallel']} ({b['gpus']} GPUs): token round {_ms(ta_us)} → {_ms(tb_us)}, "
                f"speedup {p['speedup']:.2f}x vs ideal {p['ideal_speedup']:.2f}x, scaling efficiency {_pct(p['scaling_efficiency'])}."
            )
    top = [(name, value) for name, value in ranked if abs(value) >= 50][:3]
    if top:
        parts = (
            "、".join(f"{g(name)} {_signed_ms(value)}" for name, value in top)
            if zh
            else ", ".join(f"{name} {_signed_ms(value)}" for name, value in top)
        )
        reference = (
            "" if same else ("（相对线性扩展）" if zh else " (against linear scaling)")
        )
        conclusion.append(
            f"差异主要来自{reference}：{parts}。"
            if zh
            else f"The change{reference} comes mainly from {parts}."
        )
    for side, label in (("a", "A"), ("b", "B")):
        if len(stages[side]) > 1:
            busy_list = [(stage["stage"], busy(stage)) for stage in stages[side]]
            slow = max(busy_list, key=lambda item: item[1])
            text = (
                "、".join(f"S{s} {_ms(v)}" for s, v in busy_list)
                if zh
                else ", ".join(f"S{s} {_ms(v)}" for s, v in busy_list)
            )
            bubble = p["round"][side]["bubble"]
            conclusion.append(
                f"{label} 的流水线不均衡：各 stage 每轮忙碌 {text}，S{slow[0]} 是瓶颈，其余 stage 平均每卡每轮空等 {_ms(bubble)}（流水线气泡）。"
                if zh
                else f"{label}'s pipeline is unbalanced: stages are busy {text} per token round; S{slow[0]} is the bottleneck and the others wait {_ms(bubble)} per GPU and round (pipeline bubble)."
            )
    if ra and rb:
        conclusion.append(
            f"TP all-reduce：A 每卡每轮 {ra[0]:.0f} 次 × {ra[1]:.1f} µs，B {rb[0]:.0f} 次 × {rb[1]:.1f} µs，"
            f"暴露的集合通信变化 {_signed_ms(groups.get('Exposed collective comm', 0.0))}。"
            if zh
            else f"TP all-reduce: A {ra[0]:.0f} × {ra[1]:.1f} µs per GPU and round, B {rb[0]:.0f} × {rb[1]:.1f} µs; "
            f"exposed collective communication changes by {_signed_ms(groups.get('Exposed collective comm', 0.0))}."
        )

    metric_rows = [
        (
            "Token round" if not zh else "Token 轮次",
            _ms(ta_us),
            _ms(tb_us),
            _signed_ms(tb_us - (1 if same else data["factor"]) * ta_us),
        ),
        *(
            (
                label,
                _pct(ma[key]),
                _pct(mb[key]),
                f"{100 * ((mb[key] or 0) - (ma[key] or 0)):+.1f} pp",
            )
            for key, label in (
                ("compute_share", "计算占比" if zh else "Compute share"),
                ("comm_share", "通信占比" if zh else "Communication share"),
                ("overlap_ratio", "通信被遮盖比例" if zh else "Communication hidden"),
                (
                    "exposed_comm_share",
                    "暴露通信占比" if zh else "Exposed communication share",
                ),
                ("bubble_share", "气泡占比" if zh else "Bubble share"),
                ("idle_share", "空闲占比" if zh else "Idle share"),
            )
        ),
    ]
    head = "| 指标 | A | B | 变化 |" if zh else "| Metric | A | B | Change |"
    table = [
        head,
        "|---|---:|---:|---:|",
        *(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in metric_rows),
    ]

    where = [
        (
            f"- {g(name)}：{_signed_ms(value)}（A {_ms(p['groups']['a'][name])}，B {_ms(p['groups']['b'][name])}）"
            if zh
            else f"- {name}: {_signed_ms(value)} (A {_ms(p['groups']['a'][name])}, B {_ms(p['groups']['b'][name])})"
        )
        for name, value in ranked
        if max(abs(value), p["groups"]["a"][name], p["groups"]["b"][name]) >= 10
    ]
    change = p["change"]
    where.append(
        f"- 计算内部：GEMM {_signed_ms(change['gemm'])}，Attention {_signed_ms(change['attention'])}，其他 {_signed_ms(change['other'])}。"
        if zh
        else f"- Inside compute: GEMM {_signed_ms(change['gemm'])}, attention {_signed_ms(change['attention'])}, other {_signed_ms(change['other'])}."
    )

    parallel: list[str] = []
    if ra and rb:
        parallel.append(
            f"- TP 通信：all-reduce 单次 {ra[1]:.1f} → {rb[1]:.1f} µs，每卡每轮 {ra[0]:.0f} → {rb[0]:.0f} 次；all-reduce 与计算在同一 stream 上串行，通信被遮盖比例 A {_pct(ma['overlap_ratio'])}、B {_pct(mb['overlap_ratio'])}。"
            if zh
            else f"- TP communication: all-reduce {ra[1]:.1f} → {rb[1]:.1f} µs per instance, {ra[0]:.0f} → {rb[0]:.0f} per GPU and round; hidden share A {_pct(ma['overlap_ratio'])}, B {_pct(mb['overlap_ratio'])}."
        )
    spr, seqs = p["steps_per_round"], p["sequences_per_step"]
    if spr["a"] != spr["b"]:
        parallel.append(
            f"- Micro-batch：A 每轮 {spr['a']:.0f} 个 step × {seqs['a']:.0f} 条序列，B 每轮 {spr['b']:.0f} 个 step × {seqs['b']:.0f} 条；每个 micro-batch 都要把本 stage 的权重读一遍，所以 decode 的 GEMM 时间随 micro-batch 数增加（GEMM {_signed_ms(change['gemm'])}）。"
            if zh
            else f"- Micro-batches: A runs {spr['a']:.0f} step(s) × {seqs['a']:.0f} sequences per round, B {spr['b']:.0f} × {seqs['b']:.0f}; every micro-batch streams the stage's weights again, so decode GEMM time grows (GEMM {_signed_ms(change['gemm'])})."
        )
    for side, label in (("a", "A"), ("b", "B")):
        if len(stages[side]) > 1:
            for stage in stages[side]:
                r = stage["round"]
                parallel.append(
                    f"- {label} S{stage['stage']}（L{stage['layers'][0]}-L{stage['layers'][1] - 1}）：忙碌 {_ms(busy(stage))}，气泡 {_ms(r['bubble'])}，空闲 {_ms(r['idle'])}，GEMM {_ms(r['gemm'])}。"
                    if zh
                    else f"- {label} S{stage['stage']} (L{stage['layers'][0]}-L{stage['layers'][1] - 1}): busy {_ms(busy(stage))}, bubble {_ms(r['bubble'])}, idle {_ms(r['idle'])}, GEMM {_ms(r['gemm'])}."
                )
    if not same:
        parallel.append(
            f"- 扩展效率 {_pct(p['scaling_efficiency'])}：理想情况下 {b['parallel']} 每轮应为 {_ms(data['factor'] * ta_us)}，实测 {_ms(tb_us)}。"
            if zh
            else f"- Scaling efficiency {_pct(p['scaling_efficiency'])}: ideal {b['parallel']} round {_ms(data['factor'] * ta_us)}, measured {_ms(tb_us)}."
        )

    layers = [row for row in p.get("layers", []) if row["change_us"] is not None]
    layer_lines: list[str] = []
    if layers:
        changes = [row["change_us"] for row in layers]
        worst = sorted(layers, key=lambda row: -row["change_us"])[:3]
        layer_lines.append(
            f"- 每层每轮时间变化 {min(changes):+.0f} 到 {max(changes):+.0f} µs，中位数 {statistics.median(changes):+.0f} µs；变化最大：{'、'.join(f'{row['label']} {row['change_us']:+.0f} µs' for row in worst)}。"
            if zh
            else f"- Per layer the token-round time changes by {min(changes):+.0f} to {max(changes):+.0f} µs (median {statistics.median(changes):+.0f} µs); largest: {', '.join(f'{row['label']} {row['change_us']:+.0f} µs' for row in worst)}."
        )
    non = p.get("non_layer", {})
    if non:
        layer_lines.append(
            f"- 层外部分（embedding、LM head、采样、stage 交接，含气泡）：A {_ms(non['a']['total_us'])}，B {_ms(non['b']['total_us'])}，其中气泡 A {_ms(non['a']['bubble_us'])}、B {_ms(non['b']['bubble_us'])}。"
            if zh
            else f"- Outside decoder layers (embedding, LM head, sampling, stage hand-off, including the bubble): A {_ms(non['a']['total_us'])}, B {_ms(non['b']['total_us'])}; bubble A {_ms(non['a']['bubble_us'])}, B {_ms(non['b']['bubble_us'])}."
        )

    recs: list[str] = []
    for side, label, metrics in (("b", "B", mb), ("a", "A", ma)):
        if len(stages[side]) == 2 and (metrics["bubble_share"] or 0) >= 0.05:
            s0, s1 = stages[side]
            gap = busy(s1) - busy(s0)
            per_layer = (
                statistics.fmean(
                    row[f"total_us_{side}"]
                    for row in layers
                    if row[f"total_us_{side}"] is not None
                )
                if layers
                else 0.0
            )
            move = round(abs(gap) / (2 * per_layer)) if per_layer else 0
            n0 = s0["layers"][1] - s0["layers"][0]
            n1 = s1["layers"][1] - s1["layers"][0]
            if move:
                split = (
                    f"{n0 + move},{n1 - move}"
                    if gap > 0
                    else f"{n0 - move},{n1 + move}"
                )
                left = abs(abs(gap) - 2 * move * per_layer)
                recs.append(
                    f"- 平衡流水线（{label}）：S{1 if gap > 0 else 0} 每轮多忙 {_ms(abs(gap))}，每层约 {per_layer:.0f} µs/轮；可尝试 `VLLM_PP_LAYER_PARTITION={split}`，估计 stage 间差距降到约 {_ms(left)}/轮（估算，需实测）。"
                    if zh
                    else f"- Balance the pipeline ({label}): S{1 if gap > 0 else 0} is busier by {_ms(abs(gap))} per round at about {per_layer:.0f} µs per layer; try `VLLM_PP_LAYER_PARTITION={split}`, estimated to cut the stage gap to about {_ms(left)} per round (estimate; measure it)."
                )
            break
    if spr["b"] > spr["a"] and change["gemm"] > 0:
        recs.append(
            "- decode 延迟场景下 PP 会把 batch 拆成 micro-batch、重复读取权重；同样的卡数优先用更大的 TP 或 DP×TP，PP 主要用于单卡/单节点放不下模型的情况。"
            if zh
            else "- For decode latency, pipeline parallelism splits the batch into micro-batches that re-read weights; with the same GPUs prefer larger TP or DP×TP and keep PP for models that do not fit."
        )
    if (mb["exposed_comm_share"] or 0) >= 0.2:
        recs.append(
            f"- B 的暴露通信占 {_pct(mb['exposed_comm_share'])}：all-reduce 与计算串行。可评估更小的 TP（配合 DP 提升吞吐）、融合 all-reduce+RMSNorm 或通信计算重叠方案（估计收益上限为暴露通信时间）。"
            if zh
            else f"- B spends {_pct(mb['exposed_comm_share'])} in exposed communication: all-reduces serialize with compute. Evaluate smaller TP (with DP for throughput), fused all-reduce + RMSNorm or communication/compute overlap (upper bound: the exposed time)."
        )
    if (mb["idle_share"] or 0) >= 0.15:
        recs.append(
            f"- B 有 {_pct(mb['idle_share'])} 的时间 GPU 空闲（没有 kernel 运行，多为 host 侧发射间隙）：检查 CUDA graph 是否覆盖该 batch size，减少每 step 的 Python 开销。"
            if zh
            else f"- B's GPUs are idle {_pct(mb['idle_share'])} of the time (no kernel running, mostly host launch gaps): check CUDA graph coverage for this batch size and per-step Python overhead."
        )
    if not recs:
        recs.append(
            "- 暂无明显的单项瓶颈；如需进一步优化，从时间分解中占比最大的类别入手。"
            if zh
            else "- No single dominant bottleneck; start from the largest category in the breakdown."
        )

    qa, qb = data["quality"]["a"], data["quality"]["b"]
    se = p.get("round_delta_se_us")
    quality = [
        (
            f"- 时钟对齐：{qa['alignment']} / {qb['alignment']}，残余偏差最大 {qa['residual_clock_offset_us_max_abs']:.1f} / {qb['residual_clock_offset_us_max_abs']:.1f} µs（等待/传输基于 kernel 时长，不受影响）。"
            if zh
            else f"- Clock alignment: {qa['alignment']} / {qb['alignment']}, residual offset up to {qa['residual_clock_offset_us_max_abs']:.1f} / {qb['residual_clock_offset_us_max_abs']:.1f} µs (wait and transfer use kernel durations and are unaffected)."
        ),
        (
            f"- 未分段 rank-step：{qa.get('unsegmented_rank_steps', 0)} / {qb.get('unsegmented_rank_steps', 0)}；未匹配通信的 rank-step：{qa.get('comm_count_mismatch_rank_steps', 0)} / {qb.get('comm_count_mismatch_rank_steps', 0)}。"
            if zh
            else f"- Unsegmented rank-steps: {qa.get('unsegmented_rank_steps', 0)} / {qb.get('unsegmented_rank_steps', 0)}; rank-steps with unmatched collectives: {qa.get('comm_count_mismatch_rank_steps', 0)} / {qb.get('comm_count_mismatch_rank_steps', 0)}."
        ),
    ]
    if se:
        quality.append(
            f"- 轮次差的 step 间标准误为 {se:.1f} µs；结论只覆盖本次采样窗口，跨运行波动需重复运行确认。"
            if zh
            else f"- Step-to-step standard error of the round difference is {se:.1f} µs; run-to-run variance needs repeated runs."
        )

    body = {
        names[0]: [f"- {line}" for line in conclusion],
        names[1]: table,
        names[2]: where,
        names[3]: parallel or ["-"],
        names[4]: layer_lines or ["-"],
        names[5]: recs,
        names[6]: quality,
    }
    return "\n\n".join(
        f"## {name}\n\n" + "\n".join(lines) for name, lines in body.items()
    )


def assemble(
    comparison: dict[str, Any],
    narrative: str,
    language: str,
    source: str,
    model: str | None,
) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    by = (
        f"VAP Agent ({model})"
        if source == "agent"
        else ("规则引擎" if language == "zh" else "rule engine")
    )
    meta = (
        f"> 生成：{by} · {stamp} · A = `{comparison['a']['run']}` · B = `{comparison['b']['run']}`"
        if language == "zh"
        else f"> Written by {by} · {stamp} · A = `{comparison['a']['run']}` · B = `{comparison['b']['run']}`"
    )
    appendix = ta.comparison_appendix(comparison)
    return "\n".join(
        [
            f"# {ta.comparison_title(comparison)}",
            "",
            meta,
            "",
            narrative.strip(),
            "",
            f"# {APPENDIX[language]}",
            "",
            *appendix,
        ]
    )
