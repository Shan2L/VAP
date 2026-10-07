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
        "单卡计算开销",
        "通信模式开销",
        "逐层分析",
        "优化建议",
        "数据可信度",
    ],
    "en": [
        "Conclusion",
        "Key metrics",
        "Compute per GPU",
        "Communication pattern",
        "Per-layer analysis",
        "Recommendations",
        "Data quality",
    ],
}
APPENDIX = {"zh": "附录：完整数据", "en": "Appendix: full data"}
LANGUAGES = tuple(SECTIONS)


def report_data(
    comparison: dict[str, Any],
    base: dict[str, Any],
    target: dict[str, Any],
    include_timeline: bool = False,
) -> dict[str, Any]:
    """Everything the LLM may cite, rounded; the page also gets the timeline."""
    summary = ta.compare_summary(comparison, include_timeline=include_timeline)
    for phase in summary["phases"].values():
        phase["layers"] = [
            {
                "label": row["label"],
                "stage": [row["stage_a"], row["stage_b"]],
                "change_us": row["change_us"],
                **{
                    f"wall_us_{side}": (row.get("wall_us") or {}).get(side)
                    for side in ("a", "b")
                },
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
        "Units: microseconds per generated token, averaged over the GPUs of a run "
        "(the time in which every running sequence decodes one token; comparable to "
        "TPOT; pipeline-parallel decode needs one engine step per micro-batch). With "
        "the same GPU count, change = B - A; with different GPU counts, change = "
        "B - B's ideal: A's compute x gpus_A / gpus_B (more GPUs share it) plus A's "
        "communication, pipeline wait and idle, which every GPU still spends (TP sends "
        "the same all-reduces of the same size per token at any TP size). 'ideal_us' is "
        "that ideal time per token and 'best_speedup' the speedup it allows; "
        "'linear_us' and 'ideal_speedup' describe perfect linear scaling, which A's "
        "communication and idle make unreachable. 'pillars' "
        "splits a GPU's time per token into compute, communication (exposed, "
        "including waiting for peer GPUs), pipeline wait (waiting for another stage) "
        "and idle; they add up to the time per token. 'work' gives what one GPU "
        "reads per token: weight_share of the decoder weights (decode GEMMs are bound "
        "by weight reads) and kv_share of the KV cache, plus the estimated bytes of "
        "one TP all-reduce. 'ranks' gives every GPU's categories per token. "
        "'collectives' gives per kind of communication the calls per GPU and token "
        "and the time per call split into transfer and wait. 'parts' splits the time "
        "into the decoder layers and the work outside them; per-layer values are each "
        "layer's share of the token (wall_us is its time on its stage's GPUs).\n"
        "Always give absolute times first; a share or percentage only next to the "
        "time it belongs to, never alone.\n"
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
        "directly how much faster or slower B is; then which pillars move and the "
        "mechanism behind each (work per GPU, communication pattern, pipeline). "
        f"'{sections[1]}': one compact table with A, B and change in ms for the time "
        "per token and each pillar (compute, communication, pipeline wait, idle), "
        "shares in parentheses. "
        f"'{sections[2]}': per GPU, the work per token (layers, slice of each matrix, "
        "steps per token, weight and KV reads) against the measured GEMM, attention "
        "and other compute, and any imbalance between GPUs or stages. "
        f"'{sections[3]}': per kind of communication, calls per token × time per "
        "call (transfer vs wait), GPUs per call and estimated message size; then "
        "pipeline hand-off and waiting, waits between GPUs and what is hidden. "
        f"'{sections[4]}': first how much of the change comes from the decoder "
        "layers and how much from the work outside them (parts), then how layers "
        f"differ and outliers. '{sections[5]}': 2 to 4 actionable items ordered by expected gain, "
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


PILLAR_NAMES = {
    "zh": {
        "compute": "计算",
        "communication": "通信",
        "pipeline": "流水线等待",
        "idle": "空闲",
    },
    "en": {
        "compute": "compute",
        "communication": "communication",
        "pipeline": "pipeline wait",
        "idle": "idle",
    },
}
NOTE_LABELS = {
    "zh": {"what": "这张图表示", "reading": "当前数值说明"},
    "en": {"what": "What this shows", "reading": "What the numbers say"},
}
LAYER_ROLES = ("attention", "mlp", "layers")
ROLE_NAMES = {
    "zh": {
        "embedding": "embedding",
        "attention": "attention 之后",
        "mlp": "MLP 之后",
        "pre": "层前",
        "post": "logits 汇总",
        "stage transfer": "stage 交接",
    },
    "en": {
        "attention": "after attention",
        "mlp": "after MLP",
        "pre": "before the layers",
        "post": "logits gather",
        "stage transfer": "stage hand-off",
    },
}


def _x(value: float | None) -> str:
    return "n/a" if value is None else f"×{value:.2f}"


def _ratio(b: float | None, a: float | None) -> float | None:
    return b / a if a and b is not None else None


def _kib(value: float | None) -> str | None:
    return None if not value else f"{value / 1024:.0f} KiB"


def _phase(data: dict[str, Any]) -> dict[str, Any] | None:
    phases = data["phases"]
    return phases.get("decode") or next(iter(phases.values()), None)


def _merged(
    rows: list[dict[str, Any]], side: str, keep: Any
) -> dict[str, float] | None:
    items = [row[side] for row in rows if keep(row) and row.get(side)]
    count = sum(item["count"] for item in items)
    if not count:
        return None
    return {
        "count": count,
        "duration_us": sum(i["count"] * i["duration_us"] for i in items) / count,
        "xfer_us": sum(i["count"] * i["xfer_us"] for i in items) / count,
        "wait_us": sum(i["count"] * i["wait_us"] for i in items) / count,
        "total_us": sum(i["total_us"] for i in items),
    }


def _stage_means(ranks: list[dict[str, Any]], pick: Any) -> dict[int, float]:
    by_stage: dict[int, list[float]] = {}
    for row in ranks:
        by_stage.setdefault(row["stage"], []).append(pick(row["values"]))
    return {
        stage: statistics.fmean(values) for stage, values in sorted(by_stage.items())
    }


def _compute_of(values: dict[str, float]) -> float:
    return sum(values[c] for c in (*ta.COMPUTE, "overlap"))


def _bridge_notes(data: dict[str, Any], p: dict[str, Any], zh: bool) -> dict[str, Any]:
    a, b = data["a"], data["b"]
    same = data["mode"] == "same_gpus"
    factor = 1.0 if same else data["factor"]
    names = PILLAR_NAMES["zh" if zh else "en"]
    ta_us, tb_us = p["round_us"]["a"], p["round_us"]["b"]
    pillars = p["pillars"]
    ideal = pillars["ideal"]
    ideal_us = p["ideal_us"]
    share = f"{a['gpus']}/{b['gpus']}"
    if same:
        what = (
            "最上面一组是每个 token 的总时间，下面每组是一类时间，各类加起来正好等于总时间。"
            "每组两条：上面淡色的是 A，下面实色的是 B，竖线标出 A 的值；B 比 A 多出的部分画成红色斜线，比 A 少的部分画成绿色虚框。"
            if zh
            else "The top group is the time per token; each group below is one kind of time, and the kinds add up to the total. "
            "Each group has two bars, A on top (faded) and B below (solid), and a line at A's value: where B takes longer than A the excess "
            "is hatched red, where it takes less the saving is a dashed green box."
        )
    else:
        what = (
            "最上面一组是每个 token 的总时间，下面每组是一类时间，各类加起来正好等于总时间。"
            "每组两条：上面淡色的是 A 的实测，下面实色的是 B 的实测；实线是 B 的理想值。"
            f"计算可以分给更多 GPU，理想值是 A 的计算 × {share}；"
            "通信、流水线等待和空闲是每张 GPU 都要花的时间，不会因为 GPU 变多而减少"
            "（TP 下每个 token 的 all-reduce 次数和每次的数据量都与 GPU 数无关），理想值就是 A 的值。"
            f"总时间一组另有一条虚线，是完美线性扩展（A × {share}），虚线到实线之间是 A 中不能分摊的时间，再多的 GPU 也省不掉。"
            "B 超过实线的红色斜线是真正的扩展损失，各类的红色部分加起来等于总时间的红色部分；低于实线的绿色虚框表示比理想值还好。"
            if zh
            else "The top group is the time per token; each group below is one kind of time, and the kinds add up to the total. "
            "Each group has two bars, A's measurement on top (faded) and B's below (solid); the solid line is B's ideal. "
            f"More GPUs share the compute, so its ideal is A's compute × {share}; communication, pipeline wait and idle are spent "
            "by every GPU and do not shrink with more GPUs (TP sends the same all-reduces of the same size per token at any TP "
            "size), so their ideal is A's value. The total also has a dashed line at perfect linear scaling "
            f"(A × {share}); between the dashed and the solid line is the time of A that cannot be shared, which no number "
            "of GPUs removes. The hatched red part of B beyond the solid line is the real scaling loss, and the red parts of "
            "the kinds add up to the total's; a dashed green box means better than the ideal."
        )
    what += (
        "数值是每张 GPU 生成一个 token 的平均时间（与 TPOT 同口径）："
        "计算是 GPU 在跑计算 kernel；通信是只有通信、没有计算的时间（含等待其他 GPU 到齐）；"
        "流水线等待是等另一个 stage 交来数据（气泡）；空闲是 GPU 上没有任何 kernel，多为 CPU 侧发射间隙。"
        if zh
        else " Values are the average time one GPU spends per generated token (the TPOT basis): "
        "compute is time running compute kernels; communication is time only communicating, with no compute running "
        "(including waiting for peer GPUs); pipeline wait is waiting for another stage's data (the bubble); "
        "idle is no kernel at all, mostly host launch gaps."
    )
    reading: list[str] = []
    delta = tb_us - ideal_us
    shown = [name for name in names if max(pillars["a"][name], pillars["b"][name]) >= 1]
    moves = {name: pillars["b"][name] - ideal[name] for name in names}
    if same:
        reading.append(
            f"B（{b['parallel']}）每生成一个 token 需要 {_ms(tb_us)}，A（{a['parallel']}）为 {_ms(ta_us)}，"
            f"B {'慢' if delta > 0 else '快'} {abs(delta) / ta_us:.1%}（{_signed_ms(delta)}）。"
            if zh
            else f"B ({b['parallel']}) needs {_ms(tb_us)} per generated token vs {_ms(ta_us)} for A ({a['parallel']}): "
            f"{abs(delta) / ta_us:.1%} {'slower' if delta > 0 else 'faster'} ({_signed_ms(delta)})."
        )
        reading.append(
            ("按类型拆分：" if zh else "By kind: ")
            + ("，" if zh else "; ").join(
                (
                    f"{names[name]} {_ms(pillars['a'][name])} → {_ms(pillars['b'][name])}（{_signed_ms(moves[name])}）"
                    if zh
                    else f"{names[name]} {_ms(pillars['a'][name])} → {_ms(pillars['b'][name])} ({_signed_ms(moves[name])})"
                )
                for name in shown
            )
            + ("。" if zh else ".")
        )
    else:
        gap = tb_us - ta_us
        reading.append(
            f"A（{a['parallel']}，{a['gpus']} 卡）每个 token {_ms(ta_us)}，B（{b['parallel']}，{b['gpus']} 卡）{_ms(tb_us)}，"
            f"比 A {'还慢' if gap > 0 else '快'} {_ms(abs(gap))}：加速比 {p['speedup']:.2f}×，"
            f"完美线性扩展应为 {p['ideal_speedup']:.2f}×（{_ms(p['linear_us'])}），扩展效率 {_pct(p['scaling_efficiency'])}。"
            if zh
            else f"A ({a['parallel']}, {a['gpus']} GPUs) needs {_ms(ta_us)} per token and B ({b['parallel']}, {b['gpus']} GPUs) "
            f"{_ms(tb_us)}, {_ms(abs(gap))} {'slower' if gap > 0 else 'faster'} than A: speedup {p['speedup']:.2f}x, "
            f"perfect linear scaling would be {p['ideal_speedup']:.2f}x ({_ms(p['linear_us'])}), "
            f"scaling efficiency {_pct(p['scaling_efficiency'])}."
        )
        fixed = [
            name for name in names if name != "compute" and pillars["a"][name] >= 1
        ]
        if fixed:
            listed_fixed = ("、" if zh else " and ").join(
                f"{names[name]} {_ms(pillars['a'][name])}" for name in fixed
            )
            reading.append(
                f"完美线性扩展做不到：A 的 {_ms(ta_us)} 中只有计算 {_ms(pillars['a']['compute'])} 能分给更多 GPU，"
                f"{listed_fixed} 每张 GPU 都要花。即使计算完美缩到 ×{factor:.2f}、其余不变，B 也要 {_ms(ideal_us)}，"
                f"最多快 {p['best_speedup']:.2f}×。"
                if zh
                else f"Perfect linear scaling is out of reach: of A's {_ms(ta_us)} only the compute ({_ms(pillars['a']['compute'])}) "
                f"can be shared by more GPUs, every GPU still spends {listed_fixed}. Even with the compute cut to "
                f"×{factor:.2f} and the rest unchanged, B would need {_ms(ideal_us)}, at best {p['best_speedup']:.2f}x faster."
            )
        items = []
        for name in shown:
            target, after = ideal[name], pillars["b"][name]
            over = after >= target
            items.append(
                f"{names[name]} {_ms(pillars['a'][name])} → {_ms(after)}（理想 {_ms(target)}，{'多' if over else '少'} {_ms(abs(after - target))}）"
                if zh
                else f"{names[name]} {_ms(pillars['a'][name])} → {_ms(after)} (ideal {_ms(target)}, "
                f"{_ms(abs(after - target))} {'over' if over else 'under'})"
            )
        reading.append(
            ("每类时间 A → B：" if zh else "Each kind, A → B: ")
            + ("；" if zh else "; ").join(items)
            + ("。" if zh else ".")
        )
    worse = [name for name in shown if moves[name] >= max(100.0, 0.05 * abs(delta))]
    better = [name for name in shown if moves[name] <= -max(100.0, 0.05 * abs(delta))]

    def trend(name: str) -> str:
        before, after = pillars["a"][name], pillars["b"][name]
        if before < 1:
            return "A 没有这一项" if zh else "none in A"
        ratio = after / before
        if name == "compute":
            if ratio > factor:
                return (
                    f"只降到 ×{ratio:.2f}，理想 ×{factor:.2f}"
                    if zh
                    else f"shrinks only to ×{ratio:.2f}, ideal ×{factor:.2f}"
                )
            return (
                f"降到 ×{ratio:.2f}，好于理想 ×{factor:.2f}"
                if zh
                else f"shrinks to ×{ratio:.2f}, better than the ideal ×{factor:.2f}"
            )
        if ratio >= 1.05:
            return f"是 A 的 ×{ratio:.2f}" if zh else f"×{ratio:.2f} of A"
        if ratio > 0.95:
            return "与 A 基本相同" if zh else "about the same as A"
        return f"降到 A 的 ×{ratio:.2f}" if zh else f"down to ×{ratio:.2f} of A"

    def listed(items: list[str]) -> str:
        ordered = sorted(items, key=lambda name: -abs(moves[name]))
        if not same:
            return ("；" if zh else "; ").join(
                (
                    f"{names[name]} {_signed_ms(moves[name])}（{trend(name)}）"
                    if zh
                    else f"{names[name]} {_signed_ms(moves[name])} ({trend(name)})"
                )
                for name in ordered
            )
        return ("、" if zh else " and ").join(
            (
                f"{names[name]}（{_signed_ms(moves[name])}）"
                if zh
                else f"{names[name]} ({_signed_ms(moves[name])})"
            )
            for name in ordered
        )

    main, offset = (worse, better) if delta > 0 else (better, worse)
    if main:
        if same:
            if zh:
                text = f"{'B 变慢' if delta > 0 else 'B 变快'}主要来自{listed(main)}"
                text += f"；{listed(offset)}抵消了一部分。" if offset else "。"
            else:
                text = f"B is {'slower' if delta > 0 else 'faster'} mainly because of {listed(main)}"
                text += (
                    f"; {listed(offset)} {'offset' if len(offset) > 1 else 'offsets'} part of it."
                    if offset
                    else "."
                )
        elif zh:
            text = f"B 比理想值{'多出' if delta > 0 else '少了'}的 {_ms(abs(delta))} 来自：{listed(main)}"
            text += f"。另有{listed(offset)}，抵消了一部分。" if offset else "。"
        else:
            text = f"The {_ms(abs(delta))} B takes {'over' if delta > 0 else 'under'} its ideal comes from: {listed(main)}"
            text += (
                f". {listed(offset)[0].upper()}{listed(offset)[1:]} offsets part of it."
                if offset
                else "."
            )
        reading.append(text)
    return {"what": what, "reading": reading}


def _compute_notes(data: dict[str, Any], p: dict[str, Any], zh: bool) -> dict[str, Any]:
    a, b = data["a"], data["b"]
    what = (
        "每张 GPU 生成一个 token 花在计算上的时间，按 GPU 展开，分成 GEMM、attention 和其他计算。"
        "下表按并行方式推算每张 GPU 每个 token 要读多少权重和 KV cache：decode 阶段 batch 较小，GEMM 主要受读取权重的速度限制，"
        "attention 主要受读取 KV cache 的速度限制，所以这两项的时间应大致随对应的读取量变化（推算模型，以实测为准）。"
        if zh
        else "Compute time per generated token on every GPU, split into GEMM, attention and other compute. "
        "The table derives from the parallel layout how much of the weights and of the KV cache one GPU reads per token: "
        "with small decode batches GEMMs are bound by reading weights and attention by reading the KV cache, "
        "so those times should follow the reads (a model, check it against the measurements)."
    )
    work, rounds = p["work"], p["round"]
    reading: list[str] = []
    for side, head in (("a", a), ("b", b)):
        w = work[side]
        layers = f"{w['layers_per_gpu']:.0f}" if w.get("layers_per_gpu") else "?"
        reading.append(
            f"{side.upper()}（{head['parallel']}）每张 GPU 负责 {layers} 层、每个权重矩阵的 1/{w['tp']}，"
            f"每个 token 跑 {w['steps_per_token']:g} 个 step（每个 {w['sequences_per_step']:.0f} 条序列），"
            f"读取解码层权重的 {w['weight_share']:.1%}、KV cache 的 {w['kv_share']:.1%}。"
            if zh
            else f"{side.upper()} ({head['parallel']}): each GPU holds {layers} layers and 1/{w['tp']} of every weight matrix and runs "
            f"{w['steps_per_token']:g} step(s) of {w['sequences_per_step']:.0f} sequences per token, reading "
            f"{w['weight_share']:.1%} of the decoder weights and {w['kv_share']:.1%} of the KV cache."
        )
    rw = _ratio(work["b"]["weight_share"], work["a"]["weight_share"])
    rk = _ratio(work["b"]["kv_share"], work["a"]["kv_share"])
    gemm = (rounds["a"]["gemm"], rounds["b"]["gemm"])
    attn = (rounds["a"]["attention"], rounds["b"]["attention"])
    other = tuple(rounds[s]["other"] + rounds[s]["memcpy"] for s in ("a", "b"))
    rg, ra = _ratio(gemm[1], gemm[0]), _ratio(attn[1], attn[0])
    reading.append(
        f"实测每张 GPU 每个 token：GEMM {_ms(gemm[0])} → {_ms(gemm[1])}（{_x(rg)}，权重读取量 {_x(rw)}），"
        f"attention {_ms(attn[0])} → {_ms(attn[1])}（{_x(ra)}，KV 读取量 {_x(rk)}），其他计算 {_ms(other[0])} → {_ms(other[1])}。"
        if zh
        else f"Measured per GPU and token: GEMM {_ms(gemm[0])} → {_ms(gemm[1])} ({_x(rg)}, weight reads {_x(rw)}), "
        f"attention {_ms(attn[0])} → {_ms(attn[1])} ({_x(ra)}, KV reads {_x(rk)}), other compute {_ms(other[0])} → {_ms(other[1])}."
    )
    if rw and rg:
        wa, wb = work["a"], work["b"]
        reasons = []
        if wb["steps_per_token"] > wa["steps_per_token"]:
            reasons.append(
                f"B 把 batch 拆成 {wb['steps_per_token']:g} 个 micro-batch，每个 micro-batch 都要把本卡的权重完整读一遍"
                if zh
                else f"B splits the batch into {wb['steps_per_token']:g} micro-batches and each one reads the GPU's weights again"
            )
        if wb["tp"] < wa["tp"]:
            reasons.append(
                f"B 每张卡分到的权重矩阵更大（1/{wb['tp']} 对 1/{wa['tp']}）"
                if zh
                else f"each B GPU holds a larger slice of every matrix (1/{wb['tp']} vs 1/{wa['tp']})"
            )
        if rw > 1.05:
            if rg >= 0.75 * rw:
                text = (
                    ("；".join(reasons) + "，所以" if reasons else "")
                    + f"每张卡要读的权重变为 {_x(rw)}，GEMM 时间随之变为 {_x(rg)}：多出来的计算主要是重复读取权重。"
                    if zh
                    else ("; ".join(reasons) + ", so " if reasons else "")
                    + f"each GPU reads {_x(rw)} the weights and its GEMM time follows ({_x(rg)}): the extra compute is mostly re-reading weights."
                )
            else:
                text = (
                    f"权重读取量变为 {_x(rw)}，GEMM 只变为 {_x(rg)}：GEMM 并非完全受读权重限制，更大的矩阵块效率更高。"
                    if zh
                    else f"Weight reads change {_x(rw)} but GEMM time only {_x(rg)}: GEMMs are not purely bound by weight reads and larger slices run more efficiently."
                )
        elif rw < 0.95:
            if rg > 1.15 * rw:
                text = (
                    f"每张卡读取的权重降到 {_x(rw)}，但 GEMM 只降到 {_x(rg)}：矩阵切得更小后 GEMM kernel 效率下降（固定开销占比变大），计算没有按比例缩短。"
                    if zh
                    else f"Each GPU reads {_x(rw)} the weights but its GEMM time only drops to {_x(rg)}: smaller matrix slices run less efficiently (fixed per-kernel cost weighs more), so compute does not shrink in proportion."
                )
            else:
                text = (
                    f"GEMM 时间随权重读取量按比例下降（{_x(rg)} 对 {_x(rw)}）。"
                    if zh
                    else f"GEMM time falls with the weight reads ({_x(rg)} vs {_x(rw)})."
                )
        elif abs(rg - 1) > 0.1:
            text = (
                f"两边每张卡读取的权重相同，但 GEMM 时间变为 {_x(rg)}：差异来自 GEMM kernel 本身（见下方 kernel 表）。"
                if zh
                else f"Both read the same weights per GPU, yet GEMM time changes {_x(rg)}: the GEMM kernels themselves differ (see the kernel table)."
            )
        else:
            text = (
                "两边每张卡的 GEMM 工作量相同，时间也基本相同。"
                if zh
                else "Both do the same GEMM work per GPU and take about the same time."
            )
        reading.append(text)
    for side, head in (("a", a), ("b", b)):
        stages = _stage_means(p["ranks"][side], _compute_of)
        if len(stages) > 1:
            busiest = max(stages, key=stages.get)
            parts = ("，" if zh else ", ").join(
                f"S{stage} {_ms(value)}" for stage, value in stages.items()
            )
            last = busiest == max(stages)
            reading.append(
                f"{side.upper()} 各 stage 每张卡的计算时间：{parts}；S{busiest} 最多"
                + (
                    "（最后一个 stage 还要跑 LM head、logits 和采样）。"
                    if last
                    else "。"
                )
                if zh
                else f"{side.upper()} compute per GPU by stage: {parts}; S{busiest} is busiest"
                + (
                    " (the last stage also runs the LM head, logits and sampling)."
                    if last
                    else "."
                )
            )
    return {"what": what, "reading": reading}


def _communication_notes(
    data: dict[str, Any], p: dict[str, Any], zh: bool
) -> dict[str, Any]:
    what = (
        "生成一个 token 时，每种通信发生多少次、每次多久。单次时间拆成传输（同一次通信中最快那张卡的 kernel 时间）"
        "和等待（先到的卡等最慢的卡到齐）。流水线等待是某个 stage 在等另一个 stage 交来数据（气泡），不是数据传输本身。"
        "下方逐 GPU 展示每张卡在一个 token 里花在通信和等待上的时间。"
        if zh
        else "How often each kind of communication happens per generated token and how long one call takes. "
        "A call's time splits into transfer (the fastest GPU's kernel time for that call) and wait (early GPUs waiting for the last one). "
        "Pipeline wait is a stage waiting for another stage's data (the bubble), not the transfer itself. "
        "Below, every GPU's communication and waiting per token."
    )
    rows, work = p["collectives"], p["work"]
    reading: list[str] = []
    layer = {
        side: _merged(
            rows,
            side,
            lambda row: row["dim"] == "tp"
            and row["op"] == "all_reduce"
            and row["role"] in LAYER_ROLES,
        )
        for side in ("a", "b")
    }
    sizes = [_kib(work[side].get("allreduce_bytes")) for side in ("a", "b")]
    if layer["a"] or layer["b"]:
        parts = []
        for side in ("a", "b"):
            stats = layer[side]
            if not stats:
                parts.append(f"{side.upper()} " + ("无" if zh else "none"))
                continue
            parts.append(
                f"{side.upper()} 每张卡每个 token {stats['count']:.0f} 次 × {stats['duration_us']:.1f} µs"
                f"（传输 {stats['xfer_us']:.1f} + 等待 {stats['wait_us']:.1f}）= {_ms(stats['total_us'])}，在 {work[side]['tp']} 张卡之间"
                if zh
                else f"{side.upper()} {stats['count']:.0f} calls × {stats['duration_us']:.1f} µs (transfer {stats['xfer_us']:.1f} + wait {stats['wait_us']:.1f}) "
                f"= {_ms(stats['total_us'])} per GPU and token, among {work[side]['tp']} GPUs"
            )
        text = (
            "层内 TP all-reduce：" if zh else "TP all-reduce inside the layers: "
        ) + ("；" if zh else "; ").join(parts)
        if all(sizes):
            text += (
                f"；每次数据量约 {sizes[0]} → {sizes[1]}（估算：每步序列数 × hidden size × 数据类型字节数）"
                if zh
                else f"; about {sizes[0]} → {sizes[1]} per call (estimate: sequences per step × hidden size × bytes per value)"
            )
        reading.append(text + ("。" if zh else "."))
        if layer["a"] and layer["b"]:
            ta_call, tb_call = layer["a"]["duration_us"], layer["b"]["duration_us"]
            tp_a, tp_b = work["a"]["tp"], work["b"]["tp"]
            bytes_a, bytes_b = (
                work[side].get("allreduce_bytes") for side in ("a", "b")
            )
            causes = []
            if tp_b != tp_a:
                causes.append(
                    f"参与的 GPU 从 {tp_a} 张{'减少' if tp_b < tp_a else '增加'}到 {tp_b} 张"
                    if zh
                    else f"{tp_a} → {tp_b} GPUs per call"
                )
            if bytes_a and bytes_b and abs(bytes_b / bytes_a - 1) > 0.05:
                causes.append(
                    f"每次的数据量{'减半' if abs(bytes_b / bytes_a - 0.5) < 0.05 else ('变小' if bytes_b < bytes_a else '变大')}"
                    if zh
                    else f"{'half' if abs(bytes_b / bytes_a - 0.5) < 0.05 else ('smaller' if bytes_b < bytes_a else 'larger')} messages"
                )
            if abs(tb_call / ta_call - 1) > 0.1:
                totals = layer["a"]["total_us"], layer["b"]["total_us"]
                counts = layer["a"]["count"], layer["b"]["count"]
                if zh:
                    text = f"单次 all-reduce 从 {ta_call:.1f} 变为 {tb_call:.1f} µs（{_x(tb_call / ta_call)}）"
                    text += f"：{'、'.join(causes)}" if causes else ""
                    text += f"；每个 token 调用 {counts[0]:.0f} → {counts[1]:.0f} 次，合计 {_ms(totals[0])} → {_ms(totals[1])}"
                    text += f"（{_signed_ms(totals[1] - totals[0])}）。"
                else:
                    text = f"One all-reduce takes {ta_call:.1f} → {tb_call:.1f} µs ({_x(tb_call / ta_call)})"
                    text += f": {', '.join(causes)}" if causes else ""
                    text += f"; per token {counts[0]:.0f} → {counts[1]:.0f} calls, {_ms(totals[0])} → {_ms(totals[1])}"
                    text += f" ({_signed_ms(totals[1] - totals[0])})."
                reading.append(text)
    if data["mode"] != "same_gpus" and layer["a"] and layer["b"]:
        reading.append(
            "GPU 变多后，每张卡每个 token 仍要做同样次数、同样数据量的 all-reduce（数据量取决于每步序列数和 hidden size，"
            "与 TP 大小无关），通信分摊不到更多 GPU 上，所以它的理想值就是 A 的值；参与的 GPU 越多，单次 all-reduce 通常越慢。"
            if zh
            else "With more GPUs every GPU still runs as many all-reduces of the same size per token (the size depends on "
            "the sequences per step and the hidden size, not on the TP size): communication is not shared by the GPUs, so "
            "its ideal is A's value, and more GPUs per all-reduce usually make each call slower."
        )
    others = [
        row
        for row in rows
        if row["dim"] != "pp"
        and not (row["op"] == "all_reduce" and row["role"] in LAYER_ROLES)
        and abs(
            (row.get("b") or {}).get("total_us", 0.0)
            - (row.get("a") or {}).get("total_us", 0.0)
        )
        >= 50
    ]
    if others:
        roles = ROLE_NAMES["zh" if zh else "en"]
        items = []
        for row in others:
            calls = ("，" if zh else ", ").join(
                f"{side.upper()} {row[side]['count']:.1f} {'次 ' if zh else ''}× {row[side]['duration_us']:.1f} µs"
                for side in ("a", "b")
                if row.get(side)
            )
            name = f"{row['dim'].upper()} {row['op'].replace('_', '-')}"
            role = roles.get(row["role"], row["role"])
            items.append(
                f"{name}（{role}）：{calls}" if zh else f"{name} ({role}): {calls}"
            )
        reading.append(
            (
                "其他集合通信，每张卡每个 token："
                if zh
                else "Other collectives per GPU and token: "
            )
            + ("；" if zh else "; ").join(items)
            + ("。" if zh else ".")
        )
    for side in ("a", "b"):
        waits = {row["rank"]: row["values"]["comm_wait"] for row in p["ranks"][side]}
        if len(waits) > 1 and max(waits.values()) - min(waits.values()) >= 200:
            early = max(waits, key=waits.get)
            late = min(waits, key=waits.get)
            reading.append(
                f"{side.upper()} 中 GPU {early} 每个 token 等其他卡 {_ms(waits[early])}，GPU {late} 只等 {_ms(waits[late])}：GPU {late} 通常最后到达（计算更慢或发射更晚）。"
                if zh
                else f"In {side.upper()}, GPU {early} waits {_ms(waits[early])} per token for its peers and GPU {late} only {_ms(waits[late])}: GPU {late} usually arrives last (slower compute or later launches)."
            )
    for side in ("a", "b"):
        pipe = p["pillars"][side]["pipeline"]
        if pipe < 10:
            continue
        hand = _merged(rows, side, lambda row: row["dim"] == "pp")
        bubbles = _stage_means(p["ranks"][side], lambda values: values["bubble"])
        waiting = max(bubbles, key=bubbles.get)
        busy = _stage_means(p["ranks"][side], lambda values: _compute_of(values))
        bottleneck = max(busy, key=busy.get)
        hand_text = (
            (
                f"stage 之间传激活值、广播采样结果本身很快（每次传输约 {hand['xfer_us']:.1f} µs）；"
                if zh
                else f"passing activations and sampled tokens between stages is cheap (about {hand['xfer_us']:.1f} µs per transfer); "
            )
            if hand
            else ""
        )
        reading.append(
            f"{side.upper()} 的流水线：{hand_text}主要开销是流水线等待，S{waiting} 的卡每个 token 平均等 {_ms(bubbles[waiting])}"
            f"（S{bottleneck} 计算最多，其他 stage 要等它），全部 GPU 平均 {_ms(pipe)}。"
            if zh
            else f"{side.upper()}'s pipeline: {hand_text}the cost is waiting: S{waiting} GPUs wait {_ms(bubbles[waiting])} per token "
            f"(S{bottleneck} computes most and the others wait for it), {_ms(pipe)} per GPU on average."
        )
    hidden = [p["round"][side]["overlap"] for side in ("a", "b")]
    reading.append(
        (
            "通信和计算在同一个 stream 上串行执行，没有被计算遮盖，所以通信时间直接加在每个 token 上。"
            if zh
            else "Communication runs on the compute stream, so none of it hides under compute: its time adds directly to every token."
        )
        if max(hidden) < 50
        else (
            f"被计算遮盖的通信：A {_ms(hidden[0])}，B {_ms(hidden[1])}（这部分不增加时间）。"
            if zh
            else f"Communication hidden under compute: A {_ms(hidden[0])}, B {_ms(hidden[1])} (it adds no time)."
        )
    )
    return {"what": what, "reading": reading}


def _timeline_totals(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    lanes = []
    for lane in timeline.get("ranks", []):
        totals: dict[str, float] = {}
        for start, end, kind in lane["spans"]:
            totals[kind] = totals.get(kind, 0.0) + end - start
        lanes.append({"rank": lane["rank"], "stage": lane["stage"], "totals": totals})
    return lanes


def _timeline_notes(
    data: dict[str, Any], p: dict[str, Any], zh: bool
) -> dict[str, Any]:
    timelines = p.get("timeline") or {}
    count = (
        max((t or {}).get("median_of", 0) for t in timelines.values())
        if timelines
        else 0
    )
    what = (
        f"选一个有代表性的 token（采样到的 {count} 个中时长最接近中位数的那个），把每张 GPU 在这段时间里做的事按时间顺序画出来，A 和 B 用同一时间轴。"
        "蓝色是计算，橙色是通信传输，黄色是等其他 GPU 到齐，紫色是 stage 交接，红色是等另一个 stage（流水线气泡），灰色是空闲。"
        "流水线并行时一个 token 包含每个 micro-batch 各一个 step。"
        if zh
        else f"One representative token (of {count} sampled, the one closest to the median length), every GPU's activity in time order, A and B on the same time axis. "
        "Blue is compute, orange communication transfer, yellow waiting for peer GPUs, purple stage hand-off, red waiting for another stage (pipeline bubble), gray idle. "
        "With pipeline parallelism one token spans one step per micro-batch."
    )
    reading: list[str] = []
    for side in ("a", "b"):
        timeline = timelines.get(side)
        if not timeline:
            continue
        lanes = _timeline_totals(timeline)

        def mean(kind: str, lanes: list[dict[str, Any]] = lanes) -> float:
            return (
                statistics.fmean(lane["totals"].get(kind, 0.0) for lane in lanes)
                if lanes
                else 0.0
            )

        waits = mean("bubble") >= 1
        reading.append(
            f"{side.upper()} 的这个 token 用时 {_ms(timeline['window_us'])}（{timeline['steps']} 个 step）：每张卡平均计算 {_ms(mean('compute') + mean('overlap'))}、"
            f"通信 {_ms(mean('comm_xfer') + mean('comm_wait'))}、"
            + (f"等其他 stage {_ms(mean('bubble'))}、" if waits else "")
            + f"空闲 {_ms(mean('idle'))}。"
            if zh
            else f"{side.upper()}'s token takes {_ms(timeline['window_us'])} ({timeline['steps']} step{'' if timeline['steps'] == 1 else 's'}): "
            f"per GPU on average compute {_ms(mean('compute') + mean('overlap'))}, "
            f"communication {_ms(mean('comm_xfer') + mean('comm_wait'))}, "
            + (f"waiting for another stage {_ms(mean('bubble'))}, " if waits else "")
            + f"idle {_ms(mean('idle'))}."
        )
        stages = sorted({lane["stage"] for lane in lanes})
        if len(stages) > 1:
            parts = []
            for stage in stages:
                members = [lane for lane in lanes if lane["stage"] == stage]
                parts.append(
                    f"S{stage} 计算 {_ms(mean('compute', members) + mean('overlap', members))}、等其他 stage {_ms(mean('bubble', members))}"
                    if zh
                    else f"S{stage} computes {_ms(mean('compute', members) + mean('overlap', members))} and waits {_ms(mean('bubble', members))} for another stage"
                )
            reading.append(
                f"{side.upper()} 按 stage：" + "；".join(parts) + "。"
                if zh
                else f"{side.upper()} by stage: " + "; ".join(parts) + "."
            )
    return {"what": what, "reading": reading}


def _layer_notes(data: dict[str, Any], p: dict[str, Any], zh: bool) -> dict[str, Any]:
    what = (
        "每层占每个 token 的时间：这一层在所在 stage 的 GPU 上的时间 × 该 stage 占全部 GPU 的比例。所有层加上层外部分"
        "（embedding、LM head、采样、流水线交接和气泡）正好等于每 token 时间。柱状图中每层 A 在左（浅色）、B 在右；悬停可看该层在本 stage GPU 上的实际耗时。"
        if zh
        else "Each layer's share of the time per token: its time on the GPUs of its pipeline stage × that stage's share of all GPUs. "
        "The layers plus the work outside them (embedding, LM head, sampling, pipeline hand-off and bubble) add up to the time per token. "
        "In the chart every layer shows A on the left (lighter) and B on the right; hover for its time on its stage's GPUs."
    )
    reading: list[str] = []
    factor = 1.0 if data["mode"] == "same_gpus" else data["factor"]
    parts = p.get("parts")
    layers = [row for row in p.get("layers", []) if row.get("change_us") is not None]
    if parts and layers:
        names = PILLAR_NAMES["zh" if zh else "en"]
        pieces = []
        for label, values in (
            (
                (
                    f"{len(layers)} 个解码层合计"
                    if zh
                    else f"the {len(layers)} decoder layers"
                ),
                parts["layers"],
            ),
            (
                (
                    "层外（embedding、LM head、采样、流水线交接和气泡）"
                    if zh
                    else "the work outside them (embedding, LM head, sampling, pipeline hand-off and bubble)"
                ),
                parts["outside"],
            ),
        ):
            group_a = ta._pillars(ta.ideal_values(values["a"], factor))
            group_b = ta._pillars(values["b"])
            moved = {name: group_b[name] - group_a[name] for name in group_a}
            top = [
                (name, value)
                for name, value in sorted(
                    moved.items(), key=lambda item: -abs(item[1])
                )[:2]
                if abs(value) >= 50
            ]
            detail = ("、" if zh else ", ").join(
                f"{names[name]} {_signed_ms(value)}" for name, value in top
            )
            pieces.append(
                f"{label}{'' if zh and label.endswith('）') else ' '}{_signed_ms(sum(moved.values()))}"
                + (
                    f"（{detail}）"
                    if zh and detail
                    else (f" ({detail})" if detail else "")
                )
            )
        reading.append(
            ("按位置拆分：" if zh else "By location: ")
            + ("；" if zh else "; ").join(pieces)
            + ("。" if zh else ".")
        )
    if layers:
        changes = [row["change_us"] for row in layers]
        worst = sorted(layers, key=lambda row: -abs(row["change_us"]))[:3]
        reading.append(
            f"每层占每个 token 的时间变化 {min(changes):+.0f} 到 {max(changes):+.0f} µs，中位数 {statistics.median(changes):+.0f} µs；"
            f"变化最大：{'、'.join(f'{row['label']} {row['change_us']:+.0f} µs' for row in worst)}。"
            if zh
            else f"Per layer the share of the token changes by {min(changes):+.0f} to {max(changes):+.0f} µs (median {statistics.median(changes):+.0f} µs); "
            f"largest: {', '.join(f'{row['label']} {row['change_us']:+.0f} µs' for row in worst)}."
        )
        by_stage: dict[int, list[float]] = {}
        for row in layers:
            if row.get("total_us_b") is not None:
                by_stage.setdefault(row["stage"][1], []).append(row["total_us_b"])
        if len(by_stage) > 1:
            text = ("，" if zh else ", ").join(
                f"S{stage} {statistics.fmean(values):.0f} µs"
                for stage, values in sorted(by_stage.items())
            )
            reading.append(
                f"B 各 stage 的层平均每层占 {text}。"
                if zh
                else f"B's layers take on average {text} of the token per layer by stage."
            )
    return {"what": what, "reading": reading}


def _kernel_notes(data: dict[str, Any], p: dict[str, Any], zh: bool) -> dict[str, Any]:
    what = (
        "每张 GPU 每个 token 上耗时变化最大的 kernel，用来确认具体是哪些 kernel 变快或变慢。"
        "表中是 kernel 从开始到结束的原始时长：通信 kernel 启动后要等其他 GPU 或另一个 stage，"
        "这段等待期间 GPU 可能还在跑计算，所以通信的实际开销以“通信模式开销”为准。"
        if zh
        else "The kernels whose time per GPU and token changed most, to see which kernels got faster or slower. "
        "Times are raw kernel durations from start to end: a communication kernel waits for its peers or another stage "
        "after it starts, possibly while the GPU still computes, so take the cost of communication from "
        "the communication pattern instead."
    )
    kernels = p.get("kernels") or []
    reading = []
    if kernels:
        reading.append(
            ("变化最大：" if zh else "Largest changes: ")
            + ("；" if zh else "; ").join(
                (
                    f"{ta._kernel_name(row['name'], 60)}（{row['kind']}）{row['change_us']:+.0f} µs"
                    if zh
                    else f"{ta._kernel_name(row['name'], 60)} ({row['kind']}) {row['change_us']:+.0f} µs"
                )
                for row in kernels[:3]
            )
            + ("。" if zh else ".")
        )
    for side in ("a", "b"):
        raw = sum(row[f"{side}_us"] for row in kernels if row["kind"] == "comm")
        pillars = p["pillars"][side]
        cost = pillars["communication"] + pillars["pipeline"]
        if raw - cost < max(1000.0, 0.2 * cost):
            continue
        name = (
            f"{side.upper()}（{data[side]['parallel']}）"
            if zh
            else f"{side.upper()} ({data[side]['parallel']})"
        )
        reading.append(
            f"{name}的通信 kernel 原始时长合计 {_ms(raw)}，但通信和流水线等待只占 {_ms(cost)}：多出的 "
            f"{_ms(raw - cost)} 是这些 kernel 启动后在等待，期间 GPU 同时在跑计算或其他通信，或计入了空闲"
            "（流水线并行的接收 kernel 会提前启动），不会额外加在每个 token 上。"
            if zh
            else f"{name}'s communication kernels run {_ms(raw)} in total, but communication and pipeline wait "
            f"take only {_ms(cost)}: the other {_ms(raw - cost)} is these kernels waiting after they start while "
            "the GPU computes, runs other communication or is counted as idle (pipeline receives start early), "
            "so it adds nothing to the token."
        )
    return {"what": what, "reading": reading}


def chart_notes(data: dict[str, Any], language: str) -> dict[str, Any]:
    """For every chart of the Analysis page: what it shows and what the
    current numbers say, built from the data alone."""
    zh = language == "zh"
    notes: dict[str, Any] = {"labels": NOTE_LABELS[language]}
    p = _phase(data)
    if p is None:
        return notes
    for key, build in (
        ("bridge", _bridge_notes),
        ("compute", _compute_notes),
        ("communication", _communication_notes),
        ("timeline", _timeline_notes),
        ("layers", _layer_notes),
        ("kernels", _kernel_notes),
    ):
        notes[key] = build(data, p, zh)
    return notes


def rule_narrative(data: dict[str, Any], language: str) -> str:
    """Report body with the template's sections, built from the numbers alone."""
    zh = language == "zh"
    names = SECTIONS[language]
    p = _phase(data)
    if p is None:
        return "\n\n".join(f"## {name}\n\n-" for name in names)
    same = data["mode"] == "same_gpus"
    factor = 1.0 if same else data["factor"]
    notes = chart_notes(data, language)
    ta_us, tb_us = p["round_us"]["a"], p["round_us"]["b"]
    ma, mb = p["metrics"]["a"], p["metrics"]["b"]
    spr = p["steps_per_round"]
    stages = {side: p["stages"][side] for side in ("a", "b")}
    layers = [row for row in p.get("layers", []) if row["change_us"] is not None]
    change = p["change"]

    def busy(stage: dict[str, Any]) -> float:
        return (
            stage["metrics"]["total_us"]
            - stage["round"]["bubble"]
            - stage["round"]["idle"]
        )

    bridge = notes["bridge"]["reading"]
    compute = notes["compute"]["reading"]
    communication = notes["communication"]["reading"]
    conclusion = list(bridge)
    if len(compute) >= 4:
        conclusion.append(compute[3])
    if communication:
        conclusion.append(
            communication[1]
            if len(communication) > 1 and "all-reduce" in communication[1]
            else communication[0]
        )
    pipeline = [
        line
        for line in communication
        if line.startswith(("A 的流水线", "B 的流水线", "A's pipeline", "B's pipeline"))
    ]
    conclusion.extend(pipeline[:1])

    pillar_names = PILLAR_NAMES[language]
    pillars = p["pillars"]
    metrics = [
        (
            "每 token 时间" if zh else "Time per token",
            ta_us,
            tb_us,
            p["ideal_us"],
            False,
        )
    ] + [
        (
            pillar_names[name] if zh else pillar_names[name].capitalize(),
            pillars["a"][name],
            pillars["b"][name],
            pillars["ideal"][name],
            True,
        )
        for name in pillar_names
        if max(pillars["a"][name], pillars["b"][name]) >= 1
    ]

    def timed(value: float, total: float, part: bool) -> str:
        return (
            f"{_ms(value)} ({_pct(value / total if total else None)})"
            if part
            else _ms(value)
        )

    head = ["指标" if zh else "Metric", "A", "B"]
    if same:
        head.append("变化" if zh else "Change")
    else:
        head += [
            (
                f"理想值（计算 ×{factor:.3g}，其余同 A）"
                if zh
                else f"Ideal B (compute ×{factor:.3g}, rest as in A)"
            ),
            "B − 理想值" if zh else "B − ideal",
        ]
    table = ["| " + " | ".join(head) + " |", "|---" + "|---:" * (len(head) - 1) + "|"]
    for label, before, after, target, part in metrics:
        cells = [label, timed(before, ta_us, part), timed(after, tb_us, part)]
        if not same:
            cells.append(_ms(target))
        cells.append(_signed_ms(after - target))
        table.append("| " + " | ".join(cells) + " |")

    recs: list[str] = []
    for side, label, metrics in (("b", "B", mb), ("a", "A", ma)):
        if len(stages[side]) == 2 and (metrics["bubble_share"] or 0) >= 0.05:
            s0, s1 = stages[side]
            gap = busy(s1) - busy(s0)
            per_layer = (
                statistics.fmean(
                    row[f"wall_us_{side}"]
                    for row in layers
                    if row[f"wall_us_{side}"] is not None
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
                    f"- 平衡流水线（{label}）：S{1 if gap > 0 else 0} 每个 token 多忙 {_ms(abs(gap))}，每层约 {per_layer:.0f} µs；可尝试 `VLLM_PP_LAYER_PARTITION={split}`，估计 stage 间差距降到约 {_ms(left)}（估算，需实测）。"
                    if zh
                    else f"- Balance the pipeline ({label}): S{1 if gap > 0 else 0} is busier by {_ms(abs(gap))} per token at about {per_layer:.0f} µs per layer; try `VLLM_PP_LAYER_PARTITION={split}`, estimated to cut the stage gap to about {_ms(left)} per token (estimate; measure it)."
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
            f"- B 每张卡每个 token 有 {_ms(mb['exposed_comm_us'])} 的暴露通信（占 {_pct(mb['exposed_comm_share'])}）：all-reduce 与计算串行。可评估更小的 TP（配合 DP 提升吞吐）、融合 all-reduce+RMSNorm 或通信计算重叠方案（估计收益上限为暴露通信时间）。"
            if zh
            else f"- B spends {_ms(mb['exposed_comm_us'])} per GPU and token in exposed communication ({_pct(mb['exposed_comm_share'])}): all-reduces serialize with compute. Evaluate smaller TP (with DP for throughput), fused all-reduce + RMSNorm or communication/compute overlap (upper bound: the exposed time)."
        )
    if (mb["idle_share"] or 0) >= 0.15:
        recs.append(
            f"- B 每张卡每个 token 有 {_ms(mb['idle_us'])}（{_pct(mb['idle_share'])}）GPU 空闲（没有 kernel 运行，多为 host 侧发射间隙）：检查 CUDA graph 是否覆盖该 batch size，减少每 step 的 Python 开销。"
            if zh
            else f"- B's GPUs are idle {_ms(mb['idle_us'])} per token ({_pct(mb['idle_share'])}) (no kernel running, mostly host launch gaps): check CUDA graph coverage for this batch size and per-step Python overhead."
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
            f"- 每 token 时间差的 step 间标准误为 {se:.1f} µs；结论只覆盖本次采样窗口，跨运行波动需重复运行确认。"
            if zh
            else f"- Step-to-step standard error of the per-token difference is {se:.1f} µs; run-to-run variance needs repeated runs."
        )

    body = {
        names[0]: [f"- {line}" for line in conclusion],
        names[1]: table,
        names[2]: [f"- {line}" for line in compute] or ["-"],
        names[3]: [f"- {line}" for line in communication] or ["-"],
        names[4]: [f"- {line}" for line in notes["layers"]["reading"]] or ["-"],
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
