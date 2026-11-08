"""The roofline section of the report: one log-log chart per (platform, GPU), plus the same numbers as tables.

Runs are grouped into panels by GPU *and* platform (`invariants.platform`; absent = the HUIT cluster), so a
cluster L4 and a Modal L4 never share a roof: the Modal L4 measured 3-18% slower. Panels are small
multiples, one full-size chart each on the same log axes, ordered by datasheet bf16 peak. One chart with a
series per GPU was the alternative; with five GPUs that is five roofs and ~55 labelled points on one plot,
where neither the roofs nor the labels stay readable. `roofs()` lists every panel's ceilings and
`by_gpu()` puts each model's achieved TFLOP/s side by side across panels.

Each model's latest `roofline` run is a filled point at (cost-model intensity, achieved TFLOP/s), with a faint
trajectory through its prompt-length bins; its latest `s1 ncu` run (roofline_hw_*) adds a hollow point at the
hardware-counter intensity and FLOP/s. Roofs come from the runs' measured ceilings (solid: bf16 tensor-core
GEMM, fp32 CUDA-core GEMM, DRAM copy; medians over the panel's runs) and the datasheet (dashed:
`roofline.spec_peaks(gpu)`, else the peaks the runs recorded; none drawn when neither knows the GPU).

Color encodes the architecture family, read from the metrics (MoE when active parameters are under half the
total; else hybrid when some layers are not attention): three hues, the palette's all-pairs-safe slots.
Identity is a direct label on every point, never color alone, and the table repeats every value. Decision
models are full-size points; zero-shot base models (references for the architecture, not decision models)
are small and faint.

`recipes()` lays the decision models' design choices (RECIPES: backbone, adaptation, readout, serving) next
to what they cost on the GPU and what they get right, and `findings()` reads the analysis written for the
current runs from results/roofline_findings.md (prose about the records, kept with them).
"""

from __future__ import annotations

import html
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .benchmarks import roofline as bench_roofline
from .models.catalog import CATALOG
from .record import Metrics, RunRecord, Scalar

BENCH = "roofline"
DASH = "\N{EN DASH}"  # a missing value
FAMILIES = ("Dense, hybrid mixer", "Dense, attention only", "MoE")
CSS = """
:root{--fam-0:#2a78d6;--fam-1:#eb6834;--fam-2:#1baf7a}
@media (prefers-color-scheme:dark){:root{--fam-0:#3987e5;--fam-1:#d95926;--fam-2:#199e70}}
.roof .grid{stroke:var(--rule);stroke-width:1}.roof .axis{stroke:var(--ink-2);stroke-width:1}
.roof .ceil{stroke:var(--ink-2);stroke-width:1.5;fill:none}.roof .spec{stroke-dasharray:5 4;opacity:.7}
.roof .traj{fill:none;stroke-width:1.5;opacity:.45}
.roof .legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:4px 0 10px;color:var(--ink-2);font-size:12px}
.roof .key{display:inline-flex;align-items:center;gap:6px}
.roof .sw{width:10px;height:10px;border-radius:50%;display:inline-block}
.roof g.pt:hover circle.dot{stroke-width:3}
.roof .ref{opacity:.55}
"""
FINDINGS = "roofline_findings.md"

# What each team did to a base model to make it a decision model (catalog names; see models/catalog.py).
# Columns: backbone, adaptation, readout, serving.
RECIPES = {
    "elephant-1": (
        "Qwen3.5-9B (Gated DeltaNet hybrid)",
        "full fine-tune",
        "slot-token logits, fp32 LM head",
        "own engine (torch 2.8): block-bidirectional attention, state prefilled once, questions forked "
        "from it in one batched pass",
    ),
    "jevk5-9b": (
        "Qwen3.5-9B (Gated DeltaNet hybrid)",
        "full fine-tune",
        "option-letter softmax over the vocab head, temperature 1.316",
        "plain transformers, one causal pass per question",
    ),
    "kev-4b.hybrid": (
        "Qwen3.5-4B-Base (Gated DeltaNet hybrid)",
        "LoRA (merged) + pointer head",
        "pointer over option markers: no vocab head",
        "prefix cache, fused Triton DeltaNet kernels, CUDA graphs over padded shape buckets",
    ),
    "kev-4b.qwen3": (
        "Qwen3-4B-Base (attention only)",
        "LoRA (merged) + pointer head",
        "pointer over option markers: no vocab head",
        "prefix cache; questions packed in one row under a block mask; eager",
    ),
    "laya.english": (
        "ModernBERT-large encoder (395M)",
        "fine-tuned encoder + MLP head",
        "head on each option's [MASK]",
        "fp32 weights under bf16 autocast; state cut to 512 tokens",
    ),
    "laya.multilingual": (
        "mmBERT-base encoder (307M)",
        "fine-tuned encoder + MLP head",
        "head on each option's [MASK]",
        "fp32 weights under bf16 autocast; state cut to 1024 tokens",
    ),
    "qwen35-9b.zeroshot": (
        "Qwen3.5-9B (Gated DeltaNet hybrid)",
        "none (zero-shot control: JevK5's and elephant-1's base)",
        "option-letter softmax over the vocab head",
        "plain transformers",
    ),
}


def is_decision_model(model: str) -> bool:
    """Trained for decisions, as opposed to a zero-shot base model run through the same readout."""
    return not model.endswith(".zeroshot")


def family(m: Metrics, model: str = "") -> int:
    """The catalog's group decides MoE for a known model (a small encoder's embedding is most of its
    parameters, which the active/total rule would misread); otherwise active < half of total is MoE."""
    group = next((spec.group for spec in CATALOG if spec.name == model), None)
    total, active = m.get("roofline_params_b", math.nan), m.get("roofline_active_params_b", math.nan)
    moe = group.startswith("MoE") if group is not None else active < 0.5 * total
    if moe:
        return 2
    return 1 if m.get("roofline_attention_layers", 1.0) >= 1.0 else 0


@dataclass(frozen=True)
class Point:
    model: str
    run_id: str
    m: Metrics
    hw: Metrics  # the model's latest ncu run on this GPU ({} if none)

    @property
    def family(self) -> int:
        return family(self.m, self.model)


def _model(r: RunRecord) -> str:
    return str(r.params.get("model", "?"))


def _is_ncu(r: RunRecord) -> bool:
    return bool(r.params.get("roofline_ncu", 0))


def platform(invariants: dict[str, Scalar]) -> str:
    """Where a run executed, for labels: "Modal", or "cluster" when `platform` is absent (HUIT)."""
    p = invariants.get("platform")
    if p is None:
        return "cluster"
    return "Modal" if str(p).lower() == "modal" else str(p)


def gpu_name(gpu: str) -> str:
    return gpu.removeprefix("NVIDIA ")


@dataclass(frozen=True)
class Roof:
    bf16: float  # TFLOP/s, nan when unknown
    fp32: float
    bw: float  # GB/s

    @property
    def ridge(self) -> float:
        """FLOP/byte where the bf16 roof meets the memory slope."""
        return self.bf16 * 1e3 / self.bw if self.bw > 0 else math.nan


@dataclass(frozen=True)
class Panel:
    """Every model's latest run on one GPU type on one platform: one roof, one chart."""

    gpu: str  # as nvidia-smi reports it, e.g. "NVIDIA H100 80GB HBM3"
    platform: str  # "cluster", "Modal", ...
    points: list[Point]

    @property
    def label(self) -> str:
        return f"{gpu_name(self.gpu)} ({self.platform})"

    @property
    def measured(self) -> Roof:
        return Roof(
            _median(self.points, "roofline_peak_bf16_tflops"),
            _median(self.points, "roofline_peak_fp32_tflops"),
            _median(self.points, "roofline_peak_bw_gbs"),
        )

    @property
    def datasheet(self) -> Roof:
        """The current datasheet table; the peaks the runs recorded when it does not know the GPU."""
        if (spec := bench_roofline.spec_peaks(self.gpu)) is not None:
            return Roof(*spec)
        return Roof(
            _median(self.points, "roofline_spec_bf16_tflops"),
            _median(self.points, "roofline_spec_fp32_tflops"),
            _median(self.points, "roofline_spec_bw_gbs"),
        )


def panels(recs: Sequence[RunRecord]) -> list[Panel]:
    """Latest clean run per (platform, GPU, model), joined with the latest ncu run of the same triple.
    Ordered by datasheet bf16 peak (unknown last), then GPU name, the cluster before other platforms."""
    runs = sorted((r for r in recs if BENCH in r.results), key=lambda r: r.run_id)
    clean: dict[tuple[str, str, str], RunRecord] = {}
    hw: dict[tuple[str, str, str], RunRecord] = {}
    for r in runs:
        key = (str(r.invariants.get("gpu", "?")), platform(r.invariants), _model(r))
        if _is_ncu(r):
            if "roofline_hw_intensity" in r.results[BENCH]:
                hw[key] = r
        else:
            clean[key] = r
    grouped: dict[tuple[str, str], list[Point]] = {}
    for (gpu, plat, model), r in sorted(clean.items()):
        h = hw.get((gpu, plat, model))
        grouped.setdefault((gpu, plat), []).append(
            Point(model, r.run_id, r.results[BENCH], {} if h is None else h.results[BENCH])
        )
    out = [Panel(gpu, plat, pts) for (gpu, plat), pts in grouped.items()]

    def order(p: Panel) -> tuple[bool, float, str, bool, str]:
        peak = p.datasheet.bf16
        known = math.isfinite(peak)
        return (not known, peak if known else 0.0, p.gpu, p.platform != "cluster", p.platform)

    return sorted(out, key=order)


def _median(pts: Sequence[Point], key: str) -> float:
    vals = [p.m[key] for p in pts if key in p.m and math.isfinite(p.m[key])]
    return statistics.median(vals) if vals else math.nan


def _f(v: float, digits: int = 3) -> str:
    return DASH if not math.isfinite(v) else f"{v:.{digits}g}"


def table(pts: Sequence[Point]) -> tuple[list[str], list[list[str]]]:
    header = [
        "Model",
        "Role",
        "Family",
        "Params B (active)",
        "Tokens/q",
        "GFLOP/q",
        "Intensity FLOP/B",
        "Achieved TFLOP/s",
        "Attainable",
        "% of roof",
        "HW intensity",
        "HW TFLOP/s",
        "HW / model FLOPs",
        "GPU ms/q",
        "Accuracy",
    ]
    nan = math.nan
    rows = []
    for p in sorted(pts, key=lambda p: -p.m.get("roofline_tflops", 0.0)):
        m, h = p.m, p.hw
        rows.append(
            [
                p.model,
                "decision model" if is_decision_model(p.model) else "zero-shot base",
                FAMILIES[p.family],
                f"{_f(m.get('roofline_params_b', nan))} ({_f(m.get('roofline_active_params_b', nan))})",
                _f(m.get("roofline_tokens_per_question", math.nan), 4),
                _f(m.get("roofline_gflop_per_question", math.nan)),
                _f(m.get("roofline_intensity", math.nan)),
                _f(m.get("roofline_tflops", math.nan)),
                _f(m.get("roofline_attainable_tflops", math.nan)),
                _f(100 * m.get("roofline_efficiency", math.nan), 2),
                _f(h.get("roofline_hw_intensity", math.nan)),
                _f(h.get("roofline_hw_tflops", math.nan)),
                _f(h.get("roofline_hw_flop_ratio", math.nan)),
                _f(m.get("roofline_gpu_ms", math.nan)),
                DASH if "roofline_accuracy" not in m else f"{m['roofline_accuracy']:.1%}",
            ]
        )
    return header, rows


def roofs(panels: Sequence[Panel]) -> tuple[list[str], list[list[str]]]:
    """Every panel's ceilings: measured (medians of its runs) beside the datasheet."""
    header = [
        "GPU",
        "Models",
        "Measured bf16 TFLOP/s",
        "Measured fp32 TFLOP/s",
        "Measured DRAM GB/s",
        "Measured ridge FLOP/B",
        "Datasheet bf16 TFLOP/s",
        "Datasheet fp32 TFLOP/s",
        "Datasheet DRAM GB/s",
        "Measured / datasheet bf16",
    ]
    rows = []
    for p in panels:
        meas, spec = p.measured, p.datasheet
        rows.append(
            [
                p.label,
                str(len(p.points)),
                _f(meas.bf16),
                _f(meas.fp32),
                _f(meas.bw),
                _f(meas.ridge),
                _f(spec.bf16),
                _f(spec.fp32),
                _f(spec.bw),
                DASH if not math.isfinite(r := meas.bf16 / spec.bf16) else f"{r:.0%}",
            ]
        )
    return header, rows


def by_gpu(panels: Sequence[Panel]) -> tuple[list[str], list[list[str]]]:
    """Achieved TFLOP/s (% of that panel's attainable roof) per model and panel; models in name order."""
    cells: dict[str, dict[str, str]] = {}
    for p in panels:
        for pt in p.points:
            tf, eff = pt.m.get("roofline_tflops", math.nan), pt.m.get("roofline_efficiency", math.nan)
            cell = _f(tf) + (f" ({eff:.0%})" if math.isfinite(eff) else "")
            cells.setdefault(pt.model, {})[p.label] = cell
    header = ["Model", *(p.label for p in panels)]
    rows = [[m, *(cells[m].get(p.label, DASH) for p in panels)] for m in sorted(cells)]
    return header, rows


# ---------------------------------------------------------------- chart

W, H = 820, 470
LEFT, RIGHT, TOP, BOTTOM = 64, 24, 14, 46


@dataclass(frozen=True)
class Scale:
    x0: float  # log10 bounds
    x1: float
    y0: float
    y1: float

    def x(self, v: float) -> float:
        return LEFT + (math.log10(v) - self.x0) / (self.x1 - self.x0) * (W - LEFT - RIGHT)

    def y(self, v: float) -> float:
        return H - BOTTOM - (math.log10(v) - self.y0) / (self.y1 - self.y0) * (H - TOP - BOTTOM)


def _scale(pts: Sequence[Point], peaks: Sequence[float]) -> Scale:
    nan = math.nan
    xs = [
        v for p in pts for v in (p.m.get("roofline_intensity", nan), p.hw.get("roofline_hw_intensity", nan))
    ]
    ys = [v for p in pts for v in (p.m.get("roofline_tflops", nan), p.hw.get("roofline_hw_tflops", nan))]
    xs += [p.m[k] for p in pts for k in p.m if k.startswith("roofline_t") and k.endswith("_intensity")]
    ys += [v for v in peaks if math.isfinite(v)]
    xs, ys = [v for v in xs if v > 0 and math.isfinite(v)], [v for v in ys if v > 0 and math.isfinite(v)]
    if not xs or not ys:
        return Scale(0, 4, -1, 3)
    return Scale(
        math.floor(math.log10(min(xs))),
        max(math.ceil(math.log10(max(xs))), 3),
        math.floor(math.log10(min(ys))),
        math.ceil(math.log10(max(ys) * 1.3)),
    )


def _roof(s: Scale, peak: float, bw: float, cls: str, label: str, *, slope: bool = True) -> str:
    """min(peak, bw * intensity), clipped to the plot; `label` right-aligned just above its flat part.
    A second compute roof (`slope=False`) shares the memory slope: only its flat part is drawn, labeled
    below the line, where it does not meet the points' labels above it."""
    if not (math.isfinite(peak) and math.isfinite(bw) and peak > 0 and bw > 0):
        return ""
    ridge = peak * 1e3 / bw  # FLOP/byte
    lo, hi = 10**s.x0, 10**s.x1
    start = max(lo, 10**s.y0 * 1e3 / bw)  # where the slope enters the plot
    pts = [(start, bw * start / 1e3)] if slope else []
    if ridge < hi:
        pts += [(max(ridge, lo), peak), (hi, peak)]
    else:
        pts.append((hi, bw * hi / 1e3))
    d = " ".join(f"{'M' if i == 0 else 'L'}{s.x(x):.1f},{s.y(y):.1f}" for i, (x, y) in enumerate(pts))
    tx, ty = s.x(pts[-1][0]) - 4, s.y(pts[-1][1]) + (-6 if slope else 15)
    return (
        f'<path class="ceil {cls}" d="{d}"/>'
        f'<text class="muted" x="{tx:.1f}" y="{ty:.1f}" text-anchor="end">{html.escape(label)}</text>'
    )


def _axes(s: Scale) -> str:
    out = []
    for e in range(int(s.x0), int(s.x1) + 1):
        x = s.x(10**e)
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{TOP}" y2="{H - BOTTOM}"/>')
        out.append(
            f'<text class="muted" x="{x:.1f}" y="{H - BOTTOM + 16}" text-anchor="middle">{10**e:g}</text>'
        )
    for e in range(int(s.y0), int(s.y1) + 1):
        y = s.y(10**e)
        out.append(f'<line class="grid" x1="{LEFT}" x2="{W - RIGHT}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="muted" x="{LEFT - 8}" y="{y + 4:.1f}" text-anchor="end">{10**e:g}</text>')
    out.append(
        f'<text class="muted" x="{(LEFT + W - RIGHT) / 2}" y="{H - 8}" text-anchor="middle">'
        "Operational intensity (FLOP per DRAM byte, log)</text>"
        f'<text class="muted" transform="translate(16 {(TOP + H - BOTTOM) / 2}) rotate(-90)" '
        'text-anchor="middle">Achieved TFLOP/s (log)</text>'
    )
    return "".join(out)


def _labels(placed: list[tuple[float, float, str]]) -> str:
    """Direct labels right of their points, nudged apart vertically when they would overlap."""
    out: list[str] = []
    taken: list[tuple[float, float]] = []
    for x, y, text in sorted(placed, key=lambda t: t[1]):
        ly = y + 4
        while any(abs(ly - ty) < 13 and abs(x - tx) < 130 for tx, ty in taken):
            ly += 13
        taken.append((x, ly))
        if ly - y > 8:  # nudged away from its point: a leader line keeps the pairing readable
            out.append(
                f'<line class="axis" x1="{x + 4:.1f}" y1="{y + 3:.1f}" x2="{x + 8:.1f}" y2="{ly - 4:.1f}"/>'
            )
        out.append(f'<text x="{x + 9:.1f}" y="{ly:.1f}">{html.escape(text)}</text>')
    return "".join(out)


def shared_scale(panels: Sequence[Panel]) -> Scale:
    """One set of log axes for every panel, so heights and ridge points compare across GPUs."""
    pts = [pt for p in panels for pt in p.points]
    peaks = [v for p in panels for v in (p.measured.bf16, p.datasheet.bf16)]
    return _scale(pts, peaks)


def chart(panel: Panel, s: Scale | None = None) -> str:
    """One panel's roofline; `s` is the shared scale (default: fitted to this panel alone)."""
    pts, meas, spec = panel.points, panel.measured, panel.datasheet
    if s is None:
        s = _scale(pts, [meas.bf16, spec.bf16])
    roofs = [
        _roof(s, spec.bf16, spec.bw, "spec", f"datasheet: {spec.bf16:.0f} TFLOP/s bf16, {spec.bw:.0f} GB/s"),
        _roof(
            s, meas.bf16, meas.bw, "", f"measured: {meas.bf16:.0f} TFLOP/s bf16 tensor, {meas.bw:.0f} GB/s"
        ),
        _roof(s, meas.fp32, meas.bw, "", f"measured: {meas.fp32:.0f} TFLOP/s fp32 CUDA cores", slope=False),
    ]
    marks, labels = [], []
    for p in pts:
        color = f"var(--fam-{p.family})"
        bins = sorted(
            (p.m[k], p.m[k.replace("_intensity", "_tflops")], k.split("_")[1])
            for k in p.m
            if k.startswith("roofline_t") and k.endswith("_intensity") and k != "roofline_hw_intensity"
        )
        bins = [b for b in bins if all(math.isfinite(v) and v > 0 for v in b[:2])]
        if len(bins) > 1:
            d = " ".join(
                f"{'M' if i == 0 else 'L'}{s.x(a):.1f},{s.y(t):.1f}" for i, (a, t, _) in enumerate(bins)
            )
            marks.append(f'<path class="traj" stroke="{color}" d="{d}"/>')
            for a, t, name in bins:
                tip = f"{p.model}, questions of ≤{name[1:]} tokens: {a:.0f} FLOP/B, {t:.2f} TFLOP/s"
                marks.append(
                    f'<g class="pt"><title>{html.escape(tip)}</title>'
                    f'<circle cx="{s.x(a):.1f}" cy="{s.y(t):.1f}" r="2.5" fill="{color}" opacity=".6"/></g>'
                )
        ai, tf = p.m.get("roofline_intensity", math.nan), p.m.get("roofline_tflops", math.nan)
        if not (math.isfinite(ai) and math.isfinite(tf) and ai > 0 and tf > 0):
            continue
        x, y = s.x(ai), s.y(tf)
        hai, htf = p.hw.get("roofline_hw_intensity", math.nan), p.hw.get("roofline_hw_tflops", math.nan)
        if math.isfinite(hai) and math.isfinite(htf) and hai > 0 and htf > 0:
            hx, hy = s.x(hai), s.y(htf)
            tip = f"{p.model}, hardware counters: {hai:.0f} FLOP/B, {htf:.2f} TFLOP/s"
            marks.append(
                f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" stroke="{color}" '
                'stroke-width="1" stroke-dasharray="2 3"/>'
                f'<g class="pt"><title>{html.escape(tip)}</title>'
                f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="10" fill="transparent"/>'
                f'<circle class="dot" cx="{hx:.1f}" cy="{hy:.1f}" r="5" fill="var(--surface)" '
                f'stroke="{color}" stroke-width="2"/></g>'
            )
        eff = p.m.get("roofline_efficiency", math.nan)
        tip = (
            f"{p.model} ({FAMILIES[p.family]}): {ai:.0f} FLOP/B, {tf:.2f} TFLOP/s"
            + (f", {eff:.0%} of attainable" if math.isfinite(eff) else "")
            + f" (run {p.run_id})"
        )
        ref, r = ("ref", 3.5) if not is_decision_model(p.model) else ("", 5)
        marks.append(
            f'<g class="pt {ref}"><title>{html.escape(tip)}</title>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="transparent"/>'
            f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" stroke="var(--surface)" '
            'stroke-width="2"/></g>'
        )
        labels.append((x, y, p.model))
    present = sorted({p.family for p in pts})
    legend = "".join(
        f'<span class="key"><span class="sw" style="background:var(--fam-{f})"></span>{FAMILIES[f]}</span>'
        for f in present
    )
    legend += (
        '<span class="key">● decision model · small, faint: zero-shot base model</span>'
        '<span class="key">● cost model: FLOPs ÷ compulsory bytes</span>'
        '<span class="key">○ hardware counters (ncu)</span>'
        '<span class="key">faint path: by prompt length</span>'
        '<span class="key">— measured roof · - - datasheet</span>'
    )
    missing = [
        name for name, roof in (("measured", meas), ("datasheet", spec)) if not math.isfinite(roof.bf16)
    ]
    if missing:
        legend += f'<span class="key">no {" or ".join(missing)} peaks for this GPU: roof not drawn</span>'
    title = f"Roofline on {panel.label}: each model's decision forwards, batch 1"
    return (
        f'<figure class="card chart roof"><figcaption><strong>{html.escape(title)}</strong></figcaption>'
        f'<div class="legend">{legend}</div>'
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">'
        f"{_axes(s)}{''.join(roofs)}{''.join(marks)}{_labels(labels)}</svg></figure>"
    )


# ---------------------------------------------------------------- decision recipes


def recipes(pts: Sequence[Point]) -> tuple[list[str], list[list[str]]]:
    """Each decision model's design choices beside its GPU use and accuracy on the same questions, best
    accuracy first. Models without a RECIPES entry (zero-shot references but the control) are left out."""
    header = [
        "Model",
        "Backbone",
        "Adaptation",
        "Readout",
        "Serving",
        "Accuracy",
        "Questions/s",
        "GPU ms/q",
        "GFLOP/q",
        "Intensity FLOP/B",
        "TFLOP/s",
        "% of bf16 peak",
        "% of roof",
        "Padding",
    ]
    nan = math.nan
    rows = []
    for p in sorted(pts, key=lambda p: -p.m.get("roofline_accuracy", 0.0)):
        if p.model not in RECIPES:
            continue
        m = p.m
        pad = m.get("roofline_padding", nan)
        rows.append(
            [
                p.model,
                *RECIPES[p.model],
                DASH if "roofline_accuracy" not in m else f"{m['roofline_accuracy']:.1%}",
                _f(m.get("roofline_questions_per_s", nan)),
                _f(m.get("roofline_gpu_ms", nan)),
                _f(m.get("roofline_gflop_per_question", nan)),
                _f(m.get("roofline_intensity", nan)),
                _f(m.get("roofline_tflops", nan)),
                _f(100 * m.get("roofline_mfu", nan), 2),
                _f(100 * m.get("roofline_efficiency", nan), 2),
                DASH if not math.isfinite(pad) else f"{pad:.0%}",
            ]
        )
    return header, rows


def findings(results_dir: Path) -> str:
    """The analysis written for the current runs (markdown), or "" if there is none."""
    path = results_dir / FINDINGS
    return path.read_text().strip() if path.is_file() else ""


def nested_list(items: list[tuple[int, str]], inline: Callable[[str], str]) -> str:
    """`<ul>` markup for (depth, text) bullets, nesting each deeper run under the item before it."""
    parts: list[str] = []
    depth = -1
    for d, text in items:
        if d > depth:
            parts.append("<ul>" * (d - depth))
        else:
            parts.append("</li>" + "</ul></li>" * (depth - d))
        parts.append(f"<li>{inline(text)}")
        depth = d
    parts.append("</li>" + "</ul></li>" * depth + "</ul>")
    return "".join(parts)


def markdown_to_html(text: str) -> str:
    """The markdown subset the findings use: ### headings, paragraphs, nested "- " lists, **bold**, `code`."""
    import re  # noqa: PLC0415

    def inline(t: str) -> str:
        t = html.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", t)

    out: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        if lines[0].startswith("### "):
            out.append(f"<h4>{inline(lines[0][4:])}</h4>")
            lines = lines[1:]
            if not lines:
                continue
        if all(ln.startswith(("- ", "  ")) for ln in lines):
            items: list[tuple[int, str]] = []
            for ln in lines:
                bullet = re.match(r"( *)- (.*)", ln)
                if bullet:
                    items.append((len(bullet[1]) // 2, bullet[2]))
                else:
                    depth, t = items[-1]
                    items[-1] = (depth, t + " " + ln.strip())
            out.append(nested_list(items, inline))
        else:
            out.append(f"<p>{inline(' '.join(ln.strip() for ln in lines))}</p>")
    return "\n".join(out)
