"""The roofline section of the report: one log-log chart per GPU, plus the same numbers as a table.

Each model's latest `roofline` run is a filled point at (cost-model intensity, achieved TFLOP/s), with a faint
trajectory through its prompt-length bins; its latest `s1 ncu` run (roofline_hw_*) adds a hollow point at the
hardware-counter intensity and FLOP/s. Roofs come from the runs' measured ceilings (solid: bf16 tensor-core
GEMM, fp32 CUDA-core GEMM, DRAM copy; medians over the runs on that GPU) and the datasheet (dashed).

Color encodes the architecture family, read from the metrics (MoE when active parameters are under half the
total; else hybrid when some layers are not attention): three hues, the palette's all-pairs-safe slots.
Identity is a direct label on every point, never color alone, and the table repeats every value.
"""

from __future__ import annotations

import html
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from .record import Metrics, RunRecord

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
"""


def family(m: Metrics) -> int:
    total, active = m.get("roofline_params_b", math.nan), m.get("roofline_active_params_b", math.nan)
    if active < 0.5 * total:
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
        return family(self.m)


def _model(r: RunRecord) -> str:
    return str(r.params.get("model", "?"))


def _is_ncu(r: RunRecord) -> bool:
    return bool(r.params.get("roofline_ncu", 0))


def points_by_gpu(recs: Sequence[RunRecord]) -> dict[str, list[Point]]:
    """Latest clean run per (GPU, model), joined with the latest ncu run of the same pair."""
    runs = sorted((r for r in recs if BENCH in r.results), key=lambda r: r.run_id)
    clean: dict[tuple[str, str], RunRecord] = {}
    hw: dict[tuple[str, str], RunRecord] = {}
    for r in runs:
        key = (str(r.invariants.get("gpu", "?")), _model(r))
        if _is_ncu(r):
            if "roofline_hw_intensity" in r.results[BENCH]:
                hw[key] = r
        else:
            clean[key] = r
    out: dict[str, list[Point]] = {}
    for (gpu, model), r in sorted(clean.items()):
        h = hw.get((gpu, model))
        out.setdefault(gpu, []).append(
            Point(model, r.run_id, r.results[BENCH], {} if h is None else h.results[BENCH])
        )
    return out


def _median(pts: Sequence[Point], key: str) -> float:
    vals = [p.m[key] for p in pts if key in p.m and math.isfinite(p.m[key])]
    return statistics.median(vals) if vals else math.nan


def _f(v: float, digits: int = 3) -> str:
    return DASH if not math.isfinite(v) else f"{v:.{digits}g}"


def table(pts: Sequence[Point]) -> tuple[list[str], list[list[str]]]:
    header = [
        "Model",
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


def chart(gpu: str, pts: Sequence[Point]) -> str:
    peak, bw = _median(pts, "roofline_peak_bf16_tflops"), _median(pts, "roofline_peak_bw_gbs")
    fp32 = _median(pts, "roofline_peak_fp32_tflops")
    spec_peak, spec_bw = _median(pts, "roofline_spec_bf16_tflops"), _median(pts, "roofline_spec_bw_gbs")
    s = _scale(pts, [peak, spec_peak])
    roofs = [
        _roof(s, spec_peak, spec_bw, "spec", f"datasheet: {spec_peak:.0f} TFLOP/s bf16, {spec_bw:.0f} GB/s"),
        _roof(s, peak, bw, "", f"measured: {peak:.0f} TFLOP/s bf16 tensor, {bw:.0f} GB/s"),
        _roof(s, fp32, bw, "", f"measured: {fp32:.0f} TFLOP/s fp32 CUDA cores", slope=False),
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
        marks.append(
            f'<g class="pt"><title>{html.escape(tip)}</title>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="transparent"/>'
            f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="var(--surface)" '
            'stroke-width="2"/></g>'
        )
        labels.append((x, y, p.model))
    present = sorted({p.family for p in pts})
    legend = "".join(
        f'<span class="key"><span class="sw" style="background:var(--fam-{f})"></span>{FAMILIES[f]}</span>'
        for f in present
    )
    legend += (
        '<span class="key">● cost model: FLOPs ÷ compulsory bytes</span>'
        '<span class="key">○ hardware counters (ncu)</span>'
        '<span class="key">faint path: by prompt length</span>'
        '<span class="key">— measured roof · - - datasheet</span>'
    )
    title = f"Roofline on {gpu}: each model's decision forwards, batch 1"
    return (
        f'<figure class="card chart roof"><figcaption><strong>{html.escape(title)}</strong></figcaption>'
        f'<div class="legend">{legend}</div>'
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">'
        f"{_axes(s)}{''.join(roofs)}{''.join(marks)}{_labels(labels)}</svg></figure>"
    )
