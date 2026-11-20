"""Steady state (method agreed 2026-11-19): every model x GPU pair is compared at its own steady-state batch,
where the GPU is kept busy, and judged by GPU utilization rather than TFLOP/s (which grows with batch size and
so mixes batch size into the result).

- **Batch models** (`batch` benchmark): utilization is `batch_s<L>_b<B>_kernel_busy`, GPU kernel time over
  wall time per batch from a torch.profiler trace. Values above 1 (up to ~1.2) are trace inflation and count
  as busy. The steady-state batch B* at state length L is the smallest B with kernel_busy >= 0.95 whose
  decisions/s no larger batch that ran beats by >= 5%. When no B qualifies, B* is "not reached" and the
  report gives the highest kernel_busy and the throughput plateau (the same rule without the utilization
  condition).
- **OpenJev** (vLLM, `jeval` benchmark): there is no per-batch kernel trace, only nvidia-smi. Steady state
  is the smallest `jeval_concurrency` whose questions/s gain to the next concurrency is < 5% and whose
  per-GPU utilization is >= 95% (`gpu0_util_pct`, detailed metrics). nvidia-smi's utilization only says some
  kernel was running, so this is a weaker signal than kernel_busy.
- **Best model per GPU**: JEVal accuracy, invalid answers and ECE next to each model's speed at its steady
  state on that GPU.
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import report_batch as rb
from . import report_roofline as rr
from .record import Metrics, RunRecord

BUSY = 0.95  # kernel_busy / utilization threshold
GAIN = 0.05  # a larger batch "beats" a smaller one when its throughput is >= 5% higher
LENGTHS = (128, 1024)  # state lengths the steady-state tables report
FINDINGS = "steady_state_findings.md"  # prose in results/, next to roofline_findings.md
OPENJEV = "openjev"
DASH, TIMES = "\u2013", "\u00d7"  # placeholder for missing values; "A10G x4"
GPU_ORDER = ("L4", "A10G", "A100-SXM4", "A100 80GB PCIe", "H100", "H200", "B200")


def findings(results_dir: Path) -> str:
    """The analysis written for the current steady-state runs (markdown), or "" if there is none."""
    path = results_dir / FINDINGS
    return path.read_text().strip() if path.is_file() else ""


def gpu_rank(gpu: str) -> tuple[int, str]:
    return next((i for i, o in enumerate(GPU_ORDER) if o in gpu), len(GPU_ORDER)), gpu


def busy_str(v: float) -> str:
    """kernel_busy to two decimals, rounded down so a value under the 0.95 threshold never prints as 0.95."""
    return DASH if not math.isfinite(v) else f"{math.floor(v * 100 + 1e-9) / 100:.2f}"


def _f(v: float, digits: int = 0) -> str:
    return DASH if not math.isfinite(v) else f"{v:,.{digits}f}"


# ---------------------------------------------------------------- batch models


@dataclass(frozen=True)
class Steady:
    length: int
    b_star: int | None  # None: not reached
    plateau: int | None  # smallest B that no larger batch beats by >= GAIN (no utilization condition)
    max_busy: float
    max_busy_b: int
    ooms: tuple[int, ...]
    dps: Mapping[int, float]  # decisions/s per batch size that ran
    busy: Mapping[int, float]  # kernel_busy per batch size that ran

    @property
    def at(self) -> int | None:
        """The batch the speed is quoted at: B*, else the throughput plateau."""
        return self.b_star if self.b_star is not None else self.plateau

    @property
    def speed(self) -> float:
        return self.dps[self.at] if self.at is not None else math.nan


def _unbeaten(b: int, dps: Mapping[int, float]) -> bool:
    return all(v < (1 + GAIN) * dps[b] for c, v in dps.items() if c > b)


def steady(points: Mapping[int, Metrics], length: int) -> Steady | None:
    """B* for one state length from {B: metrics}; None when no batch size ran there."""
    ooms = tuple(sorted(b for b, m in points.items() if m.get("oom", 0.0) > 0))
    dps = {
        b: m["decisions_per_s"]
        for b, m in sorted(points.items())
        if b not in ooms and math.isfinite(m.get("decisions_per_s", math.nan))
    }
    if not dps:
        return None
    busy = {b: points[b].get("kernel_busy", math.nan) for b in dps}
    b_star = next((b for b in dps if busy[b] >= BUSY and _unbeaten(b, dps)), None)
    plateau = next((b for b in dps if _unbeaten(b, dps)), None)
    finite = [(v, -b) for b, v in busy.items() if math.isfinite(v)]
    top, neg_b = max(finite) if finite else (math.nan, 0)
    return Steady(length, b_star, plateau, top, -neg_b, ooms, dps, busy)


def steady_of(s: rb.Series, length: int) -> Steady | None:
    return steady(s.grid[length], length) if length in s.grid else None


def b_star_cell(st: Steady | None) -> str:
    if st is None:
        return DASH
    if st.b_star is not None:
        return str(st.b_star)
    return "not reached" + (f" (plateau B={st.plateau})" if st.plateau is not None else "")


def busy_cell(st: Steady | None) -> str:
    return (
        DASH
        if st is None or not math.isfinite(st.max_busy)
        else f"{busy_str(st.max_busy)} (B={st.max_busy_b})"
    )


def speed_cell(st: Steady | None) -> str:
    if st is None or st.at is None:
        return DASH
    return _f(st.speed) + ("" if st.b_star is not None else " (plateau)")


def steady_table(ps: Sequence[rb.Panel]) -> tuple[list[str], list[list[str]]]:
    header: list[str] = ["Model", "GPU", "Prep mode"]
    for n in LENGTHS:
        header += [f"B* at {n}", f"Max kernel_busy at {n}", f"Decisions/s at B* ({n})"]
    header.append("OOM points")
    rows: list[list[str]] = []
    for p in ps:
        for s in p.series:
            row: list[str] = [s.name, p.label, s.prep]
            ooms: list[str] = []
            for n in LENGTHS:
                st = steady_of(s, n)
                row += [b_star_cell(st), busy_cell(st), speed_cell(st)]
                if st is not None and st.ooms:
                    ooms.append(f"B={','.join(map(str, st.ooms))} at {n}")
            rows.append([*row, "; ".join(ooms) or "none"])
    return header, rows


def busy_sizes(ps: Sequence[rb.Panel], length: int) -> list[int]:
    return sorted({b for p in ps for s in p.series for b in s.grid.get(length, {})})


def busy_table(ps: Sequence[rb.Panel], length: int) -> tuple[list[str], list[list[str]]]:
    """kernel_busy per batch size at one state length; "OOM" where the point ran out of memory."""
    bs = busy_sizes(ps, length)
    rows: list[list[str]] = []
    for p in ps:
        for s in p.series:
            pts = s.grid.get(length, {})
            if not pts:
                continue
            cells = []
            for b in bs:
                m = pts.get(b, {})
                v = m.get("kernel_busy", math.nan)
                cells.append("OOM" if m.get("oom", 0.0) > 0 else busy_str(v))
            st = steady(pts, length)
            rows.append([p.label, s.name, *cells, b_star_cell(st)])
    return ["GPU", "Series", *[f"B={b}" for b in bs], "B*"], rows


# ---------------------------------------------------------------- prep mode


def prep_rows(ps: Sequence[rb.Panel]) -> list[tuple[rb.Panel, rb.Series]]:
    """Series of models that ran both inline and pretokenized on the same GPU."""
    out: list[tuple[rb.Panel, rb.Series]] = []
    for p in ps:
        both = {
            s.slot
            for s in p.series
            if {"inline", "pretokenized"} <= {t.prep for t in p.series if t.slot == s.slot}
        }
        out += [(p, s) for s in p.series if s.slot in both]
    return out


def prep_table(ps: Sequence[rb.Panel]) -> tuple[list[str], list[list[str]]]:
    rows_in = prep_rows(ps)
    big = min((max(s.grid.get(LENGTHS[0], {0: {}})) for _, s in rows_in), default=0)  # largest B all ran
    header = [
        "GPU",
        "Series",
        "Host prep",
        "Experts",
        *[f"{c} at {n}" for n in LENGTHS for c in ("B*", "Max kernel_busy")],
        f"Decisions/s at B={big} ({LENGTHS[0]})",
        f"Host prep ms per batch at B={big}",
    ]
    rows: list[list[str]] = []
    for p, s in rows_in:
        cells = []
        for n in LENGTHS:
            st = steady_of(s, n)
            cells += [b_star_cell(st), busy_cell(st)]
        m = s.grid.get(LENGTHS[0], {}).get(big, {})
        prep = m.get("prep_ms", math.nan)
        experts = "eager" if "eager experts" in s.variant else "grouped_mm (default)"
        rows.append(
            [
                p.label,
                s.name,
                s.prep,
                experts,
                *cells,
                _f(m.get("decisions_per_s", math.nan)),
                "inside wall" if s.prep == "inline" else _f(prep, 1),
            ]
        )
    return header, rows


# ---------------------------------------------------------------- OpenJev concurrency


@dataclass(frozen=True)
class ConcPoint:
    concurrency: int
    qps: float
    util: float  # per-GPU (gpu0) nvidia-smi utilization %, pooled when per-GPU is absent
    latency_ms: float
    latency_p99_ms: float
    accuracy: float
    run_id: str


@dataclass(frozen=True)
class ConcCurve:
    platform: str
    gpu: str
    n: int
    points: list[ConcPoint]  # ascending concurrency

    @property
    def label(self) -> str:
        return f"{rr.gpu_name(self.gpu)} ({self.platform})"

    @property
    def steady(self) -> ConcPoint | None:
        """The smallest concurrency with < GAIN more q/s at the next one and per-GPU util >= 95%."""
        for a, b in zip(self.points, self.points[1:], strict=False):
            if b.qps < (1 + GAIN) * a.qps and a.util >= 100 * BUSY:
                return a
        return None

    def gain(self, i: int) -> float:
        return self.points[i + 1].qps / self.points[i].qps - 1 if i + 1 < len(self.points) else math.nan


def concurrency_of(r: RunRecord) -> int | None:
    c = r.params.get("jeval_concurrency", r.invariants.get("jeval_concurrency"))
    return None if c is None else int(c)


def openjev_curves(recs: Sequence[RunRecord], min_levels: int = 3) -> list[ConcCurve]:
    """One curve per (platform, GPU, question count) with >= min_levels concurrencies; newest per level."""
    groups: dict[tuple[str, str, int], dict[int, ConcPoint]] = {}
    for r in recs:
        m = r.results.get("jeval")
        if m is None or str(r.params.get("model", "")) != OPENJEV or (c := concurrency_of(r)) is None:
            continue
        g = r.results.get("jeval/nvidia_smi", {})
        util = g.get("gpu0_util_pct", g.get("gpu_util_pct", math.nan))
        pt = ConcPoint(
            c,
            m.get("jeval_questions_per_s", math.nan),
            util,
            m.get("jeval_latency_ms", math.nan),
            m.get("jeval_latency_ms_p99", math.nan),
            m.get("jeval_accuracy", math.nan),
            r.run_id,
        )
        key = (rr.platform(r.invariants), str(r.invariants.get("gpu", "?")), int(m.get("jeval_n", 0)))
        level = groups.setdefault(key, {})
        if c not in level or r.run_id > level[c].run_id:
            level[c] = pt
    curves = [
        ConcCurve(plat, gpu, n, [pts[c] for c in sorted(pts)])
        for (plat, gpu, n), pts in groups.items()
        if len(pts) >= min_levels
    ]
    return sorted(curves, key=lambda c: (gpu_rank(c.gpu), c.platform, c.n))


def concurrency_table(curves: Sequence[ConcCurve]) -> tuple[list[str], list[list[str]]]:
    header = ["GPU", "Questions", "Concurrency", "q/s", "Gain to next", "Per-GPU util", "Latency mean",
              "Latency p99", "Accuracy", "Steady state"]  # fmt: skip
    rows: list[list[str]] = []
    for cv in curves:
        st = cv.steady
        for i, p in enumerate(cv.points):
            g = cv.gain(i)
            rows.append(
                [
                    cv.label,
                    f"{cv.n:,}",
                    str(p.concurrency),
                    _f(p.qps, 2),
                    DASH if not math.isfinite(g) else f"{0.0 if abs(g) < 0.005 else g:+.0%}",
                    DASH if not math.isfinite(p.util) else f"{p.util:.1f}%",
                    _ms(p.latency_ms),
                    _ms(p.latency_p99_ms),
                    DASH if not math.isfinite(p.accuracy) else f"{p.accuracy:.1%}",
                    "steady" if st is not None and p is st else "",
                ]
            )
    return header, rows


def _ms(v: float) -> str:
    if not math.isfinite(v):
        return DASH
    return f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"


# ---------------------------------------------------------------- best model per GPU

NO_BATCH = {"d1-3b": "no steady-state batch: the d1 backend answers one request at a time"}
CAP_NOTE = "invalid: rows over the 16-option cap (semif_max_options = 16)"
A100_NOTE = {
    "A100 80GB PCIe": "PCIe card (300 W, 1.9 TB/s), not the SXM4 (400 W, 2.0 TB/s) of the 11-08 runs",
    "A100-SXM4-80GB": "SXM4 card, 200-question runs of 2026-11-08",
}


@dataclass(frozen=True)
class Candidate:
    gpu: str
    platform: str
    model: str
    n: int
    accuracy: float
    invalid: float
    ece: float
    qps: float
    concurrency: int | None
    gpus: int
    speed: str
    note: str
    run_id: str

    @property
    def valid_accuracy(self) -> float:
        """Correct over answered (n - invalid): approximate, it assumes accuracy counts invalid as wrong."""
        answered = self.n - self.invalid
        return self.accuracy * self.n / answered if answered > 0 else math.nan


def _batch_speed(model: str, gpu: str, ps: Sequence[rb.Panel]) -> str:
    """The fastest variant's speed at B* (else plateau) per state length on that GPU, from `batch` records."""
    best: dict[int, tuple[float, rb.Series, Steady]] = {}
    for p in ps:
        if rr.gpu_name(p.gpu) != rr.gpu_name(gpu):
            continue
        for s in p.series:
            if s.model != model:
                continue
            for n in LENGTHS:
                st = steady_of(s, n)
                if st is not None and st.at is not None and (n not in best or st.speed > best[n][0]):
                    best[n] = (st.speed, s, st)
    if not best:
        return ""
    parts = []
    for n, (v, s, st) in sorted(best.items()):
        where = f"B*={st.b_star}" if st.b_star is not None else f"plateau B={st.plateau} (B* not reached)"
        variant = f", {s.variant}" if s.variant and s.variant != "inline" else ""
        parts.append(f"{v:,.0f} dec/s at {where} ({n} tok{variant})")
    return "; ".join(parts)


def candidates(
    recs: Sequence[RunRecord], ps: Sequence[rb.Panel], curves: Sequence[ConcCurve]
) -> list[Candidate]:
    """Each (GPU, model)'s JEVal run: the most questions, then the steady-state concurrency when a curve
    exists, else the highest concurrency, then the newest. Sorted by GPU, then accuracy."""
    steady_ids = {st.run_id for cv in curves if (st := cv.steady) is not None}
    pick: dict[tuple[str, str], RunRecord] = {}

    def rank(r: RunRecord) -> tuple[float, bool, int, str]:
        n = r.results["jeval"].get("jeval_n", 0.0)
        return n, r.run_id in steady_ids, concurrency_of(r) or 0, r.run_id

    for r in recs:
        if "jeval" not in r.results:
            continue
        key = (rr.gpu_name(str(r.invariants.get("gpu", "?"))), str(r.params.get("model", "?")))
        if key not in pick or rank(r) > rank(pick[key]):
            pick[key] = r
    out: list[Candidate] = []
    for (gpu, model), r in pick.items():
        m = r.results["jeval"]
        g = r.results.get("jeval/nvidia_smi", {})
        conc = concurrency_of(r)
        if model == OPENJEV:
            cv = next((c for c in curves if c.points and any(p.run_id == r.run_id for p in c.points)), None)
            if cv is not None and cv.steady is not None and cv.steady.run_id == r.run_id:
                speed = f"{cv.steady.qps:.2f} q/s at concurrency {cv.steady.concurrency} (steady)"
            else:
                speed = "no concurrency sweep"
        else:
            family = next((v for k, v in NO_BATCH.items() if model.startswith(k)), None)
            speed = family or _batch_speed(model, gpu, ps) or "no batch record"
        notes = []
        if model.startswith(("lfm2", "qwen35", "jevk5")) and m.get("jeval_invalid", 0.0) > 0:
            notes.append(CAP_NOTE)
        if gpu in A100_NOTE:
            notes.append(A100_NOTE[gpu])
        out.append(
            Candidate(
                gpu,
                rr.platform(r.invariants),
                model,
                int(m.get("jeval_n", 0)),
                m.get("jeval_accuracy", math.nan),
                m.get("jeval_invalid", math.nan),
                m.get("jeval_ece", math.nan),
                m.get("jeval_questions_per_s", math.nan),
                conc,
                int(g.get("gpu_count", 1)),
                speed,
                "; ".join(notes),
                r.run_id,
            )
        )
    return sorted(out, key=lambda c: (gpu_rank(c.gpu), -c.accuracy if math.isfinite(c.accuracy) else 0))


def best_table(cs: Sequence[Candidate]) -> tuple[list[str], list[list[str]]]:
    header = ["GPU", "Model", "Platform", "N", "Accuracy", "Invalid", "Accuracy on valid (approx.)", "ECE",
              "JEVal q/s (concurrency)", "Speed at steady state", "Notes", "Run"]  # fmt: skip
    rows: list[list[str]] = []
    for c in cs:
        gpu = c.gpu if c.gpus == 1 else f"{c.gpu} {TIMES}{c.gpus}"
        conc = f" ({c.concurrency})" if c.concurrency is not None else ""
        rows.append(
            [
                gpu,
                c.model,
                c.platform,
                f"{c.n:,}",
                _pct(c.accuracy),
                _f(c.invalid),
                _pct(c.valid_accuracy),
                _f(c.ece, 3),
                _f(c.qps, 2) + conc,
                c.speed,
                c.note,
                c.run_id,
            ]
        )
    return header, rows


def _pct(v: float) -> str:
    return DASH if not math.isfinite(v) else f"{v:.1%}"


# ---------------------------------------------------------------- charts

W, H = 420, 260
XCONC = "Concurrent requests (jeval_concurrency)"
LEFT, RIGHT, TOP, BOTTOM = 52, 16, 12, 40


def _frame(
    xs: Sequence[int], x: Callable[[float], float], yticks: Sequence[tuple[float, str]],
    y: Callable[[float], float], *, xlabel: str, ylabel: str,
) -> list[str]:  # fmt: skip
    out: list[str] = []
    for b in xs:
        out.append(f'<line class="grid" x1="{x(b):.1f}" x2="{x(b):.1f}" y1="{TOP}" y2="{H - BOTTOM}"/>')
        out.append(
            f'<text class="muted" x="{x(b):.1f}" y="{H - BOTTOM + 15}" text-anchor="middle">{b}</text>'
        )
    for v, lab in yticks:
        out.append(f'<line class="grid" x1="{LEFT}" x2="{W - RIGHT}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        out.append(f'<text class="muted" x="{LEFT - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{lab}</text>')
    out.append(
        f'<text class="muted" x="{(LEFT + W - RIGHT) / 2}" y="{H - 6}" text-anchor="middle">{xlabel}</text>'
        f'<text class="muted" transform="translate(12 {(TOP + H - BOTTOM) / 2}) rotate(-90)" '
        f'text-anchor="middle">{ylabel}</text>'
    )
    return out


def _log2_x(xs: Sequence[int]) -> Callable[[float], float]:
    x0, x1 = math.log2(xs[0]), math.log2(xs[-1])
    return lambda b: LEFT + (math.log2(b) - x0) / max(x1 - x0, 1e-9) * (W - LEFT - RIGHT)


def _ref_line(y: float, label: str) -> str:
    return (
        f'<line class="ref" x1="{LEFT}" x2="{W - RIGHT}" y1="{y:.1f}" y2="{y:.1f}"/>'
        f'<text class="muted" x="{W - RIGHT - 2}" y="{y - 4:.1f}" text-anchor="end" font-size="11">'
        f"{label}</text>"
    )


def _dot(cx: float, cy: float, color: str, tip: str, ring: bool) -> str:
    mark = (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="var(--surface)" stroke="{color}" '
        'stroke-width="2.5"/>'
        if ring
        else f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="{color}" stroke="var(--surface)" '
        'stroke-width="2"/>'
    )
    return (
        f'<g class="pt"><title>{html.escape(tip)}</title>'
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="transparent"/>{mark}</g>'
    )


def busy_chart(ps: Sequence[rb.Panel], length: int) -> str:
    """Small multiples, one per GPU: kernel_busy vs batch size at one state length, a 95% reference line and a
    ring on each series' B*. kernel_busy above 1 (trace inflation) is drawn at 1; tooltips give the value."""
    bs = busy_sizes(ps, length)
    if not bs:
        return ""
    x = _log2_x(bs)

    def y(v: float) -> float:
        return H - BOTTOM - min(max(v, 0.0), 1.0) * (H - TOP - BOTTOM)

    figs, present = [], []
    for p in ps:
        series = [s for s in p.series if s.grid.get(length, {})]
        if not series:
            continue
        out = _frame(bs, x, [(v, f"{v:.0%}") for v in (0, 0.25, 0.5, 0.75, 1.0)], y,
                     xlabel="Batch size", ylabel="kernel_busy")  # fmt: skip
        out.append(_ref_line(y(BUSY), "95%"))
        for s in series:
            st = steady(s.grid[length], length)
            if st is None:
                continue
            pts = [(b, v) for b, v in st.busy.items() if math.isfinite(v)]
            if not pts:
                continue
            present.append(s)
            d = " ".join(f"{'M' if i == 0 else 'L'}{x(b):.1f},{y(v):.1f}" for i, (b, v) in enumerate(pts))
            out.append(rb.line(s, d))
            color = f"var(--bs-{s.slot})"
            for b, v in pts:
                star = b == st.b_star
                tip = (
                    f"{s.name} on {p.label}, B = {b}, {length} tokens: kernel_busy {v:.2f}, "
                    f"{st.dps[b]:,.0f} decisions/s{' (B*)' if star else ''}; run {s.run_id}"
                )
                out.append(_dot(x(b), y(v), color, tip, star))
        title = html.escape(p.label)
        figs.append(
            f'<figure class="card"><figcaption><strong>{title}</strong></figcaption>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="kernel_busy vs batch size on {title}, '
            f'{length} tokens">{"".join(out)}</svg></figure>'
        )
    if not figs:
        return ""
    head = (
        f"GPU utilization (kernel_busy) vs batch size, {length}-token states; dashed line 95%, "
        "ring = steady-state batch B*"
    )
    return (
        '<div class="bsw"><figure class="card chart">'
        f"<figcaption><strong>{html.escape(head)}</strong></figcaption>"
        f'<div class="legend">{rb.legend(present)}'
        '<span class="key">values above 100% (trace inflation) drawn at 100%</span></div>'
        f'<div class="multiples">{"".join(figs)}</div></figure></div>'
    )


def concurrency_chart(curves: Sequence[ConcCurve]) -> str:
    """Two panels sharing the concurrency axis: questions/s, and per-GPU utilization; one line per GPU."""
    if not curves:
        return ""
    cs = sorted({p.concurrency for cv in curves for p in cv.points})
    x = _log2_x(cs)
    top = max(p.qps for cv in curves for p in cv.points if math.isfinite(p.qps))
    step = 5.0 if top > 10 else 2.0
    ymax = math.ceil(top / step) * step

    def yq(v: float) -> float:
        return H - BOTTOM - v / ymax * (H - TOP - BOTTOM)

    def yu(v: float) -> float:
        return H - BOTTOM - min(max(v, 0.0), 100.0) / 100 * (H - TOP - BOTTOM)

    def panel(kind: str) -> str:
        if kind == "qps":
            ticks = [(v * step, f"{v * step:g}") for v in range(int(ymax / step) + 1)]
            out = _frame(cs, x, ticks, yq, xlabel=XCONC, ylabel="Questions/s")
        else:
            out = _frame(cs, x, [(v, f"{v:g}%") for v in (0, 25, 50, 75, 100)], yu,
                         xlabel=XCONC, ylabel="Per-GPU util (nvidia-smi)")  # fmt: skip
            out.append(_ref_line(yu(100 * BUSY), "95%"))
        for i, cv in enumerate(curves):
            color = f"var(--bs-{i % 6})"
            yf = yq if kind == "qps" else yu
            pts = [(p, p.qps if kind == "qps" else p.util) for p in cv.points]
            pts = [(p, v) for p, v in pts if math.isfinite(v)]
            d = " ".join(
                f"{'M' if j == 0 else 'L'}{x(p.concurrency):.1f},{yf(v):.1f}" for j, (p, v) in enumerate(pts)
            )
            out.append(f'<path class="line" stroke="{color}" d="{d}"/>')
            st = cv.steady
            for p, _ in pts:
                tip = (
                    f"OpenJev on {cv.label}, concurrency {p.concurrency}: {p.qps:.2f} q/s, per-GPU util "
                    f"{p.util:.1f}%, latency {_ms(p.latency_ms)} mean / {_ms(p.latency_p99_ms)} p99"
                    f"{' (steady state)' if p is st else ''}; run {p.run_id}"
                )
                out.append(
                    _dot(x(p.concurrency), yf(p.qps if kind == "qps" else p.util), color, tip, p is st)
                )
        name = "Questions/s" if kind == "qps" else "Per-GPU utilization"
        return (
            f'<figure class="card"><figcaption><strong>{name}</strong></figcaption>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="OpenJev {name} vs concurrency">'
            f"{''.join(out)}</svg></figure>"
        )

    keys = "".join(
        f'<span class="key"><span class="sw" style="background:var(--bs-{i % 6})"></span>'
        f"{html.escape(cv.label)}</span>"
        for i, cv in enumerate(curves)
    )
    head = (
        "OpenJev (vLLM) on JEVal: throughput and utilization vs concurrency; ring = steady-state concurrency"
    )
    return (
        '<div class="bsw"><figure class="card chart">'
        f"<figcaption><strong>{html.escape(head)}</strong></figcaption>"
        f'<div class="legend">{keys}</div>'
        f'<div class="multiples">{panel("qps")}{panel("util")}</div></figure></div>'
    )
