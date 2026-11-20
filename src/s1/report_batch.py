"""The `batch` benchmark in the report: achieved FLOP/s vs batch size per GPU, and the CPU vs GPU split of
batch-1 latency (hypothesis H2, docs/h2_batch_findings.md).

A series is one model run one way: Laya's execution mode (eager, CUDA graphs, torch.compile, compile +
graphs), and for the semif models the host prep (`batch_host_prep`, when not inline) and the MoE expert
dispatch (when not transformers' default grouped_mm). Hue is the model, line dash the variant. For each
(platform, GPU, series) the record covering the most batch sizes wins, then the newest.
Records from `s1 nsys` (`batch_nsys_n`) are left out: nsys's own tracing inflates their wall time.

FLOP/s is cost-model FLOPs over wall time (`batch_s<L>_b<B>_tflops_wall`) at the shortest state length a
record has. The split bar is the batch-1 point there: GPU kernel time (`kernel_busy`, clipped to 1: for
GPU-bound points the traced kernel time can exceed the untraced wall by up to ~20%) and everything else
(host work, launches, idle GPU between kernels). Host CPU models are in each tooltip: the eager batch-1 floor
depends on the host as much as on the GPU.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from . import report_roofline as rr
from .record import Metrics, RunRecord

BENCH = "batch"
FAMILIES = (  # fixed order = fixed categorical slot (never cycled); hue = model, line dash = variant
    ("lfm2-8b-a1b", "LFM2-8B-A1B"),
    ("laya", "Laya"),
    ("kev-4b", "Kev-4B"),
    ("qwen3-8b", "Qwen3-8B"),
    ("qwen35-9b", "Qwen3.5-9B"),
    ("jevk5-9b", "JevK5-9B"),
)
OTHER = len(FAMILIES)  # slot for a model not listed above
LAYA_MODES = {"eager": "eager", "graphs": "CUDA graphs", "compile": "torch.compile",
              "compile_graphs": "compile + graphs"}  # fmt: skip
BASE_VARIANT = {
    "kev-4b.hybrid": "fused + graphs",
    "qwen3-8b.zeroshot": "eager",
    "lfm2-8b-a1b.zeroshot": "inline",
}
DEFAULT_EXPERTS = ("", "grouped_mm")  # transformers' default MoE dispatch (absent in older records)
DASHES = ("", "6 3", "2 3", "8 3 2 3")  # variant index -> stroke-dasharray
DASH_OF = {"CUDA graphs": 1, "torch.compile": 2, "compile + graphs": 3,
           "pretokenized": 1, "eager experts": 2, "pretokenized, eager experts": 3}  # fmt: skip
CSS = """
:root{--bs-0:#2a78d6;--bs-1:#eb6834;--bs-2:#1baf7a;--bs-3:#eda100;--bs-4:#e87ba4;--bs-5:#008300;
  --bs-6:#52514e;--bs-gpu:#2a78d6;--bs-ref:#52514e}
@media (prefers-color-scheme:dark){:root{--bs-0:#3987e5;--bs-1:#d95926;--bs-2:#199e70;--bs-3:#c98500;
  --bs-4:#d55181;--bs-5:#0ca30c;--bs-6:#c3c2b7;--bs-gpu:#3987e5;--bs-ref:#c3c2b7}}
.bsw .grid{stroke:var(--rule);stroke-width:1}.bsw .axis{stroke:var(--ink-2);stroke-width:1}
.bsw .line{fill:none;stroke-width:2}.bsw g.pt:hover circle{stroke-width:3}
.bsw .ref{stroke:var(--bs-ref);stroke-width:1;stroke-dasharray:4 3}
.bsw .legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:4px 0 10px;color:var(--ink-2);font-size:12px}
.bsw .key{display:inline-flex;align-items:center;gap:6px}
.bsw .sw{width:10px;height:10px;border-radius:50%;display:inline-block}
.bsw .legend svg{display:inline-block;width:28px;height:10px}
.bsw .multiples{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(360px,100%),1fr));gap:12px}
.bsw .multiples svg{max-width:none}
.bsw .gpu{fill:var(--bs-gpu)}.bsw .host{fill:var(--track)}.bsw g.seg:hover rect{opacity:.8}
"""
_POINT = re.compile(r"^batch_s(\d+)_b(\d+)_tflops_wall$")
_ANY = re.compile(r"^batch_s(\d+)_b(\d+)_(.+)$")


def family_of(model: str) -> int:
    """FAMILIES slot of a model name ("laya.english" -> Laya), OTHER when unlisted."""
    hits = (
        i for i, (pre, _) in enumerate(FAMILIES) if model == pre or model.startswith((f"{pre}.", f"{pre} "))
    )
    return next(hits, OTHER)


def experts_of(r: RunRecord) -> str:
    """The MoE expert dispatch a run used: the param that forced it, else what the runtime recorded."""
    forced = str(r.params.get("semif_experts_implementation", ""))
    return forced or str(r.invariants.get("env_experts_implementation", ""))


def variant_of(r: RunRecord) -> str:
    """How a model ran, for labels: Laya's mode, else host prep and expert dispatch when not default."""
    model = str(r.params.get("model", r.invariants.get("model", "")))
    if model.startswith("laya"):
        mode = str(r.params.get("laya_mode", r.invariants.get("laya_mode", "eager")))
        return LAYA_MODES.get(mode, mode)
    tags = []
    if (prep := str(r.params.get("batch_host_prep", "inline"))) != "inline":
        tags.append(prep)
    if (experts := experts_of(r)) not in DEFAULT_EXPERTS:
        tags.append(f"{experts} experts")
    return ", ".join(tags) or BASE_VARIANT.get(model, "")


@dataclass(frozen=True)
class Series:
    key: str  # model, plus its variant: what makes two runs on one GPU the same series
    run_id: str
    host: str  # meta.cpu_model
    length: int  # the shortest state length the record has
    points: list[tuple[int, float]]  # (batch size, TFLOP/s on wall time), ascending B
    b1: Metrics  # batch_s<length>_b1_* without the prefix
    model: str = ""
    variant: str = ""
    prep: str = "inline"  # batch_host_prep
    grid: dict[int, dict[int, Metrics]] = field(default_factory=dict)  # length -> B -> metrics (no prefix)

    @property
    def slot(self) -> int:
        return family_of(self.model or self.key)

    @property
    def family(self) -> str:
        return FAMILIES[self.slot][1] if self.slot < OTHER else (self.model or self.key)

    @property
    def name(self) -> str:
        return f"{self.family} ({self.variant})" if self.variant else self.family

    @property
    def dash(self) -> int:
        return DASH_OF.get(self.variant, 0)

    @property
    def order(self) -> tuple[int, int, str]:
        return self.slot, self.dash, self.variant


@dataclass(frozen=True)
class Panel:
    platform: str
    gpu: str
    series: list[Series]

    @property
    def label(self) -> str:
        return f"{rr.gpu_name(self.gpu)} ({self.platform})"


def series_key(r: RunRecord) -> str:
    model = str(r.params.get("model", r.invariants.get("model", "")))
    if model.startswith("laya"):
        return f"laya {r.params.get('laya_mode', r.invariants.get('laya_mode', 'eager'))}"
    variant = variant_of(r)
    return model if variant in ("", BASE_VARIANT.get(model)) else f"{model} {variant}"


def series_of(r: RunRecord) -> Series | None:
    m = r.results.get(BENCH)
    if m is None or not m or "batch_nsys_n" in m:
        return None
    by_len: dict[int, list[tuple[int, float]]] = {}
    grid: dict[int, dict[int, Metrics]] = {}
    for k, v in m.items():
        if hit := _ANY.match(k):
            grid.setdefault(int(hit[1]), {}).setdefault(int(hit[2]), {})[hit[3]] = v
        if (hit := _POINT.match(k)) and math.isfinite(v) and v > 0:
            by_len.setdefault(int(hit[1]), []).append((int(hit[2]), v))
    if not by_len:
        return None
    length = min(by_len)
    model = str(r.params.get("model", r.invariants.get("model", "")))
    return Series(
        series_key(r),
        r.run_id,
        str(r.meta.get("cpu_model", "?")),
        length,
        sorted(by_len[length]),
        dict(grid[length].get(1, {})),
        model,
        variant_of(r),
        str(r.params.get("batch_host_prep", "inline")),
        {n: dict(sorted(bs.items())) for n, bs in sorted(grid.items())},
    )


def panels(recs: Sequence[RunRecord]) -> list[Panel]:
    """One panel per (platform, GPU), one record per series, in FAMILIES order. Of several records for a
    series the one covering the most batch sizes wins, then the newest (records are never edited; a re-run
    supersedes)."""
    best: dict[tuple[str, str, str], tuple[tuple[int, str], Series]] = {}
    for r in recs:
        if (s := series_of(r)) is None:
            continue
        key = (rr.platform(r.invariants), str(r.invariants.get("gpu", "?")), s.key)
        rank = (len(s.points), r.run_id)
        if key not in best or rank > best[key][0]:
            best[key] = (rank, s)
    grouped: dict[tuple[str, str], list[Series]] = {}
    for (plat, gpu, _), (_, s) in best.items():
        grouped.setdefault((plat, gpu), []).append(s)
    order = ("L4", "A100", "H100", "H200", "B200")

    def rank(pg: tuple[str, str]) -> tuple[int, str]:
        return next((i for i, o in enumerate(order) if o in pg[1]), len(order)), pg[1]

    return [
        Panel(plat, gpu, sorted(ss, key=lambda s: s.order))
        for (plat, gpu), ss in sorted(grouped.items(), key=lambda kv: rank(kv[0]))
    ]


_CPU = re.compile(r"^(GenuineIntel|AuthenticAMD) family (\d+) model (\d+)$")
HOSTS = {  # (vendor, family, model) -> name, for the CPUs Modal containers have landed on
    ("GenuineIntel", 6, 85): "Skylake/Cascade Lake",
    ("GenuineIntel", 6, 106): "Ice Lake",
    ("GenuineIntel", 6, 143): "Sapphire Rapids",
    ("GenuineIntel", 6, 207): "Emerald Rapids",
    ("AuthenticAMD", 23, 49): "Zen 2",
    ("AuthenticAMD", 25, 1): "Zen 3",
    ("AuthenticAMD", 25, 17): "Zen 4",
    ("AuthenticAMD", 26, 2): "Zen 5",
}


def host_name(cpu: str) -> str:
    """A short name for a meta.cpu_model ("GenuineIntel family 6 model 143" -> "Sapphire Rapids")."""
    if hit := _CPU.match(cpu.strip()):
        return HOSTS.get((hit[1], int(hit[2]), int(hit[3])), cpu)
    return cpu


def _f(v: float, digits: int = 1) -> str:
    return "-" if not math.isfinite(v) else f"{v:.{digits}f}"


def split(s: Series) -> tuple[float, float]:
    """(GPU kernel share, everything else) of the batch-1 wall time, clipped to [0, 1]."""
    busy = s.b1.get("kernel_busy", math.nan)
    if not math.isfinite(busy):
        return math.nan, math.nan
    busy = min(max(busy, 0.0), 1.0)
    return busy, 1.0 - busy


def b1_table(ps: Sequence[Panel]) -> tuple[list[str], list[list[str]]]:
    header = ["GPU", "Series", "Host CPU", "State tokens", "Wall ms", "Kernel ms", "CPU share", "Kernels"]
    rows = []
    for p in ps:
        for s in p.series:
            rows.append(
                [
                    p.label,
                    s.name,
                    host_name(s.host),
                    str(s.length),
                    _f(s.b1.get("wall_ms", math.nan)),
                    _f(s.b1.get("kernel_ms", math.nan)),
                    _f(split(s)[1], 2),
                    _f(s.b1.get("kernels", math.nan), 0),
                ]
            )
    return header, rows


def sizes(ps: Sequence[Panel]) -> list[int]:
    return sorted({b for p in ps for s in p.series for b, _ in s.points})


def curve_table(ps: Sequence[Panel]) -> tuple[list[str], list[list[str]]]:
    bs = sizes(ps)
    header = ["GPU", "Series", *[f"B={b}" for b in bs], "Gain"]
    rows = []
    for p in ps:
        for s in p.series:
            at = dict(s.points)
            gain = s.points[-1][1] / s.points[0][1] if s.points and s.points[0][0] == 1 else math.nan
            rows.append([p.label, s.name, *[_f(at.get(b, math.nan), 0) for b in bs], f"{_f(gain)}x"])
    return header, rows


# ---------------------------------------------------------------- charts

W, H = 420, 300
LEFT, RIGHT, TOP, BOTTOM = 52, 16, 12, 40


def swatch(slot: int, dash: int) -> str:
    """A short line sample in a series' hue and dash, for legends."""
    da = f' stroke-dasharray="{DASHES[dash % len(DASHES)]}"' if dash % len(DASHES) else ""
    return (
        f'<svg viewBox="0 0 28 10" aria-hidden="true"><line x1="1" x2="27" y1="5" y2="5" '
        f'stroke="var(--bs-{slot})" stroke-width="2.5"{da}/></svg>'
    )


def legend(series: Sequence[Series]) -> str:
    """One entry per distinct series name, in FAMILIES order."""
    seen: dict[str, Series] = {}
    for s in sorted(series, key=lambda s: s.order):
        seen.setdefault(s.name, s)
    return "".join(
        f'<span class="key">{swatch(s.slot, s.dash)}{html.escape(name)}</span>' for name, s in seen.items()
    )


def line(s: Series, d: str) -> str:
    da = f' stroke-dasharray="{DASHES[s.dash % len(DASHES)]}"' if s.dash % len(DASHES) else ""
    return f'<path class="line" stroke="var(--bs-{s.slot})"{da} d="{d}"/>'


def curve_charts(ps: Sequence[Panel]) -> str:
    """Small multiples, one per GPU: TFLOP/s on wall time vs batch size (log2 x, log10 y, shared y)."""
    bs = sizes(ps)
    ys = [v for p in ps for s in p.series for _, v in s.points]
    if not bs or not ys:
        return ""
    y0, y1 = math.floor(math.log10(min(ys))), math.ceil(math.log10(max(ys)))
    x0, x1 = math.log2(bs[0]), math.log2(bs[-1])

    def x(b: float) -> float:
        return LEFT + (math.log2(b) - x0) / max(x1 - x0, 1e-9) * (W - LEFT - RIGHT)

    def y(v: float) -> float:
        return H - BOTTOM - (math.log10(v) - y0) / max(y1 - y0, 1e-9) * (H - TOP - BOTTOM)

    figs = []
    for p in ps:
        out = []
        for b in bs:
            out.append(f'<line class="grid" x1="{x(b):.1f}" x2="{x(b):.1f}" y1="{TOP}" y2="{H - BOTTOM}"/>')
            out.append(
                f'<text class="muted" x="{x(b):.1f}" y="{H - BOTTOM + 15}" text-anchor="middle">{b}</text>'
            )
        for e in range(y0, y1 + 1):
            v = 10.0**e
            out.append(f'<line class="grid" x1="{LEFT}" x2="{W - RIGHT}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
            out.append(
                f'<text class="muted" x="{LEFT - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>'
            )
        for s in p.series:
            color = f"var(--bs-{s.slot})"
            d = " ".join(
                f"{'M' if i == 0 else 'L'}{x(b):.1f},{y(v):.1f}" for i, (b, v) in enumerate(s.points)
            )
            out.append(line(s, d))
            for b, v in s.points:
                tip = (
                    f"{s.name} on {p.label}, B = {b}: {v:.1f} TFLOP/s "
                    f"({s.length} tokens; host {s.host}; run {s.run_id})"
                )
                out.append(
                    f'<g class="pt"><title>{html.escape(tip)}</title>'
                    f'<circle cx="{x(b):.1f}" cy="{y(v):.1f}" r="8" fill="transparent"/>'
                    f'<circle cx="{x(b):.1f}" cy="{y(v):.1f}" r="3.5" fill="{color}" '
                    'stroke="var(--surface)" stroke-width="2"/></g>'
                )
        out.append(
            f'<text class="muted" x="{(LEFT + W - RIGHT) / 2}" y="{H - 6}" text-anchor="middle">'
            "Batch size (independent requests per forward)</text>"
            f'<text class="muted" transform="translate(13 {(TOP + H - BOTTOM) / 2}) rotate(-90)" '
            'text-anchor="middle">TFLOP/s (wall time)</text>'
        )
        title = html.escape(p.label)
        figs.append(
            f'<figure class="card"><figcaption><strong>{title}</strong></figcaption>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Achieved TFLOP/s vs batch size on {title}">'
            f"{''.join(out)}</svg></figure>"
        )
    lengths = sorted({s.length for p in ps for s in p.series})
    head = (
        "Achieved TFLOP/s vs batch size, batch-1 state "
        f"{'/'.join(map(str, lengths))} tokens (cost-model FLOPs over wall time; same log scale on every GPU)"
    )
    return (
        '<div class="bsw"><figure class="card chart">'
        f"<figcaption><strong>{html.escape(head)}</strong></figcaption>"
        f'<div class="legend">{legend([s for p in ps for s in p.series])}</div>'
        f'<div class="multiples">{"".join(figs)}</div></figure></div>'
    )


def split_chart(ps: Sequence[Panel]) -> str:
    """One horizontal bar per (GPU, series): share of batch-1 wall time in GPU kernels vs everything else."""
    rows = [(p, s) for p in ps for s in p.series if math.isfinite(split(s)[0])]
    if not rows:
        return ""
    label_w, value_w, bar_h, gap, group_gap = 250, 92, 16, 6, 12
    width = 820
    bar_w = width - label_w - value_w
    y, out, last = 24, [], None
    for p, s in rows:
        if p.label != last:
            if last is not None:
                y += group_gap
            hosts = sorted({host_name(t.host) for t in p.series})
            host = f"host {hosts[0]}" if len(hosts) == 1 else "hosts differ per row"
            out.append(
                f'<text x="0" y="{y + 12}"><tspan font-weight="600">{html.escape(p.label)}</tspan>'
                f'<tspan class="muted" dx="8">{html.escape(host)}</tspan></text>'
            )
            y += bar_h + gap
            last = p.label
        gpu, host = split(s)
        wall = s.b1.get("wall_ms", math.nan)
        kern = s.b1.get("kernel_ms", math.nan)
        tip = (
            f"{s.name} on {p.label}, batch 1, {s.length} tokens: wall {wall:.1f} ms, "
            f"GPU kernels {kern:.1f} ms "
            f"({gpu:.0%}), outside kernels {host:.0%}; host {s.host}; run {s.run_id}"
        )
        gx, gw = label_w, max(gpu * bar_w - 1, 0)
        row = s.name if len({t.host for t in p.series}) == 1 else f"{s.name} · {host_name(s.host)}"
        out.append(
            f'<g class="seg"><title>{html.escape(tip)}</title>'
            f'<text class="muted" x="12" y="{y + 12}">{html.escape(row)}</text>'
            f'<rect class="gpu" x="{gx}" y="{y}" width="{gw:.1f}" height="{bar_h}" rx="4"/>'
            f'<rect class="host" x="{gx + gw + 2:.1f}" y="{y}" width="{max(bar_w - gw - 2, 0):.1f}" '
            f'height="{bar_h}" rx="4"/>'
            f'<text x="{gx + 6}" y="{y + 12}" style="fill:#fff" font-size="11">'
            f"{'' if gpu < 0.12 else f'{gpu:.0%}'}</text>"
            f'<text class="muted" x="{width - 4}" y="{y + 12}" text-anchor="end">{wall:.1f} ms</text></g>'
        )
        y += bar_h + gap
    legend = (
        '<span class="key"><span class="sw" style="background:var(--bs-gpu)"></span>GPU kernels</span>'
        '<span class="key"><span class="sw" style="background:var(--track);outline:1px solid var(--ink-2)">'
        "</span>"
        "outside kernels: host work, launches, idle GPU</span>"
        '<span class="key">right: batch-1 wall time</span>'
    )
    head = "Where batch-1 latency goes (shortest state length; newest record per GPU and series)"
    return (
        '<div class="bsw"><figure class="card chart">'
        f"<figcaption><strong>{html.escape(head)}</strong></figcaption>"
        f'<div class="legend">{legend}</div>'
        f'<svg viewBox="0 0 {width} {y + 4}" role="img" aria-label="{html.escape(head)}">{"".join(out)}</svg>'
        "</figure></div>"
    )
