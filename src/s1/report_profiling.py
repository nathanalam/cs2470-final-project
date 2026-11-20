"""Hardware profiling in the report: the Nsight Systems length axis with GPU-metrics sampling (`s1 nsys` with
`nsys_gpu_metrics`) and sampled Nsight Compute kernel classes (`s1 ncu` with `ncu_launch_count`).

Three tables, each built from the newest record per key (records are never edited; a re-run supersedes):

- `nsys_table`: one row per (platform, GPU, model, state length, warm-up). Wall and kernel time per decision,
  the GPU's share, the kernel-time roofline point (cost-model FLOPs and bytes over kernel time), and the
  GPU counters averaged over the samples where the SMs were active (DRAM read/write, tensor, SM activity, as %
  of peak). "Warm-up" is "warm" when the profiled questions were the warm-up ones and "cold" when the warm-up
  answered other questions (`batch_nsys_warmup_offset` > 0), so every profiled input shape was new.
- `h200_table`: kernel and wall time per decision on the H100 over the H200, per (model, length, warm-up),
  for Modal records that have both GPUs. Same compute, 1.41x the measured bandwidth: ~1.41x would be
  memory-bound, ~1.0x compute-bound.
- `ncu_table`: sampled ncu records (`roofline_hw_sampled`), one row per (GPU, model, length, metric set,
  kernel class): share of profiled kernel time, kernel ms, TFLOP/s, DRAM GB/s and the speed-of-light %.
  `ncu_ratio_table` puts the H100/H200 kernel-ms ratio per class next to each other when both exist.

Records without these keys (every run before 2026-11-15) are left out, so the tables are empty, and the report
omits them, until such runs exist.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from . import report_batch as rb
from . import report_roofline as rr
from .record import Metrics, RunRecord

NSYS_BENCH = "batch"
NCU_BENCH = "roofline"
GM = "batch_nsys_gm_"
COUNTERS = (  # (column, GPU-metrics key fragment, as nsys 2025 names them)
    ("DRAM read %", "dram_read_bandwidth_throughput"),
    ("DRAM write %", "dram_write_bandwidth_throughput"),
    ("Tensor %", "tensor_active_throughput"),
    ("SMs %", "sms_active_throughput"),
)
CLASSES = ("gemm", "attention", "mixer", "routing", "norm", "elementwise", "other")
_CLASS_MS = re.compile(r"^roofline_hw_(?:s(\d+)_)?(" + "|".join(CLASSES) + r")_ms$")
GPU_ORDER = ("L4", "A100", "L40S", "H100", "H200", "B200")


def _f(v: float | None, digits: int = 1) -> str:
    return "-" if v is None or not math.isfinite(v) else f"{v:,.{digits}f}"


def _gpu(r: RunRecord) -> str:
    return str(r.invariants.get("gpu", "?")).removeprefix("NVIDIA ")


def _gpu_rank(gpu: str) -> int:
    return next((i for i, g in enumerate(GPU_ORDER) if g in gpu), len(GPU_ORDER))


def _model(r: RunRecord) -> str:
    return str(r.params.get("model", "?"))


def _newest(rows: dict[tuple[str, ...], RunRecord], key: tuple[str, ...], r: RunRecord) -> None:
    if key not in rows or r.run_id > rows[key].run_id:
        rows[key] = r


# ---- nsys length axis ----


def nsys_records(recs: Sequence[RunRecord]) -> list[RunRecord]:
    """Newest nsys record with GPU metrics per (platform, GPU, model, length, warm-up), sorted by model."""
    newest: dict[tuple[str, ...], RunRecord] = {}
    for r in recs:
        m = r.results.get(NSYS_BENCH, {})
        if f"{GM}samples" not in m or "batch_nsys_state_tokens" not in m:
            continue
        key = (
            rr.platform(r.invariants),
            _gpu(r),
            _model(r),
            str(int(m["batch_nsys_state_tokens"])),
            warmup(r),
        )
        _newest(newest, key, r)
    return sorted(
        newest.values(),
        key=lambda r: (
            _model(r),
            _gpu_rank(_gpu(r)),
            _gpu(r),
            warmup(r),
            r.results[NSYS_BENCH]["batch_nsys_state_tokens"],
        ),
    )


def warmup(r: RunRecord) -> str:
    """ "cold" when the nsys warm-up answered other rows than the profiled ones, else "warm"."""
    offset = r.params.get("batch_nsys_warmup_offset", 0)
    return (
        "cold" if isinstance(offset, int | float) and not isinstance(offset, bool) and offset > 0 else "warm"
    )


def _active(m: Metrics, frag: str) -> float | None:
    return m.get(f"{GM}{frag}_active", m.get(f"{GM}{frag}"))


def nsys_table(recs: Sequence[RunRecord]) -> tuple[list[str], list[list[str]]]:
    header = [
        "Model", "GPU", "Host CPU", "State tokens", "Warm-up", "Wall ms", "Kernel ms", "GPU share",
        "Kernel TFLOP/s", "Kernel GB/s", *[c for c, _ in COUNTERS], "Run",
    ]  # fmt: skip
    rows = []
    for r in nsys_records(recs):
        m = r.results[NSYS_BENCH]
        rows.append(
            [
                _model(r),
                f"{_gpu(r)} ({rr.platform(r.invariants)})",
                rb.host_name(str(r.meta.get("cpu_model", "?"))),
                f"{int(m['batch_nsys_state_tokens']):,}",
                warmup(r),
                _f(m.get("batch_nsys_wall_ms_per_decision")),
                _f(m.get("batch_nsys_kernel_ms")),
                _f(m.get("batch_nsys_gpu_share"), 2),
                _f(m.get("batch_nsys_kernel_tflops"), 0),
                _f(m.get("batch_nsys_kernel_gbs"), 0),
                *[_f(_active(m, frag), 0) for _, frag in COUNTERS],
                r.run_id,
            ]
        )
    return header, rows


def h200_table(recs: Sequence[RunRecord]) -> tuple[list[str], list[list[str]]]:
    """H100 / H200 kernel and wall time per decision, per (model, length, warm-up), Modal records only."""
    by: dict[tuple[str, int, str], dict[str, Metrics]] = {}
    for r in nsys_records(recs):
        if rr.platform(r.invariants) != "Modal":
            continue
        gpu = _gpu(r)
        tag = "H200" if "H200" in gpu else "H100" if "H100" in gpu else None
        if tag is None:
            continue
        m = r.results[NSYS_BENCH]
        by.setdefault((_model(r), int(m["batch_nsys_state_tokens"]), warmup(r)), {})[tag] = m
    header = ["Model", "State tokens", "Warm-up", "Kernel ms H100", "Kernel ms H200", "Kernel speedup",
              "Wall speedup", "DRAM read % H100", "DRAM read % H200"]  # fmt: skip
    rows = []
    for (model, length, warm), pair in sorted(by.items()):
        if "H100" not in pair or "H200" not in pair:
            continue
        a, b = pair["H100"], pair["H200"]

        def ratio(key: str, a: Metrics = a, b: Metrics = b) -> float | None:
            x, y = a.get(key), b.get(key)
            return x / y if x is not None and y is not None and y != 0 else None

        rows.append(
            [
                model, f"{length:,}", warm,
                _f(a.get("batch_nsys_kernel_ms")), _f(b.get("batch_nsys_kernel_ms")),
                _f(ratio("batch_nsys_kernel_ms"), 2) + "x",
                _f(ratio("batch_nsys_wall_ms_per_decision"), 2) + "x",
                _f(_active(a, COUNTERS[0][1]), 0), _f(_active(b, COUNTERS[0][1]), 0),
            ]
        )  # fmt: skip
    return header, rows


# ---- sampled ncu kernel classes ----


def ncu_records(recs: Sequence[RunRecord]) -> list[RunRecord]:
    """Newest sampled-ncu record per (platform, GPU, model, metric set, launch count)."""
    newest: dict[tuple[str, ...], RunRecord] = {}
    for r in recs:
        m = r.results.get(NCU_BENCH, {})
        if m.get("roofline_hw_sampled") != 1.0:
            continue
        key = (
            rr.platform(r.invariants),
            _gpu(r),
            _model(r),
            _metric_set(r),
            str(r.params.get("ncu_launch_count", 0)),
        )
        _newest(newest, key, r)
    return sorted(newest.values(), key=lambda r: (_model(r), _metric_set(r), _gpu_rank(_gpu(r))))


def _metric_set(r: RunRecord) -> str:
    return str(r.params.get("ncu_metrics", r.invariants.get("ncu_metrics", "full")))


def _classes(m: Metrics) -> list[tuple[int | None, str]]:
    """(state length or None for the whole profiled range, class) present in a record, length-major."""
    found = {(int(x[1]) if x[1] else None, x[2]) for k in m if (x := _CLASS_MS.match(k))}
    lengths = sorted({n for n, _ in found if n is not None})
    if lengths:  # on the length axis, report per length (the whole-range totals repeat them)
        return [(n, c) for n in lengths for c in CLASSES if (n, c) in found]
    return [(None, c) for c in CLASSES if (None, c) in found]


def ncu_table(recs: Sequence[RunRecord]) -> tuple[list[str], list[list[str]]]:
    header = ["Model", "GPU", "Metrics", "Kernels", "State tokens", "Class", "Share", "Kernel ms", "TFLOP/s",
              "DRAM GB/s", "DRAM %", "SM %", "Run"]  # fmt: skip
    rows = []
    for r in ncu_records(recs):
        m = r.results[NCU_BENCH]
        for n, cls in _classes(m):
            p = f"roofline_hw_s{n}_{cls}_" if n is not None else f"roofline_hw_{cls}_"
            rows.append(
                [
                    _model(r), f"{_gpu(r)} ({rr.platform(r.invariants)})", _metric_set(r),
                    _f(m.get("roofline_hw_kernels"), 0), f"{n:,}" if n is not None else "-", cls,
                    _f(m.get(f"{p}share", math.nan) * 100, 0) + "%", _f(m.get(f"{p}ms"), 2),
                    _f(m.get(f"{p}tflops"), 0), _f(m.get(f"{p}dram_gbs"), 0),
                    _f(m.get(f"{p}dram_pct"), 0), _f(m.get(f"{p}sm_pct"), 0), r.run_id,
                ]
            )  # fmt: skip
    return header, rows


def ncu_ratio_table(recs: Sequence[RunRecord]) -> tuple[list[str], list[list[str]]]:
    """Per (model, metric set, length, class): H100 kernel ms / H200 kernel ms, Modal sampled records."""
    by: dict[tuple[str, str], dict[str, Metrics]] = {}
    for r in ncu_records(recs):
        gpu = _gpu(r)
        tag = "H200" if "H200" in gpu else "H100" if "H100" in gpu else None
        if tag is not None and rr.platform(r.invariants) == "Modal":
            by.setdefault((_model(r), _metric_set(r)), {})[tag] = r.results[NCU_BENCH]
    header = ["Model", "Metrics", "State tokens", "Class", "Share (H100)", "Kernel ms H100", "Kernel ms H200",
              "H200 speedup"]  # fmt: skip
    rows = []
    for (model, ms), pair in sorted(by.items()):
        if "H100" not in pair or "H200" not in pair:
            continue
        a, b = pair["H100"], pair["H200"]
        total_a = total_b = 0.0
        for n, cls in _classes(a):
            p = f"roofline_hw_s{n}_{cls}_" if n is not None else f"roofline_hw_{cls}_"
            x, y = a.get(f"{p}ms"), b.get(f"{p}ms")
            if x is None or y is None or y == 0:
                continue
            total_a, total_b = total_a + x, total_b + y
            rows.append(
                [model, ms, f"{n:,}" if n is not None else "-", cls,
                 _f(a.get(f"{p}share", math.nan) * 100, 0) + "%",
                 _f(x, 2), _f(y, 2), _f(x / y, 2) + "x"]
            )  # fmt: skip
        if total_b > 0:
            rows.append(
                [model, ms, "", "all profiled", "100%", _f(total_a, 2), _f(total_b, 2),
                 _f(total_a / total_b, 2) + "x"]
            )  # fmt: skip
    return header, rows


NOTE = (
    "Hardware profiling (2026-11-15 on): Nsight Systems length runs with GPU-metrics sampling, and sampled "
    "Nsight Compute. nsys wall time includes its own tracing, so compare kernel time across GPUs; GPU "
    "counters "
    "are means over samples where the SMs were active, in % of the GPU's peak. Kernel TFLOP/s and GB/s are "
    "cost-model work over kernel time (host time removed). Sampled ncu profiles the first N kernels of one "
    "question (about three decoder layers for N = 200), so its rates are per class, not whole-forward totals."
)
