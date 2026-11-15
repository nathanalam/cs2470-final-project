"""CPU vs GPU share of a decision's latency from Nsight Systems (`s1 nsys`), and the API-name rules that the
in-process torch.profiler trace (benchmarks/batch_sweep.py) shares with it.

`s1 nsys CONFIG` runs each model of a `batch` config under

    nsys profile --trace=cuda,osrt --sample=<sample> --capture-range=cudaProfilerApi --capture-range-end=stop

with `batch_nsys = N` set, so the benchmark answers N batch-1 decisions inside one CUDA profiler range
(cudaProfilerStart/Stop) and records that region's host wall time. `nsys stats` then writes the
`cuda_api_sum` and `cuda_gpu_kern_sum` reports as CSV, and `summarize` turns them into, per decision:

    batch_nsys_kernel_ms        summed GPU kernel time (what the GPU spent computing), CUDA graphs included
    batch_nsys_kernels          kernels launched outside CUDA graphs
    batch_nsys_graph_ms         GPU time of CUDA graph replays (part of kernel_ms)
    batch_nsys_graph_launches   CUDA graph replays
    batch_nsys_launch_api_ms    host time in kernel and graph launch calls
    batch_nsys_sync_ms          host time blocked in synchronize calls (waiting for the GPU)
    batch_nsys_gpu_share        kernel time / wall time: the GPU's share of the latency
    batch_nsys_cpu_share        1 - gpu_share: host work, launch gaps and idle GPU between kernels

nsys traces a CUDA graph as one unit by default (--cuda-graph-trace=graph): its kernels are missing from
cuda_gpu_kern_sum, and each replay is one row of CUPTI_ACTIVITY_KIND_GRAPH_TRACE in the report's SQLite
export (`graph_trace`). That row spans the graph's first kernel start to its last kernel end, so it also
counts any small gaps inside the graph. Rows from before the capture range (earlier replays nsys still
reports) are dropped.

Kernel time on one stream never overlaps, so gpu_share <= 1 for these single-stream models. nsys adds a little
host time to every API call it traces, so wall time under nsys is slightly above an untraced run's.

Kernel-time roofline: the benchmark also records the cost-model work of the profiled decisions
(`batch_nsys_gflop`, `batch_nsys_mb` per decision), so

    batch_nsys_kernel_tflops    cost-model FLOPs / summed kernel time: achieved FLOP/s with host time removed
    batch_nsys_kernel_gbs       cost-model bytes / summed kernel time

place each decision on the roofline by what the GPU did while it was busy, the number end-to-end timing hides
when most of a decision is host time.

GPU metrics (`nsys_gpu_metrics = true` in a config): nsys samples the GPU's own counters
(`--gpu-metrics-devices=all`, `nsys_gpu_metrics_hz` times a second, default 10 kHz), and `gpu_metrics` reads
them from the SQLite export (tables GPU_METRICS and TARGET_INFO_GPU_METRICS) inside the captured range. Each
metric nsys reports (names differ by GPU and nsys version, e.g. "DRAM Read Bandwidth [Throughput %]") becomes

    batch_nsys_gm_<slug>            mean over every sample in the range
    batch_nsys_gm_<slug>_active     mean over samples where the SMs were active (an "SMs Active" metric > 5),
                                    i.e. what the GPU sustained while it had work, idle gaps left out

plus batch_nsys_gm_samples and batch_nsys_gm_active_share. A metric that is missing stays absent, and the
names seen are noted in meta (batch_nsys_gm_names), so a run on a GPU or nsys version with other names is
still readable.

Torch-free: only parses `nsys stats --format csv` output and reads its SQLite export.
"""

from __future__ import annotations

import contextlib
import csv
import io
import re
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

from .record import Metrics

REPORTS = ("cuda_api_sum", "cuda_gpu_kern_sum")
SAMPLES = ("process-tree", "none")  # nsys --sample: CPU IP/backtrace sampling, or off
META_SAMPLE = "nsys_sample"
LAUNCH_HINTS = ("LaunchKernel", "GraphLaunch", "LaunchCooperativeKernel")
SYNC_HINT = "Synchronize"


def is_launch(name: str) -> bool:
    """A CUDA runtime/driver call that launches GPU work (one kernel, or a whole CUDA graph)."""
    return any(h in name for h in LAUNCH_HINTS)


def is_sync(name: str) -> bool:
    """A call that blocks the host until the GPU catches up (stream/event/device synchronize)."""
    return SYNC_HINT in name


def profile_args(report: str, sample: str = "process-tree", gpu_metrics_hz: int = 0) -> tuple[str, ...]:
    """`nsys profile` arguments (before the command): CUDA + OS runtime trace, CPU sampling, and only the
    cudaProfilerStart/Stop range, written to `report` (.nsys-rep is appended by nsys). `gpu_metrics_hz` > 0
    also samples the GPU's hardware counters that many times a second."""
    if sample not in SAMPLES:
        raise ValueError(f"nsys sample must be one of {SAMPLES}, not {sample!r}")
    if gpu_metrics_hz < 0:
        raise ValueError(f"nsys_gpu_metrics_hz must be >= 0, not {gpu_metrics_hz}")
    metrics = (
        ("--gpu-metrics-devices=all", f"--gpu-metrics-frequency={gpu_metrics_hz}") if gpu_metrics_hz else ()
    )
    return (
        "profile",
        "--trace=cuda,osrt",
        "--cuda-graph-trace=node",  # a graph's kernels one by one; the default traces a graph as one range
        f"--sample={sample}",
        f"--cpuctxsw={'process-tree' if sample != 'none' else 'none'}",
        *metrics,
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        "-o",
        report,
    )


def stats_args(report_file: str, out_base: str) -> tuple[str, ...]:
    """`nsys stats` arguments: both reports as CSV files `<out_base>_<report>.csv`."""
    reports: list[str] = []
    for r in REPORTS:
        reports += ["--report", r]
    return ("stats", *reports, "--format", "csv", "--force-export=true", "--output", out_base, report_file)


def _col(header: Sequence[str], *names: str) -> int:
    for i, h in enumerate(header):
        if any(n.lower() in h.lower() for n in names):
            return i
    raise ValueError(f"nsys CSV has none of {names} in its header {list(header)}")


def parse(text: str) -> list[tuple[str, float, int]]:
    """(name, total ns, calls or instances) per row of a `cuda_api_sum` or `cuda_gpu_kern_sum` CSV. Lines
    before the header (nsys notices) are skipped; an empty report (nothing in the range) is []."""
    start = text.find('"Time')
    if start < 0:
        start = text.find("Time (%)")
    if start < 0:
        return []
    rows = list(csv.reader(io.StringIO(text[start:])))
    header = rows[0]
    name, total, count = (
        _col(header, "Name"),
        _col(header, "Total Time"),
        _col(header, "Num Calls", "Instances", "Count"),
    )
    out = []
    for r in rows[1:]:
        if len(r) != len(header):
            continue
        try:
            out.append((r[name], float(r[total].replace(",", "")), int(float(r[count].replace(",", "")))))
        except ValueError:
            continue
    return out


GRAPH_TABLE = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
RUNTIME_TABLE = "CUPTI_ACTIVITY_KIND_RUNTIME"


def graph_trace(sqlite_file: Path) -> tuple[int, float]:
    """(CUDA graph replays, their summed GPU ns) inside the captured range, from the report's SQLite export.
    The range is the CUDA runtime calls' span; (0, 0.0) when the run replayed no graph."""
    with contextlib.closing(sqlite3.connect(f"file:{sqlite_file}?mode=ro", uri=True)) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if GRAPH_TABLE not in tables or RUNTIME_TABLE not in tables:
            return 0, 0.0
        n, total = db.execute(
            f"SELECT COUNT(*), COALESCE(SUM(g.end - g.start), 0) FROM {GRAPH_TABLE} AS g, "
            f"(SELECT MIN(start) AS s, MAX(end) AS e FROM {RUNTIME_TABLE}) AS r "
            "WHERE g.start >= r.s AND g.end <= r.e"
        ).fetchone()
    return int(n), float(total)


DEFAULT_GPU_METRICS_HZ = 10_000


def gpu_metrics_hz(settings: Mapping[str, object]) -> int:
    """Samples per second for `--gpu-metrics-frequency` from a run's settings: `nsys_gpu_metrics_hz`
    (default 10 kHz) when `nsys_gpu_metrics` is true, else 0 (no GPU metrics)."""
    on = settings.get("nsys_gpu_metrics", False)
    if isinstance(on, str):
        on = on.strip().lower() in ("1", "true", "yes", "on")
    if not on:
        return 0
    hz = settings.get("nsys_gpu_metrics_hz", DEFAULT_GPU_METRICS_HZ)
    if isinstance(hz, bool) or not isinstance(hz, int | float | str):
        raise TypeError(f"nsys_gpu_metrics_hz must be a number, not {hz!r}")
    value = int(float(hz))
    if value <= 0:
        raise ValueError(f"nsys_gpu_metrics_hz must be > 0, not {value}")
    return value


METRICS_TABLE = "GPU_METRICS"
METRIC_NAMES_TABLE = "TARGET_INFO_GPU_METRICS"
ACTIVE_HINT = re.compile(r"\bSMs? Active\b", re.IGNORECASE)
ACTIVE_MIN = 5.0  # an "SMs Active" sample above this (percent) counts as the GPU having work
GM_PREFIX = "batch_nsys_gm_"
META_GM_NAMES = "batch_nsys_gm_names"


def slug(name: str) -> str:
    """A metric name as a key fragment, e.g. 'DRAM Read Bandwidth [Throughput %]' ->
    'dram_read_bandwidth_throughput'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _columns(db: sqlite3.Connection, table: str) -> dict[str, str]:
    """{lowercase column name: column name} of `table`."""
    return {str(r[1]).lower(): str(r[1]) for r in db.execute(f"PRAGMA table_info({table})")}


def gpu_metrics(sqlite_file: Path) -> tuple[Metrics, str]:
    """(batch_nsys_gm_* means, '; '-joined metric names seen) from the SQLite export's GPU-metrics samples
    inside the captured range (the CUDA runtime calls' span). ({}, "") when the run sampled no GPU metrics or
    the export lacks the expected tables or columns."""
    with contextlib.closing(sqlite3.connect(f"file:{sqlite_file}?mode=ro", uri=True)) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if METRICS_TABLE not in tables or METRIC_NAMES_TABLE not in tables:
            return {}, ""
        s, n = _columns(db, METRICS_TABLE), _columns(db, METRIC_NAMES_TABLE)
        need_s, need_n = ("timestamp", "metricid", "value"), ("metricid", "metricname")
        if any(c not in s for c in need_s) or any(c not in n for c in need_n):
            return {}, ""
        by_type = "typeid" in s and "typeid" in n
        name_rows = db.execute(
            f"SELECT {n['typeid'] if by_type else '0'}, {n['metricid']}, {n['metricname']} "
            f"FROM {METRIC_NAMES_TABLE}"
        ).fetchall()
        names = {(int(t), int(i)): str(nm) for t, i, nm in name_rows}
        where = ""
        if RUNTIME_TABLE in tables:
            lo, hi = db.execute(f"SELECT MIN(start), MAX(end) FROM {RUNTIME_TABLE}").fetchone()
            if lo is not None and hi is not None:
                where = f" WHERE {s['timestamp']} >= {int(lo)} AND {s['timestamp']} <= {int(hi)}"
        rows = db.execute(
            f"SELECT {s['typeid'] if by_type else '0'}, {s['metricid']}, {s['timestamp']}, {s['value']} "
            f"FROM {METRICS_TABLE}{where}"
        ).fetchall()
    series: dict[str, dict[int, float]] = {}
    for t, i, ts, v in rows:
        name = names.get((int(t), int(i)))
        if name is not None and v is not None:
            series.setdefault(name, {})[int(ts)] = float(v)
    return summarize_gpu_metrics(series), "; ".join(sorted(series))


def summarize_gpu_metrics(series: Mapping[str, Mapping[int, float]]) -> Metrics:
    """{metric name: {timestamp: value}} -> batch_nsys_gm_<slug> (mean of all samples) and, when an
    "SMs Active" metric exists, batch_nsys_gm_<slug>_active (mean over the timestamps where it is above
    ACTIVE_MIN) and batch_nsys_gm_active_share."""
    if not series:
        return {}
    active_name = next((name for name in sorted(series) if ACTIVE_HINT.search(name)), None)
    active = (
        {ts for ts, v in series[active_name].items() if v > ACTIVE_MIN} if active_name is not None else None
    )
    m: Metrics = {f"{GM_PREFIX}samples": float(max(len(v) for v in series.values()))}
    if active_name is not None and active is not None:
        m[f"{GM_PREFIX}active_share"] = len(active) / len(series[active_name]) if series[active_name] else 0.0
    for name, values in sorted(series.items()):
        if not values:
            continue
        key = f"{GM_PREFIX}{slug(name)}"
        m[key] = sum(values.values()) / len(values)
        if active is not None and active:
            busy = [v for ts, v in values.items() if ts in active]
            if busy:
                m[f"{key}_active"] = sum(busy) / len(busy)
    return m


def summarize(
    api: Sequence[tuple[str, float, int]],
    kernels: Sequence[tuple[str, float, int]],
    run: Mapping[str, float],
    graphs: tuple[int, float] = (0, 0.0),
) -> Metrics:
    """Per-decision CPU/GPU split of the profiled region; `run` holds the benchmark's `batch_nsys_n` and
    `batch_nsys_wall_ms` (the region's host wall time), `graphs` is `graph_trace`'s (replays, GPU ns)."""
    n = run.get("batch_nsys_n", 0.0)
    wall = run.get("batch_nsys_wall_ms", 0.0)
    if n <= 0 or wall <= 0:
        raise ValueError("the run recorded no nsys region (batch_nsys_n / batch_nsys_wall_ms)")
    graph_n, graph_ns = graphs
    kernel_ms = (sum(t for _, t, _ in kernels) + graph_ns) / 1e6
    launches = sum(c for name, _, c in api if is_launch(name))
    launch_ms = sum(t for name, t, _ in api if is_launch(name)) / 1e6
    sync_ms = sum(t for name, t, _ in api if is_sync(name)) / 1e6
    api_ms = sum(t for _, t, _ in api) / 1e6
    share = kernel_ms / wall
    per_decision_ms = kernel_ms / n
    work: Metrics = {}
    if per_decision_ms > 0 and "batch_nsys_gflop" in run:
        work["batch_nsys_kernel_tflops"] = run["batch_nsys_gflop"] / per_decision_ms  # GFLOP/ms = TFLOP/s
    if per_decision_ms > 0 and "batch_nsys_mb" in run:
        work["batch_nsys_kernel_gbs"] = run["batch_nsys_mb"] / per_decision_ms  # MB/ms = GB/s
    return work | {
        "batch_nsys_cpu_share": 1.0 - share,
        "batch_nsys_gpu_share": share,
        "batch_nsys_wall_ms_per_decision": wall / n,
        "batch_nsys_kernel_ms": kernel_ms / n,
        "batch_nsys_kernels": sum(c for _, _, c in kernels) / n,
        "batch_nsys_graph_ms": graph_ns / 1e6 / n,
        "batch_nsys_graph_launches": graph_n / n,
        "batch_nsys_launches": launches / n,
        "batch_nsys_launch_api_ms": launch_ms / n,
        "batch_nsys_sync_ms": sync_ms / n,
        "batch_nsys_api_ms": api_ms / n,
        "batch_nsys_launch_share": launch_ms / wall,
        "batch_nsys_sync_share": sync_ms / wall,
    }


def top_kernels(kernels: Sequence[tuple[str, float, int]], top: int = 8) -> str:
    """The `top` kernels by summed time, as '<share> <name>' (for `meta`)."""
    whole = sum(t for _, t, _ in kernels)
    ranked = sorted(kernels, key=lambda k: k[1], reverse=True)[:top]
    return "; ".join(f"{t / whole:.1%} x{c} {name[:100]}" for name, t, c in ranked) if whole > 0 else ""
