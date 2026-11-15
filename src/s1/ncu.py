"""Hardware-counter roofline: run a config's models under Nsight Compute (`s1 ncu`) and sum the counters.

The roofline benchmark (with `roofline_ncu = N`) re-answers its first N questions inside a CUDA profiler
range; ncu, started with `--profile-from-start off --target-processes all` (so it also follows elephant-1's
worker process), replays each kernel in that range and reads:

    FLOPs  = tensor-core ops (bf16/fp16/tf32 sources; an FMA counts 2, checked: a 4096^3 bf16 GEMM reads
             exactly 2 * 4096^3) + CUDA-core fp32 (2 * FFMA + FADD + FMUL) + fp16 (2 * HFMA + HADD + HMUL)
    bytes  = DRAM bytes read + written (real traffic: L2 misses, re-reads included)

so `roofline_hw_intensity` is the measured operational intensity, and `roofline_hw_tflops` divides the
measured FLOPs by the same questions' GPU time from the run's timed pass (ncu's own kernel durations are taken
at the clocks ncu sets, so they are kept separately as `roofline_hw_kernel_ms`). `roofline_hw_flop_ratio` and
`roofline_hw_byte_ratio` compare the counters with the cost model (s1.flops) for the same forwards.

Clock control (`ncu_clock_control`, ncu's `--clock-control`): "base" is ncu's default, which locks clocks to
base (what every Slurm run so far used, and the flag is then left off the command line, so it is unchanged);
"none" leaves the GPU's own clocks. On Modal the default is "none", because clock locking fails on Modal
workers. The value is recorded as `roofline_hw_clocks_locked` (1 = base, 0 = none) next to the kernel times it
qualifies.

Kernel classes (`summarize_named`): every kernel is put in one class by its name (`classify`: attention,
mixer, gemm, routing, norm, elementwise, other), and each class present gets `roofline_hw_<class>_*`: its
share of kernel time, kernel ms, DRAM GB/s and TFLOP/s over its own kernel time, and the time-weighted mean of
ncu's speed-of-light percentages (`dram__throughput`, `sm__throughput`, % of the GPU's peak sustained rate). A
class with no kernels has no keys (never zero-filled). `roofline_hw_busy` is summed kernel time over the
timed-pass GPU time of the same questions: the share of the forward the GPU spent running kernels.

State lengths: on the roofline length axis the benchmark profiles `roofline_ncu` questions at every length,
shortest first, and launches a marker kernel (`torch.cuda._sleep`, named "spin") before each length. The
kernels after the i-th marker belong to the i-th shortest length, and get the totals and classes above as
`roofline_hw_s<L>_*`.

Torch-free: only parses ncu's `--csv --page raw` output.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping, Sequence

from . import env
from .record import Metrics

TENSOR = (
    "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
    "sm__ops_path_tensor_src_fp16_dst_fp32.sum",
    "sm__ops_path_tensor_src_fp16_dst_fp16.sum",
    "sm__ops_path_tensor_src_tf32_dst_fp32.sum",
)
CORE = {  # CUDA-core instruction counts and the FLOPs each is worth
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 2,
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum": 2,
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum": 1,
}
DRAM = ("dram__bytes_read.sum", "dram__bytes_write.sum")
TIME = "gpu__time_duration.sum"
DRAM_PCT = "dram__throughput.avg.pct_of_peak_sustained_elapsed"  # speed of light: DRAM, % of peak
SM_PCT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"  # speed of light: SM, % of peak
METRICS = (*TENSOR, *CORE, *DRAM, TIME, DRAM_PCT, SM_PCT)
NAME = "Kernel Name"
MARKER = re.compile(r"spin", re.IGNORECASE)  # torch.cuda._sleep's kernel, launched between state lengths
LENGTH_KEY = re.compile(r"^roofline_hw_s(\d+)_timed_ms$")
# First match wins, so attention (fmha_cutlass..., cudnn ... sdpa) is taken before the generic gemm patterns.
CLASSES = (
    ("attention", re.compile(r"sdpa|fmha|flash|attention|attn", re.IGNORECASE)),
    # linear-cost sequence mixers: fla's Gated DeltaNet chunk/recurrent kernels, causal / depthwise convs
    (
        "mixer",
        re.compile(r"delta|chunk|recurrent|conv1d|conv_depthwise|causal_conv|short_conv", re.IGNORECASE),
    ),
    (
        "gemm",
        re.compile(r"gemm|nvjet|xmma|cutlass|cublas|matmul|_mm_|bmm|wgmma|s16816|grouped_mm", re.IGNORECASE),
    ),
    # MoE routing and data movement between experts
    (
        "routing",
        re.compile(r"topk|sort|gather|scatter|index|histc|bincount|moe|align|permute|cumsum", re.IGNORECASE),
    ),
    ("norm", re.compile(r"norm|reduce|softmax|mean|rms", re.IGNORECASE)),
    ("elementwise", re.compile(r"elementwise|vectorized|copy|cat|fill|pow|silu|gelu|mul|add", re.IGNORECASE)),
)
OTHER = "other"


def classify(name: str) -> str:
    """The class of a kernel, from its name (see CLASSES); "other" when nothing matches."""
    return next((cls for cls, rx in CLASSES if rx.search(name)), OTHER)


CLOCK_CONTROLS = ("base", "none")  # ncu --clock-control values that profile (ncu's "reset" just exits)
META_CLOCK_CONTROL = "ncu_clock_control"  # meta field for the resolved value (written by `s1 ncu`)
BASE_ARGS = (
    "--profile-from-start",
    "off",
    "--target-processes",
    "all",
    "--csv",
    "--page",
    "raw",
    "--metrics",
    ",".join(METRICS),
)


def clock_control(explicit: str | None = None) -> str:
    """The `ncu_clock_control` to use: `explicit` if set, else "none" on Modal, else "base" (the default)."""
    value = explicit or ("none" if env.orchestrator() == "modal" else "base")
    if value not in CLOCK_CONTROLS:
        raise ValueError(f"ncu_clock_control must be one of {CLOCK_CONTROLS}, not {value!r}")
    return value


def args(clock: str | None = None) -> tuple[str, ...]:
    """ncu's arguments for `s1 ncu`. "base" adds no flag: the cluster's command line is unchanged."""
    clock = clock_control(clock)
    return BASE_ARGS if clock == "base" else ("--clock-control", clock, *BASE_ARGS)


ARGS = args()  # resolved for this process's orchestrator (with "none" inside a Modal container)


def parse_named(text: str) -> list[tuple[str, dict[str, float]]]:
    """(kernel name, {metric: value}) per profiled kernel from ncu's raw CSV page, in launch order (log lines
    before it are skipped, as is the units row). Missing or non-numeric values read as 0."""
    start = text.find('"ID",')
    if start < 0:
        raise ValueError("no ncu CSV in the log (did the profiler range run?)")
    rows = list(csv.reader(io.StringIO(text[start:])))
    header, kernels = rows[0], [r for r in rows[2:] if len(r) == len(rows[0])]
    out = []
    for r in kernels:
        values = {}
        kernel = ""
        for name, val in zip(header, r, strict=True):
            if name == NAME:
                kernel = val
            elif name in METRICS:
                try:
                    values[name] = float(val.replace(",", ""))
                except ValueError:
                    values[name] = 0.0
        out.append((kernel, values))
    return out


def parse(text: str) -> list[dict[str, float]]:
    """One {metric: value} per profiled kernel (`parse_named` without the names)."""
    return [values for _, values in parse_named(text)]


def _flops(k: Mapping[str, float]) -> float:
    return sum(k.get(m, 0.0) for m in TENSOR) + sum(k.get(m, 0.0) * w for m, w in CORE.items())


def _dram(k: Mapping[str, float]) -> float:
    return sum(k.get(m, 0.0) for m in DRAM)


def breakdown(kernels: Sequence[tuple[str, Mapping[str, float]]], prefix: str = "roofline_hw_") -> Metrics:
    """Per kernel class (see `classify`): share of kernel time, kernel ms, DRAM GB/s, TFLOP/s and the
    time-weighted speed-of-light percentages. Rates use the class's own kernel time (ncu's clocks)."""
    total_ns = sum(k.get(TIME, 0.0) for _, k in kernels)
    groups: dict[str, list[Mapping[str, float]]] = {}
    for name, k in kernels:
        groups.setdefault(classify(name), []).append(k)
    m: Metrics = {}
    for cls in [c for c, _ in CLASSES] + [OTHER]:
        if cls not in groups:
            continue
        ks = groups[cls]
        ns = sum(k.get(TIME, 0.0) for k in ks)
        p = f"{prefix}{cls}_"
        m[f"{p}share"] = ns / total_ns if total_ns > 0 else float("nan")
        m[f"{p}ms"] = ns / 1e6
        m[f"{p}kernels"] = float(len(ks))
        m[f"{p}dram_gbs"] = sum(_dram(k) for k in ks) / ns if ns > 0 else float("nan")  # bytes/ns = GB/s
        m[f"{p}tflops"] = sum(_flops(k) for k in ks) / ns / 1e3 if ns > 0 else float("nan")
        for key, metric in (("dram_pct", DRAM_PCT), ("sm_pct", SM_PCT)):
            if ns > 0 and any(metric in k for k in ks):
                m[f"{p}{key}"] = sum(k.get(metric, 0.0) * k.get(TIME, 0.0) for k in ks) / ns
    return m


def split_markers(
    kernels: Sequence[tuple[str, Mapping[str, float]]],
) -> tuple[list[tuple[str, Mapping[str, float]]], list[list[tuple[str, Mapping[str, float]]]]]:
    """(every non-marker kernel, the kernels after each marker kernel). No markers: no segments."""
    everything: list[tuple[str, Mapping[str, float]]] = []
    segments: list[list[tuple[str, Mapping[str, float]]]] = []
    for name, k in kernels:
        if MARKER.search(name):
            segments.append([])
            continue
        everything.append((name, k))
        if segments:
            segments[-1].append((name, k))
    return everything, segments


def summarize(
    kernels: Sequence[Mapping[str, float]], run: Mapping[str, float], clock: str | None = None
) -> Metrics:
    """Totals over the profiled kernels, against the run's own `roofline_hw_*` numbers for the same work.
    `clock` is the clock control the kernels ran under (default: this process's, as in `ARGS`)."""
    tensor = sum(k.get(m, 0.0) for k in kernels for m in TENSOR)
    core = sum(k.get(m, 0.0) * w for k in kernels for m, w in CORE.items())
    dram = sum(k.get(m, 0.0) for k in kernels for m in DRAM)
    flops = tensor + core
    theory_flops = run.get("roofline_hw_theory_gflop", 0.0) * 1e9
    theory_bytes = run.get("roofline_hw_theory_mb", 0.0) * 1e6
    timed_ms = run.get("roofline_hw_timed_ms", 0.0)

    def ratio(a: float, b: float) -> float:
        return a / b if b > 0 else float("nan")

    return {
        "roofline_hw_tflops": ratio(flops, timed_ms * 1e9),
        "roofline_hw_intensity": ratio(flops, dram),
        "roofline_hw_gflop": flops / 1e9,
        "roofline_hw_dram_mb": dram / 1e6,
        "roofline_hw_tensor_share": ratio(tensor, flops),
        "roofline_hw_flop_ratio": ratio(flops, theory_flops),
        "roofline_hw_byte_ratio": ratio(dram, theory_bytes),
        "roofline_hw_kernels": float(len(kernels)),
        "roofline_hw_kernel_ms": sum(k.get(TIME, 0.0) for k in kernels) / 1e6,
        "roofline_hw_clocks_locked": 1.0 if clock_control(clock) == "base" else 0.0,
    }


def summarize_named(
    kernels: Sequence[tuple[str, Mapping[str, float]]], run: Mapping[str, float], clock: str | None = None
) -> Metrics:
    """`summarize` over every kernel except the length markers, plus `roofline_hw_busy` and the per-class
    `breakdown`; on the length axis (marker kernels in the log and `roofline_hw_s<L>_timed_ms` in `run`) the
    same again per state length as `roofline_hw_s<L>_*`, the i-th marker's kernels going to the i-th shortest
    length. A marker count that does not match the lengths raises: the kernels could not be attributed."""
    everything, segments = split_markers(kernels)
    m = summarize([k for _, k in everything], run, clock)
    m["roofline_hw_busy"] = _busy(m, run.get("roofline_hw_timed_ms", 0.0))
    m |= breakdown(everything)
    lengths = sorted(int(x.group(1)) for key in run if (x := LENGTH_KEY.match(key)))
    if not segments and not lengths:
        return m
    if len(segments) != len(lengths):
        raise ValueError(
            f"{len(segments)} length markers in the ncu log but {len(lengths)} lengths in the run"
        )
    for n, seg in zip(lengths, segments, strict=True):
        p = f"roofline_hw_s{n}_"
        view = {key.replace(p, "roofline_hw_"): val for key, val in run.items() if key.startswith(p)}
        part = summarize([k for _, k in seg], view, clock)
        part.pop("roofline_hw_clocks_locked")
        part["roofline_hw_busy"] = _busy(part, view.get("roofline_hw_timed_ms", 0.0))
        m |= {key.replace("roofline_hw_", p, 1): val for key, val in part.items()}
        m |= breakdown(seg, p)
    return m


def _busy(m: Mapping[str, float], timed_ms: float) -> float:
    return m["roofline_hw_kernel_ms"] / timed_ms if timed_ms > 0 else float("nan")
