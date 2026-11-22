"""batch: batch-size sweep and CPU share of latency (hypothesis H2: small models are launch-bound at batch 1).

B independent requests, each with its own state and one question, run as one batched forward. For every
(state length L, batch size B) point of the grid

    [grid]
    batch_state_tokens = [128, 1024]                 # reference tokens per state, as roofline's length axis
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

the benchmark takes a pool of `batch_pool` seeded JEVal rows whose own state is at least max(L) reference
tokens (`roofline.length_rows`: the same rows at every L, each state cut to its first L tokens), forms
n = max(batch_min_batches, ceil(batch_min_decisions / B)) batches of B rows (batch k takes rows kB.. kB+B-1 of
the pool, wrapping), runs them all once untimed (warm-up: CUDA-graph captures, compile, allocator), then
again timed. Per point, `batch_s<L>_b<B>_*`:

    wall_ms, wall_ms_median   host wall time per batch (tokenize, forward, copy the scores back)
    ms_per_decision           wall_ms / B
    decisions_per_s
    gpu_ms                    forwards' CUDA-event span per batch (start to end on the stream: includes the
                              GPU's idle gaps while it waits for the host to launch the next kernel)
    gpu_busy                  gpu_ms / wall_ms
    tflops                    cost-model FLOPs / gpu_ms (s1.flops, as roofline_tflops)
    tflops_wall               cost-model FLOPs / wall_ms (end-to-end achieved FLOP/s; H2's test)
    intensity, gbs, mb, gflop cost-model FLOP/byte, bytes / gpu_ms, MB and GFLOP per batch
    tokens, padding           tokens charged per batch, share of them that were padding
    kernel_ms, kernels        summed GPU kernel time and kernels per batch    (torch.profiler, `batch_trace`)
    launch_api_ms             host time in kernel/graph launch calls per batch
    kernel_busy, cpu_share    kernel_ms / wall_ms and 1 - that: how much of the latency the GPU computes
    kernels_per_forward
    peak_mb                   peak GPU memory allocated over the point (torch.cuda, when CUDA is up)
    oom                       1 when the point ran out of GPU memory (its other keys are then absent)

The kernel trace runs after the timed pass on `batch_trace` batches; its wall time is never used (tracing
slows the host). Headline keys (first): the batch-1 CPU share at the shortest length and the FLOP/s gain from
B = 1 to the largest B that ran there. `batch_b1_*` repeat the shortest length's batch-1 point.

`batch_nsys = N` (set by `s1 nsys`, see s1/nsys.py) replaces the sweep: N batch-1 decisions at the shortest
length (or at `batch_nsys_state_tokens`, one of the grid's lengths) run inside one CUDA profiler range, after
`batch_warmup` untimed ones, and the region's wall time is
recorded as batch_nsys_{n,wall_ms}; `s1 nsys` adds the Nsight Systems CPU/GPU split to the record.

The model implements, besides roofline's `profile_info` / `take_forwards`:
    decide_batch(requests: [(state, {key: question})]) -> {"decisions": int, "real_tokens": int}
        one batched pass over every request's questions, end to end (scores on the host when it returns)
`batch_host_prep = "pretokenized"` (default "inline") times only the GPU side of a batch: the model must also
implement
    prepare_batch(requests) -> prepared      host work: prompt template, tokenize, pad, CPU tensors
    run_batch(prepared) -> {"decisions": int, "real_tokens": int}   to the device, forward, scores on the host
and decide_batch(r) must equal run_batch(prepare_batch(r)). Every batch of a point is prepared before its
warm-up, outside the timed and traced passes (a server tokenizes in another process, overlapped with the
GPU), and `batch_s<L>_b<B>_prep_ms` records the host preparation time per batch that wall_ms then leaves out.
`batch_host_prep = "overlapped"` is what a serving loop sees: batch i+1 is prepared (and its host tensors
pinned, if the model has `pin_batch(prepared)`) on one CPU thread while the GPU runs batch i, and prep stays
inside the timing: each batch's wall_ms includes any wait for its own preparation to finish.
`batch_s<L>_b<B>_prep_wait_ms` records that mean exposed wait per batch (0 when prep hides fully behind the
forward). The prep thread shares the GIL with the forward's Python launches, so overlap is not free.

`batch_sizes = ["steady"]` (or `["steady-bracket"]`) in [grid] makes the runner pick the steady-state batch
per length for the GPU the run is on (src/s1/steady_state.py) and set `batch_sizes_at = "128:32,1024:16"`;
the benchmark then runs only that size at each length (or half, it and double for the bracket). If the steady
pick runs out of memory, the benchmark halves it until a batch fits and records the batch it used as
`batch_s<L>_steady_fallback` (0 if even B = 1 did not fit), next to the pick's `_oom` = 1.

The model may implement `after_warmup()` (called after each point's warm-up; kev captures pending CUDA
graphs) and `profiler(on)` (needed for batch_nsys).

Settings (all optional): batch_pool (default max batch size), batch_min_decisions 32, batch_min_batches 4,
batch_seed 0, batch_max_options 16, batch_max_question_tokens 512, batch_tokenizer_file (a tokenizer.json;
words if unset), batch_trace 2 (batches traced per point; 0 = no trace), batch_nsys 0, batch_warmup 8,
batch_min_state_tokens 0 (pool rows need a state at least this long, and at least max(L): configs with
different longest lengths then share rows).
batch_nsys_warmup_offset 0 (the nsys warm-up answers pool rows offset, offset+1, ...; with the pool at least
offset + batch_nsys, the profiled rows 0..n-1 were never seen, so every question shape is new to the profiled
region while the model, allocator and kernels are warm).
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

from .. import env, flops, nsys, steady_state
from ..record import Metrics
from ..registry import Settings, benchmark, get_int, get_str, note
from . import jeval, roofline
from .shape_sweep import FileTokenCounter, WordCounter, parse_int_list

NAME = "batch"
SIZES = "batch_sizes"
LENGTHS = "batch_state_tokens"

Request = tuple[str, Mapping[str, object]]
Call = Callable[[], Mapping[str, int]]
HOST_PREP = ("inline", "pretokenized", "overlapped")


class Batched(Protocol):
    def profile_info(self) -> Mapping[str, object]: ...
    def take_forwards(self) -> list[Mapping[str, object]]: ...
    def decide_batch(self, requests: Sequence[Request]) -> Mapping[str, int]: ...


@dataclass(frozen=True)
class Timed:
    wall_ms: float
    cost: flops.Cost
    forwards: int
    real_tokens: int
    decisions: int


@dataclass(frozen=True)
class Trace:
    kernel_us: float
    kernels: int
    launch_us: float
    sync_us: float


def batches(pool: Sequence[jeval.Row], size: int, settings: Settings) -> list[list[jeval.Row]]:
    """n = max(batch_min_batches, ceil(batch_min_decisions / size)) batches of `size` rows; batch k takes pool
    rows k*size .. k*size + size - 1, wrapping, so no batch repeats a row while size <= len(pool)."""
    if size > len(pool):
        raise ValueError(f"batch size {size} is over the pool of {len(pool)} rows (batch_pool)")
    n = max(
        get_int(settings, "batch_min_batches", 4),
        math.ceil(get_int(settings, "batch_min_decisions", 32) / size),
    )
    return [[pool[(k * size + j) % len(pool)] for j in range(size)] for k in range(n)]


def requests(rows: Sequence[jeval.Row]) -> list[Request]:
    return [(r.state, r.questions) for r in rows]


def trace_split(events: Sequence[tuple[str, str, float]]) -> Trace:
    """Kernel time and count, and host time in launch and synchronize calls, of a profiler trace."""
    kernels = [us for _, kind, us in events if kind == "kernel"]
    api = [(name, us) for name, kind, us in events if kind == "api"]
    return Trace(
        kernel_us=sum(kernels),
        kernels=len(kernels),
        launch_us=sum(us for name, us in api if nsys.is_launch(name)),
        sync_us=sum(us for name, us in api if nsys.is_sync(name)),
    )


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else math.nan


def point(prefix: str, timed: Sequence[Timed], size: int, trace: Trace | None, traced: int) -> Metrics:
    """The metrics of one (length, batch size) point; `trace` covers `traced` batches."""
    n = len(timed)
    cost = flops.ZERO
    for t in timed:
        cost = cost + t.cost
    wall = sum(t.wall_ms for t in timed) / n
    gpu = cost.ms / n
    m: Metrics = {
        f"{prefix}wall_ms": wall,
        f"{prefix}wall_ms_median": statistics.median(t.wall_ms for t in timed),
        f"{prefix}ms_per_decision": wall / size,
        f"{prefix}decisions_per_s": _ratio(
            sum(t.decisions for t in timed), sum(t.wall_ms for t in timed) / 1e3
        ),
        f"{prefix}gpu_ms": gpu,
        f"{prefix}gpu_busy": _ratio(gpu, wall),
        f"{prefix}tflops": _ratio(cost.flops, cost.ms * 1e9),
        f"{prefix}tflops_wall": _ratio(cost.flops, sum(t.wall_ms for t in timed) * 1e9),
        f"{prefix}intensity": _ratio(cost.flops, cost.bytes),
        f"{prefix}gbs": _ratio(cost.bytes, cost.ms * 1e6),
        f"{prefix}mb": cost.bytes / 1e6 / n,
        f"{prefix}gflop": cost.flops / 1e9 / n,
        f"{prefix}tokens": cost.tokens / n,
        f"{prefix}padding": 1 - _ratio(sum(t.real_tokens for t in timed), cost.tokens),
        f"{prefix}forwards": sum(t.forwards for t in timed) / n,
        f"{prefix}batches": float(n),
    }
    if trace is not None and traced:
        kernel_ms = trace.kernel_us / 1e3 / traced
        m |= {
            f"{prefix}kernel_ms": kernel_ms,
            f"{prefix}kernels": trace.kernels / traced,
            f"{prefix}launch_api_ms": trace.launch_us / 1e3 / traced,
            f"{prefix}sync_api_ms": trace.sync_us / 1e3 / traced,
            f"{prefix}kernel_busy": _ratio(kernel_ms, wall),
            f"{prefix}cpu_share": 1 - _ratio(kernel_ms, wall),
            f"{prefix}kernels_per_forward": _ratio(trace.kernels / traced, m[f"{prefix}forwards"]),
        }
    return m


def headline(m: Metrics, lengths: Sequence[int], sizes: Sequence[int]) -> Metrics:
    """batch_b1_cpu_share and batch_tflops_gain (tflops_wall at the largest B that ran / at B = 1) at the
    shortest length, then batch_b1_* copies of that point, first in the dict."""
    short, b1 = f"batch_s{min(lengths)}_", f"batch_s{min(lengths)}_b{min(sizes)}_"
    ran = [b for b in sorted(sizes) if f"{short}b{b}_tflops_wall" in m]
    out: Metrics = {
        "batch_b1_cpu_share": m.get(f"{b1}cpu_share", math.nan),
        "batch_tflops_gain": (
            _ratio(m[f"{short}b{ran[-1]}_tflops_wall"], m[f"{short}b{ran[0]}_tflops_wall"])
            if ran
            else math.nan
        ),
        "batch_max_size_ran": float(ran[-1]) if ran else math.nan,
    }
    out |= {"batch_b1_" + k.removeprefix(b1): v for k, v in m.items() if k.startswith(b1)}
    return out


def _counter(settings: Settings) -> roofline.FittingCounter:
    path = get_str(settings, "batch_tokenizer_file", "")
    return FileTokenCounter(str(env.resolve_path(path))) if path else WordCounter()


def pools(settings: Settings, lengths: Sequence[int], size: int) -> dict[int, list[jeval.Row]]:
    """{length: the pool's rows cut to it}: roofline.length_rows with this benchmark's settings."""
    keys: Settings = {
        "roofline_per_length": get_int(settings, "batch_pool", size),
        "roofline_seed": get_int(settings, "batch_seed", 0),
        "roofline_max_options": get_int(settings, "batch_max_options", 16),
        "roofline_max_question_tokens": get_int(settings, "batch_max_question_tokens", 512),
        # configs whose longest length differs still draw the same rows (roofline_min_state_tokens)
        "roofline_min_state_tokens": get_int(settings, "batch_min_state_tokens", 0),
    }
    out: dict[int, list[jeval.Row]] = {n: [] for n in lengths}
    for n, row in roofline.length_rows(jeval.load_rows(), _counter(settings), keys, lengths):
        out[n].append(row)
    return out


def _model(model: object) -> Batched:
    for attr in ("decide_batch", "profile_info", "take_forwards"):
        if not callable(getattr(model, attr, None)):
            raise TypeError(f"batch needs a model with .{attr}(); got {type(model).__name__}")
    return model  # ty: ignore[invalid-return-type]


@dataclass
class Overlap:
    """Batch i+1 prepared on a CPU thread while batch i runs (batch_host_prep = overlapped). Calls may run in
    any order (warm-up, timed, traced): a call whose batch was not prefetched prepares it on the thread and
    waits; every call prefetches the next batch of the plan (the last one, the first)."""

    reqs: list[list[Request]]
    prepare: Callable[[Sequence[Request]], object]
    run: Callable[[object], Mapping[str, int]]
    pin: Callable[[object], object] | None = None
    pool: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(1, "batch-prep"))
    pending: dict[int, Future[object]] = field(default_factory=dict)
    waits_ms: list[float] = field(default_factory=list)

    def _prep(self, i: int) -> object:
        p = self.prepare(self.reqs[i])
        return self.pin(p) if self.pin is not None else p

    def _submit(self, i: int) -> Future[object]:
        if i not in self.pending:
            self.pending[i] = self.pool.submit(self._prep, i)
        return self.pending[i]

    def call(self, i: int) -> Mapping[str, int]:
        t0 = time.perf_counter()
        prepared = self._submit(i).result()
        self.waits_ms.append((time.perf_counter() - t0) * 1e3)
        del self.pending[i]
        self._submit((i + 1) % len(self.reqs))  # wraps, so the next pass's first batch is ready too
        return self.run(prepared)

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()


def _split(m: Batched, host_prep: str) -> tuple[Callable[..., object], Callable[..., Mapping[str, int]]]:
    prepare, run_batch = getattr(m, "prepare_batch", None), getattr(m, "run_batch", None)
    if not callable(prepare) or not callable(run_batch):
        raise TypeError(
            f"batch_host_prep = {host_prep} needs .prepare_batch() and .run_batch(); got {type(m).__name__}"
        )
    return prepare, run_batch


def plan_calls(
    m: Batched, plan: Sequence[Sequence[jeval.Row]], host_prep: str
) -> tuple[list[Call], float, Overlap | None]:
    """One call per batch of `plan`, the mean host preparation ms per batch (nan unless pretokenized: inline
    and overlapped prepare inside each call) and the overlap pipeline (overlapped only)."""
    if host_prep == "inline":
        return [lambda b=b: m.decide_batch(requests(b)) for b in plan], math.nan, None
    prepare, run_batch = _split(m, host_prep)
    if host_prep == "overlapped":
        pin = getattr(m, "pin_batch", None)
        ov = Overlap([requests(b) for b in plan], prepare, run_batch, pin if callable(pin) else None)
        return [lambda i=i: ov.call(i) for i in range(len(plan))], math.nan, ov
    t0 = time.perf_counter()
    prepared = [prepare(requests(b)) for b in plan]
    prep_ms = (time.perf_counter() - t0) * 1e3 / len(plan)
    return [lambda p=p: run_batch(p) for p in prepared], prep_ms, None


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def _reset_peak() -> bool:
    """Start a point's peak-memory window; False (no peak_mb) without CUDA, e.g. under test fakes."""
    import sys  # noqa: PLC0415

    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return False
    torch.cuda.reset_peak_memory_stats()
    return True


def _peak_mb() -> float:
    import sys  # noqa: PLC0415

    return float(sys.modules["torch"].cuda.max_memory_allocated()) / 1e6


def _free_cuda() -> None:
    import gc  # noqa: PLC0415
    import sys  # noqa: PLC0415

    gc.collect()
    if "torch" in sys.modules:
        sys.modules["torch"].cuda.empty_cache()


@dataclass
class _Sweep:
    """What every (length, batch size) point of one sweep shares."""

    m: Batched
    pool: Mapping[int, Sequence[jeval.Row]]
    settings: Settings
    host_prep: str
    arch: flops.Arch
    sizes_kw: Mapping[str, int]
    traced_n: int

    def point(self, out: Metrics, length: int, size: int) -> bool:
        """Measure one point into `out`; False (and `_oom` = 1) when it ran out of GPU memory."""
        m, prefix = self.m, f"batch_s{length}_b{size}_"
        plan = batches(self.pool[length], size, self.settings)
        after_warmup = getattr(m, "after_warmup", None)
        peak = _reset_peak()
        try:
            timed, trace, prep_ms, waits = measure_point(
                m, plan, self.host_prep, after_warmup=after_warmup, traced_n=self.traced_n, arch=self.arch,
                sizes_kw=self.sizes_kw,
            )  # fmt: skip
        except Exception as e:
            if not _is_oom(e):
                raise
            m.take_forwards()
            _free_cuda()
            out[f"{prefix}oom"] = 1.0
            note(f"{prefix}oom", f"{type(e).__name__}: {e}"[:200])
            print(f"batch: L={length} B={size} out of memory", flush=True)
            return False
        out |= point(prefix, timed, size, trace, min(self.traced_n, len(plan)))
        if peak:
            out[f"{prefix}peak_mb"] = _peak_mb()
        if not math.isnan(prep_ms):
            out[f"{prefix}prep_ms"] = prep_ms
        if waits:
            out[f"{prefix}prep_wait_ms"] = statistics.fmean(waits)
        print(
            f"batch: L={length} B={size} wall {out[f'{prefix}wall_ms']:.2f} ms, "
            f"{out[f'{prefix}tflops_wall']:.2f} TFLOP/s, "
            f"cpu_share {out.get(f'{prefix}cpu_share', math.nan):.2f}",
            flush=True,
        )
        return True


@benchmark(NAME)
def run(model: object, settings: Settings) -> Metrics:
    m = _model(model)
    info = m.profile_info()
    config = info["config"]
    if not isinstance(config, Mapping):
        raise TypeError("profile_info()['config'] must be the model's HF config dict")
    arch = flops.arch(config)
    sizes_kw = {
        "head_bytes": int(roofline._num({"head_bytes": flops.BF16_BYTES} | dict(info), "head_bytes", int)),
        "weight_bytes": int(
            roofline._num({"weight_bytes": flops.BF16_BYTES} | dict(info), "weight_bytes", int)
        ),
    }
    lengths, sizes = sorted(parse_int_list(settings, LENGTHS)), sorted(parse_int_list(settings, SIZES))
    if n_nsys := get_int(settings, "batch_nsys", 0):
        at = nsys_length(settings, lengths)
        return nsys_region(
            m, pools(settings, lengths, 1)[at], n_nsys, settings, at, arch=arch, sizes_kw=sizes_kw
        )
    host_prep = get_str(settings, "batch_host_prep", "inline")
    if host_prep not in HOST_PREP:
        raise ValueError(f"batch_host_prep must be one of {', '.join(HOST_PREP)}, got {host_prep!r}")
    sweep = _Sweep(m, pools(settings, lengths, max(sizes)), settings, host_prep, arch, sizes_kw,
                   get_int(settings, "batch_trace", 2))  # fmt: skip
    out: Metrics = {}
    for length in lengths:
        for size in steady_state.sizes_for(settings, length, sizes):
            if sweep.point(out, length, size) or not steady_state.halves_on_oom(settings):
                continue
            # the steady pick ran out of memory here: halve until a batch fits, so the run still measures
            # this length, and say which batch it used (it is no longer the table's pick)
            ran = next((b for b in steady_state.fallback_sizes(size) if sweep.point(out, length, b)), 0)
            out[f"batch_s{length}_steady_fallback"] = float(ran)
            note(f"batch_s{length}_steady_fallback", f"B={size} out of memory; ran B={ran or 'none'}")
    return (
        headline(out, lengths, sizes)
        | {"batch_lengths": float(len(lengths)), "batch_sizes": float(len(sizes))}
        | out
    )


def measure_point(
    m: Batched,
    plan: Sequence[Sequence[jeval.Row]],
    host_prep: str,
    *,
    after_warmup: object,
    traced_n: int,
    arch: flops.Arch,
    sizes_kw: Mapping[str, int],
) -> tuple[list[Timed], Trace | None, float, list[float]]:
    """One point: warm-up (every shape once, untimed), timed passes, then the trace. Returns the timed
    batches, the trace, pretokenized's prep ms (else nan) and overlapped's per-batch prep waits."""
    calls, prep_ms, overlap = plan_calls(m, plan, host_prep)
    try:
        for call in calls:
            call()
        if callable(after_warmup):
            after_warmup()
        m.take_forwards()
        if overlap is not None:
            overlap.waits_ms.clear()
        timed = [time_batch(m, call, arch, sizes_kw) for call in calls]
        waits = list(overlap.waits_ms) if overlap is not None else []
        trace = trace_batches(m, calls[:traced_n]) if traced_n else None
    finally:
        if overlap is not None:
            overlap.close()
    return timed, trace, prep_ms, waits


def time_batch(m: Batched, call: Call, arch: flops.Arch, sizes_kw: Mapping[str, int]) -> Timed:
    t0 = time.perf_counter()
    done = call()
    wall = (time.perf_counter() - t0) * 1e3
    fwd = [roofline.forward(f) for f in m.take_forwards()]
    return Timed(
        wall_ms=wall,
        cost=flops.total(arch, fwd, **sizes_kw),
        forwards=len(fwd),
        real_tokens=int(done.get("real_tokens", 0)),
        decisions=int(done["decisions"]),
    )


def trace_batches(m: Batched, calls: Sequence[Call]) -> Trace | None:
    """torch.profiler over `calls` (best effort: a trace failure leaves the point's other keys)."""
    from .. import profiling  # noqa: PLC0415 — imports torch lazily inside

    def go() -> None:
        for call in calls:
            call()

    try:
        return trace_split(profiling.trace_activity(go))
    except Exception as e:
        if _is_oom(e):
            raise
        note("batch_trace_error", f"{type(e).__name__}: {e}"[:300])
        return None
    finally:
        m.take_forwards()


def nsys_length(settings: Settings, lengths: Sequence[int]) -> int:
    """The state length `batch_nsys` profiles: `batch_nsys_state_tokens` (one of the grid's lengths, so the
    rows are the same at every length), else the shortest."""
    at = get_int(settings, "batch_nsys_state_tokens", 0) or min(lengths)
    if at not in lengths:
        raise ValueError(f"batch_nsys_state_tokens {at} is not in [grid] {LENGTHS} {sorted(lengths)}")
    return at


def nsys_region(
    m: Batched,
    pool: Sequence[jeval.Row],
    n: int,
    settings: Settings,
    length: int,
    *,
    arch: flops.Arch | None = None,
    sizes_kw: Mapping[str, int] | None = None,
) -> Metrics:
    """`batch_warmup` untimed batch-1 decisions (then the model's `after_warmup`, as the sweep does), then `n`
    inside one CUDA profiler range (cudaProfilerStart/Stop), the range Nsight Systems captures under
    `s1 nsys`. With `arch`, also the cost-model GFLOP and MB per profiled decision (batch_nsys_gflop,
    batch_nsys_mb), which `nsys.summarize` divides by kernel time."""
    profiler = getattr(m, "profiler", None)
    if not callable(profiler):
        raise TypeError(f"batch_nsys needs a model with .profiler(on); got {type(m).__name__}")
    offset = get_int(settings, "batch_nsys_warmup_offset", 0)  # > 0: warm on other rows, profile cold shapes
    for i in range(get_int(settings, "batch_warmup", 8)):
        m.decide_batch(requests([pool[(i + offset) % len(pool)]]))
    if callable(after_warmup := getattr(m, "after_warmup", None)):
        after_warmup()
    m.take_forwards()
    profiler(True)
    t0 = time.perf_counter()
    try:
        for i in range(n):
            m.decide_batch(requests([pool[i % len(pool)]]))
    finally:
        wall = (time.perf_counter() - t0) * 1e3
        profiler(False)
    forwards = m.take_forwards()
    out: Metrics = {
        "batch_nsys_n": float(n),
        "batch_nsys_wall_ms": wall,
        "batch_nsys_state_tokens": float(length),
    }
    if arch is not None and forwards and n > 0:
        cost = flops.total(arch, [roofline.forward(f) for f in forwards], **(sizes_kw or {}))
        out["batch_nsys_gflop"] = cost.flops / 1e9 / n
        out["batch_nsys_mb"] = cost.bytes / 1e6 / n
    return out
