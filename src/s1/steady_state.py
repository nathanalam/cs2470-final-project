"""The steady-state batch for a run: the batch size (or, for vLLM, the concurrency) that keeps this GPU busy.

Method (agreed 2026-11-19): every model x GPU pair is compared at its own steady state, the smallest batch
with GPU kernel time >= 95% of wall time that no larger batch beats by >= 5% in throughput
(`report_steady.steady`). This module stores the answers of the sweeps we ran, so a config can say

    [grid]
    batch_sizes = ["steady"]            # or ["steady-bracket"]: half, the pick and double, to re-check it
    [params]
    jeval_concurrency = "steady"        # vLLM-served models only

and the runner fills in the number for the GPU the run landed on (docs/steady_state_batch.md).

- **The table decides.** `TABLE` holds every measured pick with the run IDs it came from;
  tests/test_steady_state.py recomputes each one from those records. With the default
  `steady_policy = "table"`, a model x GPU x setup the table lacks fails before the model loads.
- **The formula is opt-in.** `steady_policy = "table+formula"` lets a missing pair run on an estimate,
  `tokens = T[gpu] * max(1, 8B / active params)`, divided by the request length and rounded to a power of
  two, capped by memory. Records mark such picks `predicted`.
- **Setup is part of the key.** An entry carries the settings it was measured under (`variant`, e.g.
  pretokenized host prep). They are applied to the run unless the config sets them; a config that sets them
  differently does not match that entry.

Torch-free: importable on the login node.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from .config import RunSpec
from .models.catalog import CATALOG
from .record import Scalar

RULE = 1  # bump when the table schema, the lookup or the formula changes (recorded as steady_rule)
STEADY, BRACKET = "steady", "steady-bracket"
SENTINELS = (STEADY, BRACKET)
POLICIES = ("table", "table+formula")
SIZE_AXIS, LENGTH_AXIS = "batch_sizes", "batch_state_tokens"  # the `batch` benchmark's [grid] axes
CONCURRENCY = "jeval_concurrency"
SIZES_AT = "batch_sizes_at"  # resolved per-length batch: "128:32,1024:16"
QUESTION_TOKENS = 270  # JEVal question + options on top of the reference state length (records: ~263-277)
REF_ACTIVE = 8e9  # the model size T[gpu] is quoted for
MB_PER_TOKEN = 0.3  # transient activation memory per token, conservative (L4 OOM evidence: <= ~0.25-0.3)
DEFAULT_CONCURRENCY = 8  # vLLM: steady at 8 on every GPU measured (A100 PCIe, H100, H200, B200)
NEAR = 2.0  # a run length uses a measured length's entry when within this factor of it
SERVER_BATCHED = frozenset({"vllm_shim", "kev"})  # backends that batch concurrent jeval requests
NO_BATCH = {  # catalog prefix -> why "steady" cannot apply (the backend has no decide_batch)
    "elephant-1": "elephant-1's engine has no batched path here",
    "bongard-mini": "the bongard backend has no batched path",
}
SETUP_WORD = {"batch_host_prep": "prep", "laya_mode": "mode", "semif_experts_implementation": "experts"}
VARIANT_DEFAULTS: dict[str, Scalar] = {  # what a run uses when it does not set the key
    "batch_host_prep": "inline",
    "laya_mode": "eager",
    "semif_experts_implementation": "",
}

Knob = Literal["batch", "concurrency"]
Status = Literal["reached", "plateau", "predicted"]
REACHED, PLATEAU, PREDICTED = "reached", "plateau", "predicted"


class SteadyStateError(ValueError):
    """A "steady" request this module cannot resolve (unknown pair under the table policy, bad config)."""


@dataclass(frozen=True)
class Entry:
    """One measured pick. `at` maps state length to batch size ({0: c} for a concurrency); `status` says
    per length whether kernel_busy >= 0.95 was reached there or only the throughput plateau."""

    model: str
    gpu: str  # canonical key, see gpu_key
    at: Mapping[int, int]
    status: Mapping[int, Status]
    runs: tuple[str, ...]
    variant: Mapping[str, Scalar] = field(default_factory=dict)
    knob: Knob = "batch"
    gpus: int = 1  # tensor-parallel width (vLLM)
    note: str = ""


@dataclass(frozen=True)
class Gpu:
    tokens: int  # T[gpu]: tokens per forward that keep it busy for an ~8B dense model
    mem_gb: float  # usable for weights + activations
    measured: bool  # False: a proposed constant, no batch sweep on this GPU yet


GPUS: dict[str, Gpu] = {
    "L4": Gpu(1_500, 21.5, True),
    "A10G": Gpu(1_500, 21.5, False),
    "RTX-PRO-4500": Gpu(2_000, 29.0, False),
    "A100-SXM4": Gpu(3_000, 72.0, True),
    "A100-PCIe": Gpu(3_000, 72.0, True),  # same chip as the SXM4; LFM2 swept here (#21)
    "L40S": Gpu(3_000, 43.0, False),
    "H100": Gpu(6_000, 72.0, True),
    "H200": Gpu(6_000, 127.0, True),
    "B200": Gpu(12_000, 162.0, True),
}
_GPU_NAMES = (  # substring of nvidia-smi's name (upper case) -> key; order matters (L40S before L4)
    ("A100-SXM", "A100-SXM4"),
    ("A100", "A100-PCIe"),
    ("H100", "H100"),
    ("H200", "H200"),
    ("B200", "B200"),
    ("L40S", "L40S"),
    ("A10G", "A10G"),
    ("RTX PRO 4500", "RTX-PRO-4500"),
    ("L4", "L4"),
)


def gpu_key(detected: str) -> str:
    """The table's name for what `detect_gpu()` reported ("NVIDIA H100 80GB HBM3" -> "H100")."""
    names = {n.strip().upper() for n in detected.split(",") if n.strip()}
    keys = {next((k for sub, k in _GPU_NAMES if sub in n), "") for n in names}
    if len(keys) != 1 or "" in keys:
        raise SteadyStateError(f"no steady-state GPU key for {detected!r} (known: {', '.join(GPUS)})")
    return keys.pop()


# ---------------------------------------------------------------- the table (from results/*.json)

# Batch entries: the `batch` benchmark at state lengths 128 and 1024 (rows are ~L + 270 tokens), Modal hosts,
# transformers' default MoE dispatch. Status R = kernel_busy >= 0.95 reached, P = only the throughput plateau.
# Of several entries for one model x GPU, `lookup` prefers more lengths reached, then the first listed: LFM2's
# overlapped rows come first (prep runs on a CPU thread during the previous batch and stays in the wall time).
# granite4-h-tiny has no row: on the L4 image (no mamba-ssm) every B >= 2 runs out of memory (#21).
# tests/test_steady_state.py recomputes every row from its run.
_BATCH = """
# model               gpu        b128 b1024 s128 s1024 run                       setup
qwen3-8b.zeroshot     L4         4    1     R    R     20261109-231713-a38ec98c  batch_host_prep=inline
qwen35-9b.zeroshot    L4         4    1     R    R     20261119-225425-e0d0badf  batch_host_prep=inline
jevk5-9b              L4         2    1     R    R     20261119-230530-ed7206a6  batch_host_prep=inline
kev-4b.hybrid         L4         8    1     R    R     20261109-230438-4ddb0b2d  -
lfm2-8b-a1b.zeroshot  L4         2    1     P    P     20261119-224843-9162d873  batch_host_prep=inline,semif_experts_implementation=
kev-4b.qwen3          L4         1    1     P    R     20261122-174224-2b9402d0  -
olmoe-1b-7b.zeroshot  L4         2    1     P    R     20261122-182750-d8dc8419  batch_host_prep=inline,semif_experts_implementation=
laya.multilingual     L4         8    4     P    P     20261122-182531-9bfa76ff  laya_mode=eager
laya.english          L4         8    4     P    P     20261109-205307-30a4666a  laya_mode=eager
laya.english          L4         4    4     P    P     20261109-205548-b5716441  laya_mode=graphs
laya.english          L4         8    16    P    P     20261109-205821-652d33c6  laya_mode=compile
laya.english          L4         16   16    P    P     20261109-210356-5f94a8be  laya_mode=compile_graphs
qwen3-8b.zeroshot     A100-SXM4  1    1     R    R     20261109-231247-cc3371f7  batch_host_prep=inline
kev-4b.hybrid         A100-SXM4  16   2     R    P     20261109-230438-7a7d5a81  -
lfm2-8b-a1b.zeroshot  A100-SXM4  32   32    P    P     20261119-222604-2a4972b9  batch_host_prep=inline,semif_experts_implementation=
lfm2-8b-a1b.zeroshot  A100-PCIe  32   32    R    R     20261122-181032-8ea3ce5e  batch_host_prep=overlapped,semif_experts_implementation=
laya.english          A100-SXM4  32   32    P    P     20261109-230437-6351524c  laya_mode=eager
laya.english          A100-SXM4  32   32    P    P     20261109-230556-bc2c7922  laya_mode=graphs
laya.english          A100-SXM4  32   32    P    P     20261109-230701-057912aa  laya_mode=compile
laya.english          A100-SXM4  32   32    P    P     20261109-230957-e4075dcf  laya_mode=compile_graphs
qwen3-8b.zeroshot     H100       4    4     R    R     20261109-211259-b63a8b3e  batch_host_prep=inline
lfm2-8b-a1b.zeroshot  H100       32   16    R    R     20261122-175440-dc068906  batch_host_prep=overlapped,semif_experts_implementation=
lfm2-8b-a1b.zeroshot  H100       32   16    R    R     20261119-230040-acb6fcff  batch_host_prep=pretokenized,semif_experts_implementation=
lfm2-8b-a1b.zeroshot  H100       32   16    P    P     20261119-222604-5a3f7682  batch_host_prep=inline,semif_experts_implementation=
kev-4b.hybrid         H100       16   2     P    P     20261109-205016-9d8b32e6  -
laya.english          H100       64   32    P    P     20261109-230439-caedab1a  laya_mode=eager
laya.english          H100       64   16    P    P     20261109-230623-02c3b499  laya_mode=graphs
laya.english          H100       64   32    P    P     20261109-230730-559ab4d1  laya_mode=compile
laya.english          H100       64   32    P    P     20261109-231109-e21280ed  laya_mode=compile_graphs
qwen3-8b.zeroshot     H200       4    4     R    R     20261109-231248-7eb18713  batch_host_prep=inline
lfm2-8b-a1b.zeroshot  H200       32   8     R    R     20261122-181037-5b2031f6  batch_host_prep=overlapped,semif_experts_implementation=
lfm2-8b-a1b.zeroshot  H200       32   16    P    P     20261119-222601-ced5a93f  batch_host_prep=inline,semif_experts_implementation=
kev-4b.hybrid         H200       16   2     P    P     20261109-230438-b785789f  -
laya.english          H200       64   32    P    P     20261109-230438-3115efc1  laya_mode=eager
laya.english          H200       64   64    P    P     20261109-230603-94e3dd99  laya_mode=graphs
laya.english          H200       32   32    P    P     20261109-230702-5f3feeec  laya_mode=compile
laya.english          H200       32   32    P    P     20261109-230958-c55ccaa2  laya_mode=compile_graphs
qwen3-8b.zeroshot     B200       64   4     P    R     20261109-231135-fd87886c  batch_host_prep=inline
lfm2-8b-a1b.zeroshot  B200       32   16    R    R     20261122-181855-63c8017d  batch_host_prep=overlapped,semif_experts_implementation=
lfm2-8b-a1b.zeroshot  B200       32   16    P    P     20261119-222600-33e72d92  batch_host_prep=inline,semif_experts_implementation=
kev-4b.hybrid         B200       16   8     P    P     20261109-230440-c7b126b6  -
laya.english          B200       32   16    P    P     20261109-230438-6835673b  laya_mode=eager
laya.english          B200       32   16    P    P     20261109-230607-9ea70f2f  laya_mode=graphs
laya.english          B200       64   16    P    P     20261109-230700-7702ff03  laya_mode=compile
laya.english          B200       32   32    P    P     20261109-230930-a0cb6370  laya_mode=compile_graphs
"""  # noqa: E501

# OpenJev on vLLM: jeval_concurrency, 1,000 JEVal questions, steady = < 5% more q/s at the next step and
# per-GPU nvidia-smi util >= 95%. Each row lists the whole concurrency curve (1, 8, 16, 32, 64, 128).
_CONCURRENCY = """
# model   gpu        c  runs
openjev   A100-PCIe  8  20261119-222616-4119a7fc,20261119-224354-d73f77fb,20261119-225417-0776681c,20261119-230436-6b219825,20261119-231507-a273cec1,20261119-232521-b90e1912
openjev   H100       8  20261119-222601-d16bbfd0,20261119-223636-995400ee,20261119-224148-f9dd29ed,20261119-224650-0b61edbe,20261119-225152-e2af36be,20261119-225651-791ecf60
openjev   H200       8  20261119-222601-cb08832a,20261119-223601-41b4cce3,20261119-224046-b2cabe59,20261119-224521-fe4e5a6c,20261119-224954-47690596,20261119-225427-4bc2fe3b
openjev   B200       8  20261119-222602-58b3adbb,20261119-223618-6c6bc83f,20261119-223912-33ef5899,20261119-224205-4acfddf6,20261119-224458-a001d3cf,20261119-224749-e9aa1791
"""  # noqa: E501

NOTES = {  # (model, gpu) -> what a reader of a plateau entry should know
    ("lfm2-8b-a1b.zeroshot", "L4"): "max kernel_busy 0.94",
    ("lfm2-8b-a1b.zeroshot", "A100-SXM4"): "max kernel_busy 0.93 / 0.91; no grouped_mm on sm80",
    ("lfm2-8b-a1b.zeroshot", "B200"): "max kernel_busy 0.92 / 0.94 with inline prep",
    ("kev-4b.hybrid", "H100"): "kernel_busy 0.23-0.32 at 1024 tokens; batching does not fix it",
    ("kev-4b.hybrid", "H200"): "kernel_busy 0.23-0.32 at 1024 tokens; batching does not fix it",
    ("kev-4b.hybrid", "B200"): "kernel_busy 0.23-0.32 at 1024 tokens; batching does not fix it",
}
_STATUS: dict[str, Status] = {"R": REACHED, "P": PLATEAU}


def _rows(text: str) -> list[list[str]]:
    return [line.split() for line in text.strip().splitlines() if not line.startswith("#")]


def _setup(field: str) -> dict[str, Scalar]:
    out: dict[str, Scalar] = {}
    for kv in [] if field == "-" else field.split(","):
        k, _, v = kv.partition("=")
        out[k] = v
    return out


def _batch_entry(row: list[str]) -> Entry:
    m, g, b1, b2, s1, s2, run, setup = row
    at = {128: int(b1), 1024: int(b2)}
    return Entry(
        m, g, at, {128: _STATUS[s1], 1024: _STATUS[s2]}, (run,), _setup(setup), note=NOTES.get((m, g), "")
    )


def _concurrency_entry(row: list[str]) -> Entry:
    m, g, c, runs = row
    return Entry(m, g, {0: int(c)}, {0: REACHED}, tuple(runs.split(",")), knob="concurrency")


def _table() -> tuple[Entry, ...]:
    return (*map(_batch_entry, _rows(_BATCH)), *map(_concurrency_entry, _rows(_CONCURRENCY)))


TABLE: tuple[Entry, ...] = _table()


# ---------------------------------------------------------------- lookup and formula


def _matches(e: Entry, settings: Mapping[str, Scalar]) -> bool:
    """Every setting the entry was measured under is unset in the run or set the same way."""
    return all(k not in settings or str(settings[k]) == str(v) for k, v in e.variant.items())


def lookup(model: str, gpu: str, settings: Mapping[str, Scalar], *, knob: Knob = "batch") -> Entry | None:
    """The table entry for this model on this GPU (canonical key) whose setup the run's settings allow:
    the one with the most lengths reached, then the first listed. None when there is none."""
    gpus = int(settings.get("tensor_parallel_size", 1)) if knob == "concurrency" else 1
    hits = [
        e for e in TABLE
        if e.model == model and e.gpu == gpu and e.knob == knob and e.gpus == gpus and _matches(e, settings)
    ]  # fmt: skip
    if not hits:
        return None
    return max(hits, key=lambda e: (sum(s == REACHED for s in e.status.values()), -TABLE.index(e)))


def nearest(e: Entry, length: int) -> int | None:
    """The measured length whose entry a run length uses (within a factor NEAR of it), or None."""
    near = [m for m in e.at if m > 0 and max(m, length) / min(m, length) <= NEAR]
    return min(near, key=lambda m: abs(math.log(m / length))) if near else None


def pow2_round(x: float) -> int:
    return 1 << round(math.log2(max(1.0, x)))


def mem_cap(gpu: str, weights_gb: float, row_tokens: float, mb_per_token: float = MB_PER_TOKEN) -> int:
    """The largest power-of-two batch whose activations fit next to the weights (at least 1)."""
    free_mb = (GPUS[gpu].mem_gb - weights_gb) * 1000
    fit = free_mb / (row_tokens * mb_per_token)
    return 1 << int(math.log2(fit)) if fit >= 2 else 1


def formula(gpu: str, length: int, *, active_params: float, weights_gb: float) -> int:
    """The opt-in estimate: tokens = T[gpu] * max(1, 8B / active params), over row tokens, power of two."""
    tokens = GPUS[gpu].tokens * max(1.0, REF_ACTIVE / active_params)
    row = length + QUESTION_TOKENS
    return max(1, min(pow2_round(tokens / row), mem_cap(gpu, weights_gb, row)))


def bracket(b: int) -> list[int]:
    """Half the pick, the pick and double it: a cheap re-check that the pick is still the steady point."""
    return sorted({max(1, b // 2), b, 2 * b})


def parse_sizes_at(raw: str) -> dict[int, int]:
    """ "128:32,1024:16" -> {128: 32, 1024: 16}."""
    out: dict[int, int] = {}
    for part in raw.split(","):
        length, _, b = part.partition(":")
        out[int(length)] = int(b)
    return out


def format_sizes_at(at: Mapping[int, int]) -> str:
    return ",".join(f"{n}:{b}" for n, b in sorted(at.items()))


# ---------------------------------------------------------------- resolving a run


@dataclass
class Resolution:
    """What `resolve` changes: params to set (merged over the run's) and grid axes to replace."""

    params: dict[str, Scalar] = field(default_factory=dict)
    grid: dict[str, list[Scalar]] = field(default_factory=dict)
    pending: bool = False  # a knob still says "steady" and needs the loaded model (formula)

    def apply(self, params: dict[str, Scalar], grid: dict[str, list[Scalar]]) -> None:
        for k, v in self.params.items():
            params[k] = v
        grid.update(self.grid)


def requested(params: Mapping[str, Scalar], grid: Mapping[str, Sequence[Scalar]]) -> tuple[str | None, bool]:
    """(the batch sentinel or None, whether jeval_concurrency is "steady"); errors on a malformed request."""
    sizes = list(grid.get(SIZE_AXIS, []))
    batch: str | None = None
    if any(v in SENTINELS for v in sizes):
        if len(sizes) != 1:
            raise SteadyStateError(
                f'[grid] {SIZE_AXIS} = {sizes}: use ["steady"] alone, not mixed with sizes'
            )
        batch = str(sizes[0])
    conc = params.get(CONCURRENCY)
    if isinstance(conc, str) and conc != STEADY:
        raise SteadyStateError(f'{CONCURRENCY} must be an integer or "steady", got {conc!r}')
    return batch, conc == STEADY


def _spec_of(model: str) -> tuple[str | None, dict[str, Scalar]]:
    spec = next((s for s in CATALOG if s.name == model), None)
    return (spec.backend, spec.settings()) if spec is not None else (None, {})


def _policy(settings: Mapping[str, Scalar]) -> str:
    policy = str(settings.get("steady_policy", "table"))
    if policy not in POLICIES:
        raise SteadyStateError(f"steady_policy must be one of {', '.join(POLICIES)}, got {policy!r}")
    return policy


def _missing(what: str, model: str, gpu: str, settings: Mapping[str, Scalar]) -> SteadyStateError:
    have_e = [e for e in TABLE if e.model == model]
    near = sorted({f"{e.gpu} {format_sizes_at(e.at)} {dict(e.variant) or ''}".strip() for e in have_e})
    have = "; table has " + " | ".join(near) if near else "; the table has no entry for this model"
    setup = {k: settings[k] for k in VARIANT_DEFAULTS if k in settings}
    return SteadyStateError(
        f"no measured steady-state {what} for {model} on {gpu} with {setup or 'default setup'}{have}. "
        "Run a batch sweep for it (see docs/steady_state_batch.md), set the size explicitly, or opt into the "
        'estimate with steady_policy = "table+formula".'
    )


def _active_params(model_obj: object) -> float:
    from . import flops  # noqa: PLC0415 (flops is torch-free; local import keeps this module light)

    info = getattr(model_obj, "profile_info", None)
    cfg = info().get("config") if callable(info) else None
    if not isinstance(cfg, Mapping):
        raise SteadyStateError("steady_policy formula needs the model's HF config (profile_info()['config'])")
    return float(flops.arch(cfg).active_params)


@dataclass
class _Run:
    """One resolution in progress: who and where, and what has been picked so far."""

    model: str
    backend: str | None
    spec_settings: dict[str, Scalar]
    settings: dict[str, Scalar]
    policy: str
    gpu: str
    model_obj: object | None
    out: Resolution
    runs: list[str] = field(default_factory=list)
    statuses: dict[int, str] = field(default_factory=dict)
    sources: set[str] = field(default_factory=set)


def _concurrency(r: _Run) -> None:
    if r.backend not in SERVER_BATCHED:
        raise SteadyStateError(
            f'{CONCURRENCY} = "steady" needs a server-batched backend ({", ".join(sorted(SERVER_BATCHED))}); '
            f"{r.model} ({r.backend}) serializes forwards, so concurrency does not batch it"
        )
    e = lookup(r.model, r.gpu, r.settings, knob="concurrency")
    if e is not None:
        r.out.params[CONCURRENCY] = e.at[0]
        r.statuses[0] = e.status[0]
        r.sources.add("table")
        r.runs += e.runs
    elif r.policy == "table":
        raise _missing("concurrency", r.model, r.gpu, r.settings)
    else:
        r.out.params[CONCURRENCY] = DEFAULT_CONCURRENCY
        r.statuses[0] = PREDICTED
        r.sources.add("formula")


def _batch(r: _Run, sentinel: str, lengths: Sequence[int]) -> bool:
    """Resolve the batch per length into r.out; False when the formula is needed but the model is not
    loaded yet."""
    why = next((v for p, v in NO_BATCH.items() if r.model.startswith(p)), None)
    if why is not None:
        raise SteadyStateError(f"{r.model} has no steady-state batch: {why}")
    if not lengths:
        raise SteadyStateError(f'[grid] {SIZE_AXIS} = ["steady"] needs [grid] {LENGTH_AXIS}')
    e = lookup(r.model, r.gpu, r.settings)
    at: dict[int, int] = {}
    for n in lengths:
        m = nearest(e, n) if e is not None else None
        if e is not None and m is not None:
            at[n], r.statuses[n] = e.at[m], e.status[m]
            r.sources.add("table")
        elif r.policy == "table":
            raise _missing("batch", r.model, r.gpu, r.settings)
        elif r.model_obj is None:
            return False
        else:
            weights = float(r.spec_settings.get("spec_weights_gb", 0.0) or 0.0)
            at[n] = formula(r.gpu, n, active_params=_active_params(r.model_obj), weights_gb=weights)
            r.statuses[n] = PREDICTED
            r.sources.add("formula")
    if e is not None and "table" in r.sources:
        r.runs += e.runs
        for k, v in e.variant.items():  # run the setup the entry was measured under
            if k not in r.settings and v != VARIANT_DEFAULTS.get(k):
                r.out.params[k] = v
    sizes = sorted({s for b in at.values() for s in (bracket(b) if sentinel == BRACKET else [b])})
    r.out.grid[SIZE_AXIS] = list(sizes)
    r.out.params[SIZES_AT] = format_sizes_at(at)
    r.out.params["steady_bracket"] = sentinel == BRACKET
    return True


def resolve(spec: RunSpec, detected_gpu: str, model_obj: object | None = None) -> Resolution:
    """Fill in every "steady" knob of `spec` for the GPU the run is on.

    Without `model_obj` (before the model loads) only the table is used: a pair it lacks raises under the
    default policy and is left pending under "table+formula". With `model_obj` (after loading) pending
    knobs get the formula. The table always wins where it has an entry."""
    sentinel, conc = requested(spec.params, spec.grid)
    if sentinel is None and not conc:
        return Resolution()
    model = str(spec.params["model"])
    backend, spec_settings = _spec_of(model)
    settings = spec_settings | spec.invariants | spec.params
    policy = _policy(settings)
    gpu = gpu_key(detected_gpu)
    out = Resolution(params={"steady_rule": RULE, "steady_policy": policy})
    r = _Run(model, backend, spec_settings, settings, policy, gpu, model_obj, out)
    if conc:
        _concurrency(r)
    if sentinel is not None and not _batch(r, sentinel, [int(v) for v in spec.grid.get(LENGTH_AXIS, [])]):
        return Resolution(pending=True)  # the formula needs the loaded model; the runner resolves again
    out.params |= {
        "steady_requested": sentinel or STEADY,
        "steady_source": next(iter(r.sources)) if len(r.sources) == 1 else "table+formula",
        "steady_status": ",".join(f"{n}:{st}" for n, st in sorted(r.statuses.items())),
        "steady_runs": ",".join(r.runs),
        "steady_gpu": gpu,
    }
    return out


def sizes_for(settings: Mapping[str, Scalar], length: int, grid_sizes: Sequence[int]) -> list[int]:
    """The batch sizes the `batch` benchmark runs at one length: the resolved steady pick (or its bracket)
    when the runner set `batch_sizes_at`, else every size of the grid."""
    raw = settings.get(SIZES_AT)
    if not isinstance(raw, str) or not raw:
        return list(grid_sizes)
    b = parse_sizes_at(raw)[length]
    return bracket(b) if settings.get("steady_bracket") in (True, "true", "True", 1) else [b]


def halves_on_oom(settings: Mapping[str, Scalar]) -> bool:
    """Whether the `batch` benchmark should halve a steady pick that runs out of memory: a plain "steady"
    run, which would otherwise measure nothing at that length. Explicit sweeps and brackets keep the OOM
    point as data (the bracket's smaller size is already in the run)."""
    raw = settings.get(SIZES_AT)
    return (
        isinstance(raw, str) and bool(raw) and settings.get("steady_bracket") not in (True, "true", "True", 1)
    )


def fallback_sizes(b: int) -> list[int]:
    """The batches to try, in order, after `b` ran out of memory: b/2, b/4, ..., 1."""
    out: list[int] = []
    while b > 1:
        b //= 2
        out.append(b)
    return out


def markdown_table() -> str:
    """The table as markdown, one row per entry (docs/steady_state_batch.md embeds it; a test checks)."""
    cols = ("Model", "GPU", "Setup", "Batch at 128 tokens", "Batch at 1024 tokens", "Steady state reached?")
    cols += ("Source run",)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for e in TABLE:
        if e.knob != "batch":
            continue
        setup = ", ".join(f"{v} {SETUP_WORD.get(k, k)}" for k, v in e.variant.items() if v) or "default"
        reached = {e.status[128], e.status[1024]}
        word = (
            "yes"
            if reached == {REACHED}
            else "no (plateau)"
            if reached == {PLATEAU}
            else (f"128: {e.status[128]}, 1024: {e.status[1024]}")
        )
        rows.append(f"| {e.model} | {e.gpu} | {setup} | {e.at[128]} | {e.at[1024]} | {word} | {e.runs[0]} |")
    rows += [
        "",
        "| Model | GPU | Concurrency | Steady state reached? | Source runs |",
        "|---|---|---|---|---|",
    ]
    for e in TABLE:
        if e.knob == "concurrency":
            rows.append(f"| {e.model} | {e.gpu} | {e.at[0]} | yes | {e.runs[0]} .. {e.runs[-1]} |")
    return "\n".join(rows)


GPU_COLUMNS = ("L4", "A100-SXM4", "A100-PCIe", "H100", "H200", "B200")  # GPUs the table has entries for


def summary_table() -> str:
    """One row per model, one column per GPU: what a run with default settings gets (markdown).
    Batch cells read "B at 128 / B at 1024 tokens"; * marks a pick where 95% GPU-busy was not reached
    (throughput plateau); "-" means no entry (the run fails unless it opts into the formula)."""
    models = list(dict.fromkeys(e.model for e in TABLE))
    rows = ["| Model | " + " | ".join(GPU_COLUMNS) + " |", "|---|" + "---|" * len(GPU_COLUMNS)]
    for model in models:
        cells = []
        for gpu in GPU_COLUMNS:
            knob: Knob = (
                "concurrency" if any(e.model == model and e.knob == "concurrency" for e in TABLE) else "batch"
            )
            e = lookup(model, gpu, {}, knob=knob)
            if e is None:
                cells.append("-")
                continue
            star = "" if set(e.status.values()) == {REACHED} else "*"
            if knob == "concurrency":
                cells.append(f"concurrency {e.at[0]}{star}")
            else:
                cell = f"{e.at[128]} / {e.at[1024]}" + star
                prep = e.variant.get("batch_host_prep")
                cells.append(cell + (f" ({prep})" if prep in ("pretokenized", "overlapped") else ""))
        rows.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(rows)
