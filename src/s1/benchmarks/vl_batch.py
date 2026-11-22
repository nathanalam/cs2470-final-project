"""vl_batch: what an image in the state costs a vision-language decision model, against text of the same
length, at the steady-state batch size (the multimodal experiment, docs/multimodal.md).

Requests come from ImajevBench's rows with one picture (`imajev`; every split, labels are not needed). For
each image size `px` (the longer side; every picture is resized to px x round(2 px / 3), the dataset's 3:2,
so all rows of a point have the same image layout) and batch size B, three arms run the same rows and
questions:

    image   the row's state + its picture                                  (SigLIP2 + projector + the LM)
    text    the row's state with the picture replaced by text filler of the same LM token count
            (a JEVal state cut to model.image_tokens(w, h) reference tokens, a different one per row)
    none    the row's state alone, no picture and no filler (px-independent: one series, key p0)

so `image` vs `text` is the cost of the modality at equal LM length, and `image` vs `none` what a picture adds
to a decision. Batches use the model's packed path (`prepare_batch` / `run_batch`: one tree, no padding, one
vision call per batch); host preparation (image resize, processor, tokenizer) runs before the timed pass, as
`batch_host_prep = "pretokenized"`.
Per point, `vl_batch_<arm>_p<px>_b<B>_*`, the `batch` benchmark's keys
(wall_ms, ms_per_decision, decisions_per_s, gpu_ms, tflops, tflops_wall, intensity, gbs, mb, gflop, tokens,
kernel_ms, kernel_busy, cpu_share, ...; see batch_sweep.py) plus

    image_tokens        LM placeholder tokens per batch (0 for text / none)
    vision_ms           SigLIP2 + projector GPU time per batch; vision_share = vision_ms / gpu_ms
    vision_gflop        their cost-model GFLOP per batch; vision_intensity their FLOP/byte
    peak_mb             peak CUDA memory allocated above the weights during the timed batches
    peak_mb_per_decision
    oom                 1 when the point ran out of memory

The FLOP and byte counts charge the vision tower (rows = tiles, tokens = padded patches) and the projector
(tokens = image tokens) with `flops.vision_stacks`, and the language model with `flops.arch`.

Steady state (the project's rule, 2026-11-19): per (arm, px), B* is the smallest B with kernel_busy >= 0.95
whose decisions_per_s no larger B beats by more than 5%; if no B reaches 0.95, the B with the most
decisions_per_s, flagged `_bstar_busy = 0`. `vl_batch_<arm>_p<px>_bstar` and `_bstar_*` copies of that point
are recorded, and the headlines compare the arms at their own B* at the largest px:
    vl_batch_image_text_ms   ms_per_decision image / text      vl_batch_image_text_mem   peak_mb_per_decision
    vl_batch_image_none_ms   ms_per_decision image / none

With `vl_batch_check = true`, before the sweep, the packed path's probabilities for `vl_batch_check_rows` rows
of each arm (largest px) are compared with the engine's own one-request answers (decide): the largest absolute
difference is `vl_batch_check_max_diff` (bf16 kernels differ by batch shape, so expect ~1e-2, not 0).

[grid] axes: vl_batch_image_px (e.g. [384, 768, 1024]) and vl_batch_sizes ([1, 2, 4, ..., 64]).
Settings (all optional): vl_batch_arms "image,text,none", vl_batch_tokenizer_file (the model's tokenizer.json,
to cut the filler; required for the text arm), vl_batch_pool (rows; default the largest B), vl_batch_seed 0,
vl_batch_min_decisions 32, vl_batch_min_batches 4, vl_batch_trace 2 (batches traced per point),
vl_batch_check false, vl_batch_check_rows 8.

The model implements roofline's `profile_info` / `take_forwards`, `prepare_batch(requests)`, `run_batch(p)`
(-> {"decisions", "real_tokens", "image_tokens"}), `image_tokens(width, height)`, `decide(state, questions,
images=...)`, and records its passes as spans with "stack": "vision" for the image path (backends/d1.py).
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from .. import env, flops
from ..record import Metrics
from ..registry import Settings, benchmark, get_bool, get_int, get_str, note
from . import batch_sweep, imajev, jeval, roofline
from .shape_sweep import FileTokenCounter, parse_int_list

NAME = "vl_batch"
PX = "vl_batch_image_px"
SIZES = "vl_batch_sizes"
ARMS = ("image", "text", "none")
BUSY = 0.95  # steady state: kernel_busy at least this
BEATEN = 1.05  # ...and no larger batch more than 5% faster
Request = tuple[object, ...]
Call = Callable[[], Mapping[str, int]]
BSTAR_KEYS = (
    "ms_per_decision",
    "decisions_per_s",
    "tflops",
    "intensity",
    "kernel_busy",
    "peak_mb_per_decision",
    "vision_share",
    "tokens",
)


@dataclass(frozen=True)
class Source:
    """One ImajevBench row with a picture, the raw material of every arm's request."""

    item: imajev.Item
    filler: str  # a long JEVal state, cut per px for the text arm


@dataclass(frozen=True)
class Timed:
    wall_ms: float
    cost: flops.Cost  # everything (LM + vision)
    vision: flops.Cost  # the vision tower + projector part
    forwards: int
    real_tokens: int
    decisions: int
    image_tokens: int


def frame(px: int) -> tuple[int, int]:
    """The picture size of a point: px wide, 3:2 (ImajevBench's pictures are 1536x1024 and 1264x848)."""
    return px, max(1, round(px * 2 / 3))


def sources(settings: Settings, root: Path, size: int) -> list[Source]:
    """`vl_batch_pool` (default `size`) seeded rows with exactly one picture, each with its own filler."""
    items = []
    for line in (root / imajev.RECORDS).read_text().splitlines():
        record: dict[str, Any] = json.loads(line)
        if len(record.get("images") or []) != 1:
            continue
        record = {**record, "gold": True, "gold_withheld": False}  # labels unused: parse every split
        (field,) = record["request"]["fields"]
        if field.get("type") == "ordinal":
            record["gold"] = field["levels"][0]["value"]
        elif field.get("type") == "choice":
            record["gold"] = field["options"][0]["value"]
        if (item := imajev.parse(record)) is not None:
            items.append(item)
    n = get_int(settings, "vl_batch_pool", size)
    if n > len(items):
        raise ValueError(f"vl_batch_pool {n} is over the {len(items)} ImajevBench rows with one picture")
    rng = random.Random(get_int(settings, "vl_batch_seed", 0))
    picked = rng.sample(items, n)
    states = sorted((r.state for r in jeval.load_rows() if len(r.state) > 20_000), key=len)
    if len(states) < n:
        raise ValueError(f"only {len(states)} JEVal states are long enough to cut filler from")
    fillers = rng.sample(states, n)
    return [Source(it, f) for it, f in zip(picked, fillers, strict=True)]


def resized(root: Path, item: imajev.Item, px: int) -> list[object]:
    Image = imajev.pil_image()

    (im,) = imajev.open_images(root, item, 0)
    return [im.resize(frame(px), Image.Resampling.BICUBIC)]  # ty: ignore[unresolved-attribute]


def arm_requests(
    arm: str, srcs: Sequence[Source], root: Path, px: int, fill: Callable[[str], str]
) -> list[Request]:
    out: list[Request] = []
    for s in srcs:
        row = s.item.row
        if arm == "image":
            out.append((row.state, row.questions, resized(root, s.item, px)))
        elif arm == "text":
            out.append((f"{fill(s.filler)}\n\n{row.state}", row.questions))
        else:
            out.append((row.state, row.questions))
    return out


def cost_of(
    fwd: Mapping[str, object], text: flops.Arch, vision: Mapping[str, flops.Arch]
) -> tuple[flops.Cost, bool]:
    """A recorded span's cost, and whether it is the image path."""
    f = roofline.forward(fwd)
    if fwd.get("stack") != "vision":
        return flops.cost(text, f), False
    tower = flops.cost(vision["vision"], f)
    proj = flops.cost(
        vision["projector"],
        flops.Forward(rows=1, tokens=int(cast("int", fwd.get("image_tokens", 0))), kept=0),
    )
    return flops.Cost(tower.flops + proj.flops, tower.bytes + proj.bytes, f.ms, tower.tokens), True


class VlModel(Protocol):
    def profile_info(self) -> Mapping[str, object]: ...
    def take_forwards(self) -> list[Mapping[str, object]]: ...
    def decide_batch(self, requests: Sequence[Request]) -> Mapping[str, int]: ...
    def prepare_batch(self, requests: Sequence[Request]) -> Any: ...  # noqa: ANN401 — the model's own
    def run_batch(self, prepared: Any) -> Mapping[str, int]: ...  # noqa: ANN401
    def batch_probs(self, prepared: Any) -> list[list[float]]: ...  # noqa: ANN401
    def image_tokens(self, width: int, height: int) -> int: ...
    def decide(
        self, state: str, questions: Mapping[str, object], images: Sequence[object] | None = None
    ) -> Mapping[str, Any]: ...


def time_call(m: VlModel, call: Call, text: flops.Arch, vision: Mapping[str, flops.Arch]) -> Timed:
    t0 = time.perf_counter()
    done = call()
    wall = (time.perf_counter() - t0) * 1e3
    total, vis = flops.ZERO, flops.ZERO
    forwards = m.take_forwards()
    for fwd in forwards:
        c, is_vision = cost_of(fwd, text, vision)
        if is_vision:
            vis = vis + c
            total = total + flops.Cost(c.flops, c.bytes, c.ms, 0)  # `tokens` counts LM tokens only
        else:
            total = total + c
    return Timed(
        wall_ms=wall,
        cost=total,
        vision=vis,
        forwards=len(forwards),
        real_tokens=int(done.get("real_tokens", 0)),
        decisions=int(done["decisions"]),
        image_tokens=int(done.get("image_tokens", 0)),
    )


def point(
    prefix: str, timed: Sequence[Timed], size: int, trace: batch_sweep.Trace | None, traced: int
) -> Metrics:
    as_batch = [batch_sweep.Timed(t.wall_ms, t.cost, t.forwards, t.real_tokens, t.decisions) for t in timed]
    m = batch_sweep.point(prefix, as_batch, size, trace, traced)
    n = len(timed)
    vision = flops.ZERO
    for t in timed:
        vision = vision + t.vision
    gpu = m[f"{prefix}gpu_ms"]
    m[f"{prefix}image_tokens"] = sum(t.image_tokens for t in timed) / n
    m[f"{prefix}vision_ms"] = vision.ms / n
    m[f"{prefix}vision_share"] = vision.ms / n / gpu if gpu > 0 else math.nan
    m[f"{prefix}vision_gflop"] = vision.flops / 1e9 / n
    m[f"{prefix}vision_intensity"] = vision.flops / vision.bytes if vision.bytes else math.nan
    return m


def steady_state(m: Metrics, series: str, sizes: Sequence[int]) -> int | None:
    """B* of one series (keys `<series>b<B>_*`), or None when no point ran."""
    ran = [b for b in sorted(sizes) if f"{series}b{b}_decisions_per_s" in m]
    if not ran:
        return None
    rate = {b: m[f"{series}b{b}_decisions_per_s"] for b in ran}
    for b in (b for b in ran if m.get(f"{series}b{b}_kernel_busy", 0.0) >= BUSY):
        if all(rate[c] <= BEATEN * rate[b] for c in ran if c > b):
            return b
    return max(ran, key=lambda b: rate[b])


def bstar(m: Metrics, series: str, sizes: Sequence[int]) -> Metrics:
    """`<series>bstar`, `_bstar_busy` and `_bstar_<key>` copies of the steady-state point."""
    b = steady_state(m, series, sizes)
    if b is None:
        return {}
    out: Metrics = {
        f"{series}bstar": float(b),
        f"{series}bstar_busy": float(m.get(f"{series}b{b}_kernel_busy", 0.0) >= BUSY),
    }
    for key in BSTAR_KEYS:
        if f"{series}b{b}_{key}" in m:
            out[f"{series}bstar_{key}"] = m[f"{series}b{b}_{key}"]
    return out


def peak_reset() -> float | None:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return float(torch.cuda.memory_allocated()) / 2**20


def peak_mb(base: float | None) -> float:
    torch = sys.modules.get("torch")
    if base is None or torch is None:
        return math.nan
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated()) / 2**20 - base


def check(m: VlModel, reqs: Sequence[Request]) -> float:
    """Largest |p_packed - p_engine| over the options of `reqs` (one question each)."""
    prepared = m.prepare_batch(reqs)
    m.run_batch(prepared)
    packed = m.batch_probs(prepared)
    worst = 0.0
    for r, probs in zip(reqs, packed, strict=True):
        images = list(cast("Sequence[object]", r[2])) if len(r) > 2 else None
        (answer,) = m.decide(str(r[0]), cast("Mapping[str, object]", r[1]), images=images).values()
        engine = (
            [answer["noul"], 1 - answer["noul"]]
            if answer["type"] == "noul"
            else list(answer["probabilities"].values())
        )
        worst = max(worst, *(abs(float(a) - float(b)) for a, b in zip(probs, engine, strict=True)))
    m.take_forwards()
    return worst


def sweep_point(
    m: VlModel,
    plan: Sequence[Sequence[Request]],
    arch: tuple[flops.Arch, Mapping[str, flops.Arch]],
    traced_n: int,
) -> tuple[list[Timed], batch_sweep.Trace | None, float, float]:
    """Prepare every batch (untimed), run them all once (warm-up), then timed, then traced: the timings, the
    trace, the host preparation ms per batch and the peak MB allocated above the weights."""
    t0 = time.perf_counter()
    prepared = [m.prepare_batch(b) for b in plan]
    prep_ms = (time.perf_counter() - t0) * 1e3 / len(plan)
    calls: list[Call] = [lambda p=p: m.run_batch(p) for p in prepared]
    for call in calls:
        call()
    m.take_forwards()
    base = peak_reset()
    timed = [time_call(m, call, *arch) for call in calls]
    peak = peak_mb(base)
    trace = batch_sweep.trace_batches(m, calls[:traced_n]) if traced_n else None
    return timed, trace, prep_ms, peak


def _model(model: object) -> VlModel:
    for attr in ("profile_info", "take_forwards", "prepare_batch", "run_batch", "image_tokens", "decide"):
        if not callable(getattr(model, attr, None)):
            raise TypeError(f"vl_batch needs a model with .{attr}(); got {type(model).__name__}")
    return cast("VlModel", model)


def _arms(settings: Settings) -> list[str]:
    arms = [a.strip() for a in get_str(settings, "vl_batch_arms", ",".join(ARMS)).split(",") if a.strip()]
    if bad := [a for a in arms if a not in ARMS]:
        raise ValueError(f"vl_batch_arms: unknown {bad}; known {', '.join(ARMS)}")
    return arms


def _filler(settings: Settings, arms: Sequence[str]) -> FileTokenCounter | None:
    if "text" not in arms:
        return None
    path = get_str(settings, "vl_batch_tokenizer_file", "")
    if not path:
        raise ValueError("the text arm needs vl_batch_tokenizer_file (the model's tokenizer.json)")
    return FileTokenCounter(str(env.resolve_path(path)))


@benchmark(NAME)
def run(model: object, settings: Settings) -> Metrics:
    m = _model(model)
    config = m.profile_info()["config"]
    if not isinstance(config, Mapping):
        raise TypeError("profile_info()['config'] must be the model's HF config dict")
    arch = (flops.arch(config), flops.vision_stacks(config))
    pxs, sizes = sorted(parse_int_list(settings, PX)), sorted(parse_int_list(settings, SIZES))
    arms = _arms(settings)
    root = imajev.dataset_dir()
    srcs = sources(settings, root, max(sizes))
    counter = _filler(settings, arms)
    tokens_at = {px: int(m.image_tokens(*frame(px))) for px in pxs}
    out: Metrics = {f"vl_batch_p{px}_image_tokens_per_picture": float(n) for px, n in tokens_at.items()}

    def requests_for(arm: str, px: int, rows: Sequence[Source]) -> list[Request]:
        n = tokens_at[px]
        return arm_requests(arm, rows, root, px, lambda s: counter.fit(s, n) if counter is not None else s)

    if get_bool(settings, "vl_batch_check", False):
        k = get_int(settings, "vl_batch_check_rows", 8)
        out["vl_batch_check_max_diff"] = max(check(m, requests_for(a, max(pxs), srcs[:k])) for a in arms)
        print(f"vl_batch: packed vs engine max |dp| = {out['vl_batch_check_max_diff']:.4f}", flush=True)

    traced_n = get_int(settings, "vl_batch_trace", 2)
    series = [(a, px) for a in arms if a != "none" for px in pxs] + ([("none", 0)] if "none" in arms else [])
    for arm, px in series:
        reqs = requests_for(arm, px or max(pxs), srcs)
        for size in sizes:
            prefix = f"vl_batch_{arm}_p{px}_b{size}_"
            n = max(
                get_int(settings, "vl_batch_min_batches", 4),
                math.ceil(get_int(settings, "vl_batch_min_decisions", 32) / size),
            )
            plan = [[reqs[(k * size + j) % len(reqs)] for j in range(size)] for k in range(n)]
            try:
                timed, trace, prep_ms, peak = sweep_point(m, plan, arch, traced_n)
            except Exception as e:
                if not batch_sweep._is_oom(e):
                    raise
                m.take_forwards()
                batch_sweep._free_cuda()
                out[f"{prefix}oom"] = 1.0
                note(f"{prefix}oom", f"{type(e).__name__}: {e}"[:200])
                print(f"vl_batch: {arm} px={px} B={size} out of memory", flush=True)
                continue
            out |= point(prefix, timed, size, trace, min(traced_n, len(plan)))
            out |= {
                f"{prefix}prep_ms": prep_ms,
                f"{prefix}peak_mb": peak,
                f"{prefix}peak_mb_per_decision": peak / size,
            }
            print(
                f"vl_batch: {arm} px={px} B={size} {out[f'{prefix}ms_per_decision']:.2f} ms/decision, "
                f"{out[f'{prefix}tflops']:.1f} TFLOP/s, "
                f"busy {out.get(f'{prefix}kernel_busy', math.nan):.2f}, "
                f"vision {out[f'{prefix}vision_share']:.2f}, peak {peak:.0f} MB",
                flush=True,
            )
        out |= bstar(out, f"vl_batch_{arm}_p{px}_", sizes)
    return headline(out, max(pxs)) | out


def headline(m: Metrics, px: int) -> Metrics:
    def at(arm: str, key: str) -> float:
        p = 0 if arm == "none" else px
        return m.get(f"vl_batch_{arm}_p{p}_bstar_{key}", math.nan)

    def ratio(a: float, b: float) -> float:
        return a / b if b and not math.isnan(a) and not math.isnan(b) else math.nan

    return {
        "vl_batch_image_text_ms": ratio(at("image", "ms_per_decision"), at("text", "ms_per_decision")),
        "vl_batch_image_none_ms": ratio(at("image", "ms_per_decision"), at("none", "ms_per_decision")),
        "vl_batch_image_text_mem": ratio(
            at("image", "peak_mb_per_decision"), at("text", "peak_mb_per_decision")
        ),
        "vl_batch_largest_px": float(px),
    }
