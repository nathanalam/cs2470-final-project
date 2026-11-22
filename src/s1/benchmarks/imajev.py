"""ImajevBench (huggingface.co/datasets/mohit67890/imajev-bench, v2.0-lite preview, CC-BY-4.0): typed
decisions from a picture, a written state, or both. The multimodal experiment's labeled task
(docs/multimodal.md).

Each record is one /v1/systemone-style request: a JSON state, 0-1 images (AI-generated, content-addressed
under assets/) and one typed question. Only the dev and calibration splits carry labels (254 rows; the test
split's are withheld), so those are what runs. Tracks: `text` (no image, 38 labeled rows), `visual` (the
image is the evidence, 103), `joint` (image + state, 113).

Questions are mapped to the JEVal shapes our backends read and graded by `jeval.grade` (the same accuracy,
ECE and Brier definitions as `jeval`):
    boolean -> noul    instructions = question; criteria true/false from yes/no_description   label 1/0
    choice  -> choice  criteria {value: description or value}                               label value
    ordinal -> score   criteria [level descriptions, lowest value first]                     label index
A row whose reference is Unknown (24 labeled rows: the evidence does not settle it) is not scored: our
decision models have no abstain output. They are counted in `imajev_unknown_skipped`. So `imajev_accuracy` is
accuracy on answerable rows, not ImajevBench's leaderboard score (which also rewards abstaining on Unknown).

A model that reads images sets `image_input = True` and takes `decide(state, questions, images=[PIL
images])`; it may also define `decide_with_usage(state, questions, images)` -> (answers, prompt tokens
read), and the benchmark then records prompt tokens per decision. A row with an image is skipped for a
text-only model, counted in `imajev_image_skipped` (run such models with `imajev_tracks = "text"`).

Settings (all optional):
    imajev_images       pass the images (false: the same rows with the images removed)   default true
    imajev_image_px     resize each image so its longer side is this many pixels first
                        (0 = as shipped; d1 caps pictures at 1 MP itself)                default 0
    imajev_tracks       comma list of tracks                                default "text,visual,joint"
    imajev_splits       comma list of labeled splits                        default "dev,calibration"
    imajev_limit        rows to run after filtering, 0 = all                default 0
    imajev_seed         seed for the limit sample                           default 0
    imajev_concurrency  requests in flight (d1 holds one engine lock anyway)   default 1
    imajev_warmup       answer the rows once, untimed, before measuring     default false

Metrics: imajev_accuracy (headline, every answerable row), _ece, _brier, _choice/_noul/_score_accuracy, _n,
_invalid, _<track>_accuracy and _<track>_n, imajev_input_tokens{,_p90,...} (prompt tokens per decision, when
the model reports them; per track imajev_<track>_input_tokens), imajev_latency_ms*, imajev_questions_per_s,
and on CUDA imajev_weights_mb (allocated before the first row) and imajev_peak_activation_mb (peak allocated
above that).
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import random
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..record import Metrics, summarize
from ..registry import Settings, benchmark, get_bool, get_int, get_str, warmup
from . import jeval

NAME = "imajev"
REPO = "mohit67890/imajev-bench"
REVISION = "043ab1f7425ef7b60e6206199cff43e1f80bc64f"  # v2.0-lite, pinned 2026-11-20
RECORDS = "records/records-public.jsonl"
RECORDS_SHA256 = (
    "8551501a51853e91689d5970a2a7f7218b1de48729b9743ba620d2ca32af6ba8"  # the card's stated digest
)
TRACKS = ("text", "visual", "joint")


def pil_image() -> Any:  # noqa: ANN401 — PIL.Image (Pillow comes with the GPU extra, via vllm)
    return importlib.import_module("PIL.Image")


@dataclass(frozen=True)
class Item:
    row: jeval.Row  # state (JSON text), one mapped question, label; row.source is the track
    track: str
    images: tuple[str, ...]  # asset paths relative to the dataset root
    image_sha256: tuple[str, ...]


@dataclass(frozen=True)
class Answered:
    outcome: jeval.Outcome
    track: str
    tokens: int | None


def question(field: Mapping[str, Any]) -> tuple[str, dict[str, object]]:
    """An ImajevBench field as (qtype, a JEVal-shaped question)."""
    kind, text = field.get("type"), str(field.get("question") or "")
    if kind == "boolean":
        q: dict[str, object] = {"type": "noul", "instructions": text}
        yes, no = field.get("yes_description"), field.get("no_description")
        if yes or no:
            q["criteria"] = {
                "true": str(yes or "The answer is yes."),
                "false": str(no or "The answer is no."),
            }
        return "noul", q
    if kind == "choice":
        options = field.get("options")
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError("choice field needs at least two options")
        criteria = {str(o["value"]): str(o.get("description") or o["value"]) for o in options}
        return "choice", {"type": "choice", "instructions": text, "criteria": criteria}
    if kind == "ordinal":
        levels = field.get("levels")
        if not isinstance(levels, list) or len(levels) < 2:
            raise ValueError("ordinal field needs at least two levels")
        ordered = sorted(levels, key=lambda lv: lv["value"])
        criteria = [str(lv.get("description") or lv["value"]) for lv in ordered]
        return "score", {"type": "score", "instructions": text, "criteria": criteria}
    raise ValueError(f"unknown field type {kind!r}")


def label(qtype: str, field: Mapping[str, Any], gold: object) -> str | int:
    if qtype == "noul":
        return int(bool(gold))
    if qtype == "choice":
        return str(gold)
    levels = sorted(v["value"] for v in field["levels"])
    return levels.index(gold)


def parse(record: Mapping[str, Any]) -> Item | None:
    """One labeled record as an Item; None for a row without a usable reference (test split, or Unknown)."""
    if record.get("gold_withheld") or "gold" not in record or record["gold"] is None:
        return None
    request = record["request"]
    if not isinstance(request, Mapping):
        raise TypeError(f"{record.get('id')}: request is not an object")
    (field,) = request["fields"]
    qtype, q = question(field)
    state = request.get("state")
    images = record.get("images") or []
    if not isinstance(images, list):
        raise TypeError(f"{record.get('id')}: images is not a list")
    row = jeval.Row(
        id=str(record["id"]),
        source=str(record["track"]),
        qtype=qtype,
        state=state if isinstance(state, str) else json.dumps(state, ensure_ascii=False),
        questions={str(field.get("id") or "decision"): q},
        label=label(qtype, field, record["gold"]),
    )
    return Item(
        row=row,
        track=str(record["track"]),
        images=tuple(str(im["path"]) for im in images),
        image_sha256=tuple(str(im["sha256"]) for im in images),
    )


def dataset_dir() -> Path:
    """The pinned snapshot (records + images, ~95 MB) in the HF cache."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    root = Path(
        snapshot_download(
            REPO, repo_type="dataset", revision=REVISION, allow_patterns=[RECORDS, "assets/*", "README.md"]
        )
    )
    digest = hashlib.sha256((root / RECORDS).read_bytes()).hexdigest()
    if digest != RECORDS_SHA256:
        raise ValueError(f"{REPO}@{REVISION[:8]}: {RECORDS} sha256 {digest}, expected {RECORDS_SHA256}")
    return root


def load(root: Path, splits: Sequence[str], tracks: Sequence[str]) -> tuple[list[Item], int]:
    """The labeled items of `splits` and `tracks`, and how many of them have an Unknown reference."""
    items: list[Item] = []
    unknown = 0
    for line in (root / RECORDS).read_text().splitlines():
        record: dict[str, Any] = json.loads(line)
        if record.get("split") not in splits or record.get("track") not in tracks:
            continue
        if "gold" in record and record["gold"] is None and not record.get("gold_withheld"):
            unknown += 1
        if (item := parse(record)) is not None:
            items.append(item)
    return items, unknown


def _list(settings: Settings, key: str, default: str, allowed: Sequence[str] | None = None) -> list[str]:
    out = [s.strip() for s in get_str(settings, key, default).split(",") if s.strip()]
    if allowed is not None and (bad := [s for s in out if s not in allowed]):
        raise ValueError(f"{key}: unknown {bad}; known {', '.join(allowed)}")
    return out


def sample(settings: Settings, root: Path) -> tuple[list[Item], int]:
    splits = _list(settings, "imajev_splits", "dev,calibration", ("dev", "calibration"))
    items, unknown = load(root, splits, _list(settings, "imajev_tracks", ",".join(TRACKS), TRACKS))
    if limit := get_int(settings, "imajev_limit", 0):
        items = random.Random(get_int(settings, "imajev_seed", 0)).sample(items, min(limit, len(items)))
    return items, unknown


def open_images(root: Path, item: Item, px: int) -> list[object]:
    """The item's pictures as RGB PIL images (sha256-checked), the longer side resized to `px` if px > 0."""
    Image = pil_image()

    out: list[object] = []
    for path, digest in zip(item.images, item.image_sha256, strict=True):
        data = (root / path).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"{path}: sha256 does not match the record")
        im = Image.open(io.BytesIO(data)).convert("RGB")
        if px > 0:
            scale = px / max(im.size)
            im = im.resize(
                (max(1, round(im.size[0] * scale)), max(1, round(im.size[1] * scale))), Image.BICUBIC
            )
        out.append(im)
    return out


def ask(model: object, root: Path, item: Item, use_images: bool, px: int) -> Answered:
    t = time.perf_counter()
    tokens: int | None = None
    try:
        images = open_images(root, item, px) if use_images and item.images else []
        with_usage = getattr(model, "decide_with_usage", None)
        if callable(with_usage):
            answers, used = with_usage(item.row.state, item.row.questions, images)
            tokens = int(used)
        elif images:
            decide = model.decide  # ty: ignore[unresolved-attribute]
            answers = decide(item.row.state, item.row.questions, images=images)
        else:
            answers = model.decide(item.row.state, item.row.questions)  # ty: ignore[unresolved-attribute]
        ms = (time.perf_counter() - t) * 1000
        (answer,) = answers.values()
        return Answered(jeval.grade(item.row, answer, ms), item.track, tokens)
    except Exception:  # any failure is an invalid answer, counted wrong (as jeval)
        ms = (time.perf_counter() - t) * 1000
        return Answered(jeval.Outcome(item.row.qtype, False, False, 0.0, 0.0, ms), item.track, tokens)


def metrics(answered: Sequence[Answered], wall_s: float) -> Metrics:
    m: Metrics = {
        "imajev_" + k.removeprefix("jeval_"): v
        for k, v in jeval.metrics([a.outcome for a in answered], wall_s).items()
    }
    # jeval's headline covers choice + noul only; here every answerable row counts (the 8 ordinal ones too)
    m = {"imajev_accuracy": _acc([a.outcome for a in answered])} | {
        k: v for k, v in m.items() if k != "imajev_accuracy"
    }
    for track in TRACKS:
        rows = [a for a in answered if a.track == track]
        if rows:
            m[f"imajev_{track}_accuracy"] = _acc([a.outcome for a in rows])
            m[f"imajev_{track}_n"] = float(len(rows))
            if toks := [a.tokens for a in rows if a.tokens is not None]:
                m[f"imajev_{track}_input_tokens"] = sum(toks) / len(toks)
    if toks := [float(a.tokens) for a in answered if a.tokens is not None]:
        m |= summarize("imajev_input_tokens", toks)
    return m


def _acc(outcomes: Sequence[jeval.Outcome]) -> float:
    return sum(o.correct for o in outcomes) / len(outcomes) if outcomes else float("nan")


def _cuda_mb() -> tuple[float, object] | None:
    """(MB allocated now, torch) on CUDA, after resetting the peak counter; None without a GPU."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return float(torch.cuda.memory_allocated()) / 2**20, torch


def evaluate(model: object, root: Path, items: Sequence[Item], settings: Settings) -> Metrics:
    use_images = get_bool(settings, "imajev_images", True)
    px = get_int(settings, "imajev_image_px", 0)
    reads_images = bool(getattr(model, "image_input", False))
    runnable = [it for it in items if not (it.images and use_images and not reads_images)]
    mem = _cuda_mb()
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=get_int(settings, "imajev_concurrency", 1)) as pool:
        answered = list(pool.map(lambda it: ask(model, root, it, use_images, px), runnable))
    m = metrics(answered, time.perf_counter() - t)
    m["imajev_image_skipped"] = float(len(items) - len(runnable))
    m["imajev_images"] = float(use_images)
    if mem is not None:
        base, torch = mem
        torch.cuda.synchronize()  # ty: ignore[unresolved-attribute]
        m["imajev_weights_mb"] = base
        peak = torch.cuda.max_memory_allocated() / 2**20  # ty: ignore[unresolved-attribute]
        m["imajev_peak_activation_mb"] = peak - base
    return m


@warmup(NAME)
def warm(model: object, settings: Settings) -> bool:
    if not get_bool(settings, "imajev_warmup", False):
        return False
    root = dataset_dir()
    evaluate(model, root, sample(settings, root)[0], settings)
    if callable(after := getattr(model, "after_warmup", None)):
        after()
    return True


@benchmark(NAME)
def run(model: object, settings: Settings) -> Metrics:
    if not callable(getattr(model, "decide", None)):
        raise TypeError(f"imajev needs a decision model with .decide(); got {type(model).__name__}")
    root = dataset_dir()
    items, unknown = sample(settings, root)
    return evaluate(model, root, items, settings) | {"imajev_unknown_skipped": float(unknown)}
