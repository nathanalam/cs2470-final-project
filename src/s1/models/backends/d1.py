"""The `d1` backend: Liquid AI's open d1 decision models (huggingface.co/LiquidAI/d1-3B), in-process.

Runs the model repo's own System One code (`modeling_d1.D1Model`, `runner.SystemOne` over `hybrid.py`'s
language model), like laya runs its shipped helper. The six code files are checked against the sha256 we
reviewed before `trust_remote_code` imports them: they make no network calls and read only the checkpoint
directory (tokenizer, processor). A question is rendered as `<|im_start|>user\\n<state>\\n\\nQUESTION:\\n
<question block><|im_end|>\\n<|im_start|>assistant\\n` and answered by a softmax over its options' tokens at
the last position (yes/no forms for a noul, option codes for a choice, digits for a score); nothing is decoded
and nothing is cached. The answers already have the /v1/systemone shape. LFM Open License v1.0.

A one-question request is one plain pass, `model(input_ids, logits_to_keep=1)` (`SystemOne._one_pass`). With
`d1_compile` that pass is `torch.compile(model.forward, mode="reduce-overhead")`, the card's CUDA-graph path:
each new prompt length compiles/captures on first use, so measure warm (the roofline's warm-up answers its
subset once, untimed). A request with several questions runs as one packed tree (`model.answer`: the state
once, the questions after it with no padding).

Roofline (benchmarks/roofline.py): both kinds of pass are timed with `ForwardRecorder.span` around the call,
so the eager and the graphed path are recorded the same way (hooks would miss a compiled forward). The pass
writes no cache (`cached` false) and keeps the logits at 1 position per question (the tied LM head runs
there). A tree is recorded as one row of trunk + all branch tokens, which overstates its attention slightly
(branches do not see each other). FlopCounterMode cannot see a graph replay, so `count_flops` is nan when
compiled. The SigLIP2 tower is loaded but never runs on a text state; the cost model reads the text tower.

Weights and code: `scripts/download_model.sh LiquidAI/d1-3B <revision> d1-3B` (or `s1 download` on Modal):
    model_dir         the repo                         default $MODELS_DIR/<repo name>
Settings (all optional):
    d1_compile        CUDA graphs via torch.compile     from the spec
    max_prompt_tokens longest pass (a tree: state + its longest question) before the answer is invalid
                      default 16384 (the model reads 32,768; 16384 matches openjev's max_model_len)
    measure_ceilings  measure the GPU's roofline ceilings before loading   default false
    dtype, require_gpu see backends/gpu.py
"""

from __future__ import annotations

import contextlib
import importlib
import math
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, cast

from ... import profiling
from ...record import Scalar
from ...registry import get_bool, get_int
from ..spec import ModelSpec
from . import gpu

# sha256 of the shipped System One code at d1-3B@05b80fa (reviewed 2026-11-12)
CODE_SHA256 = {
    "api.py": "94e9f82eabfdf92c4abe1153e9a6dc4dd9821f9cc2fec3221e79e70723d8ada0",
    "hybrid.py": "ce579827b952a6978a3560f70e74be9e228f9029de46048996b0c5f58d7f1f39",
    "lfm2_vl.py": "f9e8d3354572a169237865283594b0364209fdb858901121b9f99147328b5d95",
    "modeling_d1.py": "7fc6a001f80fc2d146f2a51128478f3b3bd4f1423a3ac129bc10f3fd0caf1d98",
    "prompt.py": "a20b5f52e41a6d29e694bb8c861a5a4803ce3eb69364edad79511ccf8edb5c31",
    "runner.py": "24d3ae718e9f9717771c0df39c93d324d00aa27bf3e12bab085c0922096cb0dd",
}


def check_code(model_dir: Path) -> None:
    """Refuse a checkout whose System One code is not the reviewed one."""
    for name, digest in CODE_SHA256.items():
        gpu.check_sha256(model_dir / name, digest)


def chain_shape(rows: int, tokens: int, keep: int) -> dict[str, object]:
    """A plain pass: `rows` x `tokens`, logits at `keep` positions per row (0 = all), no cache."""
    return {"rows": rows, "tokens": tokens, "past": 0, "kept": keep or tokens, "cached": False}


def tree_shape(trunk: int, branches: list[int]) -> dict[str, object]:
    """A packed tree: the trunk and every branch on one axis, logits at each branch's last token."""
    return {"rows": 1, "tokens": trunk + sum(branches), "past": 0, "kept": len(branches), "cached": False}


def check_length(tokens: int, max_tokens: int) -> None:
    if tokens > max_tokens:
        raise ValueError(f"{tokens} prompt tokens, over max_prompt_tokens={max_tokens}")


def vision_shape(tiles: int, patches: int, image_tokens: int) -> dict[str, object]:
    """The image path of a packed batch: SigLIP2 over `tiles` x `patches` (padded), then the projector over
    `image_tokens` (benchmarks/vl_batch.py charges each to its flops.vision_stacks arch)."""
    return {
        "stack": "vision",
        "rows": tiles,
        "tokens": patches,
        "past": 0,
        "kept": 0,
        "cached": False,
        "image_tokens": image_tokens,
    }


def shared_prefix(rows: Sequence[Sequence[int]], stop: int | None = None) -> int:
    """Tokens every row starts with, leaving each row at least its last token (the engine's `_logz_ids`
    rule), and stopping before the first `stop` token (an image placeholder: rows with the same image layout
    share placeholder ids but not the features scattered into them)."""
    shared, shortest = 0, min(len(r) for r in rows)
    while shared < shortest - 1 and len({r[shared] for r in rows}) == 1 and rows[0][shared] != stop:
        shared += 1
    return shared


def option_columns(groups: Sequence[Sequence[Sequence[int]]]) -> tuple[list[int], list[list[list[int]]]]:
    """The union of every row's readout token ids (the columns run_batch copies back), and each row's option
    groups as indices into it."""
    columns = sorted({t for g in groups for option in g for t in option})
    at = {t: i for i, t in enumerate(columns)}
    return columns, [[[at[t] for t in option] for option in g] for g in groups]


def option_probs(scores: Sequence[float], groups: Sequence[Sequence[int]]) -> list[float]:
    """prompt.readout from copied-back logits: each option's best form, softmax over options (no calibration;
    the per-row log-normalizer cancels)."""
    best = [max(scores[i] for i in g) for g in groups]
    top = max(best)
    exps = [math.exp(b - top) for b in best]
    return [e / sum(exps) for e in exps]


@dataclass
class Packed:
    """A batch prepared on the host (prepare_batch): one tree of `trunk` + each row's branch, no padding."""

    trunk: Any  # (1, P) long, CPU
    packed: Any  # (sum(lengths),) long, CPU
    lengths: Any  # (B,) long, CPU
    vision: dict[str, object]  # the processor's pixel inputs (pixel_values, ...), CPU; {} for text
    columns: Any  # (K,) long: readout token ids
    groups: list[list[list[int]]]  # per row, per option: indices into columns
    decisions: int
    real_tokens: int
    image_tokens: int
    tiles: int
    patches: int


@dataclass
class D1:
    image_input: ClassVar[bool] = True  # decide(..., images=[PIL images]) (benchmarks/imajev.py)

    model: object  # modeling_d1.D1Model
    engine: object  # runner.SystemOne, model.engine
    max_tokens: int
    compiled: bool
    hf_config: dict[str, object] = field(
        default_factory=dict
    )  # the VLM config; the cost model reads text_config
    ceilings: dict[str, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    recorder: profiling.ForwardRecorder | None = None

    def install(self) -> None:
        """Route the engine's two kinds of pass through the length guard and the recorder."""
        chain: Callable[..., object] = self.engine._one_pass  # ty: ignore[unresolved-attribute]
        tree: Callable[..., object] = self.model.answer  # ty: ignore[unresolved-attribute]

        def one_pass(**kwargs: object) -> object:
            rows, tokens = kwargs["input_ids"].shape  # ty: ignore[unresolved-attribute]
            check_length(int(tokens), self.max_tokens)
            keep = kwargs.get("logits_to_keep", 0)
            return self._timed(chain_shape(int(rows), int(tokens), int(keep)), lambda: chain(**kwargs))  # ty: ignore[invalid-argument-type]

        def answer(trunk: object, questions: object, lengths: object, **vision: object) -> object:
            p, branches = int(trunk.shape[1]), [int(n) for n in lengths.tolist()]  # ty: ignore[unresolved-attribute]
            check_length(p + max(branches), self.max_tokens)
            return self._timed(tree_shape(p, branches), lambda: tree(trunk, questions, lengths, **vision))

        self.engine._one_pass = one_pass  # ty: ignore[unresolved-attribute]
        self.model.answer = answer  # ty: ignore[unresolved-attribute]

    def _timed(self, shape: dict[str, object], run: Callable[[], object]) -> object:
        if self.recorder is None:
            return run()
        with self.recorder.span(shape):
            return run()

    last_scores: list[list[float]] = field(default_factory=list)  # run_batch's copied-back option logits

    def decide(
        self, state: str, questions: Mapping[str, object], images: Sequence[object] | None = None
    ) -> dict[str, object]:
        return self.decide_with_usage(state, questions, images or ())[0]

    def decide_with_usage(
        self, state: str, questions: Mapping[str, object], images: Sequence[object]
    ) -> tuple[dict[str, object], int]:
        """The engine's own path (a request with pictures is one plain pass, the SigLIP2 tower inside it), and
        the prompt tokens it read (image placeholders included)."""
        engine: Any = self.engine
        with self.lock:  # one engine, and CUDA-graph outputs are reused by the next replay
            args = (state, dict(questions), list(images)) if images else (state, dict(questions))
            out = engine.system_one(*args)
        return {str(k): v for k, v in out["answers"].items()}, int(
            out.get("usage", {}).get("input_tokens", 0)
        )

    # ------------------------------------------------------------- batched (benchmarks/vl_batch.py)

    def _module(self, name: str) -> Any:  # noqa: ANN401 — a module of the shipped code
        """A module of the shipped code (prompt, hybrid, runner), as trust_remote_code imported it."""
        package = type(self.engine).__module__.rpartition(".")[0]
        return sys.modules[f"{package}.{name}"]

    def _processor(self) -> Any:  # noqa: ANN401 — the checkpoint's AutoProcessor
        engine: Any = self.engine
        if engine.processor is None:
            engine.processor = engine._load_processor()
        return engine.processor

    def image_tokens(self, width: int, height: int) -> int:
        """LM placeholder tokens one `width` x `height` picture becomes (after the engine's 1 MP cap)."""
        Image = importlib.import_module("PIL.Image")  # Pillow comes with the GPU extra

        engine, model = cast("Any", self.engine), cast("Any", self.model)
        picture = self._module("runner").cap_pixels(Image.new("RGB", (width, height)))
        inputs = self._processor()(
            text=[engine._image_markup(1)], images=[[picture]], return_tensors="pt", add_special_tokens=False
        )
        return int((inputs["input_ids"] == model.config.image_token_id).sum())

    def decide_batch(self, requests: Sequence[tuple[object, ...]]) -> dict[str, int]:
        return self.run_batch(self.prepare_batch(requests))

    def prepare_batch(self, requests: Sequence[tuple[object, ...]]) -> Packed:
        """Host half: every request's questions as rows of one packed tree (the engine's `_logz_ids` layout:
        the rows' common start is the trunk, read once; each row's rest is a branch, no padding). Requests are
        `(state, questions)` or `(state, questions, images)`; a batch is all text or all with pictures, which
        then go through the processor together (capped at 1 MP as the engine does) and run as one SigLIP2
        call. The engine itself runs a request with pictures alone, so this is the batched path it lacks."""
        import torch  # noqa: PLC0415

        engine, model = cast("Any", self.engine), cast("Any", self.model)
        prompt, runner = self._module("prompt"), self._module("runner")
        tok = engine.tokenizer
        rows: list[list[int]] = []
        questions: list[object] = []
        texts: list[str] = []
        pictures: list[list[object]] = []
        for request in requests:
            state, qs = request[0], cast("Mapping[str, object]", request[1])
            images = list(cast("Sequence[object]", request[2])) if len(request) > 2 and request[2] else []
            for q in qs.values():
                question = prompt.as_question(q)
                questions.append(question)
                if not images:
                    rows.append(tok.encode(engine.render(state, question), add_special_tokens=False))
                    continue
                capped = [runner.cap_pixels(im) for im in images]
                markup = engine._image_markup(len(capped))
                prefix = prompt.prefix_text(tok, state, engine.bos, engine.state_style, engine.system, markup)
                texts.append(prefix + prompt.suffix_text(tok, question, engine.lead, engine.option_style))
                pictures.append(capped)
        if rows and texts:
            raise ValueError("a batch is all text or all with pictures")
        vision: dict[str, object] = {}
        image_id = int(model.config.image_token_id)
        tiles = patches = 0
        if texts:
            inputs = self._processor()(
                text=texts, images=pictures, return_tensors="pt", add_special_tokens=False, padding=True
            )
            mask = inputs.pop("attention_mask")
            rows = [ids[m.bool()].tolist() for ids, m in zip(inputs.pop("input_ids"), mask, strict=True)]
            if "pixel_attention_mask" in inputs:  # trim the patch padding, as the engine's _image_inputs does
                n = int(inputs["pixel_attention_mask"].sum(1).max())
                inputs["pixel_values"] = inputs["pixel_values"][:, :n]
                inputs["pixel_attention_mask"] = inputs["pixel_attention_mask"][:, :n]
            vision = dict(inputs)
            tiles, patches = int(inputs["pixel_values"].shape[0]), int(inputs["pixel_values"].shape[1])
        p = shared_prefix(rows, image_id)
        branches = [r[p:] for r in rows]
        check_length(p + max(len(b) for b in branches), self.max_tokens)
        columns, groups = option_columns([prompt.readout_ids(tok, q) for q in questions])
        return Packed(
            trunk=torch.tensor([rows[0][:p]], dtype=torch.long),
            packed=torch.tensor([t for b in branches for t in b], dtype=torch.long),
            lengths=torch.tensor([len(b) for b in branches], dtype=torch.long),
            vision=vision,
            columns=torch.tensor(columns, dtype=torch.long),
            groups=groups,
            decisions=len(rows),
            real_tokens=sum(len(r) for r in rows),
            image_tokens=sum(r.count(image_id) for r in rows),
            tiles=tiles,
            patches=patches,
        )

    def run_batch(self, b: Packed) -> dict[str, int]:
        """Device half: one SigLIP2 + projector call for every picture (its own span), then one packed pass of
        the hybrid language model over [trunk | branches] with the image features scattered into the
        placeholders (its own span), logits at each branch's last token; the readout columns come back to
        the host. The language-model pass is `hybrid.LanguageModel.answer` with the embeddings built here, so
        it runs the same kernels as the engine's text batches."""
        import torch  # noqa: PLC0415

        model: Any = self.model
        hybrid = self._module("hybrid")
        device = next(model.parameters()).device
        bt: Any = b
        with self.lock, torch.inference_mode():
            trunk = bt.trunk.to(device, non_blocking=True)
            packed = bt.packed.to(device, non_blocking=True)
            lengths = bt.lengths.to(device, non_blocking=True)
            ids = torch.cat((trunk[0], packed))[None]
            embeds = model.get_input_embeddings()(ids)
            if b.vision:
                vision = {
                    k: v.to(device, non_blocking=True) if hasattr(v, "to") else v
                    for k, v in bt.vision.items()
                }
                with self._span(vision_shape(b.tiles, b.patches, b.image_tokens)):
                    out = model.model.get_image_features(**vision)
                    features = torch.cat(out.pooler_output).to(embeds.dtype)
                slots = (ids == model.config.image_token_id).unsqueeze(-1).expand_as(embeds)
                embeds = embeds.masked_scatter(slots, features)
            p = int(trunk.shape[1])
            with self._span(tree_shape(p, [int(n) for n in bt.lengths.tolist()])):
                tree = hybrid.Tree(p, lengths)
                h = model.model.language_model.run(embeds, tree)
                logits = model.lm_head(h[0, tree.leaves])
            picked = logits[:, bt.columns.to(device)].float().cpu()
        self.last_scores = picked.tolist()
        return {"decisions": b.decisions, "real_tokens": b.real_tokens, "image_tokens": b.image_tokens}

    def batch_probs(self, b: Packed) -> list[list[float]]:
        """The option probabilities of the last run_batch (each row's readout, as the engine would give)."""
        return [option_probs(s, g) for s, g in zip(self.last_scores, b.groups, strict=True)]

    def _span(self, shape: dict[str, object]) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext() if self.recorder is None else self.recorder.span(shape)

    def profile_info(self) -> dict[str, object]:
        if self.recorder is None:
            self.recorder = profiling.ForwardRecorder(self.model)  # its hooks stay off: every pass is a span
        return {"config": self.hf_config, "head_bytes": 2, "ceilings": self.ceilings}

    def take_forwards(self) -> list[dict[str, object]]:
        return [] if self.recorder is None else self.recorder.take()

    def count_flops(self, state: str, questions: Mapping[str, object]) -> float:
        if self.compiled:
            return math.nan
        return profiling.count_flops(lambda: self.decide(state, questions))[1]

    def profiler(self, on: bool) -> None:
        """Open/close the CUDA profiler range Nsight Compute records (`s1 ncu`)."""
        import torch  # noqa: PLC0415

        torch.cuda.synchronize()
        (torch.cuda.profiler.start if on else torch.cuda.profiler.stop)()

    def run_info(self) -> dict[str, Scalar]:
        env: dict[str, Scalar] = {f"env_{k}": v for k, v in profiling.kernel_env(self.model).items()}
        env["env_d1_compiled"] = self.compiled
        return env

    def close(self) -> None:
        """Drop the weights and any compiled graphs, so the next run in this process can load."""
        import gc  # noqa: PLC0415

        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None
        self.engine = self.model = None
        if "torch" in sys.modules:
            torch = sys.modules["torch"]
            if self.compiled:
                torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()


def build(spec: ModelSpec, settings: dict[str, Scalar]) -> D1:
    name = spec.repo.rpartition("/")[2]
    model_dir = gpu.local_dir(settings, "model_dir", name, gpu.download_hint(spec.repo, spec.revision, name))
    check_code(model_dir)
    device = gpu.cuda_device(settings, spec.name)
    ceilings = (
        profiling.measure_ceilings(str(device)) if get_bool(settings, "measure_ceilings", False) else {}
    )

    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    model = transformers.AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, dtype=gpu.torch_dtype(settings)
    )
    model = model.to(device).eval()
    engine = model.engine  # SystemOne(model=self, tokenizer from model_dir), compile=False
    compiled = get_bool(settings, "d1_compile", False)
    if compiled:  # what SystemOne(compile=True) does
        engine._one_pass = torch.compile(model.forward, mode="reduce-overhead")
    d1 = D1(
        model,
        engine,
        get_int(settings, "max_prompt_tokens", 16384),
        compiled,
        hf_config=dict(model.config.to_dict()),
        ceilings=ceilings,
    )
    d1.install()
    print(f"{spec.name}: {type(model).__name__} on {device}, compiled={compiled}", flush=True)
    return d1


class D1Backend:
    def build(self, spec: ModelSpec, settings: dict[str, Scalar]) -> D1:
        return build(spec, settings)
