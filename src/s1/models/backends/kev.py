"""The `kev` backend: Kev-4B (github.com/jaredpalmer/kev), a LoRA adapter plus a pointer head on a Qwen base.

A reimplementation of upstream's inference path at the catalog's code pin (kev/model.py `encode`,
`PointerHead`; kev/api.py `to_record`, `to_answers`; kev/checkpoint.py loading), because upstream's loader
downloads from the Hub. Each question is one causal row, `<state> state <q> instr (<opt> option </opt>)*
<decide>`, using Qwen's spare FIM/box tokens as delimiters. The pointer head scores each option's `</opt>`
hidden state against `<decide>`'s, and the logits are divided by the temperature stored in head.pt
(`kev_temperature` overrides it). The base loads in bf16 with the adapter merged (upstream's serving path),
and the head runs in fp32. Apache-2.0.

Weights (download each with scripts/download_model.sh at the catalog revision):
    kev_dir            adapter + head.pt     default $MODELS_DIR/kev-4b-<variant>
    base_dir           base model            default $MODELS_DIR/<base repo name>
Settings (all optional):
    max_prompt_tokens  longest row (state + one question) before the answer is invalid   default 16384
                       (upstream serves up to 65536 state tokens; 16384 matches openjev's max_model_len)
    kev_temperature    override head.pt's temperature                                    default: head.pt's
    dtype, require_gpu see backends/gpu.py
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ...record import Scalar
from ...registry import get_float, get_int
from ..answers import choice_answer, noul_from_logits, score_answer
from ..spec import ModelSpec
from . import gpu

if TYPE_CHECKING:
    import torch

# state, question, option, end of option, decide (kev/model.py SPECIAL)
SPECIAL = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def render(v: object, indent: int = 0) -> str:
    """kev/api.py `render`: JSON-ish values as indented text."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, str | int | float | bool):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    if isinstance(v, dict):
        return "\n".join(
            f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, dict | list) else f"{pad}{k}: {render(x)}"
            for k, x in v.items()
        )
    raise TypeError(f"cannot render {type(v).__name__}")


def option_text(name: str, desc: object) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


@dataclass(frozen=True)
class Row:
    kind: str  # choice | noul | score
    keys: list[str]  # answer keys, in option order (noul: ["false", "true"])
    instr: str
    options: list[str]


def to_row(q: object) -> Row:
    """kev/api.py `to_record` for one question. Noul options are [no, yes], so P(true) is the second."""
    if not isinstance(q, Mapping):
        raise TypeError("question must be an object")
    kind, criteria = q.get("type"), q.get("criteria")
    instr = render(q.get("instructions", ""))
    if kind == "noul":
        c = criteria if isinstance(criteria, Mapping) else {}
        return Row(
            "noul",
            ["false", "true"],
            instr,
            [option_text("no", c.get("false")), option_text("yes", c.get("true"))],
        )
    if kind == "choice":
        if not isinstance(criteria, Mapping) or len(criteria) < 2:
            raise ValueError("choice needs at least two criteria")
        keys = [str(k) for k in criteria]
        return Row(
            "choice", keys, instr, [option_text(k, v) for k, v in zip(keys, criteria.values(), strict=True)]
        )
    if kind == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score needs a list of at least two levels")
        levels = [render(x) for x in criteria]
        return Row("score", [str(i) for i in range(len(levels))], instr, levels)
    raise ValueError(f"unknown question type {kind!r}")


def escape(text: str) -> str:
    """kev/model.py `user_tokens`: `<|name|>` in user text becomes `<¦name¦>`, never a delimiter."""
    return _SPECIAL_RE.sub(r"<¦\1¦>", text)


@dataclass(frozen=True)
class Encoded:
    ids: list[int]
    decide: int  # index of <decide>
    opts: list[int]  # index of each option's </opt>


def encode_row(
    tokenize: Callable[[str], list[int]], special: Sequence[int], state: list[int], row: Row
) -> Encoded:
    """One causal row: `state` (already starting with <state>) + `<q> instr (<opt> o </opt>)* <decide>`."""
    state_id, q_id, o_id, c_id, d_id = special
    if not state or state[0] != state_id:
        raise ValueError("state must start with the <state> token")
    ids = [*state, q_id, *tokenize(escape(row.instr))]
    opts = []
    for o in row.options:
        ids += [o_id, *tokenize(escape(o)), c_id]
        opts.append(len(ids) - 1)
    ids.append(d_id)
    return Encoded(ids, len(ids) - 1, opts)


def answer(row: Row, logits: Sequence[float], temperature: float) -> dict[str, object]:
    """kev/api.py `to_answers`, from the raw pointer logits (softmax of logits / T)."""
    if row.kind == "noul":
        return noul_from_logits(logits[1], logits[0], temperature)
    if row.kind == "choice":
        return choice_answer(dict(zip(row.keys, logits, strict=True)), temperature)
    return score_answer(logits, temperature)


@dataclass
class Kev:
    tokenizer: object
    lm: torch.nn.Module
    head_q: torch.nn.Linear
    head_k: torch.nn.Linear
    special: tuple[int, ...]
    temperature: float
    device: torch.device
    max_tokens: int
    lock: threading.Lock = field(default_factory=threading.Lock)

    def tokenize(self, text: str) -> list[int]:
        out = self.tokenizer(text, add_special_tokens=False)  # ty: ignore[call-non-callable]
        return list(out["input_ids"])

    def logits(self, enc: Encoded) -> list[float]:
        import torch  # noqa: PLC0415

        if len(enc.ids) > self.max_tokens:
            raise ValueError(f"row is {len(enc.ids)} tokens, over max_prompt_tokens={self.max_tokens}")
        ids = torch.tensor([enc.ids], device=self.device)
        with self.lock, torch.inference_mode():
            h = self.lm(input_ids=ids).last_hidden_state[0].float()
            q = self.head_q(h[enc.decide])
            k = self.head_k(h[torch.tensor(enc.opts, device=self.device)])
            z = (k @ q) / (q.shape[-1] ** 0.5)
        return [float(x) for x in z.cpu()]

    def decide(self, state: str, questions: Mapping[str, object]) -> dict[str, object]:
        rows = {key: to_row(q) for key, q in questions.items()}
        s = [self.special[0], *self.tokenize(escape(render(state)))]
        return {
            key: answer(row, self.logits(encode_row(self.tokenize, self.special, s, row)), self.temperature)
            for key, row in rows.items()
        }


def build(spec: ModelSpec, settings: dict[str, Scalar]) -> Kev:
    import importlib  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    peft = importlib.import_module("peft")  # in the GPU extra only

    if spec.base_repo is None:
        raise ValueError(f"{spec.name}: kev needs a base_repo")
    variant = spec.name.partition(".")[2]
    kev_dir = gpu.local_dir(
        settings,
        "kev_dir",
        f"kev-4b-{variant}",
        gpu.download_hint(spec.repo, spec.revision, f"kev-4b-{variant}"),
    )
    base_name = spec.base_repo.rpartition("/")[2]
    base_dir = gpu.local_dir(
        settings, "base_dir", base_name, gpu.download_hint(spec.base_repo, spec.base_revision, base_name)
    )
    device = gpu.cuda_device(settings, spec.name)
    dtype = gpu.torch_dtype(settings)

    meta = torch.load(kev_dir / "head.pt", map_location="cpu", weights_only=True)
    if (
        meta.get("option_isolation")
        or meta.get("special_embeddings")
        or meta.get("weights", "lora") != "lora"
    ):
        raise ValueError(f"{kev_dir}/head.pt uses options this backend does not implement")
    tok = AutoTokenizer.from_pretrained(base_dir)
    if tok is None:
        raise RuntimeError(f"no tokenizer in {base_dir}")
    lm = AutoModelForCausalLM.from_pretrained(base_dir, dtype=dtype, attn_implementation="sdpa").model
    lm = peft.PeftModel.from_pretrained(lm, kev_dir).merge_and_unload().to(device=device, dtype=dtype).eval()

    head = meta["head"]
    dp, d = head["q.weight"].shape
    head_q, head_k = torch.nn.Linear(d, dp), torch.nn.Linear(d, dp)
    head_q.load_state_dict({"weight": head["q.weight"], "bias": head["q.bias"]})
    head_k.load_state_dict({"weight": head["k.weight"], "bias": head["k.bias"]})
    special = tuple(t for t in (tok.convert_tokens_to_ids(name) for name in SPECIAL) if isinstance(t, int))
    if len(set(special)) != len(SPECIAL) or tok.unk_token_id in special:
        raise ValueError(f"{base_dir} tokenizer lacks kev's delimiter tokens: {special}")
    temperature = get_float(settings, "kev_temperature", float(meta.get("temperature", 1.0)))
    print(f"{spec.name}: head.pt temperature {meta.get('temperature')!r}, using {temperature}", flush=True)
    return Kev(
        tokenizer=tok,
        lm=lm,
        head_q=head_q.to(device).float().eval(),
        head_k=head_k.to(device).float().eval(),
        special=special,
        temperature=temperature,
        device=device,
        max_tokens=get_int(settings, "max_prompt_tokens", 16384),
    )


class KevBackend:
    def build(self, spec: ModelSpec, settings: dict[str, Scalar]) -> Kev:
        return build(spec, settings)
