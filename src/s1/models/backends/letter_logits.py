"""The `letter_logits` backend: a stock causal LM read through option-letter logits (SemIf "direct").

SemIf (github.com/TheoLeeCJ/SemIf, MIT) is the zero-training control: no adapter, no head. Each
question becomes a chat prompt whose options carry letters A, B, C, ...; one forward pass gives the
logits of the first answer token, and only the logits of the option letters are kept. This
reimplements the small prompt and readout of `src/semif_phase1/{core,direct}.py` at the commit
pinned in the catalog (`code_revision`) so that the only dependency is `transformers`, not their
package. `DIRECT_SYSTEM`, the payload layout, the `enable_thinking=False` template flag, the slot
round-trip checks and the no-truncation rule are theirs; the prompt text is reproduced under their MIT
licence. Prompt version `direct-options-v1`.

`decide` fans out one forward pass per question (batch size 1). Question types:
    choice  options are the criteria in order; answer via `choice_answer`.
    noul    two options, the criteria's "true" and "false" texts (a fixed pair if absent); the true
            letter is A, and P(true) comes from `noul_from_logits`.
    score   the criteria are an ordered list of levels (a dict is read in key order); level i gets a
            letter, and the answer is `score_answer` (expected 0-based level), as for the other models.
More than 16 options, a prompt over `max_input_tokens`, or a letter that is not one clean token raises
(`jeval` counts that as invalid and wrong). Nothing is truncated.

Settings:
    temperature        REQUIRED, no default. Divides the letter logits before the softmax. SemIf's
                       docs/CALIBRATION.md fits it per workload (1.23 authored, 2.50 WANLI, 1.71
                       Every); which one to use is an experiment decision, so it is always explicit.
    semif_model_dir    local model directory    default /shared/home/$USER/models/<repo name>
    max_input_tokens   whole-prompt token limit default spec_max_state_tokens (4096, SemIf's default)
    device             torch device             default "cuda" (exactly one GPU, as SemIf requires)
    dtype              bfloat16|float16|float32 default "bfloat16"
    noul_bias          added to the true-minus-false margin before the softmax    default 0.0
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ...record import Scalar
from ...registry import get_float, get_int, get_str
from ..answers import choice_answer, noul_from_logits, score_answer
from ..spec import ModelSpec

# SemIf src/semif_phase1/core.py @ 23cf1f39 (MIT): letters and system prompt.
LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
PROMPT_VERSION = "direct-options-v1"
DEFAULT_MAX_INPUT_TOKENS = 4096  # SemIf's `max_tokens` default in direct.score
NOUL_FALLBACK = ("The statement is true.", "The statement is false.")


class Tokenizer(Protocol):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str: ...
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class Scorer(Protocol):
    """Last-position logits at `slots` for one prompt, in slot order."""

    def slot_logits(self, ids: list[int], slots: list[int]) -> list[float]: ...


@dataclass(frozen=True)
class Prompt:
    ids: list[int]
    slots: list[int]


def _options(question: Mapping[str, object]) -> list[tuple[str, str]]:
    """(key, text) per option. noul gets its true/false pair; score levels are keyed "0", "1", ..."""
    criteria, qtype = question.get("criteria"), question["type"]
    if qtype == "noul":
        pairs = criteria if isinstance(criteria, dict) else {}
        return [
            ("true", str(pairs.get("true", NOUL_FALLBACK[0]))),
            ("false", str(pairs.get("false", NOUL_FALLBACK[1]))),
        ]
    if isinstance(criteria, dict):
        return [(str(k), str(v)) for k, v in criteria.items()]
    if qtype == "score" and isinstance(criteria, list):
        return [(str(i), str(v)) for i, v in enumerate(criteria)]
    raise TypeError(f"{qtype} question needs criteria, got {type(criteria).__name__}")


def build_messages(
    state: str, question: Mapping[str, object], options: Sequence[tuple[str, str]]
) -> list[dict[str, str]]:
    if not 2 <= len(options) <= len(LETTERS):
        raise ValueError(f"need 2-{len(LETTERS)} options, got {len(options)}")
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions:
        raise ValueError("question has no instructions")
    payload = {
        "evidence": state,
        "criterion": instructions,
        "options": [{"letter": LETTERS[i], "description": text} for i, (_, text) in enumerate(options)],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def slot_ids(tokenizer: Tokenizer, count: int) -> list[int]:
    """Token id of each option letter; each must be one exact round-trip token (SemIf `_slot_ids`)."""
    slots: list[int] = []
    for letter in LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"answer slot {letter!r} is not one exact round-trip token")
        slots.append(encoded[0])
    if len(set(slots)) != len(slots):
        raise ValueError("answer-slot tokens collide")
    return slots


def encode_prompt(
    tokenizer: Tokenizer, messages: list[dict[str, str]], count: int, max_tokens: int
) -> Prompt:
    """Template, tokenize and check the answer boundary (SemIf `encode_prompt`); never truncates."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"{len(ids)} input tokens exceed the limit {max_tokens}; no truncation allowed")
    slots = slot_ids(tokenizer, count)
    for letter, token in zip(LETTERS, slots, strict=False):
        if tokenizer.encode(text + letter, add_special_tokens=False) != [*ids, token]:
            raise ValueError(f"answer boundary changes tokenization for slot {letter}")
    return Prompt(ids, slots)


class LetterLogits:
    def __init__(
        self,
        tokenizer: Tokenizer,
        scorer: Scorer,
        *,
        temperature: float,
        max_input_tokens: int,
        noul_bias: float = 0.0,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.tokenizer, self.scorer = tokenizer, scorer
        self.temperature, self.max_input_tokens, self.noul_bias = temperature, max_input_tokens, noul_bias
        self.on_close = on_close

    def decide(self, state: str, questions: Mapping[str, object]) -> dict[str, object]:
        return {key: self._answer(state, key, q) for key, q in questions.items()}

    def _answer(self, state: str, key: str, question: object) -> dict[str, object]:
        if not isinstance(question, dict):
            raise TypeError(f"question {key!r} is not an object")
        options = _options(question)
        messages = build_messages(state, question, options)
        prompt = encode_prompt(self.tokenizer, messages, len(options), self.max_input_tokens)
        logits = self.scorer.slot_logits(prompt.ids, prompt.slots)
        if len(logits) != len(options):
            raise ValueError(f"got {len(logits)} letter logits for {len(options)} options")
        match question["type"]:
            case "choice":
                return choice_answer(
                    dict(zip((k for k, _ in options), logits, strict=True)), self.temperature
                )
            case "noul":
                return noul_from_logits(logits[0], logits[1], self.temperature, self.noul_bias)
            case "score":
                return score_answer(logits, self.temperature)
        raise ValueError(f"unknown question type {question['type']!r}")

    def close(self) -> None:
        if self.on_close is not None:
            self.on_close()


class TorchScorer:
    """One batch-1 forward pass per prompt, as in SemIf `direct.score`; calls are serialized."""

    def __init__(self, model: Callable[..., object], device: str) -> None:
        self.model, self.device = model, device
        self.lock = threading.Lock()

    def slot_logits(self, ids: list[int], slots: list[int]) -> list[float]:
        torch = importlib.import_module("torch")
        kwargs: dict[str, object] = {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=self.device),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=self.device),
            "use_cache": False,
            "return_dict": True,
        }
        if "logits_to_keep" in inspect.signature(getattr(self.model, "forward", self.model)).parameters:
            kwargs["logits_to_keep"] = 1
        with self.lock, torch.inference_mode():
            out = self.model(**kwargs)
            vocabulary = getattr(out, "logits")[0, -1, :].float()  # noqa: B009
            return [float(v) for v in vocabulary[slots].cpu().tolist()]


def _load(model_dir: Path, device: str, dtype: str) -> tuple[Callable[..., object], Tokenizer]:
    """Load a local checkpoint like SemIf `core.load_causal_model` (Qwen3.5 text model, strict load)."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"dtype must be bfloat16, float16 or float32, got {dtype!r}")
    common = {"local_files_only": True, "trust_remote_code": False}
    config = transformers.AutoConfig.from_pretrained(str(model_dir), **common)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir), **common)
    cls = transformers.AutoModelForCausalLM
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:
        cls = getattr(transformers, "Qwen3_5ForCausalLM", None)
        if cls is None:
            raise RuntimeError("installed transformers lacks the native Qwen3.5 model")
        config = config.get_text_config()
    model, loading = cls.from_pretrained(
        str(model_dir),
        config=config,
        dtype=getattr(torch, dtype),
        low_cpu_mem_usage=True,
        output_loading_info=True,
        **common,
    )
    if any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"checkpoint did not load completely: {loading}")
    return model.to(device).eval(), tokenizer


def _temperature(settings: dict[str, Scalar]) -> float:
    if "temperature" not in settings:
        raise ValueError(
            "letter_logits needs an explicit `temperature` (no default): set it under [params] or "
            "[invariants]. SemIf's docs/CALIBRATION.md fits 1.23 (authored), 2.50 (WANLI), 1.71 (Every)."
        )
    return get_float(settings, "temperature", 1.0)


class LetterLogitsBackend:
    def build(self, spec: ModelSpec, settings: dict[str, Scalar]) -> LetterLogits:
        temperature = _temperature(settings)
        default_dir = f"/shared/home/{os.environ.get('USER', '')}/models/{spec.repo.split('/')[-1]}"
        model_dir = Path(get_str(settings, "semif_model_dir", default_dir))
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"{model_dir} missing: sbatch scripts/download_model.sh {spec.repo} {spec.revision}"
            )
        limit = get_int(
            settings, "max_input_tokens", get_int(settings, "spec_max_state_tokens", DEFAULT_MAX_INPUT_TOKENS)
        )
        device = get_str(settings, "device", "cuda")
        if not re.fullmatch(r"cuda(:\d+)?|cpu", device):
            raise ValueError(f"device must be cuda, cuda:N or cpu, got {device!r}")
        model, tokenizer = _load(model_dir, device, get_str(settings, "dtype", "bfloat16"))
        return LetterLogits(
            tokenizer,
            TorchScorer(model, device),
            temperature=temperature,
            max_input_tokens=limit,
            noul_bias=get_float(settings, "noul_bias", 0.0),
        )
