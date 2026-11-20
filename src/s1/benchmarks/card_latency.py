"""Model-card latency: the d1-3B card's own GPU speed tests, re-run as the card describes them.

Liquid AI's card (huggingface.co/LiquidAI/d1-3B, "Speed") times warm calls, one request at a time, bf16,
median of 20 runs: one question (8 ms on an RTX 4090 with CUDA graphs, 16 ms without), three questions over
one state in one pass (21 ms), and a 3.4k-token state (102 ms). Its how-to example supplies the inputs: the
state "I was charged twice this month, please refund one of them." and the questions `refund` (noul), `team`
(choice of 3) and `urgency` (score of 3). This benchmark sends exactly those, so its numbers sit next to the
card's without a shape mismatch (the roofline's shortest length-axis point is ~230 tokens; the card's one
question is a few dozen).

Cases (`card_cases`, comma-separated, default all):
    single  the example state, `refund` only
    three   the example state, all three questions (one packed pass)
    long    a seeded synthetic state of `card_long_tokens` tokens, `refund` only

Each case is answered `card_warmup` times untimed (torch's CUDA graph trees record a new input size on its
2nd call and replay from the 3rd, so this must be at least 3 for a compiled model), then `card_runs` times
timed. Wall time is `perf_counter` around `decide`, as a user of the card's API sees it (tokenizing and the
answer readout included; the answers come back as Python floats, so the GPU has finished). When the model
records its passes (`profile_info`/`take_forwards`), the GPU span of each call is recorded too.

Settings (all optional):
    card_cases          cases to run                                   default "single,three,long"
    card_warmup         untimed calls per case                         default 5
    card_runs           timed calls per case                           default 20 (the card's count)
    card_long_tokens    tokens of the long case's state                default 3400
    state_tokenizer, state_tokenizer_file  how the long state is counted (see shape_sweep); use the model's
                        own tokenizer.json so 3,400 means the model's tokens

Results: headline `card_latency_ms` (median wall ms of `single`, lower is better), then per case
`card_<case>_{wall_ms_median,wall_ms_min,wall_ms_max,gpu_ms_median,tokens,passes,invalid}` and
`card_<case>_wall_ms` (mean, std, p90, p95, p99).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Mapping, Sequence

from ..record import Metrics, summarize
from ..registry import Settings, benchmark, get_int, get_str
from .shape_sweep import TOKENIZERS, DecisionModel, SyntheticStates

NAME = "card_latency"
CASES = ("single", "three", "long")

# The card's how-to example, verbatim
CARD_STATE = "I was charged twice this month, please refund one of them."
CARD_QUESTIONS: dict[str, object] = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Charges, refunds, invoices",
            "technical": "App or site faults",
            "fraud": "Suspected unauthorised use",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["Can wait", "Today", "Blocking the customer now"],
    },
}


def case_request(case: str, settings: Settings) -> tuple[str, dict[str, object]]:
    """(state, questions) for one case."""
    refund = {"refund": CARD_QUESTIONS["refund"]}
    if case == "single":
        return CARD_STATE, refund
    if case == "three":
        return CARD_STATE, dict(CARD_QUESTIONS)
    if case == "long":
        counter = TOKENIZERS[get_str(settings, "state_tokenizer", "approx_words")](settings)
        return SyntheticStates().state(get_int(settings, "card_long_tokens", 3400), counter), refund
    raise ValueError(f"unknown card_latency case {case!r}; expected one of {CASES}")


def valid(questions: Mapping[str, object], answers: object) -> bool:
    return isinstance(answers, Mapping) and all(
        isinstance(answers.get(k), Mapping) and "type" in answers[k] for k in questions
    )


def _median(xs: Sequence[float]) -> float:
    return statistics.median(xs)


def _forwards(model: object) -> list[Mapping[str, object]]:
    take = getattr(model, "take_forwards", None)
    return list(take()) if callable(take) else []


def measure(
    model: DecisionModel, case: str, state: str, questions: Mapping[str, object], *, warmup: int, runs: int
) -> Metrics:
    for _ in range(warmup):
        model.decide(state, questions)
    _forwards(model)  # drop the warm-up passes
    wall: list[float] = []
    gpu: list[float] = []
    tokens: list[float] = []
    passes = 0
    invalid = 0
    for _ in range(runs):
        t = time.perf_counter()
        try:
            answers = model.decide(state, questions)
            ok = valid(questions, answers)
        except Exception:  # a failed call is invalid, as everywhere else
            ok = False
        wall.append((time.perf_counter() - t) * 1000)
        invalid += not ok
        fw = _forwards(model)
        passes += len(fw)
        if fw:
            gpu.append(sum(float(f["ms"]) for f in fw))  # ty: ignore[invalid-argument-type]
            tokens.append(sum(float(f["tokens"]) for f in fw))  # ty: ignore[invalid-argument-type]
    p = f"{NAME}_{case}_"
    m: Metrics = {
        f"{p}wall_ms_median": _median(wall),
        f"{p}wall_ms_min": min(wall),
        f"{p}wall_ms_max": max(wall),
        f"{p}invalid": float(invalid),
        f"{p}passes": passes / runs,
    }
    m |= summarize(f"{p}wall_ms", wall)
    if gpu:
        m[f"{p}gpu_ms_median"] = _median(gpu)
        m[f"{p}tokens"] = _median(tokens)
    return m


@benchmark(NAME)
def run(model: object, settings: Settings) -> Metrics:
    if not callable(getattr(model, "decide", None)):
        raise TypeError(
            f"{NAME} needs a decision model with .decide(state, questions); got {type(model).__name__}"
        )
    dm: DecisionModel = model  # ty: ignore[invalid-assignment]
    cases = [c.strip() for c in get_str(settings, "card_cases", ",".join(CASES)).split(",") if c.strip()]
    warmup = get_int(settings, "card_warmup", 5)
    runs = get_int(settings, "card_runs", 20)
    if runs < 1:
        raise ValueError("card_runs must be at least 1")
    info = getattr(model, "profile_info", None)
    if callable(info):
        info()  # starts the model's pass recorder, if it has one
    out: Metrics = {}
    for case in cases:
        state, questions = case_request(case, settings)
        out |= measure(dm, case, state, questions, warmup=warmup, runs=runs)
    first = f"{NAME}_{cases[0]}_wall_ms_median"
    head: Metrics = {f"{NAME}_latency_ms": out[first], f"{NAME}_n": float(runs * len(cases))}
    head[f"{NAME}_invalid"] = sum(out[f"{NAME}_{c}_invalid"] for c in cases)
    return head | out
