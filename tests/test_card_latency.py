"""card_latency: the d1-3B card's speed tests (one question, three in one pass, a long state)."""

from collections.abc import Mapping

import pytest

from s1.benchmarks import card_latency
from s1.registry import BENCHMARKS


class FakeD1:
    """Answers every question and records one pass per call, like the d1 backend's recorder."""

    def __init__(self, fail_every: int = 0) -> None:
        self.calls = 0
        self.fail_every = fail_every
        self.forwards: list[dict[str, object]] = []
        self.started = False

    def decide(self, state: str, questions: Mapping[str, object]) -> dict[str, object]:
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            raise RuntimeError("boom")
        self.forwards.append({"ms": 2.0 * len(questions), "tokens": float(len(state.split()) + 10)})
        return {k: {"type": "noul", "noul": 0.9} for k in questions}

    def profile_info(self) -> dict[str, object]:
        self.started = True
        return {}

    def take_forwards(self) -> list[dict[str, object]]:
        out, self.forwards = self.forwards, []
        return out


def test_registered():
    assert BENCHMARKS["card_latency"] is card_latency.run


def test_cases_use_the_card_example():
    state, single = card_latency.case_request("single", {})
    assert state == card_latency.CARD_STATE and list(single) == ["refund"]
    _, three = card_latency.case_request("three", {})
    assert list(three) == ["refund", "team", "urgency"]
    long_state, _ = card_latency.case_request("long", {"card_long_tokens": 50})
    assert len(long_state.split()) == 50  # approx_words: one token per word
    with pytest.raises(ValueError, match="unknown"):
        card_latency.case_request("huge", {})


def test_run_warms_then_times_each_case():
    model = FakeD1()
    m = card_latency.run(model, {"card_warmup": 3, "card_runs": 4, "card_long_tokens": 40})
    assert model.started and model.calls == 3 * 7  # 3 cases x (3 warm + 4 timed)
    assert next(iter(m)) == "card_latency_latency_ms"
    assert m["card_latency_latency_ms"] == m["card_latency_single_wall_ms_median"]
    assert m["card_latency_n"] == 12 and m["card_latency_invalid"] == 0
    assert m["card_latency_single_gpu_ms_median"] == 2.0  # warm-up passes are not counted
    assert m["card_latency_three_gpu_ms_median"] == 6.0
    assert m["card_latency_long_tokens"] == 50.0
    assert m["card_latency_single_passes"] == 1.0


def test_failures_are_invalid_and_cases_selectable():
    model = FakeD1(fail_every=2)
    m = card_latency.run(model, {"card_cases": "single", "card_warmup": 0, "card_runs": 4})
    assert m["card_latency_invalid"] == 2 and "card_latency_three_wall_ms_median" not in m


def test_model_without_recorder_has_no_gpu_keys():
    class Plain:
        def decide(self, state: str, questions: Mapping[str, object]) -> dict[str, object]:
            return {k: {"type": "noul", "noul": 0.5} for k in questions}

    m = card_latency.run(Plain(), {"card_cases": "single", "card_runs": 2})
    assert "card_latency_single_gpu_ms_median" not in m and m["card_latency_single_invalid"] == 0
