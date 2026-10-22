import math

import pytest

from s1.benchmarks.jeval import Outcome, Row, ece, evaluate, grade

from .conftest import EchoModel

CHOICE = {"q": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None, "c": None}}}
NOUL = {"q": {"type": "noul", "instructions": "?"}}
SCORE = {"q": {"type": "score", "instructions": "?", "criteria": ["low", "mid", "high"]}}

ROWS = [
    Row("r0", "s", "choice", "x", CHOICE, "a"),  # echo picks "a": correct
    Row("r1", "s", "choice", "x", CHOICE, "b"),  # wrong
    Row("r2", "s", "noul", "x", NOUL, 1),  # p=0.8: correct
    Row("r3", "s", "noul", "x", NOUL, 0),  # wrong
    Row("r4", "s", "score", "x", SCORE, 2),  # predicts 0: wrong, |0-2| = 2
]


def test_grade_choice_brier():
    o = grade(ROWS[0], {"choice": "a", "probabilities": {"a": 0.7, "b": 0.15, "c": 0.15}}, 1.0)
    assert o.correct and o.confidence == 0.7
    assert o.brier == pytest.approx(0.3**2 + 0.15**2 * 2)


def test_grade_rejects_unknown_option():
    with pytest.raises(ValueError):
        grade(ROWS[0], {"choice": "zzz", "probabilities": {}}, 1.0)


def test_evaluate_with_echo_model():
    m = evaluate(EchoModel(), ROWS, concurrency=2)
    assert m["jeval_accuracy"] == 0.5  # 2 of the 4 choice + noul rows
    assert m["jeval_choice_accuracy"] == 0.5
    assert m["jeval_noul_accuracy"] == 0.5
    assert m["jeval_score_accuracy"] == 0.0
    assert m["jeval_score_mae"] == 2.0
    assert m["jeval_invalid"] == 0
    assert m["jeval_n"] == 5
    assert {"jeval_latency_ms", "jeval_latency_ms_p95", "jeval_latency_ms_p99"} <= set(m)


def test_failures_count_as_invalid_and_wrong():
    class Broken:
        def decide(self, state, questions):
            raise ConnectionResetError

    m = evaluate(Broken(), ROWS[:2])
    assert m["jeval_invalid"] == 2
    assert m["jeval_accuracy"] == 0.0
    assert math.isnan(m["jeval_brier"])
    assert "jeval_latency_ms" not in m


def test_ece_perfect_and_overconfident():
    assert ece([Outcome("choice", True, True, 1.0, 0, 0)] * 4) == 0.0
    assert ece([Outcome("choice", True, False, 0.9, 0, 0)] * 4) == pytest.approx(0.9)
