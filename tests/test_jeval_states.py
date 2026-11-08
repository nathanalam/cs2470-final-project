"""JEVal-backed state source for shape_sweep, on a FAKE corpus: no dataset, no network."""

import pytest

from s1.benchmarks import jeval, shape_sweep
from s1.benchmarks.jeval import Row
from s1.benchmarks.jeval_states import JevalStates, cut_to, shared_prefix
from s1.benchmarks.shape_sweep import SyntheticStates, WordCounter, check_answers, sweep

from .test_shape_sweep import Clock

C = WordCounter()


def words(tag: str, n: int) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def choice_row(i: int, state: str) -> Row:
    q = {"type": "choice", "instructions": f"pick {i}", "criteria": {"a": "x", "b": "y", "c": "z"}}
    return Row(f"r{i}", "src", "choice", state, {f"k{i}": q}, "a")


def noul_row(i: int, state: str) -> Row:
    q = {"type": "noul", "instructions": f"true {i}?", "criteria": "claim"}
    return Row(f"n{i}", "src", "noul", state, {f"k{i}": q}, 1)


def score_row(i: int) -> Row:
    q = {"type": "score", "instructions": "level?", "criteria": ["lo", "hi"]}
    return Row(f"s{i}", "src", "score", "some state", {f"k{i}": q}, 0)


def corpus() -> list[Row]:
    lengths = [5, 12, 40, 60, 100, 250, 600, 2500]
    rows = [choice_row(i, words(f"s{i}_", n)) for i, n in enumerate(lengths)]
    return [*rows, noul_row(90, words("nn", 70)), score_row(91)]


def test_state_has_exact_length_and_is_deterministic_per_salt():
    a, b = JevalStates(corpus(), seed=1), JevalStates(corpus(), seed=1)
    for tokens in (1, 8, 32, 100, 700, 2048):
        for salt in (0, 1, 2, 5):
            s = a.state(tokens, C, salt)
            assert C.count(s) == tokens
            assert s == b.state(tokens, C, salt) == a.state(tokens, C, salt)


def test_call_order_does_not_change_a_state():
    a, b = JevalStates(corpus(), seed=2), JevalStates(corpus(), seed=2)
    forward = [a.state(32, C, salt) for salt in range(1, 6)]
    backward = [b.state(32, C, salt) for salt in range(5, 0, -1)][::-1]
    assert forward == backward


def test_different_salts_give_different_states():
    src = JevalStates(corpus(), seed=0)
    for tokens in (8, 32, 100, 700):
        got = [src.state(tokens, C, salt) for salt in range(1, 6)]
        assert len(set(got)) == len(got)


def test_states_that_cut_to_the_same_text_are_not_repeated():
    boiler = words("boiler", 50)  # three rows start with the same 50 words
    rows = [choice_row(i, f"{boiler} {words(f'tail{i}_', 100)}") for i in range(3)]
    got = [JevalStates(rows).state(40, C, salt) for salt in range(1, 4)]
    assert len(set(got)) == 1  # a fresh source each: all cut to the boilerplate ...
    src = JevalStates(rows)
    assert len({src.state(40, C, salt) for salt in range(1, 4)}) == 1  # ... only one distinct text exists
    assert len({src.state(60, C, salt) for salt in range(1, 4)}) == 3  # ... but at 60 tokens there are three


def test_salt_wraps_after_every_distinct_text_was_used():
    src = JevalStates([choice_row(0, words("a", 20)), choice_row(1, words("b", 20))])
    seen = [src.state(10, C, salt) for salt in range(0, 6)]
    assert len(set(seen)) == 2 and seen[0] == seen[2] == seen[4]


def test_long_rows_are_cut_and_keep_their_beginning():
    row = words("x", 100)
    s = JevalStates([choice_row(0, row)]).state(10, C)
    assert row.startswith(s) and C.count(s) == 10


def test_prefers_rows_close_to_the_target_length():
    src = JevalStates(corpus())
    # at 50 tokens the closest rows above are the 60-word one (band 0) before the 100/250-word ones
    first = src.state(50, C, salt=-2)  # salt -2 is the first candidate
    assert first.startswith("s3_0 ")
    assert all(not src.state(50, C, s).startswith("s7_") for s in range(-2, 3))


def test_short_rows_are_concatenated_until_long_enough():
    rows = [choice_row(i, words(f"r{i}_", 5)) for i in range(6)]
    src = JevalStates(rows)
    s = src.state(23, C)
    assert C.count(s) == 23
    assert len({w.split("_")[0] for w in s.split()}) >= 4  # at least four different rows contribute
    first_row = next(r.state for r in rows if s.startswith(r.state))
    assert s.count(first_row) == 1  # a row is never used twice
    assert src.state(23, C, salt=1) != s


def test_text_is_never_empty_and_empty_state_rows_are_ignored():
    src = JevalStates([choice_row(0, ""), choice_row(1, words("a", 9))])
    assert src.n_states == 1
    assert C.count(src.state(5, C)) == 5
    with pytest.raises(ValueError, match="non-empty state"):
        JevalStates([choice_row(0, "")])


def test_cut_to_nudges_when_the_prefix_recounts_differently():
    class Wide:
        """Every 'é' is two tokens; `cut(text, k)` cuts k characters, so it undershoots until nudged."""

        name = "wide"

        def count(self, text):
            return len(text) + text.count("é")

        def cut(self, text, k):
            return text[:k]

    text = "é" * 3 + "x" * 100
    got = cut_to(text, 40, Wide())
    assert Wide().count(got) == 40 and text.startswith(got)
    assert cut_to("abc", 10, Wide()) == "abc"


def test_questions_are_real_distinct_and_deterministic():
    src = JevalStates(corpus(), seed=3)
    assert src.n_questions == 9  # eight choice + one noul, never the score row
    qs = src.questions(4, salt=1)
    assert list(qs) == ["q0", "q1", "q2", "q3"]
    assert len({q["instructions"] for q in qs.values()}) == 4
    assert qs == JevalStates(corpus(), seed=3).questions(4, salt=1)
    assert qs != src.questions(4, salt=2)
    assert {q["type"] for q in src.questions(9).values()} == {"choice", "noul"}
    assert len(src.questions(20)) == 20  # more than the pool: repeats are allowed


def test_check_answers_knows_noul_questions():
    qs = {"q0": {"type": "noul", "criteria": "c"}, "q1": {"type": "choice", "criteria": {"a": "x"}}}
    assert check_answers(qs, {"q0": {"noul": 0.4}, "q1": {"choice": "a"}}) == 0
    assert check_answers(qs, {"q0": {"choice": "a"}, "q1": {"noul": 0.4}}) == 2


def test_shared_prefix_counts_what_a_cache_could_reuse():
    a, b, c = "w1 w2 w3 w4 end", "w1 w2 w3 zzz", "q1 q2"
    assert shared_prefix([a], C) == 0.0
    assert shared_prefix([a, c], C) == 0.0
    assert shared_prefix([a, b, c], C) == (3 + 0) / 2  # b shares "w1 w2 w3"; c shares nothing


class TypedModel:
    """Answers each question by its type; keeps every request."""

    def __init__(self):
        self.requests = []

    def decide(self, state, questions):
        self.requests.append((state, dict(questions)))
        out = {}
        for k, q in questions.items():
            out[k] = {"type": "noul", "noul": 0.5} if q["type"] == "noul" else {"choice": "a"}
        return out


def test_sweep_with_jeval_source_sends_unique_real_requests():
    src, model = JevalStates(corpus(), seed=0), TypedModel()
    m = sweep(
        model,
        [32],
        [3],
        states=src,
        question_source=src,
        report_shared_prefix=True,
        min_point_s=2.0,
        clock=Clock(),
    )
    reps = int(m["shape_sweep_t32_q3_reps"])
    sent = model.requests[1:]  # [0] is the warm-up
    assert reps >= 3 and len(sent) == reps and m["shape_sweep_invalid"] == 0
    assert len({s for s, _ in sent}) == reps  # a fresh state ...
    assert len({tuple(q["instructions"] for q in qs.values()) for _, qs in sent}) == reps  # ... and questions
    assert {C.count(s) for s, _ in sent} == {32} and m["shape_sweep_t32_q3_state_tokens_actual"] == 32
    assert "shape_sweep_t32_q3_shared_prefix_tokens" in m


def test_default_source_is_synthetic_and_unchanged(monkeypatch):
    monkeypatch.setattr(jeval, "load_rows", lambda *a, **k: pytest.fail("synthetic must not load JEVal"))
    base = {"grid_state_tokens": "16", "grid_questions_per_request": "2", "shape_min_point_s": 0.0}
    runs = []
    for extra in ({}, {"shape_state_source": "synthetic"}):
        model = TypedModel()
        out = shape_sweep.run(model, base | extra)
        runs.append((model.requests, out))
    assert runs[0][0] == runs[1][0]
    assert runs[0][0][0][0] == SyntheticStates(0).state(16, C, salt=-1)  # warm-up state
    assert all(
        q["type"] == "choice" and q["instructions"].startswith("Question") for q in runs[0][0][1][1].values()
    )
    assert not any("shared_prefix" in k or "jeval" in k for k in runs[0][1])


def test_unknown_source_is_an_error():
    with pytest.raises(KeyError, match="unknown shape_state_source"):
        shape_sweep.run(
            TypedModel(),
            {"shape_state_source": "nope", "grid_state_tokens": "8", "grid_questions_per_request": "1"},
        )


def test_run_with_jeval_source_uses_the_loaded_rows(monkeypatch):
    monkeypatch.setattr(jeval, "load_rows", lambda *a, **k: corpus())  # no download
    model = TypedModel()
    settings = {
        "shape_state_source": "jeval",
        "grid_state_tokens": "20",
        "grid_questions_per_request": "2",
        "shape_min_point_s": 0.0,
        "shape_warmup": 0,
    }
    out = shape_sweep.run(model, settings)
    state, questions = model.requests[0]
    assert C.count(state) == 20 and state.split()[0].startswith(("s", "nn"))
    assert {q["instructions"] for q in questions.values()} <= {f"pick {i}" for i in range(8)} | {"true 90?"}
    assert out["shape_sweep_jeval_pool_states"] == 10 and out["shape_sweep_jeval_pool_questions"] == 9
    assert out["shape_sweep_invalid"] == 0
