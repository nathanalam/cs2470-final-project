"""letter_logits backend with a stub tokenizer and scorer: no torch, transformers, GPU or weights."""

import json
import math
import sys

import pytest

from s1 import registry
from s1.models.backends import BACKENDS, load_backend
from s1.models.backends import letter_logits as ll
from s1.models.catalog import CATALOG, SEMIF_DIRECT, SEMIF_RERANKER


class CharTokenizer:
    """One token per character (id = code point), so every letter is a clean single token."""

    def __init__(self, broken_letter: str = "") -> None:
        self.broken_letter = broken_letter
        self.templates: list[dict[str, object]] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        self.templates.append(
            {"tokenize": tokenize, "gen": add_generation_prompt, "thinking": enable_thinking}
        )
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text == self.broken_letter:
            return [1, 2]
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class StubScorer:
    """Returns the configured logit per letter, and records what it was asked."""

    def __init__(self, by_letter: dict[str, float]) -> None:
        self.by_letter, self.calls = by_letter, []

    def slot_logits(self, ids, slots):
        self.calls.append((ids, slots))
        return [self.by_letter[chr(s)] for s in slots]


def model(by_letter, *, temperature=1.0, max_tokens=100_000, tokenizer=None, **kw):
    scorer = StubScorer(by_letter)
    return ll.LetterLogits(
        tokenizer or CharTokenizer(), scorer, temperature=temperature, max_input_tokens=max_tokens, **kw
    ), scorer


def choice(criteria, instructions="pick one"):
    return {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}}


def test_choice_letters_follow_option_order_and_prompt_matches_semif():
    m, scorer = model({"A": 0.0, "B": 3.0, "C": 1.0})
    out = m.decide("the state", choice({"opt_1": "red", "opt_2": "green", "opt_3": "blue"}))["q"]
    assert out["choice"] == "opt_2"
    assert list(out["probabilities"]) == ["opt_1", "opt_2", "opt_3"]
    assert sum(out["probabilities"].values()) == pytest.approx(1)
    ids, slots = scorer.calls[0]
    assert [chr(s) for s in slots] == ["A", "B", "C"]
    text = m.tokenizer.decode(ids)
    assert text.startswith(f"<system>{ll.DIRECT_SYSTEM}<user>")
    payload = json.loads(text.split("<user>")[1].removesuffix("<assistant>"))
    assert payload == {
        "evidence": "the state",
        "criterion": "pick one",
        "options": [
            {"letter": "A", "description": "red"},
            {"letter": "B", "description": "green"},
            {"letter": "C", "description": "blue"},
        ],
    }
    assert m.tokenizer.templates[0] == {"tokenize": False, "gen": True, "thinking": False}


def test_temperature_scales_logits():
    logits = {"A": 0.0, "B": math.log(4)}
    cold = model(logits, temperature=1.0)[0].decide("s", choice({"x": "1", "y": "2"}))["q"]
    hot = model(logits, temperature=2.0)[0].decide("s", choice({"x": "1", "y": "2"}))["q"]
    assert cold["probabilities"]["y"] == pytest.approx(0.8)
    assert hot["probabilities"]["y"] == pytest.approx(2 / 3)
    assert hot["choice"] == "y"


def test_noul_true_is_letter_a_and_uses_criteria_texts():
    m, scorer = model({"A": math.log(3), "B": 0.0})
    q = {
        "h": {
            "type": "noul",
            "instructions": "harmful?",
            "criteria": {"true": "It is.", "false": "It is not."},
        }
    }
    assert m.decide("s", q)["h"]["noul"] == pytest.approx(0.75)
    text = m.tokenizer.decode(scorer.calls[0][0])
    assert '"description": "It is."' in text
    assert text.index("It is.") < text.index("It is not.")


def test_noul_bias_and_missing_criteria_fallback():
    m, _ = model({"A": 0.0, "B": 0.0}, noul_bias=math.log(3))
    out = m.decide("s", {"h": {"type": "noul", "instructions": "x"}})["h"]
    assert out == {"type": "noul", "noul": pytest.approx(0.75)}


def test_score_levels_from_list_criteria():
    m, scorer = model({"A": 0.0, "B": 0.0, "C": 0.0})
    q = {"s": {"type": "score", "instructions": "how bad?", "criteria": ["none", "mild", "severe"]}}
    out = m.decide("s", q)["s"]
    assert out["type"] == "score"
    assert out["score"] == pytest.approx(1.0)
    assert out["probabilities"] == pytest.approx({"0": 1 / 3, "1": 1 / 3, "2": 1 / 3})
    assert [chr(s) for s in scorer.calls[0][1]] == ["A", "B", "C"]


def test_score_accepts_dict_criteria_in_key_order():
    m, _ = model({"A": 9.0, "B": 0.0})
    q = {"s": {"type": "score", "instructions": "x", "criteria": {"lo": "low", "hi": "high"}}}
    assert m.decide("s", q)["s"]["score"] == pytest.approx(0.0, abs=1e-3)


def test_one_forward_per_question():
    m, scorer = model({"A": 1.0, "B": 0.0})
    qs = {
        "a": {"type": "choice", "instructions": "i", "criteria": {"x": "1", "y": "2"}},
        "b": {"type": "noul", "instructions": "j"},
    }
    out = m.decide("s", qs)
    assert list(out) == ["a", "b"]
    assert len(scorer.calls) == 2


def test_sixteen_options_ok_seventeen_raises():
    sixteen = {f"o{i}": f"d{i}" for i in range(16)}
    m, scorer = model({c: float(i) for i, c in enumerate(ll.LETTERS)})
    assert m.decide("s", choice(sixteen))["q"]["choice"] == "o15"
    with pytest.raises(ValueError, match="2-16 options, got 17"):
        m.decide("s", choice({**sixteen, "o16": "d16"}))
    assert len(scorer.calls) == 1  # the 17-option question never reached the model


def test_too_few_options_raises():
    with pytest.raises(ValueError, match="options"):
        model({"A": 0.0})[0].decide("s", choice({"only": "one"}))


def test_over_length_prompt_raises_instead_of_truncating():
    m, scorer = model({"A": 0.0, "B": 0.0}, max_tokens=50)
    with pytest.raises(ValueError, match="exceed the limit 50; no truncation"):
        m.decide("x" * 500, choice({"a": "1", "b": "2"}))
    assert scorer.calls == []
    ok, _ = model({"A": 0.0, "B": 0.0}, max_tokens=10_000)
    ok.decide("x" * 500, choice({"a": "1", "b": "2"}))


def test_letter_that_is_not_one_token_raises():
    m, _ = model({"A": 0.0, "B": 0.0}, tokenizer=CharTokenizer(broken_letter="B"))
    with pytest.raises(ValueError, match="'B' is not one exact round-trip token"):
        m.decide("s", choice({"a": "1", "b": "2"}))


def test_bad_questions_raise():
    m, _ = model({"A": 0.0, "B": 0.0})
    with pytest.raises(ValueError, match="no instructions"):
        m.decide("s", {"q": {"type": "choice", "criteria": {"a": "1", "b": "2"}}})
    with pytest.raises(TypeError, match="needs criteria"):
        m.decide("s", {"q": {"type": "choice", "instructions": "i"}})
    with pytest.raises(ValueError, match="unknown question type"):
        m.decide("s", {"q": {"type": "rank", "instructions": "i", "criteria": {"a": "1", "b": "2"}}})
    with pytest.raises(TypeError, match="not an object"):
        m.decide("s", {"q": "nope"})


def test_close_runs_callback():
    closed = []
    m, _ = model({"A": 0.0}, on_close=lambda: closed.append(1))
    m.close()
    assert closed == [1]


# --- catalog and build -------------------------------------------------------------------------


def test_catalog_entry_is_ready_and_pins_survive():
    assert SEMIF_DIRECT.status == "ready"
    assert SEMIF_DIRECT.backend == "letter_logits"
    assert SEMIF_DIRECT.revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert SEMIF_DIRECT.code_revision == "23cf1f39fc9534fe81437200959b6dfc7106e45a"
    assert SEMIF_DIRECT.options["question_mode"] == "row"
    assert "temperature" not in SEMIF_DIRECT.options  # must come from the run config
    assert BACKENDS["letter_logits"] == "s1.models.backends.letter_logits:LetterLogitsBackend"
    assert isinstance(load_backend("letter_logits"), ll.LetterLogitsBackend)
    assert registry.get_model("semif.direct")


def test_reranker_stays_planned():
    assert SEMIF_RERANKER.status == "planned"
    assert SEMIF_RERANKER.backend is None
    assert any(s.name == "semif.reranker" for s in CATALOG)
    with pytest.raises(NotImplementedError, match=r"no backend yet: semif\.reranker"):
        registry.get_model("semif.reranker")({"model": "semif.reranker"})


def test_build_requires_temperature():
    with pytest.raises(ValueError, match="explicit `temperature`"):
        registry.get_model("semif.direct")({"model": "semif.direct"})
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_build_missing_model_dir_names_the_download(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"download_model\.sh Qwen/Qwen3\.5-4B 851bf6e8"):
        registry.get_model("semif.direct")(
            {"model": "semif.direct", "temperature": 1.23, "semif_model_dir": str(tmp_path / "nope")}
        )
    assert "torch" not in sys.modules


def test_build_rejects_bad_device(tmp_path):
    with pytest.raises(ValueError, match="device must be"):
        registry.get_model("semif.direct")(
            {"model": "semif.direct", "temperature": 1.0, "semif_model_dir": str(tmp_path), "device": "tpu"}
        )
    assert "torch" not in sys.modules


def test_build_wires_settings(tmp_path, monkeypatch):
    seen = {}

    def fake_load(model_dir, device, dtype):
        seen.update(model_dir=model_dir, device=device, dtype=dtype)
        return object(), CharTokenizer()

    monkeypatch.setattr(ll, "_load", fake_load)
    built = registry.get_model("semif.direct")(
        {"model": "semif.direct", "temperature": 2.5, "semif_model_dir": str(tmp_path)}
    )
    assert isinstance(built, ll.LetterLogits)
    assert built.temperature == 2.5
    assert built.max_input_tokens == 4096  # spec max_state_tokens
    assert seen == {"model_dir": tmp_path, "device": "cuda", "dtype": "bfloat16"}
    override = registry.get_model("semif.direct")(
        {
            "model": "semif.direct",
            "temperature": 1.0,
            "semif_model_dir": str(tmp_path),
            "max_input_tokens": 16384,
        }
    )
    assert isinstance(override, ll.LetterLogits)
    assert override.max_input_tokens == 16384
