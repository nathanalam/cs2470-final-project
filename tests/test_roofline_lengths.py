"""The roofline benchmark's state-length axis ([grid] roofline_state_tokens)."""

from __future__ import annotations

import dataclasses

import pytest

from s1 import registry, runner
from s1.benchmarks import jeval, roofline, shape_sweep
from s1.config import RunSpec
from tests.test_roofline import FakeProfiled, _no_gpu, row


def test_file_counter_fit_is_exact_with_a_real_tokenizer(tmp_path):
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: PLC0415

    vocab = {w: i for i, w in enumerate([*shape_sweep.WORDS, "[UNK]"])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    counter = shape_sweep.FileTokenCounter(str(path))
    text = " ".join(shape_sweep.WORDS[i % len(shape_sweep.WORDS)] for i in range(300))
    for n in (1, 32, 128, 299):
        assert counter.count(counter.fit(text, n)) == n


def test_length_rows_cut_long_states_and_reuse_the_questions():
    rows = [row(i, state_words=10 + 4 * i) for i in range(60)]  # states of 10 .. 246 words
    rows = [
        dataclasses.replace(r, state=" ".join(f"{r.id}w{j}" for j in range(len(r.state.split()))))
        for r in rows
    ]
    counter = shape_sweep.WordCounter()
    tagged = roofline.length_rows(rows, counter, {"roofline_per_length": 5}, [8, 64, 200])
    assert [n for n, _ in tagged] == [8] * 5 + [64] * 5 + [200] * 5  # length-major
    ids = {n: [r.id for m, r in tagged if m == n] for n in (8, 64, 200)}
    assert ids[8] == ids[64] == ids[200]  # only the state length differs between points
    assert {counter.count(r.state) for _, r in tagged} == {8, 64, 200}
    by_id = {r.id: r for r in rows}
    for n, r in tagged:  # each point is a prefix of that row's own state (no padding from other rows)
        original = by_id[r.id].state.split()
        assert len(original) >= 200
        assert r.state.split() == original[:n]


def test_length_rows_skip_short_states_and_long_questions():
    long_q = jeval.Row(
        "long-q", "src", "noul", " ".join(["w"] * 100), {"q": {"type": "noul", "instructions": "x " * 300}}, 1
    )
    rows = [
        row(0, state_words=10),
        row(1, "score", state_words=100),
        row(2, n_options=17, state_words=100),
        long_q,
    ]
    rows += [row(10 + i, state_words=100) for i in range(3)]
    tagged = roofline.length_rows(rows, shape_sweep.WordCounter(), {"roofline_per_length": 3}, [16, 64])
    assert {r.id for _, r in tagged} == {"r10", "r11", "r12"}


def test_length_rows_raise_when_too_few_states_are_long_enough():
    rows = [row(i, state_words=50) for i in range(10)]
    with pytest.raises(ValueError, match="only 0 JEVal rows have a state of at least 2048"):
        roofline.length_rows(rows, shape_sweep.WordCounter(), {"roofline_per_length": 4}, [32, 2048])


def _run(tmp_path, monkeypatch, *, grid):
    rows = [row(i, "noul" if i % 2 else "choice", state_words=10 + 8 * i) for i in range(80)]
    monkeypatch.setattr(jeval, "load_rows", lambda: rows)
    registry.MODELS["fake-profiled"] = lambda settings: FakeProfiled()
    try:
        spec = RunSpec(
            benchmarks=["roofline"],
            invariants={"gpu": "NVIDIA H100 80GB HBM3"},
            params={
                "model": "fake-profiled",
                "roofline_per_length": 6,
                "roofline_limit": 20,
                "roofline_counted": 0,
                "roofline_kernel_profile": 0,
            },
            grid=grid,
        )
        return runner.run(spec, results_dir=tmp_path, log=lambda _: None, gpu_sampler=_no_gpu)
    finally:
        registry.MODELS.pop("fake-profiled")


def test_length_axis_records_a_point_per_length(tmp_path, monkeypatch):
    rec = _run(tmp_path, monkeypatch, grid={roofline.LENGTH_AXIS: [32, 128, 512]})
    m = rec.results["roofline"]
    assert m["roofline_n"] == 18  # 6 questions x 3 lengths
    assert m["roofline_lengths"] == 3
    for n in (32, 128, 512):
        assert m[f"roofline_s{n}_n"] == 6
        assert m[f"roofline_s{n}_tokens"] == n  # the fake forwards one pass over exactly the state words
    # intensity grows with state length (more tokens per weight byte read)
    assert m["roofline_s32_intensity"] < m["roofline_s128_intensity"] < m["roofline_s512_intensity"]
    assert rec.grid == {roofline.LENGTH_AXIS: [32, 128, 512]}  # recorded with the run


def test_without_the_axis_the_natural_mode_is_unchanged(tmp_path, monkeypatch):
    rec = _run(tmp_path, monkeypatch, grid={})
    m = rec.results["roofline"]
    assert m["roofline_n"] == 20  # roofline_limit, natural JEVal lengths
    assert not any(
        k.startswith(("roofline_s", "roofline_lengths")) for k in m if not k.startswith("roofline_spec")
    )
