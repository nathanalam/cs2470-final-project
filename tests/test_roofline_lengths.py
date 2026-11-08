"""The roofline benchmark's state-length axis ([grid] roofline_state_tokens)."""

from __future__ import annotations

import pytest

from s1 import registry, runner
from s1.benchmarks import jeval, roofline, shape_sweep
from s1.config import RunSpec
from tests.test_roofline import FakeProfiled, _no_gpu, row


def test_fit_state_cuts_and_pads_with_real_filler():
    counter = shape_sweep.WordCounter()
    long = " ".join(f"s{i}" for i in range(50))
    assert counter.count(roofline.fit_state(long, 16, counter, [])) == 16
    short = "a b c"
    out = roofline.fit_state(short, 12, counter, ["d e f g h", "i j k l m n"])
    assert out == "a b c d e f g h i j k l"  # the state first, then filler, cut (words rejoined)
    with pytest.raises(ValueError, match="not enough filler"):
        roofline.fit_state(short, 100, counter, ["x"])


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


def test_length_rows_use_the_same_questions_at_every_length():
    rows = [row(i, state_words=10 + i) for i in range(60)]
    counter = shape_sweep.WordCounter()
    tagged = roofline.length_rows(rows, counter, {"roofline_per_length": 5}, [8, 64])
    assert [n for n, _ in tagged] == [8] * 5 + [64] * 5  # length-major
    ids8 = [r.id for n, r in tagged if n == 8]
    ids64 = [r.id for n, r in tagged if n == 64]
    assert ids8 == ids64  # only the state length differs between points
    assert {counter.count(r.state) for _, r in tagged} == {8, 64}
    by_id = {r.id: r for r in rows}
    for n, r in tagged:  # cut states keep the original's start; padded ones keep all of it
        original = by_id[r.id].state.split()
        assert r.state.split()[: min(n, len(original))] == original[:n]


def _run(tmp_path, monkeypatch, *, grid):
    rows = [row(i, "noul" if i % 2 else "choice", state_words=10 + i) for i in range(80)]
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
