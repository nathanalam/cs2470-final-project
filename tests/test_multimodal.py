"""The multimodal experiment's torch-free pieces: the vision cost model, ImajevBench's mapping and grading,
d1's packed-batch helpers, and vl_batch's steady-state rule (docs/multimodal.md)."""

import json
import math

import pytest

from s1 import flops
from s1.benchmarks import imajev, vl_batch
from s1.models import catalog
from s1.models.backends import d1, semif
from s1.registry import Settings
from tests.conftest import EchoModel
from tests.test_flops import D1_3B

# d1-3B@05b80fa / LFM2.5-VL-3B@35a118d config.json (vision part)
VL = D1_3B | {
    "downsample_factor": 2,
    "projector_hidden_size": 2048,
    "projector_bias": True,
    "projector_use_layernorm": False,
    "vision_config": {
        "model_type": "siglip2_vision_model",
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_patches": 256,
        "patch_size": 16,
    },
}


def test_vision_and_projector_complete_the_hub_parameter_count():
    """Hub total of d1-3B@05b80fa is 3,123,483,888 parameters; the text tower holds 2,697,198,592."""
    parts = flops.vision_stacks(VL)
    assert flops.arch(VL).params + parts["vision"].params + parts["projector"].params == 3_123_483_888


def test_vision_tower_is_bidirectional_and_headless():
    tower = flops.vision_stacks(VL)["vision"]
    attention = [ly for ly in tower.layers if ly.mixer == "attention"]
    assert len(attention) == 27 and not any(ly.causal for ly in attention)
    cost = flops.cost(tower, flops.Forward(rows=2, tokens=1024, kept=0))
    # every patch sees all 1024 of its tile: 4 * d * pairs per layer on top of the weight matmuls
    pairs = 2 * 1024 * 1024
    assert cost.flops == pytest.approx(2 * 2 * 1024 * tower.active_params + 27 * 4 * 1152 * pairs)


def test_vision_stacks_refuses_other_models():
    with pytest.raises(KeyError):
        flops.vision_stacks({"model_type": "lfm2"})


def test_matched_pair_shares_a_base_and_the_vl_one_is_bigger():
    vl, text = catalog.LFM25_VL_3B, catalog.LFM25_2B6
    assert vl.base_repo == text.base_repo == "LiquidAI/LFM2.5-2.6B-Base"
    assert vl.base_revision == text.base_revision
    assert vl.options["mode"] == text.options["mode"] == "direct"
    assert (vl.size_bytes or 0) - (text.size_bytes or 0) == pytest.approx(0.85e9, rel=0.01)


# --------------------------------------------------------------------------- ImajevBench


def record(kind: str, gold: object, **extra: object) -> dict[str, object]:
    fields = {
        "boolean": {"id": "decision", "question": "Approved?", "type": "boolean"},
        "choice": {
            "id": "decision",
            "question": "Price?",
            "type": "choice",
            "options": [{"value": "1.00", "description": None}, {"value": "2.00", "description": "two"}],
        },
        "ordinal": {
            "id": "decision",
            "question": "Level?",
            "type": "ordinal",
            "levels": [{"value": 3, "description": "high"}, {"value": 1, "description": "low"}],
        },
    }
    return {
        "id": f"r-{kind}",
        "track": "text",
        "split": "dev",
        "images": [],
        "request": {"state": {"rule": "x"}, "fields": [fields[kind]]},
        "gold": gold,
    } | extra


def test_questions_map_to_the_jeval_shapes():
    b = imajev.parse(record("boolean", False))
    c = imajev.parse(record("choice", "2.00"))
    o = imajev.parse(record("ordinal", 3))
    assert b is not None and c is not None and o is not None
    assert (b.row.qtype, b.row.label) == ("noul", 0)
    assert c.row.questions["decision"] == {
        "type": "choice",
        "instructions": "Price?",
        "criteria": {"1.00": "1.00", "2.00": "two"},
    }
    assert c.row.label == "2.00"
    assert o.row.questions["decision"] == {
        "type": "score",
        "instructions": "Level?",
        "criteria": ["low", "high"],  # lowest value first
    }
    assert o.row.label == 1
    assert json.loads(b.row.state) == {"rule": "x"}


def test_unknown_and_withheld_references_are_not_scored():
    assert imajev.parse(record("boolean", None)) is None
    assert imajev.parse(record("choice", "1.00", gold_withheld=True)) is None


def write_dataset(tmp_path):
    rows = [
        record("boolean", True),
        record("choice", "1.00"),
        record("ordinal", 1),
        record("boolean", None),
        record("boolean", True, split="test", gold_withheld=True),
        record("boolean", True, track="visual", images=[{"path": "assets/a.jpg", "sha256": "0" * 64}]),
    ]
    (tmp_path / "records").mkdir()
    (tmp_path / imajev.RECORDS).write_text("\n".join(json.dumps(r) for r in rows))
    return tmp_path


def test_load_filters_splits_tracks_and_counts_unknown(tmp_path):
    root = write_dataset(tmp_path)
    items, unknown = imajev.load(root, ["dev", "calibration"], list(imajev.TRACKS))
    assert [it.track for it in items] == ["text", "text", "text", "visual"]
    assert unknown == 1
    assert imajev.load(root, ["dev"], ["visual"])[0][0].images == ("assets/a.jpg",)


def test_text_model_skips_image_rows_and_echo_scores_the_rest(tmp_path):
    root = write_dataset(tmp_path)
    items, _ = imajev.load(root, ["dev"], list(imajev.TRACKS))
    m = imajev.evaluate(EchoModel(), root, items, {})
    assert m["imajev_image_skipped"] == 1
    assert m["imajev_n"] == 3 and m["imajev_invalid"] == 0
    # echo: yes (right), first option "1.00" (right), level 0 = value 1 (right)
    assert m["imajev_accuracy"] == 1.0
    assert next(iter(m)) == "imajev_accuracy"


def test_images_off_runs_image_rows_as_text(tmp_path):
    root = write_dataset(tmp_path)
    items, _ = imajev.load(root, ["dev"], list(imajev.TRACKS))
    m = imajev.evaluate(EchoModel(), root, items, {"imajev_images": False})
    assert m["imajev_image_skipped"] == 0 and m["imajev_visual_n"] == 1


class UsageModel(EchoModel):
    image_input = True

    def decide_with_usage(self, state, questions, images):
        return self.decide(state, questions), 100 + 1000 * len(images)


def test_tokens_per_decision_come_from_the_model(tmp_path):
    root = write_dataset(tmp_path)
    items, _ = imajev.load(root, ["dev"], ["text"])
    m = imajev.evaluate(UsageModel(), root, items, {})
    assert m["imajev_input_tokens"] == 100
    assert m["imajev_text_input_tokens"] == 100


def test_unknown_track_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="imajev_tracks"):
        imajev.sample({"imajev_tracks": "audio"}, write_dataset(tmp_path))


# --------------------------------------------------------------------------- d1 packed batches


def test_shared_prefix_stops_before_image_placeholders():
    image = 9
    rows = [[1, 2, 3, image, image, 5], [1, 2, 3, image, image, 6]]
    assert d1.shared_prefix(rows) == 5
    assert d1.shared_prefix(rows, image) == 3
    assert d1.shared_prefix([[1, 2], [1, 2]]) == 1  # each row keeps its last token


def test_option_columns_and_probs_match_the_readout_rule():
    groups = [[[10, 11], [20]], [[20], [30], [10]]]
    columns, idx = d1.option_columns(groups)
    assert columns == [10, 11, 20, 30]
    scores = [0.0, 2.0, 1.0, -1.0]  # logits at columns 10, 11, 20, 30
    p = d1.option_probs(scores, idx[0])  # options: max(10, 11) = 2.0, 20 = 1.0
    assert p == pytest.approx([math.exp(1) / (math.exp(1) + 1), 1 / (math.exp(1) + 1)])
    assert sum(d1.option_probs(scores, idx[1])) == pytest.approx(1.0)


def test_vision_span_shape():
    s = d1.vision_shape(tiles=7, patches=1024, image_tokens=1536)
    assert s["stack"] == "vision" and s["kept"] == 0 and s["rows"] == 7 and s["image_tokens"] == 1536


# --------------------------------------------------------------------------- vl_batch


def test_frame_is_three_by_two():
    assert vl_batch.frame(768) == (768, 512)
    assert vl_batch.frame(1024) == (1024, 683)


def series(points: dict[int, tuple[float, float]]) -> dict[str, float]:
    """{B: (kernel_busy, decisions_per_s)} as vl_batch keys of series `s_`."""
    out: dict[str, float] = {}
    for b, (busy, rate) in points.items():
        out[f"s_b{b}_kernel_busy"], out[f"s_b{b}_decisions_per_s"] = busy, rate
    return out


def test_steady_state_is_the_smallest_busy_batch_not_beaten_by_5pct():
    m = series({1: (0.5, 10), 2: (0.96, 50), 4: (0.97, 60), 8: (0.98, 62)})
    assert vl_batch.steady_state(m, "s_", [1, 2, 4, 8]) == 4  # 2 is beaten by 4 (+20%); 4 is not (+3%)


def test_steady_state_falls_back_to_the_fastest_when_never_busy():
    m = series({1: (0.5, 10), 2: (0.6, 30), 4: (0.7, 20)})
    assert vl_batch.steady_state(m, "s_", [1, 2, 4]) == 2
    out = vl_batch.bstar(m, "s_", [1, 2, 4])
    assert out["s_bstar"] == 2 and out["s_bstar_busy"] == 0
    assert vl_batch.steady_state({}, "s_", [1]) is None


def test_vision_span_is_charged_to_tower_and_projector():
    text, vision = flops.arch(VL), flops.vision_stacks(VL)
    fwd = {**d1.vision_shape(tiles=2, patches=1024, image_tokens=512), "ms": 3.0}
    cost, is_vision = vl_batch.cost_of(fwd, text, vision)
    tower = flops.cost(vision["vision"], flops.Forward(rows=2, tokens=1024, kept=0))
    proj = flops.cost(vision["projector"], flops.Forward(rows=1, tokens=512, kept=0))
    assert is_vision and cost.ms == 3.0
    assert cost.flops == pytest.approx(tower.flops + proj.flops)
    lm, is_vision = vl_batch.cost_of({**d1.tree_shape(10, [5, 5]), "ms": 1.0}, text, vision)
    assert not is_vision and lm.tokens == 20


def test_headline_compares_arms_at_their_steady_state():
    m = {
        "vl_batch_image_p768_bstar_ms_per_decision": 6.0,
        "vl_batch_text_p768_bstar_ms_per_decision": 2.0,
        "vl_batch_none_p0_bstar_ms_per_decision": 1.0,
        "vl_batch_image_p768_bstar_peak_mb_per_decision": 30.0,
        "vl_batch_text_p768_bstar_peak_mb_per_decision": 10.0,
    }
    h = vl_batch.headline(m, 768)
    assert (h["vl_batch_image_text_ms"], h["vl_batch_image_none_ms"], h["vl_batch_image_text_mem"]) == (
        3,
        6,
        3,
    )
    assert math.isnan(vl_batch.headline({}, 768)["vl_batch_image_text_ms"])


def test_text_and_none_arms_keep_the_question_and_replace_the_picture(tmp_path):
    item = imajev.parse(record("boolean", True, track="visual", images=[{"path": "a", "sha256": "0"}]))
    assert item is not None
    src = [vl_batch.Source(item, "filler words here")]
    (text,) = vl_batch.arm_requests("text", src, tmp_path, 384, lambda s: s.upper())
    (none,) = vl_batch.arm_requests("none", src, tmp_path, 384, lambda s: s)
    assert text == ("FILLER WORDS HERE\n\n" + item.row.state, item.row.questions)
    assert none == (item.row.state, item.row.questions)


class FakeVl:
    """prepare/run_batch with recorded spans: a vision span per batch with pictures, one LM span."""

    def __init__(self):
        self.forwards = []
        self.config = VL

    def profile_info(self):
        return {"config": self.config}

    def take_forwards(self):
        out, self.forwards = self.forwards, []
        return out

    def image_tokens(self, width, height):
        return 256

    def decide(self, state, questions, images=None):
        return EchoModel().decide(state, questions)

    def decide_batch(self, requests):
        return self.run_batch(self.prepare_batch(requests))

    def prepare_batch(self, requests):
        pictures = sum(len(r[2]) for r in requests if len(r) > 2)
        tokens = [len(str(r[0]).split()) + 10 + (256 if len(r) > 2 else 0) for r in requests]
        return {"n": len(requests), "pictures": pictures, "tokens": tokens}

    def run_batch(self, p):
        if p["pictures"]:
            self.forwards.append({**d1.vision_shape(2 * p["pictures"], 1024, 256 * p["pictures"]), "ms": 2.0})
        self.forwards.append({**d1.tree_shape(5, [t - 5 for t in p["tokens"]]), "ms": 1.0})
        self.p = p
        return {"decisions": p["n"], "real_tokens": sum(p["tokens"]), "image_tokens": 256 * p["pictures"]}

    def batch_probs(self, p):
        return [[0.8, 0.2]] * p["n"]


class WordFit:
    def __init__(self, path):
        pass

    def fit(self, text, tokens):
        return " ".join(text.split()[:tokens])


def test_run_sweeps_arms_and_sizes_with_a_fake_model(tmp_path, monkeypatch):
    item = imajev.parse(record("boolean", True, track="visual", images=[{"path": "a", "sha256": "0"}]))
    assert item is not None
    srcs = [vl_batch.Source(item, "word " * 400) for _ in range(4)]
    monkeypatch.setattr(imajev, "dataset_dir", lambda: tmp_path)
    monkeypatch.setattr(vl_batch, "sources", lambda settings, root, size: srcs)
    monkeypatch.setattr(vl_batch, "resized", lambda root, it, px: ["picture"])
    monkeypatch.setattr(vl_batch, "FileTokenCounter", WordFit)
    monkeypatch.setattr(vl_batch.batch_sweep, "trace_batches", lambda m, calls: None)
    settings: Settings = {
        "grid_vl_batch_image_px": "384",
        "grid_vl_batch_sizes": "1,4",
        "vl_batch_tokenizer_file": "tok.json",
        "vl_batch_min_decisions": 4,
        "vl_batch_min_batches": 1,
        "vl_batch_check": True,
        "vl_batch_check_rows": 2,
    }
    m = vl_batch.run(FakeVl(), settings)
    assert next(iter(m)) == "vl_batch_image_text_ms"
    assert m["vl_batch_check_max_diff"] == pytest.approx(0.0)  # echo's noul is 0.8, the fake's packed 0.8
    assert m["vl_batch_p384_image_tokens_per_picture"] == 256
    for arm, px in (("image", 384), ("text", 384), ("none", 0)):
        assert f"vl_batch_{arm}_p{px}_b4_ms_per_decision" in m
        assert f"vl_batch_{arm}_p{px}_bstar" in m  # no trace: falls back to the fastest
    assert m["vl_batch_image_p384_b4_vision_share"] == pytest.approx(2 / 3)
    assert m["vl_batch_text_p384_b4_vision_share"] == 0
    assert m["vl_batch_image_p384_b4_image_tokens"] == 4 * 256
    # the text arm's filler is the picture's 256 tokens: same LM length as the image arm
    assert m["vl_batch_text_p384_b4_tokens"] == m["vl_batch_image_p384_b4_tokens"]


def test_close_think_gives_a_reasoning_template_an_empty_thought():
    assert semif.close_think("<|im_start|>assistant\n<think>") == "<|im_start|>assistant\n<think></think>"
    assert semif.close_think("<|im_start|>assistant\n") == "<|im_start|>assistant\n"
    assert catalog.LFM25_2B6.options["semif_close_think"] is True
    assert "semif_close_think" not in catalog.LFM25_VL_3B.options  # its template opens no thought
