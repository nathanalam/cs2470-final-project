import math
import sys
from pathlib import Path
from typing import ClassVar

import pytest

from s1 import models, registry
from s1.config import RunSpec
from s1.models import make_factory, resolve_settings
from s1.models.answers import choice_answer, noul_answer, noul_from_logits, score_answer, softmax
from s1.models.backends import BACKENDS, load_backend
from s1.models.backends import vllm_shim as vs
from s1.models.catalog import CATALOG, OPENJEV
from s1.models.spec import ModelSpec
from s1.runner import run
from tests.conftest import EchoModel


def test_catalog_names_unique_and_registered():
    names = [s.name for s in CATALOG]
    assert len(names) == len(set(names))
    for s in CATALOG:
        assert registry.get_model(s.name)  # populates registries on first call


def test_every_spec_backend_resolvable_without_heavy_import():
    for s in CATALOG:
        assert s.backend in BACKENDS, s.name
        assert s.weights_gb > 0, s.name
    assert "torch" not in sys.modules
    assert "vllm" not in sys.modules


def test_load_backend_unknown():
    with pytest.raises(KeyError, match="unknown backend"):
        load_backend("nope")


def test_variant_diff_is_exactly_the_changes():
    base = ModelSpec(
        name="kev-4b.hybrid", group="D-hybrid", backend="b", repo="r", license="mit", weights_gb=8,
        options={"mode": "hybrid", "temperature": 1.0},
    )  # fmt: skip
    v = base.variant("attn", group="D-attn", options={"mode": "attn"})
    assert v.name == "kev-4b.attn"
    diff = {k for k in vars(base) if getattr(base, k) != getattr(v, k)}
    assert diff == {"name", "group", "options"}
    assert dict(v.options) == {"mode": "attn", "temperature": 1.0}
    with pytest.raises(ValueError, match="name"):
        base.variant("x", name="y")


def test_spec_is_frozen():
    with pytest.raises(AttributeError):
        OPENJEV.name = "x"  # ty: ignore[invalid-assignment]


def test_duplicate_registration_raises():
    registry.get_model("openjev")
    with pytest.raises(ValueError, match="already registered"):
        registry.model("openjev")(lambda settings: object())
    with pytest.raises(ValueError, match="already registered"):
        registry.register_model("openjev", lambda settings: object())


def test_resolve_settings_precedence():
    spec = ModelSpec(
        name="m", group="control", backend="b", repo="r", revision="abc", license="mit", weights_gb=1,
        options={"temperature": 0.5, "mode": "x"},
    )  # fmt: skip
    out = resolve_settings(spec, {"temperature": 2.0, "n": 3})
    assert out["temperature"] == 2.0  # run settings beat spec options
    assert out["mode"] == "x"
    assert out["spec_revision"] == "abc"
    assert "spec_base_revision" not in out  # None omitted
    assert out["n"] == 3


class FakeBackend:
    seen: ClassVar[dict[str, object]] = {}

    def build(self, spec, settings):
        FakeBackend.seen = dict(settings)
        return EchoModel()


def test_fake_backend_end_to_end_through_runner(tmp_path, monkeypatch):
    monkeypatch.setitem(BACKENDS, "fake", f"{__name__}:FakeBackend")
    spec = ModelSpec(
        name="fake.v1", group="control", backend="fake", repo="r", revision="rev1", license="mit",
        weights_gb=1, options={"temperature": 0.5},
    )  # fmt: skip
    monkeypatch.setitem(registry.MODELS, spec.name, make_factory(spec))

    @registry.benchmark("probe_bench")
    def probe(model, settings):
        return {"probe_x": 1.0}

    try:
        rs = RunSpec(
            invariants={"gpu": "cpu"},
            params={"model": "fake.v1", "temperature": 0.9},
            benchmarks=["probe_bench"],
        )
        rec = run(rs, results_dir=tmp_path, log=lambda _: None)
    finally:
        registry.BENCHMARKS.pop("probe_bench", None)
    assert rec.results["probe_bench"] == {"probe_x": 1.0}
    assert FakeBackend.seen["temperature"] == 0.9
    assert FakeBackend.seen["spec_revision"] == "rev1"


# --- openjev: same settings and behavior as before the refactor ------------------------------


class FakeProc:
    def __init__(self, cmd, **kw):
        self.args, self.pid, self.returncode = cmd, 1, None

    def poll(self):
        return None


def test_openjev_endpoint_shortcut():
    jev = registry.get_model("openjev")({"model": "openjev", "openjev_endpoint": "http://h:1/"})
    assert isinstance(jev, vs.OpenJev)
    assert jev.endpoint == "http://h:1"


def test_openjev_launch_flags_unchanged(tmp_path, monkeypatch):
    (tmp_path / "helper").mkdir()
    (tmp_path / "helper" / "shim.py").write_text("")
    spawned = []

    def fake_spawn(cmd, log, env=None):
        spawned.append((cmd, env))
        return FakeProc(cmd)

    monkeypatch.setattr(vs, "_spawn", fake_spawn)
    monkeypatch.setattr(vs, "_ready", lambda url: True)
    monkeypatch.setattr(vs, "LOG_DIR", tmp_path / "logs")
    jev = registry.get_model("openjev")(
        {"model": "openjev", "openjev_dir": str(tmp_path), "tensor_parallel_size": 4, "quantization": "none"}
    )
    (vllm_cmd, _), (shim_cmd, shim_env) = spawned
    assert vllm_cmd[1:3] == ["serve", str(tmp_path)]
    assert "--quantization" not in vllm_cmd
    assert vllm_cmd[vllm_cmd.index("--tensor-parallel-size") + 1] == "4"
    assert vllm_cmd[vllm_cmd.index("--max-model-len") + 1] == "16384"
    assert vllm_cmd[vllm_cmd.index("--served-model-name") + 1] == "qwen"
    assert shim_cmd[:2] == [sys.executable, str(Path(tmp_path) / "helper" / "shim.py")]
    assert shim_env is not None
    assert all(shim_env[k] == v for k, v in vs.SHIM_ENV.items())
    assert shim_env["TOKENIZER"] == str(tmp_path)
    assert isinstance(jev, vs.OpenJev)
    assert len(jev.procs) == 2


def test_openjev_default_quantization_is_fp8(tmp_path, monkeypatch):
    (tmp_path / "helper").mkdir()
    (tmp_path / "helper" / "shim.py").write_text("")
    spawned = []
    monkeypatch.setattr(vs, "_spawn", lambda cmd, log, env=None: spawned.append(cmd) or FakeProc(cmd))
    monkeypatch.setattr(vs, "_ready", lambda url: True)
    registry.get_model("openjev")({"model": "openjev", "openjev_dir": str(tmp_path)})
    assert spawned[0][spawned[0].index("--quantization") + 1] == "fp8"


# --- answers ----------------------------------------------------------------------------------


def test_softmax():
    p = softmax([0.0, 0.0, math.log(2)])
    assert p == pytest.approx([0.25, 0.25, 0.5])
    assert sum(softmax([1000.0, 999.0])) == pytest.approx(1)
    flat = softmax([1.0, 0.0], temperature=1000)
    assert flat[0] == pytest.approx(0.5, abs=1e-3)
    for bad in ([], [math.nan], [math.inf, 0.0]):
        with pytest.raises(ValueError, match="score"):
            softmax(bad)
    for t in (0, -1, math.nan):
        with pytest.raises(ValueError, match="temperature"):
            softmax([1.0], t)


def test_choice_answer():
    a = choice_answer({"a": 0.0, "b": 2.0, "c": 1.0})
    assert a["type"] == "choice"
    assert a["choice"] == "b"
    probs = a["probabilities"]
    assert isinstance(probs, dict)
    assert list(probs) == ["a", "b", "c"]
    assert sum(probs.values()) == pytest.approx(1)
    assert choice_answer({"x": 1.0, "y": 1.0})["choice"] == "x"  # tie -> first


def test_noul_answers():
    assert noul_answer(0.8) == {"type": "noul", "noul": 0.8}
    for bad in (-0.1, 1.1, math.nan):
        with pytest.raises(ValueError, match="P"):
            noul_answer(bad)
    a = noul_from_logits(1.0, 1.0)
    assert a["noul"] == pytest.approx(0.5)
    assert noul_from_logits(0.0, 0.0, bias=math.log(3))["noul"] == pytest.approx(0.75)


def test_score_answer():
    a = score_answer([0.0, 0.0, 0.0])
    assert a["type"] == "score"
    assert a["score"] == pytest.approx(1.0)
    assert a["probabilities"] == pytest.approx({"0": 1 / 3, "1": 1 / 3, "2": 1 / 3})
    assert score_answer([50.0, 0.0])["score"] == pytest.approx(0.0, abs=1e-9)


def test_runner_records_resolved_spec_values(tmp_path, monkeypatch):
    spec = ModelSpec(
        name="fake.rec", group="control", backend="fake", repo="r", revision="rev1", base_revision="b1",
        license="mit", weights_gb=1, options={"temperature": 0.5, "mode": "x"},
    )  # fmt: skip
    monkeypatch.setitem(BACKENDS, "fake", f"{__name__}:FakeBackend")
    monkeypatch.setitem(registry.MODELS, spec.name, make_factory(spec))
    monkeypatch.setattr(models, "CATALOG", [*CATALOG, spec])
    rs = RunSpec(invariants={"gpu": "cpu"}, params={"model": "fake.rec", "temperature": 0.9}, benchmarks=[])
    rec = run(rs, results_dir=tmp_path, log=lambda _: None)
    assert rec.params["spec_revision"] == "rev1"
    assert rec.params["spec_base_revision"] == "b1"
    assert rec.params["mode"] == "x"
    assert rec.params["temperature"] == 0.9  # config value is not overwritten
    assert "spec_revision" not in rec.invariants


def test_runner_leaves_uncataloged_models_alone(tmp_path, echo_model):
    rs = RunSpec(invariants={"gpu": "cpu"}, params={"model": "echo"}, benchmarks=[])
    assert run(rs, results_dir=tmp_path, log=lambda _: None).params == {"model": "echo"}
