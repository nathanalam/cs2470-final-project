"""The steady-state table (src/s1/steady_state.py): every entry recomputed from its records, the lookup and
its setup matching, the opt-in formula, and the runner / batch benchmark wiring."""

import math
import re
from pathlib import Path

import pytest

from s1 import registry, report_batch, report_steady, runner, steady_state
from s1.benchmarks import batch_sweep, jeval
from s1.config import RunSpec
from s1.models.catalog import CATALOG
from s1.record import RESULTS_DIR, RunRecord, Scalar, load_all
from s1.steady_state import PLATEAU, REACHED, TABLE, SteadyStateError, gpu_key, lookup, resolve
from tests.test_batch_sweep import FakeBatched
from tests.test_roofline import _no_gpu, row

DOC = Path(__file__).resolve().parents[1] / "docs" / "steady_state_batch.md"
BACKENDS_DIR = Path(steady_state.__file__).parent / "models" / "backends"


def _load(run_id: str) -> RunRecord:
    return RunRecord.load(RESULTS_DIR / f"{run_id}.json")


def _spec(model: str, *, grid: dict[str, list[Scalar]] | None = None, **params: Scalar) -> RunSpec:
    return RunSpec(
        invariants={"gpu": "auto"},
        params={"model": model, **params},
        benchmarks=["batch"],
        grid=grid if grid is not None else {"batch_state_tokens": [128, 1024], "batch_sizes": ["steady"]},
    )


# ---------------------------------------------------------------- the table matches its evidence

BATCH = [e for e in TABLE if e.knob == "batch"]
CONC = [e for e in TABLE if e.knob == "concurrency"]


@pytest.mark.parametrize("e", BATCH, ids=lambda e: f"{e.model}-{e.gpu}-{dict(e.variant)}")
def test_batch_entry_recomputes_from_its_record(e):
    (run,) = e.runs
    rec = _load(run)
    assert str(rec.params["model"]) == e.model
    assert gpu_key(str(rec.invariants["gpu"])) == e.gpu
    for k, v in e.variant.items():  # the record ran the setup the entry claims
        assert str(rec.params.get(k, steady_state.VARIANT_DEFAULTS[k])) == str(v)
    s = report_batch.series_of(rec)
    assert s is not None
    for n, b in e.at.items():
        st = report_steady.steady(s.grid[n], n)
        assert st is not None
        if e.status[n] == REACHED:
            assert st.b_star == b, (n, st)
        else:
            assert e.status[n] == PLATEAU and st.b_star is None and st.plateau == b, (n, st)


@pytest.mark.parametrize("e", CONC, ids=lambda e: e.gpu)
def test_concurrency_entry_recomputes_from_its_records(e):
    curves = report_steady.openjev_curves([_load(r) for r in e.runs])
    (cv,) = curves
    assert gpu_key(cv.gpu) == e.gpu
    assert cv.steady is not None and cv.steady.concurrency == e.at[0]
    assert {p.run_id for p in cv.points} == set(e.runs)


def test_entries_name_real_models_that_can_batch():
    specs = {s.name: s for s in CATALOG}
    for e in TABLE:
        spec = specs[e.model]
        assert spec.backend is not None
        if e.knob == "concurrency":
            assert spec.backend in steady_state.SERVER_BATCHED
        else:
            src = (BACKENDS_DIR / f"{spec.backend}.py").read_text()
            assert "def decide_batch" in src, e.model
        assert e.gpu in steady_state.GPUS
        assert set(e.at) == set(e.status)


def test_no_batch_models_have_no_decide_batch():
    for prefix in steady_state.NO_BATCH:
        spec = next(s for s in CATALOG if s.name.startswith(prefix))
        assert "def decide_batch" not in (BACKENDS_DIR / f"{spec.backend}.py").read_text()


def test_every_recorded_gpu_has_a_key():
    names = {str(r.invariants.get("gpu")) for r in load_all()} - {"cpu", "None", "test", "auto"}
    for name in names:
        assert gpu_key(name) in steady_state.GPUS
    assert gpu_key("NVIDIA A100-SXM4-80GB") != gpu_key("NVIDIA A100 80GB PCIe")
    assert gpu_key("NVIDIA L40S") == "L40S" and gpu_key("NVIDIA L4") == "L4"
    assert gpu_key("NVIDIA A10G,NVIDIA A10G") == "A10G"
    with pytest.raises(SteadyStateError):
        gpu_key("NVIDIA L4,NVIDIA H100 80GB HBM3")
    with pytest.raises(SteadyStateError):
        gpu_key("cpu")


def test_doc_embeds_the_current_tables():
    text = DOC.read_text()
    assert steady_state.summary_table() in text and steady_state.markdown_table() in text


# ---------------------------------------------------------------- lookup


def test_lookup_prefers_reached_and_respects_explicit_setup():
    e = lookup("lfm2-8b-a1b.zeroshot", "H100", {})
    assert e is not None and e.variant["batch_host_prep"] == "overlapped" and e.at == {128: 32, 1024: 16}
    pretok = lookup("lfm2-8b-a1b.zeroshot", "H100", {"batch_host_prep": "pretokenized"})
    assert pretok is not None and pretok.runs == ("20261119-230040-acb6fcff",)
    inline = lookup("lfm2-8b-a1b.zeroshot", "H100", {"batch_host_prep": "inline"})
    assert inline is not None and set(inline.status.values()) == {PLATEAU}
    assert lookup("lfm2-8b-a1b.zeroshot", "H100", {"semif_experts_implementation": "eager"}) is None
    assert lookup("qwen3-8b.zeroshot", "L40S", {}) is None
    graphs = lookup("laya.english", "L4", {"laya_mode": "graphs"})
    assert graphs is not None and graphs.at == {128: 4, 1024: 4}


def test_nearest_length_within_a_factor_of_two():
    e = lookup("qwen3-8b.zeroshot", "L4", {})
    assert e is not None
    assert steady_state.nearest(e, 64) == 128 and steady_state.nearest(e, 256) == 128
    assert steady_state.nearest(e, 512) == 1024 and steady_state.nearest(e, 2048) == 1024
    assert steady_state.nearest(e, 4096) is None and steady_state.nearest(e, 32) is None


# ---------------------------------------------------------------- resolve


def test_resolve_table_pick_and_record_fields():
    out = resolve(_spec("lfm2-8b-a1b.zeroshot"), "NVIDIA H100 80GB HBM3")
    assert not out.pending
    assert out.grid == {"batch_sizes": [16, 32]}
    p = out.params
    assert p["batch_sizes_at"] == "128:32,1024:16" and p["steady_bracket"] is False
    assert p["batch_host_prep"] == "overlapped"  # the entry's setup is applied
    assert "semif_experts_implementation" not in p  # defaults are not written
    assert p["steady_source"] == "table" and p["steady_status"] == "128:reached,1024:reached"
    assert p["steady_runs"] == "20261122-175440-dc068906" and p["steady_gpu"] == "H100"
    assert p["steady_rule"] == steady_state.RULE and p["steady_policy"] == "table"


def test_resolve_bracket_runs_half_pick_and_double():
    grid: dict[str, list[Scalar]] = {"batch_state_tokens": [128], "batch_sizes": ["steady-bracket"]}
    out = resolve(_spec("qwen3-8b.zeroshot", grid=grid), "NVIDIA L4")
    assert out.grid == {"batch_sizes": [2, 4, 8]} and out.params["steady_bracket"] is True
    assert steady_state.sizes_for(out.params, 128, [2, 4, 8]) == [2, 4, 8]
    assert steady_state.sizes_for({}, 128, [1, 2]) == [1, 2]


def test_resolve_without_steady_changes_nothing():
    out = resolve(_spec("qwen3-8b.zeroshot", grid={"batch_sizes": [1, 2]}), "NVIDIA L4")
    assert out.params == {} and out.grid == {} and not out.pending


def test_missing_pair_fails_by_default_and_waits_for_the_model_when_opted_in():
    with pytest.raises(SteadyStateError, match=r"no measured steady-state batch.*L40S"):
        resolve(_spec("qwen3-8b.zeroshot"), "NVIDIA L40S")
    with pytest.raises(SteadyStateError, match="no measured"):  # an explicit setup no entry was measured with
        resolve(_spec("lfm2-8b-a1b.zeroshot", semif_experts_implementation="eager"), "NVIDIA H100 80GB HBM3")
    opted = resolve(_spec("qwen3-8b.zeroshot", steady_policy="table+formula"), "NVIDIA L40S")
    assert opted.pending and opted.params == {}


def test_table_wins_over_the_formula_even_when_opted_in():
    out = resolve(_spec("qwen3-8b.zeroshot", steady_policy="table+formula"), "NVIDIA L4", FakeBatched())
    assert out.params["batch_sizes_at"] == "128:4,1024:1" and out.params["steady_source"] == "table"


def test_formula_after_load_is_marked_predicted():
    spec = _spec("qwen3-8b.zeroshot", steady_policy="table+formula")
    out = resolve(spec, "NVIDIA L40S", FakeBatched())
    at = steady_state.parse_sizes_at(str(out.params["batch_sizes_at"]))
    assert at == {128: 8, 1024: 2}  # the design doc's L40S dense 8-9B default
    assert out.params["steady_source"] == "formula"
    assert out.params["steady_status"] == "128:predicted,1024:predicted"


def test_concurrency_only_for_server_batched_backends():
    spec = RunSpec({"gpu": "auto"}, {"model": "openjev", "jeval_concurrency": "steady"}, ["jeval"], {})
    out = resolve(spec, "NVIDIA B200")
    assert out.params["jeval_concurrency"] == 8 and out.params["steady_status"] == "0:reached"
    semif = RunSpec({}, {"model": "qwen3-8b.zeroshot", "jeval_concurrency": "steady"}, ["jeval"], {})
    with pytest.raises(SteadyStateError, match="server-batched"):
        resolve(semif, "NVIDIA L4")
    with pytest.raises(SteadyStateError, match="no measured steady-state concurrency"):
        resolve(spec, "NVIDIA L4")


@pytest.mark.parametrize(
    ("grid", "params", "match"),
    [
        ({"batch_state_tokens": [128], "batch_sizes": ["steady", 4]}, {}, "alone"),
        ({"batch_sizes": ["steady"]}, {}, "batch_state_tokens"),
        (
            {"batch_state_tokens": [128], "batch_sizes": ["steady"]},
            {"steady_policy": "guess"},
            "steady_policy",
        ),
        ({}, {"jeval_concurrency": "auto"}, "integer or"),
    ],
)
def test_malformed_requests(grid, params, match):
    with pytest.raises(SteadyStateError, match=match):
        resolve(_spec("qwen3-8b.zeroshot", grid=grid, **params), "NVIDIA L4")


def test_no_batch_models_refuse_steady():
    with pytest.raises(SteadyStateError, match="no batched path"):
        resolve(_spec("elephant-1"), "NVIDIA L4")
    with pytest.raises(SteadyStateError, match="no measured"):  # d1 batches (#18) but has no entry yet
        resolve(_spec("d1-3b.eager"), "NVIDIA L4")


# ---------------------------------------------------------------- formula


def test_formula_properties():
    for gpu in steady_state.GPUS:
        for active in (1.5e9, 3.5e9, 8e9, 27e9):
            picks = [
                steady_state.formula(gpu, n, active_params=active, weights_gb=10) for n in (64, 128, 1024)
            ]
            assert all(b >= 1 and b & (b - 1) == 0 for b in picks)
            assert picks == sorted(picks, reverse=True)
            small = steady_state.formula(gpu, 128, active_params=active / 4, weights_gb=10)
            assert small >= picks[1]
    assert steady_state.formula("L4", 1024, active_params=1.5e9, weights_gb=21) == 1  # memory cap
    assert steady_state.bracket(1) == [1, 2] and steady_state.bracket(8) == [4, 8, 16]


def test_formula_backtest_against_measured_series():
    """The opt-in formula reaches >= 80% of the throughput at the measured pick on every non-Laya entry
    (31 of 36 series are >= 95%; the design doc's back-test). Picks beyond the largest batch that ran are
    clamped to it."""
    weights = {s.name: s.weights_gb or 0.0 for s in CATALOG}
    worst = math.inf
    for e in BATCH:
        if e.model.startswith("laya"):
            continue
        s = report_batch.series_of(_load(e.runs[0]))
        assert s is not None
        for n, ref in e.at.items():
            pts = s.grid[n]
            ran = max(b for b, m in pts.items() if m.get("oom", 0) == 0)
            active = pts[ref]["gflop"] / pts[ref]["tokens"] / 2 * 1e9
            b = min(steady_state.formula(e.gpu, n, active_params=active, weights_gb=weights[e.model]), ran)
            assert pts[b].get("oom", 0) == 0
            worst = min(worst, pts[b]["decisions_per_s"] / pts[ref]["decisions_per_s"])
    assert worst >= 0.8


# ---------------------------------------------------------------- runner and benchmark wiring


def _run(
    tmp_path, monkeypatch, model, *, gpu="NVIDIA L4", name="qwen3-8b.zeroshot", sizes="steady", **params
):
    rows = [row(i, state_words=300 + i) for i in range(20)]
    monkeypatch.setattr(jeval, "load_rows", lambda: rows)
    monkeypatch.setattr(runner, "detect_gpu", lambda: gpu)
    from s1 import profiling  # noqa: PLC0415

    monkeypatch.setattr(profiling, "trace_activity", lambda fn: (fn(), [("k", "kernel", 500.0)])[1])
    monkeypatch.setitem(registry.MODELS, name, lambda settings: model)
    spec = RunSpec(
        benchmarks=["batch"],
        invariants={"gpu": "auto"},
        params={"model": name, "batch_pool": 16, "batch_min_decisions": 8, **params},
        grid={batch_sweep.LENGTHS: [64, 128], batch_sweep.SIZES: [sizes]},
    )
    return runner.run(spec, results_dir=tmp_path, log=lambda _: None, gpu_sampler=_no_gpu)


def test_runner_runs_only_the_steady_point(tmp_path, monkeypatch):
    rec = _run(tmp_path, monkeypatch, FakeBatched())
    m = rec.results["batch"]
    assert rec.params["batch_sizes_at"] == "64:4,128:4" and rec.grid[batch_sweep.SIZES] == [4]
    assert rec.params["steady_runs"] == "20261109-231713-a38ec98c"
    assert {int(h[2]) for k in m if (h := re.match(r"batch_s(\d+)_b(\d+)_", k))} == {4}
    assert "batch_s64_b4_peak_mb" not in m  # no CUDA under the fake


def test_runner_fails_before_loading_the_model(tmp_path, monkeypatch):
    def boom(settings):
        raise AssertionError("model must not load")

    monkeypatch.setattr(runner, "detect_gpu", lambda: "NVIDIA L40S")
    monkeypatch.setitem(registry.MODELS, "qwen3-8b.zeroshot", boom)
    spec = RunSpec(
        {"gpu": "auto"},
        {"model": "qwen3-8b.zeroshot"},
        ["batch"],
        {batch_sweep.LENGTHS: [128], batch_sweep.SIZES: ["steady"]},
    )
    with pytest.raises(SteadyStateError, match="L40S"):
        runner.run(spec, results_dir=tmp_path, log=lambda _: None, gpu_sampler=_no_gpu)


def test_runner_formula_after_load(tmp_path, monkeypatch):
    rec = _run(tmp_path, monkeypatch, FakeBatched(), gpu="NVIDIA L40S", steady_policy="table+formula")
    at = steady_state.parse_sizes_at(str(rec.params["batch_sizes_at"]))
    assert rec.params["steady_source"] == "formula"
    assert all(f"batch_s{n}_b{b}_wall_ms" in rec.results["batch"] for n, b in at.items())


def test_steady_pick_that_runs_out_of_memory_is_halved(tmp_path, monkeypatch):
    rec = _run(tmp_path, monkeypatch, FakeBatched(oom_at=4))  # the L4 pick is 4 at both lengths
    m = rec.results["batch"]
    for n in (64, 128):
        assert m[f"batch_s{n}_b4_oom"] == 1 and m[f"batch_s{n}_steady_fallback"] == 2
        assert m[f"batch_s{n}_b2_wall_ms"] > 0
    assert "B=4 out of memory; ran B=2" in str(rec.meta["batch_s64_steady_fallback"])


def test_nothing_fits_records_zero(tmp_path, monkeypatch):
    m = _run(tmp_path, monkeypatch, FakeBatched(oom_at=1)).results["batch"]
    assert m["batch_s64_steady_fallback"] == 0 and m["batch_s64_b1_oom"] == 1


def test_bracket_and_explicit_sweeps_keep_the_oom_point(tmp_path, monkeypatch):
    m = _run(tmp_path, monkeypatch, FakeBatched(oom_at=8), sizes="steady-bracket").results["batch"]
    assert m["batch_s64_b8_oom"] == 1 and "batch_s64_steady_fallback" not in m
    assert m["batch_s64_b4_wall_ms"] > 0 and m["batch_s64_b2_wall_ms"] > 0
    assert not steady_state.halves_on_oom({})
    assert steady_state.fallback_sizes(16) == [8, 4, 2, 1] and steady_state.fallback_sizes(1) == []
