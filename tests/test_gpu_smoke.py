"""CI's GPU smoke: the config the workflow submits, and a checker that fails the right records."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from s1 import config
from s1.models import catalog
from s1.record import Metrics, RunRecord

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "ci_gpu_smoke_modal.toml"


def _checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_gpu_smoke", ROOT / "scripts" / "check_gpu_smoke.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


check = _checker()


def _record(n: float = 5, invalid: float = 0, gpu: str = "NVIDIA L4", detailed: bool = True) -> RunRecord:
    smi: Metrics = {"gpu_util_pct": 50.0}
    if detailed:
        smi |= {"gpu0_util_pct": 50.0, "gpu_clock_sm_mhz": 1800.0}
    return RunRecord(
        run_id="20261108-000000-00000000",
        invariants={"gpu": gpu},
        params={"model": "olmoe-1b-7b.zeroshot"},
        results={
            "jeval": {"jeval_accuracy": 0.4, "jeval_n": n, "jeval_invalid": invalid},
            "jeval/nvidia_smi": smi,
        },
    )


def test_smoke_config_is_five_fixed_questions_on_one_l4() -> None:
    (spec,) = config.load(CFG)
    settings = spec.invariants | spec.params
    assert spec.benchmarks == ["jeval"]
    assert settings["jeval_limit"] == check.EXPECTED_N
    assert settings["jeval_seed"] == 21  # the seed whose questions fit OLMoE's 4k context (see the config)
    assert settings["require_gpu"] == check.EXPECTED_GPU
    assert settings["max_prompt_tokens"] == 4096
    (model,) = [m for m in catalog.CATALOG if m.name == settings["model"]]
    assert model.status == "ready"


def test_checker_passes_a_clean_run() -> None:
    assert check.problems(_record()) == []


@pytest.mark.parametrize(
    ("rec", "expected"),
    [
        (_record(n=4), "jeval_n"),
        (_record(invalid=1), "jeval_invalid"),
        (_record(gpu="NVIDIA A10G"), "gpu"),
        (_record(detailed=False), "detailed"),
    ],
)
def test_checker_fails(rec: RunRecord, expected: str) -> None:
    (problem,) = check.problems(rec)
    assert expected in problem


def test_main_wants_exactly_one_record(tmp_path: Path) -> None:
    assert check.main([str(tmp_path)]) == 1
    _record().save(tmp_path)
    assert check.main([str(tmp_path)]) == 0
    _record(invalid=2).save(tmp_path)  # same run id: overwrites, still one record
    assert check.main([str(tmp_path)]) == 1
