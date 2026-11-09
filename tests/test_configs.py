"""Every committed experiment config must parse through the runner's own loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from s1 import config

CONFIGS = sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.toml"))


def test_configs_exist() -> None:
    assert CONFIGS


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_loads(path: Path) -> None:
    assert config.load(path)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_launch_table_parses(path: Path) -> None:
    """[launch] is optional; when present it must parse (a `gpu` list included), and it stays out of every
    RunSpec."""
    if config.load_launches(path):
        assert all(
            "orchestrator" not in s.params and "orchestrator" not in s.invariants for s in config.load(path)
        )


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_modal_configs_collect_detailed_gpu_metrics(path: Path) -> None:
    """Every Modal GPU run records the detailed (schema 1) GPU metrics: clocks, throttle, memory, PCIe."""
    launches = config.load_launches(path)
    if not any(r.orchestrator == "modal" and r.gpu is not None for r in launches):
        return
    for spec in config.load(path):
        settings = spec.invariants | spec.params
        assert settings.get("gpu_metrics") == "detailed", path.name
        assert settings.get("gpu_metrics_schema") == 1, path.name
