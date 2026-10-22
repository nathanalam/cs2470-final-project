"""Experiment configs (TOML) and their expansion into individual runs.

benchmarks = ["jev_bench"]
[invariants]   # fixed for every run
[params]       # fixed for every run
[sweep]        # each key is a list; one run per element of the cartesian product, merged into params
"""

from __future__ import annotations

import itertools
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .record import Scalar


@dataclass
class RunSpec:
    invariants: dict[str, Scalar]
    params: dict[str, Scalar]
    benchmarks: list[str]


def _scalars(table: object, where: str) -> dict[str, Scalar]:
    if not isinstance(table, dict):
        raise TypeError(f"[{where}] must be a table")
    out: dict[str, Scalar] = {}
    for k, v in table.items():
        if not isinstance(v, str | int | float | bool):
            raise TypeError(f"[{where}] {k} must be a scalar, got {type(v).__name__}")
        out[str(k)] = v
    return out


def load(path: Path) -> list[RunSpec]:
    raw = tomllib.loads(path.read_text())
    benchmarks = raw.get("benchmarks", [])
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError(f"{path}: `benchmarks` must be a non-empty list")
    invariants = _scalars(raw.get("invariants", {}), "invariants")
    params = _scalars(raw.get("params", {}), "params")

    sweep = raw.get("sweep", {})
    axes: list[tuple[str, list[Scalar]]] = []
    for k, vals in sweep.items():
        if not isinstance(vals, list) or not vals:
            raise ValueError(f"{path}: [sweep] {k} must be a non-empty list")
        axes.append((k, list(_scalars(dict(enumerate(vals)), f"sweep.{k}").values())))

    if "model" not in params and "model" not in sweep:
        raise ValueError(f"{path}: set `model` in [params] or [sweep]")

    combos = itertools.product(*(vals for _, vals in axes))
    return [
        RunSpec(
            invariants=dict(invariants),
            params=params | {k: v for (k, _), v in zip(axes, combo, strict=True)},
            benchmarks=[str(b) for b in benchmarks],
        )
        for combo in combos
    ]
