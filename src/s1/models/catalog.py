"""Every model the harness knows how to run. Add a spec here once its backend exists."""

from __future__ import annotations

from .spec import ModelSpec

OPENJEV = ModelSpec(
    name="openjev",
    group="reference",
    backend="vllm_shim",
    repo="openjev/openjev",
    license="CC BY-NC 4.0",
    weights_gb=54.0,
    base_repo="Qwen/Qwen3.5-27B",
)

CATALOG: list[ModelSpec] = [OPENJEV]
