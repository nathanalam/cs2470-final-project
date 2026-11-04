"""The `laya` backend: Laya (huggingface.co/convaiinnovations/laya), an encoder with a decision head.

Runs the model repo's own inference code, `rl_agent_api.RLAgent` (with `rl_common.py`), like openjev runs its
shipped helper. Both files are checked against the sha256 we reviewed before import: they read only the
checkpoint directory and make no network calls when given a local encoder directory, which RLAgent always
does. Each question is one encoder row, `[CLS] <type> question: <instructions> [SEP] ([MASK] option)* [SEP]
state [SEP]`, cut to the checkpoint's `max_len` (512 English, 1024 multilingual) by dropping the end of the
state. The head scores each option's [MASK]; temperatures come from the checkpoint's rl_agent_config.json, and
the probabilities are rounded to 4 decimals, as upstream. All of a request's questions run in one batched
pass. Apache-2.0.

Weights: `scripts/download_model.sh convaiinnovations/laya <revision> laya` holds both checkpoints:
    laya_dir   the repo                         default $MODELS_DIR/laya
    checkpoint subdirectory of the checkpoint    from the spec ("" English, "multilingual")
Settings: require_gpu (see backends/gpu.py). Weights stay fp32 under bf16 autocast, as upstream.
"""

from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ...record import Scalar
from ...registry import get_str
from ..spec import ModelSpec
from . import gpu

# sha256 of the shipped inference code at the catalog revision (reviewed 2026-10-30)
CODE_SHA256 = {
    "rl_agent_api.py": "be3b46819c9999c3ef88e0f2ecf6d3ab1cdfed1d9a8b466fc89811e34d44031b",
    "rl_common.py": "8d83611d480c971d640a7b7d3aa2f2219c5e8455e9cc2329fd073681bd8be23e",
}


def import_reviewed(code_dir: Path) -> type:
    """RLAgent from `code_dir`, after checking both files are the reviewed ones."""
    for name, digest in CODE_SHA256.items():
        gpu.check_sha256(code_dir / name, digest)
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    agent = importlib.import_module("rl_agent_api").RLAgent
    if not isinstance(agent, type):
        raise TypeError("rl_agent_api.RLAgent is not a class")
    return agent


@dataclass
class Laya:
    agent: object
    max_state_tokens: int | None = None  # shape_sweep skips longer states
    lock: threading.Lock = field(default_factory=threading.Lock)

    def decide(self, state: str, questions: Mapping[str, object]) -> dict[str, object]:
        with self.lock:
            out = self.agent.system_one(state, dict(questions))  # ty: ignore[unresolved-attribute]
        answers = out["answers"]
        return {str(k): v for k, v in answers.items()}


def build(spec: ModelSpec, settings: dict[str, Scalar]) -> Laya:
    code_dir = gpu.local_dir(
        settings, "laya_dir", "laya", gpu.download_hint(spec.repo, spec.revision, "laya")
    )
    checkpoint = code_dir / get_str(settings, "checkpoint", "")
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"{checkpoint}/model.safetensors missing")
    device = gpu.cuda_device(settings, spec.name)
    agent = import_reviewed(code_dir)(str(checkpoint), device=str(device))
    return Laya(agent, spec.max_state_tokens)


class LayaBackend:
    def build(self, spec: ModelSpec, settings: dict[str, Scalar]) -> Laya:
        return build(spec, settings)
