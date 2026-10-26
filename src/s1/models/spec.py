"""ModelSpec: the data that identifies a model, separate from the code that serves it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ..record import Scalar

type Group = Literal["E", "D-hybrid", "D-attn", "ED", "MoE", "control", "MoE-Diff", "reference"]


@dataclass(frozen=True)
class ModelSpec:
    """One benchmarkable model.

    `name` is `model.variant` (e.g. `kev-4b.hybrid`); a bare name like `openjev` is allowed.
    `backend` keys into `s1.models.backends.BACKENDS`. `revision=None` means unpinned: it is
    resolved at download time and recorded. `options` are backend knobs (question_mode, mode,
    temperature, ...); they become settings, so configs can override them.
    """

    name: str
    group: Group
    backend: str
    repo: str
    license: str
    weights_gb: float
    revision: str | None = None
    base_repo: str | None = None
    base_revision: str | None = None
    tokenizer: str | None = None
    max_state_tokens: int | None = None
    options: Mapping[str, Scalar] = field(default_factory=dict)

    def variant(self, tag: str, **changes: Any) -> ModelSpec:  # noqa: ANN401 — field values vary by field
        """A sibling spec named `<model>.<tag>` differing from this one only in `changes`.

        `options` changes are merged into the existing options, so a controlled pair is an
        explicit diff of the knobs that differ.
        """
        if "name" in changes:
            raise ValueError("variant() sets the name from `tag`")
        if "options" in changes:
            changes["options"] = {**self.options, **changes["options"]}
        return replace(self, name=f"{self.name.split('.', 1)[0]}.{tag}", **changes)

    def settings(self) -> dict[str, Scalar]:
        """Scalar view of the spec, to merge under the run's invariants/params (those win).

        Fields are namespaced `spec_*`; `options` keys are used as-is. None fields are omitted.
        """
        fields: dict[str, Scalar | None] = {
            "spec_group": self.group,
            "spec_repo": self.repo,
            "spec_revision": self.revision,
            "spec_base_repo": self.base_repo,
            "spec_base_revision": self.base_revision,
            "spec_license": self.license,
            "spec_weights_gb": self.weights_gb,
            "spec_tokenizer": self.tokenizer,
            "spec_max_state_tokens": self.max_state_tokens,
        }
        return {k: v for k, v in fields.items() if v is not None} | dict(self.options)
