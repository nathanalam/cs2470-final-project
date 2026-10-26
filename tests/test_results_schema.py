"""Every committed results/*.json must load and carry the fields the report and comparisons rely on."""

from __future__ import annotations

from pathlib import Path

import pytest

from s1.record import RunRecord

RESULTS = Path(__file__).resolve().parent.parent / "results"
RECORDS = sorted(RESULTS.glob("*.json"))


def test_records_exist() -> None:
    assert RECORDS, f"no run records found in {RESULTS}"


@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_record_has_required_fields(path: Path) -> None:
    record = RunRecord.load(path)
    assert record.run_id == path.stem
    assert record.params.get("model") not in (None, ""), "params.model missing"
    assert record.invariants.get("gpu") not in (None, ""), "invariants.gpu missing"
    # Benchmarks only; auxiliary entries like "jeval/nvidia_smi" carry a slash.
    benchmarks = [name for name in record.results if "/" not in name]
    assert benchmarks, "no benchmark in results"
    assert record.meta.get("git_commit") not in (None, ""), "meta.git_commit missing"
