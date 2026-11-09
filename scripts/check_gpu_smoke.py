"""Pass/fail for CI's GPU smoke run (configs/ci_gpu_smoke_modal.toml): `scripts/check_gpu_smoke.py DIR`.

DIR must hold exactly one run record, the one `s1 submit` wrote back. It passes when every one of the 5 real
JEVal questions got a valid answer on a resolved L4 with the detailed GPU keys. Accuracy is not checked.
"""

from __future__ import annotations

import sys
from pathlib import Path

from s1.record import RunRecord

EXPECTED_N = 5
EXPECTED_GPU = "L4"
DETAILED_KEYS = ("gpu0_util_pct", "gpu_clock_sm_mhz")


def problems(rec: RunRecord) -> list[str]:
    """Everything wrong with a smoke record; empty means it passed."""
    out: list[str] = []
    jeval = rec.results.get("jeval", {})
    n, invalid = jeval.get("jeval_n"), jeval.get("jeval_invalid")
    if n != EXPECTED_N:
        out.append(f"jeval_n = {n}, expected {EXPECTED_N}")
    if invalid != 0:
        out.append(f"jeval_invalid = {invalid}, expected 0")
    gpu = str(rec.invariants.get("gpu", ""))
    if EXPECTED_GPU.lower() not in gpu.lower():
        out.append(f"gpu = {gpu!r}, expected an {EXPECTED_GPU}")
    smi = rec.results.get("jeval/nvidia_smi", {})
    missing = [k for k in DETAILED_KEYS if k not in smi]
    if missing:
        out.append(f"jeval/nvidia_smi lacks detailed keys {missing}")
    return out


def main(argv: list[str]) -> int:
    (directory,) = argv
    paths = sorted(Path(directory).glob("*.json"))
    if len(paths) != 1:
        print(f"expected exactly one record in {directory}/, found {len(paths)}")
        return 1
    rec = RunRecord.load(paths[0])
    found = problems(rec)
    jeval = rec.results.get("jeval", {})
    print(f"{rec.run_id}: {rec.params.get('model')} on {rec.invariants.get('gpu')}, {jeval}")
    for p in found:
        print(f"FAIL: {p}")
    if not found:
        print("PASS")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
