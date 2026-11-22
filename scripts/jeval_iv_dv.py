"""Write analysis/jeval_iv_dv.csv: one row per JEVal IV level (model x GPU x platform x GPU count x
concurrency), every measured DV, from results/*.json.

A level's row is its largest run (most questions), the latest among equals, so a 2,000-question run stands
for its level over a 200-question smoke run. `platform` is "cluster" (HUIT Slurm, records without
`invariants.platform`) or "modal"; the same GPU on two platforms is two levels, never pooled.

    uv run python scripts/jeval_iv_dv.py            # rewrites analysis/jeval_iv_dv.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from s1.record import RunRecord, load_all  # noqa: E402

OUT = ROOT / "analysis" / "jeval_iv_dv.csv"
# Board power limits for records that predate meta.gpu_power_limit_w (W).
POWER_LIMIT_W = {"A10G": 300.0, "L4": 72.0}
GPU_SHORT = {"H100 80GB HBM3": "H100", "A100-SXM4-80GB": "A100-80GB"}
COLUMNS = [
    "iv_level", "model", "gpu", "platform", "gpu_count", "concurrency", "warmup", "n", "invalid",
    "accuracy", "choice_accuracy", "noul_accuracy", "score_accuracy", "score_mae", "ece", "brier",
    "latency_ms_mean", "latency_ms_std", "latency_ms_p90", "latency_ms_p95", "latency_ms_p99",
    "throughput_q_per_s", "gpu_util_pct", "gpu_mem_peak_gib_per_gpu", "gpu_mem_total_gib_per_gpu",
    "power_w_per_gpu", "power_pct_of_limit", "power_w_total", "energy_wh", "energy_j_per_question",
    "temp_c_max", "run_id", "git_commit",
]  # fmt: skip


def concurrency(r: RunRecord) -> int:
    """Requests in flight: the run's setting, else jeval's default (8), as the benchmark used it."""
    v = r.params.get("jeval_concurrency", r.invariants.get("jeval_concurrency", 8))
    return int(v) if isinstance(v, int | float) else 8


def gpu_short(name: str) -> str:
    name = name.removeprefix("NVIDIA ").strip()
    return GPU_SHORT.get(name, name)


def platform(r: RunRecord) -> str:
    return str(r.invariants.get("platform", "cluster")).lower()


def _r(v: object, digits: int) -> str:
    return "" if not isinstance(v, int | float) else f"{round(float(v), digits)}"


def row(r: RunRecord) -> dict[str, str]:
    m, g = r.results["jeval"], r.results.get("jeval/nvidia_smi", {})
    gpu, plat = gpu_short(str(r.invariants.get("gpu", "?"))), platform(r)
    count = int(g.get("gpu_count", 1))
    conc = concurrency(r)
    model = str(r.params.get("model", "?"))
    n = m.get("jeval_n", 0.0)
    total_w = g.get("gpu_power_w_total_mean")
    per_w = total_w / count if isinstance(total_w, float) else None
    limit = r.meta.get("gpu_power_limit_w", POWER_LIMIT_W.get(gpu))
    energy = g.get("gpu_energy_wh")
    where = f"{count}x {gpu}" + ("" if plat == "cluster" else f" ({plat.capitalize()})")
    return {
        "iv_level": f"{model} / {where} / {conc} in flight",
        "model": model,
        "gpu": gpu,
        "platform": plat,
        "gpu_count": str(count),
        "concurrency": str(conc),
        "warmup": str(bool(r.params.get("jeval_warmup", False))),
        "n": str(int(n)),
        "invalid": str(int(m.get("jeval_invalid", 0))),
        "accuracy": _r(m.get("jeval_accuracy"), 4),
        "choice_accuracy": _r(m.get("jeval_choice_accuracy"), 4),
        "noul_accuracy": _r(m.get("jeval_noul_accuracy"), 4),
        "score_accuracy": _r(m.get("jeval_score_accuracy"), 4),
        "score_mae": _r(m.get("jeval_score_mae"), 4),
        "ece": _r(m.get("jeval_ece"), 4),
        "brier": _r(m.get("jeval_brier"), 4),
        "latency_ms_mean": _r(m.get("jeval_latency_ms"), 1),
        "latency_ms_std": _r(m.get("jeval_latency_ms_std"), 1),
        "latency_ms_p90": _r(m.get("jeval_latency_ms_p90"), 1),
        "latency_ms_p95": _r(m.get("jeval_latency_ms_p95"), 1),
        "latency_ms_p99": _r(m.get("jeval_latency_ms_p99"), 1),
        "throughput_q_per_s": _r(m.get("jeval_questions_per_s"), 3),
        "gpu_util_pct": _r(g.get("gpu_util_pct"), 1),
        "gpu_mem_peak_gib_per_gpu": _r(
            g["gpu_mem_used_mb_max"] / 1024 if "gpu_mem_used_mb_max" in g else None, 2
        ),
        "gpu_mem_total_gib_per_gpu": _r(g["gpu_mem_total_mb"] / 1024 if "gpu_mem_total_mb" in g else None, 2),
        "power_w_per_gpu": _r(per_w, 1),
        "power_pct_of_limit": _r(
            100 * per_w / limit if per_w is not None and isinstance(limit, int | float) and limit else None, 1
        ),
        "power_w_total": _r(total_w, 1),
        "energy_wh": _r(energy, 2),
        "energy_j_per_question": _r(energy * 3600 / n if isinstance(energy, float) and n else None, 1),
        "temp_c_max": _r(g.get("gpu_temp_c_max"), 1),
        "run_id": r.run_id,
        "git_commit": str(r.meta.get("git_commit", "")),
    }


def levels(recs: list[RunRecord]) -> list[dict[str, str]]:
    best: dict[tuple[str, str, str, int, int], RunRecord] = {}
    for r in sorted((r for r in recs if "jeval" in r.results), key=lambda r: r.run_id):
        key = (
            str(r.params.get("model", "?")),
            gpu_short(str(r.invariants.get("gpu", "?"))),
            platform(r),
            int(r.results.get("jeval/nvidia_smi", {}).get("gpu_count", 1)),
            concurrency(r),
        )
        cur = best.get(key)
        if cur is None or r.results["jeval"].get("jeval_n", 0) >= cur.results["jeval"].get("jeval_n", 0):
            best[key] = r
    return [row(r) for _, r in sorted(best.items())]


def main() -> None:
    rows = levels(load_all(ROOT / "results"))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} levels")


if __name__ == "__main__":
    main()
