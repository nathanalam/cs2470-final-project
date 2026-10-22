from s1 import report
from s1.record import RunRecord
from s1.report import DASH, TIMES


def _rec(run_id, model, acc, n, util=None):
    results = {"jeval": {"jeval_accuracy": acc, "jeval_ece": 0.05, "jeval_n": n, "jeval_latency_ms": 1500.0}}
    if util is not None:
        results["jeval/nvidia_smi"] = {"gpu_util_pct": util, "gpu_count": 4.0, "gpu_energy_wh": 36.2}
    return RunRecord(
        run_id,
        {"gpu": "NVIDIA A10G", "quantization": "none"},
        {"model": model, "jeval_limit": n},
        results,
        {"git_commit": "abc123"},
    )


RECS = [
    _rec("20260101-000000-aaaa", "openjev", 0.70, 200, util=97.0),
    _rec("20260102-000000-bbbb", "openjev", 0.77, 2000, util=99.9),
    _rec("20260103-000000-cccc", "otherjev", 0.81, 500),
]


def test_leaderboard_takes_each_models_largest_run_sorted_by_headline():
    t = report.leaderboard("jeval", RECS)
    assert t.header[:3] == ["Model", "Accuracy", "ECE"]
    assert [row[0] for row in t.rows] == ["otherjev", "openjev"]
    assert t.rows[1][1] == "77.0%"  # the 2,000-question run, not the 200 one
    assert t.rows[1][-1] == "20260102-000000-bbbb"


def test_all_runs_newest_first_with_gpu_columns():
    t = report.all_runs("jeval", RECS)
    assert [row[0] for row in t.rows] == [
        "20260103-000000-cccc",
        "20260102-000000-bbbb",
        "20260101-000000-aaaa",
    ]
    util = t.header.index("GPU util")
    assert t.rows[1][util] == "99.9%" and t.rows[0][util] == DASH
    assert f"A10G {TIMES}4, none" in t.rows[1]


def test_formatting():
    assert report.fmt("jeval_latency_ms", 12347.0) == "12.3 s"
    assert report.fmt("jeval_latency_ms_p99", 80.0) == "80 ms"
    assert report.fmt("jeval_n", 2000.0) == "2,000"
    assert report.fmt("gpu_mem_used_mb_max", 20567.0) == "20.1 GiB"
    assert report.fmt("jeval_ece", float("nan")) == DASH


def test_write_is_deterministic(tmp_path):
    for r in RECS:
        r.save(tmp_path)
    md, page = report.write(tmp_path)
    first = (md.read_text(), page.read_text())
    report.write(tmp_path)
    assert (md.read_text(), page.read_text()) == first
    assert "| otherjev | 81.0% |" in first[0]
    assert "<svg" in first[1] and "otherjev" in first[1]
