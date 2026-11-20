"""The steady-state report section: B* per batch series, OpenJev's concurrency steady state, the best
model per GPU, the prep-mode table and the charts, on fake records."""

from __future__ import annotations

import math

from s1 import report, report_batch, report_steady
from s1.record import RunRecord
from s1.report_steady import steady

H100 = "NVIDIA H100 80GB HBM3"
L4 = "NVIDIA L4"
LFM = "lfm2-8b-a1b.zeroshot"


def _pts(busy, dps, oom=()):
    out = {
        b: {"kernel_busy": k, "decisions_per_s": d}
        for b, k, d in zip((1, 2, 4, 8, 16), busy, dps, strict=False)
    }
    for b in oom:
        out[b] = {"oom": 1.0}
    return out


def test_b_star_is_the_smallest_busy_batch_no_larger_batch_beats_by_5pct():
    st = steady(_pts((0.5, 0.8, 0.96, 0.97, 0.98), (10, 18, 30, 33, 34)), 128)
    assert st is not None
    # B=4 is busy but B=8 beats it by 10%; B=8 is busy and B=16 is only 3% faster
    assert st.b_star == 8 and st.plateau == 8 and st.speed == 33
    assert (st.max_busy, st.max_busy_b) == (0.98, 16)
    assert report_steady.busy_str(0.9476) == "0.94" and report_steady.busy_str(1.15) == "1.15"


def test_trace_inflation_counts_as_busy_and_oom_points_are_skipped():
    st = steady(_pts((1.15, 1.02, 0.99), (5, 5.1, 5.0), oom=(8, 16)), 1024)
    assert st is not None
    assert st.b_star == 1 and st.ooms == (8, 16)
    assert set(st.dps) == {1, 2, 4}


def test_not_reached_reports_the_plateau():
    st = steady(_pts((0.3, 0.5, 0.7, 0.8, 0.82), (10, 20, 40, 41, 41.5)), 128)
    assert st is not None
    assert st.b_star is None and st.plateau == 4 and st.speed == 40
    assert report_steady.b_star_cell(st) == "not reached (plateau B=4)"
    assert report_steady.speed_cell(st) == "40 (plateau)"
    assert report_steady.busy_cell(st) == "0.82 (B=16)"
    assert steady({4: {"oom": 1.0}}, 128) is None
    assert report_steady.b_star_cell(None) == report_steady.DASH


def _batch(run_id, gpu, model, busy, dps, *, prep=None, experts=None, oom_1024=()):
    m = {}
    for length in (128, 1024):
        for b, k, d in zip((1, 2, 4, 8), busy, dps, strict=True):
            if length == 1024 and b in oom_1024:
                m[f"batch_s{length}_b{b}_oom"] = 1.0
                continue
            p = f"batch_s{length}_b{b}_"
            m |= {
                f"{p}kernel_busy": k,
                f"{p}decisions_per_s": d,
                f"{p}tflops_wall": d / 10,
                f"{p}wall_ms": 10.0,
            }
            if prep == "pretokenized":
                m[f"{p}prep_ms"] = 2.5
    params = {"model": model} | ({"batch_host_prep": prep} if prep else {})
    params |= {"semif_experts_implementation": experts} if experts else {}
    return RunRecord(run_id, {"gpu": gpu, "platform": "modal"}, params, {"batch": m}, meta={"cpu_model": "x"})


BATCH = [
    _batch("20261119-01", H100, LFM, (0.4, 0.7, 0.85, 0.88), (30, 50, 90, 100)),
    _batch("20261119-02", H100, LFM, (0.4, 0.8, 0.96, 0.97), (30, 50, 100, 104), prep="pretokenized"),
    _batch("20261119-03", L4, LFM, (0.9, 0.96, 0.97, 0.97), (9, 13, 13.5, 13), oom_1024=(4, 8)),
    _batch("20261119-04", L4, "qwen35-9b.zeroshot", (1.1, 1.0, 1.0, 1.0), (5, 5, 5, 5)),
]


def test_steady_table_rows_per_series_with_oom_points():
    ps = report_batch.panels(BATCH)
    h, rows = report_steady.steady_table(ps)
    assert h[:4] == ["Model", "GPU", "Prep mode", "B* at 128"] and h[-1] == "OOM points"
    by = {(r[1], r[0]): r for r in rows}
    l4 = by[("L4 (Modal)", "LFM2-8B-A1B (inline)")]
    assert l4[3] == "2" and l4[5] == "13" and l4[-1] == "B=4,8 at 1024"
    assert by[("L4 (Modal)", "Qwen3.5-9B")][3] == "1"
    inline = by[("H100 80GB HBM3 (Modal)", "LFM2-8B-A1B (inline)")]
    pretok = by[("H100 80GB HBM3 (Modal)", "LFM2-8B-A1B (pretokenized)")]
    assert inline[3] == "not reached (plateau B=8)" and pretok[3] == "4" and pretok[2] == "pretokenized"
    h, rows = report_steady.busy_table(ps, 128)
    assert h == ["GPU", "Series", "B=1", "B=2", "B=4", "B=8", "B*"]
    h, rows = report_steady.busy_table(ps, 1024)
    assert rows[0][1] == "LFM2-8B-A1B (inline)" and rows[0][4:6] == ["OOM", "OOM"]


def test_prep_table_pairs_inline_and_pretokenized_on_the_same_gpu():
    ps = report_batch.panels(BATCH)
    assert [s.name for _, s in report_steady.prep_rows(ps)] == [
        "LFM2-8B-A1B (inline)",
        "LFM2-8B-A1B (pretokenized)",
    ]
    h, rows = report_steady.prep_table(ps)
    assert h[-2:] == ["Decisions/s at B=8 (128)", "Host prep ms per batch at B=8"]
    assert rows[0][-2:] == ["100", "inside wall"] and rows[1][-2:] == ["104", "2.5"]


def _jeval(run_id, gpu, model, conc, qps, *, util, n=1000, acc=0.786, invalid=21.0):
    m = {
        "jeval_accuracy": acc,
        "jeval_ece": 0.03,
        "jeval_n": float(n),
        "jeval_invalid": invalid,
        "jeval_latency_ms": 1000.0 * conc / qps,
        "jeval_latency_ms_p99": 5000.0,
        "jeval_questions_per_s": qps,
    }
    smi = {"gpu_util_pct": util, "gpu0_util_pct": util, "gpu_count": 1.0}
    params = {"model": model, "jeval_concurrency": conc}
    return RunRecord(run_id, {"gpu": gpu, "platform": "modal"}, params, {"jeval": m, "jeval/nvidia_smi": smi})


JEVAL = [
    _jeval("20261119-11", H100, "openjev", 1, 4.5, util=70.0),
    _jeval("20261119-12", H100, "openjev", 8, 6.8, util=98.0),
    _jeval("20261119-13", H100, "openjev", 16, 7.0, util=99.0),
    _jeval("20261119-14", H100, "openjev", 32, 7.0, util=99.0),
    _jeval("20261108-01", H100, "openjev", 8, 6.7, util=89.0, n=200),  # too few levels for a curve
    _jeval("20261119-21", L4, LFM, 1, 8.2, util=86.0, n=200, acc=0.429, invalid=49.0),
    _jeval("20261115-01", L4, "d1-3b.eager", 1, 4.7, util=94.0, n=200, acc=0.653, invalid=6.0),
]


def test_openjev_steady_concurrency_needs_flat_throughput_and_busy_gpu():
    (cv,) = report_steady.openjev_curves(JEVAL)
    assert [p.concurrency for p in cv.points] == [1, 8, 16, 32] and cv.n == 1000
    assert cv.steady is not None and cv.steady.concurrency == 8
    busy_low = [_jeval(f"r{c}", H100, "openjev", c, q, util=90.0) for c, q in ((1, 4.0), (8, 4.1), (16, 4.1))]
    (cv2,) = report_steady.openjev_curves(busy_low)
    assert cv2.steady is None  # flat throughput, but utilization never reaches 95%
    h, rows = report_steady.concurrency_table([cv])
    assert h[3:6] == ["q/s", "Gain to next", "Per-GPU util"]
    assert rows[1][4] == "+3%" and rows[1][-1] == "steady" and rows[3][4] == report_steady.DASH
    assert rows[0][6] == "222 ms" and rows[3][6] == "4.6 s"


def test_best_model_per_gpu():
    recs = [*BATCH, *JEVAL]
    ps = report_batch.panels(recs)
    curves = report_steady.openjev_curves(recs)
    cs = report_steady.candidates(recs, ps, curves)
    assert [(c.gpu, c.model) for c in cs] == [
        ("L4", "d1-3b.eager"),
        ("L4", LFM),
        ("H100 80GB HBM3", "openjev"),
    ]
    d1, lfm, oj = cs
    assert oj.run_id == "20261119-12" and oj.speed == "6.80 q/s at concurrency 8 (steady)"
    assert "one request at a time" in d1.speed
    assert lfm.speed == "13 dec/s at B*=2 (128 tok); 13 dec/s at B*=2 (1024 tok)"
    assert "16-option cap" in lfm.note
    assert math.isclose(lfm.valid_accuracy, 0.429 * 200 / 151)
    h, rows = report_steady.best_table(cs)
    assert h[6] == "Accuracy on valid (approx.)"
    assert rows[1][4:7] == ["42.9%", "49", "56.8%"] and rows[1][8] == "8.20 (1)"


def test_report_has_the_steady_state_section(tmp_path):
    recs = [*BATCH, *JEVAL]
    (tmp_path / report_steady.FINDINGS).write_text("### What it shows\n\nLFM2 is **busy** at B\\* = 32.")
    md = report.to_markdown(recs, tmp_path)
    for head in (
        "## Steady state",
        "### Findings",
        "### Best model per GPU",
        "### Steady-state batch per model and GPU",
        "### OpenJev: throughput, utilization and latency vs concurrency",
        "### Prep mode: host prep inside the timed batch",
        "### GPU utilization (kernel_busy) vs batch size, 128-token states",
        "### GPU utilization (kernel_busy) vs batch size, 1024-token states",
    ):
        assert head in md
    assert md.index("## Steady state") < md.index("## batch")
    page = report.to_html(recs, tmp_path)
    assert "<strong>busy</strong> at B* = 32" in page
    assert "GPU utilization (kernel_busy) vs batch size, 128-token states" in page
    assert "OpenJev (vLLM) on JEVal" in page and 'class="ref"' in page
    busy = report_steady.busy_chart(report_batch.panels(BATCH), 128)
    assert busy.count('role="img"') == 2  # one panel per GPU
    assert 'r="6" fill="var(--surface)"' in busy  # ring on B*
    assert report_steady.busy_chart([], 128) == "" and report_steady.concurrency_chart([]) == ""
    assert "## Steady state" not in report.to_markdown([], tmp_path)
