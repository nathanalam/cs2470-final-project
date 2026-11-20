"""The hardware-profiling tables in the report: nsys length axis with GPU metrics, H100/H200 ratios, and
sampled ncu kernel classes."""

from __future__ import annotations

from s1 import report
from s1.record import RunRecord, Scalar
from s1.report_profiling import h200_table, ncu_ratio_table, ncu_table, nsys_table

MODAL: dict[str, Scalar] = {"platform": "modal"}


def _nsys(rid: str, gpu: str, length: int, kernel_ms: float, wall_ms: float, *, offset: int = 0) -> RunRecord:
    m = {
        "batch_nsys_state_tokens": float(length),
        "batch_nsys_wall_ms_per_decision": wall_ms,
        "batch_nsys_kernel_ms": kernel_ms,
        "batch_nsys_gpu_share": kernel_ms / wall_ms,
        "batch_nsys_kernel_tflops": 400.0,
        "batch_nsys_kernel_gbs": 500.0,
        "batch_nsys_gm_samples": 1000.0,
        "batch_nsys_gm_dram_read_bandwidth_throughput": 15.0,
        "batch_nsys_gm_dram_read_bandwidth_throughput_active": 19.0,
        "batch_nsys_gm_sms_active_throughput_active": 74.0,
    }
    return RunRecord(
        rid,
        MODAL | {"gpu": f"NVIDIA {gpu}"},
        {"model": "qwen3-8b.zeroshot", "batch_nsys_warmup_offset": offset},
        {"batch": m},
        {"cpu_model": "GenuineIntel family 6 model 143"},
    )


def _ncu(rid: str, gpu: str, gemm_ms: float, elem_ms: float) -> RunRecord:
    m = {
        "roofline_hw_sampled": 1.0,
        "roofline_hw_kernels": 199.0,
        "roofline_hw_gemm_ms": gemm_ms,
        "roofline_hw_gemm_share": gemm_ms / (gemm_ms + elem_ms),
        "roofline_hw_gemm_tflops": 795.0,
        "roofline_hw_elementwise_ms": elem_ms,
        "roofline_hw_elementwise_share": elem_ms / (gemm_ms + elem_ms),
        "roofline_hw_elementwise_dram_gbs": 1540.0,
    }
    return RunRecord(
        rid,
        MODAL | {"gpu": f"NVIDIA {gpu}"},
        {"model": "qwen3-8b.zeroshot", "ncu_metrics": "lean"},
        {"roofline": m},
    )


def test_nsys_table_newest_per_point_and_active_counters():
    recs = [
        _nsys("20261115-010000-a", "H100 80GB HBM3", 512, 27.0, 34.0),
        _nsys("20261115-020000-b", "H100 80GB HBM3", 512, 26.0, 33.0),  # newer: supersedes
        _nsys("20261115-020000-c", "H100 80GB HBM3", 512, 26.0, 60.0, offset=8),  # cold: its own row
        RunRecord(
            "x", dict(MODAL), {"model": "m"}, {"batch": {"batch_s16_b1_tflops_wall": 1.0}}
        ),  # no GPU metrics
    ]
    h, rows = nsys_table(recs)
    assert len(rows) == 2
    warm = next(r for r in rows if r[h.index("Warm-up")] == "warm")
    assert warm[h.index("Run")] == "20261115-020000-b"
    assert warm[h.index("DRAM read %")] == "19"  # the active-sample mean, not the all-sample one
    assert warm[h.index("Host CPU")] == "Sapphire Rapids"
    assert next(r for r in rows if r[h.index("Warm-up")] == "cold")[h.index("Wall ms")] == "60.0"


def test_h200_table_pairs_the_two_gpus():
    recs = [_nsys("a", "H100 80GB HBM3", 2048, 30.0, 40.0), _nsys("b", "H200", 2048, 25.0, 38.0)]
    h, rows = h200_table(recs)
    assert len(rows) == 1
    assert rows[0][h.index("Kernel speedup")] == "1.20x"
    assert h200_table(recs[:1])[1] == []  # one GPU only: no ratio


def test_ncu_tables_per_class_and_h200_ratio():
    recs = [_ncu("a", "H100 80GB HBM3", 3.6, 2.6), _ncu("b", "H200", 3.6, 1.85)]
    h, rows = ncu_table(recs)
    assert {r[h.index("Class")] for r in rows} == {"gemm", "elementwise"}
    rh, rrows = ncu_ratio_table(recs)
    by_class = {r[rh.index("Class")]: r[rh.index("H200 speedup")] for r in rrows}
    assert by_class == {"gemm": "1.00x", "elementwise": "1.41x", "all profiled": "1.14x"}


def test_report_includes_the_sections_only_when_records_exist():
    empty = report.to_markdown([])
    assert "Nsight Systems: kernel-time roofline" not in empty and "Sampled Nsight Compute" not in empty
    recs = [
        _nsys("a", "H100 80GB HBM3", 512, 27.0, 34.0),
        _nsys("b", "H200", 512, 26.0, 33.0),
        _ncu("c", "H100 80GB HBM3", 3.6, 2.6),
        _ncu("d", "H200", 3.6, 1.85),
    ]
    md = report.to_markdown(recs)
    for title in (
        "Nsight Systems: kernel-time roofline and GPU counters by state length",
        "Nsight Systems: H100 over H200 by state length",
        "Sampled Nsight Compute: kernel classes",
        "Sampled Nsight Compute: H100 over H200 per kernel class",
    ):
        assert title in md
    page = report.to_html(recs)
    assert "Sampled Nsight Compute: H100 over H200 per kernel class" in page
