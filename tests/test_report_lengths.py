"""The state-length axis in the report: curves per (platform, GPU), never mixed with natural runs, and the
H200-vs-H100 speedup by length."""

import pytest

from s1 import report, report_lengths, report_roofline
from s1.benchmarks import roofline
from s1.record import Metrics, RunRecord, Scalar

H100, H200 = "NVIDIA H100 80GB HBM3", "NVIDIA H200"
LENGTHS = [32, 512, 2048]
GRID: dict[str, list[Scalar]] = {roofline.LENGTH_AXIS: [32, 512, 2048]}


@pytest.fixture(autouse=True)
def fake_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    table = {H100: (989.0, 67.0, 3350.0), H200: (989.0, 67.0, 4800.0)}
    monkeypatch.setattr(roofline, "spec_peaks", table.get)


def _lengths(ms: dict[int, float], *, mean_scale: float = 1.0, bw: float = 3000.0) -> Metrics:
    m: Metrics = {
        "roofline_tflops": 50.0,
        "roofline_intensity": 200.0,
        "roofline_params_b": 8.0,
        "roofline_active_params_b": 8.0,
        "roofline_attention_layers": 1.0,
        "roofline_peak_bf16_tflops": 700.0,
        "roofline_peak_bw_gbs": bw,
    }
    if ms:
        m["roofline_lengths"] = float(len(ms))
    for n, med in ms.items():
        m |= {
            f"roofline_s{n}_intensity": n / 4,
            f"roofline_s{n}_tflops": 10.0,
            f"roofline_s{n}_tokens": n + 200.0,
            f"roofline_s{n}_gpu_ms": med * mean_scale,
            f"roofline_s{n}_gpu_ms_median": med,
            f"roofline_s{n}_gpu_ms_max": med * 3,
        }
    return m


def _rec(
    run_id: str, gpu: str, model: str, m: Metrics, *, grid: bool = True, mamba: str = "absent"
) -> RunRecord:
    inv: dict[str, str | int | float | bool] = {"gpu": gpu, "platform": "modal", "env_mamba_ssm": mamba}
    return RunRecord(run_id, inv, {"model": model}, {"roofline": m}, grid=GRID if grid else {})


RECS = [
    _rec("20261108-000001-a", H100, "qwen3-8b.zeroshot", _lengths({32: 60.0, 512: 70.0, 2048: 120.0})),
    _rec(
        "20261108-000002-b", H200, "qwen3-8b.zeroshot", _lengths({32: 50.0, 512: 70.0, 2048: 100.0}, bw=4300)
    ),
    # same model, other runtime: its own curve, never merged with the plain one
    _rec("20261108-000003-c", H100, "granite4-h-tiny.zeroshot", _lengths({32: 100.0, 2048: 110.0})),
    _rec(
        "20261108-000004-d", H100, "granite4-h-tiny.zeroshot", _lengths({32: 30.0, 2048: 40.0}), mamba="2.2.6"
    ),
    # a natural-mode run: never on the length panels
    _rec("20261108-000005-e", H100, "jevk5-9b", _lengths({}), grid=False),
]


def test_panels_hold_only_length_runs_split_by_gpu_and_variant():
    panels = report_lengths.panels(RECS)
    assert [p.label for p in panels] == ["H100 80GB HBM3 (Modal)", "H200 (Modal)"]
    names = sorted(c.name for c in panels[0].curves)
    assert names == [
        "granite4-h-tiny.zeroshot",
        "granite4-h-tiny.zeroshot (fused Mamba kernels)",
        "qwen3-8b.zeroshot",
    ]
    assert panels[0].lengths == LENGTHS


def test_curve_tflops_use_the_median_gpu_time():
    (pt,) = report_lengths.length_points(_lengths({512: 10.0}, mean_scale=4.0))
    assert pt.ms_median == 10.0 and pt.ms_mean == 40.0
    assert pt.tflops == pytest.approx(40.0)  # 10 TFLOP/s at the mean time is 40 at the median


def test_speedup_is_the_median_ms_ratio_per_length():
    (sp,) = report_lengths.speedups(report_lengths.panels(RECS))
    assert sp.lengths == LENGTHS
    rows = {name: ratios for name, _, ratios in sp.rows}
    assert rows == {"qwen3-8b.zeroshot": [pytest.approx(1.2), pytest.approx(1.0), pytest.approx(1.2)]}
    assert sp.bw_ratio == pytest.approx(4300 / 3000)  # measured bandwidths, not the datasheet
    header, table = report_lengths.speedup_table(sp)
    assert header[-1] == "Mean" and table[0][-1] == "1.13"


def test_ms_table_and_charts_render():
    panel = report_lengths.panels(RECS)[0]
    header, rows = report_lengths.ms_table(panel)
    assert header[1:4] == ["32 tok", "512 tok", "2048 tok"]
    plain = next(r for r in rows if r[0] == "granite4-h-tiny.zeroshot")
    assert plain[1:4] == ["100", report_roofline.DASH, "110"]
    svg = report_lengths.length_chart(panel)
    assert svg.count('class="traj curve"') == 3 and "fused Mamba kernels" in svg
    (sp,) = report_lengths.speedups(report_lengths.panels(RECS))
    assert "1.0x: no gain" in report_lengths.speedup_chart(sp)


def test_report_has_the_length_sections_and_keeps_natural_points(tmp_path):
    md = report.to_markdown(RECS, tmp_path)
    assert "### H200 vs H100 80GB HBM3 speedup by state length (Modal)" in md
    assert "### State-length axis on H200 (Modal): median GPU ms per question" in md
    natural = report_roofline.panels(RECS)
    assert [p.model for panel in natural for p in panel.points] == ["jevk5-9b"]
    assert "State-length axis on" in report.to_html(RECS, tmp_path)


def test_no_length_runs_no_length_sections(tmp_path):
    md = report.to_markdown(RECS[-1:], tmp_path)
    assert "state length" not in md


def test_a_length_where_every_question_failed_has_no_point():
    m = _lengths({32: 50.0, 2048: 0.0})
    m["roofline_s2048_tokens"] = 0.0
    assert [p.length for p in report_lengths.length_points(m)] == [32]
