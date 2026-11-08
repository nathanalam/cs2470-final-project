"""The roofline section across several GPUs and both platforms: one roof per (platform, GPU), never mixed.

Datasheet peaks come from `roofline.spec_peaks`, which workstream P fills in; these tests swap in their own
table so they do not depend on its contents.
"""

import math
import re

import pytest

from s1 import report, report_roofline
from s1.benchmarks import roofline
from s1.record import Metrics, RunRecord

# bf16 TFLOP/s, fp32 TFLOP/s, GB/s; "NVIDIA X9" is a GPU the datasheet table does not know.
FAKE_SPEC = {
    "NVIDIA L4": (121.0, 30.3, 300.0),
    "NVIDIA H100 80GB HBM3": (989.0, 67.0, 3350.0),
    "NVIDIA B200": (2250.0, 80.0, 7700.0),
}


@pytest.fixture(autouse=True)
def fake_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(roofline, "spec_peaks", FAKE_SPEC.get)


def _metrics(tflops: float, peak: float, bw: float, *, spec: bool = False) -> Metrics:
    m = {
        "roofline_tflops": tflops,
        "roofline_intensity": 300.0,
        "roofline_efficiency": 0.5,
        "roofline_params_b": 8.95,
        "roofline_active_params_b": 6.9,
        "roofline_attention_layers": 0.25,
        "roofline_peak_bf16_tflops": peak,
        "roofline_peak_fp32_tflops": peak / 4,
        "roofline_peak_bw_gbs": bw,
        "roofline_accuracy": 0.7,
    }
    if spec:  # what the run recorded from the datasheet table of its day
        m |= {
            "roofline_spec_bf16_tflops": 500.0,
            "roofline_spec_fp32_tflops": 50.0,
            "roofline_spec_bw_gbs": 2000.0,
        }
    return m


def _rec(run_id: str, model: str, gpu: str, m: Metrics, platform: str | None = None) -> RunRecord:
    inv: dict[str, str | int | float | bool] = {"gpu": gpu}
    if platform is not None:
        inv["platform"] = platform
    return RunRecord(run_id, inv, {"model": model}, {"roofline": m})


RECS = [
    # the same model and GPU on both platforms: must land in separate panels
    _rec("20261104-000001-a", "jevk5-9b", "NVIDIA L4", _metrics(30.0, 59.0, 231.0)),
    _rec("20261108-000002-b", "jevk5-9b", "NVIDIA L4", _metrics(27.8, 54.0, 230.0), "modal"),
    _rec("20261108-000003-c", "qwen3-8b.zeroshot", "NVIDIA L4", _metrics(30.7, 54.4, 229.0), "modal"),
    _rec("20261108-000004-d", "jevk5-9b", "NVIDIA B200", _metrics(400.0, 1800.0, 6500.0), "modal"),
    _rec("20261108-000005-e", "jevk5-9b", "NVIDIA H100 80GB HBM3", _metrics(210.0, 700.0, 2900.0), "modal"),
    _rec(
        "20261108-000006-f", "kev-4b.qwen3", "NVIDIA H100 80GB HBM3", _metrics(150.0, 720.0, 2950.0), "modal"
    ),
    # no datasheet entry and no recorded spec, no measured ceilings either
    _rec(
        "20261108-000007-g",
        "jevk5-9b",
        "NVIDIA X9",
        {"roofline_tflops": 50.0, "roofline_intensity": 200.0},
        "modal",
    ),
]


def test_panels_split_by_platform_and_gpu_ordered_by_datasheet_peak():
    panels = report_roofline.panels(RECS)
    assert [p.label for p in panels] == [
        "L4 (cluster)",
        "L4 (Modal)",
        "H100 80GB HBM3 (Modal)",
        "B200 (Modal)",
        "X9 (Modal)",  # unknown peak last
    ]
    by_label = {p.label: p for p in panels}
    assert [pt.run_id for pt in by_label["L4 (cluster)"].points] == ["20261104-000001-a"]
    assert {pt.model for pt in by_label["L4 (Modal)"].points} == {"jevk5-9b", "qwen3-8b.zeroshot"}
    assert len(by_label["H100 80GB HBM3 (Modal)"].points) == 2


def test_platform_labels():
    assert report_roofline.platform({}) == "cluster"
    assert report_roofline.platform({"platform": "modal"}) == "Modal"
    assert report_roofline.platform({"platform": "runpod"}) == "runpod"


def test_each_panel_has_its_own_measured_roof():
    by_label = {p.label: p for p in report_roofline.panels(RECS)}
    cluster, modal = by_label["L4 (cluster)"], by_label["L4 (Modal)"]
    assert cluster.measured.bf16 == 59.0 and cluster.measured.bw == 231.0
    assert modal.measured.bf16 == pytest.approx((54.0 + 54.4) / 2)  # median over the panel's runs only
    h100 = by_label["H100 80GB HBM3 (Modal)"].measured
    assert h100.bf16 == 710.0 and h100.fp32 == pytest.approx(177.5)
    assert h100.ridge == pytest.approx(710.0 * 1e3 / 2925.0)


def test_datasheet_from_spec_peaks_then_the_records_then_none():
    by_label = {p.label: p for p in report_roofline.panels(RECS)}
    assert by_label["B200 (Modal)"].datasheet == report_roofline.Roof(2250.0, 80.0, 7700.0)
    unknown = by_label["X9 (Modal)"].datasheet
    assert all(math.isnan(v) for v in (unknown.bf16, unknown.fp32, unknown.bw))  # no roof
    recorded = report_roofline.panels(
        [
            _rec(
                "20261108-000008-h",
                "jevk5-9b",
                "NVIDIA X9",
                _metrics(50.0, 400.0, 1800.0, spec=True),
                "modal",
            )
        ]
    )[0]
    assert recorded.datasheet == report_roofline.Roof(500.0, 50.0, 2000.0)


def test_charts_never_mix_platforms_or_gpus():
    panels = report_roofline.panels(RECS)
    scale = report_roofline.shared_scale(panels)
    svgs = {p.label: report_roofline.chart(p, scale) for p in panels}
    cluster, modal = svgs["L4 (cluster)"], svgs["L4 (Modal)"]
    assert "Roofline on L4 (cluster)" in cluster and "Roofline on L4 (Modal)" in modal
    assert "(run 20261104-000001-a)" in cluster and "(run 20261108-000002-b)" not in cluster
    assert "(run 20261108-000002-b)" in modal and "(run 20261104-000001-a)" not in modal
    assert ">qwen3-8b.zeroshot</text>" not in cluster
    assert "measured: 59 TFLOP/s bf16 tensor, 231 GB/s" in cluster
    assert "measured: 54 TFLOP/s bf16 tensor, 230 GB/s" in modal
    assert "datasheet: 989 TFLOP/s bf16, 3350 GB/s" in svgs["H100 80GB HBM3 (Modal)"]
    assert svgs["H100 80GB HBM3 (Modal)"].count("datasheet:") == 1  # one roof per panel


def test_gpu_without_peaks_draws_points_and_says_so():
    (panel,) = [p for p in report_roofline.panels(RECS) if p.gpu == "NVIDIA X9"]
    svg = report_roofline.chart(panel)
    assert "datasheet:" not in svg and "measured:" not in svg
    assert 'class="ceil' not in svg
    assert "no measured or datasheet peaks for this GPU" in svg
    assert ">jevk5-9b</text>" in svg


def test_shared_scale_spans_every_panel():
    panels = report_roofline.panels(RECS)
    s = report_roofline.shared_scale(panels)
    assert 10**s.y1 >= 2250.0  # the B200 datasheet roof fits
    assert 10**s.y0 <= 27.8  # and the slowest L4 point
    axes = {
        re.findall(r'y="[\d.]+" text-anchor="end">([\d.e+]+)</text>', report_roofline.chart(p, s))[0]
        for p in panels
    }
    assert len(axes) == 1  # every panel's y axis starts at the same tick


def test_roofs_and_by_gpu_tables():
    panels = report_roofline.panels(RECS)
    header, rows = report_roofline.roofs(panels)
    assert all(len(r) == len(header) for r in rows)
    by = {r[0]: r for r in rows}
    assert by["L4 (cluster)"][header.index("Measured bf16 TFLOP/s")] == "59"
    assert by["L4 (cluster)"][header.index("Measured / datasheet bf16")] == "49%"
    assert by["X9 (Modal)"][header.index("Datasheet bf16 TFLOP/s")] == report_roofline.DASH
    assert by["X9 (Modal)"][header.index("Measured / datasheet bf16")] == report_roofline.DASH
    header, rows = report_roofline.by_gpu(panels)
    assert header[1:3] == ["L4 (cluster)", "L4 (Modal)"]
    jev = next(r for r in rows if r[0] == "jevk5-9b")
    assert jev[1:3] == ["30 (50%)", "27.8 (50%)"]
    assert jev[header.index("X9 (Modal)")] == "50"  # no efficiency recorded
    kev = next(r for r in rows if r[0] == "kev-4b.qwen3")
    assert kev[header.index("L4 (cluster)")] == report_roofline.DASH


def test_report_has_a_panel_per_platform_and_gpu(tmp_path):
    for r in RECS:
        r.save(tmp_path)
    md, page = report.write(tmp_path)
    text, html = md.read_text(), page.read_text()
    for label in ("L4 (cluster)", "L4 (Modal)", "H100 80GB HBM3 (Modal)", "B200 (Modal)", "X9 (Modal)"):
        assert f"### Roofline on {label}" in text
        assert f"Roofline on {label}: each model" in html
    assert text.count("### Roofs") == 1 and "### Achieved TFLOP/s by GPU" in text
    assert html.count('class="card chart roof"') == 5
    # the roofline leaderboard and all-runs tables say which platform each run was on
    assert "| Platform | Hardware |" in text


def test_single_panel_report_has_no_cross_gpu_matrix(tmp_path):
    RECS[0].save(tmp_path)
    md, _ = report.write(tmp_path)
    assert "### Roofs" in md.read_text() and "### Achieved TFLOP/s by GPU" not in md.read_text()
