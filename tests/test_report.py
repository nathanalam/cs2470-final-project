from s1 import report, report_roofline
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


def _modal(r: RunRecord) -> RunRecord:
    return RunRecord(r.run_id, r.invariants | {"platform": "modal"}, r.params, r.results, r.meta)


def test_platform_column_in_leaderboard_and_all_runs():
    recs = [*RECS[:2], _modal(RECS[2])]
    lb, runs = report.leaderboard("jeval", recs), report.all_runs("jeval", recs)
    for t in (lb, runs):
        i = t.header.index("Platform")
        assert t.header[i + 1] == "Hardware"
    i = lb.header.index("Platform")
    assert {row[0]: row[i] for row in lb.rows} == {"otherjev": "Modal", "openjev": "cluster"}
    i = runs.header.index("Platform")
    assert [row[i] for row in runs.rows] == ["Modal", "cluster", "cluster"]


def test_platform_column_in_shape_sweep_tables():
    sweep = RunRecord(
        "20261108-000000-dddd",
        {"gpu": "NVIDIA H100 80GB HBM3", "platform": "modal"},
        {"model": "openjev"},
        {"shape_sweep": {"shape_sweep_latency_ms": 72.0, "shape_sweep_n": 3060.0}},
    )
    t = report.all_runs("shape_sweep", [sweep])
    assert t.rows[0][t.header.index("Platform")] == "Modal"
    assert t.rows[0][t.header.index("Hardware")] == "H100 80GB HBM3"


DETAILED = {
    "gpu_util_pct": 90.0,
    "gpu_count": 2.0,
    "gpu_mem_util_pct": 41.5,
    "gpu_clock_sm_mhz": 1755.0,
    "gpu_clock_sm_mhz_min": 1410.0,
    "gpu_clock_mem_mhz": 6251.0,
    "gpu_throttle_frac": 0.125,
    "gpu_throttle_mask_seen": 4.0,
    "gpu_util_pct_spread": 3.0,
    "gpu_pcie_rx_mb_s_total_mean": 120.0,
    "gpu_pcie_tx_mb_s_total_mean": 30.0,
    "gpu_pcie_samples": 50.0,
    "gpu0_util_pct": 91.5,
    "gpu0_mem_util_pct": 40.0,
    "gpu0_clock_sm_mhz": 1760.0,
    "gpu1_util_pct": 88.5,
    "gpu1_mem_util_pct": 43.0,
    "gpu1_pcie_rx_mb_s": 60.0,
}


def _telemetry_rec() -> RunRecord:
    return RunRecord(
        "20261108-000000-eeee",
        {"gpu": "NVIDIA H100 80GB HBM3", "platform": "modal", "gpu_metrics": "detailed"},
        {"model": "jevk5-9b"},
        {"roofline": ROOF, "roofline/nvidia_smi": DETAILED},
        {
            "cpu_model": "AMD EPYC 9R14",
            "cpus_requested": 8,
            "cpu_count": 32,
            "roofline_kernels_seen": 57,
            "roofline_kernels_top": "41.2% sm90_xmma_gemm_bf16; 9.8% void at::native::foo<float, 4>",
            "roofline_attention_kernel": "flash_fwd_kernel<Flash_fwd_kernel_traits>",
            "roofline_attention_share": 0.0712,
        },
    )


def test_telemetry_digest_of_detailed_metrics_cpu_and_kernels():
    rows = {k: (v, code) for k, v, code in report.telemetry(_telemetry_rec())}
    assert rows["CPU"] == ("AMD EPYC 9R14", False)
    assert rows["CPUs"][0] == "8 requested, 32 visible"
    pooled = rows["GPUs, pooled"][0]
    assert "memory controller busy 41.5%" in pooled
    assert "SM 1,755 MHz (min 1,410 MHz)" in pooled and "memory 6,251 MHz" in pooled
    assert "throttled 12% of samples, reasons 0x4" in pooled
    assert "PCIe rx 120 MB/s, tx 30 MB/s" in pooled and "util spread across GPUs 3.0 pts" in pooled
    assert rows["GPU 0"][0] == "util 91.5%; memory controller busy 40.0%; SM 1,760 MHz"
    assert rows["GPU 1"][0] == f"util 88.5%; memory controller busy 43.0%; PCIe rx 60 MB/s, tx {DASH}"
    assert rows["CUDA kernels seen"] == ("57", False)
    assert rows["Kernel 1 (41.2% of GPU time)"] == ("sm90_xmma_gemm_bf16", True)
    assert rows["Kernel 2 (9.8% of GPU time)"] == ("void at::native::foo<float, 4>", True)
    assert rows["Attention kernel"] == ("flash_fwd_kernel<Flash_fwd_kernel_traits>", True)
    assert rows["Attention kernel share"] == ("7.1%", False)


def test_telemetry_absent_for_records_without_it():
    assert report.telemetry(RECS[0]) == []
    unknown = RunRecord(
        "x", {"gpu": "NVIDIA L4", "platform": "modal"}, {"model": "m"}, {}, {"cpu_model": "unknown"}
    )
    assert report.telemetry(unknown) == []  # early Modal records' placeholder carries nothing
    none_seen = RunRecord(
        "y",
        {"gpu": "NVIDIA L4"},
        {"model": "elephant-1"},
        {},
        {"roofline_kernels_top": "none seen", "roofline_attention_kernel": "none seen"},
    )
    assert report.telemetry(none_seen) == [
        ("Top CUDA kernels", "none seen", False),
        ("Attention kernel", "none seen", False),
    ]


def test_run_details_render_telemetry_only_when_present(tmp_path):
    RECS[0].save(tmp_path)
    md, page = report.write(tmp_path)
    assert "**Telemetry**" not in md.read_text() and "telemetry ·" not in page.read_text()
    _telemetry_rec().save(tmp_path)
    md, page = report.write(tmp_path)
    text = md.read_text()
    assert "**Telemetry**" in text
    assert "| Kernel 2 (9.8% of GPU time) | `void at::native::foo<float, 4>` |" in text
    assert "telemetry · GPU 1" in page.read_text()
    assert "void at::native::foo&lt;float, 4&gt;" in page.read_text()  # escaped in the page


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


# --- roofline section ---------------------------------------------------------------------------


def _roof_rec(run_id, model, metrics, gpu="NVIDIA L4", **params):
    return RunRecord(run_id, {"gpu": gpu}, {"model": model, **params}, {"roofline": metrics})


ROOF = {
    "roofline_tflops": 30.0,
    "roofline_intensity": 400.0,
    "roofline_t512_intensity": 300.0,
    "roofline_t512_tflops": 25.0,
    "roofline_t1024_intensity": 600.0,
    "roofline_t1024_tflops": 35.0,
    "roofline_params_b": 8.95,
    "roofline_active_params_b": 6.9,
    "roofline_attention_layers": 0.25,
    "roofline_peak_bf16_tflops": 90.0,
    "roofline_peak_fp32_tflops": 25.0,
    "roofline_peak_bw_gbs": 270.0,
    "roofline_spec_bf16_tflops": 121.0,
    "roofline_spec_bw_gbs": 300.0,
    "roofline_accuracy": 0.6,
}


def test_roofline_family_comes_from_the_metrics():
    assert report_roofline.family(ROOF) == 0  # dense, 1 in 4 layers attention
    assert report_roofline.family(ROOF | {"roofline_attention_layers": 1.0}) == 1
    assert report_roofline.family(ROOF | {"roofline_active_params_b": 1.1, "roofline_params_b": 6.9}) == 2
    small_encoder = ROOF | {
        "roofline_active_params_b": 0.11,
        "roofline_params_b": 0.31,
        "roofline_attention_layers": 1.0,
    }
    assert (
        report_roofline.family(small_encoder, "laya.multilingual") == 1
    )  # the catalog says encoder, not MoE
    assert report_roofline.family(ROOF, "olmoe-1b-7b.zeroshot") == 2


NCU = ROOF | {"roofline_hw_intensity": 150.0, "roofline_hw_tflops": 29.0}


def test_roofline_points_pair_clean_runs_with_their_ncu_run():
    recs = [
        _roof_rec("20261104-000001-a", "m1", ROOF),
        _roof_rec("20261104-000002-b", "m1", ROOF | {"roofline_tflops": 31.0}),  # latest clean run wins
        _roof_rec("20261104-000003-c", "m1", NCU, roofline_ncu=4),
        _roof_rec("20261104-000004-d", "m2", ROOF, gpu="NVIDIA A10G"),
    ]
    by_gpu = {p.gpu: p.points for p in report_roofline.panels(recs)}
    assert set(by_gpu) == {"NVIDIA L4", "NVIDIA A10G"}
    (p,) = by_gpu["NVIDIA L4"]
    assert p.m["roofline_tflops"] == 31.0
    assert p.hw["roofline_hw_intensity"] == 150.0
    assert by_gpu["NVIDIA A10G"][0].hw == {}


def test_roofline_chart_and_table():
    (panel,) = report_roofline.panels([_roof_rec("20261104-000001-a", "jevk5-9b", ROOF)])
    pts = panel.points
    svg = report_roofline.chart(panel)
    assert ">jevk5-9b</text>" in svg  # direct label
    assert "measured: 90 TFLOP/s bf16 tensor, 270 GB/s" in svg
    assert "datasheet: 121 TFLOP/s bf16" in svg
    assert 'class="traj"' in svg  # path through the prompt-length bins
    header, rows = report_roofline.table(pts)
    assert len(rows) == 1
    assert len(rows[0]) == len(header)
    assert rows[0][header.index("Accuracy")] == "60.0%"
    assert rows[0][header.index("HW TFLOP/s")] == report_roofline.DASH


def test_report_includes_the_roofline_section(tmp_path):
    _roof_rec("20261104-000001-a", "jevk5-9b", ROOF).save(tmp_path)
    md, page = report.write(tmp_path)
    assert "### Roofline on L4 (cluster)" in md.read_text()
    assert 'class="card chart roof"' in page.read_text()


def test_recipes_list_decision_models_and_the_control_only():
    recs = [
        _roof_rec("20261104-000001-a", "jevk5-9b", ROOF),
        _roof_rec("20261104-000002-b", "kev-4b.hybrid", ROOF | {"roofline_padding": 0.25}),
        _roof_rec("20261104-000003-c", "olmoe-1b-7b.zeroshot", ROOF),  # a reference, no recipe
        _roof_rec("20261104-000004-d", "qwen35-9b.zeroshot", ROOF),  # the control has one
    ]
    (panel,) = report_roofline.panels(recs)
    header, rows = report_roofline.recipes(panel.points)
    assert [r[0] for r in rows] == ["jevk5-9b", "kev-4b.hybrid", "qwen35-9b.zeroshot"]
    assert all(len(r) == len(header) for r in rows)
    assert rows[1][header.index("Padding")] == "25%"
    assert rows[0][header.index("Padding")] == report_roofline.DASH
    assert report_roofline.is_decision_model("kev-4b.hybrid")
    assert not report_roofline.is_decision_model("olmoe-1b-7b.zeroshot")


def test_report_includes_findings_when_written(tmp_path):
    _roof_rec("20261104-000001-a", "jevk5-9b", ROOF).save(tmp_path)
    md, page = report.write(tmp_path)
    assert "### Findings" not in md.read_text()
    (tmp_path / report_roofline.FINDINGS).write_text("Some **bold** claim.\n\n- one `x`\n- two\n  wrapped\n")
    md, page = report.write(tmp_path)
    assert "### Findings" in md.read_text() and "### Decision recipes on L4 (cluster)" in md.read_text()
    assert "<p>Some <strong>bold</strong> claim.</p>" in page.read_text()
    assert "<ul><li>one <code>x</code></li><li>two wrapped</li></ul>" in page.read_text()


def test_findings_nested_lists():
    text = "- a\n  - b\n    wrapped\n    - c\n  - d\n- e\n"
    assert report_roofline.markdown_to_html(text) == (
        "<ul><li>a<ul><li>b wrapped<ul><li>c</li></ul></li><li>d</li></ul></li><li>e</li></ul>"
    )
