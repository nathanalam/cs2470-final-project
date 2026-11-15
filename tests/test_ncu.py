"""Parsing Nsight Compute's raw CSV and summing it into roofline_hw_* metrics. Values from a real run
(logs/probe3: a 4096^3 bf16 GEMM and a 1024^3 fp32 GEMM on an A10G)."""

import math
import re

import pytest

from s1 import ncu
from s1.cli import _scalar

HEADER = [
    "ID",
    "Kernel Name",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "gpu__time_duration.sum",
    "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
]
UNITS = ["", "", "byte", "byte", "ns", "", "inst", "inst"]
BF16_GEMM = ["0", "ampere_bf16_s16816gemm", "248846208", "40539904", "2848832", "137438953472", "0", "0"]
SGEMM = ["1", "ampere_sgemm_128x64_nn", "19724160", "15972864", "149568", "0", "1094713344", "5242880"]


def csv_text(*rows):
    lines = ["==PROF== Connected to process 1", "done NVIDIA A10G"]
    lines += [",".join(f'"{v}"' for v in r) for r in (HEADER, UNITS, *rows)]
    return "\n".join(lines) + "\n"


def test_parse_skips_log_lines_and_units():
    kernels = ncu.parse(csv_text(BF16_GEMM, SGEMM))
    assert len(kernels) == 2
    assert kernels[0]["sm__ops_path_tensor_src_bf16_dst_fp32.sum"] == 2 * 4096**3
    assert "Kernel Name" not in kernels[0]


def test_parse_without_profiled_kernels():
    with pytest.raises(ValueError, match="no ncu CSV"):
        ncu.parse("==PROF== Disconnected\n")


def test_summarize_flops_bytes_and_ratios():
    kernels = ncu.parse(csv_text(BF16_GEMM, SGEMM))
    run = {
        "roofline_hw_theory_gflop": 137.4389 + 2.1475,
        "roofline_hw_theory_mb": 100.0,
        "roofline_hw_timed_ms": 3.0,
    }
    m = ncu.summarize(kernels, run)
    flops = 2 * 4096**3 + 2 * 1094713344 + 5242880
    dram = 248846208 + 40539904 + 19724160 + 15972864
    assert next(iter(m)) == "roofline_hw_tflops"
    assert m["roofline_hw_gflop"] == pytest.approx(flops / 1e9)
    assert m["roofline_hw_dram_mb"] == pytest.approx(dram / 1e6)
    assert m["roofline_hw_intensity"] == pytest.approx(flops / dram)
    assert m["roofline_hw_tflops"] == pytest.approx(flops / 3e9)
    assert m["roofline_hw_tensor_share"] == pytest.approx(2 * 4096**3 / flops)
    # the counters match 2 * n^3 to within the sgemm's alpha/beta epilogue
    assert m["roofline_hw_flop_ratio"] == pytest.approx(1.0, abs=0.02)
    assert m["roofline_hw_byte_ratio"] == pytest.approx(dram / 1e8)
    assert m["roofline_hw_kernels"] == 2
    assert m["roofline_hw_kernel_ms"] == pytest.approx((2848832 + 149568) / 1e6)


def test_summarize_without_run_numbers_is_nan_not_an_error():
    m = ncu.summarize(ncu.parse(csv_text(BF16_GEMM)), {})
    assert math.isnan(m["roofline_hw_tflops"])
    assert math.isnan(m["roofline_hw_flop_ratio"])


def test_ncu_profiles_children_from_a_range():
    for clock in ncu.CLOCK_CONTROLS:
        args = list(ncu.args(clock))
        assert args[args.index("--profile-from-start") + 1] == "off"
        assert args[args.index("--target-processes") + 1] == "all"  # elephant-1's worker is a child


@pytest.fixture
def on_slurm(monkeypatch):
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "4233")


@pytest.fixture
def on_modal(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("MODAL_TASK_ID", "ta-01TEST")


def test_clock_control_default_is_unchanged_off_modal(on_slurm):
    assert ncu.clock_control() == "base"
    assert ncu.args() == ncu.BASE_ARGS  # no --clock-control flag: the cluster command line is as before
    assert "--clock-control" not in ncu.ARGS  # tests import ncu off Modal


def test_clock_control_is_none_on_modal(on_modal):
    assert ncu.clock_control() == "none"
    args = list(ncu.args())
    assert args[args.index("--clock-control") + 1] == "none"
    assert args[2:] == list(ncu.BASE_ARGS)


def test_explicit_clock_control_wins(on_modal):
    assert ncu.clock_control("base") == "base"
    assert ncu.args("none")[:2] == ("--clock-control", "none")


def test_explicit_none_off_modal(on_slurm):
    assert ncu.args("none")[:2] == ("--clock-control", "none")


def test_explicit_base_on_modal_adds_no_flag(on_modal):
    assert ncu.args("base") == ncu.BASE_ARGS


def test_bad_clock_control():
    with pytest.raises(ValueError, match="ncu_clock_control"):
        ncu.clock_control("boost")


def test_clock_control_is_recorded_on_modal(on_modal):
    kernels = ncu.parse(csv_text(BF16_GEMM))
    assert ncu.summarize(kernels, {})["roofline_hw_clocks_locked"] == 0.0  # Modal default: none
    assert ncu.summarize(kernels, {}, clock="base")["roofline_hw_clocks_locked"] == 1.0


def test_clock_control_recorded_as_locked_on_slurm(on_slurm):
    assert ncu.summarize(ncu.parse(csv_text(BF16_GEMM)), {})["roofline_hw_clocks_locked"] == 1.0


@pytest.mark.parametrize(
    ("text", "val"), [("4", 4), ("1.5", 1.5), ("true", True), ("false", False), ("L4", "L4")]
)
def test_set_values(text, val):
    assert _scalar(text) == val
    assert type(_scalar(text)) is type(val)


# --- kernel classes and the length axis -----------------------------------------------------------------

NAMED_HEADER = [*HEADER, ncu.DRAM_PCT, ncu.SM_PCT]
NAMED_UNITS = [*UNITS, "%", "%"]


def kernel(i, name, *, dram=0, ns=1000, bf16=0, dram_pct=0.0, sm_pct=0.0):
    return [str(i), name, str(dram), "0", str(ns), str(bf16), "0", "0", str(dram_pct), str(sm_pct)]


def named_csv(*rows):
    lines = ["==PROF== Connected to process 1"]
    lines += [",".join(f'"{v}"' for v in r) for r in (NAMED_HEADER, NAMED_UNITS, *rows)]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7", "attention"),
        ("fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel)", "attention"),
        ("void pytorch_flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits>", "attention"),
        ("nvjet_sm90_tst_128x224_64x4_2x1_v_bz_coopA_TNT", "gemm"),
        ("ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn", "gemm"),
        ("chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "mixer"),
        ("void at::native::conv_depthwise2d_forward_kernel_generic<c10::BFloat16, int>", "mixer"),
        ("void at::native::(anonymous namespace)::indexSelectLargeIndex<c10::BFloat16>", "routing"),
        ("void at::native::sbtopk::gatherTopK<float, unsigned int, 2, false>", "routing"),
        ("void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, MeanOps<float>>>", "norm"),
        ("void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor>", "elementwise"),
        ("some_unknown_kernel", "other"),
    ],
)
def test_classify(name, cls):
    assert ncu.classify(name) == cls


def test_parse_named_keeps_names_in_launch_order():
    named = ncu.parse_named(named_csv(kernel(0, "spin_kernel"), kernel(1, "nvjet_x", dram=10)))
    assert [n for n, _ in named] == ["spin_kernel", "nvjet_x"]
    assert ncu.NAME not in named[1][1]
    assert named[1][1]["dram__bytes_read.sum"] == 10
    assert ncu.parse(named_csv(kernel(0, "nvjet_x"))) == [named[1][1] | {"dram__bytes_read.sum": 0.0}]


def test_breakdown_shares_rates_and_time_weighted_speed_of_light():
    named = [
        ("nvjet_a", {ncu.TIME: 3000.0, "dram__bytes_read.sum": 3000.0, TENSOR_BF16: 6e6, ncu.SM_PCT: 80.0}),
        ("nvjet_b", {ncu.TIME: 1000.0, "dram__bytes_read.sum": 1000.0, ncu.SM_PCT: 40.0}),
        (
            "vectorized_elementwise_kernel",
            {ncu.TIME: 4000.0, "dram__bytes_read.sum": 8000.0, ncu.DRAM_PCT: 90.0},
        ),
    ]
    m = ncu.breakdown(named)
    assert m["roofline_hw_gemm_share"] == pytest.approx(0.5)
    assert m["roofline_hw_gemm_kernels"] == 2
    assert m["roofline_hw_gemm_ms"] == pytest.approx(4e-3)
    assert m["roofline_hw_gemm_dram_gbs"] == pytest.approx(1.0)  # 4000 B / 4000 ns
    assert m["roofline_hw_gemm_tflops"] == pytest.approx(6e6 / 4000 / 1e3)
    assert m["roofline_hw_gemm_sm_pct"] == pytest.approx(70.0)  # (80*3 + 40*1) / 4
    assert m["roofline_hw_elementwise_dram_gbs"] == pytest.approx(2.0)
    assert m["roofline_hw_elementwise_dram_pct"] == pytest.approx(90.0)
    assert "roofline_hw_attention_share" not in m  # absent class: no keys, never zero-filled
    assert "roofline_hw_elementwise_sm_pct" not in m  # metric missing from every kernel of the class


TENSOR_BF16 = "sm__ops_path_tensor_src_bf16_dst_fp32.sum"


def test_summarize_named_natural_mode_adds_busy_and_classes():
    named = ncu.parse_named(
        named_csv(kernel(0, "nvjet_x", ns=2_000_000), kernel(1, "rms_norm", ns=2_000_000))
    )
    m = ncu.summarize_named(named, {"roofline_hw_timed_ms": 10.0})
    assert m["roofline_hw_kernel_ms"] == pytest.approx(4.0)
    assert m["roofline_hw_busy"] == pytest.approx(0.4)
    assert m["roofline_hw_gemm_share"] == pytest.approx(0.5)
    assert not any(k.startswith("roofline_hw_s") and k[13].isdigit() for k in m)


def test_summarize_named_splits_lengths_on_markers():
    log = named_csv(
        kernel(0, "spin_kernel", ns=999_999),
        kernel(1, "nvjet_x", ns=1_000_000, dram=100),
        kernel(2, "spin_kernel", ns=999_999),
        kernel(3, "nvjet_x", ns=3_000_000, dram=300),
        kernel(4, "cudnn_sdpa_fprop", ns=1_000_000),
    )
    run = {
        "roofline_hw_timed_ms": 10.0,
        "roofline_hw_s32_timed_ms": 2.0,
        "roofline_hw_s32_theory_mb": 1e-4,
        "roofline_hw_s2048_timed_ms": 8.0,  # sorted numerically, not as strings
    }
    m = ncu.summarize_named(ncu.parse_named(log), run)
    assert m["roofline_hw_kernels"] == 3  # markers excluded from the totals
    assert m["roofline_hw_kernel_ms"] == pytest.approx(5.0)
    assert m["roofline_hw_s32_kernel_ms"] == pytest.approx(1.0)
    assert m["roofline_hw_s32_busy"] == pytest.approx(0.5)
    assert m["roofline_hw_s32_byte_ratio"] == pytest.approx(1.0)
    assert m["roofline_hw_s2048_kernels"] == 2
    assert m["roofline_hw_s2048_busy"] == pytest.approx(0.5)
    assert m["roofline_hw_s2048_attention_share"] == pytest.approx(0.25)
    assert "roofline_hw_s32_attention_share" not in m
    assert "roofline_hw_s32_clocks_locked" not in m


def test_summarize_named_refuses_unattributable_kernels():
    log = named_csv(kernel(0, "spin_kernel"), kernel(1, "nvjet_x"))
    with pytest.raises(ValueError, match="1 length markers"):
        ncu.summarize_named(
            ncu.parse_named(log), {"roofline_hw_s32_timed_ms": 1.0, "roofline_hw_s64_timed_ms": 1.0}
        )


# ---- sampled profiling (ncu_launch_count, ncu_kernel_regex, ncu_metrics) ----


def test_sampling_settings():
    assert ncu.sampling({}) == ncu.Sampling()
    assert not ncu.Sampling().sampled and not ncu.Sampling(lean=True).sampled
    s = ncu.sampling({"ncu_launch_count": 200, "ncu_kernel_regex": "spin|gemm", "ncu_metrics": "Lean"})
    assert s == ncu.Sampling(launch_count=200, kernel_regex="spin|gemm", lean=True) and s.sampled
    with pytest.raises(ValueError, match="ncu_metrics"):
        ncu.sampling({"ncu_metrics": "some"})
    with pytest.raises(ValueError, match=">= 0"):
        ncu.sampling({"ncu_launch_count": -1})
    with pytest.raises(TypeError):
        ncu.sampling({"ncu_launch_count": True})
    with pytest.raises(re.error):
        ncu.sampling({"ncu_kernel_regex": "("})


def test_args_with_sampling():
    assert ncu.args("base", ncu.Sampling()) == ncu.BASE_ARGS  # the default sampling leaves the line as before
    lean = list(ncu.args("base", ncu.Sampling(launch_count=200, kernel_regex="spin|gemm", lean=True)))
    metrics = lean[lean.index("--metrics") + 1].split(",")
    assert metrics == list(ncu.METRICS_LEAN) and not any("sass" in m for m in metrics)
    assert lean[lean.index("--launch-count") + 1] == "200"
    assert lean[lean.index("--kernel-name") + 1] == "regex:spin|gemm"
    full = list(ncu.args("none", ncu.Sampling(launch_count=50)))
    assert full[:2] == ["--clock-control", "none"]
    assert full[full.index("--metrics") + 1].split(",") == list(ncu.METRICS)
    assert "--kernel-name" not in full


def _k(name, ns=1000.0):
    return (name, {ncu.TIME: ns, "dram__bytes_read.sum": 10.0, ncu.TENSOR[0]: 100.0})


def test_summarize_named_sampled_drops_whole_run_keys_and_allows_cut_off_lengths():
    kernels = [_k("spin"), _k("gemm_a"), _k("gemm_b")]  # only the first length's marker was reached
    run = {"roofline_hw_timed_ms": 1.0, "roofline_hw_s32_timed_ms": 0.5, "roofline_hw_s512_timed_ms": 0.5}
    m = ncu.summarize_named(kernels, run, "none", sampled=True)
    assert m["roofline_hw_sampled"] == 1.0
    assert not any(k in m for k in ncu.WHOLE_RUN_KEYS)
    assert "roofline_hw_s32_busy" not in m and "roofline_hw_s32_tflops" not in m
    assert m["roofline_hw_s32_kernels"] == 2 and m["roofline_hw_s32_gemm_kernels"] == 2
    assert not any(k.startswith("roofline_hw_s512_") for k in m)  # its marker never came
    assert m["roofline_hw_intensity"] == pytest.approx(10.0)  # 200 FLOP / 20 bytes: still per profiled kernel
    with pytest.raises(ValueError, match="length markers"):
        ncu.summarize_named(kernels, run, "none")  # unsampled: every length must be there
