import time

import pytest

from s1.gpu_stats import GpuSampler, Sample, parse_line, summarize_samples


def test_parse_line():
    s = parse_line("3, 87, 20480, 23034, 71.25, 66\n")
    assert s is not None and (s.gpu, s.util_pct, s.mem_used_mb, s.power_w) == (3, 87.0, 20480.0, 71.25)
    assert parse_line("0, [N/A], 1, 2, 3, 4") is None
    assert parse_line("") is None


def test_sampler_reads_fake_nvidia_smi(fake_nvidia_smi):
    with GpuSampler(cmd=fake_nvidia_smi) as smi:
        deadline = time.monotonic() + 5
        while len(smi.samples) < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
    m = smi.metrics()
    assert m["gpu_count"] == 2
    assert m["gpu_samples"] == 4  # the [N/A] row is dropped
    assert m["gpu_util_pct"] == 60.0
    assert m["gpu_util_pct_p99"] == pytest.approx(89.4)
    assert m["gpu_mem_used_mb_max"] == 4000
    assert m["gpu_power_w_total_mean"] == 65.0 + 65.0
    assert m["gpu_temp_c_max"] == 55


def test_energy_uses_window_length():
    m = summarize_samples([Sample(0, 0, 0, 0, 100.0, 0), Sample(1, 0, 0, 0, 200.0, 0)], duration_s=36.0)
    assert m["gpu_energy_wh"] == pytest.approx(3.0)  # 300 W for 36 s


def test_no_nvidia_smi_means_no_metrics():
    with GpuSampler(cmd=["/nonexistent/nvidia-smi"]) as smi:
        pass
    assert smi.metrics() == {}
