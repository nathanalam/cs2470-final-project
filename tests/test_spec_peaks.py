import pytest

from s1.benchmarks.roofline import SPEC, spec_peaks

REAL = {
    "NVIDIA L4": "L4",
    "NVIDIA A100-SXM4-80GB": "A100-80GB",
    "NVIDIA H100 80GB HBM3": "H100",
    "NVIDIA H200": "H200",
    "NVIDIA B200": "B200",
}
LOOKALIKES = [
    "NVIDIA A100-SXM4-40GB",
    "NVIDIA A100-PCIE-40GB",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 PCIe",
    "NVIDIA H100 NVL",
    "NVIDIA H200 NVL",
    "NVIDIA L40S",
    "NVIDIA L40",
    "NVIDIA A10G",
    "NVIDIA A10",
    "NVIDIA B100",
    "",
]
RIDGE = {"L4": 403, "A100-80GB": 153, "H100": 295, "H200": 206, "B200": 292}


@pytest.mark.parametrize("name,key", REAL.items())
def test_real_names(name: str, key: str) -> None:
    assert spec_peaks(name) == SPEC[key]


@pytest.mark.parametrize("name", LOOKALIKES)
def test_lookalikes_do_not_match(name: str) -> None:
    assert spec_peaks(name) is None


@pytest.mark.parametrize("key,ridge", RIDGE.items())
def test_ridge_points(key: str, ridge: int) -> None:
    bf16, _, bw = SPEC[key]
    assert round(bf16 * 1e3 / bw) == ridge


def test_l4_unchanged() -> None:
    assert spec_peaks("NVIDIA L4") == (121.0, 30.3, 300.0)
