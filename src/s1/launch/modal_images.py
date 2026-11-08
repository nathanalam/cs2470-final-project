"""Modal images, one per runtime. Frozen interface for M3 (owner of the body: workstream m3-images).

A config picks its runtime with `[launch.modal] runtime = "<name>"` (default "default"); `modal_app` builds
`image(modal, runtime=..., openjev=...)` for every run. The interface below is the contract between
`modal_app.py` (workstream m3-fanout) and this module (workstream m3-images): change it only through the
coordinator.

    RUNTIMES                     the runtime names a config may ask for
    image(modal, *, runtime, openjev) -> modal.Image   the image a run of that runtime uses
    describe(runtime, *, openjev) -> str               one line for `s1 submit --dry-run`

Runtimes:
    default    this repo's uv.lock (`--extra openjev` for GPU runs) on the CUDA 13.0 devel base (nvcc for
               FlashInfer's JIT, as on the cluster nodes), or on debian_slim for CPU-only runs.
    elephant   default + runtimes/elephant's own uv project (torch 2.8) for elephant-1's worker.
               TODO(m3-images)
    granite    default + Granite's fused Mamba kernels (mamba-ssm, causal-conv1d). TODO(m3-images)

`modal` is imported by the caller and passed in, so importing this module never needs it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

PYTHON = "3.12"
UV_VERSION = "0.13.5"  # uv.lock is lock revision 5; pin the uv that reads it
CUDA_DEVEL_IMAGE = "nvidia/cuda:13.0.3-devel-ubuntu24.04"  # GPU image base: nvcc for FlashInfer's JIT
REPO_ROOT = Path(__file__).resolve().parents[3]
WORKDIR = "/root/s1"  # container cwd: src/, configs/ and the submitted config live here
MODELS_MOUNT = "/models"
HF_MOUNT = "/hf-cache"
IMAGE_ENV: Mapping[str, str] = {
    "PYTHONPATH": f"{WORKDIR}/src",
    "MODELS_DIR": MODELS_MOUNT,
    "HF_HOME": HF_MOUNT,
    "PYTHONUNBUFFERED": "1",
}
RUNTIMES: tuple[str, ...] = ("default", "elephant", "granite")


def check_runtime(runtime: str) -> str:
    """`runtime` if it is one of RUNTIMES, else a ValueError naming them."""
    if runtime not in RUNTIMES:
        raise ValueError(f"unknown [launch.modal] runtime {runtime!r}; use one of {', '.join(RUNTIMES)}")
    return runtime


def describe(runtime: str, *, openjev: bool) -> str:
    """One line for the dry run: what the image of `runtime` contains."""
    extra = "--extra openjev" if openjev else "(base dependencies)"
    base = f"python {PYTHON} + uv {UV_VERSION} sync --frozen {extra} + src/ configs/ at {WORKDIR}"
    return base if check_runtime(runtime) == "default" else f"{base} + runtime {runtime!r}"


def _default(modal: ModuleType, *, openjev: bool) -> object:
    if openjev:
        # GPU runs need a CUDA toolkit at /usr/local/cuda, as on the cluster nodes: FlashInfer JIT-compiles
        # its sampling kernel with nvcc during vLLM's warmup (debian_slim has no nvcc: vLLM failed to start).
        # 13.0 matches torch 2.13.0's CUDA build and the cluster's toolkit. Triton needs gcc for its launcher.
        base = (
            modal.Image.from_registry(CUDA_DEVEL_IMAGE, add_python=PYTHON)
            .apt_install("gcc", "libc6-dev")
            .env({"CUDA_HOME": "/usr/local/cuda"})
        )
    else:
        base = modal.Image.debian_slim(python_version=PYTHON)
    return (
        base.uv_sync(
            str(REPO_ROOT), extras=["openjev"] if openjev else [], frozen=True, uv_version=UV_VERSION
        )
        .env(dict(IMAGE_ENV))
        .add_local_dir(REPO_ROOT / "src", f"{WORKDIR}/src", ignore=["**/__pycache__", "**/*.pyc"])
        .add_local_dir(REPO_ROOT / "configs", f"{WORKDIR}/configs")
    )


def image(modal: ModuleType, *, runtime: str = "default", openjev: bool) -> object:
    """The modal.Image for a run of `runtime` (`openjev`: the GPU stack, else a CPU-only image)."""
    if check_runtime(runtime) == "default":
        return _default(modal, openjev=openjev)
    raise NotImplementedError(f"runtime {runtime!r} is not built yet (workstream m3-images)")
