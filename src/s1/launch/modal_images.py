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
               FlashInfer's JIT, as on the cluster nodes), or on debian_slim for CPU-only runs. GPU images
               also carry Nsight Compute and Nsight Systems (NSIGHT: the CUDA redist archives
               scripts/modal_probe.py uses, sha256-checked) as `ncu`/`nsys` on PATH and at
               /usr/local/cuda/bin/{ncu,nsys}, which is `s1 ncu`'s default `--ncu`. Both images have git
               (kev's commit check, `modal_stage fetch-code`).
    elephant   default + runtimes/elephant's own uv project (torch 2.8, Python 3.11) at ELEPHANT_DIR, which
               is where backends/elephant.py looks by default (<repo>/runtimes/elephant), so a config needs
               no setting; `elephant_python = "<ELEPHANT_PYTHON>"` names it explicitly. Built as setup.sh
               does: OneJev at ONEJEV_COMMIT + elephant-1's serve.patch (fetched from the Hub at the catalog
               revision and refused unless it is the reviewed file), `uv sync --frozen`, and causal-conv1d
               1.7.0 from its prebuilt release wheel (pinned by sha256) instead of setup.sh's best effort.
    granite    default + mamba-ssm and causal-conv1d (GRANITE_KERNELS) for Granite-4.0-H's Mamba2 layers.
               Neither project publishes a wheel for torch 2.13, so both are compiled with nvcc 13.0 in the
               image from their hash-pinned PyPI sdists (upstream's own gencode list: sm_75 to sm_121). Only
               this runtime has them: on the default image transformers runs LFM2's short convolutions and
               Granite's scan in plain torch, as on the cluster.

`modal` is imported by the caller and passed in, so importing this module never needs it. Build one image
without running anything (the coordinator's step; it rents no GPU):

    uv run --group modal python -m s1.launch.modal_images build <runtime> [--cpu]
"""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

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
UV_VENV = "/.uv/.venv"  # where Image.uv_sync puts the harness's environment (modal_app.VENV_PYTHON)
APT_COMMON = ("git", "ca-certificates")  # git: kev's check_git_commit and modal_stage's checkouts


@dataclass(frozen=True)
class Pinned:
    """A file fetched at image build time, refused unless its sha256 matches."""

    url: str
    sha256: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[1].replace("%2B", "+")


def fetch(pin: Pinned, dest: str) -> str:
    """Shell: download `pin` to `dest` and check its sha256 (the build fails on a mismatch)."""
    return (
        f"curl -fsSL --retry 3 -o {shlex.quote(dest)} {shlex.quote(pin.url)}"
        f" && echo {shlex.quote(f'{pin.sha256}  {dest}')} | sha256sum -c -"
    )


# ---- Nsight Compute / Nsight Systems (every GPU image) ----

# CUDA redist 13.4.3 (redistrib_13.4.3.json), the archives scripts/modal_probe.py ran on the Modal L4 and H100
# (a test keeps the two in step). Each unpacks (first path component stripped) into NSIGHT_DIR/<tool>; the
# launcher's path inside it is the third field (the probe found ncu at the top level and nsys in bin/).
REDIST = "https://developer.download.nvidia.com/compute/cuda/redist/"
NSIGHT: Mapping[str, tuple[str, str, str]] = {
    "ncu": (
        "nsight_compute/linux-x86_64/nsight_compute-linux-x86_64-2026.3.1.2-archive.tar.xz",
        "a0039be47cfe551f89d82087256080a9f33c6257d99a72c1164a87466ab072aa",
        "ncu",
    ),
    "nsys": (
        "nsight_systems/linux-x86_64/nsight_systems-linux-x86_64-2026.3.2.476-archive.tar.xz",
        "af8daa081e7333e3fb4a7fc3505e379593f05a8014a703f268560cec22338aca",
        "bin/nsys",
    ),
}
NSIGHT_DIR = "/opt/nvidia"
NSIGHT_BIN = f"{NSIGHT_DIR}/bin"  # wrapper scripts; linked from /usr/local/bin and /usr/local/cuda/bin
NSIGHT_LINKS = (
    "/usr/local/bin",
    "/usr/local/cuda/bin",
)  # on PATH; /usr/local/cuda/bin/ncu is `s1 ncu`'s default


def nsight_commands() -> list[str]:
    """Shell steps installing NSIGHT: fetch + sha256, unpack, a wrapper per tool, links onto PATH."""
    cmds: list[str] = [f"mkdir -p {NSIGHT_BIN}"]
    for tool, (rel, sha, exe) in NSIGHT.items():
        archive, dest = f"/tmp/{tool}.tar.xz", f"{NSIGHT_DIR}/{tool}"
        launcher = f"{dest}/{exe}"
        wrapper = f"{NSIGHT_BIN}/{tool}"
        cmds += [
            fetch(Pinned(REDIST + rel, sha), archive),
            f"mkdir -p {dest} && tar -xJf {archive} -C {dest} --strip-components=1 && rm {archive}",
            f"test -x {launcher}",
            f"echo '#!/bin/sh' > {wrapper} && echo 'exec \"{launcher}\" \"$@\"' >> {wrapper}"
            f" && chmod 755 {wrapper}",
            *(f"ln -sf {wrapper} {d}/{tool}" for d in NSIGHT_LINKS),
        ]
    cmds.append("ncu --version && nsys --version && /usr/local/cuda/bin/ncu --version")
    return cmds


# ---- elephant: runtimes/elephant (torch 2.8) ----

ELEPHANT_SRC = REPO_ROOT / "runtimes" / "elephant"
ELEPHANT_DIR = f"{WORKDIR}/runtimes/elephant"  # backends/elephant.py's RUNTIME inside the container
ELEPHANT_PYTHON = f"{ELEPHANT_DIR}/.venv/bin/python"
ELEPHANT_PYTHON_VERSION = "3.11"  # runtimes/elephant/pyproject.toml requires-python
UV_PYTHON_DIR = "/opt/uv-python"  # uv-managed interpreters (elephant's 3.11)
ONEJEV_REPO = "https://github.com/OmniJev/OneJev.git"
ONEJEV_COMMIT = (
    "81ce62f1597c91e46767d4d02d6ac2e18534fe94"  # = setup.sh ONEJEV = catalog ELEPHANT.code_revision
)
ELEPHANT_REVISION = "b3f8a93462fb1051a9de031e94a7ff380e254072"  # catalog ELEPHANT.revision
SERVE_PATCH = Pinned(  # setup.sh PATCH_SHA256, reviewed 2026-11-04; public on the Hub
    f"https://huggingface.co/baselabs/elephant-1/resolve/{ELEPHANT_REVISION}/serve.patch",
    "f234c125f1d38d82104b9e1f24e75bde51e4401928acfba2fbf5c59ef2fb46b8",
)
ELEPHANT_CAUSAL_CONV1D = Pinned(  # release asset digest from the GitHub API (v1.7.0)
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/"
    "causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
    "b28c3657a9d6855e1f95bc912ab94f893434276b437dec8008c92c9f38b6a1f4",
)


def elephant_commands() -> list[str]:
    """Shell steps building runtimes/elephant at ELEPHANT_DIR (pyproject.toml and uv.lock already copied)."""
    onejev = f"{ELEPHANT_DIR}/OneJev"
    wheel = f"/tmp/{ELEPHANT_CAUSAL_CONV1D.filename}"
    uv_env = f"UV_PROJECT_ENVIRONMENT={ELEPHANT_DIR}/.venv UV_PYTHON_INSTALL_DIR={UV_PYTHON_DIR}"
    check = (
        "import torch, transformers, causal_conv1d, qev.readout; "
        'print("elephant runtime: torch", torch.__version__, "cuda", torch.version.cuda, '
        '"transformers", transformers.__version__, "causal_conv1d", causal_conv1d.__version__)'
    )
    return [
        fetch(SERVE_PATCH, "/tmp/serve.patch"),
        f"git clone --quiet {ONEJEV_REPO} {onejev}",
        f"git -C {onejev} fetch --quiet origin {ONEJEV_COMMIT}",
        f"git -C {onejev} -c advice.detachedHead=false checkout --quiet -f {ONEJEV_COMMIT}",
        f'test "$(git -C {onejev} rev-parse HEAD)" = {ONEJEV_COMMIT}',
        f"git -C {onejev} apply /tmp/serve.patch && rm /tmp/serve.patch",
        f"cd {ELEPHANT_DIR} && {uv_env} uv sync --frozen --python {ELEPHANT_PYTHON_VERSION}",
        fetch(ELEPHANT_CAUSAL_CONV1D, wheel),
        f"uv pip install --python {ELEPHANT_PYTHON} --no-deps {shlex.quote(wheel)}",
        f"rm {shlex.quote(wheel)}",
        f"{ELEPHANT_PYTHON} -c {shlex.quote(check)}",
        "rm -rf /root/.cache/uv",
    ]


# ---- granite: mamba-ssm + causal-conv1d for torch 2.13 / CUDA 13.0 / Python 3.12 ----

# PyPI sdists (sha256 from PyPI's JSON API). mamba-ssm 2.2.6.post3, not 2.3.x: 2.3's package __init__ imports
# Mamba3, which needs tilelang, apache-tvm-ffi and quack-kernels; 2.2.6.post3 already knows CUDA 13 (drops the
# pre-sm_75 targets nvcc 13 refuses). `wheel` is a build requirement of both setup.py files (not in uv.lock).
GRANITE_BUILD_REQS: Mapping[str, Pinned] = {
    "wheel": Pinned(
        "https://files.pythonhosted.org/packages/2e/29/69cfbb602cd91690c55d38ba9fe53e6a7e76a6fa647bf38f19c138d25449/"
        "wheel-0.48.0-py3-none-any.whl",
        "3217dcc807155e45db462d7ef2431f5ddda0d7273b700d05a67b271ceb1287ab",
    ),
}
GRANITE_KERNELS: Mapping[str, Pinned] = {
    "causal-conv1d": Pinned(
        "https://files.pythonhosted.org/packages/f4/d1/eba27735f31bd5527d39a3c7f693e6ded7b844cab275db719a15ad37b6cd/"
        "causal_conv1d-1.7.0.tar.gz",
        "3202758494eaa7b597ce1c282dfa188889506bfcb92cad3c407d26736bfcd32b",
    ),
    "mamba-ssm": Pinned(
        "https://files.pythonhosted.org/packages/b6/0c/9373a469ff7a33bdd0644e55fa45165ba3900274dcf7fe9f10ccc232aef9/"
        "mamba_ssm-2.2.6.post3.tar.gz",
        "826a3cdb651959f191dac64502f8a29627d9116fe6bb7c57e4f562da1aea7bf3",
    ),
}
GRANITE_BUILD_ENV: Mapping[str, str] = {
    "CAUSAL_CONV1D_FORCE_BUILD": "TRUE",  # never look for a prebuilt wheel (there is none for torch 2.13)
    "MAMBA_FORCE_BUILD": "TRUE",
    "MAX_JOBS": "4",  # ninja jobs; each nvcc also runs 4 threads (setup.py's --threads 4)
    # Modal's add_python is a clang-built CPython, so torch's extension builder defaults to clang++, which the
    # image does not have (the first build failed: "clang++ (0.0.0) is less than the minimum ... CUDA 13.0").
    "CC": "gcc",
    "CXX": "g++",
}


def granite_commands() -> list[str]:
    """Shell steps compiling GRANITE_KERNELS against the harness's torch (UV_VENV), no dependency changes."""
    py = f"{UV_VENV}/bin/python"
    build = "/tmp/granite-build"
    files = {
        name: f"{build}/{pin.filename}" for name, pin in {**GRANITE_BUILD_REQS, **GRANITE_KERNELS}.items()
    }
    env = " ".join(f"{k}={v}" for k, v in GRANITE_BUILD_ENV.items())
    kernels = " ".join(files[name] for name in GRANITE_KERNELS)  # causal-conv1d first
    check = (
        "import torch, causal_conv1d, causal_conv1d_cuda, selective_scan_cuda, mamba_ssm; "
        "from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; "
        'print("granite kernels: torch", torch.__version__, "cuda", torch.version.cuda, '
        '"mamba_ssm", mamba_ssm.__version__, "causal_conv1d", causal_conv1d.__version__)'
    )
    return [
        f"mkdir -p {build}",
        *(fetch(pin, files[name]) for name, pin in {**GRANITE_BUILD_REQS, **GRANITE_KERNELS}.items()),
        f"uv pip install --python {py} --no-deps {files['wheel']}",
        f"{env} uv pip install --python {py} --no-deps --no-build-isolation {kernels}",
        f"{py} -c {shlex.quote(check)}",
        f"rm -rf {build} /root/.cache/uv",
    ]


# ---- assembling the images ----


def check_runtime(runtime: str) -> str:
    """`runtime` if it is one of RUNTIMES, else a ValueError naming them."""
    if runtime not in RUNTIMES:
        raise ValueError(f"unknown [launch.modal] runtime {runtime!r}; use one of {', '.join(RUNTIMES)}")
    return runtime


_EXTRA: Mapping[str, str] = {
    "elephant": f"runtimes/elephant (torch 2.8, OneJev@{ONEJEV_COMMIT[:8]} + serve.patch) at {ELEPHANT_DIR}",
    "granite": "mamba-ssm 2.2.6.post3 + causal-conv1d 1.7.0 (built with nvcc)",
}


def describe(runtime: str, *, openjev: bool) -> str:
    """One line for the dry run: what the image of `runtime` contains."""
    extra = "--extra openjev" if openjev else "(base dependencies)"
    base = f"python {PYTHON} + uv {UV_VERSION} sync --frozen {extra} + src/ configs/ at {WORKDIR}"
    if openjev:
        base += " + ncu/nsys"
    return base if check_runtime(runtime) == "default" else f"{base} + runtime {runtime!r}: {_EXTRA[runtime]}"


def _synced(modal: ModuleType, *, openjev: bool) -> Any:  # noqa: ANN401  (a modal.Image; modal is untyped here)
    """The base image up to the harness's environment (everything but the local files)."""
    if openjev:
        # GPU runs need a CUDA toolkit at /usr/local/cuda, as on the cluster nodes: FlashInfer JIT-compiles
        # its sampling kernel with nvcc during vLLM's warmup (debian_slim has no nvcc: vLLM failed to start).
        # 13.0 matches torch 2.13.0's CUDA build and the cluster's toolkit. Triton needs gcc for its launcher.
        # Nsight comes before the uv sync so a lock change does not refetch it; uv (pinned) is for the
        # runtimes' own installs (elephant's venv, granite's kernels).
        base = (
            modal.Image.from_registry(CUDA_DEVEL_IMAGE, add_python=PYTHON)
            .apt_install("gcc", "libc6-dev", "curl", "xz-utils", *APT_COMMON)
            .env({"CUDA_HOME": "/usr/local/cuda"})
            .run_commands(*nsight_commands())
            .pip_install(f"uv=={UV_VERSION}")
        )
    else:
        base = modal.Image.debian_slim(python_version=PYTHON).apt_install(*APT_COMMON)
    return base.uv_sync(
        str(REPO_ROOT), extras=["openjev"] if openjev else [], frozen=True, uv_version=UV_VERSION
    ).env(dict(IMAGE_ENV))


def _with_local(img: Any, runtime: str) -> Any:  # noqa: ANN401
    """The local files, last (Modal mounts them at start, so editing src/ rebuilds nothing)."""
    img = img.add_local_dir(REPO_ROOT / "src", f"{WORKDIR}/src", ignore=["**/__pycache__", "**/*.pyc"])
    img = img.add_local_dir(REPO_ROOT / "configs", f"{WORKDIR}/configs")
    if runtime == "elephant":
        img = img.add_local_file(ELEPHANT_SRC / "worker.py", f"{ELEPHANT_DIR}/worker.py")
    return img


def image(modal: ModuleType, *, runtime: str = "default", openjev: bool) -> object:
    """The modal.Image for a run of `runtime` (`openjev`: the GPU stack, else a CPU-only image)."""
    check_runtime(runtime)
    if runtime != "default" and not openjev:
        raise ValueError(f"runtime {runtime!r} is GPU-only; it needs openjev = true (the CUDA image)")
    img = _synced(modal, openjev=openjev)
    if runtime == "elephant":
        img = (
            img.add_local_file(ELEPHANT_SRC / "pyproject.toml", f"{ELEPHANT_DIR}/pyproject.toml", copy=True)
            .add_local_file(ELEPHANT_SRC / "uv.lock", f"{ELEPHANT_DIR}/uv.lock", copy=True)
            .run_commands(*elephant_commands())
        )
    elif runtime == "granite":
        img = img.apt_install("g++").run_commands(*granite_commands())  # g++: torch's C++ extension build
    return _with_local(img, runtime)


# ---- `python -m s1.launch.modal_images build <runtime> [--cpu]` ----


def build(runtime: str, *, openjev: bool) -> None:
    """Build (and cache) one image without running anything, like modal_app.build_image."""
    image_runtime = check_runtime(runtime)
    from . import modal_app  # noqa: PLC0415  (modal_app imports this module)

    modal_app._check_python()
    modal = modal_app._modal()
    app = modal.App(modal_app.APP_NAME)
    with modal.enable_output(), app.run():
        img: Any = image(modal, runtime=image_runtime, openjev=openjev)
        img.build(app)  # instance method: Image.build(self, app)
    print(f"built image for runtime {runtime!r}: {describe(runtime, openjev=openjev)}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m s1.launch.modal_images", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build one runtime's image on Modal (no GPU, no run)")
    b.add_argument("runtime", choices=RUNTIMES)
    b.add_argument(
        "--cpu", action="store_true", help="the CPU-only image (debian_slim; default runtime only)"
    )
    return p


def main(argv: Sequence[str] | None = None, *, run: Callable[..., None] = build) -> None:
    args = parser().parse_args(argv)
    if args.cpu and args.runtime != "default":
        parser().error(f"runtime {args.runtime!r} is GPU-only; --cpu builds the default runtime only")
    run(args.runtime, openjev=not args.cpu)


if __name__ == "__main__":
    main(sys.argv[1:])
