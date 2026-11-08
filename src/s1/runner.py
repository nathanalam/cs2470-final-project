"""Execute a RunSpec: build the model, run each benchmark, write the record."""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from . import env, models, registry
from .config import RunSpec
from .gpu_stats import SCHEMA, GpuSampler, metrics_mode, resolve_mode
from .record import RESULTS_DIR, RunRecord, Scalar, new_run_id
from .registry import get_int, get_str


def detect_gpu() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "cpu"
    names = sorted({line.strip() for line in out.splitlines() if line.strip()})
    return ",".join(names) or "cpu"


def _run(cmd: list[str]) -> str | None:
    """stdout of `cmd`, or None if the tool is missing, fails or hangs: a missing tool never fails a run."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_model() -> str | None:
    """Host CPU model: /proc/cpuinfo "model name" (Linux), else sysctl's brand string (macOS), else None."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, val = line.partition(":")
            if key.strip() == "model name" and val.strip():
                return val.strip()
    except OSError:
        pass
    out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return (out or "").strip() or None


def hardware_meta() -> dict[str, Scalar]:
    """CPU model, GPU 0's driver and power limit, saved `nvidia-smi -q` path; absent fields omitted."""
    meta: dict[str, Scalar] = {}
    if (cpu := cpu_model()) is not None:
        meta[env.META_CPU_MODEL] = cpu
    out = _run(["nvidia-smi", "--query-gpu=driver_version,power.limit", "--format=csv,noheader,nounits"])
    if out is not None and (rows := [r for r in out.splitlines() if r.strip()]):
        driver, _, power = (f.strip() for f in rows[0].partition(","))
        if driver and driver != "[N/A]":
            meta[env.META_GPU_DRIVER] = driver
        with contextlib.suppress(ValueError):  # "[N/A]" or empty
            meta[env.META_GPU_POWER_LIMIT_W] = float(power)
    if (q := _run(["nvidia-smi", "-q"])) is not None:
        path = env.log_dir() / f"nvidia-smi-q-{env.job_id()}.txt"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(q)
        except OSError:
            pass
        else:
            meta[env.META_NVIDIA_SMI_Q_LOG] = str(path)
    return meta


def orchestrator_meta() -> dict[str, Scalar]:
    """Who ran this: orchestrator on every record, `slurm_job_id` (as before) or the Modal task id."""
    orch = env.orchestrator()
    meta: dict[str, Scalar] = {env.META_ORCHESTRATOR: orch}
    if orch == "slurm":
        meta["slurm_job_id"] = env.job_id()
    elif orch == "modal":
        meta[env.META_MODAL_TASK_ID] = env.job_id()
    return meta


def grid_settings(grid: dict[str, list[Scalar]]) -> dict[str, Scalar]:
    """Expose each grid axis to benchmarks as `grid_<axis>` = comma-joined values ("32,128,512")."""
    return {f"grid_{k}": ",".join(str(v) for v in vals) for k, vals in grid.items()}


def run(
    spec: RunSpec,
    *,
    config_path: Path | None = None,
    results_dir: Path = RESULTS_DIR,
    log: Callable[[str], None] = print,
    gpu_sampler: Callable[[], GpuSampler] = GpuSampler,
) -> RunRecord:
    invariants = dict(spec.invariants)
    if invariants.get("gpu", "auto") == "auto":
        invariants["gpu"] = detect_gpu()
    settings0 = invariants | spec.params
    gpu_mode = resolve_mode(
        get_str(settings0, "gpu_metrics", "basic"), get_int(settings0, "gpu_metrics_schema", SCHEMA)
    )
    if gpu_mode == "detailed":  # basic records stay byte-identical; detailed ones state their key set
        invariants.setdefault("gpu_metrics_schema", SCHEMA)
    # absent means the HUIT cluster, so Slurm and local runs keep exactly today's invariants
    if env.orchestrator() == "modal":
        invariants[env.INV_PLATFORM] = "modal"

    meta: dict[str, Scalar] = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "git_commit": env.git_commit(),
        "host": os.uname().nodename,
    }
    if config_path is not None:
        meta["config"] = str(config_path)
    meta |= orchestrator_meta() | hardware_meta()

    rec = RunRecord(
        run_id=new_run_id(),
        invariants=invariants,
        params=dict(spec.params),
        meta=meta,
        grid={k: list(v) for k, v in spec.grid.items()},
    )
    log(f"[{rec.run_id}] params={rec.params}")

    settings = rec.invariants | rec.params | grid_settings(rec.grid)
    t0 = time.perf_counter()
    model = registry.get_model(str(spec.params["model"]))(settings)
    for key, val in models.spec_params(str(spec.params["model"])).items():
        rec.params.setdefault(key, val)  # resolved spec values (revision, options); config wins
    if callable(run_info := getattr(model, "run_info", None)):
        for key, val in run_info().items():
            rec.invariants.setdefault(key, val)  # software the model actually ran with (versions, kernels)
    meta["load_s"] = round(time.perf_counter() - t0, 3)
    try:
        with metrics_mode(gpu_mode):
            for name in spec.benchmarks:
                t = time.perf_counter()
                if (warm := registry.get_warmup(name)) is not None and warm(model, settings):
                    meta[f"{name}_warmup_s"] = round(time.perf_counter() - t, 3)
                t = time.perf_counter()
                with gpu_sampler() as smi:
                    rec.results[name] = registry.get_benchmark(name)(model, settings)
                meta[f"{name}_s"] = round(time.perf_counter() - t, 3)
                if gpu := smi.metrics():
                    rec.results[f"{name}/nvidia_smi"] = gpu
                log(f"[{rec.run_id}] {name}: {rec.results[name]}")
    finally:
        if isinstance(model, registry.Closeable):
            model.close()

    meta["duration_s"] = round(time.perf_counter() - t0, 3)
    path = rec.save(results_dir)
    log(f"[{rec.run_id}] wrote {path}")
    return rec
