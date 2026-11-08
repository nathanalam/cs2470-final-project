"""`s1` command line: run experiments locally or via sbatch, and inspect results."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import config, ncu, report, runner
from .models.catalog import CATALOG
from .record import RESULTS_DIR, RunRecord, Scalar, load_all

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

ResultsDir = typer.Option(RESULTS_DIR, "--results-dir", "-o", help="Where run records are written/read.")


def _scalar(text: str) -> Scalar:
    """A `--set` value: true/false, an int, a float, else the string."""
    if text in ("true", "false"):
        return text == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


@app.command()
def run(
    cfg: Path,
    results_dir: Path = ResultsDir,
    index: int | None = typer.Option(None, "--index", help="Run only this combination (0-based)."),
    sets: list[str] = typer.Option([], "--set", help="Override a param: key=value (repeatable)."),
) -> None:
    """Run every combination in CONFIG in this process (use inside an allocation)."""
    specs = config.load(cfg)
    if index is not None:
        specs = [specs[index]]
    overrides = {k: _scalar(v) for k, _, v in (s.partition("=") for s in sets)}
    console.print(f"{cfg}: {len(specs)} run(s)")
    for spec in specs:
        spec.params |= overrides
        runner.run(spec, config_path=cfg, results_dir=results_dir, log=console.print)


@app.command("ncu")
def ncu_cmd(
    cfg: Path,
    questions: int = typer.Option(4, "--questions", help="Questions profiled per model."),
    results_dir: Path = ResultsDir,
    ncu_bin: str = typer.Option("/usr/local/cuda/bin/ncu", "--ncu", help="Nsight Compute CLI."),
) -> None:
    """Run each model of CONFIG under Nsight Compute (use inside an allocation): the roofline benchmark
    re-answers `--questions` questions in a CUDA profiler range, and the summed hardware counters (FLOPs,
    DRAM bytes) are added to that run's record as roofline_hw_*. See s1/ncu.py."""
    specs = config.load(cfg)
    job = os.environ.get("SLURM_JOB_ID", "local")
    Path("logs").mkdir(exist_ok=True)
    for i in range(len(specs)):
        log = Path("logs") / f"ncu-{job}-{cfg.stem}-{i}.csv"
        before = set(results_dir.glob("*.json"))
        child = [sys.executable, "-m", "s1.cli", "run", str(cfg), "--index", str(i)]
        child += ["--results-dir", str(results_dir), "--set", f"roofline_ncu={questions}"]
        subprocess.run([ncu_bin, *ncu.ARGS, "--log-file", str(log), *child], check=True)
        (path,) = set(results_dir.glob("*.json")) - before
        rec = RunRecord.load(path)
        hw = ncu.summarize(ncu.parse(log.read_text()), rec.results["roofline"])
        rec.results["roofline"] |= hw
        rec.meta["ncu_log"] = str(log)
        rec.save(results_dir)
        console.print(f"[{rec.run_id}] hardware counters: {hw}")


@app.command()
def submit(
    cfg: Path,
    *,
    partition: str | None = typer.Option(None, "-p", "--partition", help="Default: cluster default."),
    gpus: int = typer.Option(1, "-g", "--gpus"),
    gpu_type: str | None = typer.Option(None, "--gpu-type", help="GRES type, e.g. l4 (gpu-spec types them)."),
    cpus: int = typer.Option(8, "-c", "--cpus"),
    mem: str = typer.Option("32G", "--mem"),
    time: str = typer.Option("01:00:00", "-t", "--time"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the sbatch command only."),
    ncu_questions: int = typer.Option(0, "--ncu", help="Queue `s1 ncu` with this many questions instead."),
) -> None:
    """Queue `s1 run CONFIG` (or `s1 ncu CONFIG`) as a batch job; logs go to logs/<jobid>.out."""
    config.load(cfg)  # fail fast on a bad config before queueing
    cmd = ["sbatch", "-J", f"s1-{cfg.stem}", "-c", str(cpus), f"--mem={mem}", "-t", time]
    cmd += ["-o", "logs/%j.out"]
    if partition is not None:
        cmd += ["-p", partition]
    if gpus:
        cmd += [f"--gres=gpu:{'' if gpu_type is None else gpu_type + ':'}{gpus}"]
    s1 = ["ncu", str(cfg), "--questions", str(ncu_questions)] if ncu_questions else ["run", str(cfg)]
    cmd += ["--wrap", shlex.join(["uv", "run", "--extra", "openjev", "s1", *s1])]
    console.print(shlex.join(cmd))
    if not dry_run:
        Path("logs").mkdir(exist_ok=True)
        subprocess.run(cmd, check=True)


@app.command("ls")
def ls(results_dir: Path = ResultsDir) -> None:
    """Table of runs: params and each benchmark's headline metric."""
    recs = load_all(results_dir)
    if not recs:
        console.print(f"no runs in {results_dir}/")
        return
    param_keys = sorted({k for r in recs for k in r.params})
    benches = sorted({b for r in recs for b in r.results})
    t = Table(header_style="bold")
    for col in ["run_id", "gpu", *param_keys, *benches]:
        t.add_column(col, no_wrap=True)
    for r in recs:
        headline = [_headline(r.results.get(b)) for b in benches]
        params = [str(r.params.get(k, "")) for k in param_keys]
        t.add_row(r.run_id, str(r.invariants.get("gpu", "")), *params, *headline)
    console.print(t)


def _headline(metrics: dict[str, float] | None) -> str:
    """First metric, with its _std alongside when present."""
    if metrics is None or not metrics:
        return ""
    key, val = next(iter(metrics.items()))
    std = metrics.get(f"{key}_std")
    return f"{val:.4g}" + (f" ± {std:.2g}" if std is not None else "")


@app.command("models")
def models_cmd() -> None:
    """Table of the model catalog (pins, sizes, status); reads metadata only."""
    t = Table(header_style="bold")
    for col in ["name", "group", "status", "repo@rev", "GB", "license", "notes"]:
        t.add_column(col, no_wrap=col != "notes")
    for m in CATALOG:
        gb = "" if m.weights_gb is None else f"{m.weights_gb:.1f}"
        t.add_row(
            m.name, m.group, m.status, f"{m.repo}@{(m.revision or 'unpinned')[:8]}", gb, m.license, m.notes
        )
    console.print(t)


@app.command()
def collect(
    results_dir: Path = ResultsDir,
    out: Path | None = typer.Option(None, "--out", help="Write here instead of stdout."),
) -> None:
    """Merge all run records into one {run_id: record} JSON document."""
    merged: dict[str, object] = {}
    for r in load_all(results_dir):
        merged |= r.to_json()
    text = json.dumps(merged, indent=4)
    if out is None:
        print(text)
    else:
        out.write_text(text + "\n")
        console.print(f"wrote {len(merged)} run(s) to {out}")


@app.command("report")
def report_cmd(results_dir: Path = ResultsDir) -> None:
    """Regenerate results/REPORT.md and results/report.html from every run record."""
    md, page = report.write(results_dir)
    console.print(f"wrote {md} and {page}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
