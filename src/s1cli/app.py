"""s1 — interactive SLURM/tmux helper for the System One profiling project."""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from . import gpus, menus, slurm, state, tmux
from .prompts import Aborted, confirm, select

console = Console()
app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    help="Cluster helper: inspect partitions, allocate GPUs in tmux, monitor jobs.",
)


def _err(msg: str) -> None:
    console.print(f"[bold red]error:[/] {msg}")


def _interactive() -> bool:
    return os.isatty(0) and os.isatty(1)


# ------------------------------------------------------------------ info


def _partitions() -> list[slurm.Partition]:
    with console.status("Querying sinfo…"):
        return slurm.partitions()


def show_info() -> None:
    parts = _partitions()
    t = Table(title=f"Partitions on {state.hostname()}", header_style="bold")
    t.add_column("Partition", no_wrap=True)
    t.add_column("GPU/node", no_wrap=True)
    t.add_column("VRAM", justify="right")
    t.add_column("CPU", justify="right")
    t.add_column("Mem", justify="right")
    t.add_column("Max time")
    t.add_column("ready", justify="right", style="green")
    t.add_column("cloud", justify="right", style="cyan")
    t.add_column("busy", justify="right", style="yellow")
    t.add_column("down", justify="right", style="red")
    for p in parts:
        s = p.states
        name = p.name + (" [dim](default)[/]" if p.default else "")
        if p.avail.lower() != "up":
            name += f" [red]({p.avail})[/]"
        t.add_row(
            name,
            p.gpu_label(),
            gpus.fmt_vram(p.vram_gb) if p.has_gpu else "-",
            str(p.cpus or "?"),
            f"{p.mem_mb / 1024:.0f}G" if p.mem_mb else "?",
            p.timelimit,
            str(s["idle"] or ""),
            str((s["cloud"] + s["booting"]) or ""),
            str(s["busy"] or ""),
            str(s["down"] or ""),
        )
    console.print(t)
    console.print(
        "[dim]ready = powered on and idle · [cyan]cloud[/cyan] = powered down / booting, "
        "starts on demand (~5-10 min) · [red]down[/red] on AWS usually means no capacity "
        "right now, not broken hardware.[/]"
    )


# ------------------------------------------------------------------ alloc


def _alloc_summary(a: slurm.Alloc, p: slurm.Partition | None) -> None:
    lines = [f"[bold]{shlex.join(a.srun_cmd())}[/]"]
    if p is not None:
        if p.states["idle"] == 0 and (p.states["cloud"] or p.states["booting"]):
            lines.append("[cyan]No powered-on idle nodes: a cloud node will boot (~5-10 min).[/]")
        if p.states["down"] and p.states["down"] == p.nodes:
            lines.append("[red]All nodes are down — AWS may be out of capacity; job may queue.[/]")
        vram = p.vram_gb
        if p.has_gpu and vram is not None:
            lines.append(f"VRAM available: {a.gpus} x {vram} GB = {a.gpus * vram} GB")
    console.print(Panel("\n".join(lines), title=f"Allocation {a.name}", expand=False))


def launch_alloc(a: slurm.Alloc, *, dry_run: bool = False) -> None:
    srun = shlex.join(a.srun_cmd())
    if dry_run:
        console.print(srun)
        return
    if slurm.fake_mode():
        console.print(f"[yellow]\\[fake][/] would launch in tmux session [bold]{a.name}[/]: {srun}")
        return
    shell_cmd = (
        f"echo {shlex.quote('[s1] ' + srun)}; "
        "echo '[s1] waiting for resources (cloud nodes can take 5-10 min). Detach anytime: Ctrl-b d'; "
        f"{srun}; "
        'echo; echo "[s1] allocation ended (exit $?)."; exec bash -l'
    )
    if not tmux.available():
        console.print("[yellow]tmux not found[/] — the allocation will die if this terminal disconnects.")
        if _interactive() and confirm("Run srun directly anyway?", default=False):
            os.execvp(a.srun_cmd()[0], a.srun_cmd())
        return
    target = tmux.launch(a.name, shell_cmd)
    state.record_session(a.name, target, srun)
    console.print(
        f"[green]Started[/] in tmux target [bold]{target}[/] on login node [bold]{state.hostname()}[/].\n"
        f"Reattach later with [bold]uv run s1 attach[/] (or `tmux attach -t {target.split(':')[0]}`)."
    )
    if _interactive():
        tmux.attach(target)


def do_alloc(
    *,
    partition: str | None = None,
    gpus_: int | None = None,
    gpu_type: str | None = None,
    cpus: int | None = None,
    mem: str | None = None,
    time_: str | None = None,
    dry_run: bool = False,
    yes: bool = False,
) -> None:
    parts = _partitions()
    by_name = {p.name: p for p in parts}
    if partition is not None:
        # Non-interactive path: flags fully specify the request.
        if partition not in by_name:
            raise slurm.SlurmError(f"unknown partition {partition!r}; have: {', '.join(by_name)}")
        p = by_name[partition]
        a = slurm.Alloc(
            partition=partition,
            gpus=gpus_ if gpus_ is not None else (1 if p.has_gpu else 0),
            gpu_type=gpu_type,
            cpus=cpus if cpus is not None else 8,
            mem=mem if mem is not None else "32G",
            time=time_ if time_ is not None else "01:00:00",
            name=f"{slurm.JOB_PREFIX}{partition}-{time.strftime('%H%M')}",
        )
    else:
        a = menus.alloc_wizard(parts)
        p = by_name.get(a.partition)
    _alloc_summary(a, p)
    if not (yes or dry_run) and _interactive() and not confirm("Submit this allocation?"):
        return
    launch_alloc(a, dry_run=dry_run)


# ------------------------------------------------------------------ jobs


def _job_windows() -> dict[str, str]:
    """Map s1 job names -> tmux targets on this host."""
    out: dict[str, str] = {}
    for w in tmux.s1_windows():
        out[w.name if w.name.startswith(tmux.PREFIX) else w.session] = w.target
    return out


def show_jobs(*, pick: bool = True) -> None:
    with console.status("Querying squeue…"):
        jobs = slurm.my_jobs()
    if not jobs:
        console.print(f"No jobs for [bold]{slurm.user()}[/].")
        return
    wins = _job_windows()
    t = Table(title=f"Jobs for {slurm.user()}", header_style="bold")
    for col in ("JobID", "Partition", "Name", "State", "Reason", "Elapsed", "Limit", "Node", "GRES", "tmux"):
        t.add_column(col)
    for j in jobs:
        color = {"RUNNING": "green", "PENDING": "yellow"}.get(j.state, "white")
        gres = j.gres.replace("gres/", "").replace("gres:", "")
        t.add_row(
            j.id,
            j.partition,
            j.name,
            f"[{color}]{j.state}[/]",
            "" if j.reason == "None" else j.reason,
            j.elapsed,
            j.timelimit,
            j.nodes,
            "" if gres == "N/A" else gres,
            wins.get(j.name, ""),
        )
    console.print(t)
    if not (pick and _interactive()):
        return

    job_opts: list[tuple[str, slurm.Job | None]] = [(f"{j.id}  {j.name}  ({j.state})", j) for j in jobs]
    job = select("Select a job", [*job_opts, ("← back", None)])
    if job is None:
        return
    actions: list[tuple[str, str | None]] = []
    if job.name in wins:
        actions.append(("Attach to its tmux window", "attach"))
    if job.running:
        actions.append(("GPU status", "gpu"))
    actions += [
        ("Show details (scontrol)", "detail"),
        ("Cancel job (scancel)", "cancel"),
        ("← back", None),
    ]
    action = select(f"Job {job.id}", actions)
    if action == "attach":
        tmux.attach(wins[job.name])
    elif action == "gpu":
        show_gpu(job_id=job.id)
    elif action == "detail":
        console.print(slurm.job_detail(job.id))
    elif action == "cancel" and confirm(f"Cancel job {job.id} ({job.name})?", default=False):
        slurm.cancel(job.id)
        console.print(f"[green]Cancelled {job.id}.[/]")
        if job.name in wins and confirm("Also close its tmux window?"):
            tmux.kill(wins[job.name])
        state.forget_session(job.name)


# ------------------------------------------------------------------ attach


def _remote_hint(rec: state.SessionRecord) -> None:
    console.print(
        f"Session {rec['name']} lives on login node [bold]{rec['host']}[/]: "
        f"run [bold]ssh {rec['host']}[/] then `uv run s1 attach {rec['name']}`."
    )


def do_attach(name: str | None = None) -> None:
    here = state.hostname()
    local = tmux.s1_windows()
    remote = [s for s in state.load_sessions() if s["host"] != here]

    if name is not None:
        match = next((w for w in local if name in (w.name, w.session, w.target)), None)
        if match is not None:
            tmux.attach(match.target)
            return
        rem = next((s for s in remote if s["name"] == name), None)
        if rem is not None:
            _remote_hint(rem)
            return
        raise slurm.SlurmError(f"no s1 tmux session named {name!r}")

    if not local and not remote:
        console.print("No s1 tmux sessions found. Start one with [bold]uv run s1 alloc[/].")
        if tmux.available() and not tmux.inside():
            others = tmux.list_windows()
            if others:
                console.print(
                    f"[dim]Other tmux sessions here: {', '.join(sorted({w.session for w in others}))}[/]"
                )
        return

    if len(local) == 1 and not remote:
        tmux.attach(local[0].target)
        return
    options: list[tuple[str, tmux.Window | state.SessionRecord]] = [
        (f"{w.name:<24} {w.target:<20} {'(attached elsewhere)' if w.attached else ''}", w) for w in local
    ]
    options += [(f"{s['name']:<24} on {s['host']}  (other login node)", s) for s in remote]
    if not _interactive():
        for title, _ in options:
            console.print(title)
        return
    val = select(f"Attach (you are on {here})", options)
    if isinstance(val, tmux.Window):
        tmux.attach(val.target)
    else:
        _remote_hint(val)


# ------------------------------------------------------------------ gpu

NVSMI = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits",
]


def _gpu_table(job_id: str | None) -> Table:
    cmd = NVSMI if job_id is None else ["srun", "--jobid", job_id, "--overlap", "-n", "1", *NVSMI]
    out = slurm.run(cmd, timeout=60)
    where = "this node" if job_id is None else f"job {job_id}"
    t = Table(title=f"GPUs ({where}) — {time.strftime('%H:%M:%S')}", header_style="bold")
    for col in ("#", "GPU", "Mem used", "Mem total", "Mem %", "Util %"):
        t.add_column(col, justify="right" if col != "GPU" else "left")
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5:
            continue
        idx, name, used, total, util = parts[:5]
        try:
            pct_s = f"{100 * float(used) / float(total):.0f}"
        except (ValueError, ZeroDivisionError):
            pct_s = "?"
        t.add_row(idx, name, f"{used} MiB", f"{total} MiB", pct_s, util)
    return t


def show_gpu(job_id: str | None = None, *, watch: bool = False, interval: float = 0) -> None:
    if job_id is None and not slurm.on_compute_node() and not slurm.fake_mode():
        running = [j for j in slurm.my_jobs() if j.running]
        if not running:
            console.print(
                "You're on a login node and have no running jobs. Allocate one with [bold]uv run s1 alloc[/]."
            )
            return
        if len(running) == 1 or not _interactive():
            job_id = running[0].id
        else:
            job_id = select("Which job?", [(f"{j.id} {j.name} on {j.nodes}", j.id) for j in running])
    interval = interval or (2 if job_id is None else 5)
    if not watch:
        console.print(_gpu_table(job_id))
        return
    with (
        contextlib.suppress(KeyboardInterrupt),
        Live(_gpu_table(job_id), console=console, refresh_per_second=1) as live,
    ):
        while True:
            time.sleep(interval)
            live.update(_gpu_table(job_id))


# ------------------------------------------------------------------ doctor


def do_doctor() -> None:
    t = Table(title="s1 doctor", header_style="bold", show_lines=False)
    t.add_column("")
    t.add_column("Check")
    t.add_column("Detail")
    ok, warn, bad = "[green]✔[/]", "[yellow]![/]", "[red]✘[/]"

    def row(status: str, check: str, detail: str = "") -> None:
        t.add_row(status, check, detail)

    def which_row(binary: str, missing_status: str, missing_detail: str) -> None:
        path = shutil.which(binary)
        row(ok if path is not None else missing_status, binary, path or missing_detail)

    user = slurm.user()
    for b in ("sinfo", "squeue", "srun", "scontrol"):
        which_row(b, bad, "not on PATH (not on the cluster?)")
    which_row("tmux", bad, "missing — allocations won't survive disconnects")
    session = tmux.current_session()
    row(
        ok if session is not None else warn,
        "inside tmux",
        f"session {session}" if session is not None else "no (s1 will create sessions for you)",
    )
    which_row("uv", warn, "not on PATH")
    if slurm.on_compute_node():
        row(ok, "node", f"compute node {state.hostname()} (job {os.environ['SLURM_JOB_ID']})")
        which_row("nvidia-smi", warn, "not found on this node")
    else:
        row(ok, "node", f"login node {state.hostname()} — don't run heavy compute here")
    scratch = Path(f"/scratch/{user}")
    row(
        ok if scratch.is_dir() else warn,
        "scratch",
        str(scratch) if scratch.is_dir() else f"{scratch} missing — run `srun --pty true` once to create it",
    )
    hf = os.environ.get("HF_HOME")
    row(
        ok if hf is not None else warn,
        "HF_HOME",
        hf or f"unset — suggest `export HF_HOME=/scratch/{user}/hf` (fast, but scratch can be wiped)",
    )
    row(ok, "python", sys.version.split()[0])
    row(ok, "config", str(state.config_dir()))
    console.print(t)


# ------------------------------------------------------------------ main menu


def _header() -> None:
    where = f"compute node (job {os.environ['SLURM_JOB_ID']})" if slurm.on_compute_node() else "login node"
    session = tmux.current_session()
    if session is not None:
        tm = f"tmux:{session}"
    elif tmux.available():
        tm = "not in tmux"
    else:
        tm = "tmux missing"
    fake = "  [yellow]FAKE SLURM[/]" if slurm.fake_mode() else ""
    console.print(Panel(f"[bold]s1[/] · {state.hostname()} · {where} · {tm}{fake}", expand=False))


def main_menu() -> None:
    _header()
    actions: list[tuple[str, Callable[[], None] | None]] = [
        ("Cluster info (partitions, GPUs, node states)", show_info),
        ("Allocate GPU (srun in tmux)", do_alloc),
        ("My jobs", show_jobs),
        ("Attach to an s1 tmux session", do_attach),
        ("GPU status", show_gpu),
        ("Doctor (environment checks)", do_doctor),
        ("Quit", None),
    ]
    while True:
        try:
            fn = select("What do you want to do?", actions)
        except Aborted:
            return
        if fn is None:
            return
        _guard(fn)
        console.print()


def _guard(fn: Callable[[], None]) -> bool:
    try:
        fn()
    except Aborted:
        console.print("[dim]cancelled[/]")
    except slurm.SlurmError as e:
        _err(str(e))
    except RuntimeError as e:  # tmux failures
        _err(str(e))
    else:
        return True
    return False


def _cli(fn: Callable[[], None]) -> None:
    if not _guard(fn):
        raise typer.Exit(1)


# ------------------------------------------------------------------ typer commands


@app.callback()
def root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        if not _interactive():
            console.print(ctx.get_help())
            raise typer.Exit
        main_menu()


@app.command()
def info() -> None:
    """Show partitions, GPU types, VRAM and node states."""
    _cli(show_info)


@app.command()
def alloc(
    *,
    partition: str | None = typer.Option(None, "--partition", "-p", help="Skip the wizard."),
    gpus_: int | None = typer.Option(None, "--gpus", "-g"),
    gpu_type: str | None = typer.Option(None, "--gpu-type"),
    cpus: int | None = typer.Option(None, "--cpus", "-c"),
    mem: str | None = typer.Option(None, "--mem"),
    time_: str | None = typer.Option(None, "--time", "-t"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the srun command only."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation."),
) -> None:
    """Allocate GPUs interactively and run the shell inside tmux."""
    _cli(
        lambda: do_alloc(
            partition=partition,
            gpus_=gpus_,
            gpu_type=gpu_type,
            cpus=cpus,
            mem=mem,
            time_=time_,
            dry_run=dry_run,
            yes=yes,
        )
    )


@app.command()
def jobs(*, no_pick: bool = typer.Option(False, "--no-pick", help="Just print the table.")) -> None:
    """List your jobs; pick one to attach, inspect or cancel."""
    _cli(lambda: show_jobs(pick=not no_pick))


@app.command()
def attach(name: str | None = typer.Argument(None)) -> None:
    """Reattach to an s1 tmux session (handles multiple login nodes)."""
    _cli(lambda: do_attach(name))


@app.command()
def gpu(
    *,
    job: str | None = typer.Option(None, "--job", "-j", help="Job ID (from a login node)."),
    watch: bool = typer.Option(False, "--watch", "-w"),
    interval: float = typer.Option(0, "--interval", "-n", help="Seconds between refreshes."),
) -> None:
    """Show GPU memory/utilization on this node or inside a running job."""
    _cli(lambda: show_gpu(job, watch=watch, interval=interval))


@app.command()
def doctor() -> None:
    """Check the environment (Slurm, tmux, uv, scratch, HF_HOME)."""
    _cli(do_doctor)


def main() -> None:
    app()
