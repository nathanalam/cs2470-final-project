"""Interactive allocation wizard."""

from __future__ import annotations

import re
import time

from . import gpus, slurm, state
from .prompts import select, text

CPU_CHOICES = [1, 2, 4, 8, 16, 32, 48, 64]
MEM_CHOICES = ["4G", "8G", "16G", "32G", "64G", "128G", "256G"]
TIME_CHOICES = [
    ("30 min", "00:30:00"),
    ("1 hour", "01:00:00"),
    ("2 hours", "02:00:00"),
    ("4 hours", "04:00:00"),
    ("8 hours", "08:00:00"),
    ("24 hours", "1-00:00:00"),
]
TIME_RE = re.compile(r"(\d+-)?\d{1,2}:\d{2}:\d{2}")


def _states_summary(p: slurm.Partition) -> str:
    s = p.states
    bits: list[str] = []
    if s["idle"]:
        bits.append(f"{s['idle']} ready")
    if s["cloud"] or s["booting"]:
        bits.append(f"{s['cloud'] + s['booting']} cloud")
    if s["busy"]:
        bits.append(f"{s['busy']} busy")
    if s["down"]:
        bits.append(f"{s['down']} down")
    return ", ".join(bits) or "no nodes"


def partition_title(p: slurm.Partition) -> str:
    name = p.name + ("*" if p.default else "")
    gpu = p.gpu_label() if p.has_gpu else "CPU only"
    vram = p.vram_gb
    vram_s = f"{vram} GB" if vram is not None else ("? GB" if p.has_gpu else "")
    return f"{name:<14} {gpu:<14} {vram_s:>6}   max {p.timelimit:<11} {_states_summary(p)}"


def _mem_gb(mem: str) -> float:
    m = re.match(r"(\d+(?:\.\d+)?)([KMGT]?)", mem.upper())
    if m is None:
        return 0
    n, unit = float(m.group(1)), m.group(2) or "M"
    return n * {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1, "T": 1024}[unit]


def _validate_time(s: str) -> bool | str:
    return bool(TIME_RE.fullmatch(s.strip())) or "Use HH:MM:SS or D-HH:MM:SS"


def _fits(choice: str, limit: int | None) -> bool:
    minutes = slurm.parse_timelimit_minutes(choice)
    return limit is None or (minutes is not None and minutes <= limit)


def alloc_wizard(parts: list[slurm.Partition]) -> slurm.Alloc:
    defaults = state.load_defaults()
    usable = [p for p in parts if p.avail.lower() == "up"]
    if not usable:
        raise slurm.SlurmError("No partitions are up.")

    last = defaults.get("partition")
    p = select(
        "Partition  (* = cluster default; 'cloud' nodes boot on demand, ~5-10 min)",
        [(partition_title(x), x) for x in usable],
        default=next((x for x in usable if x.name == last), usable[0]),
    )
    prev = defaults.get("by_partition", {}).get(p.name, state.AllocDefaults())

    # GPU type / count
    gpu_type: str | None = None
    n_gpus = 0
    if p.has_gpu:
        typed = [t for t in p.gpu_types if t is not None]
        if len(typed) > 1:
            prev_type = prev.get("gpu_type")
            gpu_type = select(
                "GPU type",
                [(f"{t.upper():<8} {gpus.fmt_vram(gpus.vram_gb(t))}", t) for t in typed],
                default=prev_type if prev_type in typed else None,
            )
        max_n = p.gpu_types.get(gpu_type, p.max_gpus) or 1
        n_gpus = (
            select(
                f"Number of GPUs (max {max_n} per node)",
                [(str(i), i) for i in range(1, max_n + 1)],
                default=min(prev.get("gpus", 1), max_n),
            )
            if max_n > 1
            else 1
        )

    # CPUs
    cpu_opts = [c for c in CPU_CHOICES if p.cpus == 0 or c <= p.cpus]
    if p.cpus and p.cpus not in cpu_opts:
        cpu_opts.append(p.cpus)
    cpu_default = prev.get("cpus", min(8, p.cpus or 8))
    if cpu_default not in cpu_opts:
        cpu_default = cpu_opts[-1] if cpu_default > cpu_opts[-1] else cpu_opts[0]
    cpus = select(
        f"CPU cores (node has {p.cpus or '?'})",
        [(str(c), c) for c in cpu_opts],
        default=cpu_default,
    )

    # Memory
    node_gb = p.mem_mb / 1024 if p.mem_mb else None
    mem_opts = [m for m in MEM_CHOICES if node_gb is None or _mem_gb(m) <= node_gb]
    mem_default = prev.get("mem")
    if mem_default not in [*mem_opts, "0"]:
        mem_default = "32G" if "32G" in mem_opts else (mem_opts[-1] if mem_opts else "0")
    node_mem = f"{node_gb:.0f} GB" if node_gb is not None else "?"
    mem = select(
        f"Memory (node has {node_mem})",
        [*((m, m) for m in mem_opts), ("all memory on the node (--mem=0)", "0")],
        default=mem_default,
    )

    # Time
    limit = slurm.parse_timelimit_minutes(p.timelimit)
    t_opts = [(label, v) for label, v in TIME_CHOICES if _fits(v, limit)]
    tval = select(
        f"Time limit (partition max {p.timelimit})",
        [*t_opts, ("custom…", "custom")],
        default=prev.get("time", t_opts[0][1] if t_opts else "custom"),
    )
    if tval == "custom":
        tval = text("Time (HH:MM:SS or D-HH:MM:SS)", validate=_validate_time).strip()

    name = f"{slurm.JOB_PREFIX}{p.name}-{time.strftime('%H%M')}"
    alloc = slurm.Alloc(
        partition=p.name, gpus=n_gpus, gpu_type=gpu_type, cpus=cpus, mem=mem, time=tval, name=name
    )

    defaults["partition"] = p.name
    defaults.setdefault("by_partition", {})[p.name] = state.AllocDefaults(
        gpu_type=gpu_type, gpus=n_gpus, cpus=cpus, mem=mem, time=tval
    )
    state.save_defaults(defaults)
    return alloc
