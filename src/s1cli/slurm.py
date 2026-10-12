"""Thin wrappers around Slurm commands plus pure parsers for their output."""

from __future__ import annotations

import contextlib
import getpass
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field

from . import fake, gpus

SINFO_FMT = "%P|%a|%l|%D|%T|%G|%c|%m|%N"
SQUEUE_FMT = "%i|%P|%j|%T|%r|%M|%l|%N|%b"
JOB_PREFIX = "s1-"


class SlurmError(RuntimeError):
    pass


def fake_mode() -> bool:
    return os.environ.get("S1_FAKE_SLURM") == "1"


def run(cmd: list[str], timeout: float = 20) -> str:
    """Run a Slurm (or other) command and return stdout, raising SlurmError on failure."""
    if fake_mode():
        return fake.respond(cmd)
    if shutil.which(cmd[0]) is None:
        raise SlurmError(f"`{cmd[0]}` not found on PATH — are you on the cluster login node?")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise SlurmError(f"`{shlex.join(cmd)}` timed out after {timeout:.0f}s") from None
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        raise SlurmError(f"`{shlex.join(cmd)}` failed: {msg}")
    return proc.stdout


def available() -> bool:
    return fake_mode() or shutil.which("sinfo") is not None


def user() -> str:
    return os.environ.get("USER") or getpass.getuser()


# ---------------------------------------------------------------- node states

# Bucket names, in display order.
STATE_BUCKETS = ("idle", "cloud", "booting", "busy", "down")


def state_bucket(state: str) -> str:
    """Map a sinfo %T state (e.g. "idle~", "mix", "down~", "idle#") to a coarse bucket.

    On AWS ParallelCluster: "~" = powered down (will boot on demand, ~5-10 min),
    "#" / "%" = powering up / down, and "down" usually means AWS has no capacity.
    """
    s = state.lower().rstrip("*")
    base = s.rstrip("~#%!$@^-")
    if base.startswith(("down", "drain", "fail", "inval", "unk", "no_resp")):
        return "down"
    if s.endswith("#") or s.endswith("%") or "power" in base:
        return "booting"
    if s.endswith("~"):
        return "cloud"
    if base.startswith(("alloc", "mix", "comp", "resv", "plnd")):
        return "busy"
    if base.startswith("idle"):
        return "idle"
    return "down"


# ---------------------------------------------------------------- partitions


@dataclass
class Partition:
    name: str
    default: bool = False
    avail: str = "up"
    timelimit: str = ""
    nodes: int = 0
    states: dict[str, int] = field(default_factory=lambda: dict.fromkeys(STATE_BUCKETS, 0))
    gpu_types: dict[str | None, int] = field(default_factory=dict)  # type -> max GPUs per node
    cpus: int = 0  # max CPUs per node
    mem_mb: int = 0  # max memory per node (MB)
    node_names: list[str] = field(default_factory=list)

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpu_types)

    @property
    def max_gpus(self) -> int:
        return max(self.gpu_types.values(), default=0)

    @property
    def vram_gb(self) -> int | None:
        vals = [v for t in self.gpu_types if (v := gpus.vram_gb(t)) is not None]
        return max(vals) if vals else None

    def gpu_label(self) -> str:
        if not self.gpu_types:
            return "-"
        parts = []
        for t, n in self.gpu_types.items():
            parts.append(f"{(t or 'gpu').upper()} x{n}")
        return ", ".join(parts)


def _int(s: str) -> int:
    m = re.match(r"\d+", s.strip())
    return int(m.group()) if m else 0


def parse_sinfo(text: str) -> list[Partition]:
    """Parse `sinfo -h -o SINFO_FMT` output (one line per partition/state group)."""
    parts: dict[str, Partition] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        cols = line.split("|")
        if len(cols) < 8:
            continue
        name, avail, tl, nn, state, gres, cpus, mem = cols[:8]
        nodelist = cols[8] if len(cols) > 8 else ""
        default = name.endswith("*")
        name = name.rstrip("*")
        p = parts.setdefault(name, Partition(name=name))
        p.default = p.default or default
        p.avail = avail
        p.timelimit = tl
        count = _int(nn)
        p.nodes += count
        p.states[state_bucket(state)] += count
        p.cpus = max(p.cpus, _int(cpus))
        p.mem_mb = max(p.mem_mb, _int(mem))
        if nodelist:
            p.node_names.append(nodelist)
        for gtype, n in gpus.parse_gres(gres):
            p.gpu_types[gtype] = max(p.gpu_types.get(gtype, 0), n)
    return list(parts.values())


_NODE_KV = re.compile(r"(\w+)=(\S*)")


def parse_scontrol_nodes(text: str) -> list[dict[str, str]]:
    """Parse `scontrol show node -o` (one node per line, key=value tokens)."""
    nodes: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        nodes.append({str(m[1]): str(m[2]) for m in _NODE_KV.finditer(line)})
    return nodes


def refine_gpu_types(parts: list[Partition], nodes: list[dict[str, str]]) -> None:
    """Fill in untyped GPUs ("gpu:1") from scontrol GRES, features, or AWS instance names."""
    by_part: dict[str, list[dict[str, str]]] = {}
    for n in nodes:
        for pname in n.get("Partitions", "").split(","):
            by_part.setdefault(pname, []).append(n)
    for p in parts:
        if None not in p.gpu_types:
            continue
        for n in by_part.get(p.name, []):
            typed = [(t, c) for t, c in gpus.parse_gres(n.get("Gres", "")) if t is not None]
            if not typed:
                hint = " ".join(
                    [n.get("NodeName", ""), n.get("AvailableFeatures", ""), n.get("ActiveFeatures", "")]
                )
                guess = gpus.guess_from_instance(hint)
                if guess is not None:
                    typed = [(guess[0], p.gpu_types[None])]
            for t, c in typed:
                p.gpu_types[t] = max(p.gpu_types.get(t, 0), c)
        if len(p.gpu_types) > 1:
            p.gpu_types.pop(None, None)


def partitions() -> list[Partition]:
    parts = parse_sinfo(run(["sinfo", "-h", "-o", SINFO_FMT]))
    if any(None in p.gpu_types for p in parts):
        with contextlib.suppress(SlurmError):
            refine_gpu_types(parts, parse_scontrol_nodes(run(["scontrol", "show", "node", "-o"])))
    # GPU partitions first, then default, then name.
    parts.sort(key=lambda p: (not p.has_gpu, not p.default, p.name))
    return parts


# ---------------------------------------------------------------- jobs


@dataclass
class Job:
    id: str
    partition: str
    name: str
    state: str
    reason: str
    elapsed: str
    timelimit: str
    nodes: str
    gres: str

    @property
    def running(self) -> bool:
        return self.state == "RUNNING"

    @property
    def is_s1(self) -> bool:
        return self.name.startswith(JOB_PREFIX)


def parse_squeue(text: str) -> list[Job]:
    jobs = []
    for line in text.splitlines():
        cols = line.split("|")
        if len(cols) < 9:
            continue
        jobs.append(Job(*[c.strip() for c in cols[:9]]))
    return jobs


def my_jobs() -> list[Job]:
    return parse_squeue(run(["squeue", "-h", "-u", user(), "-o", SQUEUE_FMT]))


def cancel(job_id: str) -> None:
    run(["scancel", job_id])


def job_detail(job_id: str) -> str:
    return run(["scontrol", "show", "job", job_id])


# ---------------------------------------------------------------- allocation


@dataclass
class Alloc:
    partition: str
    gpus: int = 1
    gpu_type: str | None = None
    cpus: int = 8
    mem: str = "32G"
    time: str = "01:00:00"
    name: str = "s1-job"

    def gres(self) -> str:
        return f"gpu:{self.gpu_type}:{self.gpus}" if self.gpu_type is not None else f"gpu:{self.gpus}"

    def srun_cmd(self) -> list[str]:
        cmd = ["srun", "-p", self.partition]
        if self.gpus > 0:
            cmd += [f"--gres={self.gres()}"]
        cmd += [
            "-c",
            str(self.cpus),
            f"--mem={self.mem}",
            "-t",
            self.time,
            "-J",
            self.name,
            "--pty",
            "bash",
            "-l",
        ]
        return cmd


def parse_timelimit_minutes(tl: str) -> int | None:
    """ "1-00:00:00" / "4:00:00" / "30:00" / "infinite" -> minutes (None = unlimited)."""
    tl = tl.strip().lower()
    if not tl or tl in ("infinite", "unlimited", "n/a"):
        return None
    days = 0
    if "-" in tl:
        d, tl = tl.split("-", 1)
        days = int(d)
    nums = [int(x) for x in tl.split(":")]
    if len(nums) == 3:
        h, m, _ = nums
    elif len(nums) == 2:
        h, m = 0, nums[0]
    else:
        h, m = 0, nums[0]
    return days * 1440 + h * 60 + m


def on_compute_node() -> bool:
    return "SLURM_JOB_ID" in os.environ
