"""GRES parsing and GPU -> VRAM lookup."""

from __future__ import annotations

import re

# VRAM in GB, keyed by lowercase GPU name as Slurm GRES usually reports it.
GPU_VRAM_GB: dict[str, int] = {
    "t4": 16,
    "v100": 16,
    "v100s": 32,
    "a10": 24,
    "a10g": 24,
    "l4": 24,
    "a40": 48,
    "l40": 48,
    "l40s": 48,
    "a100": 40,
    "a100-40gb": 40,
    "a100-80gb": 80,
    "a100_80gb": 80,
    "h100": 80,
    "h200": 141,
    "rtx6000": 24,
    "a6000": 48,
}

# AWS ParallelCluster often names nodes/features after instance families.
AWS_INSTANCE_GPU: dict[str, tuple[str, int]] = {
    "g4dn": ("t4", 16),
    "g5": ("a10g", 24),
    "g6": ("l4", 24),
    "g6e": ("l40s", 48),
    "p3": ("v100", 16),
    "p4d": ("a100", 40),
    "p4de": ("a100-80gb", 80),
    "p5": ("h100", 80),
}

_GRES_ITEM = re.compile(r"^gpu(?::(?P<type>[^:(]+))?(?::(?P<count>\d+))?")


def _group(m: re.Match[str], name: str) -> str | None:
    """`Match.group` is typed `str | Any` (None when unmatched); narrow it to `str | None`."""
    g = m.group(name)
    return g if isinstance(g, str) else None


def parse_gres(gres: str) -> list[tuple[str | None, int]]:
    """Parse a Slurm GRES string into [(gpu_type, count)].

    Handles forms like "gpu:a10g:4(S:0)", "gpu:4", "gpu:a100:2,shard:8", "(null)".
    gpu_type is None when the GRES does not name one.
    """
    if not gres or gres in ("(null)", "N/A"):
        return []
    out: list[tuple[str | None, int]] = []
    # Split on commas that are not inside parentheses.
    depth, cur, items = 0, "", []
    for ch in gres:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(cur)
            cur = ""
        else:
            cur += ch
    items.append(cur)
    for item in items:
        m = _GRES_ITEM.match(item.strip())
        if m is None:
            continue
        gtype = _group(m, "type")
        count = _group(m, "count")
        # "gpu:4" -> type slot actually holds the count
        if gtype is not None and gtype.isdigit() and count is None:
            gtype, count = None, gtype
        out.append((gtype.lower() if gtype is not None else None, int(count) if count is not None else 1))
    return out


def vram_gb(gpu_type: str | None) -> int | None:
    if gpu_type is None:
        return None
    t = gpu_type.lower()
    if t in GPU_VRAM_GB:
        return GPU_VRAM_GB[t]
    # Fuzzy: "nvidia_a100-sxm4-80gb" etc.
    if "80g" in t and "a100" in t:
        return 80
    for name, gb in sorted(GPU_VRAM_GB.items(), key=lambda kv: -len(kv[0])):
        if name in t:
            return gb
    return None


def guess_from_instance(text: str) -> tuple[str, int] | None:
    """Guess GPU from an AWS instance family found in a node name or feature list."""
    t = text.lower()
    for fam in sorted(AWS_INSTANCE_GPU, key=len, reverse=True):
        if re.search(rf"(^|[^a-z0-9]){fam}(?=[^a-z0-9]|\d|x|$)", t):
            return AWS_INSTANCE_GPU[fam]
    return None


def fmt_vram(gb: int | None) -> str:
    return f"{gb} GB" if gb is not None else "?"
