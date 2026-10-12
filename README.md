# cs2470-final-project — `s1` cluster CLI

Interactive helper for the HUIT OnDemand cluster (AWS ParallelCluster) used to
profile open-source System One decision models. It discovers partitions/GPUs,
allocates GPUs through arrow-key menus, and runs every allocation inside **tmux**
so a dropped web terminal doesn't kill your job.

## Quickstart (OnDemand → Clusters → Shell Access)

```bash
# one-time: install uv into ~/.local/bin
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <this repo> && cd cs2470-final-project
uv run s1            # main menu
```

| Command | What it does |
|---|---|
| `uv run s1` | Main menu |
| `uv run s1 info` | Partitions, GPU type, VRAM, node states (ready / cloud / busy / down) |
| `uv run s1 alloc` | Wizard → `srun … --pty bash` inside a tmux session |
| `uv run s1 alloc -p gpu -g 1 -c 8 --mem 32G -t 01:00:00 -y` | Same, no prompts |
| `uv run s1 alloc -p gpu --dry-run` | Just print the `srun` command |
| `uv run s1 jobs` | Your jobs → attach / GPU status / details / cancel |
| `uv run s1 attach` | Reattach after a disconnect (tells you if it's on another login node) |
| `uv run s1 gpu [--watch]` | `nvidia-smi` summary, locally or inside a running job via `srun --overlap` |
| `uv run s1 doctor` | Checks Slurm, tmux, uv, `/scratch/$USER`, `HF_HOME` |

### Cluster notes
- **cloud** nodes (`idle~`) are powered down and boot on demand: expect 5–10 min in PENDING.
- **down** usually means AWS has no capacity for that instance type right now.
- The shell terminal is a login node — don't run Python/models there; allocate first.
- `/scratch/$USER` is fast but can be wiped any time; it's created by your first job (`srun --pty true`).

### tmux cheatsheet
`Ctrl-b d` detach · `Ctrl-b w` list windows · `tmux ls` · `uv run s1 attach`

## Development (off-cluster)
```bash
uv run ruff format && uv run ruff check   # lint + format
uv run ty check                           # strict type checking (all rules = error)
uv run pytest
S1_FAKE_SLURM=1 uv run s1                 # canned sinfo/squeue output, nothing is launched
```
State lives in `~/.config/s1cli/` (last allocation choices, tmux session registry).
