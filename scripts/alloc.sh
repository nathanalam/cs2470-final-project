#!/usr/bin/env bash
# Interactive GPU shell inside tmux, so a dropped OnDemand terminal doesn't kill it.
#   scripts/alloc.sh [partition] [gpus] [time]      e.g. scripts/alloc.sh gpu 1 02:00:00
# Reattach later with: tmux attach -t s1
set -euo pipefail
partition=${1:-}
gpus=${2:-1}
time=${3:-01:00:00}
cmd=(srun ${partition:+-p "$partition"} --gres=gpu:"$gpus" -c 8 --mem=32G -t "$time" -J s1 --pty bash -l)
echo "${cmd[*]}   (cloud nodes can take 5-10 min to boot; detach with Ctrl-b d)"
exec tmux new-session -A -s s1 "${cmd[*]}; echo \"allocation ended (exit \$?)\"; exec bash -l"
