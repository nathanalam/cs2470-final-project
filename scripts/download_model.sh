#!/usr/bin/env bash
# Download a Hub model on a compute node (login nodes cap virtual memory at 2 GB, which kills hf).
#   sbatch scripts/download_model.sh openjev/openjev            -> $MODELS_DIR/openjev
#SBATCH -J s1-download
#SBATCH -p general
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/download-%j.out
set -euo pipefail
repo=${1:?usage: download_model.sh <org/name>}
dest=${MODELS_DIR:-/shared/home/$USER/models}/${repo#*/}
uvx --from 'huggingface_hub[hf_xet]' hf download "$repo" --local-dir "$dest"
( cd "$dest" && sha256sum -c --quiet SHA256SUMS ) && echo "verified $dest"
