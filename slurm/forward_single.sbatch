#!/bin/bash
# ==============================================================================
# ESRF (SLURM) — single dfxm-forward simulation
# ==============================================================================
# Submit with:   sbatch slurm/forward_single.sbatch
# Monitor with:  squeue -u $USER -j <JOBID>
# See docs/cluster-runs.md for the full walkthrough.
# ==============================================================================
#
# >>> EDIT THESE >>>
#SBATCH --job-name=dfxm-forward
# Verify available partitions on your cluster:  sinfo -o "%P %a %D"
#SBATCH --partition=nice               # ESRF default; check `sinfo` to confirm
#SBATCH --time=24:00:00                # HH:MM:SS
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/forward-%j.out   # %j = job id
#SBATCH --error=logs/forward-%j.err
# #SBATCH --mail-user=your.address@example.org
# #SBATCH --mail-type=BEGIN,END,FAIL
# <<< EDIT THESE <<<

set -euo pipefail

WORKDIR="${PWD}"
CONFIG="${WORKDIR}/configs/default.toml"
OUTPUT="${WORKDIR}/output/forward_${SLURM_JOB_ID:-local}"

mkdir -p "$(dirname "${OUTPUT}")" logs

# Guard: a loaded `python3/...` module shadows conda's Python interpreter and
# produces a "Fatal Python error: init_fs_encoding" crash. Mirror the LSF
# template's warning here — the same conflict can hit any HPC site that
# ships a site-wide `python3/...` module.
if command -v module >/dev/null 2>&1 && module list 2>&1 | grep -Eqi '(^|[[:space:]])python3?/'; then
    echo "ERROR: a python3 module is loaded; this shadows conda's Python." >&2
    echo "Run 'module unload python3' (or 'module purge') before submitting." >&2
    exit 1
fi

# Module + env activation (ESRF site convention). Adjust to match your
# institution's module path or a local miniforge install.
module load conda 2>/dev/null || source ~/miniforge3/etc/profile.d/conda.sh
conda activate dfxm-geo

# Stage 0 — ensure the reciprocal-space kernel pickle exists. dfxm-bootstrap
# --if-missing is idempotent and reads pkl_fn from forward_model, so this
# guard doesn't hardcode the filename.
dfxm-bootstrap --if-missing --config "${CONFIG}"

dfxm-forward --config "${CONFIG}" --output "${OUTPUT}"

echo "Done. Output in ${OUTPUT}"
