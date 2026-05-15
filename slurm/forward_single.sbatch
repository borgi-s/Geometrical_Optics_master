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

# Module + env activation (ESRF site convention). Adjust to match your
# institution's module path or a local miniforge install.
module load conda 2>/dev/null || source ~/miniforge3/etc/profile.d/conda.sh
conda activate dfxm-geo

if [ ! -f reciprocal_space/pkl_files/Resq_i_20230913_1308.pkl ]; then
    dfxm-bootstrap --config "${CONFIG}"
fi

dfxm-forward --config "${CONFIG}" --output "${OUTPUT}"

echo "Done. Output in ${OUTPUT}"
