#!/bin/bash
# ==============================================================================
# ESRF (SLURM) — ML-training-data array job for dfxm-identify
# ==============================================================================
# Submit with:   sbatch slurm/identify_array.sbatch
# Monitor with:  squeue -u $USER -j <ARRAYID>
# ==============================================================================
#
# >>> EDIT THESE >>>
#SBATCH --job-name=dfxm-identify
#SBATCH --partition=nice               # check `sinfo` to confirm
#SBATCH --array=1-100                  # 100 tasks × 10 samples = 1000-sample sweep
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/identify-%A_%a.out  # %A = array job id, %a = task id
#SBATCH --error=logs/identify-%A_%a.err
# <<< EDIT THESE <<<

set -euo pipefail

WORKDIR="${PWD}"
CONFIG="${WORKDIR}/configs/identification_multi.toml"
OUTPUT_BASE="${WORKDIR}/output/identify_${SLURM_ARRAY_JOB_ID:-local}"
TASK_OUTPUT="${OUTPUT_BASE}/task_$(printf '%04d' "${SLURM_ARRAY_TASK_ID}")"

mkdir -p "${OUTPUT_BASE}" logs

module load conda 2>/dev/null || source ~/miniforge3/etc/profile.d/conda.sh
conda activate dfxm-geo

# Stage 0 — ensure the reciprocal-space kernel pickle exists. dfxm-bootstrap
# --if-missing is idempotent and reads pkl_fn from forward_model, so this
# guard doesn't hardcode the filename.
dfxm-bootstrap --if-missing --config "${CONFIG}"

dfxm-identify --config "${CONFIG}" --output "${TASK_OUTPUT}"

echo "Task ${SLURM_ARRAY_TASK_ID} done. Output in ${TASK_OUTPUT}"
