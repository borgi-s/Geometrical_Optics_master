# Running DFXM forward simulations on a cluster

This guide walks through running `dfxm-forward` and `dfxm-identify` on the two
clusters this codebase has been validated against: **DTU HPC** (LSF
scheduler, `bsub`) and **ESRF** (SLURM scheduler, `sbatch`). Both flows boil
down to the same two-step workflow.

## Two-step workflow

The reciprocal-space resolution kernel is a ~128 MB pickle that takes ~50 s
to regenerate on a laptop at the default `Nrays=1e8`. The pickle is **not**
in the repo; you generate it once per environment with `dfxm-bootstrap`,
then run as many `dfxm-forward` / `dfxm-identify` jobs against it as you
like.

```bash
# Step 1 — one-time per environment.
dfxm-bootstrap --config configs/default.toml

# Step 2 — many times.
dfxm-forward --config configs/default.toml --output output/run01/
```

Why two steps? Kernel regeneration is expensive; silently regenerating it
inside `dfxm-forward` would risk burning an hour of cluster time on a typo
while the user thinks the "simulation" is running. `dfxm-forward` therefore
fails loud with a `FileNotFoundError` and a `dfxm-bootstrap` instruction
when the pickle is missing.

`dfxm-bootstrap` writes to the canonical path that `dfxm-forward` reads
(`reciprocal_space/pkl_files/<pkl_fn>`, where `pkl_fn` is defined in
`dfxm_geo.direct_space.forward_model` — currently `Resq_i_20230913_1308.pkl`,
but it rotates whenever the kernel is regenerated). If you want a different
destination, pass `--output <path>`; if you want to regenerate an existing
pickle, pass `--force`. The cluster templates use `dfxm-bootstrap --if-missing`
as an idempotent guard so they don't have to hardcode the rotating filename.

## DTU HPC (LSF) walkthrough

The DTU HPC cluster uses the LSF scheduler. Templates live in
[`lsf/forward_single.bsub`](../lsf/forward_single.bsub) and
[`lsf/identify_array.bsub`](../lsf/identify_array.bsub).

```bash
# On the cluster (after `ssh login.hpc.dtu.dk`):
cd ~/Geometrical_Optics_master
git pull

# One-time conda env setup. Your conda must be initialized — if you
# installed miniforge yourself, it usually auto-activates via ~/.bashrc.
# DO NOT `module load python3/...` on DTU; it conflicts with conda's
# Python and causes a `Fatal Python error: init_fs_encoding` crash.

# Fast path (DTU only, ~2 min): skip the conda-forge solve via a captured
# lock file. Use this on linux-64 hosts; for other platforms, use the
# slow path below.
conda env create -f locks/environment-dtu-linux-64.yml

# Slow path (any cluster, ~30 min): re-solve from the >= ranges. Use
# this if the lock file is out of date for your platform.
# conda env create -f environment.yml

conda activate dfxm-geo
pip install -e .

# Run a single forward simulation.
bsub < lsf/forward_single.bsub
bjobs                                    # check status

# Once it's running, follow stdout:
tail -f logs/forward-<JOBID>.out
```

The default LSF template targets the `hpc` queue with a 24 h walltime cap,
4 GB/slot, 8 slots, single host. Override these in the `>>> EDIT THESE >>>`
block at the top.

For ML training data — sweeping many random crystal configurations — use
the array template:

```bash
bsub < lsf/identify_array.bsub
bjobs -A <ARRAYID>                       # array status
```

## ESRF (SLURM) walkthrough

The ESRF cluster uses the SLURM scheduler. Templates live in
[`slurm/forward_single.sbatch`](../slurm/forward_single.sbatch) and
[`slurm/identify_array.sbatch`](../slurm/identify_array.sbatch).

```bash
# On the cluster:
cd ~/Geometrical_Optics_master
git pull

# One-time conda env setup.
module load conda
conda env create -f environment.yml
conda activate dfxm-geo
pip install -e .

# Confirm the partition name (ESRF partitions vary per cluster).
sinfo -o "%P %a %D"

# Run a single forward simulation.
sbatch slurm/forward_single.sbatch
squeue -u $USER                          # check status
```

Array jobs:

```bash
sbatch slurm/identify_array.sbatch
squeue -u $USER -j <ARRAYID>
```

## Output handling

Templates write `output/<run-tag>_<jobid>/` relative to the directory the
job was submitted from (`bsub`/`sbatch` inherits CWD). On both clusters,
that's typically a shared scratch directory the templates assume is writable.

To pull results back to your laptop:

```bash
rsync -avh --partial \
    cluster-login:~/Geometrical_Optics_master/output/forward_<JOBID>/ \
    ./local-output/
```

## Memory + walltime sizing

| Workload | Wall time | Memory |
|---|---|---|
| `dfxm-bootstrap` (Nrays=1e8) | ~50 s | ~5 GB peak (chunked truncnorm) |
| `dfxm-forward` (61×61 grid, Nsub=2, ndis=151) | ~10–20 min | ~4 GB |
| `dfxm-forward` (61×61, Nsub=1 — fast iteration) | ~2–3 min | ~2 GB |
| `dfxm-identify --mode multi` (10 samples) | ~2 min | ~4 GB |

These numbers are rough — verify against your cluster before scaling up.

## DTU vs ESRF specifics

| | DTU HPC | ESRF |
|---|---|---|
| Scheduler | LSF | SLURM |
| Submit | `bsub < file` | `sbatch file` |
| Status | `bjobs` | `squeue` |
| Time format | `HH:MM` | `HH:MM:SS` |
| Partition flag | `#BSUB -q <queue>` | `#SBATCH --partition=<part>` |
| Memory flag | `#BSUB -R "rusage[mem=4GB]"` | `#SBATCH --mem=4G` |
| Array syntax | `#BSUB -J "name[1-N]"` | `#SBATCH --array=1-N` |
| Array index env var | `$LSB_JOBINDEX` | `$SLURM_ARRAY_TASK_ID` |
| Account flag (if required) | `#BSUB -P <project>` | `#SBATCH --account=<account>` |

If your cluster requires an account/project flag for billing, add it to the
`EDIT THESE` block at the top of each template.
