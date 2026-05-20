# Getting started — DTU HPC cluster

This guide gets `dfxm-geo` running end-to-end on the DTU HPC cluster (LSF scheduler, `bsub`). From SSH-ing in to a finished `dfxm-forward` job on the queue takes 15–25 minutes the first time; subsequent runs take 60 seconds to submit.

For laptop instructions, see [getting-started-windows-laptop.md](getting-started-windows-laptop.md). For deeper workflow reference (array jobs, `dfxm-identify` modes, SLURM equivalents for the ESRF cluster), see [cluster-runs.md](cluster-runs.md).

---

## Prerequisites

- **DTU HPC account.** You should already have SSH access via `ssh <username>@login.hpc.dtu.dk`. If not, request via DTU IT.
- **A working conda on the login node.** Either DTU's `module load miniforge3` (system conda), or your own miniforge install under `~/miniforge3`. Verify with `which conda` and `conda --version`. **Do not** `module load python3` — it conflicts with conda's Python and crashes with `Fatal Python error: init_fs_encoding` (see "Common pitfalls" below).
- **Outbound HTTPS** from the login node (most DTU HPC nodes have this).
- **About 1.5 GB of home-directory space** for the conda env, plus whatever you need for outputs.

> **Aside on the DTU lock file:** v1.0.3 ships `locks/environment-dtu-linux-64.yml`, a captured conda lock that lets you skip the slow conda-forge solve (~30 minutes → ~2 minutes) on DTU's linux-64 login node. The lock was re-exported from an activated `dfxm-geo` env in PR #7; if it ever drifts from the underlying `environment.yml`, fall back to the slow path (`conda env create -f environment.yml`).

---

## 1. SSH in and clone the repository

```bash
ssh <username>@login.hpc.dtu.dk
cd ~
git clone https://github.com/borgi-s/Geometrical_Optics_master.git
cd Geometrical_Optics_master
```

If you already have a clone, just `cd` in and `git pull`.

To work with v1.1.0 HDF5-outputs before it merges to main:

```bash
git fetch origin
git checkout feature/hdf5-output-v1.1
```

(Once v1.1.0 is on `main`, just `git checkout main`.)

---

## 2. Create the conda environment (one-time, ~2 min via lock)

The fast path uses the captured lock file at `locks/environment-dtu-linux-64.yml`:

```bash
# Confirm conda is on PATH
which conda

# Fast path (DTU linux-64 only): ~2 minutes
conda env create -f locks/environment-dtu-linux-64.yml

# Slow path (any platform, or if the lock is stale): ~30 minutes
# conda env create -f environment.yml

conda activate dfxm-geo
pip install -e ".[dev]"
```

Verify:

```bash
python --version                    # 3.11.x or 3.12.x
which python                        # ~/.../envs/dfxm-geo/bin/python
dfxm-forward --help
dfxm-bootstrap --help
dfxm-identify --help
dfxm-migrate-output --help
```

All four CLI entry points should print their help text.

---

## 3. Run the test suite (sanity check)

```bash
cd ~/Geometrical_Optics_master
python -m pytest -q
```

Expected: **300+ passed, 0 failed** in 30–60 seconds. The slow CLI smoke test may be deselected — that's fine.

If anything fails, **stop here** and report — don't proceed to a queued cluster run while the test suite is red.

---

## 4. Generate the reciprocal-space kernel (one-time per env)

The kernel is a ~128 MB `.npz` file (~50 seconds to generate at default `Nrays=1e8`). It's not in the repo; you generate it once per environment, then run as many simulations as you want against it.

Generate it on the **login node** (cheap, no SLURM/LSF needed):

```bash
dfxm-bootstrap --config configs/default.toml --if-missing
```

The `--if-missing` flag is idempotent: it generates if absent, prints a confirmation if present. Output:

```
Generating kernel: reciprocal_space/pkl_files/Resq_i_<date>_<time>.npz ...
Wrote reciprocal_space/pkl_files/Resq_i_<date>_<time>.npz (138 MB)
```

The kernel filename rotates each time it regenerates (encodes the timestamp). The canonical filename for this build is set in `src/dfxm_geo/direct_space/forward_model.py:57` and is the one `dfxm-forward` looks for at runtime. The LSF templates use `dfxm-bootstrap --if-missing` as an idempotent pre-step so they don't have to hardcode the rotating filename.

> If the login node has a strict CPU/mem cap and the bootstrap is killed: run a tiny interactive shell instead via `qrsh` (or `bsub -Is`) before running the bootstrap. The 5-GB peak memory needs a small dedicated slot, not the shared login node.

---

## 5. Submit your first forward simulation

The repository ships an LSF template at [`lsf/forward_single.bsub`](../lsf/forward_single.bsub). Review the `>>> EDIT THESE >>>` block at the top and adjust:

- `QUEUE` — typically `hpc` (24 h walltime cap, general purpose)
- `WALLTIME` — `04:00` is plenty for a single 61×61 forward run at `Nsub=2`
- `MEM_PER_SLOT` — `4GB` default; bump to `6GB` if you see OOMs
- `SLOTS` — `8` slots default; one host
- `CONFIG` — `configs/default.toml` or your own
- `RUN_TAG` — short slug for the output directory name

Submit:

```bash
bsub < lsf/forward_single.bsub
```

Check status:

```bash
bjobs                                  # all your jobs
bjobs -l <JOBID>                       # detailed view of one job
```

Once it starts, follow stdout live:

```bash
tail -f logs/forward-<JOBID>.out
```

Wall time on the DTU HPC `hpc` queue: **10–15 minutes** at `Nsub=2` for the default 61×61 grid.

---

## 6. Find your output

The template writes to:

```
~/Geometrical_Optics_master/output/<run-tag>_<JOBID>/
```

After completion you'll see:

```
output/<run-tag>_<JOBID>/
├── dfxm_geo.h5                       ← single HDF5 file, ~1.5–2 GB compressed
└── figures/
    ├── mosaicity_maps.svg
    └── qi_cross_section.svg
```

The `dfxm_geo.h5` follows BLISS layout and includes provenance (git SHA, kernel hash, full TOML config embedded) plus both scans (dislocated `/1.1` and perfect-crystal `/2.1`) and the postprocess analysis arrays. See [output-format.md](output-format.md) for the full schema.

---

## 7. Pull results back to your laptop

From your laptop (Windows PowerShell or any Linux/macOS):

```powershell
# Replace JOBID and tag with your actual values.
rsync -avhP <username>@login.hpc.dtu.dk:~/Geometrical_Optics_master/output/forward_<JOBID>/ `
    .\local-output\forward_<JOBID>\
```

(On Windows: install `rsync` via WSL2 or use `scp` / WinSCP if rsync isn't available.)

Or, if the cluster output dir is small (~2 GB), just `scp -r ...`.

Once it's on the laptop, open the `dfxm_geo.h5` per [getting-started-windows-laptop.md § 7](getting-started-windows-laptop.md#7-inspect-the-output) — same file, same tools.

---

## 8. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `Fatal Python error: init_fs_encoding: failed to get the Python codec ...` | Did `module load python3/3.11` — it sets `PYTHONHOME`/`PYTHONPATH` to DTU's site-packages, crashing conda's Python | Do NOT `module load python3`. Use only `conda activate dfxm-geo`. The v1.0.1 fix removed `module load python3` from the LSF templates — don't add it back. |
| `~/miniforge3/etc/profile.d/conda.sh: No such file or directory` | Hardcoded conda path doesn't match yours | Edit the `CONDA_BASE` variable in the `>>> EDIT THESE >>>` block of the LSF template to point at your conda install root |
| `FileNotFoundError: Reciprocal-space kernel npz not found at ...` | Forgot Step 4 | `dfxm-bootstrap --config configs/default.toml --if-missing` on the login node |
| Job killed mid-run with no traceback | OOM (out of memory) | Bump `MEM_PER_SLOT` in the LSF template from 4GB to 6GB or 8GB |
| Job killed at the walltime | Default walltime too short for your grid size or `Nsub` | Bump `WALLTIME` in the template; for `Nsub=2` 61×61, 4h is comfortable |
| `bsub: command not found` | Not on the login/submit node | SSH to `login.hpc.dtu.dk` specifically (not a compute node) |
| `conda activate` works in interactive shell but not in LSF script | `conda init` not in the right shell rc, or batch shell doesn't source `~/.bashrc` | The LSF template sources `${CONDA_BASE}/etc/profile.d/conda.sh` explicitly before activating — verify your `CONDA_BASE` points where conda is actually installed |
| Output written somewhere weird | `bsub` inherits the current working directory; if you submitted from `~/scratch`, output lands in `~/scratch/output/...` | Submit from `~/Geometrical_Optics_master`, or set `OUTPUT_DIR` explicitly in the template |

---

## 9. Next steps

- **Array jobs for many crystal configurations** (Monte Carlo sweep, parameter scan): use the template at `lsf/identify_array.bsub`. See [cluster-runs.md § Array jobs](cluster-runs.md) for index-handling details.
- **Different scan geometry**: copy `configs/default.toml`, edit, re-submit with `--config configs/my-experiment.toml`.
- **Skip the perfect-crystal reference** (faster, no χ-shift correction): in your config, set `[io] include_perfect_crystal = false`.
- **Migrate legacy `.npy` outputs** from pre-v1.1.0 runs: `dfxm-migrate-output <old_dir>` converts to the new HDF5 format.
- **ESRF (SLURM) instead of DTU (LSF)**: same workflow shape, different scheduler — see [cluster-runs.md § ESRF (SLURM) walkthrough](cluster-runs.md).

For the science behind the simulator, see [physics.md](physics.md) and the canonical paper: [Borgi et al., J. Appl. Cryst. 57(2):358–368, 2024](https://journals.iucr.org/j/issues/2024/02/00/nb5370/).
