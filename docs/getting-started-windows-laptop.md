# Getting started — Windows laptop

This guide gets `dfxm-geo` running end-to-end on a Windows 11 laptop with PowerShell. From a fresh clone to a finished `dfxm-forward` run with `dfxm_geo.h5` opened in Python takes 10–15 minutes.

For the DTU cluster, see [getting-started-dtu-cluster.md](getting-started-dtu-cluster.md). For deeper workflow reference (LSF/SLURM templates, array jobs, sizing), see [cluster-runs.md](cluster-runs.md).

---

## Prerequisites

- **Windows 11** with PowerShell (PowerShell 5.1 or 7+). The commands below assume PowerShell, not bash.
- **Python 3.11 or 3.12.** Install from python.org (recommended) or via the Microsoft Store. **Do not use** the Python 2.7 that may be bundled with git-bash — it is incompatible.
- **Git for Windows.** Includes `git` and a bash for occasional Unix-style commands. `git --version` should be 2.30+.
- **~20 GB free disk space.** The reciprocal-space kernel is ~128 MB; one forward-sim output is 1.5–2 GB; tailoring across runs you'll want headroom.
- **Optional: an HDF5 viewer.** [HDFView](https://www.hdfgroup.org/downloads/hdfview/) is the easiest standalone GUI. Or just use Python (`h5py`) as shown below.

> **Note on conda:** Anaconda / Miniforge on this machine has a broken HTTP stack (curl works fine to the same URLs, but `conda` calls hang). Use the `venv` + `pip` flow below; do not try to follow the cluster instructions that use `conda env create`.

---

## 1. Clone the repository

```powershell
cd $HOME\Documents
mkdir GM-reworked -ErrorAction SilentlyContinue
cd GM-reworked
git clone https://github.com/borgi-s/Geometrical_Optics_master.git
cd Geometrical_Optics_master
```

If you already have a clone, just `cd` into it and `git pull`.

To work with the v1.1.0 HDF5-outputs feature before it merges to main:

```powershell
git fetch origin
git checkout feature/hdf5-output-v1.1
```

(Once v1.1.0 is on `main`, just `git checkout main` instead.)

---

## 2. Create the Python virtual environment

```powershell
cd $HOME\Documents\GM-reworked
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` prefix your PowerShell prompt.

Verify the Python version:

```powershell
python --version
```

Expected: `Python 3.11.x` or `3.12.x`. If you see `Python 2.7.x`, you accidentally invoked git-bash's bundled Python; close the shell, open a fresh PowerShell, and re-activate.

> **Activation gotcha:** if PowerShell refuses to run `Activate.ps1` with "execution of scripts is disabled", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry activation.

---

## 3. Install dfxm-geo and its dev tools

From inside the activated venv:

```powershell
cd $HOME\Documents\GM-reworked\Geometrical_Optics_master
pip install -e ".[dev]"
```

This installs the package in editable mode (so your code edits hit immediately) plus the dev extras (pytest, mypy, ruff, jupyter). First install takes ~3 minutes — `numba` and `scipy` carry heavy wheels.

Verify the install:

```powershell
python -c "import dfxm_geo; print(dfxm_geo.__file__)"
dfxm-forward --help
dfxm-bootstrap --help
dfxm-identify --help
dfxm-migrate-output --help
```

The four CLI entry points should print their help text. If any are missing, re-run `pip install -e ".[dev]"`.

---

## 4. Run the test suite (sanity check)

```powershell
python -m pytest -q
```

Expected: **300+ passed, 0 failed** in ~30–60 seconds. One test (`TestDfxmForwardSampleRemountCLI::test_dfxm_forward_with_sample_remount_S2_runs`) is a slow CLI smoke test that takes 5+ minutes and may be deselected; that's normal.

If anything fails, stop here and surface the failures — the rest of the guide assumes a green suite.

```powershell
python -m mypy src/dfxm_geo/
python -m ruff check src/dfxm_geo/ tests/
```

Both should report 0 errors. Pre-existing `import-untyped` warnings for `h5py`, `tqdm`, `psutil`, etc. are expected.

---

## 5. Generate the reciprocal-space kernel (one-time)

The kernel is a ~128 MB `.npz` file that takes ~50 seconds to generate at default `Nrays=1e8`. It's not committed to the repo; you generate it once per environment with `dfxm-bootstrap`.

```powershell
dfxm-bootstrap --config configs/default.toml --if-missing
```

The `--if-missing` flag makes this idempotent: it generates the kernel if absent, prints a confirmation if present. Output:

```
Generating kernel: <path-to-pkl_files>\Resq_i_<date>_<time>.npz ...
Wrote <path>\Resq_i_<date>_<time>.npz (138 MB)
```

The kernel filename rotates each time it regenerates (encodes the timestamp); the canonical filename for the current build is set in `src/dfxm_geo/direct_space/forward_model.py:57` (`pkl_fn = "..."`).

To force regeneration of an existing kernel (e.g., after a code change to the kernel module), drop `--if-missing` or pass `--force`.

---

## 6. Run your first simulation

```powershell
dfxm-forward --config configs/default.toml --output .\output\first-run
```

This runs the canonical 61×61 (φ, χ) rocking grid at the default parameters from `configs/default.toml` (4 µm dislocation spacing, 151 dislocations, S1 sample remount). Wall time on a laptop: **10–20 minutes** at `Nsub=2`, or **2–3 minutes** at `Nsub=1` (faster but lower-quality kernel sampling).

Progress is shown live via `tqdm` bars; expect two: the dislocated-crystal sweep and the perfect-crystal reference sweep. The postprocess stage runs after both sweeps complete and writes the COM-map / qi-field analysis into the same HDF5.

Final state on disk:

```
output\first-run\
├── dfxm_geo.h5                      ← single HDF5 with everything
└── figures\
    ├── mosaicity_maps.svg
    └── qi_cross_section.svg
```

Versus the legacy `.npy` format (pre-v1.1.0), which produced 7,442 `.npy` files in `images10/` + `images10_perf_crystal/`. The new HDF5 is ~3.5× smaller (gzip-4 + shuffle compression) and contains everything you need to reproduce or analyse the run.

---

## 7. Inspect the output

### Quick check via Python

```powershell
python -c @'
import h5py
with h5py.File("output/first-run/dfxm_geo.h5", "r") as f:
    print("Version:", f["/dfxm_geo/version"][()].decode())
    print("Git SHA:", f["/dfxm_geo/git_sha"][()].decode())
    print("Generated:", f["/dfxm_geo/generated_at"][()].decode())
    print("Kernel:", f["/dfxm_geo/kernel/pkl_fn"][()].decode())
    print()
    print("Dislocated scan /1.1:")
    print("  data shape:", f["/1.1/instrument/dfxm_sim_detector/data"].shape)
    print("  data dtype:", f["/1.1/instrument/dfxm_sim_detector/data"].dtype)
    print("  phi (degrees):", f["/1.1/instrument/positioners/phi"][:3], "...")
    print("  chi (degrees):", f["/1.1/instrument/positioners/chi"][:3], "...")
    print("  title:", f["/1.1/title"][()].decode())
    print()
    print("Perfect-crystal scan /2.1 present:", "/2.1" in f)
    print()
    print("Analysis (from postprocess):")
    print("  phi_list shape:", f["/1.1/dfxm_geo/analysis/phi_list"].shape)
    print("  chi_shift_deg:", float(f["/1.1/dfxm_geo/analysis/chi_shift_deg"][()]))
'@
```

Expected: version, git SHA, kernel filename, scan shapes, both scans present, analysis arrays populated.

### Open in HDFView (GUI)

Launch HDFView and `File → Open` your `dfxm_geo.h5`. You'll see the tree:

- `/dfxm_geo/` — provenance (version, git_sha, kernel hash, full config TOML embedded)
- `/1.1/` — dislocated scan (BLISS layout): instrument/, positioners, measurement/ soft-links, sample/, dfxm_geo/ per-scan group
- `/2.1/` — perfect-crystal scan

Double-click any dataset to see values. For the image stack, right-click → Open As → Image to render frames.

### Open in darfix or darling

If you have the ESRF analysis stack installed (`pip install darfix` or `pip install -e <darling-clone>`):

```python
# darfix
from darfix.io.utils import open_h5
ds = open_h5("output/first-run/dfxm_geo.h5", "/1.1")

# darling
import darling
dset = darling.DataSet("output/first-run/dfxm_geo.h5", scan_id="1.1")
print(dset.data.shape, dset.motors)
```

The HDF5 follows BLISS layout (see [output-format.md](output-format.md) for the full schema) so both tools should open it without patches.

---

## 8. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'dfxm_geo'` | Venv not active, or pip install skipped | `& .\.venv\Scripts\Activate.ps1; pip install -e ".[dev]"` |
| `Python 2.7.x` shows in `python --version` | git-bash's bundled Python on PATH | Use PowerShell, not git-bash; or use absolute path `& .\.venv\Scripts\python.exe ...` |
| `FileNotFoundError: Reciprocal-space kernel npz not found at ...` | Forgot Step 5 | `dfxm-bootstrap --config configs/default.toml --if-missing` |
| `RuntimeError: ... requires forward kernel ... but it was not loaded` at import time | Stale install or moved files | `pip install -e ".[dev]"` from the repo root |
| Memory error on the postprocess stage | Default config + 16 GB RAM laptop with other apps open | Edit `configs/default.toml` to drop `chi_oversample` from 20 to 10; or run with `DFXM_MAX_WORKERS=2` env var to cap thread pool |
| `Activate.ps1 cannot be loaded ... execution of scripts is disabled` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (once, then retry) |
| Pre-commit hook fails on commit with mixed-line-ending | Editor saved file with CRLF on Windows | Re-stage the auto-fixed file and re-commit — hook fixes it for you |

---

## 9. Next steps

- **Run a different scan geometry**: copy `configs/default.toml` to `configs/my-experiment.toml`, edit `crystal` / `scan` / `postprocess` sections, run with `--config configs/my-experiment.toml --output output/my-experiment`.
- **Skip the perfect-crystal reference** (faster, but no χ-shift correction): in your config, set `[io] include_perfect_crystal = false`. Postprocess won't run.
- **Compare against a legacy `.npy` directory** (pre-v1.1.0 outputs): `dfxm-migrate-output <old_dir>` converts it to the new HDF5 format. Defaults to IUCrJ-2024 params; use `--config <toml>` to override.
- **Run on the cluster** for production-scale sweeps: see [getting-started-dtu-cluster.md](getting-started-dtu-cluster.md).
- **Deep workflow reference**: [cluster-runs.md](cluster-runs.md) covers `dfxm-identify` modes, array jobs, and sizing for both LSF and SLURM clusters.

For the science behind the simulator, see [physics.md](physics.md) and the canonical paper: [Borgi et al., J. Appl. Cryst. 57(2):358–368, 2024](https://journals.iucr.org/j/issues/2024/02/00/nb5370/).
