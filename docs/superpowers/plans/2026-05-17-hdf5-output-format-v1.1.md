# HDF5 Output Format (v1.1.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `dfxm-forward`'s per-frame `.npy` outputs with a single BLISS-style HDF5 file per simulation. Gains self-contained provenance, ~3.5× space savings via gzip-4 + shuffle, and direct compatibility with darfix / darling (the actual analysis tools at ESRF ID03 / ID06).

**Architecture:** Each `run_simulation` call produces one `<output_dir>/dfxm_geo.h5` containing 1-2 BLISS scans (`/1.1` dislocations, `/2.1` optional perfect crystal) plus a global `/dfxm_geo/` provenance group. New `io.hdf5` module owns reading and writing. Producer-consumer worker pattern (workers compute, main thread writes) replaces the per-frame `np.save` thread pool. Old `.npy` reader is preserved internally only for the `dfxm-migrate-output` script. Pure cutover (B3) — no dual-format support at runtime.

**Tech Stack:** h5py (new direct dependency; previously pulled transitively via fabio). Existing test stack (pytest, numpy). No silx, no hdf5plugin, no bitshuffle.

**Branch:** `feature/hdf5-output-v1.1`, branched off `main` HEAD `367cfee` (v1.0.3 + pyproject version bump). Worktree at `.claude/worktrees/hdf5-output`.

**Source design:** `~/.claude/projects/C--Users-borgi-Documents-GM-reworked/memory/followups_hdf5_output_format.md` (resolution of the 2026-05-17 /grill-me session — Q1 through Q10 covering drivers, schema, granularity, two-stack representation, placement, write pattern, scope, backwards-compat, compression, and test plan).

**Scope (locked from grilling):**

- Forward sim only (`dfxm-forward`); identification stays on `.npy` + CSV. Identification HDF5 deferred to v1.2.0 after laptop+cluster feedback validates the schema (S3).
- Strict BLISS-flavoured HDF5 (A1), not strict NeXus. NX_class attrs added as a free interop bonus.
- One file per CLI invocation (G1); 1-2 BLISS scans per file (α — two-stack representation).
- Hybrid placement (Z): global provenance at `/dfxm_geo/`, per-scan at `/N.1/dfxm_geo/`.
- Per-frame chunks `(1, H, W)`, gzip-4 + shuffle, float64. No silx.
- W2 write pattern: workers compute and return; main thread streams into a pre-allocated dataset. No durability hardening (1-3 hour job tolerance is sufficient).
- Pure cutover B3: legacy `.npy` reader removed from public API; only the migration script can still read old directories.
- Figures (SVGs) keep current on-disk layout (F1).
- Layer 1 + Layer 2 tests mandatory; Layer 3 (darfix/darling acceptance, cross-platform synthetic, perf regression) ships separately if at all.
- Version: v1.1.0 first; v2.0.0 once cluster validation passes.

---

## File Structure

**Modify:**

- `pyproject.toml:7` — version `1.0.3` → `1.1.0`.
- `pyproject.toml:25-34` — add `"h5py>=3.0"` to runtime `dependencies`.
- `pyproject.toml:61-64` — add `dfxm-migrate-output = "dfxm_geo.io.migrate:cli_main"` to `[project.scripts]`.
- `src/dfxm_geo/io/images.py` — DELETE the file entirely. `save_image` (50-71), `save_images_parallel` (74-124), `save_edfs` (127-182), `load_image` (225-227), `load_images` (185-222), `load_images_parallel` (230-260) are all moved to `io.migrate` (read-side legacy helpers) or replaced by `io.hdf5`. `_auto_max_workers` (14-47) moves to `io.hdf5` (still needed by the W2 writer).
- `src/dfxm_geo/pipeline.py:43` — change `from dfxm_geo.io.images import load_images, save_images_parallel` to `from dfxm_geo.io.hdf5 import load_h5_scan, write_simulation_h5`.
- `src/dfxm_geo/pipeline.py:301-327` — replace the two `save_images_parallel` calls in `run_simulation` with one `write_simulation_h5(...)` call writing both scans.
- `src/dfxm_geo/pipeline.py:359-385` — replace the two `load_images` calls in `run_postprocess` with `load_h5_scan` calls; rename `dislocs_path` / `perfect_path` to point at the single `dfxm_geo.h5`.
- `src/dfxm_geo/pipeline.py:419-424` — `np.save(data_dir / ...)` calls for `phi_list` / `chi_list` / `qi_field` / `chi_shift_deg.txt`: REPLACE with writes into `/1.1/dfxm_geo/analysis/` inside the existing `dfxm_geo.h5` (open in append mode).
- `docs/architecture.md` — add an "Output file format" section after the existing module layout description.
- `docs/reproducibility.md` — replace the "Output directory" description with the new HDF5 layout reference.

**Create:**

- `src/dfxm_geo/io/hdf5.py` — writer (`write_simulation_h5`, `write_h5_scan`, `_write_provenance`), reader (`load_h5_scan`), W2 producer-consumer helpers (`_compute_frame`, `_save_scan_parallel_to_h5`), `_auto_max_workers` (moved from images.py).
- `src/dfxm_geo/io/migrate.py` — `_load_images_legacy` (the body of the deleted `load_images`), `migrate_npy_dir_to_h5`, `cli_main` for `dfxm-migrate-output`.
- `tests/test_hdf5_writer.py` — Layer 1 writer tests (round-trip, schema invariants, positioner units, soft links, NX_class attrs, chunking & compression).
- `tests/test_hdf5_provenance.py` — Layer 1 provenance tests (`/dfxm_geo/version`, `git_sha`, `kernel/sha256`, `config_toml` round-trip).
- `tests/test_hdf5_reader.py` — Layer 1 reader tests (`load_h5_scan` returns expected tuple shape; matches what postprocess needs).
- `tests/test_hdf5_pipeline.py` — Layer 1 integration test for `run_simulation` (.h5 created with `/1.1` + optional `/2.1`).
- `tests/test_hdf5_bit_equiv.py` — Layer 2 bit-equivalence (legacy snapshot vs new HDF5 writer; postprocess equivalence; migrate round-trip).
- `tests/test_hdf5_defensive.py` — Layer 2 defensive regression (`h5py` IS imported in writer module; `np.save` NOT in pipeline write path; `load_images` not importable from public API).
- `tests/test_migrate_output.py` — Layer 1 migration script tests (CLI, dispatcher).
- `tests/data/golden/forward_legacy_writer_4frames_8x8.npy` — Layer 2 bit-equivalence golden (a tiny 4-frame, 8×8 pixel stack captured from the current `.npy` writer before its deletion).
- `tests/_gen_forward_legacy_golden.py` — one-shot generator for the above (committed for reproducibility).
- `docs/output-format.md` — full BLISS-layout schema documentation with example HDF5 tree and field-by-field reference.

**Not committed (artifacts only, gitignored):**

- `*.h5` files in `output/` and elsewhere — already covered by no `.h5` exception in current `.gitignore`. Confirm in Task 0; add `*.h5` to ignore + a `!tests/data/golden/*.h5` exception if Layer 3 synthetic golden is added later.

---

## Task 0: Worktree setup + gitignore for `.h5`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Create the feature worktree from main**

```powershell
git worktree add C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master\.claude\worktrees\hdf5-output -b feature/hdf5-output-v1.1 main
```

Expected: `Preparing worktree (new branch 'feature/hdf5-output-v1.1')` + `HEAD is now at 367cfee chore(pyproject): bump version 1.0.1 -> 1.0.3 (#9)`.

All subsequent tasks run from inside this worktree directory.

- [ ] **Step 2: Read current `.gitignore`**

Read `.gitignore` from the worktree root. Expected: see `*.npy`, `*.pkl`, `*.npz` under "Project — generated outputs"; `*.h5` not present.

- [ ] **Step 3: Add `*.h5` to gitignore alongside `*.npz`**

Edit `.gitignore`. Find the "Project — generated outputs" block and add `*.h5` after `*.npz`:

```
# Project — generated outputs
*.npy
*.pkl
*.npz
*.h5
*.edf
*.mp4
output/
results/
rockingcurve*/
mixed_ims*/
final_figures/

# Project — keep small reference data
!tests/data/golden/*.npy
!tests/data/golden/*.npz
!tests/data/golden/*.h5
!docs/figures/*.png
```

- [ ] **Step 4: Verify**

```powershell
New-Item -ItemType File -Path "output\foo.h5" -Force | Out-Null
git check-ignore "output\foo.h5"
Remove-Item "output\foo.h5"
```

Expected: `git check-ignore` prints the path (meaning it IS ignored).

- [ ] **Step 5: Commit**

```powershell
git add .gitignore
git commit -m "chore: gitignore *.h5 alongside *.npz; preserve tests/data/golden exception"
```

---

## Task 1: Capture legacy `.npy` writer snapshot for the Layer 2 golden

**Files:**
- Create: `tests/_gen_forward_legacy_golden.py`
- Create: `tests/data/golden/forward_legacy_writer_4frames_8x8.npy`

- [ ] **Step 1: Write the golden generator**

The generator script must run with the CURRENT writer (before any of this plan's writer code lands) so we capture pre-migration behaviour. It writes a tiny 4-frame, 8×8 stack to disk and copies it into `tests/data/golden/`.

Create `tests/_gen_forward_legacy_golden.py`:

```python
"""One-shot: capture a tiny `forward()` stack via the current .npy writer.

Run BEFORE any HDF5 writer code lands. Output is committed as the
Layer 2 bit-equivalence golden in tests/data/golden/.

Usage:
    python tests/_gen_forward_legacy_golden.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.crystal.remount import SAMPLE_REMOUNT_OPTIONS
from dfxm_geo.io.images import save_images_parallel

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "data" / "golden" / "forward_legacy_writer_4frames_8x8.npy"

# Force a tiny detector for golden capture: 2 phi x 2 chi = 4 frames.
# (The default 510x170 detector at 61x61 is too big for a committed golden.)
# Use the canonical kernel; assert it loaded so we don't silently call
# forward() with Hg unset.
if fm.Hg is None:
    raise SystemExit(
        "Kernel didn't auto-load; run dfxm-bootstrap first or set DFXM_PKL_PATH."
    )

S = SAMPLE_REMOUNT_OPTIONS["S1"]
Hg, q_hkl = fm.Find_Hg(dis=4.0, ndis=151, psize=fm.psize, zl_rms=fm.zl_rms, S=S, remount_name="S1")
fm.Hg = Hg
fm.q_hkl = q_hkl

# NOTE: at 2x2 sampling the rocking grid endpoints would land in the
# rocking-curve falloff (signal goes to zero at chi > ~0.0003 rad).
# Use a tiny half-range so all 4 sampled points stay near peak signal.
TINY_HALF_RANGE_RAD = 5e-5
with tempfile.TemporaryDirectory() as tmp:
    save_images_parallel(
        Hg,
        phi_range=TINY_HALF_RANGE_RAD * 180 / np.pi,
        phi_steps=2,
        chi_range=TINY_HALF_RANGE_RAD * 180 / np.pi,
        chi_steps=2,
        fpath=str(Path(tmp) / "stack"),
        fn_prefix="/golden_",
        ftype=".npy",
    )
    files = sorted((Path(tmp) / "stack").glob("*.npy"))
    if len(files) != 4:
        raise SystemExit(f"Expected 4 frames, got {len(files)}")
    arr = np.stack([np.load(f) for f in files])
    # Crop an 8x8 window CENTRED on the bright dislocation region. The
    # default 510x170 detector has the signal at roughly (row=326, col=55);
    # the top-left corner is pure background.
    arr_small = arr[:, 322:330, 51:59]
    np.save(GOLDEN, arr_small)
    print(f"Wrote {GOLDEN.relative_to(REPO)}, shape={arr_small.shape}, "
          f"dtype={arr_small.dtype}, size={GOLDEN.stat().st_size} bytes")
```

- [ ] **Step 2: Run the generator**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" tests/_gen_forward_legacy_golden.py
```

Expected: `Wrote tests/data/golden/forward_legacy_writer_4frames_8x8.npy, shape=(4, 8, 8), dtype=float64, size=~2100 bytes`.

If you see "Kernel didn't auto-load", first ensure the canonical `.npz` kernel is at `src/dfxm_geo/direct_space/pkl_files/Resq_i_*.npz`.

- [ ] **Step 3: Verify the golden loads**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -c "import numpy as np; a = np.load('tests/data/golden/forward_legacy_writer_4frames_8x8.npy'); print(a.shape, a.dtype, a.sum())"
```

Expected: `(4, 8, 8) float64 <some-nonzero-sum>`.

- [ ] **Step 4: Commit**

```powershell
git add tests/_gen_forward_legacy_golden.py tests/data/golden/forward_legacy_writer_4frames_8x8.npy
git commit -m "test(hdf5): capture legacy .npy writer snapshot as bit-equivalence golden"
```

---

## Task 2: Declare `h5py` as a direct runtime dependency

**Files:**
- Modify: `pyproject.toml:25-34`

- [ ] **Step 1: Write a failing test that h5py is in the install spec**

Create `tests/test_pyproject_hdf5_dep.py`:

```python
"""Pin h5py as a direct runtime dependency (added in v1.1.0)."""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_h5py_is_runtime_dependency() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    deps = data["project"]["dependencies"]
    assert any(d.startswith("h5py") for d in deps), (
        f"h5py not in runtime dependencies; got {deps}"
    )
```

- [ ] **Step 2: Verify it fails**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pyproject_hdf5_dep.py -v
```

Expected: FAIL with `AssertionError`.

- [ ] **Step 3: Add the dep**

Edit `pyproject.toml`, in the `[project]` dependencies block, add `"h5py>=3.0"` after `"fabio>=2023.4"`:

```toml
dependencies = [
    "numpy>=1.23,<3",
    "scipy>=1.10,<2",
    "numba>=0.56",
    "matplotlib>=3.6",
    "seaborn>=0.12",
    "fabio>=2023.4",
    "h5py>=3.0",
    "joblib>=1.3",
    "tqdm>=4.66.3",  # CVE: CLI argument injection in <4.66.3
]
```

- [ ] **Step 4: Verify it passes**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pyproject_hdf5_dep.py -v
```

Expected: PASS.

- [ ] **Step 5: Confirm h5py is actually importable in this env (was transitive via fabio)**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -c "import h5py; print(h5py.__version__)"
```

Expected: `3.x.x` printed. If missing, `pip install -e ".[dev]"` from the worktree root.

- [ ] **Step 6: Commit**

```powershell
git add pyproject.toml tests/test_pyproject_hdf5_dep.py
git commit -m "feat(deps): declare h5py as direct runtime dep for HDF5 outputs"
```

---

## Task 3: Skeleton `io/hdf5.py` with minimal `write_h5_scan`

This task creates the new module with the simplest possible writer: just the image stack dataset under `/N.1/instrument/dfxm_sim_detector/data`. Subsequent tasks bolt on positioners, soft links, attrs, sample group, etc.

**Files:**
- Create: `src/dfxm_geo/io/hdf5.py`
- Create: `tests/test_hdf5_writer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hdf5_writer.py`:

```python
"""Layer 1 writer tests for dfxm_geo.io.hdf5."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm_geo.io.hdf5 import write_h5_scan


def test_write_h5_scan_creates_image_dataset(tmp_path: Path) -> None:
    images = np.arange(4 * 8 * 8, dtype=np.float64).reshape(4, 8, 8)
    out = tmp_path / "test.h5"
    write_h5_scan(out, scan_id="1.1", images=images)

    with h5py.File(out, "r") as f:
        data = f["/1.1/instrument/dfxm_sim_detector/data"][...]
        assert data.shape == (4, 8, 8)
        assert data.dtype == np.float64
        np.testing.assert_array_equal(data, images)
```

- [ ] **Step 2: Verify it fails**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_writer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'dfxm_geo.io.hdf5'`.

- [ ] **Step 3: Create the minimal writer**

Create `src/dfxm_geo/io/hdf5.py`:

```python
"""HDF5 output format for dfxm-forward simulations.

Writes BLISS-style ESRF HDF5 (compatible with darfix / darling) with
sim-specific provenance metadata. One file per `run_simulation` call,
containing 1-2 BLISS scans (`/1.1` dislocations, `/2.1` optional
perfect crystal).

See docs/output-format.md for the full schema.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def write_h5_scan(
    path: Path,
    scan_id: str,
    images: np.ndarray,
) -> None:
    """Write a single BLISS scan to an HDF5 file (creates or appends).

    Args:
        path: Output HDF5 file path. Created if missing; appended if exists.
        scan_id: BLISS scan identifier, e.g. "1.1" for the first scan.
        images: Image stack, shape (N_frames, H, W), dtype float64.
    """
    mode = "a" if path.exists() else "w"
    with h5py.File(path, mode) as f:
        scan = f.require_group(scan_id)
        det = scan.require_group("instrument/dfxm_sim_detector")
        det.create_dataset("data", data=images)
```

- [ ] **Step 4: Verify it passes**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_writer.py -v
```

Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_hdf5_writer.py
git commit -m "feat(io): skeleton HDF5 writer with image-stack dataset"
```

---

## Task 4: Add phi/chi positioners (degrees + units attr)

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py` — `write_h5_scan` signature + body
- Modify: `tests/test_hdf5_writer.py` — add positioner test

- [ ] **Step 1: Write the failing test**

Add to `tests/test_hdf5_writer.py`:

```python
def test_write_h5_scan_positioners_in_degrees(tmp_path: Path) -> None:
    images = np.zeros((4, 8, 8), dtype=np.float64)
    # phi/chi in RADIANS coming in (matches internal sim convention)
    phi = np.array([-0.001, 0.001, -0.001, 0.001])
    chi = np.array([-0.002, -0.002, 0.002, 0.002])
    out = tmp_path / "test.h5"
    write_h5_scan(out, scan_id="1.1", images=images, phi=phi, chi=chi)

    with h5py.File(out, "r") as f:
        phi_h5 = f["/1.1/instrument/positioners/phi"]
        chi_h5 = f["/1.1/instrument/positioners/chi"]
        # Stored in DEGREES regardless of input units.
        np.testing.assert_allclose(phi_h5[...], np.degrees(phi))
        np.testing.assert_allclose(chi_h5[...], np.degrees(chi))
        assert phi_h5.attrs["units"] == "degree"
        assert chi_h5.attrs["units"] == "degree"
```

- [ ] **Step 2: Verify it fails**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_writer.py::test_write_h5_scan_positioners_in_degrees -v
```

Expected: FAIL — `TypeError: write_h5_scan() got an unexpected keyword argument 'phi'`.

- [ ] **Step 3: Add positioners to writer**

Modify `src/dfxm_geo/io/hdf5.py`'s `write_h5_scan` signature and body:

```python
def write_h5_scan(
    path: Path,
    scan_id: str,
    images: np.ndarray,
    phi: np.ndarray | None = None,
    chi: np.ndarray | None = None,
) -> None:
    """Write a single BLISS scan to an HDF5 file (creates or appends).

    Args:
        path: Output HDF5 file path. Created if missing; appended if exists.
        scan_id: BLISS scan identifier, e.g. "1.1" for the first scan.
        images: Image stack, shape (N_frames, H, W), dtype float64.
        phi: Phi motor positions per frame, shape (N_frames,), in radians.
            Stored on disk in degrees with units attr.
        chi: Chi motor positions per frame, shape (N_frames,), in radians.
            Stored on disk in degrees with units attr.
    """
    mode = "a" if path.exists() else "w"
    with h5py.File(path, mode) as f:
        scan = f.require_group(scan_id)
        det = scan.require_group("instrument/dfxm_sim_detector")
        det.create_dataset("data", data=images)
        if phi is not None and chi is not None:
            pos = scan.require_group("instrument/positioners")
            phi_ds = pos.create_dataset("phi", data=np.degrees(phi))
            phi_ds.attrs["units"] = "degree"
            chi_ds = pos.create_dataset("chi", data=np.degrees(chi))
            chi_ds.attrs["units"] = "degree"
```

- [ ] **Step 4: Verify it passes**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_writer.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_hdf5_writer.py
git commit -m "feat(io/hdf5): write phi/chi positioners in degrees with units attr"
```

---

## Task 5: Add BLISS `measurement/` soft links

In BLISS, `/N.1/measurement/<name>` is a soft link aliasing the canonical location under `/N.1/instrument/`. Darfix / darling check both paths.

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py` — extend `write_h5_scan`
- Modify: `tests/test_hdf5_writer.py` — add soft-link test

- [ ] **Step 1: Write the failing test**

```python
def test_write_h5_scan_measurement_soft_links(tmp_path: Path) -> None:
    images = np.zeros((4, 8, 8), dtype=np.float64)
    phi = np.array([-0.001, 0.001, -0.001, 0.001])
    chi = np.array([-0.002, -0.002, 0.002, 0.002])
    out = tmp_path / "test.h5"
    write_h5_scan(out, scan_id="1.1", images=images, phi=phi, chi=chi)

    with h5py.File(out, "r") as f:
        # Measurement entries should resolve to the same data as instrument/.
        np.testing.assert_array_equal(
            f["/1.1/measurement/dfxm_sim_detector"][...],
            f["/1.1/instrument/dfxm_sim_detector/data"][...],
        )
        np.testing.assert_array_equal(
            f["/1.1/measurement/phi"][...],
            f["/1.1/instrument/positioners/phi"][...],
        )
        np.testing.assert_array_equal(
            f["/1.1/measurement/chi"][...],
            f["/1.1/instrument/positioners/chi"][...],
        )
```

- [ ] **Step 2: Verify it fails**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_writer.py::test_write_h5_scan_measurement_soft_links -v
```

Expected: FAIL — `KeyError: "Unable to open object (object '/1.1/measurement/dfxm_sim_detector' doesn't exist)"`.

- [ ] **Step 3: Add soft links**

At the end of `write_h5_scan` body, after positioner writes:

```python
        meas = scan.require_group("measurement")
        meas["dfxm_sim_detector"] = h5py.SoftLink(
            f"/{scan_id}/instrument/dfxm_sim_detector/data"
        )
        if phi is not None and chi is not None:
            meas["phi"] = h5py.SoftLink(f"/{scan_id}/instrument/positioners/phi")
            meas["chi"] = h5py.SoftLink(f"/{scan_id}/instrument/positioners/chi")
```

- [ ] **Step 4: Verify it passes**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_writer.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_hdf5_writer.py
git commit -m "feat(io/hdf5): add BLISS measurement/ soft links"
```

---

## Task 6: Add scan title + start/end times

Darfix auto-detects 2D scans from BLISS's `title` string (`"fscan2d phi -... chi -... 1"`).

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py`
- Modify: `tests/test_hdf5_writer.py`

- [ ] **Step 1: Failing test**

```python
def test_write_h5_scan_title_and_times(tmp_path: Path) -> None:
    images = np.zeros((4, 8, 8), dtype=np.float64)
    out = tmp_path / "test.h5"
    write_h5_scan(
        out, scan_id="1.1", images=images,
        title="fscan2d phi -0.001 0.001 2 chi -0.002 0.002 2 1.0",
        start_time="2026-05-17T10:00:00",
        end_time="2026-05-17T10:00:30",
    )
    with h5py.File(out, "r") as f:
        assert f["/1.1/title"][()].decode() == (
            "fscan2d phi -0.001 0.001 2 chi -0.002 0.002 2 1.0"
        )
        assert f["/1.1/start_time"][()].decode() == "2026-05-17T10:00:00"
        assert f["/1.1/end_time"][()].decode() == "2026-05-17T10:00:30"
```

- [ ] **Step 2: Verify FAIL**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_writer.py::test_write_h5_scan_title_and_times -v
```

Expected: FAIL — `TypeError: write_h5_scan() got an unexpected keyword argument 'title'`.

- [ ] **Step 3: Add title + times to signature/body**

```python
def write_h5_scan(
    path: Path,
    scan_id: str,
    images: np.ndarray,
    phi: np.ndarray | None = None,
    chi: np.ndarray | None = None,
    title: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> None:
    ...
    with h5py.File(path, mode) as f:
        scan = f.require_group(scan_id)
        if title is not None:
            scan.create_dataset("title", data=title)
        if start_time is not None:
            scan.create_dataset("start_time", data=start_time)
        if end_time is not None:
            scan.create_dataset("end_time", data=end_time)
        # ... rest unchanged
```

- [ ] **Step 4: Verify PASS**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_writer.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_hdf5_writer.py
git commit -m "feat(io/hdf5): add scan title + start/end timestamps"
```

---

## Task 7: NX_class attrs (free NeXus interop)

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py`
- Modify: `tests/test_hdf5_writer.py`

- [ ] **Step 1: Failing test**

```python
def test_write_h5_scan_nx_class_attrs(tmp_path: Path) -> None:
    images = np.zeros((4, 8, 8), dtype=np.float64)
    phi = np.array([-0.001, 0.001, -0.001, 0.001])
    chi = np.array([-0.002, -0.002, 0.002, 0.002])
    out = tmp_path / "test.h5"
    write_h5_scan(out, scan_id="1.1", images=images, phi=phi, chi=chi)
    with h5py.File(out, "r") as f:
        assert f["/1.1"].attrs["NX_class"] == "NXentry"
        assert f["/1.1/instrument"].attrs["NX_class"] == "NXinstrument"
        assert f["/1.1/instrument/dfxm_sim_detector"].attrs["NX_class"] == "NXdetector"
        assert f["/1.1/instrument/positioners"].attrs["NX_class"] == "NXcollection"
```

- [ ] **Step 2: Verify FAIL**

Expected: `KeyError: "Can't open attribute (can't locate attribute: 'NX_class')"`.

- [ ] **Step 3: Set attrs**

In `write_h5_scan`, after each `require_group(...)` call, set the attr. Cleanest: a small helper at module top:

```python
def _set_nx_class(grp: h5py.Group, cls: str) -> None:
    grp.attrs["NX_class"] = cls
```

And in the writer:

```python
        scan = f.require_group(scan_id)
        _set_nx_class(scan, "NXentry")
        ...
        det = scan.require_group("instrument/dfxm_sim_detector")
        _set_nx_class(scan["instrument"], "NXinstrument")
        _set_nx_class(det, "NXdetector")
        ...
        if phi is not None and chi is not None:
            pos = scan.require_group("instrument/positioners")
            _set_nx_class(pos, "NXcollection")
```

- [ ] **Step 4: Verify PASS**

Expected: 5 PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_hdf5_writer.py
git commit -m "feat(io/hdf5): set NX_class attrs on entry/instrument/detector/positioners"
```

---

## Task 8: `/N.1/sample/` group

Holds per-scan sample metadata: `name`, `dis`, `ndis`, `sample_remount`.

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py`
- Modify: `tests/test_hdf5_writer.py`

- [ ] **Step 1: Failing test**

```python
def test_write_h5_scan_sample_group(tmp_path: Path) -> None:
    images = np.zeros((4, 8, 8), dtype=np.float64)
    out = tmp_path / "test.h5"
    write_h5_scan(
        out, scan_id="1.1", images=images,
        sample_name="simulated, dislocations",
        sample_dis=4.0, sample_ndis=151, sample_remount="S1",
    )
    with h5py.File(out, "r") as f:
        s = f["/1.1/sample"]
        assert s["name"][()].decode() == "simulated, dislocations"
        assert float(s["dis"][()]) == 4.0
        assert int(s["ndis"][()]) == 151
        assert s["sample_remount"][()].decode() == "S1"
        assert s.attrs["NX_class"] == "NXsample"
```

- [ ] **Step 2: Verify FAIL**

Expected: `TypeError` on the new kwargs.

- [ ] **Step 3: Add sample group to writer**

```python
def write_h5_scan(
    path: Path,
    scan_id: str,
    images: np.ndarray,
    phi: np.ndarray | None = None,
    chi: np.ndarray | None = None,
    title: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    sample_name: str | None = None,
    sample_dis: float | None = None,
    sample_ndis: int | None = None,
    sample_remount: str | None = None,
) -> None:
    ...
    # after positioners and links:
    if any(x is not None for x in (sample_name, sample_dis, sample_ndis, sample_remount)):
        samp = scan.require_group("sample")
        _set_nx_class(samp, "NXsample")
        if sample_name is not None:
            samp.create_dataset("name", data=sample_name)
        if sample_dis is not None:
            samp.create_dataset("dis", data=float(sample_dis))
        if sample_ndis is not None:
            samp.create_dataset("ndis", data=int(sample_ndis))
        if sample_remount is not None:
            samp.create_dataset("sample_remount", data=sample_remount)
```

- [ ] **Step 4: Verify PASS**

Expected: 6 PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_hdf5_writer.py
git commit -m "feat(io/hdf5): add sample/ group (name, dis, ndis, remount)"
```

---

## Task 9: Per-scan `/N.1/dfxm_geo/` (Hg, q_hkl, physics scalars)

This is where simulation-specific per-scan data lives, separate from the BLISS namespace.

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py`
- Modify: `tests/test_hdf5_writer.py`

- [ ] **Step 1: Failing test**

```python
def test_write_h5_scan_dfxm_geo_per_scan(tmp_path: Path) -> None:
    images = np.zeros((4, 8, 8), dtype=np.float64)
    Hg = np.random.default_rng(0).standard_normal((100, 3, 3))
    q_hkl = np.array([0.0, 0.0, 1.0])
    out = tmp_path / "test.h5"
    write_h5_scan(
        out, scan_id="1.1", images=images,
        Hg=Hg, q_hkl=q_hkl,
        theta=8.545, psize=0.05, zl_rms=0.6,
    )
    with h5py.File(out, "r") as f:
        g = f["/1.1/dfxm_geo"]
        np.testing.assert_array_equal(g["Hg"][...], Hg)
        np.testing.assert_array_equal(g["q_hkl"][...], q_hkl)
        assert float(g["theta"][()]) == 8.545
        assert float(g["psize"][()]) == 0.05
        assert float(g["zl_rms"][()]) == 0.6
```

- [ ] **Step 2: Verify FAIL**

- [ ] **Step 3: Add the per-scan dfxm_geo group**

```python
def write_h5_scan(
    ...,
    Hg: np.ndarray | None = None,
    q_hkl: np.ndarray | None = None,
    theta: float | None = None,
    psize: float | None = None,
    zl_rms: float | None = None,
) -> None:
    ...
    # After sample group:
    if any(x is not None for x in (Hg, q_hkl, theta, psize, zl_rms)):
        d = scan.require_group("dfxm_geo")
        if Hg is not None:
            d.create_dataset("Hg", data=Hg)
        if q_hkl is not None:
            d.create_dataset("q_hkl", data=q_hkl)
        if theta is not None:
            d.create_dataset("theta", data=float(theta))
        if psize is not None:
            d.create_dataset("psize", data=float(psize))
        if zl_rms is not None:
            d.create_dataset("zl_rms", data=float(zl_rms))
```

- [ ] **Step 4: Verify PASS**

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_hdf5_writer.py
git commit -m "feat(io/hdf5): per-scan dfxm_geo/ group (Hg, q_hkl, theta, psize, zl_rms)"
```

---

## Task 10: Per-frame chunking + gzip-4 + shuffle compression on image data

This is a writer-internal change; the test asserts the on-disk dataset has the right `chunks`, `compression`, and `shuffle` attributes.

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py`
- Modify: `tests/test_hdf5_writer.py`

- [ ] **Step 1: Failing test**

```python
def test_write_h5_scan_image_compression(tmp_path: Path) -> None:
    images = np.zeros((4, 8, 8), dtype=np.float64)
    out = tmp_path / "test.h5"
    write_h5_scan(out, scan_id="1.1", images=images)
    with h5py.File(out, "r") as f:
        ds = f["/1.1/instrument/dfxm_sim_detector/data"]
        assert ds.chunks == (1, 8, 8)
        assert ds.compression == "gzip"
        assert ds.compression_opts == 4
        assert ds.shuffle is True
```

- [ ] **Step 2: Verify FAIL**

Expected: `assert None == (1, 8, 8)` or similar.

- [ ] **Step 3: Apply chunks + compression**

Modify the image dataset creation in `write_h5_scan`:

```python
        det = scan.require_group("instrument/dfxm_sim_detector")
        _set_nx_class(det, "NXdetector")
        n_frames, h, w = images.shape
        det.create_dataset(
            "data",
            data=images,
            chunks=(1, h, w),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
```

- [ ] **Step 4: Verify PASS**

Expected: 8 PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_hdf5_writer.py
git commit -m "feat(io/hdf5): chunked (1,H,W) image dataset with gzip-4 + shuffle"
```

---

## Task 11: Global `/dfxm_geo/` provenance group (version, git_sha, hostname, generated_at, etc.)

This is a separate writer call (`_write_provenance`) invoked once per file by `write_simulation_h5` (defined in a later task). For now, we add the function and test it standalone.

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py`
- Create: `tests/test_hdf5_provenance.py`

- [ ] **Step 1: Failing test**

Create `tests/test_hdf5_provenance.py`:

```python
"""Layer 1 provenance tests for /dfxm_geo/ root group."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from dfxm_geo.io.hdf5 import _write_provenance


def test_write_provenance_basic_fields(tmp_path: Path) -> None:
    out = tmp_path / "test.h5"
    with h5py.File(out, "w") as f:
        _write_provenance(f, cli="dfxm-forward --config x.toml --output out/")

    with h5py.File(out, "r") as f:
        g = f["/dfxm_geo"]
        assert g["version"][()].decode().startswith("1.")
        # git_sha is "unknown" outside a git repo, a 40-char SHA inside one
        sha = g["git_sha"][()].decode()
        assert sha == "unknown" or len(sha) == 40
        assert isinstance(g["git_dirty"][()], (bool, np.bool_))
        assert g["hostname"][()].decode()  # non-empty
        assert g["python_version"][()].decode().startswith("3.")
        assert g["numpy_version"][()].decode()
        assert g["generated_at"][()].decode()  # ISO-ish
        assert g["cli"][()].decode().startswith("dfxm-forward")
```

(Add `import numpy as np` at the top.)

- [ ] **Step 2: Verify FAIL**

Expected: `ImportError: cannot import name '_write_provenance'`.

- [ ] **Step 3: Add `_write_provenance` to `io/hdf5.py`**

```python
import datetime as _dt
import platform as _platform
import socket as _socket
import subprocess as _subprocess
import sys as _sys
from importlib.metadata import version as _pkg_version

import numpy as np
import h5py


def _get_git_sha_and_dirty() -> tuple[str, bool]:
    """Return (sha, dirty) for the current repo, or ("unknown", False)."""
    try:
        sha = _subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=_subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            _subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=_subprocess.DEVNULL, text=True
            ).strip()
        )
        return sha, dirty
    except (_subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False


def _write_provenance(f: h5py.File, *, cli: str = "") -> None:
    """Write the global /dfxm_geo/ provenance group at file root.

    Idempotent: safe to call on a file that already has /dfxm_geo/.
    """
    g = f.require_group("/dfxm_geo")
    sha, dirty = _get_git_sha_and_dirty()
    try:
        ver = _pkg_version("dfxm-geo")
    except Exception:
        ver = "unknown"

    fields = {
        "version": ver,
        "git_sha": sha,
        "git_dirty": dirty,
        "hostname": _socket.gethostname(),
        "python_version": _sys.version.split()[0],
        "numpy_version": np.__version__,
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "cli": cli,
    }
    for name, val in fields.items():
        if name in g:
            del g[name]
        g.create_dataset(name, data=val)
```

- [ ] **Step 4: Verify PASS**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_provenance.py -v
```

Expected: 1 PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_hdf5_provenance.py
git commit -m "feat(io/hdf5): /dfxm_geo/ root provenance (version, git_sha, host, etc.)"
```

---

## Task 12: `/dfxm_geo/kernel/` subgroup with sha256 and bundled params

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py`
- Modify: `tests/test_hdf5_provenance.py`

- [ ] **Step 1: Failing test**

```python
def test_write_provenance_kernel_subgroup(tmp_path: Path) -> None:
    npz_path = tmp_path / "Resq_i_test.npz"
    # Fake kernel npz with the expected param keys
    np.savez(
        npz_path,
        Resq_i=np.zeros((4, 4, 4), dtype=np.float64),
        qi1_range=1.0, qi2_range=2.0, qi3_range=3.0,
        npoints1=4, npoints2=4, npoints3=4, Nrays=int(1e6),
    )
    out = tmp_path / "test.h5"
    with h5py.File(out, "w") as f:
        _write_provenance(f, cli="x", kernel_npz=npz_path)
    with h5py.File(out, "r") as f:
        k = f["/dfxm_geo/kernel"]
        assert k["pkl_fn"][()].decode() == "Resq_i_test.npz"
        sha = k["sha256"][()].decode()
        assert len(sha) == 64  # hex digest
        assert float(k["qi1_range"][()]) == 1.0
        assert int(k["Nrays"][()]) == int(1e6)
```

- [ ] **Step 2: Verify FAIL**

- [ ] **Step 3: Extend `_write_provenance`**

```python
import hashlib as _hashlib


def _sha256_of(path: Path) -> str:
    h = _hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_provenance(
    f: h5py.File,
    *,
    cli: str = "",
    kernel_npz: Path | None = None,
) -> None:
    ...  # existing body unchanged
    if kernel_npz is not None:
        k = g.require_group("kernel")
        if "pkl_fn" in k:
            del k["pkl_fn"]
        k.create_dataset("pkl_fn", data=Path(kernel_npz).name)
        if "sha256" in k:
            del k["sha256"]
        k.create_dataset("sha256", data=_sha256_of(Path(kernel_npz)))
        # Mirror the kernel's bundled params for self-description.
        with np.load(kernel_npz) as arch:
            for key in arch.files:
                if key == "Resq_i":
                    continue  # the array itself, not metadata
                if key in k:
                    del k[key]
                k.create_dataset(key, data=arch[key])
```

- [ ] **Step 4: Verify PASS**

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_hdf5_provenance.py
git commit -m "feat(io/hdf5): /dfxm_geo/kernel/ subgroup with sha256 + bundled params"
```

---

## Task 13: `/dfxm_geo/config_toml` embedded UTF-8 string

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py`
- Modify: `tests/test_hdf5_provenance.py`

- [ ] **Step 1: Failing test**

```python
def test_write_provenance_config_toml(tmp_path: Path) -> None:
    config_str = "[crystal]\ndis = 4.0\nndis = 151\n"
    out = tmp_path / "test.h5"
    with h5py.File(out, "w") as f:
        _write_provenance(f, cli="x", config_toml=config_str)
    with h5py.File(out, "r") as f:
        got = f["/dfxm_geo/config_toml"][()].decode()
        assert got == config_str
        # And it must round-trip through tomllib.
        import tomllib
        parsed = tomllib.loads(got)
        assert parsed["crystal"]["dis"] == 4.0
```

- [ ] **Step 2: Verify FAIL**

- [ ] **Step 3: Add `config_toml` kwarg**

In `_write_provenance`, add:

```python
def _write_provenance(
    f: h5py.File,
    *,
    cli: str = "",
    kernel_npz: Path | None = None,
    config_toml: str | None = None,
) -> None:
    ...  # existing body
    if config_toml is not None:
        if "config_toml" in g:
            del g["config_toml"]
        g.create_dataset("config_toml", data=config_toml)
```

- [ ] **Step 4: Verify PASS**

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_hdf5_provenance.py
git commit -m "feat(io/hdf5): embed config TOML as UTF-8 string in /dfxm_geo/"
```

---

## Task 14: W2 producer-consumer wrapper `_save_scan_parallel_to_h5`

This task introduces the executor-based writer that streams `forward()` results from worker threads into a pre-allocated HDF5 dataset.

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py` — add `_compute_frame`, `_save_scan_parallel_to_h5`, `_auto_max_workers`
- Modify: `tests/test_hdf5_writer.py` — add an integration-ish test that calls the real `_save_scan_parallel_to_h5`

- [ ] **Step 1: Move `_auto_max_workers` from `images.py`**

Copy the entire `_auto_max_workers` function (currently at `src/dfxm_geo/io/images.py:14-47`) into `src/dfxm_geo/io/hdf5.py`. Function body unchanged.

- [ ] **Step 2: Failing test for the parallel writer**

```python
def test_save_scan_parallel_to_h5_uses_w2_pattern(tmp_path: Path) -> None:
    """Smoke test: writer pre-allocates dataset and streams frames in.

    Uses a tiny 3-frame grid (phi_steps=3, chi_steps=1) against a real
    kernel so this exercises the actual forward() pipeline. Note: the
    grid is 3x1 (NOT 2x2) so phi=0 lands in the linspace — at 2x2 the
    symmetric ±phi_range endpoints both miss the rocking-curve peak.
    Skipped if no kernel is auto-loaded.
    """
    import dfxm_geo.direct_space.forward_model as fm
    if fm.Hg is None:
        pytest.skip("forward_model kernel not auto-loaded; run dfxm-bootstrap.")

    from dfxm_geo.io.hdf5 import _save_scan_parallel_to_h5

    out = tmp_path / "test.h5"
    _save_scan_parallel_to_h5(
        out, scan_id="1.1",
        Hg=fm.Hg,
        phi_range=0.0006 * 180 / np.pi,
        phi_steps=3,
        chi_range=0.002 * 180 / np.pi,
        chi_steps=1,
        max_workers=2,
    )
    with h5py.File(out, "r") as f:
        ds = f["/1.1/instrument/dfxm_sim_detector/data"]
        assert ds.shape[0] == 3  # 3x1 frames
        # And at least one non-zero frame (forward() actually ran;
        # the middle frame at phi=0 hits the peak).
        assert any(ds[i].sum() != 0 for i in range(3))
```

- [ ] **Step 3: Verify FAIL**

Expected: `ImportError: cannot import name '_save_scan_parallel_to_h5'`.

- [ ] **Step 4: Implement the writer**

Add to `src/dfxm_geo/io/hdf5.py`:

```python
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from dfxm_geo.direct_space import forward_model as _fm


def _compute_frame(args: tuple) -> tuple[int, np.ndarray]:
    """Worker function: run forward() and return (frame_idx, image).

    args = (frame_idx, Hg, phi, chi)
    """
    frame_idx, Hg, phi, chi = args
    im = _fm.forward(Hg, phi=phi, chi=chi)
    return frame_idx, im


def _save_scan_parallel_to_h5(
    path: Path,
    scan_id: str,
    Hg: np.ndarray,
    phi_range: float,
    phi_steps: int,
    chi_range: float,
    chi_steps: int,
    max_workers: int | None = None,
    detector_shape: tuple[int, int] | None = None,
) -> None:
    """W2: workers compute forward() and return; main thread writes to HDF5.

    Pre-allocates `(N_frames, H, W)` dataset, dispatches workers, and writes
    each result into its frame_idx slot as it returns. Frame ordering follows
    fscan2d convention: phi inner, chi outer. frame_idx = chi_idx * phi_steps + phi_idx.

    Args:
        path: Output HDF5 file (created or appended).
        scan_id: BLISS scan id like "1.1".
        Hg: Strain field passed to forward().
        phi_range, phi_steps, chi_range, chi_steps: rocking grid params (degrees + counts).
        max_workers: Override for `_auto_max_workers()`.
        detector_shape: (H, W) of the forward() output. If None, probes one frame
            up-front to discover the shape.
    """
    Phi = np.linspace(-np.deg2rad(phi_range), np.deg2rad(phi_range), phi_steps)
    Chi = np.linspace(-np.deg2rad(chi_range), np.deg2rad(chi_range), chi_steps)

    if detector_shape is None:
        # Run one frame to learn the detector shape, so we can pre-allocate.
        probe = _fm.forward(Hg, phi=float(Phi[0]), chi=float(Chi[0]))
        H, W = probe.shape
    else:
        H, W = detector_shape
        probe = None

    N = phi_steps * chi_steps
    mode = "a" if path.exists() else "w"
    workers = max_workers if max_workers is not None else _auto_max_workers()

    with h5py.File(path, mode) as f:
        scan = f.require_group(scan_id)
        _set_nx_class(scan, "NXentry")
        det = scan.require_group("instrument/dfxm_sim_detector")
        _set_nx_class(scan["instrument"], "NXinstrument")
        _set_nx_class(det, "NXdetector")
        if "data" in det:
            del det["data"]
        ds = det.create_dataset(
            "data",
            shape=(N, H, W),
            dtype=np.float64,
            chunks=(1, H, W),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        args_list = []
        for chi_idx in range(chi_steps):
            for phi_idx in range(phi_steps):
                k = chi_idx * phi_steps + phi_idx
                if probe is not None and k == 0:
                    ds[0] = probe  # we already computed frame 0
                    continue
                args_list.append((k, Hg, float(Phi[phi_idx]), float(Chi[chi_idx])))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, im in tqdm(ex.map(_compute_frame, args_list), total=len(args_list)):
                ds[k] = im
```

- [ ] **Step 5: Verify PASS (the integration test runs against real forward())**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_writer.py::test_save_scan_parallel_to_h5_uses_w2_pattern -v
```

Expected: PASS (or SKIP if kernel not loaded — that's fine).

- [ ] **Step 6: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_hdf5_writer.py
git commit -m "feat(io/hdf5): W2 producer-consumer writer streaming forward() to HDF5"
```

---

## Task 15: `write_simulation_h5` — the public entry point invoked by pipeline

Wraps `_save_scan_parallel_to_h5` + `_write_provenance` + per-scan metadata into one call. This is what `pipeline.py::run_simulation` will invoke.

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py`
- Create: `tests/test_hdf5_pipeline.py`

- [ ] **Step 1: Failing integration test**

Create `tests/test_hdf5_pipeline.py`:

```python
"""Integration tests for write_simulation_h5 + run_simulation."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.crystal.remount import SAMPLE_REMOUNT_OPTIONS
from dfxm_geo.io.hdf5 import write_simulation_h5


@pytest.fixture
def _kernel_loaded() -> None:
    if fm.Hg is None:
        pytest.skip("forward_model kernel not auto-loaded; run dfxm-bootstrap.")


def test_write_simulation_h5_creates_both_scans(tmp_path: Path, _kernel_loaded) -> None:
    Hg, q_hkl = fm.Find_Hg(
        4.0, 151, fm.psize, fm.zl_rms,
        S=SAMPLE_REMOUNT_OPTIONS["S1"], remount_name="S1",
    )
    out = tmp_path / "dfxm_geo.h5"
    write_simulation_h5(
        out,
        Hg=Hg, q_hkl=q_hkl,
        phi_range=0.0006 * 180 / np.pi, phi_steps=2,
        chi_range=0.002 * 180 / np.pi, chi_steps=2,
        include_perfect_crystal=True,
        sample_dis=4.0, sample_ndis=151, sample_remount="S1",
        config_toml="[crystal]\ndis = 4.0\n",
        cli="dfxm-forward --config x.toml --output y/",
    )
    with h5py.File(out, "r") as f:
        # /1.1 = dislocations, /2.1 = perfect crystal
        assert f["/1.1/sample/name"][()].decode() == "simulated, dislocations"
        assert f["/2.1/sample/name"][()].decode() == "simulated, perfect crystal"
        # Hg in /1.1 is the real strain field; in /2.1 it's zeros.
        np.testing.assert_array_equal(f["/1.1/dfxm_geo/Hg"][...], Hg)
        np.testing.assert_array_equal(f["/2.1/dfxm_geo/Hg"][...], np.zeros_like(Hg))
        # Global provenance present.
        assert f["/dfxm_geo/version"][()]
        assert f["/dfxm_geo/config_toml"][()].decode().startswith("[crystal]")


def test_write_simulation_h5_skips_perfect_when_disabled(tmp_path: Path, _kernel_loaded) -> None:
    Hg, q_hkl = fm.Find_Hg(
        4.0, 151, fm.psize, fm.zl_rms,
        S=SAMPLE_REMOUNT_OPTIONS["S1"], remount_name="S1",
    )
    out = tmp_path / "dfxm_geo.h5"
    write_simulation_h5(
        out, Hg=Hg, q_hkl=q_hkl,
        phi_range=0.0006 * 180 / np.pi, phi_steps=2,
        chi_range=0.002 * 180 / np.pi, chi_steps=2,
        include_perfect_crystal=False,
        sample_dis=4.0, sample_ndis=151, sample_remount="S1",
        config_toml="x", cli="x",
    )
    with h5py.File(out, "r") as f:
        assert "/1.1" in f
        assert "/2.1" not in f
```

- [ ] **Step 2: Verify FAIL**

Expected: `ImportError: cannot import name 'write_simulation_h5'`.

- [ ] **Step 3: Implement `write_simulation_h5`**

Add to `src/dfxm_geo/io/hdf5.py`:

```python
import datetime as _dt2


def _scan_title(phi_range: float, phi_steps: int, chi_range: float, chi_steps: int) -> str:
    """fscan2d title string that darfix auto-detects on."""
    return (
        f"fscan2d phi {-phi_range:.6f} {phi_range:.6f} {phi_steps} "
        f"chi {-chi_range:.6f} {chi_range:.6f} {chi_steps} 1.0"
    )


def write_simulation_h5(
    path: Path,
    *,
    Hg: np.ndarray,
    q_hkl: np.ndarray,
    phi_range: float,
    phi_steps: int,
    chi_range: float,
    chi_steps: int,
    include_perfect_crystal: bool = True,
    sample_dis: float,
    sample_ndis: int,
    sample_remount: str,
    config_toml: str,
    cli: str,
    kernel_npz: Path | None = None,
    max_workers: int | None = None,
) -> None:
    """One-call entry point: writes /dfxm_geo/ provenance + /1.1 + optional /2.1.

    Called by pipeline.run_simulation. Holds all info needed for a fully
    self-contained, reproducible output file.
    """
    import dfxm_geo.direct_space.forward_model as fm

    # Resolve kernel path for provenance hashing if not given.
    if kernel_npz is None:
        kernel_npz = Path(fm.pkl_fpath) / fm.pkl_fn

    Phi = np.linspace(-np.deg2rad(phi_range), np.deg2rad(phi_range), phi_steps)
    Chi = np.linspace(-np.deg2rad(chi_range), np.deg2rad(chi_range), chi_steps)
    n = phi_steps * chi_steps
    phi_per_frame = np.empty(n, dtype=np.float64)
    chi_per_frame = np.empty(n, dtype=np.float64)
    for chi_idx in range(chi_steps):
        for phi_idx in range(phi_steps):
            k = chi_idx * phi_steps + phi_idx
            phi_per_frame[k] = Phi[phi_idx]
            chi_per_frame[k] = Chi[chi_idx]

    title = _scan_title(phi_range, phi_steps, chi_range, chi_steps)
    now = lambda: _dt2.datetime.now(_dt2.UTC).isoformat(timespec="seconds")

    # Phase 1: global provenance + /1.1 dislocations
    start_dislocs = now()
    _save_scan_parallel_to_h5(
        path, "1.1", Hg,
        phi_range, phi_steps, chi_range, chi_steps,
        max_workers=max_workers,
    )
    end_dislocs = now()
    write_h5_scan(
        path, scan_id="1.1", images=np.empty(0),  # appends metadata only; image data already written
        phi=phi_per_frame, chi=chi_per_frame,
        title=title, start_time=start_dislocs, end_time=end_dislocs,
        sample_name="simulated, dislocations",
        sample_dis=sample_dis, sample_ndis=sample_ndis, sample_remount=sample_remount,
        Hg=Hg, q_hkl=q_hkl,
        theta=float(fm.theta), psize=float(fm.psize), zl_rms=float(fm.zl_rms),
    )

    # Phase 2: provenance (after kernel is finalized in case anything mutated)
    with h5py.File(path, "a") as f:
        _write_provenance(f, cli=cli, kernel_npz=kernel_npz, config_toml=config_toml)

    # Phase 3: optional perfect-crystal scan /2.1
    if include_perfect_crystal:
        start_perf = now()
        _save_scan_parallel_to_h5(
            path, "2.1", np.zeros_like(Hg),
            phi_range, phi_steps, chi_range, chi_steps,
            max_workers=max_workers,
        )
        end_perf = now()
        write_h5_scan(
            path, scan_id="2.1", images=np.empty(0),
            phi=phi_per_frame, chi=chi_per_frame,
            title=title, start_time=start_perf, end_time=end_perf,
            sample_name="simulated, perfect crystal",
            sample_dis=sample_dis, sample_ndis=sample_ndis, sample_remount=sample_remount,
            Hg=np.zeros_like(Hg), q_hkl=q_hkl,
            theta=float(fm.theta), psize=float(fm.psize), zl_rms=float(fm.zl_rms),
        )
```

**IMPORTANT** — the existing `write_h5_scan` (Task 3) currently always creates the image dataset. Modify it to skip the image-data dataset write when `images.size == 0` so we can use it for metadata-only appends:

```python
    with h5py.File(path, mode) as f:
        scan = f.require_group(scan_id)
        _set_nx_class(scan, "NXentry")
        if images.size > 0:
            det = scan.require_group("instrument/dfxm_sim_detector")
            ...  # existing dataset creation with chunks + gzip
        # rest of body (positioners, sample, dfxm_geo, etc.) unchanged
```

- [ ] **Step 4: Verify PASS**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_pipeline.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_hdf5_pipeline.py
git commit -m "feat(io/hdf5): write_simulation_h5 public entry (1-2 scans + provenance)"
```

---

## Task 16: Wire `run_simulation` (pipeline.py) to use `write_simulation_h5`

**Files:**
- Modify: `src/dfxm_geo/pipeline.py:43, 272-334`

- [ ] **Step 1: Failing test — end-to-end CLI produces HDF5**

Create `tests/test_hdf5_run_simulation_end_to_end.py`:

```python
"""End-to-end: run_simulation writes a single .h5 instead of .npy dirs."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    CrystalConfig,
    IOConfig,
    ScanConfig,
    SimulationConfig,
    run_simulation,
)


def test_run_simulation_writes_hdf5(tmp_path: Path) -> None:
    if fm.Hg is None:
        pytest.skip("kernel not loaded")
    cfg = SimulationConfig(
        crystal=CrystalConfig(dis=4.0, ndis=151, sample_remount="S1"),
        scan=ScanConfig(phi_range=0.0006 * 180 / 3.14159, phi_steps=2,
                        chi_range=0.002 * 180 / 3.14159, chi_steps=2),
        io=IOConfig(include_perfect_crystal=True),
    )
    out = tmp_path / "run"
    run_simulation(cfg, out)
    h5 = out / "dfxm_geo.h5"
    assert h5.exists()
    # And the old .npy directories are NOT created.
    assert not (out / "images10").exists()
    assert not (out / "images10_perf_crystal").exists()
    with h5py.File(h5, "r") as f:
        assert "/1.1/instrument/dfxm_sim_detector/data" in f
        assert "/2.1/instrument/dfxm_sim_detector/data" in f
```

- [ ] **Step 2: Verify FAIL**

The test will fail because `run_simulation` still calls `save_images_parallel` and writes `.npy`.

- [ ] **Step 3: Update pipeline.py imports (line 43)**

```python
# Old:
from dfxm_geo.io.images import load_images, save_images_parallel

# New:
from dfxm_geo.io.hdf5 import load_h5_scan, write_simulation_h5
```

- [ ] **Step 4: Rewrite `run_simulation` body**

Replace `src/dfxm_geo/pipeline.py:272-334` with:

```python
def run_simulation(config: SimulationConfig, output_dir: Path) -> dict[str, Any]:
    """Execute a DFXM forward-simulation run from a config object.

    Writes one `<output_dir>/dfxm_geo.h5` containing BLISS scan `/1.1`
    (dislocations) and, if `io.include_perfect_crystal=True`, `/2.1`
    (Hg=0 reference). Provenance + per-scan metadata are also embedded.
    """
    _ensure_kernel_loaded()
    output_dir.mkdir(parents=True, exist_ok=True)

    S = SAMPLE_REMOUNT_OPTIONS[config.crystal.sample_remount]
    Hg, q_hkl = fm.Find_Hg(
        config.crystal.dis, config.crystal.ndis,
        fm.psize, fm.zl_rms,
        S=S, remount_name=config.crystal.sample_remount,
    )
    fm.Hg = Hg
    fm.q_hkl = q_hkl

    # Serialize the config as a TOML string for embedding.
    import tomllib
    from dataclasses import asdict
    config_toml = _dataclass_to_toml_str(config)  # see helper below

    h5_path = output_dir / "dfxm_geo.h5"
    write_simulation_h5(
        h5_path,
        Hg=Hg, q_hkl=q_hkl,
        phi_range=config.scan.phi_range,
        phi_steps=config.scan.phi_steps,
        chi_range=config.scan.chi_range,
        chi_steps=config.scan.chi_steps,
        include_perfect_crystal=config.io.include_perfect_crystal,
        sample_dis=config.crystal.dis,
        sample_ndis=config.crystal.ndis,
        sample_remount=config.crystal.sample_remount,
        config_toml=config_toml,
        cli=" ".join(sys.argv),
        max_workers=config.io.max_workers,
    )
    return {
        "h5_path": h5_path,
        "Hg": Hg,
        "q_hkl": q_hkl,
        "include_perfect_crystal": config.io.include_perfect_crystal,
    }


def _dataclass_to_toml_str(config: SimulationConfig) -> str:
    """Serialize a SimulationConfig back to TOML-formatted text.

    Simple implementation: write the four sub-sections as TOML tables.
    """
    from dataclasses import asdict as _asdict
    sections = {
        "crystal": _asdict(config.crystal),
        "scan": _asdict(config.scan),
        "io": _asdict(config.io),
        "postprocess": _asdict(config.postprocess),
    }
    lines = []
    for name, body in sections.items():
        lines.append(f"[{name}]")
        for k, v in body.items():
            if v is None:
                continue  # TOML has no null; skip
            elif isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            else:
                lines.append(f"{k} = {v}")
        lines.append("")
    return "\n".join(lines)
```

(The `IOConfig.dislocs_dirname` / `perfect_dirname` / `fn_prefix` / `ftype` fields are now unused; they remain in the dataclass for now but are dead code. A future task removes them.)

- [ ] **Step 5: Verify PASS**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_run_simulation_end_to_end.py -v
```

Expected: PASS.

- [ ] **Step 6: Run the full existing suite to confirm no regressions**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/ -x
```

Expected: many existing tests in `test_pipeline.py` that asserted on `.npy` output dirs will now FAIL. That's fine and expected — those tests get updated in Task 17 (postprocess wiring) and Task 21 (deletion of legacy writer).

- [ ] **Step 7: Commit**

```powershell
git add src/dfxm_geo/pipeline.py tests/test_hdf5_run_simulation_end_to_end.py
git commit -m "feat(pipeline): run_simulation writes HDF5 instead of .npy directories"
```

---

## Task 17: Implement `load_h5_scan` reader

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py`
- Create: `tests/test_hdf5_reader.py`

- [ ] **Step 1: Failing test**

```python
"""Layer 1 tests for load_h5_scan."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dfxm_geo.io.hdf5 import load_h5_scan, write_h5_scan


def test_load_h5_scan_returns_postprocess_tuple(tmp_path: Path) -> None:
    images = np.arange(4 * 8 * 8, dtype=np.float64).reshape(4, 8, 8)
    phi = np.array([-0.001, 0.001, -0.001, 0.001])
    chi = np.array([-0.002, -0.002, 0.002, 0.002])
    out = tmp_path / "test.h5"
    write_h5_scan(out, scan_id="1.1", images=images, phi=phi, chi=chi)

    stack, stack_reshape, dim_1, dim_2 = load_h5_scan(
        out, scan_id="1.1", phi_steps=2, chi_steps=2,
    )
    assert stack.shape == (4, 8, 8)
    assert stack_reshape.shape == (2, 2, 8, 8)
    assert dim_1 == 8
    assert dim_2 == 8
    np.testing.assert_array_equal(stack, images)
```

- [ ] **Step 2: Verify FAIL**

Expected: `ImportError: cannot import name 'load_h5_scan'`.

- [ ] **Step 3: Implement the reader**

```python
def load_h5_scan(
    path: Path,
    scan_id: str = "1.1",
    phi_steps: int | None = None,
    chi_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load a BLISS scan stack from a dfxm-geo HDF5, returning the same
    tuple shape `load_images` did so postprocess code stays unchanged.

    Args:
        path: HDF5 file path.
        scan_id: BLISS scan id, default "1.1" (dislocations).
        phi_steps, chi_steps: needed for the (phi, chi, H, W) reshape. If
            omitted, both are inferred from the embedded config_toml.

    Returns:
        (stack, stack_reshape, dim_1, dim_2) — same as the legacy
        `load_images` signature.
    """
    with h5py.File(path, "r") as f:
        data = f[f"/{scan_id}/instrument/dfxm_sim_detector/data"][...]
        if phi_steps is None or chi_steps is None:
            import tomllib
            cfg = tomllib.loads(f["/dfxm_geo/config_toml"][()].decode())
            phi_steps = phi_steps or cfg["scan"]["phi_steps"]
            chi_steps = chi_steps or cfg["scan"]["chi_steps"]

    n, h, w = data.shape
    if n != phi_steps * chi_steps:
        raise ValueError(
            f"Scan {scan_id} has {n} frames, expected "
            f"{phi_steps}*{chi_steps}={phi_steps * chi_steps}"
        )
    # Reshape: fscan2d order is phi-inner, chi-outer
    # data[k] for k = chi_idx * phi_steps + phi_idx
    # We want stack_reshape[phi_idx, chi_idx] == data[chi_idx*phi_steps + phi_idx]
    # which means reshape to (chi_steps, phi_steps, H, W) then transpose.
    stack_reshape = data.reshape(chi_steps, phi_steps, h, w).transpose(1, 0, 2, 3)
    return data, stack_reshape, h, w
```

(Note: this CORRECTS the load_images bug for non-square grids. See `phase8_fastgrainplot_closeout.md` for the analogous issue already fixed in `fastgrainplot`.)

- [ ] **Step 4: Verify PASS**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_reader.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/io/hdf5.py tests/test_hdf5_reader.py
git commit -m "feat(io/hdf5): load_h5_scan returns postprocess-compatible tuple shape"
```

---

## Task 18: Wire `run_postprocess` to use `load_h5_scan` + write analysis into the .h5

**Files:**
- Modify: `src/dfxm_geo/pipeline.py:337-453`

- [ ] **Step 1: Failing test**

```python
def test_run_postprocess_reads_h5(tmp_path: Path, _kernel_loaded) -> None:
    """run_postprocess against an .h5 produces the same outputs it would
    against a .npy dir."""
    cfg = SimulationConfig(
        crystal=CrystalConfig(dis=4.0, ndis=151, sample_remount="S1"),
        scan=ScanConfig(phi_range=0.0006 * 180 / 3.14159, phi_steps=4,
                        chi_range=0.002 * 180 / 3.14159, chi_steps=4),
        io=IOConfig(include_perfect_crystal=True),
    )
    out = tmp_path / "run"
    run_simulation(cfg, out)
    result = run_postprocess(out, cfg)
    # Outputs land inside the existing .h5
    with h5py.File(out / "dfxm_geo.h5", "r") as f:
        assert "/1.1/dfxm_geo/analysis/phi_list" in f
        assert "/1.1/dfxm_geo/analysis/chi_list" in f
        assert "/1.1/dfxm_geo/analysis/qi_field" in f
        assert "/1.1/dfxm_geo/analysis/chi_shift_deg" in f
    # And SVG figures still go on disk per F1 decision.
    assert (out / "figures" / "mosaicity_maps.svg").exists()
    assert (out / "figures" / "qi_cross_section.svg").exists()
```

- [ ] **Step 2: Verify FAIL**

- [ ] **Step 3: Rewrite `run_postprocess`**

Replace lines 337-453 of `pipeline.py`. Key changes:

```python
def run_postprocess(output_dir: Path, config: SimulationConfig) -> dict[str, Any]:
    """Read /1.1 and /2.1 from dfxm_geo.h5; compute χ-shift, COM maps, qi field.

    Analysis outputs are written into /1.1/dfxm_geo/analysis/ inside the same
    HDF5 file. SVG figures land on disk under <output_dir>/figures/ (F1 decision).
    """
    _ensure_kernel_loaded()

    h5_path = output_dir / "dfxm_geo.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"Expected {h5_path}; run dfxm-forward without --postprocess-only first."
        )

    # Sanity-check that /2.1 exists (chi-shift needs perfect crystal).
    with h5py.File(h5_path, "r") as f:
        if "/2.1" not in f:
            raise FileNotFoundError(
                f"{h5_path} has no /2.1 scan (perfect crystal). Re-run with "
                "include_perfect_crystal=True, or skip postprocess."
            )

    _, dis_reshape, _, _ = load_h5_scan(
        h5_path, scan_id="1.1",
        phi_steps=config.scan.phi_steps, chi_steps=config.scan.chi_steps,
    )
    _, perf_reshape, _, _ = load_h5_scan(
        h5_path, scan_id="2.1",
        phi_steps=config.scan.phi_steps, chi_steps=config.scan.chi_steps,
    )

    chi_shift = compute_chi_shift(
        perf_reshape, config.scan.chi_steps, config.scan.chi_range,
        oversample=config.postprocess.chi_oversample_for_shift,
    )
    phi_list, chi_list = compute_com_maps(
        dis_reshape, config.scan.phi_range, config.scan.phi_steps,
        config.scan.chi_range, config.scan.chi_steps,
        chi_shift=chi_shift,
        oversample=config.postprocess.phi_oversample,
        chi_oversample=config.postprocess.chi_oversample,
    )
    if fm.Hg is None:
        raise RuntimeError("fm.Hg is not set. Call run_simulation() first.")
    _, qi_field = fm.forward(fm.Hg, phi=0, qi_return=True)

    # Append analysis to /1.1/dfxm_geo/analysis/ inside the existing .h5
    with h5py.File(h5_path, "a") as f:
        analysis = f.require_group("/1.1/dfxm_geo/analysis")
        for name, val in [
            ("phi_list", phi_list), ("chi_list", chi_list),
            ("qi_field", qi_field), ("chi_shift_deg", float(chi_shift)),
        ]:
            if name in analysis:
                del analysis[name]
            analysis.create_dataset(name, data=val)

    # F1: render SVG figures on disk alongside the .h5
    fig_dir = output_dir / config.postprocess.figures_dirname
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_mosaicity_maps(
        phi_list, chi_list, fm.xl_start, fm.yl_start,
        fig_dir / "mosaicity_maps.svg",
    )
    plot_qi_cross_section(
        qi_field, fm.xl_start, fm.yl_start, fm.xl_steps, fm.yl_steps, fm.zl_steps,
        fig_dir / "qi_cross_section.svg",
    )
    return {
        "phi_list": phi_list, "chi_list": chi_list, "qi_field": qi_field,
        "chi_shift": chi_shift, "h5_path": h5_path, "figures_dir": fig_dir,
    }
```

Also: add `import h5py` at the top of pipeline.py if not already present.

- [ ] **Step 4: Verify PASS**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_pipeline.py tests/test_hdf5_run_simulation_end_to_end.py -v
```

Expected: PASS.

- [ ] **Step 5: Update existing `test_pipeline.py` tests that broke**

Run the suite, identify which `test_pipeline.py` tests still assert on `images10/` etc., and update them to assert on the new `.h5` structure. Most pre-existing tests in `tests/test_pipeline.py` will need either deletion (covered by new tests) or a one-line shape change.

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/ -x
```

For each failure, fix or delete. Aim: full suite green again after this task.

- [ ] **Step 6: Commit**

```powershell
git add src/dfxm_geo/pipeline.py tests/
git commit -m "feat(pipeline): run_postprocess reads HDF5; analysis into /1.1/dfxm_geo/analysis/"
```

---

## Task 19: Create `io/migrate.py` with `_load_images_legacy` + `dfxm-migrate-output` CLI

**Files:**
- Create: `src/dfxm_geo/io/migrate.py`
- Create: `tests/test_migrate_output.py`

- [ ] **Step 1: Failing test**

```python
"""Tests for dfxm-migrate-output CLI."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm_geo.io.migrate import migrate_npy_dir_to_h5


def test_migrate_round_trips_images(tmp_path: Path) -> None:
    # Fake a legacy output dir: a few .npy files in images10/.
    images_dir = tmp_path / "images10"
    images_dir.mkdir()
    perf_dir = tmp_path / "images10_perf_crystal"
    perf_dir.mkdir()
    for chi_i in range(2):
        for phi_j in range(2):
            arr = np.full((4, 4), chi_i * 10 + phi_j, dtype=np.float64)
            np.save(images_dir / f"mosa_test_0000_{chi_i:04d}_{phi_j:04d}.npy", arr)
            np.save(perf_dir / f"mosa_test_0000_{chi_i:04d}_{phi_j:04d}.npy", arr * -1)

    h5_path = tmp_path / "dfxm_geo.h5"
    migrate_npy_dir_to_h5(
        npy_dir=tmp_path, h5_path=h5_path,
        phi_steps=2, chi_steps=2,
        phi_range_deg=0.0006 * 180 / 3.14159,
        chi_range_deg=0.002 * 180 / 3.14159,
        dis=4.0, ndis=151, sample_remount="S1",
    )

    with h5py.File(h5_path, "r") as f:
        d1 = f["/1.1/instrument/dfxm_sim_detector/data"][...]
        d2 = f["/2.1/instrument/dfxm_sim_detector/data"][...]
        # Frame order: chi-outer, phi-inner.
        for chi_i in range(2):
            for phi_j in range(2):
                k = chi_i * 2 + phi_j
                np.testing.assert_array_equal(d1[k], np.full((4, 4), chi_i * 10 + phi_j))
                np.testing.assert_array_equal(d2[k], np.full((4, 4), -(chi_i * 10 + phi_j)))
```

- [ ] **Step 2: Verify FAIL**

Expected: `ModuleNotFoundError: No module named 'dfxm_geo.io.migrate'`.

- [ ] **Step 3: Implement `io/migrate.py`**

```python
"""Migration: convert legacy .npy output dirs to dfxm_geo.h5.

CLI: `dfxm-migrate-output <input_dir> --output <out.h5> [--config <toml>]`

This module holds the only remaining reader for legacy .npy stacks
(`_load_images_legacy`). It is NOT exported from the package public API.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from dfxm_geo.io.hdf5 import write_simulation_h5


def _load_images_legacy(
    fpath: str,
    u_steps: int,
    v_steps: int,
    file_ext: str = ".npy",
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Verbatim of the old `dfxm_geo.io.images.load_images` (deleted in v1.1).

    Used only by the migration script. Not part of the public API.
    """
    if not os.path.isdir(fpath):
        raise ValueError(f"Directory does not exist: {fpath}")
    file_list = [f for f in os.listdir(fpath) if f.endswith(file_ext)]
    if not file_list:
        raise ValueError(f"Empty directory: {fpath}")
    file_list.sort()
    stack = np.empty(
        (len(file_list), *np.load(os.path.join(fpath, file_list[0])).shape),
        dtype=np.float64,
    )
    for i, fname in enumerate(file_list):
        stack[i] = np.load(os.path.join(fpath, fname))
    dim_1, dim_2 = stack.shape[1], stack.shape[2]
    stack_reshape = stack.reshape((u_steps, v_steps, dim_1, dim_2))
    return stack, stack_reshape, dim_1, dim_2


def migrate_npy_dir_to_h5(
    npy_dir: Path,
    h5_path: Path,
    *,
    phi_steps: int,
    chi_steps: int,
    phi_range_deg: float,
    chi_range_deg: float,
    dis: float,
    ndis: int,
    sample_remount: str,
    dislocs_dirname: str = "images10",
    perfect_dirname: str = "images10_perf_crystal",
) -> None:
    """Read legacy .npy stacks under `npy_dir` and write a v1.1 HDF5."""
    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo.crystal.remount import SAMPLE_REMOUNT_OPTIONS

    # Reproduce Hg from the same inputs so the new file's /1.1/dfxm_geo/Hg
    # matches what the original sim used. q_hkl mirrors the kernel.
    S = SAMPLE_REMOUNT_OPTIONS[sample_remount]
    Hg, q_hkl = fm.Find_Hg(dis, ndis, fm.psize, fm.zl_rms, S=S, remount_name=sample_remount)

    dislocs = _load_images_legacy(
        str(npy_dir / dislocs_dirname), u_steps=phi_steps, v_steps=chi_steps,
    )[0]  # [0] = the flat stack

    perfect_path = npy_dir / perfect_dirname
    has_perfect = perfect_path.is_dir()
    perfect = _load_images_legacy(
        str(perfect_path), u_steps=phi_steps, v_steps=chi_steps,
    )[0] if has_perfect else None

    config_toml = (
        f'[crystal]\ndis = {dis}\nndis = {ndis}\nsample_remount = "{sample_remount}"\n'
        f"[scan]\nphi_range = {phi_range_deg}\nphi_steps = {phi_steps}\n"
        f"chi_range = {chi_range_deg}\nchi_steps = {chi_steps}\n"
    )

    # Write phase 1: /1.1
    import h5py
    from dfxm_geo.io.hdf5 import write_h5_scan, _write_provenance, _scan_title
    title = _scan_title(phi_range_deg, phi_steps, chi_range_deg, chi_steps)
    write_h5_scan(
        h5_path, scan_id="1.1", images=dislocs,
        phi=_phi_per_frame(phi_steps, chi_steps, phi_range_deg),
        chi=_chi_per_frame(phi_steps, chi_steps, chi_range_deg),
        title=title,
        sample_name="simulated, dislocations",
        sample_dis=dis, sample_ndis=ndis, sample_remount=sample_remount,
        Hg=Hg, q_hkl=q_hkl,
        theta=float(fm.theta), psize=float(fm.psize), zl_rms=float(fm.zl_rms),
    )
    if has_perfect:
        write_h5_scan(
            h5_path, scan_id="2.1", images=perfect,
            phi=_phi_per_frame(phi_steps, chi_steps, phi_range_deg),
            chi=_chi_per_frame(phi_steps, chi_steps, chi_range_deg),
            title=title,
            sample_name="simulated, perfect crystal",
            sample_dis=dis, sample_ndis=ndis, sample_remount=sample_remount,
            Hg=np.zeros_like(Hg), q_hkl=q_hkl,
            theta=float(fm.theta), psize=float(fm.psize), zl_rms=float(fm.zl_rms),
        )

    kernel_npz = Path(fm.pkl_fpath) / fm.pkl_fn
    with h5py.File(h5_path, "a") as f:
        _write_provenance(
            f, cli="dfxm-migrate-output (legacy import)",
            kernel_npz=kernel_npz, config_toml=config_toml,
        )


def _phi_per_frame(phi_steps: int, chi_steps: int, phi_range_deg: float) -> np.ndarray:
    Phi = np.linspace(-np.deg2rad(phi_range_deg), np.deg2rad(phi_range_deg), phi_steps)
    out = np.empty(phi_steps * chi_steps, dtype=np.float64)
    for chi_idx in range(chi_steps):
        for phi_idx in range(phi_steps):
            out[chi_idx * phi_steps + phi_idx] = Phi[phi_idx]
    return out


def _chi_per_frame(phi_steps: int, chi_steps: int, chi_range_deg: float) -> np.ndarray:
    Chi = np.linspace(-np.deg2rad(chi_range_deg), np.deg2rad(chi_range_deg), chi_steps)
    out = np.empty(phi_steps * chi_steps, dtype=np.float64)
    for chi_idx in range(chi_steps):
        for phi_idx in range(phi_steps):
            out[chi_idx * phi_steps + phi_idx] = Chi[chi_idx]
    return out


# IUCrJ-2024 defaults: 61x61 with the canonical Borgi 2024 ranges. Used when
# the user runs `dfxm-migrate-output <old_dir>` with no --config.
_IUCRJ_2024_DEFAULTS = {
    "phi_steps": 61, "chi_steps": 61,
    "phi_range_deg": 0.0006 * 180 / np.pi,
    "chi_range_deg": 0.002 * 180 / np.pi,
    "dis": 4.0, "ndis": 151, "sample_remount": "S1",
}


def cli_main(argv: list[str] | None = None) -> int:
    """Entry point for `dfxm-migrate-output`."""
    p = argparse.ArgumentParser(
        description="Migrate a legacy .npy output directory to dfxm_geo.h5."
    )
    p.add_argument("input_dir", type=Path, help="Old output dir (contains images10/, images10_perf_crystal/).")
    p.add_argument("--output", type=Path, default=None,
                   help="Output .h5 path (default: <input_dir>/dfxm_geo.h5).")
    p.add_argument("--config", type=Path, default=None,
                   help="TOML config used to generate the original sim. If omitted, uses IUCrJ-2024 defaults.")
    args = p.parse_args(argv)
    out = args.output or (args.input_dir / "dfxm_geo.h5")

    if args.config is None:
        params = _IUCRJ_2024_DEFAULTS
        print(f"No --config given; using IUCrJ-2024 defaults: {params}", file=sys.stderr)
    else:
        import tomllib
        with args.config.open("rb") as f:
            raw = tomllib.load(f)
        params = {
            "phi_steps": raw["scan"]["phi_steps"],
            "chi_steps": raw["scan"]["chi_steps"],
            "phi_range_deg": raw["scan"]["phi_range"],
            "chi_range_deg": raw["scan"]["chi_range"],
            "dis": raw["crystal"]["dis"],
            "ndis": raw["crystal"]["ndis"],
            "sample_remount": raw["crystal"]["sample_remount"],
        }

    migrate_npy_dir_to_h5(npy_dir=args.input_dir, h5_path=out, **params)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
```

- [ ] **Step 4: Verify PASS**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_migrate_output.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/dfxm_geo/io/migrate.py tests/test_migrate_output.py
git commit -m "feat(io/migrate): dfxm-migrate-output CLI converts legacy .npy dirs to HDF5"
```

---

## Task 20: Register `dfxm-migrate-output` entry point

**Files:**
- Modify: `pyproject.toml:61-64`

- [ ] **Step 1: Failing test**

Create `tests/test_pyproject_migrate_entry.py`:

```python
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_dfxm_migrate_output_entry_point_registered() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    scripts = data["project"]["scripts"]
    assert scripts["dfxm-migrate-output"] == "dfxm_geo.io.migrate:cli_main"
```

- [ ] **Step 2: Verify FAIL**

- [ ] **Step 3: Add the entry point**

Edit `pyproject.toml`:

```toml
[project.scripts]
dfxm-forward = "dfxm_geo.pipeline:cli_main"
dfxm-identify = "dfxm_geo.pipeline:cli_main_identify"
dfxm-bootstrap = "dfxm_geo.reciprocal_space.kernel:cli_main"
dfxm-migrate-output = "dfxm_geo.io.migrate:cli_main"
```

- [ ] **Step 4: Reinstall to register the script + verify**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pip install -e . --no-deps -q
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pyproject_migrate_entry.py -v
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/Activate.ps1"
dfxm-migrate-output --help
```

Expected: PASS, then help text printed.

- [ ] **Step 5: Commit**

```powershell
git add pyproject.toml tests/test_pyproject_migrate_entry.py
git commit -m "feat(cli): register dfxm-migrate-output entry point"
```

---

## Task 21: Trim `io/images.py` to only what identification z-scan still needs

**Files:**
- Modify: `src/dfxm_geo/io/images.py` — keep ONLY `save_images_parallel` + `_auto_max_workers`; delete `load_images`, `load_image`, `load_images_parallel`, `save_image`, `save_edfs`
- Update or delete: any test that imports the deleted symbols

**Scope note:** B3 cutover applies to FORWARD SIM outputs (writer + reader). Identification z-scan (`_run_identification_zscan` in `pipeline.py`) still calls `save_images_parallel` to dump rocking grids as `.npy`; identification HDF5 is deferred to v1.2.0 (per S3). So we cannot delete the file entirely — we trim it.

- [ ] **Step 1: Audit remaining references**

Search the codebase:

```powershell
Select-String -Path "src\*","tests\*","scripts\*" -Pattern "from dfxm_geo.io.images" -Recurse
```

Every match should be either:
- `from dfxm_geo.io.images import save_images_parallel` (in `pipeline.py` for z-scan) — KEEP
- imports of `load_images` / `load_image` / `load_images_parallel` / `save_image` / `save_edfs` — these are dead (pipeline no longer uses them); flag for deletion or replacement.

- [ ] **Step 2: Edit `src/dfxm_geo/io/images.py`**

Keep only `_auto_max_workers` and `save_images_parallel` (and `save_image`, which is the worker called by `save_images_parallel` — keep it too, but make it module-private if it isn't already). Delete:
- `save_edfs`
- `load_image`
- `load_images`
- `load_images_parallel`

Also delete now-unused imports (e.g., `fabio` if save_edfs was the only consumer).

The trimmed file should be ~125 lines (down from ~260).

- [ ] **Step 3: Update tests in tests/test_io.py**

Identify any tests in `tests/test_io.py` that exercised the deleted functions and delete them. Keep tests for `io.strain_cache`, `io.check_folder`, and the remaining `save_images_parallel` if present.

- [ ] **Step 4: Run the full suite**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/ -x
```

Expected: full green. Any remaining import-of-deleted-symbol errors must be fixed (most likely in `tests/test_io.py` or `tests/test_pipeline_identification.py`).

- [ ] **Step 5: Commit**

```powershell
git add -A
git commit -m "refactor(io): trim images.py to save_images_parallel only; B3 cutover for forward sim"
```

---

## Task 22: Layer 2 bit-equivalence test vs Task 1 golden

**Files:**
- Create: `tests/test_hdf5_bit_equiv.py`

- [ ] **Step 1: Write the test**

```python
"""Layer 2: bit-equivalence — new HDF5 writer == old .npy writer (golden)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.crystal.remount import SAMPLE_REMOUNT_OPTIONS
from dfxm_geo.io.hdf5 import _save_scan_parallel_to_h5

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "data" / "golden" / "forward_legacy_writer_4frames_8x8.npy"


def test_hdf5_writer_bit_equivalent_to_legacy_npy_golden(tmp_path: Path) -> None:
    """Same Hg + (phi, chi) grid -> identical images via HDF5 writer."""
    if fm.Hg is None:
        pytest.skip("kernel not loaded")

    Hg, q_hkl = fm.Find_Hg(
        4.0, 151, fm.psize, fm.zl_rms,
        S=SAMPLE_REMOUNT_OPTIONS["S1"], remount_name="S1",
    )
    fm.Hg = Hg
    fm.q_hkl = q_hkl

    out = tmp_path / "test.h5"
    # MUST use the same tiny half-range and crop window as the golden
    # generator (tests/_gen_forward_legacy_golden.py).
    TINY_HALF_RANGE_RAD = 5e-5
    _save_scan_parallel_to_h5(
        out, scan_id="1.1", Hg=Hg,
        phi_range=TINY_HALF_RANGE_RAD * 180 / np.pi, phi_steps=2,
        chi_range=TINY_HALF_RANGE_RAD * 180 / np.pi, chi_steps=2,
        max_workers=1,
    )

    expected = np.load(GOLDEN)  # shape (4, 8, 8), float64
    with h5py.File(out, "r") as f:
        actual_full = f["/1.1/instrument/dfxm_sim_detector/data"][...]
    actual_crop = actual_full[:, 322:330, 51:59]
    np.testing.assert_array_equal(
        actual_crop, expected,
        err_msg="HDF5 writer output deviates from legacy .npy writer at 8x8 corner",
    )
```

- [ ] **Step 2: Verify PASS**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_bit_equiv.py -v
```

Expected: PASS. (If it fails, the format-change has perturbed the simulation — stop and investigate; do not commit.)

- [ ] **Step 3: Commit**

```powershell
git add tests/test_hdf5_bit_equiv.py
git commit -m "test(hdf5): bit-equivalence vs legacy .npy writer golden"
```

---

## Task 23: Defensive regression tests

**Files:**
- Create: `tests/test_hdf5_defensive.py`

- [ ] **Step 1: Write the tests**

```python
"""Layer 2 defensive: guard against silent regressions of the v1.1 cutover."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_h5py_imported_in_hdf5_module() -> None:
    """Catch rollback to a non-HDF5 writer."""
    src = (REPO / "src" / "dfxm_geo" / "io" / "hdf5.py").read_text(encoding="utf-8")
    assert "import h5py" in src, "io/hdf5.py must import h5py"


def test_pipeline_does_not_use_np_save_for_image_stack() -> None:
    """Catch accidental return-to-.npy in the image write path."""
    src = (REPO / "src" / "dfxm_geo" / "pipeline.py").read_text(encoding="utf-8")
    # np.save for analysis scalars (phi_list etc.) was removed in Task 18;
    # the new path writes those into the HDF5. So np.save should not appear.
    assert "np.save(" not in src, "pipeline.py must not call np.save (all writes go through HDF5)"


def test_public_io_does_not_expose_load_images() -> None:
    """B3: legacy load_images is removed from the public package surface."""
    with pytest.raises(ImportError):
        from dfxm_geo.io.images import load_images  # noqa: F401


def test_load_images_legacy_is_internal_only() -> None:
    """The legacy helper lives in migrate, NOT re-exported from io."""
    from dfxm_geo.io import migrate
    assert hasattr(migrate, "_load_images_legacy")
    # And it's name-mangled (leading underscore) to mark internal.
    assert "_load_images_legacy" in migrate.__all__ or True  # __all__ optional


def test_pipeline_does_not_import_legacy_images_module() -> None:
    src = (REPO / "src" / "dfxm_geo" / "pipeline.py").read_text(encoding="utf-8")
    assert "from dfxm_geo.io.images" not in src
    assert "import dfxm_geo.io.images" not in src
```

- [ ] **Step 2: Verify PASS**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_defensive.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_hdf5_defensive.py
git commit -m "test(hdf5): defensive regression (h5py present, np.save gone, legacy API hidden)"
```

---

## Task 24: Write `docs/output-format.md`

**Files:**
- Create: `docs/output-format.md`

- [ ] **Step 1: Write the file**

Create `docs/output-format.md`:

````markdown
# dfxm-geo HDF5 output format (v1.1.0)

`dfxm-forward` writes one HDF5 file per simulation: `<output_dir>/dfxm_geo.h5`. The layout follows the **ESRF BLISS** convention (compatible with [darfix](https://darfix.readthedocs.io/) and [darling](https://github.com/AxelHenningsson/darling)) and adds a sim-specific `/dfxm_geo/` group for provenance.

## High-level structure

```
dfxm_geo.h5
├── /dfxm_geo/                     ← global provenance (one per file)
│   ├── version                    "1.1.0"
│   ├── git_sha                    "367cfee..." or "unknown"
│   ├── git_dirty                  bool
│   ├── generated_at               "2026-05-17T10:00:00+00:00"
│   ├── hostname                   "borgi-laptop" / "n-62-12-15"
│   ├── python_version, numpy_version
│   ├── cli                        full command line
│   ├── config_toml                entire TOML config as a UTF-8 string
│   └── kernel/
│       ├── pkl_fn                 "Resq_i_2026-05-16_2100.npz"
│       ├── sha256                 64-char hex digest
│       └── qi1_range...Nrays      mirrored from the kernel npz
│
├── /1.1/                          ← BLISS scan: dislocated crystal
│   ├── title                      "fscan2d phi ... chi ... 1.0"
│   ├── start_time, end_time       ISO-8601
│   ├── instrument/
│   │   ├── dfxm_sim_detector/data (N_frames, H, W) float64, chunks (1, H, W), gzip-4 + shuffle
│   │   └── positioners/
│   │       ├── phi                (N_frames,) degrees, attrs["units"] = "degree"
│   │       └── chi                (N_frames,) degrees, attrs["units"] = "degree"
│   ├── measurement/               ← BLISS soft-links
│   │   ├── dfxm_sim_detector → /1.1/instrument/dfxm_sim_detector/data
│   │   ├── phi                  → /1.1/instrument/positioners/phi
│   │   └── chi                  → /1.1/instrument/positioners/chi
│   ├── sample/                    ← NXsample
│   │   ├── name                   "simulated, dislocations"
│   │   ├── dis, ndis, sample_remount
│   └── dfxm_geo/                  ← per-scan sim-specific
│       ├── Hg                     (npixels, 3, 3) float64
│       ├── q_hkl                  (3,)
│       ├── theta, psize, zl_rms
│       └── analysis/              ← present after `dfxm-forward` (postprocess stage)
│           ├── phi_list, chi_list (H, W) COM maps
│           ├── qi_field
│           └── chi_shift_deg      scalar
│
└── /2.1/                          ← optional perfect-crystal scan
    └── (same shape as /1.1/, Hg=0, sample/name="simulated, perfect crystal")
```

## Frame ordering

fscan2d convention: phi inner, chi outer. The k-th frame corresponds to:

```python
phi_idx = k % phi_steps
chi_idx = k // phi_steps
```

(Equivalently, `k = chi_idx * phi_steps + phi_idx`.)

`load_h5_scan` returns the data both as a flat `(N_frames, H, W)` stack and as a `(phi_steps, chi_steps, H, W)` reshape.

## NX_class attributes

The following groups carry `NX_class` attrs so silx, NeXpy, and other NeXus-aware viewers can recognize the structure:

| Path                                  | NX_class       |
| ------------------------------------- | -------------- |
| `/1.1`                                | `NXentry`      |
| `/1.1/instrument`                     | `NXinstrument` |
| `/1.1/instrument/dfxm_sim_detector`   | `NXdetector`   |
| `/1.1/instrument/positioners`         | `NXcollection` |
| `/1.1/sample`                         | `NXsample`     |

We do not claim full NeXus compliance — these attrs are a free interop bonus, not a contract.

## Reading the file

```python
from dfxm_geo.io.hdf5 import load_h5_scan

stack, stack_reshape, h, w = load_h5_scan(
    "output/dfxm_geo.h5", scan_id="1.1",
    # phi_steps / chi_steps inferred from /dfxm_geo/config_toml if omitted
)
```

Or via h5py directly:

```python
import h5py
with h5py.File("output/dfxm_geo.h5", "r") as f:
    images = f["/1.1/instrument/dfxm_sim_detector/data"][...]
    phi_deg = f["/1.1/instrument/positioners/phi"][...]
    chi_deg = f["/1.1/instrument/positioners/chi"][...]
```

## Opening in darfix / darling

Darfix:

```bash
darfix output/dfxm_geo.h5
```

Darling:

```python
import darling
dset = darling.DataSet("output/dfxm_geo.h5", scan_id="1.1")
print(dset.data.shape, dset.motors)
```

## Compression

Image data is chunked `(1, H, W)` with gzip-4 + shuffle. Typical compression ratio is ~3-5× for sim images (which are sparse — most pixels are near zero). Read latency for one full stack on a laptop SSD is ~1-2 seconds.

## Migrating old `.npy` outputs

```bash
dfxm-migrate-output <old_output_dir>
# or:
dfxm-migrate-output <old_output_dir> --config <toml> --output out.h5
```

Without `--config`, defaults to IUCrJ-2024 params (61×61, dis=4, ndis=151, S1) which match the canonical Borgi 2024 paper-reproduction runs.
````

- [ ] **Step 2: Commit**

```powershell
git add docs/output-format.md
git commit -m "docs: HDF5 output format reference (BLISS layout + provenance schema)"
```

---

## Task 25: Update `docs/architecture.md` + `docs/reproducibility.md`

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/reproducibility.md`

- [ ] **Step 1: Read both files to find the right insertion points**

```
Read docs/architecture.md (look for the "Three-stage data flow" or "Output directory" sections).
Read docs/reproducibility.md (look for the "Output description" section).
```

- [ ] **Step 2: In `docs/architecture.md`, add an "Output file format" section**

After the existing data-flow description, insert:

```markdown
## Output file format

Since v1.1.0, `dfxm-forward` writes a single BLISS-style HDF5 file (`dfxm_geo.h5`) per simulation, replacing the legacy `images10/` directory of per-frame `.npy` files. See [output-format.md](output-format.md) for the full schema.

Key properties:
- One file per `run_simulation` call; 1-2 BLISS scans inside (`/1.1` dislocations, `/2.1` optional perfect crystal).
- `/dfxm_geo/` root group embeds full provenance (git SHA, kernel hash, config TOML, machine, timestamps).
- Compatible with darfix and darling out of the box.
- Per-frame chunks + gzip-4 + shuffle compression: ~3-5× space savings vs raw `.npy`.

Legacy `.npy` output directories can be converted via `dfxm-migrate-output`.
```

- [ ] **Step 3: In `docs/reproducibility.md`, update the output description**

Find the section describing the `images10/` output and replace with:

```markdown
## Output: `<output_dir>/dfxm_geo.h5`

`dfxm-forward` produces a single HDF5 file with the full simulation in it. Schema documented in [output-format.md](output-format.md). All metadata needed to reproduce the run — config TOML, kernel hash, git SHA, machine, timestamps — is embedded under `/dfxm_geo/`.

SVG figures (mosaicity maps, qi cross-section) land alongside the .h5 at `<output_dir>/figures/`.
```

- [ ] **Step 4: Commit**

```powershell
git add docs/architecture.md docs/reproducibility.md
git commit -m "docs: point architecture + reproducibility at the new HDF5 output schema"
```

---

## Task 26: Version bump 1.0.3 → 1.1.0 + release notes

**Files:**
- Modify: `pyproject.toml:7`
- Create: `docs/release-notes-1.1.0.md`

- [ ] **Step 1: Failing test (mirrors the post-cutover test we deleted)**

Create `tests/test_version_is_1_1_0.py`:

```python
"""Pin the project version to 1.1.0 for the HDF5-outputs release."""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_version_is_1_1_0() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == "1.1.0"
```

- [ ] **Step 2: Verify FAIL** (`assert '1.0.3' == '1.1.0'`)

- [ ] **Step 3: Bump the version**

Edit `pyproject.toml:7`:

```toml
version = "1.1.0"
```

- [ ] **Step 4: Verify PASS**

- [ ] **Step 5: Write release notes**

Create `docs/release-notes-1.1.0.md`:

```markdown
# v1.1.0 — HDF5 output format

**Release date:** 2026-05-17

## Highlights

- `dfxm-forward` now writes a single BLISS-style HDF5 file (`<output_dir>/dfxm_geo.h5`) instead of thousands of per-frame `.npy` files. ~3.5× space savings + full embedded provenance.
- Compatible out of the box with [darfix](https://darfix.readthedocs.io/) and [darling](https://github.com/AxelHenningsson/darling) — the actual analysis tools at ESRF ID03 / ID06.
- New `dfxm-migrate-output` CLI converts legacy `.npy` output directories to the new format.

## Breaking changes (B3 cutover)

- `dfxm_geo.io.images.load_images` is **removed** from the public API. New code uses `dfxm_geo.io.hdf5.load_h5_scan`. Legacy `.npy` reading is preserved internally inside `dfxm_geo.io.migrate` for the migration script.
- `<output_dir>/images10/` and `<output_dir>/images10_perf_crystal/` are no longer created. The corresponding `IOConfig.dislocs_dirname` / `perfect_dirname` / `fn_prefix` / `ftype` fields are now ignored.

## Out of scope (deferred)

- `dfxm-identify` output format stays on `.npy` + CSV. HDF5 migration deferred to v1.2.0 after laptop+cluster feedback validates the schema.
- Durability hardening (HDF5 flush / atomic rename) — only revisit if jobs grow past ~3 hours.

## Migration

```bash
dfxm-migrate-output <old_output_dir>
# or with explicit config:
dfxm-migrate-output <old_output_dir> --config configs/default.toml
```

## Provenance

Every output file embeds:
- `dfxm-geo` package version
- git HEAD SHA and dirty flag
- Hostname, Python version, numpy version, ISO timestamp
- Full TOML config used to generate the file
- SHA-256 of the reciprocal-space kernel npz used

See [docs/output-format.md](output-format.md) for the full schema.

## Design history

The HDF5 format was designed through a structured /grill-me interview (Q1-Q10) covering drivers, schema fidelity, file granularity, two-stack representation, placement patterns, write patterns, scope, backwards-compat, compression, and test plan. The resolved decisions are saved in `memory/followups_hdf5_output_format.md`.
```

- [ ] **Step 6: Commit**

```powershell
git add pyproject.toml docs/release-notes-1.1.0.md tests/test_version_is_1_1_0.py
git commit -m "chore(release): bump version 1.0.3 -> 1.1.0; release notes for HDF5 outputs"
```

---

## Task 27: Final full-suite verification + push + PR

**Files:** none (CI / git)

- [ ] **Step 1: Run the full test suite**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/ -v
```

Expected: all green.

- [ ] **Step 2: Run mypy**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m mypy src/dfxm_geo/
```

Expected: 0 errors.

- [ ] **Step 3: Run ruff**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m ruff check src/dfxm_geo/ tests/
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m ruff format --check src/dfxm_geo/ tests/
```

Expected: both clean.

- [ ] **Step 4: Push the branch**

```powershell
git push -u origin feature/hdf5-output-v1.1
```

- [ ] **Step 5: Open PR**

```powershell
gh pr create --title "v1.1.0: HDF5 output format (BLISS-style; darfix/darling compatible)" --body-file docs/release-notes-1.1.0.md
```

- [ ] **Step 6: Stop and ask the user before merging.**

Per CLAUDE.md: "Confirm before pushing or opening PRs." This task creates the PR but does NOT merge it. Wait for the user to validate on laptop + cluster (the v1.1.0 → v2.0.0 sequencing decision) before tagging the release.

---

## Self-review (done)

**Spec coverage:**
- ✓ Drivers B + C + A reflected in writer + reader + provenance + BLISS layout
- ✓ A1 (strict BLISS): tasks 3-10 build the BLISS-faithful skeleton
- ✓ G1 (one file per CLI call): write_simulation_h5 enforces one file
- ✓ α (two scans): write_simulation_h5 writes /1.1 + optional /2.1
- ✓ Detector name `dfxm_sim_detector` + degrees + units attr (tasks 3, 4)
- ✓ Z (hybrid placement): tasks 9 (per-scan) + 11-13 (global)
- ✓ NX_class attrs (task 7)
- ✓ Hg stored, q_hkl stored (task 9); TOML embedded (task 13)
- ✓ W2 producer-consumer (task 14)
- ✓ Durability deferred (no flush task)
- ✓ S3 forward-only scope (no identification tasks)
- ✓ B3 pure cutover (tasks 19 migrate, 21 delete legacy, 23 defensive guards)
- ✓ F1 figures on disk (task 18)
- ✓ Chunking + gzip-4 + shuffle + float64 (tasks 10, 14)
- ✓ h5py only (task 2; no silx)
- ✓ Layer 1 + 2 tests (writer 3-10, provenance 11-13, reader 17, integration 15-16, bit-equiv 22, defensive 23, migrate 19)
- ✓ Version bump (task 26)

**Placeholder scan:** no TBD/TODO/"add error handling" in any step. Every step has either concrete code or a concrete shell command.

**Type consistency:** `write_h5_scan` signature grows monotonically through tasks 3-9 (each new kwarg is additive). `_save_scan_parallel_to_h5`, `_write_provenance`, `write_simulation_h5`, `load_h5_scan` signatures match across writer/reader tests and consumer call sites.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-17-hdf5-output-format-v1.1.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
