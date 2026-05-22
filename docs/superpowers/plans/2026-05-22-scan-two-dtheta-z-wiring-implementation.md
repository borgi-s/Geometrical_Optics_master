# `[scan.two_dtheta]` / `[scan.z]` Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the eager `ValueError` in `run_simulation` that today rejects `[scan.two_dtheta]` and `[scan.z]` scans, and thread both axes through forward + identification (single/multi) so the canonical 4-axis `(phi, chi, two_dtheta, z)` trajectory produces real BLISS HDF5 output.

**Architecture:** Approach A from the spec — extend in place. Introduce a `ScanFrames` dataclass + helpers that flatten the existing `ScanGrid` into the per-frame trajectory. Forward runs an outer loop over unique z values (per-z `Find_Hg` recompute, with a z-aware disk cache for wall mode) and parallelizes the inner `(phi × chi × two_dtheta)` Cartesian product. Identification single/multi yield one `ScanSpec` per `(z, plane, b, alpha)` configuration. `forward(Hg, phi, chi, TwoDeltaTheta)` and `Z_shift(offset_um)` already exist; the work is almost entirely orchestration plumbing.

**Tech Stack:** numpy, h5py, existing `dfxm_geo` machinery (`build_scan_grid`, `_compute_and_write_detector_file_parallel`, `MasterWriter`).

**Spec:** `docs/superpowers/specs/2026-05-22-scan-two-dtheta-z-wiring-design.md`.

---

## File touch list (locked in here, drives task decomposition)

| Path | Role |
|---|---|
| `src/dfxm_geo/pipeline.py` | `ScanFrames` + `_build_scan_frames` + `_build_scan_frames_at_z` + extended `_scan_frames_args` + extended `_positioners_for_scan` + new `_iterate_simulation_frames` orchestrator; lift the ValueError in `run_simulation`; thread z into `_iter_identification_single` + `_iter_identification_multi`. |
| `src/dfxm_geo/direct_space/forward_model.py` | `Find_Hg`: new `z_offset_um` kwarg + extended Fg-cache filename + internal `Z_shift(z)` plumbing. `Find_Hg_from_population`: new optional `rl` kwarg defaulting to `fm.rl`. |
| `src/dfxm_geo/io/hdf5.py` | `_FrameArgs` alias becomes a 5-tuple; `_compute_frame` unpacks `two_dtheta`; `write_simulation_h5` signature drops `phi_range/phi_steps/chi_range/chi_steps`, accepts `frames: ScanFrames`. |
| `tests/test_scan_frames.py` | **NEW.** Unit tests for `ScanFrames` + `_build_scan_frames` + `_build_scan_frames_at_z`. |
| `tests/test_find_hg_z_cache.py` | **NEW.** Unit tests for `Find_Hg` z-cache filename + load. |
| `tests/test_pipeline_scan_modes.py` | Extend: end-to-end `run_simulation` with each new axis alone + 4-axis `mosa_strain_layer`; bit-equivalence guard. |
| `tests/test_identification_scan_modes.py` | Extend: same axis combinations on `mode="single"` and `mode="multi"`. |
| `tests/test_hdf5_pipeline.py` | Update `write_simulation_h5` callers to pass `frames=` instead of `phi_range/...`. |
| `docs/output-format.md` | Document the 4-axis positioner block + scan-count scaling note. |
| `configs/variants/forward_strain_scan.toml` | **NEW (optional).** Showcase `[scan.two_dtheta]`. |
| `configs/variants/forward_z_scan.toml` | **NEW (optional).** Showcase `[scan.z]`. |

---

## Task 1: `ScanFrames` dataclass + `_build_scan_frames`

**Files:**
- Create: `tests/test_scan_frames.py`
- Modify: `src/dfxm_geo/pipeline.py` (add `ScanFrames` + `_build_scan_frames` near the existing `_frame_grid_from_scan` at ~line 957)

- [ ] **Step 1: Write the failing tests** in `tests/test_scan_frames.py`:

```python
"""Unit tests for ScanFrames + _build_scan_frames (v1.3.0-A)."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.pipeline import (
    AxisScanConfig,
    ScanConfig,
    ScanFrames,
    _build_scan_frames,
)


def test_zero_scanned_axes_yields_one_frame():
    """Single mode: all axes fixed -> n_frames = 1, all per-frame arrays length 1."""
    cfg = ScanConfig()
    frames = _build_scan_frames(cfg)
    assert isinstance(frames, ScanFrames)
    assert frames.n_frames == 1
    assert frames.phi_pf.shape == (1,)
    assert frames.chi_pf.shape == (1,)
    assert frames.two_dtheta_pf.shape == (1,)
    assert frames.z_pf.shape == (1,)
    # All values come from the axis `.value` (defaults: 0).
    assert frames.phi_pf[0] == 0.0
    assert frames.z_pf[0] == 0.0


def test_phi_only_scanned_matches_legacy_layout():
    """Phi-only rocking: phi values walk linspace, others repeat."""
    cfg = ScanConfig(phi=AxisScanConfig(range=1e-3, steps=5))
    frames = _build_scan_frames(cfg)
    assert frames.n_frames == 5
    np.testing.assert_allclose(frames.phi_pf, np.linspace(-1e-3, 1e-3, 5))
    np.testing.assert_array_equal(frames.chi_pf, np.zeros(5))
    np.testing.assert_array_equal(frames.two_dtheta_pf, np.zeros(5))
    np.testing.assert_array_equal(frames.z_pf, np.zeros(5))


def test_phi_chi_ordering_phi_innermost():
    """Mosa: 3 phi x 2 chi = 6 frames; phi cycles inside chi."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=3),
        chi=AxisScanConfig(range=2e-3, steps=2),
    )
    frames = _build_scan_frames(cfg)
    assert frames.n_frames == 6
    phi_grid = np.linspace(-1e-3, 1e-3, 3)
    chi_grid = np.linspace(-2e-3, 2e-3, 2)
    # phi-innermost: frames 0,1,2 share chi[0]; frames 3,4,5 share chi[1].
    np.testing.assert_allclose(frames.phi_pf, np.tile(phi_grid, 2))
    np.testing.assert_allclose(
        frames.chi_pf, np.repeat(chi_grid, 3)
    )


def test_four_axes_cartesian_product():
    """All 4 axes scanned: n_frames = product; phi-innermost, z-outermost."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1.0, steps=2),
        chi=AxisScanConfig(range=2.0, steps=2),
        two_dtheta=AxisScanConfig(range=3.0, steps=2),
        z=AxisScanConfig(range=4.0, steps=2),
    )
    frames = _build_scan_frames(cfg)
    assert frames.n_frames == 16  # 2^4
    # phi-innermost means phi values toggle every frame.
    # z-outermost means the first 8 frames share z[0]=-4, the last 8 share z[1]=+4.
    np.testing.assert_allclose(frames.z_pf[:8], -4.0)
    np.testing.assert_allclose(frames.z_pf[8:], 4.0)
    # Two_dtheta cycles every 4 frames (between phi/chi inner and z outer).
    np.testing.assert_allclose(frames.two_dtheta_pf[:4], -3.0)
    np.testing.assert_allclose(frames.two_dtheta_pf[4:8], 3.0)


def test_n_frames_matches_array_length():
    """frames.n_frames is consistent with the per-frame array length."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=4),
        two_dtheta=AxisScanConfig(range=1e-3, steps=3),
    )
    frames = _build_scan_frames(cfg)
    assert frames.n_frames == 12
    assert frames.phi_pf.size == 12
    assert frames.two_dtheta_pf.size == 12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_scan_frames.py -v`
Expected: ImportError on `ScanFrames` (not yet defined).

- [ ] **Step 3: Implement `ScanFrames` + `_build_scan_frames`** in `src/dfxm_geo/pipeline.py`. Add near the existing `_frame_grid_from_scan` definition (~line 957):

```python
@dataclass(frozen=True)
class ScanFrames:
    """Per-frame trajectory for one scan, all parallel of length n_frames.

    Frame ordering: phi-innermost, chi, two_dtheta, z-outermost.
    Units: phi/chi/two_dtheta in radians; z in micrometers.
    """

    phi_pf: np.ndarray
    chi_pf: np.ndarray
    two_dtheta_pf: np.ndarray
    z_pf: np.ndarray
    n_frames: int


def _build_scan_frames(scan: ScanConfig) -> ScanFrames:
    """Flatten the 4-axis Cartesian product of `build_scan_grid` into per-frame arrays.

    Order: phi-innermost (stride 1), then chi, then two_dtheta,
    then z-outermost (largest stride). Fixed axes contribute a
    singleton sample, so they degenerate to a constant column.
    """
    from dfxm_geo.direct_space.forward_model import build_scan_grid

    grid = build_scan_grid(scan)
    phi, chi, two_dtheta, z = grid.samples
    # `np.meshgrid(..., indexing="ij")` returns arrays ordered (phi, chi, two_dtheta, z).
    # Ravel in Fortran order so the FIRST index (phi) varies fastest -- giving
    # phi-innermost, z-outermost flat layout.
    phi_g, chi_g, twodt_g, z_g = np.meshgrid(phi, chi, two_dtheta, z, indexing="ij")
    phi_pf = phi_g.ravel(order="F")
    chi_pf = chi_g.ravel(order="F")
    two_dtheta_pf = twodt_g.ravel(order="F")
    z_pf = z_g.ravel(order="F")
    return ScanFrames(
        phi_pf=phi_pf,
        chi_pf=chi_pf,
        two_dtheta_pf=two_dtheta_pf,
        z_pf=z_pf,
        n_frames=int(phi_pf.size),
    )
```

Also make sure `ScanFrames` is importable from `dfxm_geo.pipeline` (it will be — it's at module scope).

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_scan_frames.py -v`
Expected: 5 passed.

- [ ] **Step 5: mypy check**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: `Success: no issues found in 28 source files`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_scan_frames.py src/dfxm_geo/pipeline.py
git commit -m "pipeline: ScanFrames + _build_scan_frames (v1.3.0-A task 1)"
```

---

## Task 2: `_build_scan_frames_at_z` helper

**Files:**
- Modify: `tests/test_scan_frames.py`
- Modify: `src/dfxm_geo/pipeline.py`

- [ ] **Step 1: Add failing tests** at the end of `tests/test_scan_frames.py`:

```python
from dfxm_geo.pipeline import _build_scan_frames_at_z


def test_build_scan_frames_at_z_fixes_z_axis():
    """At a specific z, z_pf is full-length constant; other axes walk product."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=2),
        chi=AxisScanConfig(range=2e-3, steps=3),
    )
    frames = _build_scan_frames_at_z(cfg, z_value=12.5)
    assert frames.n_frames == 6  # phi x chi only
    np.testing.assert_array_equal(frames.z_pf, np.full(6, 12.5))


def test_build_scan_frames_at_z_with_two_dtheta_scanned():
    """If two_dtheta is scanned, inner trajectory is phi x chi x two_dtheta."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=2),
        two_dtheta=AxisScanConfig(range=3.0, steps=2),
    )
    frames = _build_scan_frames_at_z(cfg, z_value=0.0)
    assert frames.n_frames == 4  # 2 * 2
    np.testing.assert_array_equal(frames.z_pf, np.zeros(4))
    # phi cycles fastest: [-1e-3, +1e-3, -1e-3, +1e-3]
    np.testing.assert_allclose(frames.phi_pf, [-1e-3, 1e-3, -1e-3, 1e-3])


def test_build_scan_frames_at_z_ignores_z_scan_config():
    """z range/steps in the config are ignored; only the passed-in z_value is used."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=2),
        z=AxisScanConfig(range=10.0, steps=5),  # scanned in cfg
    )
    # Identification iterators handle z themselves; helper takes one z at a time.
    frames = _build_scan_frames_at_z(cfg, z_value=7.7)
    assert frames.n_frames == 2  # phi only; z collapsed to the passed value
    np.testing.assert_array_equal(frames.z_pf, [7.7, 7.7])
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_scan_frames.py::test_build_scan_frames_at_z_fixes_z_axis -v`
Expected: ImportError on `_build_scan_frames_at_z`.

- [ ] **Step 3: Implement `_build_scan_frames_at_z`** in `src/dfxm_geo/pipeline.py`, immediately after `_build_scan_frames`:

```python
def _build_scan_frames_at_z(scan: ScanConfig, z_value: float) -> ScanFrames:
    """Inner (phi × chi × two_dtheta) trajectory with z_pf fixed to z_value.

    Used by identification iterators that loop outer over z themselves.
    The scan's `[scan.z]` configuration is ignored; only z_value is used.
    """
    from dfxm_geo.direct_space.forward_model import build_scan_grid

    grid = build_scan_grid(scan)
    phi, chi, two_dtheta, _z_ignored = grid.samples
    phi_g, chi_g, twodt_g = np.meshgrid(phi, chi, two_dtheta, indexing="ij")
    phi_pf = phi_g.ravel(order="F")
    chi_pf = chi_g.ravel(order="F")
    two_dtheta_pf = twodt_g.ravel(order="F")
    n = int(phi_pf.size)
    return ScanFrames(
        phi_pf=phi_pf,
        chi_pf=chi_pf,
        two_dtheta_pf=two_dtheta_pf,
        z_pf=np.full(n, float(z_value), dtype=np.float64),
        n_frames=n,
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_scan_frames.py -v`
Expected: 8 passed.

- [ ] **Step 5: mypy check**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: `Success: no issues found in 28 source files`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_scan_frames.py src/dfxm_geo/pipeline.py
git commit -m "pipeline: _build_scan_frames_at_z helper (v1.3.0-A task 2)"
```

---

## Task 3: Extend `_scan_frames_args` to emit 5-tuples (and update `_FrameArgs`)

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py` (change `_FrameArgs` alias + `_compute_frame`)
- Modify: `src/dfxm_geo/pipeline.py` (rewrite `_scan_frames_args` to consume `ScanFrames`)
- Test: extend `tests/test_scan_frames.py`

- [ ] **Step 1: Add failing test for the new `_scan_frames_args` signature** at the end of `tests/test_scan_frames.py`:

```python
from dfxm_geo.pipeline import _scan_frames_args


def test_scan_frames_args_returns_5_tuples():
    """_scan_frames_args(Hg, frames) emits (idx, Hg, phi, chi, two_dtheta)."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=2),
        two_dtheta=AxisScanConfig(range=3e-4, steps=2),
    )
    frames = _build_scan_frames_at_z(cfg, z_value=0.0)
    Hg = np.zeros((10, 3, 3))
    args_list, positioners = _scan_frames_args(Hg, frames)
    assert len(args_list) == 4
    for tup in args_list:
        assert len(tup) == 5  # idx, Hg, phi, chi, two_dtheta
    indices = [tup[0] for tup in args_list]
    assert indices == [0, 1, 2, 3]
    # Per-frame phi/chi/two_dtheta come straight from frames
    np.testing.assert_allclose([tup[2] for tup in args_list], frames.phi_pf)
    np.testing.assert_allclose([tup[4] for tup in args_list], frames.two_dtheta_pf)


def test_scan_frames_args_positioners_contain_all_four_axes():
    """positioners dict has phi/chi/two_dtheta/z (per-frame arrays or scalars)."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=3),
        two_dtheta=AxisScanConfig(range=3e-4, steps=2),
    )
    frames = _build_scan_frames_at_z(cfg, z_value=5.0)
    Hg = np.zeros((10, 3, 3))
    _, positioners = _scan_frames_args(Hg, frames)
    assert set(positioners.keys()) == {"phi", "chi", "two_dtheta", "z"}
    # Scanned axes are arrays
    assert isinstance(positioners["phi"], np.ndarray)
    assert isinstance(positioners["two_dtheta"], np.ndarray)
    # Fixed axes are scalars (chi default value is 0.0)
    assert positioners["chi"] == 0.0
    # z is fixed at 5.0 by _build_scan_frames_at_z
    assert positioners["z"] == 5.0
```

- [ ] **Step 2: Run failing tests**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_scan_frames.py::test_scan_frames_args_returns_5_tuples -v`
Expected: failure (the current `_scan_frames_args` returns 4-tuples + 2 separate phi/chi arrays).

- [ ] **Step 3: Update `_FrameArgs` alias in `src/dfxm_geo/io/hdf5.py`** (line 40):

```python
# Was: _FrameArgs = tuple[int, np.ndarray, float, float]
_FrameArgs = tuple[int, np.ndarray, float, float, float]
"""(frame_idx, Hg, phi_rad, chi_rad, two_dtheta_rad)"""
```

- [ ] **Step 4: Update `_compute_frame` in `src/dfxm_geo/io/hdf5.py`** (line 138):

```python
def _compute_frame(args: _FrameArgs) -> tuple[int, np.ndarray]:
    """Worker function: run forward() and return (frame_idx, image).

    args = (frame_idx, Hg, phi, chi, two_dtheta)
    """
    frame_idx, Hg, phi, chi, two_dtheta = args
    im = cast(
        np.ndarray, _fm.forward(Hg, phi=phi, chi=chi, TwoDeltaTheta=two_dtheta)
    )
    return frame_idx, im
```

- [ ] **Step 5: Replace `_scan_frames_args` in `src/dfxm_geo/pipeline.py`** (~line 979). Delete the old definition and write:

```python
def _scan_frames_args(
    Hg: np.ndarray, frames: ScanFrames
) -> tuple[list[tuple[int, np.ndarray, float, float, float]], dict[str, np.ndarray | float]]:
    """Build (args_list, positioners) for one ScanSpec.

    args_list elements: (frame_idx, Hg, phi_rad, chi_rad, two_dtheta_rad).
    positioners: dict keyed by canonical axis; scanned axes -> per-frame array,
    fixed axes -> scalar (matching `_positioners_for_scan`'s previous shape).
    """
    args_list: list[tuple[int, np.ndarray, float, float, float]] = []
    for k in range(frames.n_frames):
        args_list.append(
            (
                k,
                Hg,
                float(frames.phi_pf[k]),
                float(frames.chi_pf[k]),
                float(frames.two_dtheta_pf[k]),
            )
        )
    positioners = _positioners_for_scan_frames(frames)
    return args_list, positioners
```

- [ ] **Step 6: Stub `_positioners_for_scan_frames`** (full impl lands in Task 4). For now make it pass the new tests at least:

```python
def _positioners_for_scan_frames(frames: ScanFrames) -> dict[str, np.ndarray | float]:
    """Temporary stub: always returns per-frame arrays.

    Task 4 replaces this with the scanned/fixed-axis collapse logic.
    """
    return {
        "phi": frames.phi_pf,
        "chi": frames.chi_pf,
        "two_dtheta": frames.two_dtheta_pf,
        "z": frames.z_pf,
    }
```

Note: the Task 3 tests check `positioners["chi"] == 0.0` (scalar collapse). To make them pass under this stub, the stub returns the per-frame array `np.zeros(N)`, which is NOT == 0.0. So the second test (`test_scan_frames_args_positioners_contain_all_four_axes`) is expected to fail under the stub and pass after Task 4 lands the collapse logic.

Skip `test_scan_frames_args_positioners_contain_all_four_axes` for this commit with `@pytest.mark.xfail(reason="collapse logic lands in Task 4", strict=True)` decorator:

```python
@pytest.mark.xfail(reason="collapse logic lands in Task 4", strict=True)
def test_scan_frames_args_positioners_contain_all_four_axes():
    # ... (same body)
```

- [ ] **Step 7: Adjust the legacy `_scan_frames_args` call sites in `_iter_identification_single`/`_iter_identification_multi`/`_iter_identification_zscan`** to match the new signature.

Search current usage:
```bash
grep -n "_scan_frames_args" "src/dfxm_geo/pipeline.py"
```

For each call, the old shape was `_scan_frames_args(Hg, Phi, Chi)` returning `(args_list, phi_pf, chi_pf)`. The new shape is `_scan_frames_args(Hg, frames)` returning `(args_list, positioners)`.

At each call site, locally build a `frames` object via `_build_scan_frames_at_z(config.scan, z_value=...)`:

```python
# Old:
# args_list, phi_pf, chi_pf = _scan_frames_args(Hg, Phi, Chi)
# (then args_list went to ScanSpec.detectors, phi_pf/chi_pf went via _positioners_for_scan)

# New:
frames_at_z = _build_scan_frames_at_z(config.scan, z_value=0.0)  # z=0 if no z scan
args_list, positioners = _scan_frames_args(Hg, frames_at_z)
```

For the existing `_iter_identification_zscan` (z-scan mode), pass `z_value=z_off`. For `_iter_identification_single` / `_iter_identification_multi`, pass `z_value=0.0` for now — Tasks 11+12 add full z-awareness.

- [ ] **Step 8: Delete the now-orphan `_frame_grid_from_scan` in `pipeline.py`** (~line 957). All call sites should be migrated by step 7; if any remain, finish that migration first.

```bash
grep -n "_frame_grid_from_scan" src/dfxm_geo/
```

Expected: 0 hits after migration. Then delete the function definition.

- [ ] **Step 9: Run scan-mode-related tests**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_scan_frames.py tests/test_pipeline_scan_modes.py tests/test_identification_scan_modes.py tests/test_pipeline_identification.py tests/test_pipeline_identification_hdf5.py -v`
Expected: scan_frames tests pass (with the one xfail); identification tests still pass (still using z=0). May need targeted fixes if a call site was missed.

- [ ] **Step 10: mypy check**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: clean.

- [ ] **Step 11: Commit**

```bash
git add src/dfxm_geo/io/hdf5.py src/dfxm_geo/pipeline.py tests/test_scan_frames.py
git commit -m "pipeline+hdf5: _scan_frames_args yields 5-tuples; _FrameArgs gains two_dtheta (v1.3.0-A task 3)"
```

---

## Task 4: Extend `_positioners_for_scan_frames` to collapse fixed axes

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (replace the Task 3 stub)
- Modify: `tests/test_scan_frames.py` (remove the xfail)

- [ ] **Step 1: Drop the xfail marker** on `test_scan_frames_args_positioners_contain_all_four_axes`. The test as written must now pass.

- [ ] **Step 2: Run that single test to confirm it fails again**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_scan_frames.py::test_scan_frames_args_positioners_contain_all_four_axes -v`
Expected: FAIL (stub returns array, test wants scalar 0.0 for fixed axes).

- [ ] **Step 3: Replace `_positioners_for_scan_frames` with the real collapse logic** in `src/dfxm_geo/pipeline.py`. The function signature gains a `scan: ScanConfig` parameter so it can consult `is_scanned`:

```python
def _positioners_for_scan_frames(
    frames: ScanFrames, scan: ScanConfig
) -> dict[str, np.ndarray | float]:
    """Build the positioners dict for a ScanSpec.

    Scanned axes -> the full per-frame array (np.ndarray of length n_frames).
    Fixed axes -> the scalar axis value (float). Matches v1.2.0 convention
    for phi/chi; now extended to two_dtheta + z.
    """
    pf_arrays = {
        "phi": frames.phi_pf,
        "chi": frames.chi_pf,
        "two_dtheta": frames.two_dtheta_pf,
        "z": frames.z_pf,
    }
    out: dict[str, np.ndarray | float] = {}
    for axis_name in _CANONICAL_AXES:
        axis = getattr(scan, axis_name)
        if axis.is_scanned:
            out[axis_name] = pf_arrays[axis_name]
        else:
            out[axis_name] = float(axis.value)
    return out
```

- [ ] **Step 4: Update `_scan_frames_args` to thread `scan` through**:

```python
def _scan_frames_args(
    Hg: np.ndarray, frames: ScanFrames, scan: ScanConfig
) -> tuple[list[tuple[int, np.ndarray, float, float, float]], dict[str, np.ndarray | float]]:
    args_list: list[tuple[int, np.ndarray, float, float, float]] = []
    for k in range(frames.n_frames):
        args_list.append(
            (
                k,
                Hg,
                float(frames.phi_pf[k]),
                float(frames.chi_pf[k]),
                float(frames.two_dtheta_pf[k]),
            )
        )
    positioners = _positioners_for_scan_frames(frames, scan)
    return args_list, positioners
```

- [ ] **Step 5: Update the Task 3 test signature** in `tests/test_scan_frames.py`. The two existing tests (`test_scan_frames_args_returns_5_tuples`, `test_scan_frames_args_positioners_contain_all_four_axes`) need `scan` passed through:

```python
def test_scan_frames_args_returns_5_tuples():
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=2),
        two_dtheta=AxisScanConfig(range=3e-4, steps=2),
    )
    frames = _build_scan_frames_at_z(cfg, z_value=0.0)
    Hg = np.zeros((10, 3, 3))
    args_list, positioners = _scan_frames_args(Hg, frames, cfg)  # <-- added cfg
    # ... rest unchanged


def test_scan_frames_args_positioners_contain_all_four_axes():
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=3),
        two_dtheta=AxisScanConfig(range=3e-4, steps=2),
    )
    frames = _build_scan_frames_at_z(cfg, z_value=5.0)
    Hg = np.zeros((10, 3, 3))
    _, positioners = _scan_frames_args(Hg, frames, cfg)  # <-- added cfg
    # ... rest unchanged
```

- [ ] **Step 6: Migrate identification call sites** in `pipeline.py` to pass `scan`:

```python
# Old: args_list, positioners = _scan_frames_args(Hg, frames_at_z)
# New: args_list, positioners = _scan_frames_args(Hg, frames_at_z, config.scan)
```

- [ ] **Step 7: Drop the old `_positioners_for_scan(phi_pf, chi_pf, scan)`** (~line 1000). It's superseded by `_positioners_for_scan_frames(frames, scan)`. Verify no callers remain:

```bash
grep -n "_positioners_for_scan\b" src/dfxm_geo/
```

Expected: only the new `_positioners_for_scan_frames` (note the trailing `_frames`).

- [ ] **Step 8: Run tests**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_scan_frames.py tests/test_pipeline_scan_modes.py tests/test_identification_scan_modes.py tests/test_pipeline_identification.py tests/test_pipeline_identification_hdf5.py -v`
Expected: all pass.

- [ ] **Step 9: mypy check**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_scan_frames.py
git commit -m "pipeline: _positioners_for_scan_frames collapses fixed axes (v1.3.0-A task 4)"
```

---

## Task 5: Find_Hg z-aware disk cache

**Files:**
- Create: `tests/test_find_hg_z_cache.py`
- Modify: `src/dfxm_geo/direct_space/forward_model.py` (`Find_Hg` signature + cache filename + rl plumbing)

- [ ] **Step 1: Write failing tests** in `tests/test_find_hg_z_cache.py`:

```python
"""Unit tests for the Find_Hg z-aware disk cache (v1.3.0-A task 5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm


def _require_kernel() -> None:
    """Skip unless a bootstrapped (-1,1,-1) 17 keV kernel npz is on disk."""
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip(f"no kernel npz found in {kernel_dir}")


def test_z_offset_zero_keeps_legacy_filename(tmp_path, monkeypatch):
    """z_offset_um=0.0 must produce the same Fg cache filename as v1.2.0."""
    _require_kernel()
    monkeypatch.setattr(fm, "_REPO_ROOT", tmp_path)
    # Force fresh cache dir so we observe the file that actually lands.
    Fg_dir = tmp_path / "direct_space" / "deformation_gradient_tensors"
    fm._lookup_and_load_kernel = lambda *a, **kw: None  # no-op; Find_Hg won't lookup
    # ... (run Find_Hg at wall mode; observe the file)

    fm.Hg = None
    fm._loaded_kernel_path = None
    from dfxm_geo.pipeline import _lookup_and_load_kernel
    _lookup_and_load_kernel((-1, 1, -1), 17.0)

    fm.Find_Hg(dis=4.0, ndis=1, psize=fm.psize, zl_rms=fm.zl_rms, h=-1, k=1, l=-1)
    files = list(Fg_dir.glob("Fg_*.npy"))
    assert files, f"no Fg cache file landed in {Fg_dir}"
    name = files[0].name
    assert "_z" not in name, f"z=0 should not add z suffix; got {name!r}"


def test_z_offset_nonzero_adds_z_suffix(tmp_path, monkeypatch):
    """z_offset_um=12.5 produces a filename containing _z12500nm (round(z*1000))."""
    _require_kernel()
    monkeypatch.setattr(fm, "_REPO_ROOT", tmp_path)
    Fg_dir = tmp_path / "direct_space" / "deformation_gradient_tensors"
    from dfxm_geo.pipeline import _lookup_and_load_kernel
    fm.Hg = None
    fm._loaded_kernel_path = None
    _lookup_and_load_kernel((-1, 1, -1), 17.0)

    fm.Find_Hg(
        dis=4.0, ndis=1, psize=fm.psize, zl_rms=fm.zl_rms,
        h=-1, k=1, l=-1, z_offset_um=12.5,
    )
    files = list(Fg_dir.glob("Fg_*_z*nm.npy"))
    assert files, f"expected file with _z…nm suffix in {Fg_dir}"
    assert "_z12500nm" in files[0].name


def test_z_offset_nonzero_uses_shifted_rl(tmp_path, monkeypatch):
    """Find_Hg with z_offset_um!=0 must build rl via Z_shift, not use module rl."""
    _require_kernel()
    monkeypatch.setattr(fm, "_REPO_ROOT", tmp_path)
    from dfxm_geo.pipeline import _lookup_and_load_kernel
    fm.Hg = None
    fm._loaded_kernel_path = None
    _lookup_and_load_kernel((-1, 1, -1), 17.0)

    # Spy on Z_shift to confirm it's invoked.
    calls: list[float] = []
    real_z_shift = fm.Z_shift

    def spy(offset_um: float):
        calls.append(float(offset_um))
        return real_z_shift(offset_um)

    monkeypatch.setattr(fm, "Z_shift", spy)
    fm.Find_Hg(
        dis=4.0, ndis=1, psize=fm.psize, zl_rms=fm.zl_rms,
        h=-1, k=1, l=-1, z_offset_um=5.0,
    )
    assert calls == [5.0], f"Z_shift should be called once with 5.0; got {calls}"


def test_z_offset_zero_does_not_call_z_shift(tmp_path, monkeypatch):
    """z_offset_um=0.0 must NOT call Z_shift (keep the module-level rl)."""
    _require_kernel()
    monkeypatch.setattr(fm, "_REPO_ROOT", tmp_path)
    from dfxm_geo.pipeline import _lookup_and_load_kernel
    fm.Hg = None
    fm._loaded_kernel_path = None
    _lookup_and_load_kernel((-1, 1, -1), 17.0)

    calls: list[float] = []
    monkeypatch.setattr(
        fm, "Z_shift", lambda off: calls.append(off) or fm.rl
    )

    fm.Find_Hg(
        dis=4.0, ndis=1, psize=fm.psize, zl_rms=fm.zl_rms, h=-1, k=1, l=-1
    )
    assert calls == [], f"Z_shift should not be called for z=0; got {calls}"
```

- [ ] **Step 2: Run tests, expect failures**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_find_hg_z_cache.py -v`
Expected: TypeError (`z_offset_um` is not a kwarg yet) or wrong filename.

- [ ] **Step 3: Modify `Find_Hg`** in `src/dfxm_geo/direct_space/forward_model.py`. Signature change + cache filename + rl plumbing:

```python
def Find_Hg(
    dis: float,
    ndis: int,
    psize: float,
    zl_rms: float,
    h: int = -1,
    k: int = 1,
    l: int = -1,
    *,
    S: np.ndarray = _S_IDENTITY,
    remount_name: str = "S1",
    z_offset_um: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    # ... (docstring + remount validation unchanged) ...

    Q_norm = np.sqrt(h * h + k * k + l * l)
    q_hkl = np.asarray([h, k, l]) / Q_norm

    Fg_dir = _REPO_ROOT / "direct_space" / "deformation_gradient_tensors"
    Fg_dir.mkdir(parents=True, exist_ok=True)

    # z-aware cache filename. When z_offset_um == 0.0, identical to v1.2.0
    # filename — non-z scans hit the same cache file as today.
    z_suffix = "" if z_offset_um == 0.0 else f"_z{round(z_offset_um * 1000)}nm"
    Fg_path = str(
        Fg_dir
        / "Fg_{}_{}nm_{}nm_px{}_sub{}_remount{}{}.npy".format(
            str(dis).replace(".", ""),
            int(psize * 1e9),
            int(zl_rms * 2.35e9),
            Npixels,
            Nsub,
            remount_name,
            z_suffix,
        )
    )

    # Pick the rl grid: shifted if z_offset_um != 0, else the module-level rl.
    rl_eff = Z_shift(z_offset_um) if z_offset_um != 0.0 else rl

    Hg = load_or_generate_Hg(rl_eff, Ud, Us, Theta, dis, ndis, Fg_path, S=S)

    # ... (sidecar _vars.txt logic unchanged) ...
```

- [ ] **Step 4: Run tests**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_find_hg_z_cache.py -v`
Expected: 4 passed.

- [ ] **Step 5: Sanity-check the existing Find_Hg tests still pass**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_io.py tests/test_remount.py tests/test_dislocations.py -v`
Expected: all pass.

- [ ] **Step 6: mypy check**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add tests/test_find_hg_z_cache.py src/dfxm_geo/direct_space/forward_model.py
git commit -m "forward_model: Find_Hg gains z_offset_um kwarg + z-aware Fg cache (v1.3.0-A task 5)"
```

---

## Task 6: `Find_Hg_from_population` rl kwarg

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` (`Find_Hg_from_population`)
- Add a test in `tests/test_find_hg_z_cache.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_find_hg_z_cache.py`):

```python
def test_find_hg_from_population_accepts_rl_kwarg():
    """Find_Hg_from_population(population, rl=Z_shift(z)) uses the passed rl."""
    _require_kernel()
    from dfxm_geo.pipeline import _lookup_and_load_kernel
    fm.Hg = None
    fm._loaded_kernel_path = None
    _lookup_and_load_kernel((-1, 1, -1), 17.0)

    # Single centered dislocation
    from dfxm_geo.pipeline import CenteredCrystalConfig, CrystalConfig
    cfg = CrystalConfig(
        mode="centered",
        centered=CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 1, -2)),
    )
    pop = fm.build_dislocation_population(cfg, fov_lateral_um=20.0, rng=None)

    rl_shifted = fm.Z_shift(3.0)
    Hg_shifted, _ = fm.Find_Hg_from_population(pop, h=-1, k=1, l=-1, rl=rl_shifted)
    Hg_zero, _ = fm.Find_Hg_from_population(pop, h=-1, k=1, l=-1)
    # Different rl -> different Hg
    assert not np.allclose(Hg_shifted, Hg_zero), (
        "Hg from z-shifted rl should differ from Hg at z=0"
    )
```

- [ ] **Step 2: Run failing test**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_find_hg_z_cache.py::test_find_hg_from_population_accepts_rl_kwarg -v`
Expected: TypeError (`rl` not a kwarg).

- [ ] **Step 3: Modify `Find_Hg_from_population`** in `src/dfxm_geo/direct_space/forward_model.py` (~line 757):

```python
def Find_Hg_from_population(
    population: DislocationPopulation,
    h: int = -1,
    k: int = 1,
    l: int = -1,
    *,
    S: np.ndarray = _S_IDENTITY,
    rl: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Hg + q_hkl from an arbitrary DislocationPopulation.

    ...

    Args:
        population: ...
        h, k, l: Miller indices.
        S: 3x3 sample-remount rotation (default identity).
        rl: Detector ray grid to evaluate the strain on. Defaults to the
            module-level `fm.rl`. Pass `Z_shift(z_um)` to evaluate at a
            non-zero sample-depth offset (z-scan support).

    Returns:
        (Hg, q_hkl) ...
    """
    from dfxm_geo.crystal.dislocations import Fd_find_multi_dislocs_mixed, MixedDislocSpec

    rl_eff = rl if rl is not None else globals()["rl"]

    Q_norm = np.sqrt(h * h + k * k + l * l)
    q_hkl = np.asarray([h, k, l]) / Q_norm

    crystals = [
        MixedDislocSpec(
            Ud_mix=population.Ud[i],
            rotation_deg=0.0,
            position_lab_um=(
                float(population.positions_um[i, 0]),
                float(population.positions_um[i, 1]),
                float(population.positions_um[i, 2]),
            ),
        )
        for i in range(len(population.positions_um))
    ]

    Fg = Fd_find_multi_dislocs_mixed(rl_eff, Us, crystals, Theta, S=S)
    Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1])
    Hg -= np.identity(3)
    return Hg, q_hkl
```

- [ ] **Step 4: Run tests**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_find_hg_z_cache.py -v`
Expected: 5 passed.

- [ ] **Step 5: mypy check**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add tests/test_find_hg_z_cache.py src/dfxm_geo/direct_space/forward_model.py
git commit -m "forward_model: Find_Hg_from_population gains optional rl kwarg (v1.3.0-A task 6)"
```

---

## Task 7: `_iterate_simulation_frames` orchestrator

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (new function)
- Test: extend `tests/test_scan_frames.py`

- [ ] **Step 1: Write failing tests** (append to `tests/test_scan_frames.py`):

```python
from collections.abc import Callable

from dfxm_geo.pipeline import _iterate_simulation_frames


def test_iterate_simulation_frames_single_z_calls_provider_once():
    """When no z-scan, Hg_provider is called exactly once with z=0."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=3),
        chi=AxisScanConfig(range=2e-3, steps=2),
    )
    frames = _build_scan_frames(cfg)
    calls: list[float] = []
    fake_Hg = np.zeros((10, 3, 3))

    def Hg_provider(z: float) -> np.ndarray:
        calls.append(z)
        return fake_Hg

    out = list(_iterate_simulation_frames(frames, Hg_provider))
    assert calls == [0.0]
    assert len(out) == 6  # 3 phi x 2 chi
    # Each tuple is (idx, Hg, phi, chi, two_dtheta).
    indices = [t[0] for t in out]
    assert indices == list(range(6))
    for t in out:
        assert t[1] is fake_Hg


def test_iterate_simulation_frames_z_scan_calls_provider_per_unique_z():
    """When z is scanned, Hg_provider is called once per unique z."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=2),
        z=AxisScanConfig(range=5.0, steps=3),
    )
    frames = _build_scan_frames(cfg)
    calls: list[float] = []

    def Hg_provider(z: float) -> np.ndarray:
        calls.append(z)
        return np.full((10, 3, 3), z)  # marker

    out = list(_iterate_simulation_frames(frames, Hg_provider))
    assert len(out) == 6  # 2 phi x 3 z
    # 3 unique z values, provider called 3 times in z order.
    assert calls == sorted(set(frames.z_pf.tolist()))
    # All frames at z=z[i] share the same Hg.
    Hg0 = out[0][1]
    Hg1 = out[1][1]
    assert Hg0 is Hg1  # same z, same Hg pointer


def test_iterate_simulation_frames_memory_release_pops_prior_z():
    """Once we move past a z, the prior Hg is no longer referenced from the iterator."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=2),
        z=AxisScanConfig(range=5.0, steps=2),
    )
    frames = _build_scan_frames(cfg)
    seen: list[np.ndarray] = []

    def Hg_provider(z: float) -> np.ndarray:
        Hg = np.zeros((100, 3, 3))
        seen.append(Hg)
        return Hg

    out = []
    for tup in _iterate_simulation_frames(frames, Hg_provider):
        out.append(tup)
    assert len(out) == 4
    # The iterator should not hold references to old Hg arrays at the end.
    # We just check it doesn't leak the dict ad infinitum; this is mostly a
    # structural test.
    assert len(seen) == 2
```

- [ ] **Step 2: Run failing tests**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_scan_frames.py -v -k "iterate_simulation_frames"`
Expected: ImportError on `_iterate_simulation_frames`.

- [ ] **Step 3: Implement the orchestrator** in `src/dfxm_geo/pipeline.py` (place after `_build_scan_frames_at_z`):

```python
from collections.abc import Callable, Iterator


def _iterate_simulation_frames(
    frames: ScanFrames,
    Hg_provider: Callable[[float], np.ndarray],
) -> Iterator[tuple[int, np.ndarray, float, float, float]]:
    """Yield (idx, Hg, phi, chi, two_dtheta) per frame; Hg_provider called once per unique z.

    Memory mitigation: because z is the outermost loop in `_build_scan_frames`
    (z-outermost frame order), all frames sharing a z value are contiguous in
    `frames.z_pf`. After the last frame at a given z is yielded, the cached
    Hg is dropped — only one Hg lives in memory at any time during the walk.
    """
    z_to_Hg: dict[float, np.ndarray] = {}
    z_pf = frames.z_pf
    n = frames.n_frames
    for k in range(n):
        z = float(z_pf[k])
        if z not in z_to_Hg:
            z_to_Hg[z] = Hg_provider(z)
        yield (
            k,
            z_to_Hg[z],
            float(frames.phi_pf[k]),
            float(frames.chi_pf[k]),
            float(frames.two_dtheta_pf[k]),
        )
        # If this is the last frame at this z (either we're at the end,
        # or the next frame has a different z), free the Hg array.
        if k == n - 1 or float(z_pf[k + 1]) != z:
            del z_to_Hg[z]
```

- [ ] **Step 4: Run tests**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_scan_frames.py -v`
Expected: all pass.

- [ ] **Step 5: mypy check**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_scan_frames.py
git commit -m "pipeline: _iterate_simulation_frames orchestrator with per-z Hg cleanup (v1.3.0-A task 7)"
```

---

## Task 8: `write_simulation_h5` signature change to accept `ScanFrames`

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py` (`write_simulation_h5`)
- Modify: `src/dfxm_geo/pipeline.py` (`run_simulation` caller — Task 9 handles full body change)
- Modify: `tests/test_hdf5_pipeline.py` (callers)

- [ ] **Step 1: Update `write_simulation_h5` signature** in `src/dfxm_geo/io/hdf5.py` (~line 624):

```python
def write_simulation_h5(
    path: Path,
    *,
    Hg: np.ndarray,
    q_hkl: np.ndarray,
    frames: "ScanFrames",  # NEW: replaces phi_range/phi_steps/chi_range/chi_steps
    include_perfect_crystal: bool = True,
    sample_dis: float | None,
    sample_ndis: int,
    sample_remount: str,
    config_toml: str,
    cli: str,
    kernel_npz: Path | None = None,
    max_workers: int | None = None,
    crystal_mode: str | None = None,
    scan_mode: str | None = None,
    scanned_axes: list[str] | None = None,
    positioners: dict[str, np.ndarray | float] | None = None,  # NEW
) -> None:
```

Add import at top:

```python
from dfxm_geo.pipeline import ScanFrames  # local-only; circular-safe via TYPE_CHECKING
```

Actually use TYPE_CHECKING to avoid cycles:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dfxm_geo.pipeline import ScanFrames
```

Inside the function body, replace the linspace + ravel block (lines ~665-685 of the existing `write_simulation_h5`):

```python
# Old:
#   Phi = np.linspace(-np.deg2rad(phi_range), np.deg2rad(phi_range), phi_steps)
#   Chi = np.linspace(-np.deg2rad(chi_range), np.deg2rad(chi_range), chi_steps)
#   n = phi_steps * chi_steps
#   phi_per_frame = ...
#   chi_per_frame = ...

# New:
n = frames.n_frames
phi_per_frame = frames.phi_pf
chi_per_frame = frames.chi_pf
two_dtheta_per_frame = frames.two_dtheta_pf
z_per_frame = frames.z_pf
```

The `_build_args` closure becomes:

```python
def _build_args(Hg_in: np.ndarray) -> list[_FrameArgs]:
    return [
        (
            i,
            Hg_in,
            float(phi_per_frame[i]),
            float(chi_per_frame[i]),
            float(two_dtheta_per_frame[i]),
        )
        for i in range(n)
    ]
```

The two `master.add_scan(...)` calls already in the loop (from `cb53f76`) pull `positioners` — extend that:

```python
# inside the for-loop over scans:
if positioners is None:
    pos_dict: dict[str, np.ndarray | float] = {
        "phi": phi_per_frame,
        "chi": chi_per_frame,
    }
    if scan_mode and "two_dtheta" in (scanned_axes or []):
        pos_dict["two_dtheta"] = two_dtheta_per_frame
    else:
        pos_dict["two_dtheta"] = 0.0
    if scan_mode and "z" in (scanned_axes or []):
        pos_dict["z"] = z_per_frame
    else:
        pos_dict["z"] = 0.0
else:
    pos_dict = positioners

master.add_scan(
    scan_id=f"{scan_idx}.1",
    title=title,
    start_time=start,
    end_time=end,
    sample=sample,
    positioners=pos_dict,
    detector_links={...},
    dfxm_geo={...},
    attrs=attrs_1_1,
)
```

(Simpler: just accept `positioners` from the caller and trust it. The caller — `run_simulation` in Task 9 — passes the dict from `_positioners_for_scan_frames`.)

- [ ] **Step 2: Update the immediate caller in `pipeline.run_simulation`** (~line 693) to pass `frames=` and `positioners=` instead of `phi_range/...`:

```python
frames = _build_scan_frames(config.scan)
positioners = _positioners_for_scan_frames(frames, config.scan)

write_simulation_h5(
    h5_path,
    Hg=Hg,
    q_hkl=q_hkl,
    frames=frames,
    include_perfect_crystal=config.io.include_perfect_crystal,
    sample_dis=sample_dis,
    sample_ndis=sample_ndis,
    sample_remount=sample_remount,
    config_toml=config_toml,
    cli=" ".join(sys.argv),
    max_workers=config.io.max_workers,
    crystal_mode=config.crystal.mode,
    scan_mode=config.scan.derived_mode_name(),
    scanned_axes=list(config.scan.scanned_axes()),
    positioners=positioners,
)
```

Note: at this task's checkpoint, `run_simulation` still passes a *single* Hg (no per-z loop yet). Task 9 will fold in the orchestrator. For now we keep it working with z=0 only.

- [ ] **Step 3: Update test callers in `tests/test_hdf5_pipeline.py`** (~lines 41, 80):

The existing tests call `write_simulation_h5(path, Hg=..., phi_range=..., phi_steps=..., chi_range=..., chi_steps=..., ...)`. Replace those four kwargs with a `frames=` built from a synthetic `ScanFrames`:

```python
from dfxm_geo.pipeline import ScanFrames

phi = np.linspace(-1e-3, 1e-3, 5)
chi = np.linspace(-2e-3, 2e-3, 5)
phi_pf = np.tile(phi, 5)
chi_pf = np.repeat(chi, 5)
n = 25
frames = ScanFrames(
    phi_pf=phi_pf,
    chi_pf=chi_pf,
    two_dtheta_pf=np.zeros(n),
    z_pf=np.zeros(n),
    n_frames=n,
)

write_simulation_h5(
    path,
    Hg=Hg,
    q_hkl=q_hkl,
    frames=frames,
    sample_dis=4.0,
    sample_ndis=1,
    sample_remount="S1",
    config_toml="...",
    cli="pytest",
)
```

- [ ] **Step 4: Run tests**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_hdf5_pipeline.py tests/test_pipeline.py -v`
Expected: pass (the kernel-loaded ones still skip if no bootstrap kernel; the rest pass).

- [ ] **Step 5: mypy check**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/io/hdf5.py src/dfxm_geo/pipeline.py tests/test_hdf5_pipeline.py
git commit -m "hdf5+pipeline: write_simulation_h5 takes ScanFrames instead of phi/chi range+steps (v1.3.0-A task 8)"
```

---

## Task 9: `run_simulation` z-aware via `_iterate_simulation_frames`

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (`run_simulation` body)
- Modify: `src/dfxm_geo/io/hdf5.py` (`write_simulation_h5` accepts an Hg-per-frame source)

**This is the structural change.** Until now, `write_simulation_h5` called `_compute_and_write_detector_file_parallel` with one Hg shared by all frames. With z-scanning, Hg varies per z.

- [ ] **Step 1: Write a failing integration test in `tests/test_pipeline_scan_modes.py`**:

```python
def test_run_simulation_two_dtheta_scan_lifts_value_error(tmp_path):
    """Setting [scan.two_dtheta] no longer raises; produces a 4D scan."""
    from dfxm_geo.pipeline import (
        AxisScanConfig, CenteredCrystalConfig, CrystalConfig, IOConfig,
        PostprocessConfig, ReciprocalConfig, ScanConfig, SimulationConfig,
        run_simulation,
    )
    import dfxm_geo.direct_space.forward_model as fm
    from pathlib import Path

    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip(f"no kernel npz found in {kernel_dir}")

    cfg = SimulationConfig(
        crystal=CrystalConfig(
            mode="centered",
            centered=CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 1, -2)),
        ),
        scan=ScanConfig(
            phi=AxisScanConfig(range=6e-4, steps=2),
            chi=AxisScanConfig(range=2e-3, steps=2),
            two_dtheta=AxisScanConfig(range=1e-4, steps=2),
        ),
        io=IOConfig(include_perfect_crystal=False),
        postprocess=PostprocessConfig(enabled=False),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_simulation(cfg, tmp_path)

    import h5py
    with h5py.File(tmp_path / "dfxm_geo.h5", "r") as f:
        # 2 phi * 2 chi * 2 two_dtheta = 8 frames
        assert f["/1.1/instrument/dfxm_sim_detector/data"].shape[0] == 8
        assert f["/1.1"].attrs["scan_mode"] == "mosa_strain"
        assert sorted(f["/1.1"].attrs["scanned_axes"]) == ["chi", "phi", "two_dtheta"]
        # two_dtheta is a per-frame positioner array
        assert f["/1.1/instrument/positioners/two_dtheta"].shape == (8,)
```

- [ ] **Step 2: Run failing test**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_pipeline_scan_modes.py::test_run_simulation_two_dtheta_scan_lifts_value_error -v`
Expected: ValueError (the guard still fires).

- [ ] **Step 3: Replace the ValueError guard in `run_simulation`** (`pipeline.py` ~line 616). Delete the `unwired` ValueError block:

```python
# DELETE this entire block:
# unwired = [axis for axis in ("two_dtheta", "z") if config.scan.is_scanned(axis)]
# if unwired:
#     raise ValueError(...)
```

- [ ] **Step 4: Rewrite the body of `run_simulation`** to use `_iterate_simulation_frames`. Sketch (replace from "Wall mode preserves..." to the end of the function):

```python
# Wall mode preserves legacy Find_Hg path (Fg cache + sidecar _vars.txt).
# Centered + random_dislocations use Find_Hg_from_population.
def Hg_provider(z: float) -> tuple[np.ndarray, np.ndarray]:
    if config.crystal.mode == "wall":
        w = config.crystal.wall
        assert w is not None
        S = SAMPLE_REMOUNT_OPTIONS[w.sample_remount]
        return fm.Find_Hg(
            w.dis, w.ndis, fm.psize, fm.zl_rms,
            h=config.reciprocal.hkl[0],
            k=config.reciprocal.hkl[1],
            l=config.reciprocal.hkl[2],
            S=S,
            remount_name=w.sample_remount,
            z_offset_um=z,
        )
    else:
        rl_eff = fm.Z_shift(z) if z != 0.0 else None
        return fm.Find_Hg_from_population(
            population,
            h=config.reciprocal.hkl[0],
            k=config.reciprocal.hkl[1],
            l=config.reciprocal.hkl[2],
            rl=rl_eff,
        )

frames = _build_scan_frames(config.scan)
positioners = _positioners_for_scan_frames(frames, config.scan)

# For the HDF5 writer we need Hg + q_hkl for the master provenance.
# Use Hg at z=0 as the "base" provenance Hg (the per-frame variation
# is captured in the detector file's frame stack + positioners[z]).
Hg_base, q_hkl = Hg_provider(0.0)
fm.Hg = Hg_base
fm.q_hkl = q_hkl
```

Then pass `frames`, `positioners`, AND a callable `Hg_provider` to `write_simulation_h5`. That's a bigger change to `write_simulation_h5` — see step 5.

- [ ] **Step 5: Update `write_simulation_h5`** to accept an optional `frame_args_builder` callable that yields the per-frame args list with z-awareness baked in. New signature add:

```python
def write_simulation_h5(
    path: Path,
    *,
    Hg: np.ndarray,
    q_hkl: np.ndarray,
    frames: "ScanFrames",
    include_perfect_crystal: bool = True,
    sample_dis: float | None,
    sample_ndis: int,
    sample_remount: str,
    config_toml: str,
    cli: str,
    kernel_npz: Path | None = None,
    max_workers: int | None = None,
    crystal_mode: str | None = None,
    scan_mode: str | None = None,
    scanned_axes: list[str] | None = None,
    positioners: dict[str, np.ndarray | float] | None = None,
    Hg_provider: "Callable[[float], tuple[np.ndarray, np.ndarray]] | None" = None,  # NEW
) -> None:
```

In the body, when `Hg_provider` is None (back-compat: single-Hg simulation, no z-scan), build a flat args_list with a single Hg as before. When `Hg_provider` is provided, build the args_list via `_iterate_simulation_frames` and a per-z Hg cache. For perfect-crystal `/2.1`, use `Hg_zero = np.zeros_like(Hg)` regardless of z (the perfect crystal has no strain).

Concrete: inside the existing per-scan loop (`for scan_idx, sample_name, Hg_for_scan in scans:`), build `args_list` like:

```python
if Hg_provider is not None and scan_idx == 1:
    # Dislocation scan: use the Hg_provider for z-aware per-frame Hg.
    args_list = []
    z_to_Hg: dict[float, np.ndarray] = {}
    n = frames.n_frames
    for k in range(n):
        z = float(frames.z_pf[k])
        if z not in z_to_Hg:
            z_to_Hg[z] = Hg_provider(z)[0]  # discard q_hkl on per-z calls
        args_list.append(
            (
                k,
                z_to_Hg[z],
                float(frames.phi_pf[k]),
                float(frames.chi_pf[k]),
                float(frames.two_dtheta_pf[k]),
            )
        )
        if k == n - 1 or float(frames.z_pf[k + 1]) != z:
            del z_to_Hg[z]
else:
    # Perfect crystal /2.1, or back-compat single-Hg case.
    args_list = _build_args(Hg_for_scan)
```

- [ ] **Step 6: Update `run_simulation` to pass `Hg_provider`**:

```python
write_simulation_h5(
    h5_path,
    Hg=Hg_base,
    q_hkl=q_hkl,
    frames=frames,
    ...
    positioners=positioners,
    Hg_provider=Hg_provider,
)
```

- [ ] **Step 7: Run integration test**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_pipeline_scan_modes.py::test_run_simulation_two_dtheta_scan_lifts_value_error -v`
Expected: PASS (with kernel) or SKIP (no kernel).

- [ ] **Step 8: Run all scan-related + HDF5 tests**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_pipeline_scan_modes.py tests/test_hdf5_pipeline.py tests/test_pipeline.py tests/test_pipeline_identification.py tests/test_pipeline_identification_hdf5.py -v`
Expected: all pass (with kernel-loaded paths skipping if no kernel).

- [ ] **Step 9: mypy check**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add src/dfxm_geo/pipeline.py src/dfxm_geo/io/hdf5.py tests/test_pipeline_scan_modes.py
git commit -m "pipeline+hdf5: run_simulation lifts two_dtheta/z guard via Hg_provider + _iterate_simulation_frames (v1.3.0-A task 9)"
```

---

## Task 10: z-scan integration for forward mode

**Files:**
- Modify: `tests/test_pipeline_scan_modes.py`

- [ ] **Step 1: Write the failing test**:

```python
def test_run_simulation_z_scan_recomputes_hg_per_z(tmp_path, monkeypatch):
    """A [scan.z] config triggers one Find_Hg(_from_population) call per unique z."""
    from dfxm_geo.pipeline import (
        AxisScanConfig, CenteredCrystalConfig, CrystalConfig, IOConfig,
        PostprocessConfig, ReciprocalConfig, ScanConfig, SimulationConfig,
        run_simulation,
    )
    import dfxm_geo.direct_space.forward_model as fm
    from pathlib import Path

    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip(f"no kernel npz found in {kernel_dir}")

    # Spy on Find_Hg_from_population to count calls.
    calls: list[float] = []
    real = fm.Find_Hg_from_population

    def spy(pop, *args, rl=None, **kwargs):
        # Reverse-engineer z from rl identity: rl is None -> z=0; rl is shifted otherwise.
        if rl is None:
            calls.append(0.0)
        else:
            # We can't easily reverse the offset from rl alone; just record "non-zero".
            calls.append(float("nan"))
        return real(pop, *args, rl=rl, **kwargs)

    monkeypatch.setattr(fm, "Find_Hg_from_population", spy)

    cfg = SimulationConfig(
        crystal=CrystalConfig(
            mode="centered",
            centered=CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 1, -2)),
        ),
        scan=ScanConfig(
            phi=AxisScanConfig(range=6e-4, steps=2),
            z=AxisScanConfig(range=5.0, steps=3),
        ),
        io=IOConfig(include_perfect_crystal=False),
        postprocess=PostprocessConfig(enabled=False),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_simulation(cfg, tmp_path)
    # 3 unique z values + 1 baseline call (Hg_provider(0.0) for q_hkl provenance)
    assert len(calls) >= 3, f"expected >=3 Find_Hg_from_population calls, got {len(calls)}"

    import h5py
    with h5py.File(tmp_path / "dfxm_geo.h5", "r") as f:
        # 2 phi * 3 z = 6 frames
        assert f["/1.1/instrument/dfxm_sim_detector/data"].shape[0] == 6
        # z is a per-frame positioner
        assert f["/1.1/instrument/positioners/z"].shape == (6,)
```

- [ ] **Step 2: Run failing test, debug, fix any plumbing issues from Task 9**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_pipeline_scan_modes.py::test_run_simulation_z_scan_recomputes_hg_per_z -v`
Expected: PASS once Task 9's `Hg_provider` is wired correctly. If FAIL, the most likely issue is `rl_eff = fm.Z_shift(z) if z != 0.0 else None` not propagating into `Find_Hg_from_population`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_scan_modes.py
git commit -m "tests: forward z-scan recomputes Hg per unique z (v1.3.0-A task 10)"
```

---

## Task 11: Identification single — z-aware iterator

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (`_iter_identification_single`)
- Test: extend `tests/test_identification_scan_modes.py`

- [ ] **Step 1: Write failing test** in `tests/test_identification_scan_modes.py`:

```python
def test_single_with_z_scanned_emits_one_scanspec_per_z(tmp_path):
    """[scan.z] in identification single mode produces (n_z * n_planes * n_b * n_alpha) ScanSpecs."""
    _require_kernel()
    cfg = IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=(1, 1, 1),
            angle_start_deg=0.0,
            angle_stop_deg=0.0,
            angle_step_deg=10.0,
            b_vector_indices=[0],
            sweep_all_slip_planes=False,
            exclude_invisibility=False,
        ),
        scan=ScanConfig(
            phi=AxisScanConfig(value=1.5e-4),
            z=AxisScanConfig(range=2.0, steps=3),  # 3 z values
        ),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)
    with h5py.File(tmp_path / "dfxm_identify.h5", "r") as f:
        scan_keys = sorted(k for k in f if k != "dfxm_geo")
        # 3 z * 1 plane * 1 b * 1 alpha = 3 scans
        assert scan_keys == ["1.1", "2.1", "3.1"]
        for sid in scan_keys:
            # Each scan's positioners include a z entry (scalar — z is fixed within a scan)
            assert "z" in f[f"/{sid}/instrument/positioners"]
```

- [ ] **Step 2: Run failing test**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_identification_scan_modes.py::test_single_with_z_scanned_emits_one_scanspec_per_z -v`
Expected: FAIL — current `_iter_identification_single` emits 1 scan, ignoring z.

- [ ] **Step 3: Restructure `_iter_identification_single`** (~`pipeline.py:1105-1192`). Wrap the existing inner loop in an outer `for z in z_samples:`. Inside the inner loop, use `_build_scan_frames_at_z(config.scan, z)` and replace the `Fd_find_mixed(fm.rl, ...)` call with one that uses `rl=Z_shift(z) if z != 0 else fm.rl`.

Sketch:

```python
def _iter_identification_single(
    config: IdentificationConfig,
) -> Iterator[ScanSpec]:
    crystal_cfg = config.crystal
    planes = (
        _ALL_111_PLANES if crystal_cfg.sweep_all_slip_planes else [crystal_cfg.slip_plane_normal]
    )
    angles_deg = np.arange(
        crystal_cfg.angle_start_deg,
        crystal_cfg.angle_stop_deg + crystal_cfg.angle_step_deg * 0.5,
        crystal_cfg.angle_step_deg,
    )
    q_hkl = np.asarray(fm.q_hkl, dtype=float)
    scan_mode = config.scan.derived_mode_name()
    scanned_axes = list(config.scan.scanned_axes())

    # NEW: outer z loop. When z is fixed, z_samples == [scan.z.value].
    z_samples = build_scan_grid(config.scan).samples[3]

    for z in z_samples:
        z_float = float(z)
        rl_eff = fm.Z_shift(z_float) if z_float != 0.0 else fm.rl
        frames_at_z = _build_scan_frames_at_z(config.scan, z_float)

        for plane in planes:
            b_table = _burgers_vectors(plane)
            b_indices = (
                crystal_cfg.b_vector_indices
                if crystal_cfg.b_vector_indices is not None
                else list(range(len(b_table)))
            )
            n_arr = np.asarray(plane, dtype=float) / np.linalg.norm(plane)
            rotated = _rotated_t_vectors(n_arr, b_table[b_indices], angles_deg)
            Ud_all = _ud_matrices(n_arr, rotated)

            for j, b_idx in enumerate(b_indices):
                if crystal_cfg.exclude_invisibility and not _passes_invisibility(
                    q_hkl, b_table[b_idx], crystal_cfg.invisibility_threshold_deg
                ):
                    continue
                for i, alpha in enumerate(angles_deg):
                    Ud_mix = Ud_all[i, j]
                    Fg = Fd_find_mixed(
                        rl_eff,  # <-- was fm.rl
                        fm.Us,
                        Ud_mix=Ud_mix,
                        rotation_deg=float(alpha),
                        Theta=fm.Theta,
                    )
                    Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1]) - np.identity(3)

                    args_list, positioners = _scan_frames_args(
                        Hg, frames_at_z, config.scan
                    )

                    burgers_int = (
                        int(round(b_table[b_idx, 0] * np.sqrt(2))),
                        int(round(b_table[b_idx, 1] * np.sqrt(2))),
                        int(round(b_table[b_idx, 2] * np.sqrt(2))),
                    )
                    yield ScanSpec(
                        title=_identify_title(scan_mode, frames_at_z.n_frames, config.scan),
                        sample={
                            "name": "simulated, dislocation identification (single)",
                            "slip_plane_normal": np.asarray(plane, dtype=np.int32),
                            "burgers": np.asarray(burgers_int, dtype=np.int32),
                            "rotation_deg": float(alpha),
                        },
                        positioners=positioners,
                        dfxm_geo={
                            "Hg": Hg,
                            "q_hkl": q_hkl,
                            "theta": float(fm.theta),
                            "psize": float(fm.psize),
                            "zl_rms": float(fm.zl_rms),
                        },
                        detectors={"dfxm_sim_detector": args_list},
                        attrs={
                            "scan_mode": scan_mode,
                            "scanned_axes": scanned_axes,
                            "identify_mode": "single",
                        },
                    )
```

- [ ] **Step 4: Run the new test + existing identification tests**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_identification_scan_modes.py tests/test_pipeline_identification.py tests/test_pipeline_identification_hdf5.py -v`
Expected: new z-scan test passes; existing tests unchanged.

- [ ] **Step 5: mypy check**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_identification_scan_modes.py
git commit -m "pipeline: _iter_identification_single z-aware (v1.3.0-A task 11)"
```

---

## Task 12: Identification multi — z-aware iterator

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (`_iter_identification_multi`)
- Test: extend `tests/test_identification_scan_modes.py`

- [ ] **Step 1: Write failing test**:

```python
def test_multi_with_z_scanned_emits_one_scanspec_per_z(tmp_path):
    """[scan.z] in identification multi mode multiplies the scan count by n_z."""
    _require_kernel()
    cfg = IdentificationConfig(
        mode="multi",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=ScanConfig(
            phi=AxisScanConfig(value=1e-4),
            z=AxisScanConfig(range=2.0, steps=2),  # 2 z values
        ),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        multi=IdentificationMonteCarloConfig(n_samples=2, pos_std_um=5.0),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)
    with h5py.File(tmp_path / "dfxm_identify.h5", "r") as f:
        scan_keys = sorted(k for k in f if k != "dfxm_geo")
        # 2 z * 2 n_samples = 4 scans
        assert scan_keys == ["1.1", "2.1", "3.1", "4.1"]
```

- [ ] **Step 2: Run failing test**

Expected: FAIL (current multi iterator ignores z).

- [ ] **Step 3: Restructure `_iter_identification_multi`** with the same outer-z pattern as Task 11. Use `rl=Z_shift(z)` in the `Fd_find_multi_dislocs_mixed` call inside the Monte Carlo loop.

- [ ] **Step 4: Run tests + mypy + commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_identification_scan_modes.py
git commit -m "pipeline: _iter_identification_multi z-aware (v1.3.0-A task 12)"
```

---

## Task 13: Bit-equivalence smoke for phi/chi-only scans

**Files:**
- Modify: `tests/test_pipeline_scan_modes.py`

- [ ] **Step 1: Write the smoke test** — runs a 3x3 phi/chi-only `run_simulation` AND captures the same trajectory's per-frame phi/chi/two_dtheta/z arrays, then checks they match v1.2.0 conventions:

```python
def test_phi_chi_only_scan_remains_v120_compatible(tmp_path):
    """Phi/chi scan with both two_dtheta/z fixed at 0 produces output with the
    same flat ordering and frame count as v1.2.0. No two_dtheta/z positioners
    are present (they collapse to scalars when fixed)."""
    _require_kernel()
    from dfxm_geo.pipeline import (
        AxisScanConfig, CenteredCrystalConfig, CrystalConfig, IOConfig,
        PostprocessConfig, ReciprocalConfig, ScanConfig, SimulationConfig,
        run_simulation,
    )
    cfg = SimulationConfig(
        crystal=CrystalConfig(
            mode="centered",
            centered=CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 1, -2)),
        ),
        scan=ScanConfig(
            phi=AxisScanConfig(range=6e-4, steps=3),
            chi=AxisScanConfig(range=2e-3, steps=3),
        ),
        io=IOConfig(include_perfect_crystal=False),
        postprocess=PostprocessConfig(enabled=False),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_simulation(cfg, tmp_path)
    with h5py.File(tmp_path / "dfxm_geo.h5", "r") as f:
        assert f["/1.1/instrument/dfxm_sim_detector/data"].shape == (9, 510, 170)
        positioners = f["/1.1/instrument/positioners"]
        assert positioners["phi"].shape == (9,)
        assert positioners["chi"].shape == (9,)
        # Fixed axes collapsed to scalars
        assert positioners["two_dtheta"].shape == ()
        assert positioners["z"].shape == ()
        # Frame ordering: phi-innermost
        phi_pf = positioners["phi"][...]
        # First 3 frames cycle through phi values, then chi steps.
        unique_phi = sorted(set(phi_pf[:3].tolist()))
        assert len(unique_phi) == 3
```

- [ ] **Step 2: Run test**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_pipeline_scan_modes.py::test_phi_chi_only_scan_remains_v120_compatible -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_scan_modes.py
git commit -m "tests: phi/chi-only scans remain v1.2.0-compatible (v1.3.0-A task 13)"
```

---

## Task 14: Full 4-axis integration test (mosa_strain_layer)

**Files:**
- Modify: `tests/test_pipeline_scan_modes.py`

- [ ] **Step 1: Write the full-4-axis integration test**:

```python
def test_run_simulation_mosa_strain_layer_2x2x2x2(tmp_path):
    """All 4 axes scanned at 2 steps each = 16 frames, scan_mode='mosa_strain_layer'."""
    _require_kernel()
    from dfxm_geo.pipeline import (
        AxisScanConfig, CenteredCrystalConfig, CrystalConfig, IOConfig,
        PostprocessConfig, ReciprocalConfig, ScanConfig, SimulationConfig,
        run_simulation,
    )
    cfg = SimulationConfig(
        crystal=CrystalConfig(
            mode="centered",
            centered=CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 1, -2)),
        ),
        scan=ScanConfig(
            phi=AxisScanConfig(range=6e-4, steps=2),
            chi=AxisScanConfig(range=2e-3, steps=2),
            two_dtheta=AxisScanConfig(range=1e-4, steps=2),
            z=AxisScanConfig(range=1.0, steps=2),
        ),
        io=IOConfig(include_perfect_crystal=False),
        postprocess=PostprocessConfig(enabled=False),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_simulation(cfg, tmp_path)
    with h5py.File(tmp_path / "dfxm_geo.h5", "r") as f:
        assert f["/1.1/instrument/dfxm_sim_detector/data"].shape[0] == 16  # 2^4
        assert f["/1.1"].attrs["scan_mode"] == "mosa_strain_layer"
        assert sorted(f["/1.1"].attrs["scanned_axes"]) == [
            "chi", "phi", "two_dtheta", "z"
        ]
        positioners = f["/1.1/instrument/positioners"]
        # All 4 axes are per-frame arrays.
        for axis in ("phi", "chi", "two_dtheta", "z"):
            assert positioners[axis].shape == (16,), (
                f"{axis} should be per-frame; got shape {positioners[axis].shape}"
            )
        # z-outermost: first 8 frames share z[0], last 8 share z[1]
        z_pf = positioners["z"][...]
        assert len(set(z_pf[:8].tolist())) == 1
        assert len(set(z_pf[8:].tolist())) == 1
        assert z_pf[0] != z_pf[8]
```

- [ ] **Step 2: Run test**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_pipeline_scan_modes.py::test_run_simulation_mosa_strain_layer_2x2x2x2 -v`
Expected: PASS (takes ~30 s on a laptop — 16 frames + 2 Hg recomputes for centered mode).

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_scan_modes.py
git commit -m "tests: 4-axis mosa_strain_layer end-to-end (v1.3.0-A task 14)"
```

---

## Task 15: Docs + optional config variants

**Files:**
- Modify: `docs/output-format.md`
- Create (optional): `configs/variants/forward_strain_scan.toml`
- Create (optional): `configs/variants/forward_z_scan.toml`

- [ ] **Step 1: Update `docs/output-format.md`** to document:

1. The full canonical positioner block has 4 axes (`phi`, `chi`, `two_dtheta`, `z`); fixed axes collapse to scalars.
2. Frame ordering: phi-innermost, z-outermost.
3. Identification scan-count scaling: when `[scan.z]` is set in `mode="single"` or `mode="multi"`, the master HDF5 contains `n_z * n_planes * n_b * n_alpha` `/N.1` entries instead of `n_planes * n_b * n_alpha`.
4. Fg disk cache filename gains `_z{nm}nm` suffix for non-zero z offsets (wall mode only).

Search the doc for `phi_steps` and update sections that mention only phi/chi rocking.

- [ ] **Step 2: (Optional) ship two demo configs** — copy `configs/default.toml` and modify:

`configs/variants/forward_strain_scan.toml`: set `[scan.phi].steps = 1` (or unset `range`) so phi is fixed; set `[scan.two_dtheta].range = 1e-4; steps = 5`. Add a header comment explaining the demo.

`configs/variants/forward_z_scan.toml`: phi+chi default mosa, add `[scan.z].range = 5.0; steps = 4`.

- [ ] **Step 3: Run the full suite + mypy**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q`
Expected: all pass (modulo the same 4 kernel-load skips + 1 pre-existing xfail).

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add docs/output-format.md configs/variants/forward_strain_scan.toml configs/variants/forward_z_scan.toml
git commit -m "docs+configs: document v1.3.0 4-axis positioners + ship demo configs (v1.3.0-A task 15)"
```

---

## Wrap-up (after Task 15)

- [ ] **Step 1: Manual smoke** — generate a tiny 4-axis run, inspect in silx + h5py:

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m dfxm_geo.pipeline \
    --config configs/variants/forward_strain_scan.toml \
    --output C:\Users\borgi\tmp\v130_strain_smoke
silx view C:\Users\borgi\tmp\v130_strain_smoke\dfxm_geo.h5
```

Confirm: `/1.1/instrument/positioners/two_dtheta` shows up as a 1-D array.

- [ ] **Step 2: Push branch + open PR if working on a feature branch**, or push to main directly if working on main.

- [ ] **Step 3: Save handoff memory** at `C:\Users\borgi\.claude\projects\C--Users-borgi-Documents-GM-reworked\memory\session_handoff_<date>_v130-a.md` documenting:
  - Which tasks landed
  - Final commit SHA
  - Suite + mypy status
  - Open follow-ups (item B z-scan retirement, item C `render_per_dislocation` for z-scan)

---

## Self-review notes

**Spec coverage check:**

- Spec Section 1 (Helpers) → Tasks 1, 2, 3, 4. ✓
- Spec Section 2 (Forward path) → Tasks 7, 8, 9. ✓
- Spec Section 3 (Identification path) → Tasks 11, 12. ✓
- Spec Section 4 (Find_Hg z-cache) → Tasks 5, 6. ✓
- Spec Section 5 (HDF5 output) → Tasks 8, 9 (positioners path) + 14 (verification). ✓
- Spec Section 6 (Validation) → Task 9 (lift the ValueError). ✓
- Spec Section 7 (Frame ordering) → Task 1 (test_four_axes_cartesian_product) + 14 (z-outermost check). ✓
- Risks: Risk 1 (bit-eq) → Task 13. Risk 2 (memory) → Task 7 + the memory mitigation in Task 9. Risk 3 (scan-count explosion) → Task 15 docs. Risk 4 (cache dir growth) → Task 15 docs. Risk 5 (rl kwarg back-compat) → Task 6.
- Testing plan: covered by Tasks 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14.

**Type consistency check:**

- `_FrameArgs` becomes a 5-tuple in Task 3; consumed by `_compute_frame` (Task 3) and `_compute_and_write_detector_file_parallel` (already accepts the alias, ripples in Task 3). ✓
- `_scan_frames_args(Hg, frames, scan)` — 3 params from Task 4 onward (Task 3 introduces it with 2 params, Task 4 adds `scan`). The migration sequence is intentional. ✓
- `_iterate_simulation_frames(frames, Hg_provider)` — signature stable from Task 7. ✓
- `_positioners_for_scan_frames(frames, scan)` — Task 3 stubs without `scan`, Task 4 adds. ✓
- `write_simulation_h5(...frames=, positioners=, Hg_provider=...)` — Task 8 adds `frames`+`positioners`, Task 9 adds `Hg_provider`. Stable from Task 9. ✓

**Placeholder scan:**

- No "TBD" / "TODO" / "fill in" left.
- Optional Task 15 config variants are labelled "Optional"; everything else is concrete.
- "Sketch" appears once in Task 9 step 4 — the rest of the step shows the code; the sketch language hedges that the engineer may need to keep going from the example into the actual edit. Acceptable.
