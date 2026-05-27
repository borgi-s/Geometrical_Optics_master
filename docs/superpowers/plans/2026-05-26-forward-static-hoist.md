# Forward-Model Static/Dynamic Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoist the loop-invariant `qs = Us @ Hg @ q_hkl` matmul (≈73% of `forward()`) out of the per-frame scan loop so it is computed once per scan instead of once per frame, bit-exactly.

**Architecture:** Split `forward()` into `precompute_forward_static(Hg) -> base_qc` (the expensive Hg-only part, run once per scan) and `forward_from_static(base_qc, phi, chi, TwoDeltaTheta) -> image` (the cheap per-frame part). `forward()` stays as a thin bit-exact wrapper over the two. The scan seam `_scan_frames_args` computes `base_qc` once and ships it (read-only, shared across worker threads) in each frame tuple instead of `Hg`. Reference implementation already exists in `C:\Users\borgi\Documents\HFP_Book_Ch\direct_space\forward_model.py` (lines 146–207).

**Tech Stack:** Python 3.11/3.12, NumPy, pytest, mypy. Venv interpreter `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe`.

**Spec:** `docs/superpowers/specs/2026-05-26-forward-static-hoist-design.md`

---

## Background the engineer needs

- **Run all commands with the venv python**, not bare `python` (bare `python` is Python 2.7 on this machine):
  `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe`. Working dir is the repo root `C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master`.
- **`forward()` lives in** `src/dfxm_geo/direct_space/forward_model.py` (current body: lines 447–577). It reads module-level globals: `Us`, `q_hkl`, `theta_0`, `Resq_i`, `_analytic_eval`, `_analytic_eval`, `prob_z`, `qi1_start/qi2_start/qi3_start`, `qi1_step/qi2_step/qi3_step`, `npoints1/npoints2/npoints3`, `_flat_indices`, `NN1/NN2/NN3/Nsub`. The split functions read the same globals.
- **`base_qc` shape is `(3, N)`** where `N = NN1*NN2*NN3` (= 1,473,900 at the px510/Nsub=1 default). It is `(Us @ Hg @ q_hkl).squeeze().T`.
- **Bit-exactness is the whole point.** The split must reproduce the *exact* float operation order of today's `forward()` tail. Never reassociate (`Us @ (Hg @ q_hkl)`, einsum, etc.) — those drift ~1e-18 and break goldens. Just stop *repeating* the same op.
- **Critical: never `del base_qc`** inside `forward_from_static` — it is shared/reused across frames. (Today's `forward()` does `del qs` because `qs` is single-use; in the split it is not.)
- **Guard ordering matters.** `test_forward_model_smoke.py::test_forward_raises_when_kernel_not_loaded` calls `fm.forward(Hg=None)` with `fm.Resq_i = None` and expects `RuntimeError(match="not initialized")`. So the wrapper `forward()` must run the kernel guard *before* `precompute_forward_static` (otherwise `Us @ None` raises `TypeError` first).
- **Some tests skip without a kernel.** A bare CI checkout has no MC kernel `.npz`, so end-to-end `forward()` tests skip. This machine has one at `reciprocal_space/pkl_files/Resq_i_h-1_k1_l-1_17keV_*.npz` and an Fg cache `Fg_4_40nm_150nm_px510_sub1_remountS1.npy`, so the characterization test runs here.

## File structure

| File | Responsibility | Change |
|---|---|---|
| `src/dfxm_geo/direct_space/forward_model.py` | The split functions + wrapper | Modify (lines 447–577) |
| `tests/test_forward_static_split.py` | Characterization: wrapper ≡ split, byte-for-byte | Create |
| `src/dfxm_geo/pipeline.py` | `_scan_frames_args` precomputes `base_qc` once | Modify (lines 1123–1144) |
| `src/dfxm_geo/io/hdf5.py` | `_compute_frame` worker calls `forward_from_static`; `_FrameArgs` doc | Modify (lines 45, 144–151) |
| `tests/test_forward_static_split.py` | + test that `_scan_frames_args` carries `base_qc` and `_compute_frame` matches `forward` | Modify (same file) |
| `src/dfxm_geo/io/images.py` | legacy `save_images_parallel`/`save_image` precompute `base_qc` | Modify (lines 55–76, 79–130) |
| `tests/_perf_static_hoist.py` | one-shot before/after wall-clock measurement script | Create |

---

## Task 1: Split `forward()` into static + dynamic halves (bit-exact)

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py:447-577`
- Test: `tests/test_forward_static_split.py` (create)

- [ ] **Step 1: Write the failing characterization test**

Create `tests/test_forward_static_split.py`:

```python
"""Characterization: the static/dynamic split reproduces forward() byte-for-byte.

forward(Hg, phi, chi, 2theta) must equal
forward_from_static(precompute_forward_static(Hg), phi, chi, 2theta) for the
same inputs, with NO numerical drift. These tests assert exact equality, not
allclose -- any difference is a real bug (an accidental reassociation).
"""

from __future__ import annotations

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm


def _ensure_loaded() -> None:
    """Load the default MC kernel if present; skip the test otherwise."""
    from dfxm_geo.pipeline import _lookup_and_load_kernel

    if fm.Resq_i is None and fm._analytic_eval is None:
        try:
            _lookup_and_load_kernel((-1, 1, -1), 17.0)
        except Exception:  # noqa: BLE001 - any load failure -> skip
            pytest.skip("no MC kernel available on this checkout")
    if fm.Resq_i is None and fm._analytic_eval is None:
        pytest.skip("kernel state not initialized")


@pytest.fixture(scope="module")
def loaded_Hg() -> np.ndarray:
    _ensure_loaded()
    Hg, q_hkl = fm.Find_Hg(4.0, 1, fm.psize, fm.zl_rms)
    fm.q_hkl = q_hkl
    return Hg


@pytest.mark.parametrize(
    "phi, chi, two_dtheta",
    [
        (0.0, 0.0, 0.0),
        (1e-4, -2e-4, 0.0),
        (0.0, 0.0, 3e-4),       # exercises the TwoDeltaTheta != 0 Theta branch
        (5e-5, 1e-4, -1e-4),    # all three axes non-zero
    ],
)
def test_split_matches_forward_bitwise(
    loaded_Hg: np.ndarray, phi: float, chi: float, two_dtheta: float
) -> None:
    expected = fm.forward(loaded_Hg, phi=phi, chi=chi, TwoDeltaTheta=two_dtheta)
    base_qc = fm.precompute_forward_static(loaded_Hg)
    actual = fm.forward_from_static(
        base_qc, phi=phi, chi=chi, TwoDeltaTheta=two_dtheta
    )
    np.testing.assert_array_equal(actual, expected)


def test_precompute_is_reusable_across_frames(loaded_Hg: np.ndarray) -> None:
    """One base_qc drives many frames; reuse must not mutate it."""
    base_qc = fm.precompute_forward_static(loaded_Hg)
    snapshot = base_qc.copy()
    fm.forward_from_static(base_qc, phi=1e-4, chi=0.0)
    fm.forward_from_static(base_qc, phi=-1e-4, chi=2e-4)
    np.testing.assert_array_equal(base_qc, snapshot)


def test_forward_from_static_qi_return_shape(loaded_Hg: np.ndarray) -> None:
    base_qc = fm.precompute_forward_static(loaded_Hg)
    im, qi_field = fm.forward_from_static(base_qc, phi=0.0, chi=0.0, qi_return=True)
    assert im.shape == (fm.NN2 // fm.Nsub, fm.NN1 // fm.Nsub)
    assert qi_field.shape == (3, fm.NN1, fm.NN2, fm.NN3)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_forward_static_split.py -v`
Expected: FAIL — `AttributeError: module 'dfxm_geo.direct_space.forward_model' has no attribute 'precompute_forward_static'`.

- [ ] **Step 3: Implement the split + wrapper**

In `src/dfxm_geo/direct_space/forward_model.py`, replace the entire `forward()` function (current lines 447–577) with the three functions below. Keep the existing docstring on `forward()`.

```python
def precompute_forward_static(Hg: np.ndarray) -> np.ndarray:
    """Compute the phi/chi/2theta-independent part of the forward model.

    Returns ``base_qc = (Us @ Hg @ q_hkl).squeeze().T`` (shape ``(3, N)``),
    the single most expensive step in the forward model (~73% of a call) and
    the *only* part that depends solely on ``Hg``. Compute this once per scan
    and pass it to ``forward_from_static`` for every frame. The result is
    read-only and safe to share across worker threads (the dynamic half never
    mutates it).
    """
    qs = Us @ Hg @ q_hkl
    return qs.squeeze().T


def forward_from_static(
    base_qc: np.ndarray,
    phi: float = 0,
    chi: float = 0,
    TwoDeltaTheta: float = 0,
    qi_return: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Per-frame half of the forward model.

    Takes ``base_qc`` from ``precompute_forward_static`` and the goniometer
    angles, and produces the detector image. Reproduces the exact float
    operations of the historical monolithic ``forward()`` tail (no
    reassociation), so output is bit-identical.
    """
    if Resq_i is None and _analytic_eval is None:
        raise RuntimeError(
            "forward_model state is not initialized. Load a kernel "
            "(_lookup_and_load_kernel) or register the analytic backend "
            "(_load_analytic_resolution) before calling forward()."
        )

    if TwoDeltaTheta != 0:
        theta = theta_0 + TwoDeltaTheta
        Theta = np.array(
            [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
        )
    else:
        Theta = np.array(
            [
                [np.cos(theta_0), 0, np.sin(theta_0)],
                [0, 1, 0],
                [-np.sin(theta_0), 0, np.cos(theta_0)],
            ]
        )

    # Initialize forward model image with zeros
    im_1 = np.zeros([(NN2 // Nsub), NN1 // Nsub])

    # Define angles
    ang_arr = np.asarray(
        [[phi - TwoDeltaTheta / 2], [chi], [(TwoDeltaTheta / 2) / np.tan(theta_0)]]
    )

    # qc = base_qc + ang_arr. NOTE: do NOT `del base_qc` -- it is reused across
    # frames. `ang_arr` is single-use and freed.
    qc = base_qc + ang_arr
    del ang_arr

    # Calculate scattering vector in imaging space; `qc` is no longer needed.
    qi = Theta @ qc
    del qc

    if _analytic_eval is not None:
        prob = (_analytic_eval(qi) * prob_z).astype(np.float32)
        if not qi_return:
            del qi
        contribution = np.bincount(_flat_indices, weights=prob, minlength=im_1.size)
        del prob
        im_1 += contribution.reshape(im_1.shape)
        del contribution
        if qi_return:
            qi_field = qi.reshape(3, NN1, NN2, NN3)
            return im_1, qi_field
        return im_1

    assert Resq_i is not None

    index1 = np.floor((qi[0] - qi1_start) / qi1_step).astype(np.int16)
    index2 = np.floor((qi[1] - qi2_start) / qi2_step).astype(np.int16)
    index3 = np.floor((qi[2] - qi3_start) / qi3_step).astype(np.int16)
    if not qi_return:
        del qi

    idx = (
        (index3 >= 0)
        * (index2 >= 0)
        * (index1 >= 0)
        * (index1 < npoints1)
        * (index2 < npoints2)
        * (index3 < npoints3)
    )
    prob = Resq_i[index1[idx], index2[idx], index3[idx]] * prob_z[idx]
    del index1, index2, index3

    contribution = np.bincount(
        _flat_indices[idx],
        weights=prob.astype(np.float32),
        minlength=im_1.size,
    )
    del idx, prob
    im_1 += contribution.reshape(im_1.shape)
    del contribution
    if qi_return:
        qi_field = qi.reshape(3, NN1, NN2, NN3)
        return im_1, qi_field
    return im_1


def forward(
    Hg: np.ndarray,
    phi: float = 0,
    chi: float = 0,
    TwoDeltaTheta: float = 0,
    qi_return: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the DFXM forward-model image for the given goniometer angles.

    Thin wrapper over ``precompute_forward_static`` + ``forward_from_static``,
    preserved for one-shot callers (single images, tests). For scans, call
    ``precompute_forward_static(Hg)`` once and ``forward_from_static`` per frame
    to avoid recomputing the Hg-only ``base_qc`` every frame.

    Args:
        Hg: Displacement gradient field, shape (X, 3, 3) where X = NN1*NN2*NN3.
        phi: Radians off the Bragg condition (rotation around y_l axis).
        chi: Radians off the Bragg condition (rotation around x_l axis).
        TwoDeltaTheta: Radians off the Bragg angle (2theta shift).
        qi_return: If True, also return the scattering vector field qi.

    Returns:
        Forward-model image of shape (NN2//Nsub, NN1//Nsub). If qi_return is
        True, returns (image, qi_field) where qi_field has shape (3, NN1, NN2, NN3).
    """
    # Guard BEFORE precompute so an uninitialized call (e.g. forward(Hg=None)
    # with no kernel loaded) raises the clear RuntimeError rather than a
    # TypeError from `Us @ None`.
    if Resq_i is None and _analytic_eval is None:
        raise RuntimeError(
            "forward_model state is not initialized. Load a kernel "
            "(_lookup_and_load_kernel) or register the analytic backend "
            "(_load_analytic_resolution) before calling forward()."
        )
    base_qc = precompute_forward_static(Hg)
    return forward_from_static(
        base_qc, phi=phi, chi=chi, TwoDeltaTheta=TwoDeltaTheta, qi_return=qi_return
    )
```

- [ ] **Step 4: Run the characterization test to verify it passes**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_forward_static_split.py -v`
Expected: PASS (6 parametrized + 2 = all pass). If it SKIPS, the kernel isn't loading — confirm `reciprocal_space/pkl_files/Resq_i_h-1_k1_l-1_17keV_*.npz` exists.

- [ ] **Step 5: Run the existing forward + golden suite — no regression**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_forward_model_smoke.py tests/test_forward_scatter.py tests/test_forward_dispatch.py tests/test_forward_model_paths.py tests/test_hdf5_bit_equiv.py tests/test_dislocations_smoke.py -q`
Expected: PASS/xfail/skip profile unchanged from before the edit. In particular `test_forward_raises_when_kernel_not_loaded` PASSES (guard ordering) and the bit-equivalence golden keeps its prior status (xfail/skip).

- [ ] **Step 6: mypy clean**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add src/dfxm_geo/direct_space/forward_model.py tests/test_forward_static_split.py
git commit -m "perf(forward): split into precompute_forward_static + forward_from_static

Hoist the loop-invariant qs = Us @ Hg @ q_hkl (~73% of forward()) into a
once-per-scan static half. forward() stays a bit-exact wrapper. Ports the
HFP_Book_Ch split, keeping production's bincount scatter and qi_return guard.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Wire `base_qc` through the live scan path (`_scan_frames_args` + `_compute_frame`)

**Files:**
- Modify: `src/dfxm_geo/pipeline.py:1123-1144` (`_scan_frames_args`)
- Modify: `src/dfxm_geo/io/hdf5.py:45` (`_FrameArgs` doc), `:144-151` (`_compute_frame`)
- Test: `tests/test_forward_static_split.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_forward_static_split.py`:

```python
def test_scan_frames_args_carry_base_qc(loaded_Hg: np.ndarray) -> None:
    """_scan_frames_args precomputes base_qc once and ships it per frame."""
    import numpy as np

    from dfxm_geo.pipeline import (
        ScanConfig,
        _build_scan_frames,
        _scan_frames_args,
    )

    scan = ScanConfig.from_dict(
        {"phi": {"range": 6e-4, "steps": 2}, "chi": {"range": 2e-3, "steps": 2}}
    )
    frames = _build_scan_frames(scan)
    args_list, _positioners = _scan_frames_args(loaded_Hg, frames, scan)

    expected_base_qc = fm.precompute_forward_static(loaded_Hg)
    # Every frame tuple carries the SAME base_qc object (shared, read-only).
    first_base_qc = args_list[0][1]
    np.testing.assert_array_equal(first_base_qc, expected_base_qc)
    for args in args_list[1:]:
        assert args[1] is first_base_qc


def test_compute_frame_matches_forward(loaded_Hg: np.ndarray) -> None:
    """_compute_frame(base_qc, ...) image == forward(Hg, ...) image."""
    import numpy as np

    from dfxm_geo.io.hdf5 import _compute_frame

    base_qc = fm.precompute_forward_static(loaded_Hg)
    phi, chi, two_dtheta = 1e-4, -2e-4, 3e-4
    idx, im = _compute_frame((7, base_qc, phi, chi, two_dtheta))
    assert idx == 7
    expected = fm.forward(loaded_Hg, phi=phi, chi=chi, TwoDeltaTheta=two_dtheta)
    np.testing.assert_array_equal(im, expected)
```

> `ScanConfig.from_dict` takes nested per-axis dicts; `range` is in **radians** (project-wide convention). `_build_scan_frames` and `_scan_frames_args` are module-private in `dfxm_geo.pipeline` — import them directly.

- [ ] **Step 2: Run the test to verify it fails**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_forward_static_split.py::test_scan_frames_args_carry_base_qc tests/test_forward_static_split.py::test_compute_frame_matches_forward -v`
Expected: FAIL — `test_scan_frames_args_carry_base_qc` fails because `args_list[0][1]` is `Hg`, not `base_qc` (shapes `(N,3,3)` vs `(3,N)` → `assert_array_equal` raises).

- [ ] **Step 3a: Precompute `base_qc` once in `_scan_frames_args`**

In `src/dfxm_geo/pipeline.py`, replace the body of `_scan_frames_args` (lines 1123–1144) with:

```python
def _scan_frames_args(
    Hg: np.ndarray, frames: ScanFrames, scan: ScanConfig
) -> tuple[list[tuple[int, np.ndarray, float, float, float]], dict[str, np.ndarray | float]]:
    """Build (args_list, positioners) for one ScanSpec.

    args_list elements: (frame_idx, base_qc, phi_rad, chi_rad, two_dtheta_rad),
    where base_qc = precompute_forward_static(Hg) is computed ONCE here and
    shared (read-only) across every frame -- the per-frame worker runs the
    cheap forward_from_static(base_qc, ...) instead of recomputing the
    Hg-only qs matmul each frame.
    positioners: dict keyed by canonical axis; scanned axes -> per-frame array,
    fixed axes -> scalar.
    """
    base_qc = fm.precompute_forward_static(Hg)
    args_list: list[tuple[int, np.ndarray, float, float, float]] = []
    for k in range(frames.n_frames):
        args_list.append(
            (
                k,
                base_qc,
                float(frames.phi_pf[k]),
                float(frames.chi_pf[k]),
                float(frames.two_dtheta_pf[k]),
            )
        )
    positioners = _positioners_for_scan_frames(frames, scan)
    return args_list, positioners
```

- [ ] **Step 3b: Update `_compute_frame` to call `forward_from_static`**

In `src/dfxm_geo/io/hdf5.py`, change the `_FrameArgs` alias comment (line 45) and `_compute_frame` (lines 144–151):

```python
# (frame_idx, base_qc, phi, chi, two_dtheta). base_qc = precompute_forward_static(Hg),
# shared read-only across frames; replaces the per-frame Hg as of the static-hoist.
_FrameArgs = tuple[int, np.ndarray, float, float, float]
```

```python
def _compute_frame(args: _FrameArgs) -> tuple[int, np.ndarray]:
    """Worker function: run the per-frame forward model and return (frame_idx, image).

    args = (frame_idx, base_qc, phi, chi, two_dtheta)
    """
    frame_idx, base_qc, phi, chi, two_dtheta = args
    im = cast(
        np.ndarray,
        _fm.forward_from_static(base_qc, phi=phi, chi=chi, TwoDeltaTheta=two_dtheta),
    )
    return frame_idx, im
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_forward_static_split.py -v`
Expected: PASS (all, including the two new ones).

- [ ] **Step 5: Run the full suite — no regression**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q`
Expected: same pass/skip/xfail counts as the pre-change baseline. Pay attention to `tests/test_hdf5_bit_equiv.py` (uses `_compute_and_write_detector_file_parallel`), `tests/test_io.py`, and any `tests/test_pipeline_*.py` that drive scans. If the bit-equiv test's hand-built `args` (it appends `(k, Hg, phi, chi)` 4-tuples at `test_hdf5_bit_equiv.py:63`) now breaks, that is expected collateral — see Step 5b.

- [ ] **Step 5b: Fix the bit-equiv test's hand-built args (if needed)**

`tests/test_hdf5_bit_equiv.py:59-64` builds 4-tuples `(k, Hg, phi, chi)` and passes them to `_compute_and_write_detector_file_parallel`, which routes to `_compute_frame`. Those tuples are now `(k, base_qc, phi, chi, two_dtheta)`. Update the loop:

```python
    base_qc = fm.precompute_forward_static(Hg)
    args = []
    for chi_idx in range(2):
        for phi_idx in range(2):
            k = chi_idx * 2 + phi_idx
            args.append((k, base_qc, float(Phi[phi_idx]), float(Chi[chi_idx]), 0.0))
    _compute_and_write_detector_file_parallel(out, args, max_workers=1)
```

(This test is already `xfail(strict=False)` for kernel/Nsub reasons, so it will still xfail/skip — the edit just keeps the tuple contract consistent for the day it is un-xfailed.)

- [ ] **Step 6: mypy clean**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add src/dfxm_geo/pipeline.py src/dfxm_geo/io/hdf5.py tests/test_forward_static_split.py tests/test_hdf5_bit_equiv.py
git commit -m "perf(scan): precompute base_qc once per scan in _scan_frames_args

Frame tuples now carry base_qc (shared read-only) instead of Hg; the parallel
worker runs forward_from_static per frame. Drops the recomputed qs matmul from
every frame and removes the ~270 MiB qs allocation from each worker.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Apply the hoist to the legacy `.npy` writer (`save_images_parallel`/`save_image`)

> Rationale: `save_images_parallel` is no longer on the production pipeline path (only docstrings + `tests/test_io.py` + `tests/_gen_forward_legacy_golden.py` reference it), but keeping the two parallel writers consistent avoids a latent slow path. It builds its own args directly (not via `_scan_frames_args`), so it needs its own one-line precompute.

**Files:**
- Modify: `src/dfxm_geo/io/images.py:55-76` (`save_image`), `:79-130` (`save_images_parallel`)
- Test: `tests/test_io.py` (verify existing tests still pass; they monkeypatch the worker)

- [ ] **Step 1: Inspect the existing `save_images_parallel` tests**

Run: `grep -n "save_images_parallel\|save_image\|monkeypatch\|_fm" tests/test_io.py`
These tests (`test_save_images_parallel_uses_explicit_max_workers`, `test_save_images_parallel_falls_back_to_env_var`) monkeypatch the heavy compute and only assert the worker-count plumbing. Confirm whether they patch `save_image` or `_fm.forward`. This determines whether the signature change below ripples into them.

- [ ] **Step 2: Update `save_image` to consume `base_qc`**

In `src/dfxm_geo/io/images.py`, change `save_image` (lines 55–76):

```python
def save_image(args: tuple) -> None:
    """Render one frame via forward_from_static and save it as .npy.

    args = (base_qc, phi, chi, j, i, fpath, fn_prefix, ftype), where
    base_qc = forward_model.precompute_forward_static(Hg) is shared across
    all frames of the scan.
    """
    base_qc, phi, chi, j, i, fpath, fn_prefix, ftype = args
    im = _fm.forward_from_static(base_qc, phi=phi, chi=chi)
    fn_suffix = f"{i}".zfill(4) + "_" + f"{j}".zfill(4) + ftype
    np.save(os.path.join(fpath + fn_prefix + fn_suffix), im)
```

- [ ] **Step 3: Update `save_images_parallel` to precompute `base_qc` once**

In `src/dfxm_geo/io/images.py`, change the args-list construction in `save_images_parallel` (lines 118–122):

```python
    base_qc = _fm.precompute_forward_static(Hg)
    args_list = [
        (base_qc, Phi[j], Chi[i], j, i, fpath, fn_prefix, ftype)
        for i in range(chi_steps)
        for j in range(phi_steps)
    ]
```

(The `Hg` parameter of `save_images_parallel` stays in the signature — it is the input; only the per-frame tuple changes.)

- [ ] **Step 4: Run the io tests**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_io.py -v`
Expected: PASS. If a test patched `save_image` with a fake that unpacks the old 8-tuple by position, update the fake's unpacking to `(base_qc, phi, chi, j, i, ...)`.

- [ ] **Step 5: mypy clean + full suite**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/ ; & "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q`
Expected: mypy `Success`; pytest counts unchanged from baseline.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/io/images.py tests/test_io.py
git commit -m "perf(io): hoist base_qc in legacy save_images_parallel writer

Keep the .npy writer consistent with the HDF5 path: precompute base_qc once
and run forward_from_static per frame.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Measure before/after (acceptance criterion 4)

**Files:**
- Create: `tests/_perf_static_hoist.py` (one-shot measurement script, not a pytest test — leading underscore keeps it out of collection)

- [ ] **Step 1: Write the measurement script**

Create `tests/_perf_static_hoist.py`:

```python
"""One-shot wall-clock check that the static hoist landed. Not a pytest test.

Run: python tests/_perf_static_hoist.py
Reports per-frame cost of the OLD path (forward(Hg) per frame, recomputing qs)
vs the NEW path (precompute_forward_static once + forward_from_static per frame).
"""

import time

import numpy as np

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import _lookup_and_load_kernel

_lookup_and_load_kernel((-1, 1, -1), 17.0)
Hg, q_hkl = fm.Find_Hg(4.0, 1, fm.psize, fm.zl_rms)
fm.q_hkl = q_hkl

N = 25  # one 5x5 scan's worth of frames
angles = [(1e-4 * (k % 5), 2e-4 * (k // 5), 0.0) for k in range(N)]

# warm
fm.forward(Hg, phi=0.0)

# OLD: recompute qs every frame
t = time.perf_counter()
for phi, chi, dt in angles:
    fm.forward(Hg, phi=phi, chi=chi, TwoDeltaTheta=dt)
old = time.perf_counter() - t

# NEW: precompute once, dynamic per frame
t = time.perf_counter()
base_qc = fm.precompute_forward_static(Hg)
for phi, chi, dt in angles:
    fm.forward_from_static(base_qc, phi=phi, chi=chi, TwoDeltaTheta=dt)
new = time.perf_counter() - t

print(f"OLD  {N} frames (forward per frame):        {old:7.2f}s  ({old / N * 1000:6.1f} ms/frame)")
print(f"NEW  {N} frames (precompute + from_static): {new:7.2f}s  ({new / N * 1000:6.1f} ms/frame)")
print(f"per-scan speedup: {old / new:4.2f}x")
```

- [ ] **Step 2: Run it and record the numbers**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" tests/_perf_static_hoist.py`
Expected: NEW per-frame ≈ 350–400 ms vs OLD ≈ 1050–1100 ms; per-scan speedup ≈ 2.5–3.0×. If the speedup is < 1.5×, the hoist did not take effect — re-check Task 1/2.

- [ ] **Step 3: Commit the script with the measured result in the message**

```bash
git add tests/_perf_static_hoist.py
git commit -m "test: add static-hoist wall-clock measurement script

Measured on laptop (px510, Nsub=1, 25 frames): OLD ~<X>ms/frame ->
NEW ~<Y>ms/frame, ~<Z>x per-scan speedup. Fill <X>/<Y>/<Z> from the run.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

(Replace `<X>/<Y>/<Z>` with the actual numbers from Step 2 before committing.)

---

## Final verification (run before declaring done)

- [ ] **Full suite green, profile unchanged:**
  `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q`
  Compare pass/skip/xfail counts to the pre-change baseline — they must match (plus the new passing tests in `test_forward_static_split.py`).
- [ ] **mypy clean:** `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/` → `Success: no issues found`.
- [ ] **Characterization byte-exact:** `tests/test_forward_static_split.py` all pass (not skipped) on this machine.
- [ ] **Speedup measured:** `tests/_perf_static_hoist.py` shows ≥ ~2.5× per-scan.

## Notes / deferred (do NOT do in this plan — see spec "Out of scope")

- einsum reassociation (dropped: moot post-hoist + breaks goldens at 1e-18).
- GPU backend.
- Raising the `_auto_max_workers` cap now that `base_qc` is shared (the ~270 MiB/worker saving makes this safe, but measure the new per-worker footprint first).
- Optimizing the new per-frame hot path (`np.floor`/indexing ~121 ms, LUT gather ~106 ms).
- These are all candidates for a fresh profiling pass *after* this hoist ships.
```
