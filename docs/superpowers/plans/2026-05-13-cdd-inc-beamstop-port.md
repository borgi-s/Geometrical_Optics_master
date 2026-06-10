# CDD_inc Beamstop / Aperture / Knife-edge Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the beamstop / square-aperture / knife-edge / `dphi_range` physics from the `origin/CDD_inc` branch's `recspace_res.py` into the cleanup branch's `dfxm_geo.reciprocal_space.resolution.py`, so that `dfxm_geo.reciprocal_space.kernel.generate_kernel()` reproduces the CDD_inc standard reciprocal-space resolution kernel and the cleanup pipeline can be validated end-to-end against `legacy/init_forward.py`.

**Architecture:** Strictly additive — keep `reciprocal_res_func`'s existing positional signature. Add six new kwargs (`beamstop`, `bs_height`, `return_qs`, `aperture`, `knife_edge`, `dphi_range`) plus an `rng` kwarg for test seedability. All defaults reproduce current behaviour. Beamstop physics lives in module-private helpers (`_bfp_x_to_alpha`, `_bfp_alpha_to_x`, `_apply_aperture`, `_apply_knife_edge`, `_apply_wire`) so the public function body stays readable. `xraylib` is optional and lazy-imported inside `_apply_wire`; the CDD_inc standard recipe uses `aperture=True` and never touches it. `kernel.generate_kernel()` defaults flip to the CDD_inc-canonical parameter set (qi ranges `5e-4/7.5e-3/7.5e-3`, theta derived from 17 keV / Al 111, `beamstop=True, aperture=True, bs_height=25e-3`) but all params are exposed as kwargs so existing call sites can override.

**Tech Stack:** Python 3.11, numpy, scipy.stats (truncnorm), pytest, mypy, ruff. Optional: xraylib (Tungsten total cross-section at 17 keV; only used by the wire-absorption mode).

**Reference source:** `origin/CDD_inc:reciprocal_space/recspace_res.py` and `origin/CDD_inc:reciprocal_space/generate_Resq_i.py` in `C:\Users\borgi\Documents\CDD_Khaled\Geometrical_Optics_master`. Fetch with `git show origin/CDD_inc:reciprocal_space/recspace_res.py` from that clone.

**Pre-flight:**
- Working tree: `C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master\` on branch `cleanup/main-modernization` at HEAD `d5a8e6c`.
- Venv python: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe` (Python 3.11.14). All commands below use it explicitly; the bash `python` is Python 2.7 on this machine.

---

### Task 1: Add `xraylib` as optional dep + mypy override

**Files:**
- Modify: `pyproject.toml`

**Why:** The CDD_inc wire-absorption mode (`beamstop=True, aperture=False, knife_edge=False`) calls `xraylib.CS_Total(74, 17)` for Tungsten cross-section. We make it optional because the CDD_inc *standard* recipe uses `aperture=True` and never reaches the wire-mode path. Users who only want the square-aperture or knife-edge modes (i.e. everyone running the default kernel) shouldn't have to install a C-extension dep.

- [ ] **Step 1: Add optional-deps group and mypy override**

In `pyproject.toml`, after the existing `[project.optional-dependencies]` block (currently has only `dev`), add a new `beamstop-wire` group:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=4",
    "pytest-benchmark>=4",
    "ruff>=0.6",
    "mypy>=1.10",
    "pre-commit>=3.7",
    "ipykernel",
    "jupyterlab",
]
beamstop-wire = [
    "xraylib>=4.1",
]
```

And extend the mypy `module` list (currently has `numba`, `tqdm.*`, `fabio`, `fabio.*`, `joblib`, `scipy`, `scipy.*`) to include `xraylib`:

```toml
[[tool.mypy.overrides]]
module = [
    "numba",
    "tqdm.*",
    "fabio",
    "fabio.*",
    "joblib",
    "scipy",
    "scipy.*",
    "xraylib",
]
ignore_missing_imports = true
```

- [ ] **Step 2: Verify pyproject still parses + mypy still clean**

Run:
```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pip install -e ".[dev]"
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/
```

Expected: install succeeds (xraylib not installed because not in `dev`); mypy reports `Success: no issues found in 21 source files`.

- [ ] **Step 3: Commit**

```powershell
git add pyproject.toml
git commit -m "build: add optional xraylib dep + mypy override for beamstop port"
```

---

### Task 2: Add `rng` kwarg to `reciprocal_res_func` for test seedability

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/resolution.py`
- Create: `tests/test_reciprocal_resolution.py`

**Why:** The current function creates `rng = np.random.default_rng()` inline (line 85) and uses `scipy.stats.truncnorm.rvs` without `random_state`. Both are unseedable. To pin current behaviour with a regression test before changing physics, we need a seedable path. Additive kwarg: defaults to `np.random.default_rng()` so existing call sites are unaffected.

- [ ] **Step 1: Write the failing test**

Create `tests/test_reciprocal_resolution.py`:

```python
"""Regression and beamstop-physics tests for the reciprocal-space resolution kernel.

The Monte Carlo Nrays is kept tiny here (1e3-1e4) so the suite runs in seconds.
Statistical sanity checks use loose tolerances; structural checks (output
shape, masking direction, kwargs plumbed through) are tight.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func


# Canonical test parameters - CDD_inc-shaped but with tiny Nrays.
NRAYS = 10_000
NPOINTS = (40, 30, 30)
QI_RANGES = (5e-4, 7.5e-3, 7.5e-3)
ZETA_V_FWHM = 5.3e-04
ZETA_H_FWHM = 0.0
NA_RMS = 7.31e-4 / 2.35
EPS_RMS = 1.41e-4 / 2.35
THETA = 0.15662  # ~17 keV / Al 111
D = 2 * np.sqrt(50e-6 * 1.6e-3)
D1 = 0.274
PHYS_APER = D / D1


def _call(rng=None, **overrides):
    """Helper: run reciprocal_res_func with canonical params + overrides."""
    kwargs = dict(
        Nrays=NRAYS,
        npoints1=NPOINTS[0],
        npoints2=NPOINTS[1],
        npoints3=NPOINTS[2],
        qi1_range=QI_RANGES[0],
        qi2_range=QI_RANGES[1],
        qi3_range=QI_RANGES[2],
        plot_figs=False,
        save_resqi=False,
        zeta_v_fwhm=ZETA_V_FWHM,
        zeta_h_fwhm=ZETA_H_FWHM,
        NA_rms=NA_RMS,
        eps_rms=EPS_RMS,
        theta=THETA,
        phys_aper=PHYS_APER,
        date="test",
        rng=rng,
    )
    kwargs.update(overrides)
    return reciprocal_res_func(**kwargs)


def test_seeded_rng_makes_output_reproducible():
    """Same seed -> identical output. Confirms rng kwarg is plumbed through."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    out1 = _call(rng=rng1, return_qs=True)
    out2 = _call(rng=rng2, return_qs=True)
    qrock1, _, _, _, _, _ = out1
    qrock2, _, _, _, _, _ = out2
    np.testing.assert_array_equal(qrock1, qrock2)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_reciprocal_resolution.py -v
```

Expected: FAIL with `TypeError: reciprocal_res_func() got an unexpected keyword argument 'rng'` (or `'return_qs'`).

- [ ] **Step 3: Add `rng` kwarg to `reciprocal_res_func`**

In `src/dfxm_geo/reciprocal_space/resolution.py`:

1. Update the signature to add `rng: np.random.Generator | None = None` after `mem_save`:

```python
def reciprocal_res_func(
    Nrays: int,
    npoints1: int,
    npoints2: int,
    npoints3: int,
    qi1_range: float,
    qi2_range: float,
    qi3_range: float,
    plot_figs: bool,
    save_resqi: bool,
    zeta_v_fwhm: float,
    zeta_h_fwhm: float,
    NA_rms: float,
    eps_rms: float,
    theta: float,
    phys_aper: float,
    date: str,
    mem_save: bool = True,
    rng: np.random.Generator | None = None,
    return_qs: bool = False,
) -> tuple[np.ndarray, ...] | None:
```

(We add `return_qs` here too — needed for the test in step 1 to read out raw arrays. The actual `return_qs` masking logic comes in Task 8; for now we just return the raw `(qrock, qroll, qpar, qrock_prime, q2th, delta_2theta)` tuple when `return_qs=True`.)

2. Right after the `print("Defining properties of rays")` line, add:

```python
    if rng is None:
        rng = np.random.default_rng()
```

3. Replace the existing `rng = np.random.default_rng()` line (currently at ~line 85) with nothing — it's been hoisted to the top.

4. Replace the `scipy.stats.truncnorm.rvs(...)` call (currently at ~line 55) so it accepts `random_state=rng`:

```python
    zeta_v = scipy.stats.truncnorm.rvs(
        (lower - mu) / zeta_v_sigma,
        (upper - mu) / zeta_v_sigma,
        loc=mu,
        scale=zeta_v_sigma,
        size=Nrays,
        random_state=rng,
    )
```

5. At the very end of the function (after the `if plot_figs == 1:` block at the end of the file), add:

```python
    if return_qs:
        return qrock, qroll, qpar, qrock_prime, q2th, delta_2theta
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_reciprocal_resolution.py::test_seeded_rng_makes_output_reproducible -v
```

Expected: PASS.

- [ ] **Step 5: Run full suite + mypy + ruff to confirm no regression**

Run:
```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m ruff check src tests
```

Expected: 92 passed (was 91 + 1 new), 5 deselected. mypy clean (21 files). ruff clean.

- [ ] **Step 6: Commit**

```powershell
git add src/dfxm_geo/reciprocal_space/resolution.py tests/test_reciprocal_resolution.py
git commit -m "refactor(reciprocal_space): add rng + return_qs kwargs for seedable testing"
```

---

### Task 3: Pin current physics baseline with a seeded regression test

**Files:**
- Modify: `tests/test_reciprocal_resolution.py`
- Create: `tests/data/golden/reciprocal_baseline.npz`

**Why:** Lock the current `reciprocal_res_func` output before adding beamstop physics. The golden is a small Nrays seeded run with the no-beamstop path. Any later change that perturbs this branch will fail loudly.

- [ ] **Step 1: Generate the golden**

Create `tests/_gen_reciprocal_baseline.py` (NOT a pytest test — just a one-off script to produce the golden):

```python
"""One-shot golden generator for tests/test_reciprocal_resolution.py.

Run once with `python -m tests._gen_reciprocal_baseline` from the repo root.
Output saved to tests/data/golden/reciprocal_baseline.npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func


def main() -> None:
    rng = np.random.default_rng(20260513)
    result = reciprocal_res_func(
        Nrays=10_000,
        npoints1=40,
        npoints2=30,
        npoints3=30,
        qi1_range=5e-4,
        qi2_range=7.5e-3,
        qi3_range=7.5e-3,
        plot_figs=False,
        save_resqi=False,
        zeta_v_fwhm=5.3e-04,
        zeta_h_fwhm=0.0,
        NA_rms=7.31e-4 / 2.35,
        eps_rms=1.41e-4 / 2.35,
        theta=0.15662,
        phys_aper=(2 * np.sqrt(50e-6 * 1.6e-3)) / 0.274,
        date="golden",
        rng=rng,
        return_qs=True,
    )
    assert result is not None
    qrock, qroll, qpar, qrock_prime, q2th, delta_2theta = result
    out = Path(__file__).parent / "data" / "golden" / "reciprocal_baseline.npz"
    np.savez(
        out,
        qrock=qrock,
        qroll=qroll,
        qpar=qpar,
        qrock_prime=qrock_prime,
        q2th=q2th,
        delta_2theta=delta_2theta,
    )
    print(f"Wrote {out} (sizes: qrock={qrock.shape})")


if __name__ == "__main__":
    main()
```

Run it:
```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m tests._gen_reciprocal_baseline
```

Expected output: `Wrote .../reciprocal_baseline.npz (sizes: qrock=(10000,))`.

- [ ] **Step 2: Add the regression test**

Append to `tests/test_reciprocal_resolution.py`:

```python
def test_no_beamstop_baseline_matches_golden(golden_dir):
    """Seeded no-beamstop run reproduces the pinned baseline to bit-equality."""
    rng = np.random.default_rng(20260513)
    result = _call(rng=rng, return_qs=True)
    assert result is not None
    qrock, qroll, qpar, qrock_prime, q2th, delta_2theta = result

    golden = np.load(golden_dir / "reciprocal_baseline.npz")
    np.testing.assert_array_equal(qrock, golden["qrock"])
    np.testing.assert_array_equal(qroll, golden["qroll"])
    np.testing.assert_array_equal(qpar, golden["qpar"])
    np.testing.assert_array_equal(qrock_prime, golden["qrock_prime"])
    np.testing.assert_array_equal(q2th, golden["q2th"])
    np.testing.assert_array_equal(delta_2theta, golden["delta_2theta"])
```

- [ ] **Step 3: Run test to verify it passes**

Run:
```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_reciprocal_resolution.py -v
```

Expected: both tests PASS.

- [ ] **Step 4: Commit**

```powershell
git add tests/test_reciprocal_resolution.py tests/_gen_reciprocal_baseline.py tests/data/golden/reciprocal_baseline.npz
git commit -m "test(reciprocal_space): pin no-beamstop baseline as regression golden"
```

---

### Task 4: Port `dphi_range` rocking-curve sweep

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/resolution.py`
- Modify: `tests/test_reciprocal_resolution.py`

**Why:** CDD_inc's `dphi_range` kwarg uniformly randomises each ray's `dphi` over `[-dphi_range/2, +dphi_range/2]` and adds it to `qrock`. Used to simulate rocking-curve scans. Smallest physics addition; do it first so the path stays simple.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reciprocal_resolution.py`:

```python
def test_dphi_range_zero_matches_baseline():
    """dphi_range=0 must reproduce the no-beamstop baseline exactly."""
    rng = np.random.default_rng(20260513)
    result = _call(rng=rng, return_qs=True, dphi_range=0.0)
    assert result is not None
    qrock, *_ = result
    rng_baseline = np.random.default_rng(20260513)
    baseline = _call(rng=rng_baseline, return_qs=True)
    assert baseline is not None
    qrock_baseline, *_ = baseline
    np.testing.assert_array_equal(qrock, qrock_baseline)


def test_dphi_range_positive_broadens_qrock():
    """Positive dphi_range adds a uniform offset, broadening qrock std."""
    rng1 = np.random.default_rng(7)
    out_narrow = _call(rng=rng1, return_qs=True, dphi_range=0.0)
    rng2 = np.random.default_rng(7)
    out_wide = _call(rng=rng2, return_qs=True, dphi_range=1e-3)
    assert out_narrow is not None and out_wide is not None
    qrock_narrow = out_narrow[0]
    qrock_wide = out_wide[0]
    # Adding U(-5e-4, 5e-4) adds variance (1e-3)^2/12 ~= 8.3e-8 to qrock.
    # Narrow std is dominated by zeta_v and delta_2theta (both ~few e-5 rad)
    # so the relative widening should be substantial.
    assert qrock_wide.std() > qrock_narrow.std() * 1.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_reciprocal_resolution.py -v
```

Expected: `test_dphi_range_zero_matches_baseline` fails with `TypeError: ... unexpected keyword argument 'dphi_range'`.

- [ ] **Step 3: Add `dphi_range` to the signature and implementation**

In `src/dfxm_geo/reciprocal_space/resolution.py`:

1. Add to the signature after `return_qs`:

```python
    dphi_range: float = 0.0,
```

2. Right after `eps = rng.normal(size=Nrays) * eps_rms` (around line 87 currently), insert:

```python
    if dphi_range > 0.0:
        dphi = rng.uniform(-dphi_range / 2, dphi_range / 2, Nrays)
    else:
        dphi = 0.0
```

3. Update the qrock assignment (line 102):

```python
    qrock = (-zeta_v / 2) - (delta_2theta / 2) + dphi
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_reciprocal_resolution.py -v
```

Expected: all 4 tests pass. The baseline test still passes because `dphi=0.0` is a no-op scalar add when `dphi_range==0`.

- [ ] **Step 5: Full suite + mypy + ruff**

Run:
```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m ruff check src tests
```

Expected: all green.

- [ ] **Step 6: Commit**

```powershell
git add src/dfxm_geo/reciprocal_space/resolution.py tests/test_reciprocal_resolution.py
git commit -m "feat(reciprocal_space): add dphi_range rocking-curve sweep kwarg"
```

---

### Task 5: Port BFP geometry helpers + square-aperture beamstop mode

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/resolution.py`
- Modify: `tests/test_reciprocal_resolution.py`

**Why:** Square-aperture is the CDD_inc *standard* mode (`aperture=True, knife_edge=False`) used by `generate_Resq_i.py`. Geometric only — no xraylib. Folds the BFP helpers in because they're useless standalone.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reciprocal_resolution.py`:

```python
def test_aperture_beamstop_drops_rays_in_corners():
    """Square aperture absorbs rays whose |alpha_x|>bs/2 OR |alpha_y|>bs/2."""
    rng = np.random.default_rng(99)
    open_count = _call(
        rng=rng, save_resqi=False, return_qs=True
    )
    rng = np.random.default_rng(99)
    masked = _call(
        rng=rng,
        return_qs=True,
        beamstop=True,
        aperture=True,
        knife_edge=False,
        bs_height=25e-3,
    )
    assert open_count is not None and masked is not None
    # Masked output should have strictly fewer rays than unmasked.
    assert masked[0].size < open_count[0].size


def test_aperture_beamstop_requires_bs_height():
    """beamstop=True, aperture=True without bs_height should raise."""
    rng = np.random.default_rng(0)
    with pytest.raises((TypeError, ValueError)):
        _call(
            rng=rng,
            beamstop=True,
            aperture=True,
            knife_edge=False,
            bs_height=None,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_reciprocal_resolution.py -v
```

Expected: both new tests fail with `TypeError: ... unexpected keyword argument 'beamstop'`.

- [ ] **Step 3: Add BFP helpers, the kwargs, and the aperture path**

In `src/dfxm_geo/reciprocal_space/resolution.py`, near the top after the imports, add the BFP transfocator constants and helpers (private — leading underscore):

```python
# ID06 transfocator parameters (Simons 2017 eq. 22). These describe the
# back-focal-plane geometry used for the optional beamstop / aperture
# modelling in `reciprocal_res_func`. Hardcoded because they are properties
# of the physical instrument, not user-tunable.
_BFP_PHI = 0.008684440640353642  # unitless
_BFP_F = 21214.67  # mm; single-lenslet focal distance
_BFP_N = 88  # number of lenslets


def _bfp_x_to_alpha(x: np.ndarray | float) -> np.ndarray | float:
    """Convert BFP position (mm) to angle (rad)."""
    return x * np.sin(_BFP_N * _BFP_PHI) / (_BFP_F * _BFP_PHI)


def _bfp_alpha_to_x(alpha: np.ndarray | float) -> np.ndarray | float:
    """Convert angle (rad) to BFP position (mm). Inverse of _bfp_x_to_alpha."""
    return alpha / np.sin(_BFP_N * _BFP_PHI) * (_BFP_F * _BFP_PHI)


def _apply_aperture(
    alpha_x: np.ndarray, alpha_y: np.ndarray, square_half_mm: float
) -> np.ndarray:
    """Square-aperture mask: True for rays that PASS the aperture."""
    x = _bfp_alpha_to_x(alpha_x)
    y = _bfp_alpha_to_x(alpha_y)
    absorbed = (np.abs(x) > square_half_mm) | (np.abs(y) > square_half_mm)
    return ~absorbed
```

Then update the signature to add the new kwargs (after `dphi_range`):

```python
    beamstop: bool = False,
    bs_height: float | None = None,
    aperture: bool = False,
    knife_edge: bool = False,
```

Then add the beamstop dispatcher between the qrock_prime / q2th transform (after the `print("Converted to image system system ")` line) and the binning step:

```python
    if beamstop:
        if bs_height is None:
            raise ValueError("bs_height must be provided when beamstop=True")
        if aperture and not knife_edge:
            keep = _apply_aperture(
                np.abs(delta_2theta / 2), np.abs(xi / 2), bs_height / 2
            )
        elif knife_edge and not aperture:
            raise NotImplementedError("knife_edge mode added in Task 6")
        elif not aperture and not knife_edge:
            raise NotImplementedError("wire mode added in Task 7")
        else:
            raise ValueError(
                "aperture and knife_edge are mutually exclusive"
            )
        qrock = qrock[keep][:Nrays]
        qroll = qroll[keep][:Nrays]
        qpar = qpar[keep][:Nrays]
        qrock_prime = qrock_prime[keep][:Nrays]
        q2th = q2th[keep][:Nrays]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_reciprocal_resolution.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

Run:
```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m ruff check src tests
```

Expected: all green.

- [ ] **Step 6: Commit**

```powershell
git add src/dfxm_geo/reciprocal_space/resolution.py tests/test_reciprocal_resolution.py
git commit -m "feat(reciprocal_space): port BFP helpers + square-aperture beamstop mode"
```

---

### Task 6: Port knife-edge beamstop mode

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/resolution.py`
- Modify: `tests/test_reciprocal_resolution.py`

**Why:** Asymmetric mask (single-sided) in the qrock direction — `x >= edge_pos` passes. CDD_inc supports this via `knife_edge=True, aperture=False`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reciprocal_resolution.py`:

```python
def test_knife_edge_beamstop_drops_rays_below_edge():
    """Knife-edge masks rays whose BFP x is below the edge position."""
    rng = np.random.default_rng(11)
    open_count = _call(rng=rng, return_qs=True)
    rng = np.random.default_rng(11)
    masked = _call(
        rng=rng,
        return_qs=True,
        beamstop=True,
        aperture=False,
        knife_edge=True,
        bs_height=25e-3,
    )
    assert open_count is not None and masked is not None
    # Knife-edge removes ~half the rays on average.
    assert masked[0].size < open_count[0].size
    # Surviving rays should be biased: BFP x of delta_2theta/2 above edge_pos.
    delta_2theta_passed = masked[5]  # index 5 = delta_2theta
    bfp_x = _import_bfp_alpha_to_x()(delta_2theta_passed / 2)
    assert (bfp_x >= 25e-3 / 2 - 1e-12).all()


def _import_bfp_alpha_to_x():
    """Helper to import the private function under test."""
    from dfxm_geo.reciprocal_space.resolution import _bfp_alpha_to_x
    return _bfp_alpha_to_x
```

- [ ] **Step 2: Run test to verify it fails**

Expected: `NotImplementedError: knife_edge mode added in Task 6`.

- [ ] **Step 3: Add the knife-edge helper and dispatcher branch**

In `src/dfxm_geo/reciprocal_space/resolution.py`, alongside `_apply_aperture`:

```python
def _apply_knife_edge(alpha: np.ndarray, edge_pos_mm: float) -> np.ndarray:
    """Knife-edge mask: True for rays whose BFP x is at or above edge_pos."""
    x = _bfp_alpha_to_x(alpha)
    return x >= edge_pos_mm
```

Replace the `elif knife_edge and not aperture:` branch:

```python
        elif knife_edge and not aperture:
            keep = _apply_knife_edge(delta_2theta / 2, bs_height / 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_reciprocal_resolution.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

Same commands as before. Expected: all green.

- [ ] **Step 6: Commit**

```powershell
git add src/dfxm_geo/reciprocal_space/resolution.py tests/test_reciprocal_resolution.py
git commit -m "feat(reciprocal_space): port knife-edge beamstop mode"
```

---

### Task 7: Port wire-absorption beamstop mode (xraylib, lazy-imported)

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/resolution.py`
- Modify: `tests/test_reciprocal_resolution.py`

**Why:** The third mode (`aperture=False, knife_edge=False`) models a cylindrical Tungsten wire's X-ray attenuation. Uses `xraylib.CS_Total(74, 17)` (Tungsten Z=74, 17 keV). Not used by the CDD_inc standard recipe but needed for completeness.

- [ ] **Step 1: Write the test (skipped if xraylib missing)**

Append to `tests/test_reciprocal_resolution.py`:

```python
def test_wire_beamstop_drops_rays_through_wire():
    """Wire mode uses xraylib for Tungsten absorption; some rays must absorb."""
    pytest.importorskip("xraylib")
    rng = np.random.default_rng(13)
    open_count = _call(rng=rng, return_qs=True)
    rng = np.random.default_rng(13)
    masked = _call(
        rng=rng,
        return_qs=True,
        beamstop=True,
        aperture=False,
        knife_edge=False,
        bs_height=25e-3,
    )
    assert open_count is not None and masked is not None
    # Some absorption must happen.
    assert masked[0].size < open_count[0].size


def test_wire_beamstop_without_xraylib_raises_clear_error(monkeypatch):
    """If xraylib is not installed, wire mode raises a clear RuntimeError."""
    import sys

    # Simulate the import failing regardless of whether xraylib is installed.
    monkeypatch.setitem(sys.modules, "xraylib", None)
    rng = np.random.default_rng(0)
    with pytest.raises((RuntimeError, ImportError), match="xraylib"):
        _call(
            rng=rng,
            beamstop=True,
            aperture=False,
            knife_edge=False,
            bs_height=25e-3,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: `test_wire_beamstop_drops_rays_through_wire` fails with `NotImplementedError: wire mode added in Task 7` (if xraylib installed); otherwise SKIPPED. `test_wire_beamstop_without_xraylib_raises_clear_error` fails with `NotImplementedError`.

- [ ] **Step 3: Add the wire-absorption helper**

In `src/dfxm_geo/reciprocal_space/resolution.py`, alongside `_apply_aperture` and `_apply_knife_edge`:

```python
# Tungsten properties for the wire-absorption beamstop model.
_TUNGSTEN_Z = 74
_TUNGSTEN_DENSITY = 19.254  # g/cm^3
_BEAM_ENERGY_KEV = 17


def _apply_wire(
    alpha: np.ndarray, half_thick_mm: float, rng: np.random.Generator
) -> np.ndarray:
    """Stochastic Tungsten-wire absorption mask using xraylib cross sections.

    Returns True for rays that PASS (either miss the wire or survive
    absorption stochastically). Raises RuntimeError if xraylib is not
    installed (it is an optional dependency, install with
    ``pip install dfxm-geo[beamstop-wire]``).
    """
    try:
        import xraylib  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "Wire-absorption beamstop mode requires xraylib. "
            "Install it with: pip install dfxm-geo[beamstop-wire]"
        ) from e

    x = _bfp_alpha_to_x(alpha)
    inside = x < half_thick_mm
    thick = np.zeros_like(alpha)
    thick[inside] = np.sqrt(half_thick_mm**2 - x[inside] ** 2)
    mu = xraylib.CS_Total(_TUNGSTEN_Z, _BEAM_ENERGY_KEV)
    survive_prob = np.exp(-mu * thick / 10 * _TUNGSTEN_DENSITY)
    draws = rng.random(alpha.size)
    return (x >= half_thick_mm) | (draws < survive_prob)
```

Replace the `elif not aperture and not knife_edge:` branch:

```python
        elif not aperture and not knife_edge:
            keep = _apply_wire(np.abs(delta_2theta / 2), bs_height / 2, rng)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pip install "xraylib>=4.1"
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_reciprocal_resolution.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

Same commands as before. Expected: all green.

- [ ] **Step 6: Commit**

```powershell
git add src/dfxm_geo/reciprocal_space/resolution.py tests/test_reciprocal_resolution.py
git commit -m "feat(reciprocal_space): port wire-absorption beamstop mode (xraylib optional)"
```

---

### Task 8: Update `kernel.generate_kernel()` defaults to CDD_inc standard

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py`
- Modify: `tests/test_reciprocal_space_imports.py` (only if signature change breaks the existing import test — likely no change needed)

**Why:** Flip the cleanup's kernel-generation script defaults to match `origin/CDD_inc:reciprocal_space/generate_Resq_i.py` so `generate_kernel()` reproduces the canonical CDD_inc kernel. Expose all parameters as kwargs (defaults preserve CDD_inc values) so callers can override.

- [ ] **Step 1: Write a defaults test**

Append to `tests/test_reciprocal_resolution.py`:

```python
def test_kernel_defaults_match_cdd_inc_generate_Resq_i_py():
    """generate_kernel defaults reproduce the CDD_inc canonical recipe.

    Reference: origin/CDD_inc:reciprocal_space/generate_Resq_i.py.
    """
    import inspect

    from dfxm_geo.reciprocal_space import kernel

    sig = inspect.signature(kernel.generate_kernel)
    defaults = {p.name: p.default for p in sig.parameters.values()}

    # Scalar params from CDD_inc generate_Resq_i.py
    assert defaults["Nrays"] == int(1e8)
    assert defaults["npoints1"] == 400
    assert defaults["npoints2"] == 200
    assert defaults["npoints3"] == 200
    assert defaults["qi1_range"] == 5e-4
    assert defaults["qi2_range"] == 0.75e-2
    assert defaults["qi3_range"] == 0.75e-2
    assert defaults["zeta_v_fwhm"] == 5.3e-04
    assert defaults["zeta_h_fwhm"] == 0
    # Beamstop / aperture switches
    assert defaults["beamstop"] is True
    assert defaults["aperture"] is True
    assert defaults["knife_edge"] is False
    assert defaults["bs_height"] == 25e-3
    # Theta derived from 17 keV / Al 111, not the hardcoded 17.953/2 deg
    expected_a = 4.0495e-10
    expected_wavelength = 1.239841984e-9 / 17
    expected_d_111 = expected_a / np.sqrt(3)
    expected_theta = np.arcsin(expected_wavelength / (2 * expected_d_111))
    assert defaults["theta"] == pytest.approx(expected_theta, rel=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL on `Nrays` default (it's already 1e8, so this part passes) but the test will fail on `defaults["beamstop"]` because `generate_kernel` doesn't take `beamstop`. Actual failing assertion will be the `KeyError` from `defaults["beamstop"]` lookup, or AssertionError comparing positional-only defaults.

- [ ] **Step 3: Rewrite `generate_kernel` with the new signature**

Replace the body of `src/dfxm_geo/reciprocal_space/kernel.py` (keep the module docstring; replace the function):

```python
"""Driver: generates a reciprocal-space resolution kernel pickle.

Configures the Monte Carlo integration parameters and calls
:func:`dfxm_geo.reciprocal_space.resolution.reciprocal_res_func`. Side
effects: writes ``pkl_files/Resq_i_<timestamp>.pkl`` and
``pkl_files/Resq_i_<timestamp>_vars.txt`` to the current working directory.

Defaults reproduce the CDD_inc canonical recipe (Al 111 reflection at
17 keV, beamstop ON via square aperture of side 25 mm at the BFP).

Run as a script::

    python -m dfxm_geo.reciprocal_space.kernel
"""

from datetime import datetime

import numpy as np

from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func


def _default_theta_al_111(keV: float = 17) -> float:
    """Bragg angle for Al 111 at the given beam energy (default 17 keV)."""
    a = 4.0495e-10  # Al lattice parameter, m
    d_111 = a / np.sqrt(3)
    wavelength = 1.239841984e-9 / keV
    return float(np.arcsin(wavelength / (2 * d_111)))


def generate_kernel(
    date: str | None = None,
    *,
    Nrays: int = int(1e8),
    npoints1: int = 400,
    npoints2: int = 200,
    npoints3: int = 200,
    qi1_range: float = 5e-4,
    qi2_range: float = 0.75e-2,
    qi3_range: float = 0.75e-2,
    zeta_v_fwhm: float = 5.3e-04,
    zeta_h_fwhm: float = 0,
    NA_rms: float = 7.31e-4 / 2.35,
    eps_rms: float = 1.41e-4 / 2.35,
    theta: float = _default_theta_al_111(17),
    D: float = 2 * np.sqrt(50e-6 * 1.6e-3),
    d1: float = 0.274,
    beamstop: bool = True,
    bs_height: float = 25e-3,
    aperture: bool = True,
    knife_edge: bool = False,
    dphi_range: float = 0.0,
) -> str:
    """Run the kernel-generation Monte Carlo and write the pickle to ``pkl_files/``.

    Defaults reproduce the CDD_inc canonical recipe (Al 111 reflection at
    17 keV; square-aperture beamstop with 25 mm side at the BFP).

    Args:
        date: Timestamp tag for the output filenames. Defaults to
            ``YYYYmmdd_HHMM`` from the current local time.
        Nrays: number of Monte Carlo rays.
        npoints1/2/3: voxel counts for the qi grid.
        qi1_range/qi2_range/qi3_range: half-widths of the qi grid.
        zeta_v_fwhm/zeta_h_fwhm: incoming-beam divergence FWHM (rad).
        NA_rms/eps_rms: objective NA / energy-bandwidth rms.
        theta: Bragg angle (rad).
        D: physical objective aperture (m).
        d1: sample-objective distance (m).
        beamstop/bs_height/aperture/knife_edge: beamstop config; see
            :func:`dfxm_geo.reciprocal_space.resolution.reciprocal_res_func`.
        dphi_range: rocking-curve sweep half-width (rad).

    Returns:
        The timestamp tag that was used.
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d_%H%M")

    phys_aper = D / d1

    reciprocal_res_func(
        Nrays,
        npoints1,
        npoints2,
        npoints3,
        qi1_range,
        qi2_range,
        qi3_range,
        plot_figs=False,
        save_resqi=True,
        zeta_v_fwhm=zeta_v_fwhm,
        zeta_h_fwhm=zeta_h_fwhm,
        NA_rms=NA_rms,
        eps_rms=eps_rms,
        theta=theta,
        phys_aper=phys_aper,
        date=date,
        beamstop=beamstop,
        bs_height=bs_height,
        aperture=aperture,
        knife_edge=knife_edge,
        dphi_range=dphi_range,
    )

    vars_used = {
        "Nrays": Nrays,
        "npoints1": npoints1,
        "npoints2": npoints2,
        "npoints3": npoints3,
        "qi1_range": qi1_range,
        "qi2_range": qi2_range,
        "qi3_range": qi3_range,
        "zeta_v_fwhm": zeta_v_fwhm,
        "zeta_h_fwhm": zeta_h_fwhm,
        "NA_rms": NA_rms,
        "eps_rms": eps_rms,
        "theta": theta,
        "D": D,
        "d1": d1,
        "phys_aper": phys_aper,
        "beamstop": beamstop,
        "bs_height": bs_height,
        "aperture": aperture,
        "knife_edge": knife_edge,
        "dphi_range": dphi_range,
    }

    with open(f"pkl_files/Resq_i_{date}_vars.txt", "w") as data:
        data.write(str(vars_used))

    return date


if __name__ == "__main__":
    generate_kernel()
```

**Note on `_default_theta_al_111(17)` as a default:** This evaluates *at function-definition time*, so the default is a concrete float. Verify it matches the CDD_inc value (~0.156619).

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_reciprocal_resolution.py::test_kernel_defaults_match_cdd_inc_generate_Resq_i_py -v
```

Expected: PASS.

- [ ] **Step 5: Verify the import-safety test still passes (no module-level side effects)**

Run:
```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_reciprocal_space_imports.py -v
```

Expected: all 3 import-safety tests pass (no monte carlo on import).

- [ ] **Step 6: Full suite + mypy + ruff**

Same commands as before. Expected: all green.

- [ ] **Step 7: Commit**

```powershell
git add src/dfxm_geo/reciprocal_space/kernel.py tests/test_reciprocal_resolution.py
git commit -m "feat(reciprocal_space): default generate_kernel to CDD_inc canonical recipe"
```

---

### Task 9: Manual end-to-end run — produce the kernel pickle and validate pipeline

**Files:**
- (no commits; manual artefact generation)
- Produces: `pkl_files/Resq_i_<timestamp>.pkl` (uncommitted; large)

**Why:** The whole point of this port. Run the Monte Carlo at the CDD_inc-canonical Nrays=1e8 to produce the kernel pickle, then run `dfxm-forward` end-to-end and compare against `legacy/init_forward.py` outputs.

**Estimated wall-clock:** 15-30 minutes for the 1e8 Monte Carlo.

- [ ] **Step 1: Generate the kernel pickle**

Run from the repo root (don't background — the user wants to watch this finish):
```powershell
cd C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m dfxm_geo.reciprocal_space.kernel
```

Expected output: stdout traces ("Defining properties of rays", "Properties of rays defined", "Found trial delta_2theta and xi", "Converted to crystal system coordinates", "Converted to image system system ", "Resq_i filled", "Resq_i saved as Resq_i_<date>.pkl"). Creates `pkl_files/Resq_i_<date>.pkl` and `pkl_files/Resq_i_<date>_vars.txt`.

- [ ] **Step 2: Drop the kernel into the location forward_model expects**

The cleanup pipeline loads the kernel via `dfxm_geo.direct_space.forward_model._load_default_kernel(pkl_path)`. Symlink or copy the new pickle:

```powershell
Copy-Item "pkl_files\Resq_i_<date>.pkl" "reciprocal_space\pkl_files\Resq_i_<date>.pkl"
```

(Adjust the destination path to whatever forward_model expects — verify by reading `src/dfxm_geo/direct_space/forward_model.py:_load_default_kernel` first.)

- [ ] **Step 3: Run the legacy script as the reference**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" legacy\init_forward.py
```

This should produce `output/<something>.png/.svg/.npy` files representing the legacy forward-model output.

- [ ] **Step 4: Run the new pipeline as the comparison**

```powershell
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m dfxm_geo.pipeline --config configs\default.toml --output-dir output\dfxm-forward
```

Or equivalently `dfxm-forward --config configs\default.toml --output-dir output\dfxm-forward`.

- [ ] **Step 5: Compare outputs and report**

Visual diff or per-pixel ratio of the two image stacks. Report back: are the legacy and new outputs equivalent (modulo Monte Carlo noise)? Any structural differences?

Document the validation outcome in `docs/reproducibility.md` or as a comment on PR #1.

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Beamstop / aperture / knife_edge / dphi_range / return_qs all have tasks
- ✅ xraylib lazy-imported as optional dep (Task 1 + Task 7)
- ✅ Existing call sites preserved via additive kwargs (Task 2-7)
- ✅ kernel.generate_kernel() defaults updated to CDD_inc canonical (Task 8)
- ✅ Regression test pins current physics before any change (Task 3)
- ✅ End-to-end validation against legacy (Task 9)

**Type consistency:**
- `_bfp_x_to_alpha` / `_bfp_alpha_to_x` use `np.ndarray | float` — same input/output type. OK.
- `_apply_aperture`, `_apply_knife_edge`, `_apply_wire` all return `np.ndarray` (boolean mask). Consistent.
- `reciprocal_res_func` return type `tuple[np.ndarray, ...] | None` — return tuple when `return_qs=True`, else None. Consistent.

**No placeholders:** All steps have concrete code. No "TODO" / "TBD" / "implement appropriate". ✅
