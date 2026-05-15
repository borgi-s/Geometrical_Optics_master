# Dislocation Identification Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the `dislocation_identification` branch into the cleanup as a first-class workflow `dfxm-identify`, supporting (a) Borgi 2025's deterministic 864-image single-disloc test set and (b) a forward-looking Monte Carlo multi-disloc mode.

**Architecture:** New entrypoint `dfxm-identify` registered in `pyproject.toml`. Calls `dfxm_geo.pipeline.cli_main_identify()`. Physics math lands in `dfxm_geo.crystal.dislocations` (two new functions); geometry helpers in new `dfxm_geo.crystal.burgers`; plotly viz in new `dfxm_geo.viz.burgers` (lazy import). Pipeline orchestration in `dfxm_geo.pipeline` (new dataclasses + `run_identification`). Plotly added as optional dep group `[identification]`.

**Tech Stack:** Python 3.11, numpy, scipy (scipy.spatial.transform.Rotation), plotly (optional), pytest, mypy, ruff.

**Spec:** `docs/superpowers/specs/2026-05-14-dislocation-identification-port-design.md`

**Canonical reference:** Borgi, Winther, Poulsen (2025), *J. Appl. Cryst.* **58**, 813–821. DOI `10.1107/S1600576725002614`. Eqs. 1/3/5/7/8.

---

## File Structure

**Create:**
- `src/dfxm_geo/crystal/burgers.py` — slip-plane lookup, t-vector rotation, Ud matrix construction
- `src/dfxm_geo/viz/burgers.py` — interactive 3D plotly viz (lazy import)
- `configs/identification_single.toml` — example config (deterministic 864-sweep)
- `configs/identification_multi.toml` — example config (Monte Carlo)
- `tests/test_dislocations_mixed.py` — physics tests for mixed-character functions
- `tests/test_burgers.py` — geometry helper tests
- `tests/test_viz_burgers.py` — plotly viz test (lazy-import guard)
- `tests/test_pipeline_identification.py` — config + smoke run + CLI tests

**Modify:**
- `src/dfxm_geo/crystal/dislocations.py` — append `MixedDislocSpec`, `Fd_find_mixed`, `Fd_find_multi_dislocs_mixed`
- `src/dfxm_geo/pipeline.py` — append `Identification*Config` dataclasses, `run_identification`, `cli_main_identify`
- `pyproject.toml` — add `[identification]` optional-deps group + `dfxm-identify` console script

---

## Task 1: Add MixedDislocSpec dataclass

**Files:**
- Modify: `src/dfxm_geo/crystal/dislocations.py` (append at end)
- Test: `tests/test_dislocations_mixed.py` (create)

- [ ] **Step 1: Create the test file with a failing test for the dataclass**

```python
# tests/test_dislocations_mixed.py
"""Unit tests for dfxm_geo.crystal.dislocations mixed-character functions.

References:
    Borgi, S., Winther, G., Poulsen, H. F. (2025). J. Appl. Cryst. 58, 813-821.
    DOI: 10.1107/S1600576725002614. Eq. 1 defines the mixed-character F_d.
"""

import numpy as np
import pytest

from dfxm_geo.constants import BURGERS_VECTOR, POISSON_RATIO
from dfxm_geo.crystal.dislocations import (
    Fd_find,
    Fd_find_mixed,
    Fd_find_multi_dislocs_mixed,
    MixedDislocSpec,
)


@pytest.fixture
def identity_rotations():
    """Identity Us / Theta and a simple Ud_mix for isolating dislocation-frame math."""
    return np.identity(3), np.identity(3), np.identity(3)


@pytest.fixture
def simple_rl_grid():
    """A small lab-frame grid avoiding the dislocation core singularity."""
    rng = np.random.default_rng(0)
    rl = rng.normal(size=(3, 50)) * 1e-6  # 50 random points in metres
    return rl


def test_mixed_disloc_spec_defaults():
    """MixedDislocSpec stores Ud_mix, rotation_deg, and a default zero position."""
    Ud = np.identity(3)
    spec = MixedDislocSpec(Ud_mix=Ud, rotation_deg=45.0)
    assert spec.rotation_deg == 45.0
    assert spec.position_lab_um == (0.0, 0.0, 0.0)
    np.testing.assert_array_equal(spec.Ud_mix, Ud)


def test_mixed_disloc_spec_with_position():
    """MixedDislocSpec accepts an explicit position offset (lab-frame µm)."""
    spec = MixedDislocSpec(Ud_mix=np.identity(3), rotation_deg=0.0, position_lab_um=(1.0, 2.0, 3.0))
    assert spec.position_lab_um == (1.0, 2.0, 3.0)


def test_mixed_disloc_spec_is_frozen():
    """MixedDislocSpec is immutable."""
    spec = MixedDislocSpec(Ud_mix=np.identity(3), rotation_deg=0.0)
    with pytest.raises((AttributeError, Exception)):
        spec.rotation_deg = 10.0  # type: ignore[misc]
```

- [ ] **Step 2: Run the test, confirm it fails**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_dislocations_mixed.py -v
```

Expected: ImportError or `cannot import name 'MixedDislocSpec'`.

- [ ] **Step 3: Implement MixedDislocSpec at the bottom of dislocations.py**

Append to `src/dfxm_geo/crystal/dislocations.py`:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MixedDislocSpec:
    """Specification for one mixed-character dislocation.

    Attributes:
        Ud_mix: Rotation matrix from dislocation to grain frame (Eq. 3 of
            Borgi 2025), shape (3, 3). Columns are (b, n, t) basis vectors.
        rotation_deg: Rotation angle (degrees) of the line direction `t`
            around the slip-plane normal `n`, starting from `t_0 = b × n`.
            See `Fd_find_mixed` docstring for the relation to the paper's α.
        position_lab_um: Lab-frame offset (µm) applied to ``rl`` so the
            dislocation core sits at the given (x, y, z). Default (0, 0, 0).
    """

    Ud_mix: np.ndarray
    rotation_deg: float
    position_lab_um: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))
```

- [ ] **Step 4: Run the tests, confirm they pass**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_dislocations_mixed.py::test_mixed_disloc_spec_defaults tests/test_dislocations_mixed.py::test_mixed_disloc_spec_with_position tests/test_dislocations_mixed.py::test_mixed_disloc_spec_is_frozen -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```
git add src/dfxm_geo/crystal/dislocations.py tests/test_dislocations_mixed.py
git commit -m "feat(crystal): add MixedDislocSpec dataclass for mixed-character dislocations"
```

---

## Task 2: Add Fd_find_mixed

**Files:**
- Modify: `src/dfxm_geo/crystal/dislocations.py` (append)
- Test: `tests/test_dislocations_mixed.py` (append)

- [ ] **Step 1: Add failing tests for Fd_find_mixed**

Append to `tests/test_dislocations_mixed.py`:

```python
def test_Fd_find_mixed_pure_edge_matches_Fd_find_ndis1(
    identity_rotations, simple_rl_grid
):
    """At rotation_deg=0 (paper's α=90°, pure edge), Fd_find_mixed must equal
    Fd_find(ndis=1) with matching Ud. This is the regression guard tying the
    new mixed math to the cleanup's existing edge formula (Eq. 1 of Borgi
    2025 collapses to the edge term for α=90°).
    """
    Us, Ud, Theta = identity_rotations
    rl = simple_rl_grid

    Fg_edge = Fd_find(rl, Ud, Us, Theta, dis=1.0, ndis=1)
    Fg_mixed = Fd_find_mixed(rl, Us, Ud_mix=Ud, rotation_deg=0.0, Theta=Theta)

    np.testing.assert_allclose(Fg_mixed, Fg_edge, atol=1e-15, rtol=1e-12)


def test_Fd_find_mixed_pure_screw_has_only_screw_terms(identity_rotations):
    """At rotation_deg=90° (paper's α=0°, pure screw), the cos(rotation) factor
    zeros the edge contributions; the only nonzero off-diagonal entries are
    the screw out-of-plane terms ∂u_dx/∂y and ∂u_dx/∂z (Eq. 1 of Borgi 2025).
    """
    Us, Ud, Theta = identity_rotations
    # Use a deterministic rd that maps directly through identity rotations.
    rd = np.array([[1e-7, 2e-7, 3e-7], [4e-7, 5e-7, 6e-7], [7e-7, 8e-7, 9e-7]])

    Fg = Fd_find_mixed(rd, Us, Ud_mix=Ud, rotation_deg=90.0, Theta=Theta)

    # Identity-frame check: Fg = I + screw_terms (no edge).
    # Diagonal: identity (1, 1, 1)
    np.testing.assert_allclose(np.diagonal(Fg, axis1=1, axis2=2), 1.0, atol=1e-15)
    # (1, 0), (2, 0), (2, 1) — pure screw matrix in Eq. 1 has zeros here.
    np.testing.assert_allclose(Fg[:, 1, 0], 0.0, atol=1e-15)
    np.testing.assert_allclose(Fg[:, 2, 0], 0.0, atol=1e-15)
    np.testing.assert_allclose(Fg[:, 2, 1], 0.0, atol=1e-15)
    # (1, 2) is zero too in the pure-screw matrix.
    np.testing.assert_allclose(Fg[:, 1, 2], 0.0, atol=1e-15)
    # (0, 1) and (0, 2) hold the screw terms (∂u_dx/∂y, ∂u_dx/∂z).
    sqz = rd[2] ** 2
    sqy = rd[1] ** 2
    denom1 = sqz + sqy + 1e-20
    bfactor1 = BURGERS_VECTOR / (2 * np.pi)
    expected_01 = -rd[2] / denom1 * bfactor1
    expected_02 = rd[1] / denom1 * bfactor1
    np.testing.assert_allclose(Fg[:, 0, 1], expected_01, rtol=1e-12)
    np.testing.assert_allclose(Fg[:, 0, 2], expected_02, rtol=1e-12)


def test_Fd_find_mixed_position_offset_shifts_singularity(identity_rotations):
    """Translating the dislocation core by `position_lab_um` is equivalent to
    evaluating Fd_find_mixed at rl shifted in the opposite direction.
    """
    Us, Ud, Theta = identity_rotations
    rl = np.array([[5e-6, 0.0], [0.0, 0.0], [0.0, 0.0]])  # single test point at (5µm, 0, 0)

    # Same physics: dislocation at origin, evaluate at (5µm, 0, 0).
    Fg_at_origin = Fd_find_mixed(rl, Us, Ud_mix=Ud, rotation_deg=0.0, Theta=Theta)

    # Equivalent: dislocation at (5µm, 0, 0), evaluate at the offset point itself
    # (which should map to rd ≈ 0, i.e. near the singularity).
    rl_at_offset = np.array([[5e-6], [0.0], [0.0]])
    Fg_at_offset_pos = Fd_find_mixed(
        rl_at_offset, Us, Ud_mix=Ud, rotation_deg=0.0, Theta=Theta,
        position_lab_um=(5.0, 0.0, 0.0),
    )

    # The first column of Fg_at_origin (point at +5µm, disloc at 0) should NOT
    # equal Fg_at_offset_pos (point at +5µm, disloc at +5µm) — the latter is at
    # the singularity (denom regularized by alpha=1e-20, gives a finite but
    # very different field). This asserts the offset actually shifts the disloc.
    diff = np.abs(Fg_at_origin[0] - Fg_at_offset_pos[0]).max()
    assert diff > 1e-6, f"position_lab_um had no effect (diff={diff})"


def test_Fd_find_mixed_uses_module_constants_as_defaults(identity_rotations, simple_rl_grid):
    """Calling without b/ny kwargs uses BURGERS_VECTOR/POISSON_RATIO from constants."""
    Us, Ud, Theta = identity_rotations
    rl = simple_rl_grid

    Fg_default = Fd_find_mixed(rl, Us, Ud_mix=Ud, rotation_deg=30.0, Theta=Theta)
    Fg_explicit = Fd_find_mixed(
        rl, Us, Ud_mix=Ud, rotation_deg=30.0, Theta=Theta,
        b=BURGERS_VECTOR, ny=POISSON_RATIO,
    )
    np.testing.assert_array_equal(Fg_default, Fg_explicit)
```

- [ ] **Step 2: Run tests, confirm they fail with "cannot import Fd_find_mixed"**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_dislocations_mixed.py -v
```

Expected: ImportError for `Fd_find_mixed`.

- [ ] **Step 3: Implement Fd_find_mixed in dislocations.py**

Append to `src/dfxm_geo/crystal/dislocations.py`:

```python
def Fd_find_mixed(
    rl: np.ndarray,
    Us: np.ndarray,
    Ud_mix: np.ndarray,
    rotation_deg: float,
    Theta: np.ndarray,
    *,
    b: float = BURGERS_VECTOR,
    ny: float = POISSON_RATIO,
    position_lab_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Displacement gradient Fg for a single mixed-character dislocation.

    Implements Eq. 1 of Borgi, Winther & Poulsen (2025), J. Appl. Cryst. 58,
    813-821, doi:10.1107/S1600576725002614::

        F_d = I + screw_matrix * cos(α_paper) + edge_matrix * sin(α_paper)

    where α_paper is the angle between Burgers vector and dislocation line
    (α_paper = 0° / 180° pure screw; 90° / 270° pure edge).

    **Parameterization note (differs from paper)**: ``rotation_deg`` is the
    angle (degrees) by which the line direction has been rotated around the
    slip-plane normal `n`, starting from the initial in-plane reference
    ``t_0 = b × n`` (which has α_paper=90°, pure edge). The two
    parameterizations satisfy ``α_paper = 90° - rotation_deg``, so:

        rotation_deg = 0   ⇔ α_paper = 90°  (pure edge)
        rotation_deg = 90° ⇔ α_paper = 0°   (pure screw)

    Naming preserves the convention of the branch source (`disloc_identify`)
    rather than the paper to keep callers unchanged.

    Args:
        rl: Lab-frame coordinates, shape (3, X).
        Us: Sample-to-grain rotation (Eq. 5 of Borgi 2025), shape (3, 3).
        Ud_mix: Dislocation-to-grain rotation (Eq. 3 of Borgi 2025), shape (3, 3).
        rotation_deg: See parameterization note above.
        Theta: Lab-to-sample rotation (Eq. 7 of Borgi 2025), shape (3, 3).
        b: Burgers vector magnitude. Default `BURGERS_VECTOR` from constants.
        ny: Poisson ratio. Default `POISSON_RATIO` from constants.
        position_lab_um: Lab-frame offset (µm); shifts rl before transforming
            to dislocation coords so the core sits at this offset. Default 0.

    Returns:
        Fg of shape (X, 3, 3) in the grain frame, with the identity added.
    """
    # Apply lab-frame position offset before any rotations.
    if position_lab_um != (0.0, 0.0, 0.0):
        offset_m = np.asarray(position_lab_um).reshape(3, 1) * 1e-6
        rl = rl - offset_m

    # Eq. 8 of Borgi 2025: r_l = Θ^T · Us · Ud · r_d → r_d = Ud^T · Us^T · Θ · r_l.
    rs = Theta @ rl
    rc = Us.T @ rs
    rd = Ud_mix.T @ rc

    Fdd = np.zeros([rd.shape[1], 3, 3])
    alpha = 1e-20  # singular regularizer at the dislocation core

    sqx = rd[0] * rd[0]
    sqy = rd[1] * rd[1]
    denom = (sqx + sqy) * (sqx + sqy) + alpha
    bfactor = b / (4 * np.pi * (1 - ny))
    nyfactor = 2 * ny * (sqx + sqy)

    # Edge formula (the cleanup's Appendix-A sign correction is already
    # applied: +nyfactor on [1, 1], -nyfactor on the other three).
    Fdd[:, 0, 0] = -rd[1] * (3 * sqx + sqy - nyfactor) / denom
    Fdd[:, 0, 1] = rd[0] * (3 * sqx + sqy - nyfactor) / denom
    Fdd[:, 1, 0] = -rd[0] * (3 * sqy + sqx - nyfactor) / denom
    Fdd[:, 1, 1] = rd[1] * (sqx - sqy + nyfactor) / denom

    Fdd *= bfactor
    Fdd *= np.cos(np.deg2rad(rotation_deg))

    # Screw out-of-plane contributions (∂u_dx/∂y, ∂u_dx/∂z).
    # denom1 = z² + y² preserved from branch source (Eq. 1's pure-screw matrix
    # specifies ∂u_dx/∂y and ∂u_dx/∂z; the analytic denominator is the
    # squared distance in the (y, z) plane).
    sqz = rd[2] * rd[2]
    denom1 = sqz + sqy + alpha
    bfactor1 = b / (2 * np.pi)
    sin_rot = np.sin(np.deg2rad(rotation_deg))

    Fdd[:, 0, 1] += (-rd[2] / denom1) * bfactor1 * sin_rot
    Fdd[:, 0, 2] += (rd[1] / denom1) * bfactor1 * sin_rot

    Fdd += np.identity(3)
    return Ud_mix @ Fdd @ Ud_mix.T
```

- [ ] **Step 4: Run tests, confirm they pass**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_dislocations_mixed.py -v
```

Expected: 7 passed (3 from Task 1 + 4 new).

- [ ] **Step 5: Verify mypy + ruff still clean**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff check src/ tests/
```

Expected: `Success: no issues found in N source files` and `All checks passed!`.

- [ ] **Step 6: Commit**

```
git add src/dfxm_geo/crystal/dislocations.py tests/test_dislocations_mixed.py
git commit -m "feat(crystal): Fd_find_mixed — single mixed-character dislocation (Borgi 2025 Eq. 1)"
```

---

## Task 3: Add Fd_find_multi_dislocs_mixed

**Files:**
- Modify: `src/dfxm_geo/crystal/dislocations.py` (append)
- Test: `tests/test_dislocations_mixed.py` (append)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_dislocations_mixed.py`:

```python
def test_Fd_find_multi_N1_matches_single(identity_rotations, simple_rl_grid):
    """For one crystal, Fd_find_multi_dislocs_mixed must equal Fd_find_mixed."""
    Us, Ud, Theta = identity_rotations
    rl = simple_rl_grid

    spec = MixedDislocSpec(Ud_mix=Ud, rotation_deg=37.0)
    Fg_multi = Fd_find_multi_dislocs_mixed(rl, Us, [spec], Theta)
    Fg_single = Fd_find_mixed(rl, Us, Ud_mix=Ud, rotation_deg=37.0, Theta=Theta)

    np.testing.assert_allclose(Fg_multi, Fg_single, atol=1e-15, rtol=1e-12)


def test_Fd_find_multi_N2_is_superposition(identity_rotations, simple_rl_grid):
    """Two crystals: result = (Fdd₁ - I) + (Fdd₂ - I) + I.

    The Identity is added once (in the grain frame); the per-crystal
    contributions are pure Fdd (no per-crystal identity). This is the
    generalization of Eq. 1 to N dislocations.
    """
    Us, Ud, Theta = identity_rotations
    rl = simple_rl_grid

    spec1 = MixedDislocSpec(Ud_mix=Ud, rotation_deg=10.0, position_lab_um=(1.0, 0.0, 0.0))
    spec2 = MixedDislocSpec(Ud_mix=Ud, rotation_deg=80.0, position_lab_um=(-1.0, 0.0, 0.0))

    Fg_multi = Fd_find_multi_dislocs_mixed(rl, Us, [spec1, spec2], Theta)

    Fg1 = Fd_find_mixed(
        rl, Us, Ud_mix=Ud, rotation_deg=10.0, Theta=Theta, position_lab_um=(1.0, 0.0, 0.0)
    )
    Fg2 = Fd_find_mixed(
        rl, Us, Ud_mix=Ud, rotation_deg=80.0, Theta=Theta, position_lab_um=(-1.0, 0.0, 0.0)
    )
    I = np.identity(3)
    expected = (Fg1 - I) + (Fg2 - I) + I

    np.testing.assert_allclose(Fg_multi, expected, atol=1e-15, rtol=1e-12)


def test_Fd_find_multi_empty_raises(identity_rotations, simple_rl_grid):
    """Empty crystals list is a programmer error — fail loudly."""
    Us, _, Theta = identity_rotations
    with pytest.raises(ValueError, match="at least one"):
        Fd_find_multi_dislocs_mixed(simple_rl_grid, Us, [], Theta)
```

- [ ] **Step 2: Run tests, confirm failure**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_dislocations_mixed.py -v
```

Expected: ImportError for `Fd_find_multi_dislocs_mixed`.

- [ ] **Step 3: Implement Fd_find_multi_dislocs_mixed**

Append to `src/dfxm_geo/crystal/dislocations.py`:

```python
def Fd_find_multi_dislocs_mixed(
    rl: np.ndarray,
    Us: np.ndarray,
    crystals: list[MixedDislocSpec],
    Theta: np.ndarray,
    *,
    b: float = BURGERS_VECTOR,
    ny: float = POISSON_RATIO,
) -> np.ndarray:
    """Sum of mixed-dislocation contributions from N crystals.

    Generalises Eq. 1 of Borgi 2025 to multiple dislocations: each crystal's
    screw+edge contributions are summed (no per-crystal identity), and the
    identity is added once at the end. For N=1 this reduces to
    ``Fd_find_mixed``; for N=2 it is the case used by the multi-disloc Monte
    Carlo pipeline mode.

    Args:
        rl: Lab-frame coordinates, shape (3, X).
        Us: Sample-to-grain rotation, shape (3, 3).
        crystals: list of `MixedDislocSpec`, at least one.
        Theta: Lab-to-sample rotation, shape (3, 3).
        b: Burgers vector magnitude (µm).
        ny: Poisson ratio.

    Returns:
        Fg of shape (X, 3, 3) in the grain frame, with the identity added once.
    """
    if not crystals:
        raise ValueError("Fd_find_multi_dislocs_mixed requires at least one crystal")

    I = np.identity(3)
    Fg_sum = np.zeros((rl.shape[1], 3, 3))
    for spec in crystals:
        Fg_one = Fd_find_mixed(
            rl,
            Us,
            Ud_mix=spec.Ud_mix,
            rotation_deg=spec.rotation_deg,
            Theta=Theta,
            b=b,
            ny=ny,
            position_lab_um=spec.position_lab_um,
        )
        Fg_sum += Fg_one - I

    return Fg_sum + I
```

- [ ] **Step 4: Run all tests, confirm pass**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_dislocations_mixed.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Verify mypy + ruff still clean**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff check src/ tests/
```

- [ ] **Step 6: Commit**

```
git add src/dfxm_geo/crystal/dislocations.py tests/test_dislocations_mixed.py
git commit -m "feat(crystal): Fd_find_multi_dislocs_mixed — sum N mixed dislocations"
```

---

## Task 4: Create burgers.py with burgers_vectors lookup

**Files:**
- Create: `src/dfxm_geo/crystal/burgers.py`
- Test: `tests/test_burgers.py` (create)

- [ ] **Step 1: Create the test file**

Create `tests/test_burgers.py`:

```python
"""Unit tests for dfxm_geo.crystal.burgers — slip-plane geometry helpers."""

import numpy as np
import pytest

from dfxm_geo.crystal.burgers import burgers_vectors


# The four {111}-family normals from the branch source's lookup table.
SLIP_PLANE_NORMALS = [
    (1, 1, 1),
    (1, -1, 1),
    (1, 1, -1),
    (-1, 1, 1),
]


@pytest.mark.parametrize("normal", SLIP_PLANE_NORMALS)
def test_burgers_vectors_shape(normal):
    """Returns (6, 3) — 3 basis Burgers vectors + 3 negatives."""
    b = burgers_vectors(normal)
    assert b.shape == (6, 3)


@pytest.mark.parametrize("normal", SLIP_PLANE_NORMALS)
def test_burgers_vectors_perpendicular_to_normal(normal):
    """All 6 Burgers vectors satisfy b · n = 0 (b lies in the slip plane)."""
    n = np.asarray(normal, dtype=float)
    b = burgers_vectors(normal)
    dots = b @ n
    np.testing.assert_allclose(dots, 0.0, atol=1e-12)


@pytest.mark.parametrize("normal", SLIP_PLANE_NORMALS)
def test_burgers_vectors_paired_negatives(normal):
    """The 6 vectors come in 3 ± pairs: b[i+3] == -b[i]."""
    b = burgers_vectors(normal)
    np.testing.assert_array_equal(b[3:], -b[:3])


def test_burgers_vectors_unit_magnitude_in_aluminum_units(normal=(1, 1, 1)):
    """Vectors are normalized to magnitude 1/sqrt(2) (matches branch code:
    `np.vstack([basis, -basis]) / np.sqrt(2)`).
    """
    b = burgers_vectors(normal)
    mags = np.linalg.norm(b, axis=1)
    np.testing.assert_allclose(mags, 1.0 / np.sqrt(2), rtol=1e-12)


def test_burgers_vectors_invalid_normal_raises():
    """Non-{111} normal raises ValueError."""
    with pytest.raises(ValueError, match="not one of the four"):
        burgers_vectors((2, 0, 0))
```

- [ ] **Step 2: Run, confirm fail**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_burgers.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create burgers.py with the lookup function**

Create `src/dfxm_geo/crystal/burgers.py`:

```python
"""Slip-plane geometry helpers for the dislocation-identification workflow.

Provides:
    burgers_vectors(slip_plane_normal): {111}-family Burgers vector basis.
    rotated_t_vectors(...): line-direction rotation around n.
    ud_matrices(...): construct Ud_mix from (rotated_t, n, b) basis.

References:
    Borgi, S., Winther, G., Poulsen, H. F. (2025). J. Appl. Cryst. 58, 813-821.
    DOI: 10.1107/S1600576725002614. Eq. 3 defines Ud = [b | n | t] columns.
"""

from __future__ import annotations

import numpy as np

# Lookup table for the four {111}-family slip-plane normals. Each entry maps
# the slug "h k l" (with `-` for minus) to its three basis Burgers vectors,
# in the cubic crystal frame. Negatives are appended in `burgers_vectors`.
_BASIS_TABLE: dict[str, np.ndarray] = {
    "111": np.array([[-1, 1, 0], [1, 0, -1], [0, 1, -1]], dtype=float),
    "1-11": np.array([[1, 1, 0], [1, 0, -1], [0, 1, 1]], dtype=float),
    "11-1": np.array([[1, -1, 0], [1, 0, 1], [0, -1, -1]], dtype=float),
    "-111": np.array([[-1, -1, 0], [-1, 0, -1], [0, 1, -1]], dtype=float),
}


def _slug(slip_plane_normal: tuple[int, int, int]) -> str:
    """Convert (h, k, l) to a lookup key like '1-11'."""
    return "".join(str(c) if c >= 0 else f"-{abs(c)}" for c in slip_plane_normal)


def burgers_vectors(slip_plane_normal: tuple[int, int, int]) -> np.ndarray:
    """Return the 6 Burgers vectors associated with a {111}-family slip plane.

    Args:
        slip_plane_normal: One of (1,1,1), (1,-1,1), (1,1,-1), (-1,1,1).

    Returns:
        Array of shape (6, 3) — three basis vectors followed by their negatives,
        normalised to magnitude 1/√2 (matches branch source convention).

    Raises:
        ValueError if slip_plane_normal is not one of the four {111} variants.
    """
    key = _slug(slip_plane_normal)
    if key not in _BASIS_TABLE:
        raise ValueError(
            f"slip_plane_normal {slip_plane_normal} is not one of the four "
            f"{{111}}-family variants {list(_BASIS_TABLE.keys())}"
        )
    basis = _BASIS_TABLE[key]
    return np.vstack([basis, -basis]) / np.sqrt(2)
```

- [ ] **Step 4: Run tests, confirm pass**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_burgers.py -v
```

Expected: 10 passed (parametrize × 3 + 2 standalone).

- [ ] **Step 5: Commit**

```
git add src/dfxm_geo/crystal/burgers.py tests/test_burgers.py
git commit -m "feat(crystal): burgers.py with burgers_vectors lookup ({111}-family)"
```

---

## Task 5: Add rotated_t_vectors + ud_matrices

**Files:**
- Modify: `src/dfxm_geo/crystal/burgers.py` (append)
- Test: `tests/test_burgers.py` (append)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_burgers.py`:

```python
from dfxm_geo.crystal.burgers import rotated_t_vectors, ud_matrices


def test_rotated_t_vectors_shape():
    """Shape is (n_angles, n_burgers, 3)."""
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b = burgers_vectors((1, 1, 1))
    angles = np.array([0.0, 90.0, 180.0])
    result = rotated_t_vectors(n, b, angles)
    assert result.shape == (3, 6, 3)


def test_rotated_t_vectors_zero_angle_is_b_cross_n():
    """At angle=0, the rotated vector equals t_0 = b × n (initial in-plane)."""
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b = burgers_vectors((1, 1, 1))
    angles = np.array([0.0])
    result = rotated_t_vectors(n, b, angles)  # (1, 6, 3)

    t_expected = np.cross(b, n)
    np.testing.assert_allclose(result[0], t_expected, atol=1e-12)


def test_rotated_t_vectors_180_negates():
    """At angle=180°, the rotated vector is -t_0."""
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b = burgers_vectors((1, 1, 1))
    result = rotated_t_vectors(n, b, np.array([180.0]))
    t_expected = -np.cross(b, n)
    np.testing.assert_allclose(result[0], t_expected, atol=1e-12)


def test_ud_matrices_shape():
    """Shape is (n_angles, n_burgers, 3, 3)."""
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b = burgers_vectors((1, 1, 1))
    rotated = rotated_t_vectors(n, b, np.array([0.0, 45.0]))
    Ud = ud_matrices(n, rotated)
    assert Ud.shape == (2, 6, 3, 3)


def test_ud_matrices_columns_are_basis():
    """Each Ud has columns (b × n × t, n, t) per branch source convention
    (translates Eq. 3 of Borgi 2025 with the branch's column ordering).
    """
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b = burgers_vectors((1, 1, 1))
    rotated = rotated_t_vectors(n, b, np.array([30.0]))
    Ud = ud_matrices(n, rotated)
    # For each (angle, b_idx), Ud column 2 should equal the corresponding t vector.
    for j in range(6):
        np.testing.assert_allclose(Ud[0, j, :, 2], rotated[0, j], atol=1e-12)
        np.testing.assert_allclose(Ud[0, j, :, 1], n, atol=1e-12)
```

- [ ] **Step 2: Run, confirm fail**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_burgers.py -v
```

Expected: ImportError for `rotated_t_vectors`.

- [ ] **Step 3: Implement rotated_t_vectors + ud_matrices**

Append to `src/dfxm_geo/crystal/burgers.py`:

```python
from scipy.spatial.transform import Rotation as _Rotation


def rotated_t_vectors(
    slip_plane_normal: np.ndarray,
    burgers: np.ndarray,
    angles_deg: np.ndarray,
) -> np.ndarray:
    """Rotate each in-plane initial line direction `t_0 = b × n` around `n`.

    For every (angle, Burgers vector) pair, computes
    ``R_n(angle) · (b × n)`` where ``R_n`` is rotation around `n` (degrees).
    Mirrors the branch source's `BurgersVectorsPlotter.calculate_rotated_vectors`.

    Args:
        slip_plane_normal: shape (3,) — the slip-plane normal `n`.
        burgers: shape (n_burgers, 3) — the Burgers vectors.
        angles_deg: shape (n_angles,) — rotation angles in degrees.

    Returns:
        ndarray of shape (n_angles, n_burgers, 3) — rotated t-vectors.
    """
    n = np.asarray(slip_plane_normal, dtype=float)
    t_0 = np.cross(burgers, n)  # (n_burgers, 3)

    out = np.zeros((len(angles_deg), len(burgers), 3))
    for i, angle in enumerate(angles_deg):
        rot = _Rotation.from_rotvec(angle * n, degrees=True)
        out[i] = rot.apply(t_0)
    return out


def ud_matrices(
    slip_plane_normal: np.ndarray,
    rotated_vectors: np.ndarray,
) -> np.ndarray:
    """Construct Ud_mix matrices from (n × t, n, t) basis frames.

    Mirrors the branch source's `BurgersVectorsPlotter.calculate_ud_matrices`,
    which stacks ``(np.cross(n, t), n, t)`` as the three columns of each Ud
    matrix and reshapes to (n_angles, n_burgers, 3, 3).

    Args:
        slip_plane_normal: shape (3,) — the slip-plane normal `n`.
        rotated_vectors: shape (n_angles, n_burgers, 3) from rotated_t_vectors.

    Returns:
        ndarray of shape (n_angles, n_burgers, 3, 3).
    """
    n = np.asarray(slip_plane_normal, dtype=float)
    n_angles, n_burgers, _ = rotated_vectors.shape
    Ud = np.zeros((n_angles, n_burgers, 3, 3))
    for i in range(n_angles):
        for j in range(n_burgers):
            t = rotated_vectors[i, j]
            cross_nt = np.cross(n, t)
            Ud[i, j] = np.column_stack([cross_nt, n, t])
    return Ud
```

- [ ] **Step 4: Run tests, confirm pass**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_burgers.py -v
```

Expected: 15 passed.

- [ ] **Step 5: Verify mypy + ruff**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff check src/ tests/
```

- [ ] **Step 6: Commit**

```
git add src/dfxm_geo/crystal/burgers.py tests/test_burgers.py
git commit -m "feat(crystal): rotated_t_vectors + ud_matrices geometry helpers"
```

---

## Task 6: Add viz/burgers.py (plotly, lazy import)

**Files:**
- Create: `src/dfxm_geo/viz/burgers.py`
- Test: `tests/test_viz_burgers.py` (create)

- [ ] **Step 1: Create failing test file**

Create `tests/test_viz_burgers.py`:

```python
"""Unit tests for dfxm_geo.viz.burgers — interactive 3D plotly viz."""

import sys
from unittest.mock import patch

import numpy as np
import pytest

from dfxm_geo.crystal.burgers import burgers_vectors, rotated_t_vectors


def test_plot_slip_plane_3d_returns_figure_with_expected_traces():
    """Returns a plotly Figure with one surface (the plane) + N traces for vectors."""
    plotly = pytest.importorskip("plotly")
    from dfxm_geo.viz.burgers import plot_slip_plane_3d

    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b = burgers_vectors((1, 1, 1))
    rotated = rotated_t_vectors(n, b, np.array([0.0, 90.0, 180.0]))

    fig = plot_slip_plane_3d(n, b, rotated)

    # 1 surface (plane) + 6 Burgers vectors + 3 rotated t-vectors (one per
    # angle, sampled at b_idx=0). The branch source plots rotated_vectors[:, 0]
    # so we mirror that: 1 + 6 + 3 = 10 traces.
    assert len(fig.data) == 10


def test_plot_slip_plane_3d_missing_plotly_raises_runtime_error(monkeypatch):
    """If plotly is not installed, raise a clear error pointing to the extras."""
    # Force ImportError on `import plotly.graph_objects` from inside the function.
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)

    from dfxm_geo.viz.burgers import plot_slip_plane_3d

    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b = burgers_vectors((1, 1, 1))
    rotated = rotated_t_vectors(n, b, np.array([0.0]))

    with pytest.raises(RuntimeError, match="plotly is required"):
        plot_slip_plane_3d(n, b, rotated)
```

- [ ] **Step 2: Run, confirm fail**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_viz_burgers.py -v
```

Expected: ImportError for `dfxm_geo.viz.burgers`.

- [ ] **Step 3: Create viz/burgers.py**

Create `src/dfxm_geo/viz/burgers.py`:

```python
"""Interactive 3D visualization of slip-plane geometry (plotly).

Requires the optional ``[identification]`` dep group. Plotly is imported
lazily inside `plot_slip_plane_3d` so the rest of the package imports
cleanly when plotly isn't installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go


def plot_slip_plane_3d(
    slip_plane_normal: np.ndarray,
    burgers: np.ndarray,
    rotated_vectors: np.ndarray,
) -> "go.Figure":
    """Interactive 3D figure showing slip plane + Burgers vectors + rotated t-vectors.

    Caller decides whether to `.show()` (in a notebook) or `.write_html(path)`.

    Args:
        slip_plane_normal: shape (3,) — the slip-plane normal `n`.
        burgers: shape (n_burgers, 3) — Burgers vectors (typically 6).
        rotated_vectors: shape (n_angles, n_burgers, 3) — rotated t-vectors.

    Returns:
        plotly Figure.

    Raises:
        RuntimeError if plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
    except (ImportError, TypeError) as exc:
        raise RuntimeError(
            "plotly is required for plot_slip_plane_3d. "
            "Install via: pip install 'dfxm-geo[identification]'"
        ) from exc

    n = np.asarray(slip_plane_normal, dtype=float)
    fig = go.Figure()

    # The plane (z = (-n_x x - n_y y) / n_z within a (-1, 1) box).
    xx, yy = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
    zz = (-n[0] * xx - n[1] * yy) / n[2]
    fig.add_trace(
        go.Surface(
            z=zz,
            x=xx,
            y=yy,
            showscale=False,
            colorscale=[[0, "black"], [1, "black"]],
            opacity=0.6,
        )
    )

    # Burgers vectors (red) — match branch convention of scaling by sqrt(2).
    b_scaled = burgers * np.sqrt(2)
    for i, b_vec in enumerate(b_scaled):
        fig.add_trace(
            go.Scatter3d(
                x=[0, b_vec[0]],
                y=[0, b_vec[1]],
                z=[0, b_vec[2]],
                mode="lines+markers",
                name=f"b_{i}",
                marker=dict(size=4, color="red"),
                line=dict(color="red", width=4),
            )
        )

    # Rotated t-vectors at b_idx=0 (blue) — one per angle.
    for i, vec in enumerate(rotated_vectors[:, 0]):
        fig.add_trace(
            go.Scatter3d(
                x=[0, vec[0]],
                y=[0, vec[1]],
                z=[0, vec[2]],
                mode="lines+markers",
                name=f"t_{i}",
                marker=dict(size=4, color="blue"),
                line=dict(color="blue", width=3),
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="x [001]",
            yaxis_title="y [010]",
            zaxis_title="z [100]",
            aspectmode="cube",
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
        ),
    )
    return fig
```

- [ ] **Step 4: Run tests, confirm pass**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_viz_burgers.py -v
```

Expected: 2 passed (or 1 passed + 1 skipped if plotly not installed locally — that's OK).

- [ ] **Step 5: Add plotly to dev deps temporarily so the test passes locally**

(Will be moved to the proper `[identification]` group in Task 7. For now: install plotly into the venv ad-hoc.)

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pip install "plotly>=5"
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_viz_burgers.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Add mypy override for plotly to keep type-check clean**

Edit `pyproject.toml`, under the existing `[[tool.mypy.overrides]]` section that ignores numba/tqdm/etc., add `plotly.*`:

```toml
[[tool.mypy.overrides]]
module = [
    "numba.*",
    "tqdm.*",
    "fabio.*",
    "joblib.*",
    "scipy.*",
    "xraylib",
    "plotly.*",
]
ignore_missing_imports = true
```

(If the override block doesn't already list these exactly, just add `plotly.*` to whatever module list is present.)

- [ ] **Step 7: Verify mypy + ruff clean**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff check src/ tests/
```

- [ ] **Step 8: Commit**

```
git add src/dfxm_geo/viz/burgers.py tests/test_viz_burgers.py pyproject.toml
git commit -m "feat(viz): plot_slip_plane_3d (plotly, lazy import)"
```

---

## Task 7: Register [identification] optional-deps + dfxm-identify console script in pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `[identification]` to optional-dependencies and dfxm-identify to scripts**

Edit `pyproject.toml`. In the existing `[project.optional-dependencies]` section (which already has `dev`, `beamstop-wire`, `memory-aware`), add:

```toml
identification = [
    "plotly>=5",
]
```

In the existing `[project.scripts]` section (which already has `dfxm-forward = "dfxm_geo.pipeline:cli_main"`), add the new entry below `dfxm-forward`:

```toml
dfxm-identify = "dfxm_geo.pipeline:cli_main_identify"
```

Note: `cli_main_identify` doesn't exist yet — it will be created in Task 13. The pyproject change is wired up now so the console script is ready when the function lands.

- [ ] **Step 2: Verify pyproject.toml is still valid TOML by reinstalling editable**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pip install -e .
```

Expected: install succeeds, no TOML parsing errors. The `dfxm-identify` entry-point is registered but calling it will fail with `ImportError` until cli_main_identify is implemented in Task 13 — that's fine, we won't try to invoke it until then.

- [ ] **Step 3: Verify ruff still clean**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff check src/ tests/
```

- [ ] **Step 4: Commit**

```
git add pyproject.toml
git commit -m "build: register [identification] extras + dfxm-identify console script"
```

---

## Task 8: Add IdentificationCrystalConfig + IdentificationScanConfig + IdentificationMonteCarloConfig dataclasses

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (append)
- Test: `tests/test_pipeline_identification.py` (create)

- [ ] **Step 1: Create failing test file**

Create `tests/test_pipeline_identification.py`:

```python
"""Unit tests for dfxm-identify pipeline shape."""

import numpy as np
import pytest

from dfxm_geo.pipeline import (
    IdentificationCrystalConfig,
    IdentificationMonteCarloConfig,
    IdentificationScanConfig,
)


def test_identification_crystal_config_defaults():
    cfg = IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1))
    assert cfg.slip_plane_normal == (1, 1, 1)
    assert cfg.angle_start_deg == 0.0
    assert cfg.angle_stop_deg == 350.0
    assert cfg.angle_step_deg == 10.0
    assert cfg.b_vector_indices is None
    assert cfg.sweep_all_slip_planes is True
    assert cfg.exclude_invisibility is True
    assert cfg.invisibility_threshold_deg == 10.0


def test_identification_scan_config_defaults():
    cfg = IdentificationScanConfig()
    assert cfg.phi_rad == pytest.approx(150e-6)
    assert cfg.poisson_noise is True
    assert cfg.rng_seed == 0
    assert cfg.intensity_scale == 7.0


def test_identification_montecarlo_config_defaults():
    cfg = IdentificationMonteCarloConfig()
    assert cfg.n_samples == 1000
    assert cfg.pos_std_um == 5.0
    assert cfg.n_png_previews == 50


def test_identification_crystal_config_is_frozen():
    cfg = IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1))
    with pytest.raises(Exception):
        cfg.angle_step_deg = 5.0  # type: ignore[misc]
```

- [ ] **Step 2: Run, confirm fail**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py -v
```

Expected: ImportError.

- [ ] **Step 3: Append the dataclasses to pipeline.py**

Append to `src/dfxm_geo/pipeline.py` (after the existing dataclasses, before `run_simulation`):

```python
@dataclass(frozen=True, kw_only=True)
class IdentificationCrystalConfig:
    """Crystal config for `dfxm-identify`. Slip plane + Burgers vector sweep."""

    slip_plane_normal: tuple[int, int, int]
    angle_start_deg: float = 0.0
    angle_stop_deg: float = 350.0
    angle_step_deg: float = 10.0
    b_vector_indices: list[int] | None = None  # None = all 6
    sweep_all_slip_planes: bool = True
    exclude_invisibility: bool = True
    invisibility_threshold_deg: float = 10.0


@dataclass(frozen=True, kw_only=True)
class IdentificationScanConfig:
    """Forward-model scan parameters for `dfxm-identify`."""

    phi_rad: float = 150e-6
    poisson_noise: bool = True
    rng_seed: int = 0
    intensity_scale: float = 7.0


@dataclass(frozen=True, kw_only=True)
class IdentificationMonteCarloConfig:
    """Multi-disloc Monte Carlo parameters (mode='multi' only)."""

    n_samples: int = 1000
    pos_std_um: float = 5.0
    n_png_previews: int = 50
```

(Make sure `from dataclasses import dataclass, field` is already imported at the top of `pipeline.py` — it should be, but add it if not.)

- [ ] **Step 4: Run, confirm pass**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Verify mypy + ruff**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff check src/ tests/
```

- [ ] **Step 6: Commit**

```
git add src/dfxm_geo/pipeline.py tests/test_pipeline_identification.py
git commit -m "feat(pipeline): IdentificationCrystalConfig + ScanConfig + MonteCarloConfig dataclasses"
```

---

## Task 9: Add IdentificationConfig with validation

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (append)
- Test: `tests/test_pipeline_identification.py` (append)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_pipeline_identification.py`:

```python
from dfxm_geo.pipeline import IdentificationConfig, IOConfig


def _make_io_config():
    return IOConfig(
        fn_prefix="/mosa_test_0000_",
        ftype=".npy",
        dislocs_dirname="identify",
        perfect_dirname="ignored",
        include_perfect_crystal=False,
    )


def test_identification_config_mode_single_ok():
    cfg = IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=IdentificationScanConfig(),
        io=_make_io_config(),
    )
    assert cfg.mode == "single"
    assert cfg.multi is None


def test_identification_config_mode_multi_ok():
    cfg = IdentificationConfig(
        mode="multi",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=IdentificationScanConfig(),
        multi=IdentificationMonteCarloConfig(),
        io=_make_io_config(),
    )
    assert cfg.mode == "multi"
    assert cfg.multi is not None


def test_identification_config_mode_multi_requires_multi_block():
    with pytest.raises(ValueError, match="mode='multi'"):
        IdentificationConfig(
            mode="multi",
            crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
            scan=IdentificationScanConfig(),
            multi=None,
            io=_make_io_config(),
        )


def test_identification_config_invalid_slip_plane_raises():
    with pytest.raises(ValueError, match="not one of the four"):
        IdentificationConfig(
            mode="single",
            crystal=IdentificationCrystalConfig(slip_plane_normal=(2, 0, 0)),
            scan=IdentificationScanConfig(),
            io=_make_io_config(),
        )


def test_identification_config_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        IdentificationConfig(
            mode="bogus",  # type: ignore[arg-type]
            crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
            scan=IdentificationScanConfig(),
            io=_make_io_config(),
        )
```

- [ ] **Step 2: Run, confirm fail**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py -v
```

Expected: ImportError for `IdentificationConfig`.

- [ ] **Step 3: Implement IdentificationConfig with validation**

Append to `src/dfxm_geo/pipeline.py`:

```python
from typing import Literal

from dfxm_geo.crystal.burgers import burgers_vectors as _burgers_vectors


@dataclass(frozen=True, kw_only=True)
class IdentificationConfig:
    """Top-level config for dfxm-identify.

    Validates mode/multi/slip-plane consistency in __post_init__.
    """

    mode: Literal["single", "multi"]
    crystal: IdentificationCrystalConfig
    scan: IdentificationScanConfig
    io: IOConfig
    multi: IdentificationMonteCarloConfig | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("single", "multi"):
            raise ValueError(f"mode must be 'single' or 'multi', got {self.mode!r}")
        if self.mode == "multi" and self.multi is None:
            raise ValueError("mode='multi' requires a `multi` config block")
        # Validate the slip plane against the {111} family (also used in 'multi'
        # mode as the starting / fallback plane).
        try:
            _burgers_vectors(self.crystal.slip_plane_normal)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
```

- [ ] **Step 4: Run, confirm pass**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py -v
```

Expected: 9 passed (4 from Task 8 + 5 new).

- [ ] **Step 5: Verify mypy + ruff**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff check src/ tests/
```

- [ ] **Step 6: Commit**

```
git add src/dfxm_geo/pipeline.py tests/test_pipeline_identification.py
git commit -m "feat(pipeline): IdentificationConfig with mode/multi/slip-plane validation"
```

---

## Task 10: Add TOML loader for IdentificationConfig

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (append)
- Test: `tests/test_pipeline_identification.py` (append)

- [ ] **Step 1: Add failing TOML round-trip test**

Append to `tests/test_pipeline_identification.py`:

```python
from dfxm_geo.pipeline import load_identification_config


def test_load_identification_config_single(tmp_path):
    """Round-trip a single-mode TOML file → IdentificationConfig."""
    toml_text = """
mode = "single"

[crystal]
slip_plane_normal = [1, 1, 1]
angle_start_deg = 0.0
angle_stop_deg = 350.0
angle_step_deg = 10.0
sweep_all_slip_planes = true
exclude_invisibility = true
invisibility_threshold_deg = 10.0

[scan]
phi_rad = 1.5e-4
poisson_noise = true
rng_seed = 0
intensity_scale = 7.0

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify"
perfect_dirname = "ignored"
include_perfect_crystal = false
"""
    cfg_path = tmp_path / "identification_single.toml"
    cfg_path.write_text(toml_text)

    cfg = load_identification_config(cfg_path)
    assert cfg.mode == "single"
    assert cfg.crystal.slip_plane_normal == (1, 1, 1)
    assert cfg.scan.phi_rad == pytest.approx(1.5e-4)
    assert cfg.multi is None


def test_load_identification_config_multi(tmp_path):
    """Multi-mode TOML round-trips, including the [multi] block."""
    toml_text = """
mode = "multi"

[crystal]
slip_plane_normal = [1, 1, 1]

[scan]
phi_rad = 1.5e-4
rng_seed = 42

[multi]
n_samples = 100
pos_std_um = 3.0
n_png_previews = 10

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify_multi"
perfect_dirname = "ignored"
include_perfect_crystal = false
"""
    cfg_path = tmp_path / "identification_multi.toml"
    cfg_path.write_text(toml_text)

    cfg = load_identification_config(cfg_path)
    assert cfg.mode == "multi"
    assert cfg.multi is not None
    assert cfg.multi.n_samples == 100
    assert cfg.scan.rng_seed == 42


def test_load_identification_config_missing_mode_raises(tmp_path):
    """A TOML missing the top-level `mode = ...` field raises."""
    cfg_path = tmp_path / "bad.toml"
    cfg_path.write_text('[crystal]\nslip_plane_normal = [1, 1, 1]\n')
    with pytest.raises(ValueError, match="missing top-level 'mode'"):
        load_identification_config(cfg_path)
```

- [ ] **Step 2: Run, confirm fail**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py -v
```

Expected: ImportError for `load_identification_config`.

- [ ] **Step 3: Implement the TOML loader**

Append to `src/dfxm_geo/pipeline.py`:

```python
import tomllib


def load_identification_config(path: Path) -> IdentificationConfig:
    """Load and validate an `dfxm-identify` config from a TOML file."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    if "mode" not in data:
        raise ValueError(f"{path}: missing top-level 'mode' field")

    crystal_data = data.get("crystal", {})
    if "slip_plane_normal" in crystal_data:
        crystal_data = {**crystal_data, "slip_plane_normal": tuple(crystal_data["slip_plane_normal"])}
    crystal = IdentificationCrystalConfig(**crystal_data)
    scan = IdentificationScanConfig(**data.get("scan", {}))
    io = IOConfig(**data.get("io", {}))
    multi = (
        IdentificationMonteCarloConfig(**data["multi"])
        if data.get("multi") is not None
        else None
    )

    return IdentificationConfig(
        mode=data["mode"],
        crystal=crystal,
        scan=scan,
        io=io,
        multi=multi,
    )
```

- [ ] **Step 4: Run, confirm pass**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Verify mypy + ruff**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff check src/ tests/
```

- [ ] **Step 6: Commit**

```
git add src/dfxm_geo/pipeline.py tests/test_pipeline_identification.py
git commit -m "feat(pipeline): load_identification_config — TOML loader for dfxm-identify"
```

---

## Task 11: Implement run_identification_single (deterministic 864-sweep + invisibility filter)

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (append)
- Test: `tests/test_pipeline_identification.py` (append)

- [ ] **Step 1: Add failing test for the single-mode sweep**

Append to `tests/test_pipeline_identification.py`:

```python
from dfxm_geo.pipeline import _run_identification_single


def _tiny_single_config(tmp_path):
    return IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=(1, 1, 1),
            angle_start_deg=0.0,
            angle_stop_deg=90.0,
            angle_step_deg=90.0,  # only 2 angles: 0 and 90
            b_vector_indices=[0, 1],  # only 2 Burgers vectors
            sweep_all_slip_planes=False,  # just one plane
            exclude_invisibility=False,  # don't filter
        ),
        scan=IdentificationScanConfig(rng_seed=0, intensity_scale=1.0),
        io=_make_io_config(),
    )


def test_run_identification_single_writes_expected_count(tmp_path, monkeypatch):
    """Tiny sweep (1 slip plane × 2 b × 2 angles = 4 images) writes 4 .npy files
    and 4 PNGs, and one manifest.csv with 4 rows.
    """
    import dfxm_geo.direct_space.forward_model as fm
    # Stub forward() so the test doesn't need the real pickle: return a small
    # deterministic array sized like the canonical detector.
    expected_image = np.ones((170, 510))
    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))  # placeholder
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: expected_image)

    output_dir = tmp_path / "out"
    cfg = _tiny_single_config(tmp_path)

    result = _run_identification_single(cfg, output_dir)

    npys = sorted((output_dir / "n_1_1_1" / "im_data").glob("*.npy"))
    assert len(npys) == 4
    pngs = sorted((output_dir / "n_1_1_1" / "images").glob("*.png"))
    assert len(pngs) == 4
    manifest = output_dir / "manifest.csv"
    assert manifest.is_file()
    lines = manifest.read_text().strip().splitlines()
    # 1 header + 4 rows
    assert len(lines) == 5
    assert result["n_images"] == 4
```

- [ ] **Step 2: Run, confirm fail**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py::test_run_identification_single_writes_expected_count -v
```

Expected: ImportError for `_run_identification_single`.

- [ ] **Step 3: Implement _run_identification_single**

Append to `src/dfxm_geo/pipeline.py`:

```python
import csv

from dfxm_geo.crystal.burgers import (
    burgers_vectors as _burgers_vectors_lookup,
    rotated_t_vectors as _rotated_t_vectors,
    ud_matrices as _ud_matrices,
)
from dfxm_geo.crystal.dislocations import Fd_find_mixed
from dfxm_geo.crystal.rotations import fast_inverse2


def _slip_plane_slug(n: tuple[int, int, int]) -> str:
    """Convert (1, -1, 1) -> '1_m1_1' for directory names."""
    return "_".join("m" + str(abs(c)) if c < 0 else str(c) for c in n)


def _save_preview_png(arr: np.ndarray, png_path: Path) -> None:
    """Quick matplotlib heatmap snapshot for spot-checking."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.imshow(arr.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(png_path, dpi=72, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _passes_invisibility(
    q_hkl: np.ndarray,
    b_vec: np.ndarray,
    threshold_deg: float,
) -> bool:
    """True if |G·b| / (|G| |b|) >= cos(90° - threshold) — i.e. NOT near-orthogonal.

    Paper convention: configuration is excluded if Burgers vector is within
    `threshold_deg` degrees of perpendicular to G. cos(90° - 10°) = cos(80°) ≈ 0.174.
    """
    cos_angle = abs(np.dot(q_hkl, b_vec)) / (np.linalg.norm(q_hkl) * np.linalg.norm(b_vec))
    return bool(cos_angle >= np.cos(np.deg2rad(90.0 - threshold_deg)))


def _run_identification_single(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Deterministic Cartesian sweep: slip planes × Burgers vectors × angles."""
    import dfxm_geo.direct_space.forward_model as fm

    output_dir.mkdir(parents=True, exist_ok=True)
    crystal_cfg = config.crystal
    scan_cfg = config.scan

    # Slip planes to iterate.
    all_planes = [(1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, 1, 1)]
    planes = all_planes if crystal_cfg.sweep_all_slip_planes else [crystal_cfg.slip_plane_normal]

    # Angles to iterate (inclusive end if it falls on the grid).
    angles_deg = np.arange(
        crystal_cfg.angle_start_deg,
        crystal_cfg.angle_stop_deg + crystal_cfg.angle_step_deg * 0.5,
        crystal_cfg.angle_step_deg,
    )

    rng = np.random.default_rng(scan_cfg.rng_seed) if scan_cfg.poisson_noise else None
    q_hkl = np.asarray(fm.q_hkl, dtype=float)

    manifest_rows: list[dict[str, Any]] = []
    n_written = 0

    for plane in planes:
        plane_slug = _slip_plane_slug(plane)
        im_dir = output_dir / f"n_{plane_slug}" / "im_data"
        png_dir = output_dir / f"n_{plane_slug}" / "images"
        im_dir.mkdir(parents=True, exist_ok=True)
        png_dir.mkdir(parents=True, exist_ok=True)

        b_table = _burgers_vectors_lookup(plane)
        b_indices = (
            crystal_cfg.b_vector_indices
            if crystal_cfg.b_vector_indices is not None
            else list(range(len(b_table)))
        )
        b_subset = b_table[b_indices]
        n_arr = np.asarray(plane, dtype=float) / np.linalg.norm(np.asarray(plane, dtype=float))

        rotated = _rotated_t_vectors(n_arr, b_subset, angles_deg)
        Ud_all = _ud_matrices(n_arr, rotated)  # (n_angles, n_b, 3, 3)

        for j, b_idx in enumerate(b_indices):
            if crystal_cfg.exclude_invisibility and not _passes_invisibility(
                q_hkl, b_table[b_idx], crystal_cfg.invisibility_threshold_deg
            ):
                continue
            for i, alpha in enumerate(angles_deg):
                Ud_mix = Ud_all[i, j]
                Fg = Fd_find_mixed(
                    fm.rl,
                    fm.Us,
                    Ud_mix=Ud_mix,
                    rotation_deg=float(alpha),
                    Theta=fm.Theta,
                )
                Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1]) - np.identity(3)
                fm.Hg = Hg
                fm.q_hkl = q_hkl  # keep in sync

                image = fm.forward(Hg, phi=scan_cfg.phi_rad) * scan_cfg.intensity_scale
                if scan_cfg.poisson_noise:
                    assert rng is not None
                    image = rng.poisson(np.clip(image, a_min=0.0, a_max=None)).astype(float)

                stem = f"b{b_idx}_alpha{int(round(alpha)):03d}"
                npy_path = im_dir / f"{stem}.npy"
                png_path = png_dir / f"{stem}.png"
                np.save(npy_path, image)
                _save_preview_png(image, png_path)

                manifest_rows.append({
                    "image_path": str(npy_path.relative_to(output_dir)),
                    "n_h": plane[0], "n_k": plane[1], "n_l": plane[2],
                    "b_idx": b_idx,
                    "b_h": int(round(b_table[b_idx, 0] * np.sqrt(2))),
                    "b_k": int(round(b_table[b_idx, 1] * np.sqrt(2))),
                    "b_l": int(round(b_table[b_idx, 2] * np.sqrt(2))),
                    "rotation_deg": float(alpha),
                })
                n_written += 1

    # Manifest CSV.
    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as fh:
        if manifest_rows:
            writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        else:
            fh.write("")

    return {"n_images": n_written, "output_dir": output_dir, "manifest_path": manifest_path}
```

- [ ] **Step 4: Run, confirm pass**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py::test_run_identification_single_writes_expected_count -v
```

Expected: 1 passed.

- [ ] **Step 5: Run all pipeline_identification tests**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py -v
```

Expected: 13 passed.

- [ ] **Step 6: Verify mypy + ruff**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff check src/ tests/
```

- [ ] **Step 7: Commit**

```
git add src/dfxm_geo/pipeline.py tests/test_pipeline_identification.py
git commit -m "feat(pipeline): _run_identification_single — deterministic sweep + invisibility filter"
```

---

## Task 12: Implement _run_identification_multi (Monte Carlo)

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (append)
- Test: `tests/test_pipeline_identification.py` (append)

- [ ] **Step 1: Add failing test**

Append to `tests/test_pipeline_identification.py`:

```python
from dfxm_geo.pipeline import _run_identification_multi


def _tiny_multi_config():
    return IdentificationConfig(
        mode="multi",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=IdentificationScanConfig(rng_seed=0, intensity_scale=1.0),
        multi=IdentificationMonteCarloConfig(n_samples=3, pos_std_um=2.0, n_png_previews=2),
        io=_make_io_config(),
    )


def test_run_identification_multi_writes_samples_and_manifest(tmp_path, monkeypatch):
    """n_samples=3 writes 3 .npy + 2 PNGs (n_png_previews=2) + manifest with 3 rows."""
    import dfxm_geo.direct_space.forward_model as fm
    expected_image = np.ones((170, 510))
    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: expected_image)

    output_dir = tmp_path / "out"
    cfg = _tiny_multi_config()

    result = _run_identification_multi(cfg, output_dir)

    npys = sorted((output_dir / "im_data").glob("*.npy"))
    pngs = sorted((output_dir / "images").glob("*.png"))
    assert len(npys) == 3
    assert len(pngs) == 2  # only the first 2 samples get PNGs
    manifest = output_dir / "manifest.csv"
    assert manifest.is_file()
    lines = manifest.read_text().strip().splitlines()
    assert len(lines) == 4  # header + 3 rows
    assert result["n_samples"] == 3


def test_run_identification_multi_is_deterministic_for_seed(tmp_path, monkeypatch):
    """Two runs at the same seed produce identical manifests + image hashes."""
    import dfxm_geo.direct_space.forward_model as fm
    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    counter = {"n": 0}

    def fake_forward(*args, **kwargs):
        counter["n"] += 1
        return np.full((170, 510), float(counter["n"]))

    monkeypatch.setattr(fm, "forward", fake_forward)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    cfg = _tiny_multi_config()

    _run_identification_multi(cfg, out1)
    counter["n"] = 0  # reset
    _run_identification_multi(cfg, out2)

    m1 = (out1 / "manifest.csv").read_text()
    m2 = (out2 / "manifest.csv").read_text()
    assert m1 == m2
```

- [ ] **Step 2: Run, confirm fail**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py::test_run_identification_multi_writes_samples_and_manifest -v
```

Expected: ImportError for `_run_identification_multi`.

- [ ] **Step 3: Implement _run_identification_multi**

Append to `src/dfxm_geo/pipeline.py`:

```python
from dfxm_geo.crystal.dislocations import (
    Fd_find_multi_dislocs_mixed,
    MixedDislocSpec,
)


_ALL_111_PLANES: list[tuple[int, int, int]] = [
    (1, 1, 1),
    (1, -1, 1),
    (1, 1, -1),
    (-1, 1, 1),
]


def _draw_dislocation(rng: np.random.Generator, pos_std_um: float) -> dict[str, Any]:
    """Draw a single random dislocation (slip plane, Burgers idx, angle, position)."""
    plane_idx = int(rng.integers(0, len(_ALL_111_PLANES)))
    plane = _ALL_111_PLANES[plane_idx]
    b_table = _burgers_vectors_lookup(plane)
    b_idx = int(rng.integers(0, len(b_table)))
    alpha = float(rng.uniform(0.0, 360.0))
    pos = (float(rng.normal(0.0, pos_std_um)), float(rng.normal(0.0, pos_std_um)), 0.0)

    n = np.asarray(plane, dtype=float)
    rotated = _rotated_t_vectors(n / np.linalg.norm(n), b_table[b_idx:b_idx + 1], np.array([alpha]))
    Ud = _ud_matrices(n / np.linalg.norm(n), rotated)[0, 0]

    return {
        "plane": plane,
        "b_idx": b_idx,
        "b_vec": b_table[b_idx] * np.sqrt(2),
        "alpha_deg": alpha,
        "pos_um": pos,
        "Ud": Ud,
    }


def _run_identification_multi(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Monte Carlo over n_samples; each sample is 2 random mixed dislocations summed."""
    import dfxm_geo.direct_space.forward_model as fm

    assert config.multi is not None  # validated in __post_init__
    mc = config.multi
    scan_cfg = config.scan

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "im_data").mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    # Split master rng → child streams (param draws, Poisson noise).
    master = np.random.default_rng(scan_cfg.rng_seed)
    param_rng, noise_rng = master.spawn(2)

    manifest_rows: list[dict[str, Any]] = []
    q_hkl = np.asarray(fm.q_hkl, dtype=float)
    fm.q_hkl = q_hkl  # ensure in sync

    pad = max(5, len(str(mc.n_samples - 1)))
    for k in range(mc.n_samples):
        d1 = _draw_dislocation(param_rng, mc.pos_std_um)
        d2 = _draw_dislocation(param_rng, mc.pos_std_um)
        specs = [
            MixedDislocSpec(Ud_mix=d1["Ud"], rotation_deg=d1["alpha_deg"], position_lab_um=d1["pos_um"]),
            MixedDislocSpec(Ud_mix=d2["Ud"], rotation_deg=d2["alpha_deg"], position_lab_um=d2["pos_um"]),
        ]
        Fg = Fd_find_multi_dislocs_mixed(fm.rl, fm.Us, specs, fm.Theta)
        Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1]) - np.identity(3)
        fm.Hg = Hg

        image = fm.forward(Hg, phi=scan_cfg.phi_rad) * scan_cfg.intensity_scale
        if scan_cfg.poisson_noise:
            image = noise_rng.poisson(np.clip(image, a_min=0.0, a_max=None)).astype(float)

        stem = f"{k:0{pad}d}"
        np.save(output_dir / "im_data" / f"{stem}.npy", image)
        if k < mc.n_png_previews:
            _save_preview_png(image, output_dir / "images" / f"{stem}.png")

        manifest_rows.append({
            "sample_id": stem,
            "n1_h": d1["plane"][0], "n1_k": d1["plane"][1], "n1_l": d1["plane"][2],
            "b1_idx": d1["b_idx"],
            "b1_h": int(round(d1["b_vec"][0])), "b1_k": int(round(d1["b_vec"][1])), "b1_l": int(round(d1["b_vec"][2])),
            "alpha1_deg": d1["alpha_deg"],
            "x1_um": d1["pos_um"][0], "y1_um": d1["pos_um"][1],
            "n2_h": d2["plane"][0], "n2_k": d2["plane"][1], "n2_l": d2["plane"][2],
            "b2_idx": d2["b_idx"],
            "b2_h": int(round(d2["b_vec"][0])), "b2_k": int(round(d2["b_vec"][1])), "b2_l": int(round(d2["b_vec"][2])),
            "alpha2_deg": d2["alpha_deg"],
            "x2_um": d2["pos_um"][0], "y2_um": d2["pos_um"][1],
            "image_path": f"im_data/{stem}.npy",
        })

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as fh:
        if manifest_rows:
            writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    return {"n_samples": mc.n_samples, "output_dir": output_dir, "manifest_path": manifest_path}
```

- [ ] **Step 4: Run, confirm pass**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py -v
```

Expected: 15 passed.

- [ ] **Step 5: Verify mypy + ruff**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff check src/ tests/
```

- [ ] **Step 6: Commit**

```
git add src/dfxm_geo/pipeline.py tests/test_pipeline_identification.py
git commit -m "feat(pipeline): _run_identification_multi — Monte Carlo over 2 random dislocations"
```

---

## Task 13: Add run_identification dispatcher + cli_main_identify

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (append)
- Test: `tests/test_pipeline_identification.py` (append)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_pipeline_identification.py`:

```python
from dfxm_geo.pipeline import cli_main_identify, run_identification


def test_run_identification_dispatches_to_single(tmp_path, monkeypatch):
    import dfxm_geo.direct_space.forward_model as fm
    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: np.ones((170, 510)))

    cfg = _tiny_single_config(tmp_path)
    result = run_identification(cfg, tmp_path / "out")
    assert "n_images" in result


def test_run_identification_dispatches_to_multi(tmp_path, monkeypatch):
    import dfxm_geo.direct_space.forward_model as fm
    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: np.ones((170, 510)))

    cfg = _tiny_multi_config()
    result = run_identification(cfg, tmp_path / "out")
    assert "n_samples" in result


def test_cli_main_identify_parses_args(tmp_path, monkeypatch):
    """Smoke test: CLI parses --config and --output, calls run_identification, returns 0."""
    import dfxm_geo.direct_space.forward_model as fm
    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: np.ones((170, 510)))

    toml_text = """
mode = "single"

[crystal]
slip_plane_normal = [1, 1, 1]
angle_start_deg = 0.0
angle_stop_deg = 0.0
angle_step_deg = 10.0
b_vector_indices = [0]
sweep_all_slip_planes = false
exclude_invisibility = false

[scan]
phi_rad = 1.5e-4
poisson_noise = false
rng_seed = 0
intensity_scale = 1.0

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify"
perfect_dirname = "ignored"
include_perfect_crystal = false
"""
    cfg_path = tmp_path / "id.toml"
    cfg_path.write_text(toml_text)
    out_dir = tmp_path / "out"

    exit_code = cli_main_identify(["--config", str(cfg_path), "--output", str(out_dir)])
    assert exit_code == 0
    assert (out_dir / "manifest.csv").is_file()
```

- [ ] **Step 2: Run, confirm fail**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py::test_run_identification_dispatches_to_single -v
```

Expected: ImportError for `run_identification`.

- [ ] **Step 3: Implement dispatcher + CLI**

Append to `src/dfxm_geo/pipeline.py`:

```python
def run_identification(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Dispatch to single or multi runner based on config.mode."""
    if config.mode == "single":
        return _run_identification_single(config, output_dir)
    return _run_identification_multi(config, output_dir)


def cli_main_identify(argv: list[str] | None = None) -> int:
    """Argparse-driven entry point for `dfxm-identify`."""
    parser = argparse.ArgumentParser(description="DFXM dislocation identification simulation")
    parser.add_argument("--config", type=Path, required=True, help="Path to identification TOML config")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default=None,
        help="Override the config's mode field",
    )
    args = parser.parse_args(argv)

    cfg = load_identification_config(args.config)
    if args.mode is not None and args.mode != cfg.mode:
        # Rebuild the config with overridden mode (dataclass is frozen, so use dataclasses.replace).
        from dataclasses import replace
        cfg = replace(cfg, mode=args.mode)
        cfg.__post_init__()  # re-run validation

    result = run_identification(cfg, args.output)
    if cfg.mode == "single":
        print(f"Wrote {result['n_images']} images to {result['output_dir']}")
    else:
        print(f"Wrote {result['n_samples']} samples to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(cli_main_identify())
```

(If `argparse` isn't already imported at the top of `pipeline.py`, it should be — it's used by the existing `cli_main`. No new import needed.)

- [ ] **Step 4: Run, confirm pass**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py -v
```

Expected: 18 passed.

- [ ] **Step 5: Verify mypy + ruff**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff check src/ tests/
```

- [ ] **Step 6: Commit**

```
git add src/dfxm_geo/pipeline.py tests/test_pipeline_identification.py
git commit -m "feat(pipeline): run_identification dispatcher + cli_main_identify entrypoint"
```

---

## Task 14: Create example configs

**Files:**
- Create: `configs/identification_single.toml`
- Create: `configs/identification_multi.toml`
- Test: `tests/test_pipeline_identification.py` (append)

- [ ] **Step 1: Add failing test that loads both example configs**

Append to `tests/test_pipeline_identification.py`:

```python
from pathlib import Path as _Path


def test_example_single_config_loads():
    """configs/identification_single.toml parses and validates."""
    repo_root = _Path(__file__).resolve().parents[1]
    cfg = load_identification_config(repo_root / "configs" / "identification_single.toml")
    assert cfg.mode == "single"
    assert cfg.crystal.sweep_all_slip_planes is True
    assert cfg.crystal.exclude_invisibility is True


def test_example_multi_config_loads():
    """configs/identification_multi.toml parses and validates."""
    repo_root = _Path(__file__).resolve().parents[1]
    cfg = load_identification_config(repo_root / "configs" / "identification_multi.toml")
    assert cfg.mode == "multi"
    assert cfg.multi is not None
    assert cfg.multi.n_samples == 1000
```

- [ ] **Step 2: Run, confirm fail**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py::test_example_single_config_loads tests/test_pipeline_identification.py::test_example_multi_config_loads -v
```

Expected: FileNotFoundError.

- [ ] **Step 3: Create the configs**

Create `configs/identification_single.toml`:

```toml
# dfxm-identify: deterministic single-dislocation sweep
#
# Reproduces the test set of Borgi 2025 (J. Appl. Cryst. 58, 813-821):
#   4 slip planes × 6 Burgers vectors × 36 line directions = 864 images,
#   840 after excluding 24 near-invisibility (G·b ≈ 0) configurations.
#
# Override on the CLI with: dfxm-identify --config ... --output ... --mode ...

mode = "single"

[crystal]
slip_plane_normal = [1, 1, 1]      # starting plane; sweep_all_slip_planes overrides
angle_start_deg = 0.0
angle_stop_deg = 350.0
angle_step_deg = 10.0
# b_vector_indices = [0, 1, 2, 3, 4, 5]  # uncomment to restrict; default = all
sweep_all_slip_planes = true
exclude_invisibility = true
invisibility_threshold_deg = 10.0

[scan]
phi_rad = 1.5e-4                   # 8 × 10⁻⁴° ≈ 1.4e-4 rad in Borgi 2025
poisson_noise = true
rng_seed = 0
intensity_scale = 7.0              # branch source's `* 7` multiplier

[io]
fn_prefix = "/mosa_test_0000_"     # unused for dfxm-identify but IOConfig requires
ftype = ".npy"
dislocs_dirname = "identify"
perfect_dirname = "ignored"
include_perfect_crystal = false
```

Create `configs/identification_multi.toml`:

```toml
# dfxm-identify: Monte Carlo over 2 random dislocations per image.
#
# Forward-looking extension beyond Borgi 2025; intended for ML-training-data
# generation. Each sample image contains two mixed-character dislocations
# with independently drawn (slip plane, Burgers vector, rotation angle,
# in-plane position). The manifest.csv carries per-image ground-truth labels.

mode = "multi"

[crystal]
slip_plane_normal = [1, 1, 1]      # ignored in mode='multi'; required by validation
# angle_*_deg and b_vector_indices are also unused in 'multi' mode

[scan]
phi_rad = 1.5e-4
poisson_noise = true
rng_seed = 0
intensity_scale = 7.0

[multi]
n_samples = 1000
pos_std_um = 5.0                   # std of in-plane (x, y) draws
n_png_previews = 50                # write PNGs only for first N samples

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify_multi"
perfect_dirname = "ignored"
include_perfect_crystal = false
```

- [ ] **Step 4: Run, confirm pass**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py -v
```

Expected: 20 passed.

- [ ] **Step 5: Commit**

```
git add configs/identification_single.toml configs/identification_multi.toml tests/test_pipeline_identification.py
git commit -m "feat(configs): example identification_single + identification_multi TOMLs"
```

---

## Task 15: End-to-end CLI smoke test

**Files:**
- Test: `tests/test_pipeline_identification.py` (append)

- [ ] **Step 1: Add a tiny-sweep smoke test that runs the actual CLI binary**

Append to `tests/test_pipeline_identification.py`:

```python
import subprocess
import sys


def test_dfxm_identify_cli_end_to_end(tmp_path, monkeypatch):
    """Invoke `dfxm-identify` via subprocess on a tiny config; confirm exit 0.

    This is slow-ish; mark with @pytest.mark.slow to opt out via -m.
    """
    # Build a minimal config with 1 slip plane × 1 b × 1 angle = 1 image.
    toml_text = """
mode = "single"

[crystal]
slip_plane_normal = [1, 1, 1]
angle_start_deg = 0.0
angle_stop_deg = 0.0
angle_step_deg = 10.0
b_vector_indices = [0]
sweep_all_slip_planes = false
exclude_invisibility = false

[scan]
phi_rad = 1.5e-4
poisson_noise = false
rng_seed = 0
intensity_scale = 1.0

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify"
perfect_dirname = "ignored"
include_perfect_crystal = false
"""
    cfg_path = tmp_path / "smoke.toml"
    cfg_path.write_text(toml_text)
    out_dir = tmp_path / "out"

    # Invoke via `python -m dfxm_geo.pipeline ...` so we don't depend on the
    # installed console script (entry-point caching can be flaky in dev env).
    # NOTE: cli_main is the existing forward entrypoint; we run cli_main_identify directly.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from dfxm_geo.pipeline import cli_main_identify; "
            f"raise SystemExit(cli_main_identify(['--config', r'{cfg_path}', '--output', r'{out_dir}']))",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    npys = list((out_dir / "n_1_1_1" / "im_data").glob("*.npy"))
    assert len(npys) == 1
```

- [ ] **Step 2: Run the test (requires the default `Resq_i` kernel pickle to exist on this machine)**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py::test_dfxm_identify_cli_end_to_end -v
```

Expected: 1 passed (or skipped with clear message if kernel pickle missing — see Step 3 below).

- [ ] **Step 3: If kernel pickle is missing, add a fixture that generates it (or skip)**

If the test fails because `reciprocal_space/pkl_files/Resq_i_*.pkl` is missing on the CI runner:

Edit `tests/test_pipeline_identification.py` to add a `kernel_pickle` autouse skip-fixture at module scope:

```python
@pytest.fixture(autouse=True)
def _skip_if_no_kernel(request):
    """Skip CLI end-to-end tests if the default kernel pickle isn't available."""
    if "end_to_end" not in request.node.name:
        return
    from pathlib import Path as _P
    repo_root = _P(__file__).resolve().parents[1]
    pkl = repo_root / "reciprocal_space" / "pkl_files" / "Resq_i_20230913_1308.pkl"
    if not pkl.is_file():
        pytest.skip(f"kernel pickle missing: {pkl}")
```

- [ ] **Step 4: Run all pipeline_identification tests**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py -v
```

Expected: 21 passed (or 20 passed + 1 skipped if no kernel).

- [ ] **Step 5: Commit**

```
git add tests/test_pipeline_identification.py
git commit -m "test(pipeline): end-to-end smoke for dfxm-identify CLI"
```

---

## Task 16: Final validation + summary

**Files:** None modified — validation only.

- [ ] **Step 1: Run the full test suite**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest -q
```

Expected: all green. Roughly **111 + ~24 = ~135 tests** passing (existing 111 + new tests across Tasks 1-15).

- [ ] **Step 2: Verify mypy clean across the package**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
```

Expected: `Success: no issues found in 23 source files` (was 21 — added crystal/burgers.py and viz/burgers.py).

- [ ] **Step 3: Verify ruff clean**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff check src/ tests/
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff format --check src/ tests/
```

Expected: All checks passed.

- [ ] **Step 4: Verify the console script entrypoint resolves**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pip install -e .
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/dfxm-identify.exe --help
```

Expected: argparse help text printed. (On Windows the `.exe` shim; on POSIX, plain `dfxm-identify`.)

- [ ] **Step 5: Confirm git status is clean**

```
git status --short
git log --oneline main..HEAD | head -25
```

Expected: empty working tree (all commits made); 16 new commits ahead of where the port started.

- [ ] **Step 6: No final commit — Task 16 is a validation gate only**

The port is complete. Subagent-driven dev should mark this task done and report the summary stats:
- Total commits added by this port
- Test count delta (before → after)
- Coverage delta (if measured)
- mypy file count delta
- Any deviations from the spec (none expected)

---

## Spec Coverage Audit

| Spec requirement | Task |
|---|---|
| `Fd_find_mixed` (Eq. 1) — `rotation_deg` parameterization | Task 2 |
| `Fd_find_multi_dislocs_mixed` — sum of N MixedDislocSpec | Task 3 |
| `MixedDislocSpec` dataclass | Task 1 |
| `burgers_vectors({111} lookup)` | Task 4 |
| `rotated_t_vectors` + `ud_matrices` | Task 5 |
| `plot_slip_plane_3d` (plotly lazy import) | Task 6 |
| `[identification]` optional dep group + `dfxm-identify` script | Task 7 |
| `IdentificationCrystalConfig` (sweep_all_slip_planes, exclude_invisibility, threshold) | Task 8 |
| `IdentificationScanConfig` (phi_rad, poisson_noise, rng_seed, intensity_scale) | Task 8 |
| `IdentificationMonteCarloConfig` (n_samples, pos_std_um, n_png_previews) | Task 8 |
| `IdentificationConfig` with __post_init__ validation | Task 9 |
| `load_identification_config` TOML loader | Task 10 |
| `_run_identification_single` (864-sweep + invisibility filter) | Task 11 |
| `_run_identification_multi` (Monte Carlo, 2 random dislocations, deterministic on seed) | Task 12 |
| `run_identification` dispatcher | Task 13 |
| `cli_main_identify` (--config, --output, --mode override) | Task 13 |
| `configs/identification_single.toml` + `configs/identification_multi.toml` | Task 14 |
| End-to-end CLI smoke test | Task 15 |
| All tests green + mypy clean + ruff clean | Task 16 |

All 18 spec-numbered requirements have a corresponding task. ✓
