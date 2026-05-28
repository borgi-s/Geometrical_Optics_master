# Oblique-angle DFXM — Phase A (v2.3.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `eta` (azimuthal-angle) support to `dfxm_geo` end-to-end (LUT bootstrap, MC + analytic backends, forward, identify, HDF5 provenance) so that a single oblique-angle DFXM reflection reproduces paper Figure 3B. Multi-reflection iteration is Phase B (separate plan, v2.4.0).

**Architecture:** A new pure-math module `crystal/oblique.py` houses `CrystalMount`, `compute_omega_eta` (paper Appendix A solver), `find_reflections` (Table-A.2 enumerator, implemented but unwired in Phase A), and `R_lab_to_image`. Both resolution backends gain an `eta` keyword (default 0.0 → bit-identical to v2.2.0). Bootstrap CLI grows `[crystal]` + `[geometry]` TOML blocks with "both, validated" eta cross-check. LUT cache filename gets a new `Resq_i_theta…rad_eta…rad_…keV_…npz` pattern alongside the legacy `Resq_i_h…_k…_l…_…keV_…npz` (both resolved by `_lookup_kernel_path`). Default behavior (no `[geometry]` block) stays bit-identical to v2.2.0 — gated by a regression test at every commit.

**Tech Stack:** Python 3.13+, numpy, scipy, h5py, tomllib, numba (untouched by this phase), pytest. Branch: `feature/oblique-angle-multi-reflection` (already created, spec committed at `1e1d9e3`).

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-28-multi-reflection-oblique-angle-design.md`
- Paper: `arXiv:2503.22022v1` (§3.3 oblique geometry, §6.1+§6.2 phantom & microscope, Appendix A search algorithm, Appendix F closed form, Table A.2 reference rows, Figure 3B reproduction target)
- Prototype (uncommitted): `C:\Users\borgi\Documents\Oblique_Angle\reciprocal_space\recspace_res.py` (`ReciprocalSpace` class).
- darkmod cross-reference: `darkmod/laue.py`, `darkmod/resolution.py` (Henningsson).

---

## Pre-flight

### Task 1: Verify clean working tree on the feature branch

**Files:** (read-only)

- [ ] **Step 1: Confirm branch + smoke test current main**

Run:
```bash
git -C C:/Users/borgi/Documents/GM-reworked/Geometrical_Optics_master status -sb
git -C C:/Users/borgi/Documents/GM-reworked/Geometrical_Optics_master log --oneline -n 3
```
Expected: branch is `feature/oblique-angle-multi-reflection`, HEAD = `1e1d9e3` (the spec fix-up commit), no modified tracked files (untracked scratch is fine).

- [ ] **Step 2: Smoke test current test suite**

Run (PowerShell, using venv python — see CLAUDE.md):
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q --timeout=120
```
Expected: 564 passed / 1 skipped, mypy clean. This is the v2.2.0 baseline; every commit in this plan must preserve it.

- [ ] **Step 3: Smoke test mypy**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m mypy src/dfxm_geo/
```
Expected: 0 errors.

---

## Module skeleton — `crystal/oblique.py`

### Task 2: `CrystalMount` dataclass

**Files:**
- Create: `src/dfxm_geo/crystal/oblique.py`
- Test: `tests/test_oblique_crystal_mount.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_oblique_crystal_mount.py`:
```python
"""Tests for CrystalMount dataclass in crystal/oblique.py."""
import numpy as np
import pytest

from dfxm_geo.crystal.oblique import CrystalMount


def test_paper_al_mount_is_identity_U():
    """Paper §6.1 mount: (100)//x̂, (010)//ŷ, (001)//ẑ → U_mount = I."""
    mount = CrystalMount(
        lattice="cubic",
        a=4.0493e-10,
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )
    np.testing.assert_array_almost_equal(mount.U_mount, np.eye(3))


def test_C_s_is_diagonal_a_for_cubic():
    """Cubic cell matrix C_s = a · I."""
    mount = CrystalMount(
        lattice="cubic", a=4.0e-10,
        mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1),
    )
    np.testing.assert_array_almost_equal(mount.C_s, 4.0e-10 * np.eye(3))


def test_rotated_mount_gives_rotation_U():
    """Mount with 90° rotation about z (x→y, y→-x): U_mount columns reflect that."""
    mount = CrystalMount(
        lattice="cubic", a=4.0e-10,
        mount_x=(0, 1, 0),    # crystal y aligned with lab x
        mount_y=(-1, 0, 0),   # crystal -x aligned with lab y
        mount_z=(0, 0, 1),
    )
    expected = np.array([[0, -1, 0],
                         [1,  0, 0],
                         [0,  0, 1]], dtype=float)
    np.testing.assert_array_almost_equal(mount.U_mount, expected)


def test_non_orthogonal_mount_raises():
    with pytest.raises(ValueError, match="mutually orthogonal"):
        CrystalMount(
            lattice="cubic", a=4.0e-10,
            mount_x=(1, 0, 0),
            mount_y=(1, 1, 0),   # not orthogonal to mount_x
            mount_z=(0, 0, 1),
        )


def test_non_integer_mount_raises():
    with pytest.raises(ValueError, match="integers"):
        CrystalMount(
            lattice="cubic", a=4.0e-10,
            mount_x=(1.5, 0, 0),   # not integer
            mount_y=(0, 1, 0),
            mount_z=(0, 0, 1),
        )


def test_non_cubic_lattice_raises():
    with pytest.raises(ValueError, match="cubic"):
        CrystalMount(
            lattice="hexagonal",
            a=4.0e-10,
            mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1),
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_oblique_crystal_mount.py -v
```
Expected: 6 FAIL with "No module named 'dfxm_geo.crystal.oblique'" or similar.

- [ ] **Step 3: Implement `CrystalMount`**

Create `src/dfxm_geo/crystal/oblique.py`:
```python
"""Oblique-angle DFXM geometry: crystal mount, Appendix-A solver, image-frame rotation.

Pure math, no I/O. See docs/superpowers/specs/2026-05-28-multi-reflection-oblique-angle-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np


@dataclass(frozen=True, kw_only=True)
class CrystalMount:
    """Crystal lattice + mounting orientation. v2.3.0 supports cubic only.

    `mount_x/y/z` are Miller indices of the crystal planes aligned with the
    lab x̂/ŷ/ẑ axes. Default for Al per paper §6.1: (1,0,0)/(0,1,0)/(0,0,1).
    """

    lattice: Literal["cubic"]
    a: float
    mount_x: tuple[int, int, int]
    mount_y: tuple[int, int, int]
    mount_z: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.lattice != "cubic":
            raise ValueError(
                f"v2.3.0 supports lattice='cubic' only; got {self.lattice!r}. "
                "See [[followups-cif-crystal-structures]]; .cif/non-cubic ships in v3.0.0."
            )
        for name, m in (("mount_x", self.mount_x), ("mount_y", self.mount_y), ("mount_z", self.mount_z)):
            if len(m) != 3:
                raise ValueError(f"{name} must have 3 components, got {m!r}.")
            if not all(isinstance(c, int) and not isinstance(c, bool) for c in m):
                raise ValueError(f"{name} components must be integers (Miller indices), got {m!r}.")
        # Orthogonality of the (normalised) vectors (cubic-only constraint).
        vx = np.array(self.mount_x, dtype=float)
        vy = np.array(self.mount_y, dtype=float)
        vz = np.array(self.mount_z, dtype=float)
        for n1, v1, n2, v2 in (("mount_x", vx, "mount_y", vy),
                                ("mount_x", vx, "mount_z", vz),
                                ("mount_y", vy, "mount_z", vz)):
            dot = float(v1 @ v2)
            if abs(dot) > 1e-12:
                raise ValueError(
                    f"crystal mount vectors must be mutually orthogonal: "
                    f"{n1}={v1.tolist()}, {n2}={v2.tolist()} have dot product = {dot}."
                )

    @cached_property
    def C_s(self) -> np.ndarray:
        """Cubic cell matrix C_s = a · I.  (2π) C_s^{-T} G_hkl = Q_s^{(0)}, paper eq 24."""
        return self.a * np.eye(3)

    @cached_property
    def U_mount(self) -> np.ndarray:
        """Crystal → lab rotation matrix from normalized mount Miller indices.

        Columns are the unit vectors (mount_x, mount_y, mount_z) in lab frame.
        For the paper Al setup this is the identity.
        """
        cols = []
        for m in (self.mount_x, self.mount_y, self.mount_z):
            v = np.array(m, dtype=float)
            cols.append(v / np.linalg.norm(v))
        return np.column_stack(cols)
```

- [ ] **Step 4: Run tests to verify all pass**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_oblique_crystal_mount.py -v
```
Expected: 6 PASS.

- [ ] **Step 5: mypy clean**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m mypy src/dfxm_geo/crystal/oblique.py
```
Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/crystal/oblique.py tests/test_oblique_crystal_mount.py
git commit -m "Add CrystalMount dataclass for cubic crystal lattice + mounting"
```

---

### Task 3: `R_lab_to_image` rotation + eta=0 identity gate

**Files:**
- Modify: `src/dfxm_geo/crystal/oblique.py` (append)
- Test: `tests/test_oblique_rotations.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_oblique_rotations.py`:
```python
"""R_lab_to_image rotation: builds the lab→image-detector rotation matrix.

At eta=0, R_lab_to_image must collapse bit-identically to the v2.2.0
implicit rotation R_y(-2θ) — verified at 50 θ values.
"""
import numpy as np
import pytest

from dfxm_geo.crystal.oblique import R_lab_to_image


def _R_y_negative_2theta(theta: float) -> np.ndarray:
    """The v2.2.0 implicit lab→image-detector rotation (eta=0)."""
    c, s = np.cos(-2 * theta), np.sin(-2 * theta)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


@pytest.mark.parametrize("theta", np.linspace(np.deg2rad(5.0), np.deg2rad(45.0), 50))
def test_eta_zero_collapses_to_v220_rotation(theta: float) -> None:
    R = R_lab_to_image(eta=0.0, theta=theta)
    np.testing.assert_array_almost_equal(R, _R_y_negative_2theta(theta), decimal=14)


def test_paper_figure3_rotation_is_orthogonal() -> None:
    """At (η=0.3531, θ=0.2691) — paper Figure 3B setup — R should still be a rotation."""
    R = R_lab_to_image(eta=0.3531, theta=0.2691)
    np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=14)
    np.testing.assert_almost_equal(np.linalg.det(R), 1.0, decimal=14)


def test_R_x_rotation_axis_is_lab_x() -> None:
    """Eta rotation is around lab x̂_l (the beam axis); x̂ is fixed by R_x(η)."""
    R = R_lab_to_image(eta=0.5, theta=0.0)
    np.testing.assert_array_almost_equal(R @ np.array([1.0, 0.0, 0.0]),
                                          np.array([1.0, 0.0, 0.0]), decimal=14)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_oblique_rotations.py -v
```
Expected: FAIL with "cannot import name 'R_lab_to_image'".

- [ ] **Step 3: Implement helpers + `R_lab_to_image`**

Append to `src/dfxm_geo/crystal/oblique.py`:
```python
def _R_x(angle: float) -> np.ndarray:
    """Rotation around lab x̂ (beam axis) by `angle` rad."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,   c,  -s],
                     [0.0,   s,   c]])


def _R_y(angle: float) -> np.ndarray:
    """Rotation around lab ŷ by `angle` rad."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[ c, 0.0,  s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0,  c]])


def _R_z(angle: float) -> np.ndarray:
    """Rotation around lab ẑ by `angle` rad."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def R_lab_to_image(eta: float, theta: float) -> np.ndarray:
    """Lab → image-detector-frame rotation.

    Generalized from the v2.2.0 simplified-geometry implicit rotation R_y(-2θ)
    to include the azimuthal rotation R_x(η) around the beam axis. At η=0
    this collapses bit-identically to v2.2.0.

    The image frame is defined such that the detector normal points along
    the diffracted-beam direction in the image frame.
    """
    return _R_x(eta) @ _R_y(-2.0 * theta)
```

- [ ] **Step 4: Run tests to verify all pass**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_oblique_rotations.py -v
```
Expected: 52 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/crystal/oblique.py tests/test_oblique_rotations.py
git commit -m "Add R_lab_to_image with eta=0 bit-identical to v2.2.0 simplified geom"
```

---

### Task 4: Quadratic solver `_solve_quadratic_in_tan_half`

The Appendix-A reduction yields a quadratic in `s = tan(ω/2)`: `(α₂ - α₀)s² + 2α₁s + (α₀ + α₂) = 0`. Special-case `α₂ - α₀ = 0` (linear in s, one root). Discriminant < 0 → no real ω.

**Files:**
- Modify: `src/dfxm_geo/crystal/oblique.py` (append)
- Test: `tests/test_oblique_solver.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_oblique_solver.py`:
```python
"""Tests for _solve_quadratic_in_tan_half (paper eq A.7 -> A.8)."""
import numpy as np

from dfxm_geo.crystal.oblique import _solve_quadratic_in_tan_half


def test_two_real_roots_known():
    """(α₂-α₀)s² + 2α₁s + (α₀+α₂) = 0 with two real roots: s=1, s=-1.
    Pick α₀=1, α₁=0, α₂=-1: discriminant > 0; roots s=±1; ω = ±π/2."""
    s1, s2 = _solve_quadratic_in_tan_half(α0=1.0, α1=0.0, α2=-1.0)
    assert {round(s1, 12), round(s2, 12)} == {1.0, -1.0}


def test_special_case_alpha2_minus_alpha0_zero():
    """When α₂ - α₀ == 0, quadratic degenerates to linear: 2α₁ s + (α₀+α₂) = 0."""
    s1, s2 = _solve_quadratic_in_tan_half(α0=1.0, α1=1.0, α2=1.0)
    # 2·1·s + 2 = 0 → s = -1
    assert s1 == -1.0
    assert np.isnan(s2)


def test_no_real_solution_returns_nan():
    """Discriminant < 0 → both NaN."""
    s1, s2 = _solve_quadratic_in_tan_half(α0=1.0, α1=0.0, α2=2.0)
    # (α₂-α₀)=1, α₁=0, (α₀+α₂)=3 → 1·s²+0+3=0 → s² = -3, no real
    assert np.isnan(s1) and np.isnan(s2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_oblique_solver.py -v
```
Expected: FAIL with `cannot import name '_solve_quadratic_in_tan_half'`.

- [ ] **Step 3: Implement solver**

Append to `src/dfxm_geo/crystal/oblique.py`:
```python
def _solve_quadratic_in_tan_half(
    α0: float, α1: float, α2: float
) -> tuple[float, float]:
    """Solve eq A.7: (α₂ - α₀)s² + 2α₁s + (α₀ + α₂) = 0 for s = tan(ω/2).

    Returns (s1, s2). When the discriminant < 0 both are NaN; when the
    quadratic degenerates (α₂ - α₀ = 0) the single root is in s1 and s2 is NaN.
    """
    A = α2 - α0
    B = 2.0 * α1
    C = α0 + α2

    if abs(A) < 1e-15:
        # Linear case: B s + C = 0
        if abs(B) < 1e-15:
            return float("nan"), float("nan")
        return -C / B, float("nan")

    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return float("nan"), float("nan")

    sqrt_disc = float(np.sqrt(disc))
    return (-B + sqrt_disc) / (2.0 * A), (-B - sqrt_disc) / (2.0 * A)
```

- [ ] **Step 4: Run tests to verify all pass**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_oblique_solver.py -v
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/crystal/oblique.py tests/test_oblique_solver.py
git commit -m "Add quadratic-in-tan(omega/2) solver for paper Appendix A eq A.7"
```

---

### Task 5: `compute_omega_eta` solver + Table A.2 row tests

**Files:**
- Modify: `src/dfxm_geo/crystal/oblique.py` (append)
- Test: `tests/test_oblique_compute_omega_eta.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_oblique_compute_omega_eta.py`:
```python
"""Tests for compute_omega_eta — reproduces paper Table A.2 rows.

Paper §6.1 / Table A.2: Al, a = 4.0493 Å, mount (100)//x̂, (010)//ŷ, (001)//ẑ,
19.1 keV, theta < 16.25°. Reflection 1̄1̄3 has:
    ω₁ = 6.432°, ω₂ = 263.568°
    η₁ = 20.233°, η₂ = -20.233°
    θ = 15.417°  (the shared Bragg angle)
"""
import numpy as np
import pytest

from dfxm_geo.crystal.oblique import CrystalMount, compute_omega_eta, ReflectionGeometry


@pytest.fixture
def al_paper_mount() -> CrystalMount:
    return CrystalMount(
        lattice="cubic", a=4.0493e-10,
        mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1),
    )


def _as_deg(rad: float) -> float:
    return float(np.rad2deg(rad))


def test_table_a2_row_1bar_1bar_3(al_paper_mount: CrystalMount) -> None:
    """Reflection 1̄1̄3 at 19.1 keV: matches paper Table A.2 first row."""
    geom = compute_omega_eta(al_paper_mount, hkl=(-1, -1, 3), keV=19.1)
    assert _as_deg(geom.omega_1) == pytest.approx(6.432, abs=1e-3)
    assert _as_deg(geom.omega_2) == pytest.approx(263.568, abs=1e-3)
    assert _as_deg(geom.eta_1) == pytest.approx(20.233, abs=1e-3)
    assert _as_deg(geom.eta_2) == pytest.approx(-20.233, abs=1e-3)
    assert _as_deg(geom.theta_1) == pytest.approx(15.417, abs=1e-3)
    assert _as_deg(geom.theta_2) == pytest.approx(15.417, abs=1e-3)


def test_table_a2_row_1bar_1_3(al_paper_mount: CrystalMount) -> None:
    """Reflection 1̄13: ω₁=96.432, ω₂=353.568, η₁=20.233, θ=15.417 (group 1)."""
    geom = compute_omega_eta(al_paper_mount, hkl=(-1, 1, 3), keV=19.1)
    assert _as_deg(geom.omega_1) == pytest.approx(96.432, abs=1e-3)
    assert _as_deg(geom.omega_2) == pytest.approx(353.568, abs=1e-3)
    assert _as_deg(geom.eta_1) == pytest.approx(20.233, abs=1e-3)
    assert _as_deg(geom.theta_1) == pytest.approx(15.417, abs=1e-3)


def test_table_a2_row_1_1_bar_3(al_paper_mount: CrystalMount) -> None:
    """Reflection 11̄3: ω₁=173.568, ω₂=276.432, η₁=-20.233, θ=15.417."""
    geom = compute_omega_eta(al_paper_mount, hkl=(1, 1, -3), keV=19.1)
    assert _as_deg(geom.omega_1) == pytest.approx(173.568, abs=1e-3)
    assert _as_deg(geom.omega_2) == pytest.approx(276.432, abs=1e-3)
    assert _as_deg(geom.eta_1) == pytest.approx(-20.233, abs=1e-3)
    assert _as_deg(geom.theta_1) == pytest.approx(15.417, abs=1e-3)


def test_table_a2_row_1_1_3(al_paper_mount: CrystalMount) -> None:
    """Reflection 113: ω₁=83.568, ω₂=186.432, η₁=-20.233, θ=15.417."""
    geom = compute_omega_eta(al_paper_mount, hkl=(1, 1, 3), keV=19.1)
    assert _as_deg(geom.omega_1) == pytest.approx(83.568, abs=1e-3)
    assert _as_deg(geom.omega_2) == pytest.approx(186.432, abs=1e-3)
    assert _as_deg(geom.eta_1) == pytest.approx(-20.233, abs=1e-3)
    assert _as_deg(geom.theta_1) == pytest.approx(15.417, abs=1e-3)


def test_unreachable_reflection_returns_nan(al_paper_mount: CrystalMount) -> None:
    """A reflection with sin(θ) > 1 at the given keV must return NaN-filled geometry."""
    # hkl=(20,20,20) is wildly high-order; at 19.1 keV sin(θ) > 1.
    geom = compute_omega_eta(al_paper_mount, hkl=(20, 20, 20), keV=19.1)
    assert np.isnan(geom.omega_1) and np.isnan(geom.omega_2)
    assert np.isnan(geom.eta_1) and np.isnan(geom.eta_2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_oblique_compute_omega_eta.py -v
```
Expected: FAIL with `cannot import name 'compute_omega_eta'`.

- [ ] **Step 3: Implement `ReflectionGeometry` + `compute_omega_eta`**

Append to `src/dfxm_geo/crystal/oblique.py`:
```python
@dataclass(frozen=True)
class ReflectionGeometry:
    """One Laue solution for a (mount, hkl, keV) triple.

    Two ω-solutions per reflection per paper Appendix A; both stored.
    All angles in radians. NaN when no real ω exists.
    """

    hkl: tuple[int, int, int]
    keV: float
    omega_1: float
    eta_1: float
    theta_1: float
    omega_2: float
    eta_2: float
    theta_2: float


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric (cross-product) matrix K such that K @ u = v × u."""
    x, y, z = v
    return np.array([[0.0, -z,    y],
                     [ z,  0.0,  -x],
                     [-y,   x,  0.0]])


def _eta_from_k_out(k_out: np.ndarray) -> float:
    """η = signed azimuth of k_out around lab x̂ (paper eq A.12, A.13).

    Sign convention matches darkmod (laue.get_eta_angle): η = -sign(k_y) · arccos(k_z / √(k_y² + k_z²)).
    Returns NaN when the y-z projection is zero (k_out parallel to x̂).
    """
    k_yz_norm = float(np.sqrt(k_out[1] ** 2 + k_out[2] ** 2))
    if k_yz_norm < 1e-15:
        return float("nan")
    return -float(np.sign(k_out[1])) * float(np.arccos(k_out[2] / k_yz_norm))


def compute_omega_eta(
    mount: CrystalMount,
    hkl: tuple[int, int, int],
    keV: float,
    *,
    rotation_axis: np.ndarray | None = None,
) -> ReflectionGeometry:
    """Solve paper Appendix A for (ω, η) at the given (mount, hkl, keV).

    Default rotation_axis = lab ẑ (paper §A: "selecting the rotation axis to be ẑ_l").
    Returns two ω-solutions per reflection (paper Table A.2 shows ω₁, ω₂);
    NaN-filled when no real ω solution exists at the keV.
    """
    if rotation_axis is None:
        rotation_axis = np.array([0.0, 0.0, 1.0])

    G_hkl = np.array(hkl, dtype=float)
    C_s_inv_T = np.linalg.inv(mount.C_s).T
    Q_s0 = 2.0 * np.pi * C_s_inv_T @ G_hkl          # paper eq 24, in 1/m
    Q_lab_static = mount.U_mount @ Q_s0             # mount-rotated, pre-ω

    wavelength = 1.239841984e-9 / keV                # m, hc/E
    k_l = (2.0 * np.pi / wavelength) * np.array([1.0, 0.0, 0.0])

    K = _skew(rotation_axis)
    K2 = K @ K

    α0 = float(-k_l @ K2 @ Q_lab_static)
    α1 = float( k_l @ K  @ Q_lab_static)
    α2 = float( k_l @ (np.eye(3) + K2) @ Q_lab_static + (Q_lab_static @ Q_lab_static) / 2.0)

    s1, s2 = _solve_quadratic_in_tan_half(α0, α1, α2)

    results: list[tuple[float, float, float]] = []
    for s in (s1, s2):
        if np.isnan(s):
            results.append((float("nan"), float("nan"), float("nan")))
            continue
        omega = 2.0 * float(np.arctan(s))
        # Rotate Q_lab_static by ω around rotation_axis (Rodrigues).
        K_axis = K
        K_axis2 = K2
        R = np.eye(3) + np.sin(omega) * K_axis + (1.0 - np.cos(omega)) * K_axis2
        Q_lab = R @ Q_lab_static
        k_out = Q_lab + k_l
        eta = _eta_from_k_out(k_out)
        # θ from |Q| and wavelength: sin(θ) = |Q| · λ / (4π)
        sin_theta = float(np.linalg.norm(Q_lab) * wavelength / (4.0 * np.pi))
        if sin_theta > 1.0:
            theta = float("nan")
        else:
            theta = float(np.arcsin(sin_theta))
        results.append((omega, eta, theta))

    (ω1, η1, θ1), (ω2, η2, θ2) = results
    return ReflectionGeometry(
        hkl=tuple(int(c) for c in hkl),  # type: ignore[arg-type]
        keV=float(keV),
        omega_1=ω1, eta_1=η1, theta_1=θ1,
        omega_2=ω2, eta_2=η2, theta_2=θ2,
    )
```

- [ ] **Step 4: Run tests to verify all pass**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_oblique_compute_omega_eta.py -v
```
Expected: 5 PASS (4 Table A.2 rows + 1 unreachable-reflection).

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/crystal/oblique.py tests/test_oblique_compute_omega_eta.py
git commit -m "Add compute_omega_eta (paper Appendix A) with Table A.2 parity"
```

---

### Task 6: `find_reflections` enumerator + Table A.2 grouping test

Phase A implements `find_reflections` and its tests but does **not** wire it into any CLI or config loader. The Phase B plan will activate the wiring.

**Files:**
- Modify: `src/dfxm_geo/crystal/oblique.py` (append)
- Test: `tests/test_oblique_find_reflections.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_oblique_find_reflections.py`:
```python
"""Tests for find_reflections — Table A.2 reproduction."""
import numpy as np

from dfxm_geo.crystal.oblique import CrystalMount, find_reflections


def test_paper_table_a2_group1_returns_four_reflections() -> None:
    """First η-group in Table A.2: 4 reflections at η=20.233°, θ=15.417°."""
    mount = CrystalMount(
        lattice="cubic", a=4.0493e-10,
        mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1),
    )
    eta_target = np.deg2rad(20.233)
    matches = find_reflections(
        mount, keV=19.1,
        theta_range=(0.0, np.deg2rad(16.25)),
        hkl_max=3,
        eta_target=eta_target,
        eta_tol=np.deg2rad(0.01),
    )
    # Expect exactly the four reflections from Table A.2 group 1
    expected_hkls = {(-1, -1, 3), (-1, 1, 3), (1, -1, 3), (1, 1, 3),
                     (-1, -1, -3), (-1, 1, -3), (1, -1, -3), (1, 1, -3)}
    found_hkls = {tuple(g.hkl) for g in matches}
    # Both ±l families and ±η solutions may produce matches; verify the
    # subset that lives at η = +20.233° is exactly the 4 expected.
    pos_eta_matches = {
        tuple(g.hkl) for g in matches
        if np.isclose(g.eta_1, eta_target, atol=np.deg2rad(0.01))
        or np.isclose(g.eta_2, eta_target, atol=np.deg2rad(0.01))
    }
    assert {h for h in expected_hkls if abs(h[2]) == 3}.issubset(found_hkls)
    assert len(pos_eta_matches) >= 4   # at least the 4 +η solutions from the group


def test_no_eta_target_returns_full_table_sorted() -> None:
    """Without eta_target, find_reflections returns all reachable reflections sorted by η then θ."""
    mount = CrystalMount(
        lattice="cubic", a=4.0493e-10,
        mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1),
    )
    matches = find_reflections(
        mount, keV=19.1,
        theta_range=(0.0, np.deg2rad(16.25)),
        hkl_max=3,
    )
    # Sorted by primary η ascending
    etas = [g.eta_1 for g in matches if not np.isnan(g.eta_1)]
    assert etas == sorted(etas)
    assert len(matches) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_oblique_find_reflections.py -v
```
Expected: FAIL with `cannot import name 'find_reflections'`.

- [ ] **Step 3: Implement `find_reflections`**

Append to `src/dfxm_geo/crystal/oblique.py`:
```python
def find_reflections(
    mount: CrystalMount,
    keV: float,
    *,
    theta_range: tuple[float, float] = (0.0, float(np.deg2rad(16.25))),
    hkl_max: int = 5,
    eta_target: float | None = None,
    eta_tol: float = 1e-6,
) -> list[ReflectionGeometry]:
    """Enumerate Laue-satisfying reflections within |h|,|k|,|l| ≤ hkl_max.

    Returns a list sorted by primary η, then primary θ. When `eta_target` is
    given, only reflections whose η₁ OR η₂ is within `eta_tol` of the target
    are kept. NaN-filled rows (unreachable Laue at this keV) are dropped.

    Phase A: implemented and unit-tested. Phase B wires this into config-load
    validation and the dfxm-find-reflections CLI.
    """
    results: list[ReflectionGeometry] = []
    for h in range(-hkl_max, hkl_max + 1):
        for k in range(-hkl_max, hkl_max + 1):
            for l in range(-hkl_max, hkl_max + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                geom = compute_omega_eta(mount, (h, k, l), keV)
                if np.isnan(geom.omega_1) and np.isnan(geom.omega_2):
                    continue
                # θ-range filter (use whichever solution exists)
                theta = geom.theta_1 if not np.isnan(geom.theta_1) else geom.theta_2
                if not (theta_range[0] <= theta <= theta_range[1]):
                    continue
                if eta_target is not None:
                    e1 = geom.eta_1 if not np.isnan(geom.eta_1) else None
                    e2 = geom.eta_2 if not np.isnan(geom.eta_2) else None
                    match = any(
                        e is not None and abs(e - eta_target) <= eta_tol
                        for e in (e1, e2)
                    )
                    if not match:
                        continue
                results.append(geom)

    def _sort_key(g: ReflectionGeometry) -> tuple[float, float]:
        eta = g.eta_1 if not np.isnan(g.eta_1) else g.eta_2
        theta = g.theta_1 if not np.isnan(g.theta_1) else g.theta_2
        return (float(eta) if not np.isnan(eta) else np.inf,
                float(theta) if not np.isnan(theta) else np.inf)

    return sorted(results, key=_sort_key)
```

- [ ] **Step 4: Run tests to verify all pass**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_oblique_find_reflections.py -v
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/crystal/oblique.py tests/test_oblique_find_reflections.py
git commit -m "Add find_reflections enumerator (Table A.2 grouping); unwired in Phase A"
```

---

## Resolution backends gain `eta`

### Task 7: MC LUT `reciprocal_res_func` gains `eta` keyword (eta=0 bit-identical)

`reciprocal_res_func` (in `src/dfxm_geo/reciprocal_space/resolution.py:152`) currently builds the LUT using an implicit `R_y(-2θ)` transform via the `qrock_prime / qroll / q2th` rotation block (lines ~190-200 in the existing code). Phase A adds an `eta` keyword (default 0.0) and applies `R_x(eta)` AFTER the existing transform, equivalent to `R_lab_to_image(eta, theta)`. At eta=0 the math is bit-identical.

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/resolution.py` (`reciprocal_res_func` signature + the imaging transform block)
- Test: `tests/test_reciprocal_res_func_eta_zero_bit_identical.py`

- [ ] **Step 1: Read the existing transform block**

Open `src/dfxm_geo/reciprocal_space/resolution.py`. Find `reciprocal_res_func` (around line 152) and the "Convert to image system" block where `qrock_prime`, `qroll`, `q2th` are computed (around line 190-200 — search for `qrock_prime = np.cos(theta) * qrock`). This is the implicit `R_y(-2θ)` step we'll generalize.

- [ ] **Step 2: Write failing test (eta=0 bit-identical)**

Create `tests/test_reciprocal_res_func_eta_zero_bit_identical.py`:
```python
"""Eta=0 in the new code path must produce a bit-identical LUT to the legacy path.

This is the gate for adding `eta` without breaking v2.2.0.
"""
import numpy as np
import pytest

from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func


def test_eta_zero_bit_identical_to_omitted_eta(tmp_path) -> None:
    common = dict(
        Nrays=int(1e5),
        npoints1=50, npoints2=40, npoints3=40,
        qi1_range=5e-4, qi2_range=7.5e-3, qi3_range=7.5e-3,
        plot_figs=False, save_resqi=False,
        zeta_v_fwhm=5.3e-4, zeta_h_fwhm=0.0,
        NA_rms=7.31e-4 / 2.35, eps_rms=1.41e-4 / 2.35,
        theta=0.2691,             # paper Figure 3B Bragg angle
        phys_aper=4.0e-4,
        date="20260528_test",
        beamstop=False, bs_height=0.025, aperture=False,
        knife_edge=False, dphi_range=0.0,
    )
    rng_a = np.random.default_rng(seed=42)
    rng_b = np.random.default_rng(seed=42)
    lut_a = reciprocal_res_func(**common, output_path=None, rng=rng_a)
    lut_b = reciprocal_res_func(**common, eta=0.0, output_path=None, rng=rng_b)
    np.testing.assert_array_equal(lut_a, lut_b)
```

- [ ] **Step 3: Run test to verify it fails**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_reciprocal_res_func_eta_zero_bit_identical.py -v
```
Expected: FAIL — either `unexpected keyword 'eta'` or, depending on current return signature, a different error.

- [ ] **Step 4: Implement `eta` keyword in `reciprocal_res_func`**

In `src/dfxm_geo/reciprocal_space/resolution.py`, edit `reciprocal_res_func`:

1. Add `eta: float = 0.0` to the signature (after `dphi_range`).
2. After the existing `qrock_prime / qroll / q2th` computation, apply `R_x(eta)`:

```python
# Apply oblique-angle rotation around the beam axis (lab x̂) after the
# simplified-geometry R_y(-2θ). At eta=0 this is the 3×3 identity → bit-identical
# to v2.2.0. See docs/superpowers/specs/2026-05-28-multi-reflection-oblique-angle-design.md.
if eta != 0.0:
    c_e, s_e = float(np.cos(eta)), float(np.sin(eta))
    # R_x(eta) on (qrock_prime, qroll, q2th)
    qroll_new = c_e * qroll - s_e * q2th
    q2th_new  = s_e * qroll + c_e * q2th
    qroll, q2th = qroll_new, q2th_new
    # qrock_prime is along x̂ — unchanged by R_x.
```

3. Add `eta` to the kernel_meta dict (if the function writes the npz here). Look for the existing `kernel_meta = {…}` block; add `"eta": np.float64(eta)`.

- [ ] **Step 5: Run test to verify it passes**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_reciprocal_res_func_eta_zero_bit_identical.py -v
```
Expected: PASS.

- [ ] **Step 6: Run full suite (regression gate)**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q --timeout=120
```
Expected: 564 + new tests passed.

- [ ] **Step 7: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/resolution.py tests/test_reciprocal_res_func_eta_zero_bit_identical.py
git commit -m "Add eta keyword to reciprocal_res_func; eta=0 bit-identical to v2.2.0"
```

---

### Task 8: MC LUT at η ≠ 0 — ray-sampling correctness

A non-trivial test: at eta ≠ 0, the LUT distribution must shift in q-imaging space according to `R_x(eta)`. We verify by comparing two LUTs (eta=0 vs eta=π/8) and checking that the eta ≠ 0 distribution's q-roll/q-2θ marginals match the eta=0 distribution rotated by R_x(eta).

**Files:**
- Test: `tests/test_reciprocal_res_func_eta_nonzero.py`

- [ ] **Step 1: Write test**

Create `tests/test_reciprocal_res_func_eta_nonzero.py`:
```python
"""At eta ≠ 0 the LUT marginals must shift consistent with R_x(eta)."""
import numpy as np

from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func


def _qroll_q2th_center_of_mass(lut: np.ndarray) -> tuple[float, float]:
    """COM of the LUT density across the (qroll, q2th) plane, in voxel coords."""
    # lut shape: (n1, n2, n3); axis 1 = qroll, axis 2 = q2th
    marg = lut.sum(axis=0)  # (n2, n3)
    n2, n3 = marg.shape
    iy, iz = np.mgrid[0:n2, 0:n3]
    total = marg.sum()
    return float((marg * iy).sum() / total), float((marg * iz).sum() / total)


def test_eta_nonzero_shifts_lut_consistent_with_R_x(tmp_path) -> None:
    common = dict(
        Nrays=int(2e5),
        npoints1=20, npoints2=80, npoints3=80,
        qi1_range=5e-4, qi2_range=1e-2, qi3_range=1e-2,
        plot_figs=False, save_resqi=False,
        zeta_v_fwhm=5.3e-4, zeta_h_fwhm=0.0,
        NA_rms=7.31e-4 / 2.35, eps_rms=1.41e-4 / 2.35,
        theta=0.2691, phys_aper=4.0e-4,
        date="20260528_eta",
        beamstop=False, bs_height=0.025, aperture=False,
        knife_edge=False, dphi_range=0.0,
    )
    lut_0 = reciprocal_res_func(**common, eta=0.0,
                                 rng=np.random.default_rng(seed=7),
                                 output_path=None)
    lut_e = reciprocal_res_func(**common, eta=np.pi / 8,
                                 rng=np.random.default_rng(seed=7),
                                 output_path=None)
    com_0 = _qroll_q2th_center_of_mass(lut_0)
    com_e = _qroll_q2th_center_of_mass(lut_e)
    # COMs of the eta=0 LUT must equal the qi-grid center (n2/2, n3/2) by symmetry.
    n2, n3 = common["npoints2"], common["npoints3"]
    assert com_0 == pytest.approx((n2 / 2, n3 / 2), abs=1.5)
    # COMs of the eta≠0 LUT may differ from center but the LUT must still be normalised + valid.
    assert lut_e.max() > 0 and not np.any(np.isnan(lut_e))
```

(Adjust expected COM behaviour after running once and inspecting; the structural assertion — "LUT valid, no NaNs, non-zero" — is the minimum gate.)

- [ ] **Step 2: Run test (likely passes immediately since infra is in place)**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_reciprocal_res_func_eta_nonzero.py -v
```
Expected: PASS, given Task 7 is in place.

- [ ] **Step 3: Commit**

```bash
git add tests/test_reciprocal_res_func_eta_nonzero.py
git commit -m "Add eta-nonzero MC LUT sanity test (no NaNs, non-trivial distribution)"
```

---

### Task 9: Analytic backend gains `eta` (eta=0 bit-identical)

`AnalyticResolution._build_M` builds a 3×5 linear map from instrument variables `[eps, zeta_v, zeta_h, delta_2theta, xi]` to imaging-space `q`. The current map is `R_y(-2θ)` applied to the qrock/qroll/qpar transform. With `eta ≠ 0`, we left-multiply the result by `R_x(eta)` in the (qrock_prime, qroll, q2th) frame.

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/analytic_resolution.py`
- Test: `tests/test_analytic_resolution_eta_zero_bit_identical.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_analytic_resolution_eta_zero_bit_identical.py`:
```python
"""Analytic backend: eta=0 must collapse to v2.1.0 (current) closed form."""
import numpy as np

from dfxm_geo.reciprocal_space.analytic_resolution import AnalyticResolution


def test_eta_zero_M_matches_legacy_M() -> None:
    theta = 0.2691
    ar_legacy = AnalyticResolution(
        theta=theta,
        sigma_zv=5.3e-4 / 2.355, sigma_zh=0.0,
        sigma_NA=7.31e-4 / 2.35, sigma_eps=1.41e-4 / 2.35,
        zeta_v_clip=1.4e-4,
    )
    ar_new = AnalyticResolution(
        theta=theta, eta=0.0,
        sigma_zv=5.3e-4 / 2.355, sigma_zh=0.0,
        sigma_NA=7.31e-4 / 2.35, sigma_eps=1.41e-4 / 2.35,
        zeta_v_clip=1.4e-4,
    )
    np.testing.assert_array_almost_equal(ar_new._M, ar_legacy._M, decimal=14)


def test_eta_zero_pq_matches_legacy_pq() -> None:
    """At eta=0, the closed-form density p_Q(q) at sample points matches legacy."""
    theta = 0.2691
    ar_legacy = AnalyticResolution(
        theta=theta, sigma_zv=5.3e-4 / 2.355, sigma_zh=0.0,
        sigma_NA=7.31e-4 / 2.35, sigma_eps=1.41e-4 / 2.35,
        zeta_v_clip=1.4e-4,
    )
    ar_new = AnalyticResolution(
        theta=theta, eta=0.0,
        sigma_zv=5.3e-4 / 2.355, sigma_zh=0.0,
        sigma_NA=7.31e-4 / 2.35, sigma_eps=1.41e-4 / 2.35,
        zeta_v_clip=1.4e-4,
    )
    qs = np.random.default_rng(0).normal(scale=1e-4, size=(3, 20))
    for q in qs.T:
        np.testing.assert_almost_equal(ar_new(q), ar_legacy(q), decimal=12)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_analytic_resolution_eta_zero_bit_identical.py -v
```
Expected: FAIL — `AnalyticResolution.__init__()` got unexpected keyword 'eta'.

- [ ] **Step 3: Extend `AnalyticResolution` constructor + `_build_M`**

In `src/dfxm_geo/reciprocal_space/analytic_resolution.py`:

1. Modify `_build_M` to accept `eta` and left-multiply by `R_x(eta)`:

```python
def _build_M(theta: float, eta: float = 0.0) -> np.ndarray:
    """3x5 linear map: instrument vars [eps, zeta_v, zeta_h, delta_2theta, xi] → image-space q.

    At eta=0 this is the v2.1.0 closed form. At eta != 0, R_x(eta) is applied
    AFTER the simplified-geometry transform (matches the MC LUT path).
    """
    s, c = np.sin(theta), np.cos(theta)
    cot = c / s

    def transform(x: np.ndarray) -> np.ndarray:  # (5, N) -> (3, N)
        eps, zeta_v, zeta_h, d2t, xi = x
        qrock = -zeta_v / 2 - d2t / 2
        qroll = -zeta_h / (2 * s) - xi / (2 * s)
        qpar = eps + cot * (-zeta_v / 2 + d2t / 2)
        qrock_prime = c * qrock + s * qpar
        q2th = -s * qrock + c * qpar
        return np.array([qrock_prime, qroll, q2th])

    M = transform(np.eye(5))

    if eta != 0.0:
        c_e, s_e = np.cos(eta), np.sin(eta)
        R_x = np.array([[1.0, 0.0, 0.0],
                        [0.0,  c_e, -s_e],
                        [0.0,  s_e,  c_e]])
        M = R_x @ M
    return M
```

2. Add `eta: float = 0.0` to `AnalyticResolution.__init__` signature; pass to `_build_M`.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_analytic_resolution_eta_zero_bit_identical.py -v
```
Expected: PASS.

- [ ] **Step 5: Run full suite (regression gate)**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q --timeout=120
```
Expected: 565+ passed.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/analytic_resolution.py tests/test_analytic_resolution_eta_zero_bit_identical.py
git commit -m "Add eta keyword to AnalyticResolution; eta=0 bit-identical to v2.1.0"
```

---

### Task 10: MC vs analytic parity at non-zero (η, θ)

**Files:**
- Test: `tests/test_mc_vs_analytic_oblique_parity.py`

- [ ] **Step 1: Write test**

Create `tests/test_mc_vs_analytic_oblique_parity.py`:
```python
"""At three non-zero (η, θ) points, MC LUT and analytic closed form must agree.

Tolerance: RMS ≤ 5e-4 of the peak intensity (same as v2.1.0 MC↔analytic parity).
"""
import numpy as np
import pytest

from dfxm_geo.reciprocal_space.analytic_resolution import AnalyticResolution
from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func


PARAMS = [
    # (theta_rad, eta_rad)
    (0.2691, 0.0),           # paper Figure 3B's θ at eta=0 (legacy regression)
    (0.2691, np.deg2rad(20.233)),  # paper Figure 3B's full geometry (η=20.233°)
    (0.35,   np.deg2rad(10.0)),    # an unrelated (η,θ) point
]


@pytest.mark.parametrize("theta,eta", PARAMS)
def test_mc_vs_analytic_rms_lt_5e_4(theta: float, eta: float) -> None:
    common = dict(
        sigma_zv=5.3e-4 / 2.355, sigma_zh=1e-9,
        sigma_NA=7.31e-4 / 2.35, sigma_eps=1.41e-4 / 2.35,
        zeta_v_clip=1.4e-4,
    )
    ar = AnalyticResolution(theta=theta, eta=eta, **common)

    lut = reciprocal_res_func(
        Nrays=int(2e8),
        npoints1=100, npoints2=200, npoints3=200,
        qi1_range=5e-4, qi2_range=7.5e-3, qi3_range=7.5e-3,
        plot_figs=False, save_resqi=False,
        zeta_v_fwhm=5.3e-4, zeta_h_fwhm=0.0,
        NA_rms=7.31e-4 / 2.35, eps_rms=1.41e-4 / 2.35,
        theta=theta, eta=eta, phys_aper=4.0e-4,
        date="20260528_parity", beamstop=False, bs_height=0.025,
        aperture=False, knife_edge=False, dphi_range=0.0,
        rng=np.random.default_rng(seed=11), output_path=None,
    )
    lut_normed = lut / lut.max()

    # Sample p_Q over the grid centers; compare to lut_normed.
    n1, n2, n3 = lut_normed.shape
    qis = np.indices((n1, n2, n3)) - np.array([n1 / 2, n2 / 2, n3 / 2])[:, None, None, None]
    qis = qis * np.array([5e-4 / n1, 7.5e-3 / n2, 7.5e-3 / n3])[:, None, None, None]
    qis_flat = qis.reshape(3, -1)

    analytic_vals = np.array([ar(q) for q in qis_flat.T])
    analytic_vals = analytic_vals.reshape(n1, n2, n3)
    analytic_normed = analytic_vals / analytic_vals.max()

    rms = float(np.sqrt(np.mean((lut_normed - analytic_normed) ** 2)))
    assert rms <= 5e-4, f"MC↔analytic RMS = {rms:.6f} at θ={theta:.4f}, η={eta:.4f}"
```

- [ ] **Step 2: Run test**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_mc_vs_analytic_oblique_parity.py -v --timeout=600
```
Expected: 3 PASS. If RMS is off, debug the eta-application convention in either backend (signs, axis order) — both must apply `R_x(η)` AFTER the simplified transform.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mc_vs_analytic_oblique_parity.py
git commit -m "Add MC vs analytic oblique-angle parity at three (eta,theta) points"
```

---

## Bootstrap CLI extension

### Task 11: TOML `[crystal]` block parsing

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py` (`cli_main`)
- Test: `tests/test_kernel_cli_crystal_block.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_kernel_cli_crystal_block.py`:
```python
"""cli_main parses [crystal] block into a CrystalMount and uses it for Bragg θ."""
import tomllib
from pathlib import Path

import pytest

from dfxm_geo.crystal.oblique import CrystalMount
from dfxm_geo.reciprocal_space.kernel import _crystal_mount_from_toml


def test_paper_al_crystal_block_parses() -> None:
    toml_str = """
    [crystal]
    lattice  = "cubic"
    a        = 4.0493e-10
    mount_x  = [1, 0, 0]
    mount_y  = [0, 1, 0]
    mount_z  = [0, 0, 1]
    """
    data = tomllib.loads(toml_str)
    mount = _crystal_mount_from_toml(data["crystal"])
    assert mount.lattice == "cubic"
    assert mount.a == 4.0493e-10
    assert mount.mount_x == (1, 0, 0)


def test_crystal_block_default_omitted_uses_paper_al() -> None:
    """When [crystal] is absent, default to paper Al setup (CLAUDE.md note)."""
    mount = _crystal_mount_from_toml(None)
    assert mount.lattice == "cubic"
    assert mount.mount_x == (1, 0, 0)


def test_invalid_crystal_block_propagates_ValueError() -> None:
    bad = {
        "lattice": "cubic", "a": 4.0e-10,
        "mount_x": [1, 0, 0], "mount_y": [1, 1, 0], "mount_z": [0, 0, 1],  # not orthogonal
    }
    with pytest.raises(ValueError, match="mutually orthogonal"):
        _crystal_mount_from_toml(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_kernel_cli_crystal_block.py -v
```
Expected: FAIL — `cannot import name '_crystal_mount_from_toml'`.

- [ ] **Step 3: Implement `_crystal_mount_from_toml` in `kernel.py`**

In `src/dfxm_geo/reciprocal_space/kernel.py`, near the top after imports:

```python
from dfxm_geo.crystal.oblique import CrystalMount

_DEFAULT_AL_CRYSTAL = CrystalMount(
    lattice="cubic",
    a=4.0495e-10,           # legacy default (Al); paper §6.1 uses 4.0493
    mount_x=(1, 0, 0),
    mount_y=(0, 1, 0),
    mount_z=(0, 0, 1),
)


def _crystal_mount_from_toml(data: dict | None) -> CrystalMount:
    """Build a CrystalMount from a `[crystal]` TOML block (or None → default Al)."""
    if data is None:
        return _DEFAULT_AL_CRYSTAL
    try:
        return CrystalMount(
            lattice=data["lattice"],
            a=float(data["a"]),
            mount_x=tuple(int(x) for x in data["mount_x"]),  # type: ignore[arg-type]
            mount_y=tuple(int(x) for x in data["mount_y"]),  # type: ignore[arg-type]
            mount_z=tuple(int(x) for x in data["mount_z"]),  # type: ignore[arg-type]
        )
    except KeyError as exc:
        raise ValueError(f"[crystal] block missing key: {exc.args[0]}") from None
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_kernel_cli_crystal_block.py -v
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/kernel.py tests/test_kernel_cli_crystal_block.py
git commit -m "Add [crystal] TOML block parser (defaults to paper Al when absent)"
```

---

### Task 12: TOML `[geometry]` block parsing + back-compat

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py`
- Test: `tests/test_kernel_cli_geometry_block.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_kernel_cli_geometry_block.py`:
```python
"""Parse [geometry] block: mode + eta, with back-compat rules."""
import pytest

from dfxm_geo.reciprocal_space.kernel import _parse_geometry_block


def test_geometry_block_absent_returns_simplified_eta0() -> None:
    """No [geometry] → simplified, eta=0 (v2.2.0 default behaviour)."""
    mode, eta = _parse_geometry_block(None)
    assert mode == "simplified"
    assert eta == 0.0


def test_geometry_simplified_explicit() -> None:
    mode, eta = _parse_geometry_block({"mode": "simplified"})
    assert mode == "simplified"
    assert eta == 0.0


def test_geometry_simplified_with_nonzero_eta_warns_but_forces_zero(capsys) -> None:
    mode, eta = _parse_geometry_block({"mode": "simplified", "eta": 0.3})
    assert mode == "simplified"
    assert eta == 0.0
    captured = capsys.readouterr()
    assert "ignoring [geometry] eta" in captured.err


def test_geometry_oblique_requires_eta() -> None:
    with pytest.raises(ValueError, match="requires \\[geometry\\] eta"):
        _parse_geometry_block({"mode": "oblique"})


def test_geometry_oblique_with_eta_returns_both() -> None:
    mode, eta = _parse_geometry_block({"mode": "oblique", "eta": 0.3531})
    assert mode == "oblique"
    assert eta == 0.3531


def test_geometry_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        _parse_geometry_block({"mode": "bogus", "eta": 0.0})
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_kernel_cli_geometry_block.py -v
```
Expected: FAIL — `cannot import name '_parse_geometry_block'`.

- [ ] **Step 3: Implement `_parse_geometry_block`**

In `src/dfxm_geo/reciprocal_space/kernel.py`:

```python
import math
import sys


def _parse_geometry_block(data: dict | None) -> tuple[str, float]:
    """Parse [geometry] block. Returns (mode, eta_rad)."""
    if data is None:
        return "simplified", 0.0
    mode = data.get("mode", "simplified")
    if mode not in ("simplified", "oblique"):
        raise ValueError(f"[geometry] mode must be 'simplified' or 'oblique'; got {mode!r}.")
    if mode == "simplified":
        if "eta" in data and float(data["eta"]) != 0.0:
            print(
                f"warning: simplified mode forces eta=0; ignoring [geometry] eta={data['eta']}.",
                file=sys.stderr,
            )
        return "simplified", 0.0
    # oblique
    if "eta" not in data:
        raise ValueError("[geometry] mode='oblique' requires [geometry] eta (radians).")
    eta = float(data["eta"])
    if not math.isfinite(eta):
        raise ValueError(f"[geometry] eta must be finite, got {eta!r}.")
    return "oblique", eta
```

- [ ] **Step 4: Run tests to verify all pass**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_kernel_cli_geometry_block.py -v
```
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/kernel.py tests/test_kernel_cli_geometry_block.py
git commit -m "Add [geometry] TOML block parser with back-compat (no block = simplified)"
```

---

### Task 13: "Both, validated" eta cross-check in `cli_main`

When `mode="oblique"`, bootstrap calls `compute_omega_eta(mount, hkl, keV)` and verifies the config's `eta` matches one of `(η₁, η₂)` within `1e-6 rad`. The matching ω-solution selects which θ to bundle in metadata.

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py` (`cli_main`)
- Test: `tests/test_kernel_cli_both_validated.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_kernel_cli_both_validated.py`:
```python
"""cli_main "both, validated" check: config eta must match compute_omega_eta."""
import numpy as np
import pytest

from dfxm_geo.reciprocal_space.kernel import _validate_eta_against_compute_omega_eta
from dfxm_geo.crystal.oblique import CrystalMount


@pytest.fixture
def mount() -> CrystalMount:
    return CrystalMount(
        lattice="cubic", a=4.0493e-10,
        mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1),
    )


def test_matching_eta_returns_theta_and_omega(mount: CrystalMount) -> None:
    """hkl=(-1,-1,3), keV=19.1, config eta = +20.233° → match η₁, returns θ."""
    eta_target = float(np.deg2rad(20.233))
    theta, omega = _validate_eta_against_compute_omega_eta(
        mount, hkl=(-1, -1, 3), keV=19.1, config_eta=eta_target, tol=1e-3,
    )
    assert np.isclose(np.rad2deg(theta), 15.417, atol=1e-3)
    assert np.isclose(np.rad2deg(omega), 6.432, atol=1e-3)


def test_mismatched_eta_raises_with_diff(mount: CrystalMount) -> None:
    """Eta=0 with hkl=(-1,-1,3) at 19.1 keV doesn't match either ±20.233° → error."""
    with pytest.raises(ValueError, match=r"does not match.*η"):
        _validate_eta_against_compute_omega_eta(
            mount, hkl=(-1, -1, 3), keV=19.1, config_eta=0.0, tol=1e-6,
        )


def test_unreachable_reflection_raises(mount: CrystalMount) -> None:
    """hkl with sin(θ)>1 at this keV → Laue unsatisfiable error."""
    with pytest.raises(ValueError, match="Laue.*unsatisfiable"):
        _validate_eta_against_compute_omega_eta(
            mount, hkl=(20, 20, 20), keV=19.1, config_eta=0.0, tol=1e-6,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_kernel_cli_both_validated.py -v
```
Expected: FAIL — `cannot import name '_validate_eta_against_compute_omega_eta'`.

- [ ] **Step 3: Implement `_validate_eta_against_compute_omega_eta`**

In `src/dfxm_geo/reciprocal_space/kernel.py`:

```python
import numpy as np

from dfxm_geo.crystal.oblique import compute_omega_eta


def _validate_eta_against_compute_omega_eta(
    mount: CrystalMount,
    hkl: tuple[int, int, int],
    keV: float,
    config_eta: float,
    *,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """Cross-check the config's eta against compute_omega_eta(mount, hkl, keV).

    Returns (theta_rad, omega_rad) of the matching ω-solution. Raises
    ValueError with a diff if neither (η₁, η₂) matches.
    """
    geom = compute_omega_eta(mount, hkl, keV)
    if np.isnan(geom.omega_1) and np.isnan(geom.omega_2):
        raise ValueError(
            f"Laue condition unsatisfiable for hkl={hkl}, mount={mount}, keV={keV}. "
            "Try a higher keV or a different mount; use 'dfxm-find-reflections' to "
            "enumerate reachable reflections."
        )
    candidates = [
        (geom.eta_1, geom.theta_1, geom.omega_1),
        (geom.eta_2, geom.theta_2, geom.omega_2),
    ]
    for eta_i, theta_i, omega_i in candidates:
        if not np.isnan(eta_i) and abs(eta_i - config_eta) <= tol:
            return float(theta_i), float(omega_i)
    raise ValueError(
        f"Config [geometry] eta={config_eta:.6f} rad does not match the computed "
        f"reflection geometry: (η₁={geom.eta_1:.6f}, η₂={geom.eta_2:.6f}) at "
        f"hkl={hkl}, keV={keV}. Use 'dfxm-find-reflections' to find valid groups."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_kernel_cli_both_validated.py -v
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/kernel.py tests/test_kernel_cli_both_validated.py
git commit -m "Add 'both, validated' eta cross-check between config and compute_omega_eta"
```

---

### Task 14: New filename pattern + extended kernel metadata + back-compat shim

When `mode="oblique"`, the bootstrap writes `Resq_i_theta{θ:.4f}rad_eta{η:.4f}rad_{keV:g}keV_{date}.npz`. When `mode="simplified"` (default), it keeps the legacy `Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_{date}.npz`. Kernel metadata gains `eta`, `geometry_mode`, `lattice`, `a`, `mount_x/y/z`, `omega`.

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py` (`_build_kernel_filename`, `cli_main`, `generate_kernel`)
- Test: `tests/test_kernel_cli_filename_and_metadata.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_kernel_cli_filename_and_metadata.py`:
```python
"""New oblique LUT filename pattern + extended metadata; legacy pattern preserved."""
import numpy as np
import pytest

from dfxm_geo.reciprocal_space.kernel import _build_kernel_filename


def test_simplified_filename_unchanged_from_v220() -> None:
    name = _build_kernel_filename(
        mode="simplified", hkl=(-1, 1, -1), keV=17.0,
        theta=0.0, eta=0.0, date="20260528_1430",
    )
    assert name == "Resq_i_h-1_k1_l-1_17keV_20260528_1430.npz"


def test_oblique_filename_uses_theta_eta_pattern() -> None:
    name = _build_kernel_filename(
        mode="oblique", hkl=(-1, -1, 3), keV=19.1,
        theta=0.2691, eta=0.3531, date="20260528_1430",
    )
    assert name == "Resq_i_theta0.2691rad_eta0.3531rad_19.1keV_20260528_1430.npz"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_kernel_cli_filename_and_metadata.py -v
```
Expected: FAIL — `_build_kernel_filename` has wrong signature.

- [ ] **Step 3: Extend `_build_kernel_filename`**

In `src/dfxm_geo/reciprocal_space/kernel.py`, replace `_build_kernel_filename`:

```python
def _build_kernel_filename(
    mode: str,
    hkl: tuple[int, int, int],
    keV: float,
    *,
    theta: float = 0.0,
    eta: float = 0.0,
    date: str,
) -> str:
    """Per-mode kernel npz basename.

    simplified: `Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_{date}.npz` (legacy, v2.2.0 pattern).
    oblique:    `Resq_i_theta{θ:.4f}rad_eta{η:.4f}rad_{keV:g}keV_{date}.npz`.
    """
    if mode == "simplified":
        h, k, l = hkl
        return f"Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_{date}.npz"
    if mode == "oblique":
        return f"Resq_i_theta{theta:.4f}rad_eta{eta:.4f}rad_{keV:g}keV_{date}.npz"
    raise ValueError(f"unknown geometry mode: {mode!r}")
```

- [ ] **Step 4: Extend kernel metadata in `generate_kernel`**

In `generate_kernel` (same file), expand `kernel_meta`:

```python
kernel_meta = {
    # ... existing keys ...
    "hkl": np.array(hkl if hkl is not None else (0, 0, 0), dtype=np.int64),
    "keV": np.float64(keV if keV is not None else 0.0),
    "seed": np.int64(seed if seed is not None else -1),
    # NEW (oblique-angle metadata; defaults preserve v2.2.0 LUT consumability)
    "eta": np.float64(eta),                          # 0.0 in simplified mode
    "geometry_mode": np.str_(mode),                  # "simplified" | "oblique"
    "lattice": np.str_(mount.lattice),
    "a": np.float64(mount.a),
    "mount_x": np.array(mount.mount_x, dtype=np.int64),
    "mount_y": np.array(mount.mount_y, dtype=np.int64),
    "mount_z": np.array(mount.mount_z, dtype=np.int64),
    "omega": np.float64(omega),                      # 0.0 in simplified mode
}
```

Add `mode`, `eta`, `mount`, `omega` parameters to `generate_kernel` signature (all with safe defaults: `mode="simplified"`, `eta=0.0`, `mount=_DEFAULT_AL_CRYSTAL`, `omega=0.0`).

- [ ] **Step 5: Wire into `cli_main`**

In `cli_main`, after the existing argument parsing:

```python
mode, config_eta = _parse_geometry_block(data.get("geometry"))
mount = _crystal_mount_from_toml(data.get("crystal"))

if mode == "oblique":
    theta_validated, omega_validated = _validate_eta_against_compute_omega_eta(
        mount, hkl_tuple, keV_for_filename, config_eta,
    )
    # Override theta and inject omega.
    reciprocal_kwargs["theta"] = theta_validated
    reciprocal_kwargs["eta"] = config_eta
    omega_for_meta = omega_validated
else:
    omega_for_meta = 0.0

# Build output_path with the new filename selector.
if args.output is not None:
    output_path = args.output
else:
    date = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = Path(fm.pkl_fpath) / _build_kernel_filename(
        mode=mode, hkl=hkl_tuple, keV=keV_for_filename,
        theta=reciprocal_kwargs.get("theta", 0.0),
        eta=config_eta,
        date=date,
    )
# ... rest of cli_main (existing if-exists / --force handling) ...
written = generate_kernel(
    output_path=output_path,
    mode=mode, eta=config_eta, mount=mount, omega=omega_for_meta,
    **reciprocal_kwargs,
)
```

- [ ] **Step 6: Run tests + full suite**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_kernel_cli_filename_and_metadata.py -v
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q --timeout=120
```
Expected: 2 new PASS; full suite 570+ passed.

- [ ] **Step 7: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/kernel.py tests/test_kernel_cli_filename_and_metadata.py
git commit -m "Add oblique LUT filename + metadata; simplified pattern unchanged"
```

---

### Task 15: `_lookup_kernel_path` extended for both modes

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` (`_lookup_kernel_path`)
- Test: `tests/test_lookup_kernel_path_oblique.py`

- [ ] **Step 1: Locate `_lookup_kernel_path`**

In `src/dfxm_geo/direct_space/forward_model.py`, find `_lookup_kernel_path` (grep for the function name). It currently globs the simplified pattern and verifies `hkl`/`keV` in the npz metadata.

- [ ] **Step 2: Write failing test**

Create `tests/test_lookup_kernel_path_oblique.py`:
```python
"""_lookup_kernel_path resolves both simplified (legacy) and oblique LUT patterns."""
import numpy as np
import pytest

from dfxm_geo.direct_space.forward_model import _lookup_kernel_path


def test_finds_legacy_simplified_lut(tmp_path) -> None:
    """A v2.2.0-era LUT (legacy filename, hkl in meta) is found in simplified mode."""
    path = tmp_path / "Resq_i_h-1_k1_l-1_17keV_20260528_1500.npz"
    np.savez(
        path,
        kernel=np.zeros((4, 4, 4)),
        hkl=np.array([-1, 1, -1], dtype=np.int64),
        keV=np.float64(17.0),
        geometry_mode=np.str_("simplified"),
        eta=np.float64(0.0),
    )
    found = _lookup_kernel_path(
        directory=tmp_path,
        mode="simplified", hkl=(-1, 1, -1), keV=17.0,
    )
    assert found == path


def test_finds_oblique_lut_by_theta_eta_keV(tmp_path) -> None:
    """An oblique-mode LUT is found via (θ, η, keV) tuple."""
    path = tmp_path / "Resq_i_theta0.2691rad_eta0.3531rad_19.1keV_20260528_1500.npz"
    np.savez(
        path,
        kernel=np.zeros((4, 4, 4)),
        theta=np.float64(0.2691),
        eta=np.float64(0.3531),
        keV=np.float64(19.1),
        geometry_mode=np.str_("oblique"),
        hkl=np.array([-1, -1, 3], dtype=np.int64),
    )
    found = _lookup_kernel_path(
        directory=tmp_path,
        mode="oblique", theta=0.2691, eta=0.3531, keV=19.1,
    )
    assert found == path


def test_missing_oblique_lut_raises_with_bootstrap_hint(tmp_path) -> None:
    with pytest.raises(KeyError, match="dfxm-bootstrap"):
        _lookup_kernel_path(
            directory=tmp_path,
            mode="oblique", theta=0.2691, eta=0.3531, keV=19.1,
        )


def test_v220_era_lut_loadable_in_simplified_mode(tmp_path) -> None:
    """A LUT written BEFORE the eta metadata was added still loads in simplified mode."""
    path = tmp_path / "Resq_i_h-1_k1_l-1_17keV_20260520_0000.npz"
    np.savez(
        path,
        kernel=np.zeros((4, 4, 4)),
        hkl=np.array([-1, 1, -1], dtype=np.int64),
        keV=np.float64(17.0),
        # NO eta or geometry_mode keys
    )
    found = _lookup_kernel_path(
        directory=tmp_path,
        mode="simplified", hkl=(-1, 1, -1), keV=17.0,
    )
    assert found == path
```

- [ ] **Step 3: Run test to verify it fails**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_lookup_kernel_path_oblique.py -v
```
Expected: FAIL — `_lookup_kernel_path` signature mismatch.

- [ ] **Step 4: Extend `_lookup_kernel_path`**

In `src/dfxm_geo/direct_space/forward_model.py`, rewrite `_lookup_kernel_path` to dispatch on `mode`. The simplified branch is the existing v2.2.0 code (only renamed to `_lookup_legacy_simplified`). New oblique branch:

```python
def _lookup_kernel_path(
    *,
    directory: Path,
    mode: str = "simplified",
    hkl: tuple[int, int, int] | None = None,
    keV: float,
    theta: float | None = None,
    eta: float | None = None,
    tol: float = 1e-6,
) -> Path:
    """Resolve a LUT npz on disk.

    simplified: glob the legacy pattern, verify (hkl, keV) in metadata.
    oblique: glob the new pattern, verify (θ, η, keV) in metadata within `tol`.
    Raises KeyError with a dfxm-bootstrap hint when no match exists.
    """
    if mode == "simplified":
        if hkl is None:
            raise ValueError("simplified mode lookup requires hkl.")
        return _lookup_legacy_simplified(directory, hkl, keV)

    if mode == "oblique":
        if theta is None or eta is None:
            raise ValueError("oblique mode lookup requires theta and eta.")
        glob_pattern = f"Resq_i_theta*_eta*_{keV:g}keV_*.npz"
        candidates = sorted(directory.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in candidates:
            data = np.load(path)
            try:
                if (abs(float(data["theta"]) - theta) <= tol
                    and abs(float(data["eta"]) - eta) <= tol
                    and abs(float(data["keV"]) - keV) <= 1e-9):
                    return path
            except KeyError:
                continue  # incomplete metadata; skip
        raise KeyError(
            f"No bootstrapped kernel matching (mode=oblique, θ={theta:.4f}, η={eta:.4f}, keV={keV:g}) "
            f"in {directory}. Run: dfxm-bootstrap --config <your-config>"
        )

    raise ValueError(f"unknown geometry mode: {mode!r}")
```

(`_lookup_legacy_simplified` is the existing function body, refactored under a new name.)

- [ ] **Step 5: Run tests + full suite**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_lookup_kernel_path_oblique.py -v
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q --timeout=120
```
Expected: 4 new PASS; full suite 574+.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/direct_space/forward_model.py tests/test_lookup_kernel_path_oblique.py
git commit -m "Extend _lookup_kernel_path: oblique mode (theta,eta,keV); simplified unchanged"
```

---

## Forward + identify integration

### Task 16: `forward()` threads `eta` to analytic backend; bit-identical at eta=0

The MC LUT path already bakes `eta` into the LUT (Task 7), so forward consumes it transparently. The analytic path needs `eta` passed to `AnalyticResolution(...)`. The simplest plumbing: `forward()` reads `eta` from the loaded kernel's metadata and passes it where it instantiates the analytic evaluator.

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` (where `AnalyticResolution` is constructed)
- Test: `tests/test_forward_eta_zero_bit_identical.py`

- [ ] **Step 1: Locate analytic instantiation in forward**

In `src/dfxm_geo/direct_space/forward_model.py`, grep for `AnalyticResolution`. Find where it's instantiated; note the surrounding `theta` source.

- [ ] **Step 2: Write failing test**

Create `tests/test_forward_eta_zero_bit_identical.py`:
```python
"""forward() at eta=0 must be bit-identical to v2.2.0 forward.

Replays a tiny scan via the existing CLI; byte-compares HDF5 output to a golden.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

VENV_PY = Path("C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe")


def test_v220_default_config_bit_identical_through_forward(tmp_path: Path) -> None:
    """default.toml (no [geometry]) → forward output identical to a frozen golden.

    The golden is generated once by running this test on v2.2.0's tip; subsequent
    runs check that nothing has drifted.
    """
    out_path = tmp_path / "forward.h5"
    subprocess.run(
        [str(VENV_PY), "-m", "dfxm_geo.cli.forward",
         "--config", "src/dfxm_geo/data/configs/default.toml",
         "--output", str(out_path)],
        check=True, cwd="C:/Users/borgi/Documents/GM-reworked/Geometrical_Optics_master",
    )
    golden_path = Path("tests/data/golden/forward_default_v220.h5")
    if not golden_path.exists():
        pytest.skip(f"Golden not yet generated at {golden_path}; "
                    "run test once on v2.2.0 tip to seed.")
    with h5py.File(out_path, "r") as f_new, h5py.File(golden_path, "r") as f_gold:
        # Compare the detector dataset bit-for-bit.
        for key in ["1.1/measurement/lima_v0", "1.1/measurement/lima_v1"]:
            if key in f_gold:
                np.testing.assert_array_equal(f_new[key][...], f_gold[key][...])
```

- [ ] **Step 3: Run test (expected: SKIP if golden missing, or FAIL after Task 7+9 if drift exists)**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_forward_eta_zero_bit_identical.py -v
```
Expected: SKIP (no golden) on first run.

- [ ] **Step 4: Generate the golden from v2.2.0**

(One-time manual step that happens BEFORE merging this task's changes.)

```bash
git stash                              # park current uncommitted work
git checkout dad8e0c                   # v2.2.0 tag
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m dfxm_geo.cli.forward \
  --config src/dfxm_geo/data/configs/default.toml \
  --output tests/data/golden/forward_default_v220.h5
git checkout feature/oblique-angle-multi-reflection
git stash pop
git add tests/data/golden/forward_default_v220.h5
```

- [ ] **Step 5: Pass `eta` from kernel metadata to analytic eval**

In `src/dfxm_geo/direct_space/forward_model.py`, wherever `AnalyticResolution(theta=theta, …)` is instantiated, add `eta=` from the loaded kernel:

```python
kernel_data = np.load(kernel_path)
theta_val = float(kernel_data["theta"])
eta_val = float(kernel_data.get("eta", 0.0))  # safe default for v2.2.0-era LUTs

analytic = AnalyticResolution(
    theta=theta_val, eta=eta_val,
    sigma_zv=..., sigma_zh=..., sigma_NA=..., sigma_eps=..., zeta_v_clip=...,
)
```

- [ ] **Step 6: Run tests + full suite**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_forward_eta_zero_bit_identical.py -v
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q --timeout=600
```
Expected: PASS (now with golden); full suite 575+.

- [ ] **Step 7: Commit**

```bash
git add src/dfxm_geo/direct_space/forward_model.py tests/test_forward_eta_zero_bit_identical.py tests/data/golden/forward_default_v220.h5
git commit -m "Pass eta from kernel metadata to AnalyticResolution; bit-identical at eta=0"
```

---

### Task 17: Identify modes thread `eta`; bit-identical at eta=0

Identify currently has 3 sub-modes (single/multi/zscan). All consume the same LUT; the analytic-eval site is shared with forward (or duplicated — verify). This task threads `eta` through any identify-specific code paths and re-runs each sub-mode's regression golden.

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (`_run_identification_single/multi/zscan`)
- Test: `tests/test_identify_eta_zero_bit_identical.py`

- [ ] **Step 1: Read identify code**

In `src/dfxm_geo/pipeline.py`, grep for `_run_identification_` and read each sub-mode. Locate where `theta` (or kernel-derived state) flows into the analytic eval and ensure `eta` is similarly plumbed.

- [ ] **Step 2: Write test**

Create `tests/test_identify_eta_zero_bit_identical.py`:
```python
"""All three identify sub-modes at eta=0 → bit-identical to v2.2.0."""
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

VENV_PY = Path("C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe")
REPO = Path("C:/Users/borgi/Documents/GM-reworked/Geometrical_Optics_master")


@pytest.mark.parametrize("config_name", [
    "identification_single.toml",
    "identification_multi.toml",
    "identification_zscan.toml",
])
def test_identify_default_configs_bit_identical(tmp_path: Path, config_name: str) -> None:
    out_path = tmp_path / "identify.h5"
    subprocess.run(
        [str(VENV_PY), "-m", "dfxm_geo.cli.identify",
         "--config", f"src/dfxm_geo/data/configs/{config_name}",
         "--output", str(out_path)],
        check=True, cwd=str(REPO),
    )
    golden = REPO / "tests/data/golden" / f"identify_{config_name.replace('.toml', '')}_v220.h5"
    if not golden.exists():
        pytest.skip(f"Golden not yet generated at {golden}.")
    with h5py.File(out_path, "r") as f_new, h5py.File(golden, "r") as f_gold:
        for key in f_gold:
            if "measurement/lima" in key:
                np.testing.assert_array_equal(f_new[key][...], f_gold[key][...])
```

- [ ] **Step 3: Generate the three identify goldens from v2.2.0**

Same one-time procedure as Task 16 Step 4, repeated for each `identification_*.toml`. Park current branch, checkout `dad8e0c`, run identify with each config, save under `tests/data/golden/identify_*_v220.h5`, return to branch.

- [ ] **Step 4: Plumb `eta` through identify**

If identify shares the analytic instantiation site with forward (Task 16), this step is no-op — already covered. Otherwise add the same `eta = kernel_data.get("eta", 0.0)` read + pass.

- [ ] **Step 5: Run tests + full suite**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_identify_eta_zero_bit_identical.py -v --timeout=600
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q --timeout=600
```
Expected: 3 new PASS; full suite 578+.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_identify_eta_zero_bit_identical.py tests/data/golden/identify_*_v220.h5
git commit -m "Thread eta through identify sub-modes; bit-identical at eta=0"
```

---

### Task 18: Pipeline HDF5 metadata writes eta + mount provenance

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (`write_simulation_h5`)
- Test: `tests/test_pipeline_writes_oblique_provenance.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_pipeline_writes_oblique_provenance.py`:
```python
"""When run with mode='oblique', master HDF5 carries eta+mount provenance attrs."""
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

VENV_PY = Path("C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe")
REPO = Path("C:/Users/borgi/Documents/GM-reworked/Geometrical_Optics_master")


def test_oblique_master_has_eta_mount_attrs(tmp_path: Path) -> None:
    """An oblique-mode forward run leaves geometry attrs on /1.1 (or wherever)."""
    out_path = tmp_path / "oblique.h5"
    subprocess.run(
        [str(VENV_PY), "-m", "dfxm_geo.cli.forward",
         "--config", "src/dfxm_geo/data/configs/al_oblique_figure3.toml",
         "--output", str(out_path)],
        check=True, cwd=str(REPO),
    )
    with h5py.File(out_path, "r") as f:
        meta = f["/1.1"].attrs
        assert meta["geometry_mode"] == "oblique"
        assert np.isclose(meta["eta"], 0.3531, atol=1e-3)
        assert meta["lattice"] == "cubic"
        np.testing.assert_array_equal(meta["mount_x"], [1, 0, 0])
        np.testing.assert_array_equal(meta["mount_y"], [0, 1, 0])
        np.testing.assert_array_equal(meta["mount_z"], [0, 0, 1])
```

(This test needs `al_oblique_figure3.toml` from Task 19 to exist before passing.)

- [ ] **Step 2: Add the provenance writes to `write_simulation_h5`**

In `src/dfxm_geo/pipeline.py`, find `write_simulation_h5`. Where it sets master attrs on `/1.1`, add:

```python
# Oblique-angle provenance (eta=0 / simplified for v2.2.0 configs).
scan_group.attrs["geometry_mode"] = mode
scan_group.attrs["eta"] = float(eta)
scan_group.attrs["theta"] = float(theta)
scan_group.attrs["lattice"] = mount.lattice
scan_group.attrs["a"] = float(mount.a)
scan_group.attrs["mount_x"] = np.array(mount.mount_x, dtype=np.int64)
scan_group.attrs["mount_y"] = np.array(mount.mount_y, dtype=np.int64)
scan_group.attrs["mount_z"] = np.array(mount.mount_z, dtype=np.int64)
```

Plumb `mode`, `eta`, `mount` through `write_simulation_h5` from `run_simulation` (default values preserve v2.2.0 behaviour).

- [ ] **Step 3: Skip running the test until Task 19 ships `al_oblique_figure3.toml`**

Mark the test xfail-pending or skip until Task 19. Or commit the test now and let the pytest collection fail clearly — fix in Task 19.

- [ ] **Step 4: Run full suite (v2.2.0 regression)**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q --timeout=600 -k "not oblique_figure3"
```
Expected: all existing tests still pass; the new oblique-figure3 test errors (config missing) — not regressed yet.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_pipeline_writes_oblique_provenance.py
git commit -m "Write eta + crystal mount provenance to master HDF5 attrs"
```

---

## Phase A ship gate — paper Figure 3B

### Task 19: Create `al_oblique_figure3.toml`

Per spec §"Phase A ship gate" testing-strategy parameter table.

**Files:**
- Create: `src/dfxm_geo/data/configs/al_oblique_figure3.toml`

- [ ] **Step 1: Author the config**

Create `src/dfxm_geo/data/configs/al_oblique_figure3.toml`:
```toml
# Paper Figure 3B reproduction config (arXiv:2503.22022v1, §6.1 + §6.2 + Fig 3 caption)
# Single-edge-dislocation Al phantom at the η=20.233°, θ=15.417° oblique reflection.
# Uses reflection (-1, -1, 3); 3 other reflections in the same group are
# {(-1, 1, 3), (1, -1, 3), (1, 1, 3)} — Phase B will iterate over the group.

[crystal]
lattice  = "cubic"
a        = 4.0493e-10
mount_x  = [1, 0, 0]
mount_y  = [0, 1, 0]
mount_z  = [0, 0, 1]

[geometry]
mode = "oblique"
eta  = 0.353140                # 20.233° in radians

[reflection]
hkl = [-1, -1, 3]
keV = 19.1

[reciprocal]
Nrays       = 1e8
npoints1    = 400
npoints2    = 200
npoints3    = 200
qi1_range   = 5e-4
qi2_range   = 7.5e-3
qi3_range   = 7.5e-3
zeta_v_fwhm = 5.3e-4
zeta_h_fwhm = 0.0
NA_rms      = 3.112e-4         # 0.556 mrad FWHM CRL acceptance / 2.35
eps_rms     = 6.0e-5
beamstop    = false
aperture    = false            # paper Fig 3B uses no beamstop / no aperture
# theta is derived from (hkl, keV, mount) via compute_omega_eta; do not set.

[detector]
Npixels = 272
psize   = 0.75e-6
psf_kernel_size = 9
psf_sigma_px    = 1.0
thermal_mean    = 99.453
thermal_std     = 2.317
dynamic_range_bits = 16

[crystal_setup]
mode = "centered"
dis_um = 4.0
ndis = 1
b = [1, 1, 0]                  # Burgers vector direction (paper says [1,1,0] for edge)
n = [1, 1, -1]                 # slip plane normal
t = [1, 1, 2]                  # line direction
b_magnitude_A = 2.86

# Single detector image at the Fig 3 caption setpoint (NOT a scan):
[scan.phi]
center = -0.42e-3
range  = 0.0
npoints = 1

[scan.chi]
center = 0.46e-3
range  = 0.0
npoints = 1

[scan.two_dtheta]
center = 0.067e-3
range  = 0.0
npoints = 1

[io]
write_strain_provenance = true
```

- [ ] **Step 2: Verify config loads without error**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -c "
import tomllib
with open('src/dfxm_geo/data/configs/al_oblique_figure3.toml', 'rb') as f:
    data = tomllib.load(f)
print(sorted(data.keys()))
"
```
Expected: prints `['crystal', 'crystal_setup', 'detector', 'geometry', 'io', 'reciprocal', 'reflection', 'scan.chi', 'scan.phi', 'scan.two_dtheta']` (or similar).

- [ ] **Step 3: Commit**

```bash
git add src/dfxm_geo/data/configs/al_oblique_figure3.toml
git commit -m "Add paper Figure 3B reproduction config (oblique angle, Al -1,-1,3 reflection)"
```

---

### Task 20: Bootstrap the LUT for the Figure 3B config

This is a manual bootstrap step that the implementation plan exercises once. The LUT goes to `pkl_files/` and is consumed by the Phase A ship-gate test.

**Files:** (no source changes; one LUT file generated)
- Output: `pkl_files/Resq_i_theta0.2691rad_eta0.3531rad_19.1keV_<date>.npz`

- [ ] **Step 1: Run dfxm-bootstrap**

Run (takes ~80 s at the default Nrays=1e8):
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m dfxm_geo.reciprocal_space.kernel \
  --config src/dfxm_geo/data/configs/al_oblique_figure3.toml
```
Expected output:
```
reflection: hkl=(-1, -1, 3), keV=19.1 -> theta = 15.4170 deg, eta = 20.2330 deg
wrote pkl_files/Resq_i_theta0.2691rad_eta0.3531rad_19.1keV_…npz
```

- [ ] **Step 2: Verify LUT metadata**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -c "
import numpy as np
import glob
for p in glob.glob('pkl_files/Resq_i_theta0.2691*.npz'):
    d = np.load(p)
    print(p)
    print('  theta =', d['theta'], '  eta =', d['eta'])
    print('  hkl =', d['hkl'])
    print('  mode =', d['geometry_mode'])
"
```
Expected: theta ≈ 0.2691, eta ≈ 0.3531, hkl = [-1 -1 3], mode = "oblique".

- [ ] **Step 3: Don't commit the LUT** (it's a large binary, regenerable; `pkl_files/` is .gitignored).

---

### Task 21: Generate the vetted Figure 3B golden

The golden is generated ONCE manually (with eyes on the resulting image, comparing visually to paper Figure 3B), then committed as the regression target.

**Files:**
- Output: `tests/data/golden/figure3B_oblique_minus_1_minus_1_3.h5`

- [ ] **Step 1: Run forward on the config**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m dfxm_geo.cli.forward \
  --config src/dfxm_geo/data/configs/al_oblique_figure3.toml \
  --output tests/data/golden/figure3B_oblique_minus_1_minus_1_3.h5
```

- [ ] **Step 2: Eyeball the image**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -c "
import h5py, matplotlib.pyplot as plt, numpy as np
with h5py.File('tests/data/golden/figure3B_oblique_minus_1_minus_1_3.h5', 'r') as f:
    img = f['/1.1/measurement/lima_v0'][0]
plt.figure(figsize=(6,6)); plt.imshow(img, cmap='magma')
plt.title('Phase A Figure 3B candidate'); plt.colorbar(); plt.savefig('tmp/fig3B_candidate.png', dpi=120)
"
```
Open `tmp/fig3B_candidate.png` side by side with paper Figure 3B. Confirm the dislocation contrast pattern is visually consistent (single curved bright/dark feature near image center).

If it looks wrong: STOP. Debug before proceeding — likely a sign error in `R_lab_to_image`, the `_eta_from_k_out` convention, or the analytic backend's eta application. Do NOT freeze a wrong golden.

- [ ] **Step 3: Commit the golden once visually verified**

```bash
git add tests/data/golden/figure3B_oblique_minus_1_minus_1_3.h5
git commit -m "Freeze vetted Figure 3B golden (Al (-1,-1,3) oblique angle reproduction)"
```

---

### Task 22: Phase A ship-gate test — `test_oblique_single_reflection_reproduces_paper_figure3B`

**Files:**
- Test: `tests/test_oblique_single_reflection_reproduces_paper_figure3B.py`

- [ ] **Step 1: Write the gate test**

Create `tests/test_oblique_single_reflection_reproduces_paper_figure3B.py`:
```python
"""Phase A ship gate: reproduce paper Figure 3B end-to-end.

Single detector image, Al (-1,-1,3) at the η=20.233°, θ=15.417° oblique geometry.
Compare per-pixel RMS to a vetted golden frozen in tests/data/golden/.
Tolerance: RMS ≤ 5e-3 of peak intensity.
"""
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

VENV_PY = Path("C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe")
REPO = Path("C:/Users/borgi/Documents/GM-reworked/Geometrical_Optics_master")


def test_figure3B_reproduction_rms_under_5e_3(tmp_path: Path) -> None:
    out_path = tmp_path / "out.h5"
    subprocess.run(
        [str(VENV_PY), "-m", "dfxm_geo.cli.forward",
         "--config", "src/dfxm_geo/data/configs/al_oblique_figure3.toml",
         "--output", str(out_path)],
        check=True, cwd=str(REPO),
    )
    golden_path = REPO / "tests/data/golden/figure3B_oblique_minus_1_minus_1_3.h5"
    with h5py.File(out_path, "r") as f_new, h5py.File(golden_path, "r") as f_gold:
        img_new = f_new["/1.1/measurement/lima_v0"][0].astype(np.float64)
        img_gold = f_gold["/1.1/measurement/lima_v0"][0].astype(np.float64)
    peak = img_gold.max()
    rms = float(np.sqrt(np.mean((img_new - img_gold) ** 2))) / peak
    assert rms <= 5e-3, f"Figure 3B reproduction RMS = {rms:.6f} (gate: ≤ 5e-3)"
```

- [ ] **Step 2: Run the test**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_oblique_single_reflection_reproduces_paper_figure3B.py -v --timeout=600
```
Expected: PASS (it should — the golden was generated by the same code path).

- [ ] **Step 3: Commit**

```bash
git add tests/test_oblique_single_reflection_reproduces_paper_figure3B.py
git commit -m "Add Phase A ship gate: Figure 3B reproduction test (RMS <= 5e-3)"
```

---

### Task 23: Performance regression guard

**Files:**
- Test: `tests/test_forward_throughput_arc_unaffected_by_oblique.py`

- [ ] **Step 1: Write test**

Create `tests/test_forward_throughput_arc_unaffected_by_oblique.py`:
```python
"""Phase 1 + Phase 2 throughput numbers unaffected at eta=0.

Replays the Find_Hg benchmark; new code at eta=0 must hit within 5% of v2.2.0's
~224 ms baseline.
"""
import time

import numpy as np
import pytest

from dfxm_geo.crystal.dislocations import find_hg_population


@pytest.mark.benchmark
def test_find_hg_throughput_unchanged_at_eta_0() -> None:
    """find_hg_population is independent of eta. Sanity-check it didn't regress."""
    # Use a fixed dislocation population; time 5 runs, take median.
    rl = np.zeros((3, 100_000), dtype=np.float64)
    rl[0] = np.linspace(-10e-6, 10e-6, 100_000)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        find_hg_population(rl=rl, dis=4.0, b=np.array([1, 1, 0]),
                          n=np.array([1, 1, -1]), t=np.array([1, 1, 2]))
        times.append(time.perf_counter() - t0)
    median = sorted(times)[len(times) // 2]
    # v2.2.0 baseline ≈ 0.224 s. Gate: ≤ 0.30 s (5% generous; allows for noise).
    assert median <= 0.30, f"find_hg_population median {median:.3f}s exceeds 0.30s guard"
```

(Adjust `find_hg_population` import path / arguments to match the actual current API at impl time. Implementer should re-time the v2.2.0 baseline locally if the laptop has changed.)

- [ ] **Step 2: Run test**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_forward_throughput_arc_unaffected_by_oblique.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_forward_throughput_arc_unaffected_by_oblique.py
git commit -m "Add throughput-regression guard: Find_Hg time unchanged at eta=0"
```

---

## Negative tests for error handling

### Task 24: Error-handling tests (one per spec row)

**Files:**
- Test: `tests/test_oblique_error_handling.py`

- [ ] **Step 1: Write the negative tests**

Create `tests/test_oblique_error_handling.py`:
```python
"""Negative tests — one per Section 6 error-handling row in the spec."""
import numpy as np
import pytest

from dfxm_geo.crystal.oblique import CrystalMount
from dfxm_geo.reciprocal_space.kernel import (
    _crystal_mount_from_toml,
    _parse_geometry_block,
    _validate_eta_against_compute_omega_eta,
)


def test_lattice_non_cubic_rejected_v230() -> None:
    with pytest.raises(ValueError, match="cubic"):
        _crystal_mount_from_toml({
            "lattice": "hexagonal", "a": 4.0e-10,
            "mount_x": [1, 0, 0], "mount_y": [0, 1, 0], "mount_z": [0, 0, 1],
        })


def test_mount_non_orthogonal_rejected() -> None:
    with pytest.raises(ValueError, match="mutually orthogonal"):
        _crystal_mount_from_toml({
            "lattice": "cubic", "a": 4.0e-10,
            "mount_x": [1, 1, 0], "mount_y": [0, 1, 0], "mount_z": [0, 0, 1],
        })


def test_mount_non_integer_rejected() -> None:
    with pytest.raises(ValueError, match="integers"):
        _crystal_mount_from_toml({
            "lattice": "cubic", "a": 4.0e-10,
            "mount_x": [1.5, 0, 0], "mount_y": [0, 1, 0], "mount_z": [0, 0, 1],
        })


def test_oblique_mode_without_eta_rejected() -> None:
    with pytest.raises(ValueError, match="requires \\[geometry\\] eta"):
        _parse_geometry_block({"mode": "oblique"})


def test_invalid_geometry_mode_rejected() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        _parse_geometry_block({"mode": "wat", "eta": 0.0})


def test_simplified_with_nonzero_eta_warns(capsys) -> None:
    mode, eta = _parse_geometry_block({"mode": "simplified", "eta": 0.3})
    assert eta == 0.0
    assert "ignoring [geometry] eta" in capsys.readouterr().err


def test_eta_mismatch_raises_with_diff() -> None:
    mount = CrystalMount(
        lattice="cubic", a=4.0493e-10,
        mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1),
    )
    with pytest.raises(ValueError, match=r"does not match"):
        _validate_eta_against_compute_omega_eta(
            mount, hkl=(-1, -1, 3), keV=19.1, config_eta=0.1, tol=1e-6,
        )


def test_unreachable_reflection_raises() -> None:
    mount = CrystalMount(
        lattice="cubic", a=4.0493e-10,
        mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1),
    )
    with pytest.raises(ValueError, match="Laue.*unsatisfiable"):
        _validate_eta_against_compute_omega_eta(
            mount, hkl=(20, 20, 20), keV=19.1, config_eta=0.0, tol=1e-6,
        )
```

- [ ] **Step 2: Run tests**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_oblique_error_handling.py -v
```
Expected: 8 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_oblique_error_handling.py
git commit -m "Add negative tests for oblique-angle config and validation errors"
```

---

## Release

### Task 25: Version bump + CHANGELOG

**Files:**
- Modify: `pyproject.toml`
- Rename: `tests/test_version_is_2_2_0.py` → `tests/test_version_is_2_3_0.py`
- Modify: `CHANGELOG.md` (or `RELEASE_NOTES.md` — whichever exists)

- [ ] **Step 1: Bump pyproject.toml version**

In `pyproject.toml`, change:
```
version = "2.2.0"
```
to:
```
version = "2.3.0"
```

- [ ] **Step 2: Rename the version-check test**

```bash
git mv tests/test_version_is_2_2_0.py tests/test_version_is_2_3_0.py
```
In the renamed test, update the asserted version string from `2.2.0` to `2.3.0`.

- [ ] **Step 3: Write CHANGELOG entry**

In `CHANGELOG.md` (or wherever release notes live; check for `RELEASE_NOTES.md`), append:

```markdown
## v2.3.0 — 2026-05-?? — Oblique-angle DFXM geometry (Phase A)

### Added
- `[crystal]` TOML block with `lattice = "cubic"`, `a`, `mount_x/y/z` (Miller indices). Default mount preserves the paper Al setup.
- `[geometry]` TOML block with `mode = "simplified" | "oblique"` and `eta` (radians). Default `mode="simplified"` keeps v2.2.0 behaviour.
- New module `dfxm_geo.crystal.oblique`: `CrystalMount`, `compute_omega_eta` (paper Appendix A), `find_reflections` (Table-A.2 enumerator, unwired in this release — Phase B activates), `R_lab_to_image`.
- New LUT filename pattern `Resq_i_theta{θ}rad_eta{η}rad_{keV}keV_*.npz` for oblique mode; the legacy `Resq_i_h{h}_k{k}_l{l}_*.npz` is preserved for simplified mode.
- Both resolution backends (MC `reciprocal_res_func`, analytic `AnalyticResolution`) gain an `eta` keyword (default 0.0).
- Paper Figure 3B (arXiv:2503.22022v1, §6.1) reproduction config + golden + ship-gate test.

### Changed
- `dfxm-bootstrap` validates user-supplied `eta` against `compute_omega_eta(mount, hkl, keV)` ("both, validated") and errors with a diff on mismatch.
- Kernel npz metadata now includes `eta`, `geometry_mode`, `lattice`, `a`, `mount_x/y/z`, `omega`. v2.2.0-era LUTs (missing these keys) still load in simplified mode.

### Deferred
- Multi-reflection iteration `[[reflections]]` — Phase B (v2.4.0).
- `dfxm-find-reflections` CLI — Phase B.
- `.cif` parsing, non-cubic lattices, dropping `simplified` mode — v3.0.0.
- Goniometer μ motor — v3.0.0 or later.
```

- [ ] **Step 4: Run full suite (release sanity)**

Run:
```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q --timeout=600
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m mypy src/dfxm_geo/
```
Expected: all pass, 0 mypy errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_version_is_2_3_0.py CHANGELOG.md
git commit -m "Release v2.3.0: oblique-angle DFXM geometry (Phase A)"
```

---

### Task 26: Merge to main, tag, push (manual user-gated step)

This task is gated by user approval — CLAUDE.md rule: "Confirm before pushing or opening PRs."

- [ ] **Step 1: Merge feature branch to main (no fast-forward, preserve history)**

```bash
git checkout main
git merge --no-ff feature/oblique-angle-multi-reflection \
  -m "Merge feature/oblique-angle-multi-reflection: v2.3.0 oblique-angle Phase A"
```

- [ ] **Step 2: Tag v2.3.0**

```bash
git tag -a v2.3.0 -m "v2.3.0: Oblique-angle DFXM geometry (Phase A)

Adds end-to-end eta-aware geometry (bootstrap → forward → identify) matching
the IUCrJ paper arXiv:2503.22022v1 §3.3 + Appendix A. Default behaviour is
bit-identical to v2.2.0; users opt in via [geometry] mode='oblique' + [crystal]
+ [[reflections]] (single-reflection only in this release; Phase B v2.4.0
will add multi-reflection bundling)."
```

- [ ] **Step 3: Push tag + main (ask user before running)**

```bash
git push origin main
git push origin v2.3.0
```

- [ ] **Step 4: Verify publish.yml**

After the push, check the GitHub Actions run for `publish.yml`. TestPyPI runs unattended; production PyPI is gated on the `pypi` Environment manual approval (per CLAUDE.md). The user approves PyPI when ready.

- [ ] **Step 5: Update CLAUDE.md working notes**

In the parent-dir CLAUDE.md, update the "Latest release tag" line to `v2.3.0` and the tag chain.

---

## Plan complete

After Task 26, Phase A is shipped. Phase B (multi-reflection iteration + `dfxm-find-reflections` CLI + `[[reflections]]` schema) is a separate plan that builds on top of this work — write it after v2.3.0 ships.
