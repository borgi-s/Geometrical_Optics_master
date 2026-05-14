# ESRF_DTU Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the ESRF_DTU branch into the cleanup as a third `dfxm-identify` mode (`z-scan`) — 4D scans (depth × Burgers × line-direction × rocking-curve) — plus z-shift / rotation helpers and a sample-coords 3D viz. Pure-edge / pure-screw functions are NOT ported (collapse to `Fd_find_mixed` at `rotation_deg=0` / `90`; documented instead).

**Architecture:** Mirrors Round 16's pattern. New `IdentificationZScanConfig` dataclass, new `_run_identification_zscan` runner alongside `_run_identification_single` / `_run_identification_multi`. Reuses the existing `save_images_parallel` writer for each per-configuration rocking-curve grid so output is byte-for-byte interoperable with `dfxm-forward` outputs. Random secondary dislocation per (z, b, α) configuration, seeded via `master.spawn`.

**Tech Stack:** Python 3.11, numpy, scipy, matplotlib (already required), pytest, mypy, ruff.

**Spec:** `docs/superpowers/specs/2026-05-14-esrf-dtu-port-design.md`

---

## File Structure

**Create:**
- `src/dfxm_geo/viz/sample.py` — `plot_crystal_in_lab` + `euler_matrix` (port of branch `plot_sample.py`)
- `configs/identification_zscan.toml` — example config
- `tests/test_viz_sample.py` — viz tests

**Modify:**
- `src/dfxm_geo/crystal/rotations.py` — append `rotate_matrix_z_axis`, `is_valid_rotation_matrix`
- `src/dfxm_geo/direct_space/forward_model.py` — append `Z_shift`
- `src/dfxm_geo/crystal/dislocations.py` — update `Fd_find_mixed` docstring with pure-edge/screw note
- `src/dfxm_geo/pipeline.py` — append `IdentificationZScanConfig`, extend `IdentificationConfig` mode + validation, extend `load_identification_config`, append `_run_identification_zscan`, extend `run_identification` + `cli_main_identify`
- `tests/test_rotations.py` — append tests for new helpers
- `tests/test_forward_model_smoke.py` — append `Z_shift` tests
- `tests/test_pipeline_identification.py` — append z-scan tests

---

## Task 1: Add `rotate_matrix_z_axis` helper

**Files:**
- Modify: `src/dfxm_geo/crystal/rotations.py` (append)
- Test: `tests/test_rotations.py` (append)

- [ ] **Step 1: Append failing tests**

Add to `tests/test_rotations.py`:

```python
from dfxm_geo.crystal.rotations import rotate_matrix_z_axis


def test_rotate_matrix_z_axis_zero_is_identity():
    """A 0° rotation around z leaves any matrix unchanged."""
    M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    np.testing.assert_allclose(rotate_matrix_z_axis(M, 0.0), M, atol=1e-15)


def test_rotate_matrix_z_axis_90_permutes_first_two_rows():
    """Rotating identity by 90° around z swaps and signs the first two rows
    of the result (left-multiplication by R_z(90°))."""
    I = np.identity(3)
    R90 = rotate_matrix_z_axis(I, 90.0)
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(R90, expected, atol=1e-12)


def test_rotate_matrix_z_axis_360_is_identity():
    """A 360° rotation returns to identity (within FP tolerance)."""
    M = np.array([[0.5, 0.1, 0.0], [0.2, 0.7, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(rotate_matrix_z_axis(M, 360.0), M, atol=1e-12)
```

- [ ] **Step 2: Run, confirm fail**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_rotations.py -v
```

Expected: ImportError for `rotate_matrix_z_axis`.

- [ ] **Step 3: Implement**

Append to `src/dfxm_geo/crystal/rotations.py`:

```python
def rotate_matrix_z_axis(matrix: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate `matrix` by `angle_degrees` around the lab z axis.

    Returns ``R_z(angle) @ matrix``. Matches the branch source's
    ``functions.rotate_matrix_z_axis`` from the ESRF_DTU port.

    Args:
        matrix: Shape (3, 3) — the matrix to left-rotate.
        angle_degrees: Rotation angle in degrees around the lab z axis.

    Returns:
        Rotated (3, 3) matrix.
    """
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    Rz = np.array(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return Rz @ matrix
```

- [ ] **Step 4: Run tests, confirm pass**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_rotations.py -v
```

- [ ] **Step 5: mypy + ruff clean**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff check src/ tests/
```

- [ ] **Step 6: Commit** (re-stage on hook reformat)

```
git add src/dfxm_geo/crystal/rotations.py tests/test_rotations.py
git commit -m "feat(crystal): rotate_matrix_z_axis helper"
```

---

## Task 2: Add `is_valid_rotation_matrix` helper

**Files:**
- Modify: `src/dfxm_geo/crystal/rotations.py` (append)
- Test: `tests/test_rotations.py` (append)

- [ ] **Step 1: Append failing tests**

```python
from dfxm_geo.crystal.rotations import is_valid_rotation_matrix


def test_is_valid_rotation_matrix_accepts_identity():
    assert is_valid_rotation_matrix(np.identity(3)) is True


def test_is_valid_rotation_matrix_accepts_scipy_random_rotation():
    """A scipy-generated random rotation is always valid."""
    from scipy.spatial.transform import Rotation as R
    M = R.random(random_state=0).as_matrix()
    assert is_valid_rotation_matrix(M) is True


def test_is_valid_rotation_matrix_rejects_scaled_identity():
    """det != 1 → invalid."""
    assert is_valid_rotation_matrix(2.0 * np.identity(3)) is False


def test_is_valid_rotation_matrix_rejects_non_orthogonal():
    """R @ R.T != I → invalid."""
    M = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert is_valid_rotation_matrix(M) is False


def test_is_valid_rotation_matrix_atol_kwarg():
    """A slightly-non-orthogonal matrix is accepted if atol is loosened."""
    M = np.identity(3) + 1e-4 * np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert is_valid_rotation_matrix(M, atol=1e-3) is True
    assert is_valid_rotation_matrix(M, atol=1e-6) is False
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement** — append to `src/dfxm_geo/crystal/rotations.py`:

```python
def is_valid_rotation_matrix(R: np.ndarray, *, atol: float = 1e-6) -> bool:
    """Return True if R is a proper rotation matrix.

    Validates ``det(R) ≈ 1`` and ``R @ R.T ≈ I`` within `atol`. Used by the
    z-scan pipeline to fail early on a malformed Us (rather than producing
    garbage Fg arrays).

    Args:
        R: Candidate 3x3 matrix.
        atol: Absolute tolerance for the determinant and orthogonality
            checks. Default 1e-6.

    Returns:
        True if R passes both checks; False otherwise.
    """
    if R.shape != (3, 3):
        return False
    if not np.isclose(np.linalg.det(R), 1.0, atol=atol):
        return False
    return bool(np.allclose(R @ R.T, np.identity(3), atol=atol))
```

- [ ] **Step 4: Run tests, confirm pass.**

- [ ] **Step 5: mypy + ruff clean.**

- [ ] **Step 6: Commit:**

```
git add src/dfxm_geo/crystal/rotations.py tests/test_rotations.py
git commit -m "feat(crystal): is_valid_rotation_matrix helper"
```

---

## Task 3: Add `Z_shift` to forward_model

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` (append)
- Test: `tests/test_forward_model_smoke.py` (append)

- [ ] **Step 1: Append failing tests**

Add to `tests/test_forward_model_smoke.py`:

```python
def test_Z_shift_zero_offset_matches_module_rl():
    """Z_shift(0.0) reproduces the module-level rl grid bit-for-bit."""
    import dfxm_geo.direct_space.forward_model as fm
    rl_shifted = fm.Z_shift(0.0)
    np.testing.assert_array_equal(rl_shifted, fm.rl)


def test_Z_shift_shifts_z_column_only():
    """Z_shift(5.0) shifts z by 5 µm; x and y unchanged."""
    import dfxm_geo.direct_space.forward_model as fm
    offset_um = 5.0
    rl_shifted = fm.Z_shift(offset_um)
    # x and y rows match
    np.testing.assert_array_equal(rl_shifted[0], fm.rl[0])
    np.testing.assert_array_equal(rl_shifted[1], fm.rl[1])
    # z row is shifted by -offset_um * 1e-6 m (matches branch convention:
    # Z_shift moves the dislocation core *up*, equivalent to shifting rl *down*).
    np.testing.assert_allclose(
        rl_shifted[2], fm.rl[2] - offset_um * 1e-6, atol=1e-18, rtol=1e-15
    )
```

Add `import numpy as np` at the top of the test file if not already imported.

- [ ] **Step 2: Run, confirm fail** (ImportError or AttributeError for `Z_shift`).

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_forward_model_smoke.py -v
```

- [ ] **Step 3: Implement** — append to `src/dfxm_geo/direct_space/forward_model.py`:

```python
def Z_shift(offset_um: float) -> np.ndarray:
    """Return an `rl` grid shifted along the z axis by `offset_um` µm.

    Uses the module's existing detector ray-grid parameters (xl_start, NN1,
    etc.) to build the same mgrid as `rl` does at import time, but with the
    z range translated by ``-offset_um * 1e-6`` m. The module-level ``rl``
    is not modified.

    Used by the z-scan pipeline mode to scan dislocations through the
    sample depth without rebuilding the detector ray grid for each layer.

    Args:
        offset_um: z offset in micrometres. Positive values move the
            dislocation core "up" in the lab z direction (equivalent to
            shifting `rl` "down" by the same amount).

    Returns:
        (3, X) coordinates in metres, same shape as `rl`.
    """
    offset_m = offset_um * 1e-6
    shifted_rl = np.vstack(  # type: ignore[call-overload]
        np.mgrid[
            -xl_range : xl_range : complex(xl_steps),
            -yl_range : yl_range : complex(yl_steps),
            -zl_range - offset_m : zl_range - offset_m : complex(zl_steps),
        ]
    ).reshape(3, -1)
    return shifted_rl
```

The function depends on module-level constants `xl_range`, `xl_steps`, `yl_range`, `yl_steps`, `zl_range`, `zl_steps` — these already exist in `forward_model.py` (they're computed alongside `rl` at import).

- [ ] **Step 4: Run, confirm pass.**

- [ ] **Step 5: mypy + ruff clean.**

- [ ] **Step 6: Commit:**

```
git add src/dfxm_geo/direct_space/forward_model.py tests/test_forward_model_smoke.py
git commit -m "feat(direct-space): Z_shift returns rl grid shifted along z"
```

---

## Task 4: Update Fd_find_mixed docstring with pure-edge/screw note

**Files:**
- Modify: `src/dfxm_geo/crystal/dislocations.py` (docstring only)

This is a docstring-only change. No tests; the existing tests already cover the math (Round 16 Task 2 tests `Fd_find_mixed` at `rotation_deg=0` and `rotation_deg=90`).

- [ ] **Step 1: Read the current `Fd_find_mixed` docstring** to find the right insertion point:

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -c "from dfxm_geo.crystal.dislocations import Fd_find_mixed; help(Fd_find_mixed)" | head -50
```

The docstring already has a "Parameterization note (differs from paper)" section. We add the new note just after that section.

- [ ] **Step 2: Edit the docstring**

Find this section in `src/dfxm_geo/crystal/dislocations.py` (inside `Fd_find_mixed`):

```python
    Naming preserves the convention of the branch source (`disloc_identify`)
    rather than the paper to keep callers unchanged.
```

Insert this after that paragraph, before "Args:":

```python
    **Convenience equivalences (use these instead of separate functions):**

    - ``Fd_find_mixed(..., rotation_deg=0)`` is the pure-edge field
      (equivalent to ``Fd_find(..., ndis=1)`` with matching Ud).
    - ``Fd_find_mixed(..., rotation_deg=90)`` is the pure-screw field
      (only the screw out-of-plane terms ∂u_dx/∂y, ∂u_dx/∂z survive).

    We don't ship separate `Fd_find_edge` / `Fd_find_screw` wrappers; the
    ESRF_DTU branch had them but they're just `Fd_find_mixed` at the
    specific rotation angles above (Borgi 2025 Eq. 1's α=90° and α=0°
    limits respectively).
```

- [ ] **Step 3: Verify no tests broken**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_dislocations_mixed.py -v
```

Expected: 10 passed (unchanged).

- [ ] **Step 4: mypy + ruff clean.**

- [ ] **Step 5: Commit:**

```
git add src/dfxm_geo/crystal/dislocations.py
git commit -m "docs(crystal): note pure-edge/screw equivalences in Fd_find_mixed"
```

---

## Task 5: Create `viz/sample.py` (port plot_sample.py)

**Files:**
- Create: `src/dfxm_geo/viz/sample.py`
- Create: `tests/test_viz_sample.py`

- [ ] **Step 1: Create failing test file** at `tests/test_viz_sample.py`:

```python
"""Unit tests for dfxm_geo.viz.sample — 3D crystal-in-lab visualisation."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

from dfxm_geo.viz.sample import euler_matrix, plot_crystal_in_lab


def test_euler_matrix_zero_is_identity():
    np.testing.assert_allclose(euler_matrix((0.0, 0.0, 0.0)), np.identity(3), atol=1e-12)


def test_euler_matrix_is_orthonormal():
    """Euler matrix has det=1 and is orthonormal (R @ R.T = I)."""
    R = euler_matrix((30.0, 45.0, 60.0))
    np.testing.assert_allclose(R @ R.T, np.identity(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_euler_matrix_order_kwarg():
    """The order kwarg controls the rotation composition; xyz != zyx in general."""
    angles = (10.0, 20.0, 30.0)
    R_xyz = euler_matrix(angles, order="xyz")
    R_zyx = euler_matrix(angles, order="zyx")
    assert not np.allclose(R_xyz, R_zyx)


def test_plot_crystal_in_lab_returns_figure():
    """Returns a matplotlib Figure; doesn't raise."""
    fig = plot_crystal_in_lab(sample_to_lab_R=np.identity(3))
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)
```

- [ ] **Step 2: Run, confirm fail** (ImportError).

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_viz_sample.py -v
```

- [ ] **Step 3: Create `src/dfxm_geo/viz/sample.py`:**

```python
"""3D visualisation of the crystal in lab coordinates.

Pure matplotlib (no plotly). Matches the ESRF_DTU branch's `plot_sample.py`
geometry helpers + draws a cube + axis arrows.
"""

from __future__ import annotations

import numpy as np


def euler_matrix(
    angles_deg: tuple[float, float, float],
    order: str = "xyz",
) -> np.ndarray:
    """Build a 3x3 rotation matrix from Euler angles (degrees).

    Default order ``"xyz"`` composes as ``Rx @ Ry @ Rz`` (applied
    right-to-left to column vectors), matching the branch source.

    Args:
        angles_deg: ``(ax, ay, az)`` in degrees.
        order: Composition order, any permutation of "xyz" (e.g. "zyx").

    Returns:
        (3, 3) orthonormal rotation matrix.
    """
    ax, ay, az = np.radians(angles_deg)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(ax), -np.sin(ax)], [0.0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0.0, np.sin(ay)], [0.0, 1.0, 0.0], [-np.sin(ay), 0.0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0.0], [np.sin(az), np.cos(az), 0.0], [0.0, 0.0, 1.0]])
    R = np.identity(3)
    for axis in order:
        if axis == "x":
            R = Rx @ R
        elif axis == "y":
            R = Ry @ R
        elif axis == "z":
            R = Rz @ R
        else:
            raise ValueError(f"order must be a permutation of 'xyz'; got {order!r}")
    return R


def _draw_cube(
    ax: "matplotlib.axes.Axes",
    *,
    side: float = 1.2,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    R: np.ndarray | None = None,
    facecolor: tuple[float, float, float] = (0.6, 0.75, 1.0),
    edgecolor: str = "k",
    alpha: float = 0.65,
) -> None:
    """Draw an axis-aligned cube rotated by R and translated to center."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if R is None:
        R = np.identity(3)
    s = side / 2.0
    V = np.array(
        [
            [+s, +s, +s], [+s, +s, -s], [+s, -s, +s], [+s, -s, -s],
            [-s, +s, +s], [-s, +s, -s], [-s, -s, +s], [-s, -s, -s],
        ]
    )
    V = (R @ V.T).T + np.asarray(center)
    faces_idx = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    faces = [[V[i] for i in idx] for idx in faces_idx]
    poly = Poly3DCollection(
        faces, facecolors=facecolor, edgecolors=edgecolor, linewidths=1.0, alpha=alpha
    )
    ax.add_collection3d(poly)


def plot_crystal_in_lab(
    sample_to_lab_R: np.ndarray | None = None,
    *,
    side: float = 1.2,
    show_axes: bool = True,
) -> "matplotlib.figure.Figure":
    """Return a matplotlib Figure showing the sample cube in lab coordinates.

    Args:
        sample_to_lab_R: (3, 3) rotation matrix from sample to lab frame.
            Default: identity (cube axis-aligned with lab).
        side: Cube side length. Default 1.2.
        show_axes: If True, draw RGB lab-frame axis arrows. Default True.

    Returns:
        matplotlib Figure. Caller decides whether to ``.show()`` or
        ``.savefig(path)``.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers projection

    if sample_to_lab_R is None:
        sample_to_lab_R = np.identity(3)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    _draw_cube(ax, side=side, R=sample_to_lab_R)

    if show_axes:
        arrow_len = side * 0.8
        ax.quiver(0, 0, 0, arrow_len, 0, 0, color="r", arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, arrow_len, 0, color="g", arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, arrow_len, color="b", arrow_length_ratio=0.1)

    ax.set_xlabel("x (lab)")
    ax.set_ylabel("y (lab)")
    ax.set_zlabel("z (lab)")
    ax.set_box_aspect((1, 1, 1))
    return fig
```

- [ ] **Step 4: Run tests, confirm 4 passed.**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_viz_sample.py -v
```

- [ ] **Step 5: mypy + ruff clean.**

- [ ] **Step 6: Commit:**

```
git add src/dfxm_geo/viz/sample.py tests/test_viz_sample.py
git commit -m "feat(viz): plot_crystal_in_lab (matplotlib 3D crystal viz)"
```

---

## Task 6: Add IdentificationZScanConfig dataclass

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (append)
- Modify: `tests/test_pipeline_identification.py` (append)

- [ ] **Step 1: Append failing tests** to `tests/test_pipeline_identification.py`:

```python
def test_identification_zscan_config_defaults():
    from dfxm_geo.pipeline import IdentificationZScanConfig
    cfg = IdentificationZScanConfig(
        z_offsets_um=[0.0],
        phi_range_deg=0.03,
        phi_steps=11,
        chi_range_deg=0.1,
        chi_steps=11,
    )
    assert cfg.z_offsets_um == [0.0]
    assert cfg.phi_steps == 11
    assert cfg.chi_steps == 11
    assert cfg.include_secondary is True  # default
    assert cfg.secondary_rng_offset == 1  # default


def test_identification_zscan_config_is_frozen():
    from dataclasses import FrozenInstanceError
    from dfxm_geo.pipeline import IdentificationZScanConfig
    cfg = IdentificationZScanConfig(
        z_offsets_um=[0.0],
        phi_range_deg=0.03,
        phi_steps=11,
        chi_range_deg=0.1,
        chi_steps=11,
    )
    with pytest.raises(FrozenInstanceError):
        cfg.phi_steps = 21  # type: ignore[misc]
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Append the dataclass** to `src/dfxm_geo/pipeline.py` (after `IdentificationMonteCarloConfig`):

```python
@dataclass(frozen=True, kw_only=True)
class IdentificationZScanConfig:
    """z-scan mode parameters (mode='z-scan' only).

    Each (z_layer, b, α) configuration produces a (phi_steps × chi_steps)
    rocking-curve stack on disk, with a randomly-drawn secondary
    dislocation if `include_secondary` is True. The secondary is drawn
    once per (z, b, α) and shared across the rocking grid.
    """

    z_offsets_um: list[float]              # e.g. [-2.0, -1.0, 0.0, 1.0, 2.0]
    phi_range_deg: float                   # half-range, degrees
    phi_steps: int
    chi_range_deg: float
    chi_steps: int
    include_secondary: bool = True
    secondary_rng_offset: int = 1
```

- [ ] **Step 4: Run, confirm pass.**

- [ ] **Step 5: mypy + ruff clean.**

- [ ] **Step 6: Commit:**

```
git add src/dfxm_geo/pipeline.py tests/test_pipeline_identification.py
git commit -m "feat(pipeline): IdentificationZScanConfig dataclass"
```

---

## Task 7: Extend IdentificationConfig with z-scan mode + validation

**Files:**
- Modify: `src/dfxm_geo/pipeline.py`
- Modify: `tests/test_pipeline_identification.py` (append)

- [ ] **Step 1: Append failing tests** to `tests/test_pipeline_identification.py`:

```python
from dfxm_geo.pipeline import IdentificationZScanConfig


def _tiny_zscan_config(slip_plane=(1, 1, 1)):
    return IdentificationConfig(
        mode="z-scan",
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=slip_plane,
            angle_start_deg=0.0,
            angle_stop_deg=0.0,
            angle_step_deg=10.0,
            b_vector_indices=[0],
            sweep_all_slip_planes=False,
            exclude_invisibility=False,
        ),
        scan=IdentificationScanConfig(rng_seed=0, intensity_scale=1.0),
        zscan=IdentificationZScanConfig(
            z_offsets_um=[0.0],
            phi_range_deg=0.03,
            phi_steps=2,
            chi_range_deg=0.1,
            chi_steps=2,
            include_secondary=False,
        ),
        io=_make_io_config(),
    )


def test_identification_config_mode_zscan_ok():
    cfg = _tiny_zscan_config()
    assert cfg.mode == "z-scan"
    assert cfg.zscan is not None
    assert cfg.multi is None


def test_identification_config_mode_zscan_requires_zscan_block():
    with pytest.raises(ValueError, match="mode='z-scan'"):
        IdentificationConfig(
            mode="z-scan",
            crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
            scan=IdentificationScanConfig(),
            zscan=None,
            io=_make_io_config(),
        )


def test_identification_config_mode_single_rejects_zscan_block():
    """Passing a zscan block in single mode is a config error (clarity)."""
    with pytest.raises(ValueError, match="single|multi.*zscan"):
        IdentificationConfig(
            mode="single",
            crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
            scan=IdentificationScanConfig(),
            zscan=IdentificationZScanConfig(
                z_offsets_um=[0.0],
                phi_range_deg=0.03,
                phi_steps=2,
                chi_range_deg=0.1,
                chi_steps=2,
            ),
            io=_make_io_config(),
        )
```

- [ ] **Step 2: Run, confirm fail** (the new mode isn't yet in the Literal; type narrows fail or `zscan` kwarg doesn't exist).

- [ ] **Step 3: Modify IdentificationConfig in `src/dfxm_geo/pipeline.py`:**

Find:

```python
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
```

Replace with:

```python
@dataclass(frozen=True, kw_only=True)
class IdentificationConfig:
    """Top-level config for dfxm-identify.

    Validates mode / sub-config / slip-plane consistency in __post_init__.
    """

    mode: Literal["single", "multi", "z-scan"]
    crystal: IdentificationCrystalConfig
    scan: IdentificationScanConfig
    io: IOConfig
    multi: IdentificationMonteCarloConfig | None = None
    zscan: IdentificationZScanConfig | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("single", "multi", "z-scan"):
            raise ValueError(
                f"mode must be 'single', 'multi', or 'z-scan', got {self.mode!r}"
            )
        if self.mode == "multi" and self.multi is None:
            raise ValueError("mode='multi' requires a `multi` config block")
        if self.mode == "z-scan" and self.zscan is None:
            raise ValueError("mode='z-scan' requires a `zscan` config block")
        if self.mode in ("single", "multi") and self.zscan is not None:
            raise ValueError(
                f"mode={self.mode!r}: zscan config block is only valid in mode='z-scan'"
            )
```

(Slip-plane validation block below stays unchanged.)

- [ ] **Step 4: Run, confirm pass.**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py -v
```

- [ ] **Step 5: mypy + ruff clean.**

- [ ] **Step 6: Commit:**

```
git add src/dfxm_geo/pipeline.py tests/test_pipeline_identification.py
git commit -m "feat(pipeline): extend IdentificationConfig with mode='z-scan' + validation"
```

---

## Task 8: Extend `load_identification_config` to parse `[zscan]`

**Files:**
- Modify: `src/dfxm_geo/pipeline.py`
- Modify: `tests/test_pipeline_identification.py` (append)

- [ ] **Step 1: Append failing test**

```python
def test_load_identification_config_zscan(tmp_path):
    """A mode='z-scan' TOML round-trips, including the [zscan] block."""
    toml_text = """
mode = "z-scan"

[crystal]
slip_plane_normal = [1, 1, 1]

[scan]
phi_rad = 1.5e-4
rng_seed = 7

[zscan]
z_offsets_um = [-1.0, 0.0, 1.0]
phi_range_deg = 0.03
phi_steps = 21
chi_range_deg = 0.1
chi_steps = 21
include_secondary = true
secondary_rng_offset = 2

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify_zscan"
perfect_dirname = "ignored"
include_perfect_crystal = false
"""
    cfg_path = tmp_path / "id_zscan.toml"
    cfg_path.write_text(toml_text)

    cfg = load_identification_config(cfg_path)
    assert cfg.mode == "z-scan"
    assert cfg.zscan is not None
    assert cfg.zscan.z_offsets_um == [-1.0, 0.0, 1.0]
    assert cfg.zscan.phi_steps == 21
    assert cfg.zscan.include_secondary is True
    assert cfg.zscan.secondary_rng_offset == 2
    assert cfg.scan.rng_seed == 7
```

- [ ] **Step 2: Run, confirm fail** (the loader doesn't parse `[zscan]` yet).

- [ ] **Step 3: Modify `load_identification_config` in `src/dfxm_geo/pipeline.py`:**

Find:

```python
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

Replace with:

```python
    multi = (
        IdentificationMonteCarloConfig(**data["multi"])
        if data.get("multi") is not None
        else None
    )
    zscan = (
        IdentificationZScanConfig(**data["zscan"])
        if data.get("zscan") is not None
        else None
    )

    return IdentificationConfig(
        mode=data["mode"],
        crystal=crystal,
        scan=scan,
        io=io,
        multi=multi,
        zscan=zscan,
    )
```

- [ ] **Step 4: Run, confirm pass.**

- [ ] **Step 5: mypy + ruff clean.**

- [ ] **Step 6: Commit:**

```
git add src/dfxm_geo/pipeline.py tests/test_pipeline_identification.py
git commit -m "feat(pipeline): load_identification_config parses [zscan] section"
```

---

## Task 9: Implement `_run_identification_zscan`

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (append)
- Modify: `tests/test_pipeline_identification.py` (append)

- [ ] **Step 1: Append failing tests:**

```python
def test_run_identification_zscan_writes_per_config_rocking_grid(tmp_path, monkeypatch):
    """Tiny z-scan: 1 layer × 1 plane × 1 b × 1 α = 1 configuration → 1
    config directory with phi_steps*chi_steps .npy files (4 here)."""
    from dfxm_geo.pipeline import _run_identification_zscan
    import dfxm_geo.direct_space.forward_model as fm
    import numpy as np

    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: np.ones((170, 510)))

    output_dir = tmp_path / "out"
    cfg = _tiny_zscan_config()
    result = _run_identification_zscan(cfg, output_dir)

    layer_dirs = sorted(output_dir.glob("layer_*"))
    assert len(layer_dirs) == 1
    # Inside layer_0000: n_1_1_1 / b0_alpha000 / mosa_test_0000_*.npy
    config_dirs = list((layer_dirs[0] / "n_1_1_1").glob("b*"))
    assert len(config_dirs) == 1
    npys = list(config_dirs[0].glob("*.npy"))
    assert len(npys) == 4  # 2 phi * 2 chi
    manifest = output_dir / "manifest.csv"
    assert manifest.is_file()
    lines = manifest.read_text().strip().splitlines()
    assert len(lines) == 2  # header + 1 row
    assert result["n_configurations"] == 1


def test_run_identification_zscan_is_deterministic_for_seed(tmp_path, monkeypatch):
    """Two runs at same seed produce identical manifests (same secondary draws)."""
    from dfxm_geo.pipeline import _run_identification_zscan
    import dfxm_geo.direct_space.forward_model as fm
    import numpy as np

    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: np.ones((170, 510)))

    cfg = IdentificationConfig(
        mode="z-scan",
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=(1, 1, 1),
            angle_start_deg=0.0,
            angle_stop_deg=90.0,
            angle_step_deg=90.0,
            b_vector_indices=[0, 1],
            sweep_all_slip_planes=False,
            exclude_invisibility=False,
        ),
        scan=IdentificationScanConfig(rng_seed=0, intensity_scale=1.0),
        zscan=IdentificationZScanConfig(
            z_offsets_um=[0.0],
            phi_range_deg=0.03,
            phi_steps=2,
            chi_range_deg=0.1,
            chi_steps=2,
            include_secondary=True,  # exercise the secondary draw
        ),
        io=_make_io_config(),
    )

    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    _run_identification_zscan(cfg, out1)
    _run_identification_zscan(cfg, out2)

    m1 = (out1 / "manifest.csv").read_text()
    m2 = (out2 / "manifest.csv").read_text()
    assert m1 == m2
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement `_run_identification_zscan`** — append to `src/dfxm_geo/pipeline.py`:

```python
def _run_identification_zscan(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """z-scan mode: depth layers × Burgers × angle sweep × rocking-curve grid.

    For each (z_offset, slip_plane, b_idx, alpha), optionally pairs a random
    secondary dislocation drawn once per configuration, computes Hg, and
    writes the phi/chi rocking-curve grid via `save_images_parallel`.
    """
    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo.io.images import save_images_parallel

    assert config.zscan is not None  # validated in __post_init__
    zscan = config.zscan
    crystal_cfg = config.crystal
    scan_cfg = config.scan
    io_cfg = config.io

    output_dir.mkdir(parents=True, exist_ok=True)

    all_planes = [(1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, 1, 1)]
    planes = all_planes if crystal_cfg.sweep_all_slip_planes else [crystal_cfg.slip_plane_normal]

    angles_deg = np.arange(
        crystal_cfg.angle_start_deg,
        crystal_cfg.angle_stop_deg + crystal_cfg.angle_step_deg * 0.5,
        crystal_cfg.angle_step_deg,
    )

    # Reuse Round 16's RNG-spawn pattern. The secondary stream is at
    # `secondary_rng_offset` in the spawn list so a noise stream could be
    # added later without breaking determinism.
    master = np.random.default_rng(scan_cfg.rng_seed)
    spawned = master.spawn(zscan.secondary_rng_offset + 1)
    secondary_rng = spawned[zscan.secondary_rng_offset]

    q_hkl = np.asarray(fm.q_hkl, dtype=float)
    manifest_rows: list[dict[str, Any]] = []
    n_configs = 0

    # Pre-warn the user about output count.
    projected_configs = (
        len(zscan.z_offsets_um)
        * len(planes)
        * (len(crystal_cfg.b_vector_indices) if crystal_cfg.b_vector_indices else 6)
        * len(angles_deg)
    )
    projected_images = projected_configs * zscan.phi_steps * zscan.chi_steps
    print(
        f"z-scan projection (pre-invisibility filter): "
        f"{projected_configs} configs × {zscan.phi_steps * zscan.chi_steps} rocking images "
        f"= {projected_images} .npy files to {output_dir}"
    )

    for k, z_off in enumerate(zscan.z_offsets_um):
        rl_shifted = fm.Z_shift(z_off)
        layer_dir = output_dir / f"layer_{k:04d}"

        for plane in planes:
            plane_slug = _slip_plane_slug(plane)
            b_table = _burgers_vectors(plane)
            b_indices = (
                crystal_cfg.b_vector_indices
                if crystal_cfg.b_vector_indices is not None
                else list(range(len(b_table)))
            )
            b_subset = b_table[b_indices]
            n_arr_unnorm = np.asarray(plane, dtype=float)
            n_arr = n_arr_unnorm / np.linalg.norm(n_arr_unnorm)

            rotated = _rotated_t_vectors(n_arr, b_subset, angles_deg)
            Ud_all = _ud_matrices(n_arr, rotated)

            for j, b_idx in enumerate(b_indices):
                if crystal_cfg.exclude_invisibility and not _passes_invisibility(
                    q_hkl, b_table[b_idx], crystal_cfg.invisibility_threshold_deg
                ):
                    continue
                for i, alpha in enumerate(angles_deg):
                    Ud_primary = Ud_all[i, j]
                    primary_spec = MixedDislocSpec(
                        Ud_mix=Ud_primary, rotation_deg=float(alpha), position_lab_um=(0.0, 0.0, 0.0)
                    )

                    secondary_meta: dict[str, Any]
                    if zscan.include_secondary:
                        sec = _draw_dislocation(secondary_rng, pos_std_um=0.0)
                        secondary_spec = MixedDislocSpec(
                            Ud_mix=sec["Ud"],
                            rotation_deg=sec["alpha_deg"],
                            position_lab_um=sec["pos_um"],
                        )
                        Fg = Fd_find_multi_dislocs_mixed(
                            rl_shifted, fm.Us, [primary_spec, secondary_spec], fm.Theta
                        )
                        secondary_meta = {
                            "secondary_present": True,
                            "secondary_n_h": sec["plane"][0],
                            "secondary_n_k": sec["plane"][1],
                            "secondary_n_l": sec["plane"][2],
                            "secondary_b_idx": sec["b_idx"],
                            "secondary_b_h": int(round(sec["b_vec"][0])),
                            "secondary_b_k": int(round(sec["b_vec"][1])),
                            "secondary_b_l": int(round(sec["b_vec"][2])),
                            "secondary_alpha_deg": sec["alpha_deg"],
                        }
                    else:
                        Fg = Fd_find_mixed(
                            rl_shifted, fm.Us, Ud_mix=Ud_primary, rotation_deg=float(alpha), Theta=fm.Theta
                        )
                        secondary_meta = {
                            "secondary_present": False,
                            "secondary_n_h": float("nan"),
                            "secondary_n_k": float("nan"),
                            "secondary_n_l": float("nan"),
                            "secondary_b_idx": -1,
                            "secondary_b_h": float("nan"),
                            "secondary_b_k": float("nan"),
                            "secondary_b_l": float("nan"),
                            "secondary_alpha_deg": float("nan"),
                        }

                    Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1]) - np.identity(3)
                    fm.Hg = Hg

                    config_dir = layer_dir / f"n_{plane_slug}" / f"b{b_idx}_alpha{int(round(alpha)):03d}"
                    config_dir.mkdir(parents=True, exist_ok=True)
                    save_images_parallel(
                        Hg,
                        zscan.phi_range_deg,
                        zscan.phi_steps,
                        zscan.chi_range_deg,
                        zscan.chi_steps,
                        str(config_dir),
                        io_cfg.fn_prefix,
                        io_cfg.ftype,
                    )

                    manifest_rows.append({
                        "layer": f"{k:04d}",
                        "z_offset_um": z_off,
                        "n_h": plane[0],
                        "n_k": plane[1],
                        "n_l": plane[2],
                        "b_idx": b_idx,
                        "b_h": int(round(b_table[b_idx, 0] * np.sqrt(2))),
                        "b_k": int(round(b_table[b_idx, 1] * np.sqrt(2))),
                        "b_l": int(round(b_table[b_idx, 2] * np.sqrt(2))),
                        "alpha_deg": float(alpha),
                        **secondary_meta,
                        "path": str(config_dir.relative_to(output_dir)),
                    })
                    n_configs += 1

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as fh:
        if manifest_rows:
            writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    return {"n_configurations": n_configs, "output_dir": output_dir, "manifest_path": manifest_path}
```

Note: this relies on `_slip_plane_slug`, `_burgers_vectors`, `_rotated_t_vectors`, `_ud_matrices`, `_passes_invisibility`, `_draw_dislocation`, `MixedDislocSpec`, `Fd_find_mixed`, `Fd_find_multi_dislocs_mixed`, `fast_inverse2` — all already imported in pipeline.py from Round 16. No new imports needed.

- [ ] **Step 4: Run tests, confirm pass.**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_pipeline_identification.py::test_run_identification_zscan_writes_per_config_rocking_grid tests/test_pipeline_identification.py::test_run_identification_zscan_is_deterministic_for_seed -v
```

- [ ] **Step 5: mypy + ruff clean.**

- [ ] **Step 6: Commit:**

```
git add src/dfxm_geo/pipeline.py tests/test_pipeline_identification.py
git commit -m "feat(pipeline): _run_identification_zscan — 4D scan mode"
```

---

## Task 10: Extend `run_identification` dispatcher + `cli_main_identify` choices

**Files:**
- Modify: `src/dfxm_geo/pipeline.py`
- Modify: `tests/test_pipeline_identification.py` (append)

- [ ] **Step 1: Append failing tests**

```python
def test_run_identification_dispatches_to_zscan(tmp_path, monkeypatch):
    """run_identification dispatches mode='z-scan' to _run_identification_zscan."""
    import dfxm_geo.direct_space.forward_model as fm
    import numpy as np
    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: np.ones((170, 510)))

    cfg = _tiny_zscan_config()
    result = run_identification(cfg, tmp_path / "out")
    assert "n_configurations" in result


def test_cli_main_identify_zscan_mode(tmp_path, monkeypatch):
    """The CLI accepts --mode z-scan and produces a manifest."""
    import dfxm_geo.direct_space.forward_model as fm
    import numpy as np
    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: np.ones((170, 510)))

    toml_text = """
mode = "z-scan"

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

[zscan]
z_offsets_um = [0.0]
phi_range_deg = 0.03
phi_steps = 2
chi_range_deg = 0.1
chi_steps = 2
include_secondary = false

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify_zscan"
perfect_dirname = "ignored"
include_perfect_crystal = false
"""
    cfg_path = tmp_path / "zscan.toml"
    cfg_path.write_text(toml_text)
    out_dir = tmp_path / "out"

    exit_code = cli_main_identify(["--config", str(cfg_path), "--output", str(out_dir)])
    assert exit_code == 0
    assert (out_dir / "manifest.csv").is_file()
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Modify `run_identification` and `cli_main_identify`** in `src/dfxm_geo/pipeline.py`:

Find:

```python
def run_identification(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Dispatch to single or multi runner based on config.mode."""
    if config.mode == "single":
        return _run_identification_single(config, output_dir)
    return _run_identification_multi(config, output_dir)
```

Replace with:

```python
def run_identification(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Dispatch to single / multi / z-scan runner based on config.mode."""
    if config.mode == "single":
        return _run_identification_single(config, output_dir)
    if config.mode == "multi":
        return _run_identification_multi(config, output_dir)
    return _run_identification_zscan(config, output_dir)
```

Find:

```python
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default=None,
        help="Override the config's mode field",
    )
```

Replace with:

```python
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "z-scan"],
        default=None,
        help="Override the config's mode field",
    )
```

Find the per-mode print at the end of `cli_main_identify`:

```python
    result = run_identification(cfg, args.output)
    if cfg.mode == "single":
        print(f"Wrote {result['n_images']} images to {result['output_dir']}")
    else:
        print(f"Wrote {result['n_samples']} samples to {result['output_dir']}")
    return 0
```

Replace with:

```python
    result = run_identification(cfg, args.output)
    if cfg.mode == "single":
        print(f"Wrote {result['n_images']} images to {result['output_dir']}")
    elif cfg.mode == "multi":
        print(f"Wrote {result['n_samples']} samples to {result['output_dir']}")
    else:  # z-scan
        print(f"Wrote {result['n_configurations']} configurations to {result['output_dir']}")
    return 0
```

- [ ] **Step 4: Run, confirm pass.**

- [ ] **Step 5: mypy + ruff clean.**

- [ ] **Step 6: Commit:**

```
git add src/dfxm_geo/pipeline.py tests/test_pipeline_identification.py
git commit -m "feat(pipeline): wire z-scan into run_identification + cli_main_identify"
```

---

## Task 11: Create `configs/identification_zscan.toml`

**Files:**
- Create: `configs/identification_zscan.toml`
- Modify: `tests/test_pipeline_identification.py` (append)

- [ ] **Step 1: Append failing test**

```python
def test_example_zscan_config_loads():
    """configs/identification_zscan.toml parses and validates."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_identification_config(repo_root / "configs" / "identification_zscan.toml")
    assert cfg.mode == "z-scan"
    assert cfg.zscan is not None
    assert len(cfg.zscan.z_offsets_um) >= 1
```

- [ ] **Step 2: Run, confirm fail** (FileNotFoundError).

- [ ] **Step 3: Create `configs/identification_zscan.toml`:**

```toml
# dfxm-identify: z-scan mode (4D scan — depth × Burgers × angle × rocking-curve).
#
# Mirrors the ESRF_DTU branch's save_scan workflow: for each z layer, sweep
# (Burgers, line-direction angle), pair with a randomly-drawn secondary
# dislocation (seeded for reproducibility), then save a phi/chi rocking
# curve per configuration via save_images_parallel.
#
# BEFORE A REAL RUN: flip Nsub = 2 -> 1 in
# src/dfxm_geo/direct_space/forward_model.py:42 for ~8x faster per-image
# forward calls. Nsub = 1 is the typical real-run choice; the cleanup
# default Nsub = 2 is preserved as the publication-quality setting from
# Borgi 2024/2025.

mode = "z-scan"

[crystal]
slip_plane_normal = [1, 1, 1]    # starting plane; sweep_all_slip_planes overrides
angle_start_deg = 0.0
angle_stop_deg = 350.0
angle_step_deg = 10.0
sweep_all_slip_planes = true
exclude_invisibility = true
invisibility_threshold_deg = 10.0

[scan]
phi_rad = 1.5e-4                 # unused in z-scan but IdentificationScanConfig requires
poisson_noise = false            # noise lives in the rocking-curve forward calls
rng_seed = 0
intensity_scale = 7.0

[zscan]
z_offsets_um = [-2.0, -1.0, 0.0, 1.0, 2.0]   # 5 depth slices
phi_range_deg = 0.034377467707849395         # matches dfxm-forward default (0.0006 rad)
phi_steps = 21                               # downscaled example; bump to 61 for a real run
chi_range_deg = 0.11459155902616465
chi_steps = 21
include_secondary = true
secondary_rng_offset = 1

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify_zscan"
perfect_dirname = "ignored"
include_perfect_crystal = false
```

- [ ] **Step 4: Run, confirm pass.**

- [ ] **Step 5: Commit:**

```
git add configs/identification_zscan.toml tests/test_pipeline_identification.py
git commit -m "feat(configs): example identification_zscan.toml"
```

---

## Task 12: Final validation gate

**Files:** None modified — validation only.

- [ ] **Step 1: Run the full test suite.**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest -q
```

Expected: ~178 passed (163 before this port + ~15 new tests across Tasks 1-11).

- [ ] **Step 2: Verify mypy clean.**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
```

Expected: `Success: no issues found in 24 source files` (was 23 — added `viz/sample.py`).

- [ ] **Step 3: Verify ruff clean.**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff check src/ tests/
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m ruff format --check src/ tests/
```

- [ ] **Step 4: Verify console script still works.**

```
/c/Users/borgi/Documents/GM-reworked/.venv/Scripts/dfxm-identify.exe --help
```

Expected: argparse help text including `{single,multi,z-scan}` in `--mode` choices.

- [ ] **Step 5: Confirm git status is clean.**

```
git status --short
git log --oneline 059225d..HEAD | head -20
```

Expected: empty working tree; 11 new commits since the spec commit (`059225d`).

- [ ] **Step 6: No final commit — Task 12 is a validation gate only.**

Mark Task 12 done and report:
- Total commits added by this port
- Test count delta (before → after)
- mypy file count delta
- Any deviations from the spec (none expected)

---

## Spec Coverage Audit

| Spec requirement | Task |
|---|---|
| `rotate_matrix_z_axis` helper | Task 1 |
| `is_valid_rotation_matrix` helper | Task 2 |
| `Z_shift(offset_um)` in forward_model | Task 3 |
| `Fd_find_mixed` docstring note on pure-edge/screw equivalence | Task 4 |
| `viz/sample.py` with `plot_crystal_in_lab` + `euler_matrix` | Task 5 |
| `IdentificationZScanConfig` dataclass | Task 6 |
| `IdentificationConfig.mode` Literal extended + validation | Task 7 |
| `load_identification_config` parses `[zscan]` | Task 8 |
| `_run_identification_zscan` (Z_shift, primary sweep, random secondary, save_images_parallel, manifest) | Task 9 |
| `run_identification` dispatcher + `cli_main_identify` choices + per-mode print | Task 10 |
| `configs/identification_zscan.toml` | Task 11 |
| All tests green + mypy/ruff clean + CLI works | Task 12 |
| Nsub=1 preamble note in example config (per user, 2026-05-14) | Task 11 (config preamble) |

All 13 spec-numbered requirements have a corresponding task. ✓
