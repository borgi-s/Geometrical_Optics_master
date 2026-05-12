# Phase 9.2 — Post-Processing Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the COM/mosaicity and qi-field post-processing from `init_forward.py` (lines 102-269) into `dfxm_geo` so that `dfxm-forward` reproduces the paper figures end-to-end from a TOML config.

**Architecture:** Two new modules — `dfxm_geo/analysis/mosaicity.py` (numerical: `compute_chi_shift`, `compute_com_maps`) and `dfxm_geo/viz/mosaicity.py` (figures: `plot_mosaicity_maps`, `plot_qi_cross_section`) — plus a new `run_postprocess` orchestrator in `pipeline.py` with a `PostprocessConfig` dataclass. CLI gains `--no-postprocess` and `--postprocess-only` flags. After this lands, `init_forward.py` moves to `legacy/`.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`scipy.ndimage.center_of_mass`), Matplotlib (SVG backend), pytest, mypy, ruff. All commands assume `cwd = C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master` and the venv at `C:\Users\borgi\Documents\GM-reworked\.venv`.

**Design reference:** `docs/superpowers/specs/2026-05-12-phase-9-2-postprocessing-design.md`

**Conventions:**
- All test commands use the absolute venv python: `C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe`. In the plan we abbreviate as `$PY`.
- Bash POSIX paths (forward slashes) work on this machine for both PowerShell and Bash.
- Branch is `cleanup/main-modernization`. No pushes during implementation; user does pushes manually after PR review.

---

### Task 1: Scaffold `analysis/mosaicity.py` with failing test for `compute_chi_shift`

**Files:**
- Create: `tests/test_mosaicity.py`
- Create: `src/dfxm_geo/analysis/mosaicity.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mosaicity.py`:

```python
"""Unit tests for dfxm_geo.analysis.mosaicity."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.analysis.mosaicity import compute_chi_shift


class TestComputeChiShift:
    def test_corner_peaked_at_middle_returns_baseline_shift(self) -> None:
        """A perfect-crystal stack whose corner pixel peaks at the middle chi
        cell should return a small baseline shift (the systematic discretization
        offset). Asserts against the analytical formula in the docstring."""
        chi_steps = 11  # odd → integer middle index
        phi_steps = 5
        chi_range = 0.1  # degrees
        stack = np.zeros((chi_steps, phi_steps, 4, 4))
        # corner pixel (-1, -1) peaks at chi-index 5 (middle of 11)
        stack[5, :, -1, -1] = 1.0

        shift = compute_chi_shift(stack, chi_steps, chi_range, oversample=100)

        # com[0] = 5.0; shift_idx = 5*100 - 11*100/2 = -50; abs = 50
        # chi_high has 1100 elements over [-0.1, 0.1] → step = 0.2/1099
        expected = 50 * (2 * chi_range / (chi_steps * 100 - 1))
        assert shift == pytest.approx(expected, rel=1e-9)

    def test_shift_grows_when_corner_pixel_shifts(self) -> None:
        """Moving the corner-pixel peak by one chi cell increases |shift| by
        exactly one oversample-step worth of degrees."""
        chi_steps = 11
        chi_range = 0.1
        oversample = 100
        per_cell = oversample * (2 * chi_range / (chi_steps * oversample - 1))

        stack_centered = np.zeros((chi_steps, 5, 4, 4))
        stack_centered[5, :, -1, -1] = 1.0
        stack_shifted = np.zeros((chi_steps, 5, 4, 4))
        stack_shifted[4, :, -1, -1] = 1.0

        s_centered = compute_chi_shift(stack_centered, chi_steps, chi_range,
                                       oversample=oversample)
        s_shifted = compute_chi_shift(stack_shifted, chi_steps, chi_range,
                                      oversample=oversample)
        assert (s_shifted - s_centered) == pytest.approx(per_cell, rel=1e-9)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
$PY -m pytest tests/test_mosaicity.py -v
```

Expected: `ModuleNotFoundError: No module named 'dfxm_geo.analysis.mosaicity'`

- [ ] **Step 3: Implement `compute_chi_shift`**

Create `src/dfxm_geo/analysis/mosaicity.py`:

```python
"""Mosaicity-extraction analysis routines for DFXM rocking-grid stacks.

The functions in this module are a port of the per-pixel center-of-mass
extraction in the original ``init_forward.py``. They consume image stacks
already reshaped to ``(chi_steps, phi_steps, H, W)`` and return:

- ``compute_chi_shift``: a scalar χ-axis offset (in degrees) measured from the
  corner pixel of a strain-free reference stack. Use it to calibrate the χ
  axis before extracting COMs from a strained stack.
- ``compute_com_maps``: per-pixel mosaicity maps in φ and χ (radians) for a
  strained stack.

The straight port preserves the original numerical conventions, including the
``abs(shift)`` sign-loss in ``compute_chi_shift``. Performance work (vectorizing
the COM loop) is deferred to Phase 8.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass


def compute_chi_shift(
    stack_perfect: np.ndarray,
    chi_steps: int,
    chi_range: float,
    *,
    oversample: int = 100,
) -> float:
    """Measure the systematic χ offset from the corner pixel of a perfect stack.

    The corner pixel ``stack_perfect[:, :, -1, -1]`` of a strain-free crystal
    should peak at χ = 0. Any offset is interpreted as a systematic shift
    introduced by the finite rocking grid; this function returns the magnitude
    of that offset in *degrees* (matching ``init_forward.py``'s convention,
    which converts to radians downstream via ``np.deg2rad``).

    Args:
        stack_perfect: Shape ``(chi_steps, phi_steps, H, W)``. Detector frame
            of the perfect-crystal rocking sweep.
        chi_steps: Number of χ steps in the rocking grid.
        chi_range: Half-range of χ in degrees (same units as ``configs/default.toml``).
        oversample: High-resolution refinement factor on the χ axis. The
            original script uses 100.

    Returns:
        Absolute χ-axis shift in degrees.
    """
    com = center_of_mass(stack_perfect[:, :, -1, -1])
    chi_high = np.linspace(-chi_range, chi_range, chi_steps * oversample)
    shift_idx = com[0] * oversample - (chi_steps * oversample / 2)
    return float(chi_high[int(abs(shift_idx))] - chi_high[0])
```

Also create `src/dfxm_geo/analysis/__init__.py` if it doesn't already export the new function — check first; per `ls` output earlier, `__init__.py` exists. Leave it untouched; users can import the fully-qualified path.

- [ ] **Step 4: Run the test to verify it passes**

```bash
$PY -m pytest tests/test_mosaicity.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_mosaicity.py src/dfxm_geo/analysis/mosaicity.py
git commit -m "feat(analysis): add compute_chi_shift port of init_forward.py:108-117"
```

---

### Task 2: Add `compute_com_maps` to `analysis/mosaicity.py`

**Files:**
- Modify: `tests/test_mosaicity.py`
- Modify: `src/dfxm_geo/analysis/mosaicity.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mosaicity.py`:

```python
from dfxm_geo.analysis.mosaicity import compute_com_maps


class TestComputeComMaps:
    def test_planted_centroids_recovered(self) -> None:
        """Each pixel has a planted intensity peak at a known (chi, phi) cell.
        compute_com_maps must return the corresponding chi_high / phi_high
        radian values."""
        chi_steps = 11
        phi_steps = 7
        chi_range = 0.1
        phi_range = 0.05
        oversample = 20
        H, W = 3, 4
        stack = np.zeros((chi_steps, phi_steps, H, W))
        # plant pixel (i, j) → peak at chi=i+1, phi=j+1 (modulo grid)
        for i in range(H):
            for j in range(W):
                stack[(i + 1) % chi_steps, (j + 1) % phi_steps, i, j] = 1.0

        phi_list, chi_list = compute_com_maps(
            stack,
            phi_range,
            phi_steps,
            chi_range,
            chi_steps,
            chi_shift=0.0,
            oversample=oversample,
        )

        phi_high = np.deg2rad(
            np.linspace(-phi_range, phi_range, phi_steps * oversample)
        )
        chi_high = np.deg2rad(
            np.linspace(-chi_range, chi_range, chi_steps * oversample)
        )
        assert phi_list.shape == (H, W)
        assert chi_list.shape == (H, W)
        for i in range(H):
            for j in range(W):
                expected_phi = phi_high[((j + 1) % phi_steps) * oversample]
                expected_chi = chi_high[((i + 1) % chi_steps) * oversample]
                assert phi_list[i, j] == pytest.approx(expected_phi)
                assert chi_list[i, j] == pytest.approx(expected_chi)

    def test_chi_shift_is_applied_additively(self) -> None:
        """Passing chi_shift shifts the χ output by that amount (degrees → radians)."""
        chi_steps = 11
        phi_steps = 7
        chi_range = 0.1
        phi_range = 0.05
        oversample = 20
        stack = np.zeros((chi_steps, phi_steps, 1, 1))
        stack[5, 3, 0, 0] = 1.0

        _, chi_zero = compute_com_maps(
            stack, phi_range, phi_steps, chi_range, chi_steps,
            chi_shift=0.0, oversample=oversample,
        )
        _, chi_shifted = compute_com_maps(
            stack, phi_range, phi_steps, chi_range, chi_steps,
            chi_shift=0.02, oversample=oversample,
        )
        # delta on the chi grid equals deg2rad(0.02)
        assert (chi_shifted[0, 0] - chi_zero[0, 0]) == pytest.approx(
            np.deg2rad(0.02), rel=1e-3
        )

    def test_non_square_grid(self) -> None:
        """COM extraction must not assume H == W (regression guard for the
        latent fastgrainplot non-square-grid bug class)."""
        chi_steps = 5
        phi_steps = 5
        H, W = 3, 7  # asymmetric
        stack = np.zeros((chi_steps, phi_steps, H, W))
        stack[2, 2, :, :] = 1.0

        phi_list, chi_list = compute_com_maps(
            stack, phi_range=0.1, phi_steps=phi_steps,
            chi_range=0.1, chi_steps=chi_steps,
            chi_shift=0.0, oversample=10,
        )
        assert phi_list.shape == (H, W)
        assert chi_list.shape == (H, W)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
$PY -m pytest tests/test_mosaicity.py -v
```

Expected: 3 failures with `ImportError: cannot import name 'compute_com_maps'`.

- [ ] **Step 3: Implement `compute_com_maps`**

Append to `src/dfxm_geo/analysis/mosaicity.py`:

```python
def compute_com_maps(
    stack: np.ndarray,
    phi_range: float,
    phi_steps: int,
    chi_range: float,
    chi_steps: int,
    *,
    chi_shift: float = 0.0,
    oversample: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel center-of-mass extraction over the (φ, χ) rocking grid.

    For each detector pixel ``(i, j)``, computes the centroid of
    ``stack[:, :, i, j]`` (a ``chi_steps × phi_steps`` rocking-curve image)
    and looks the (φ, χ) values up on a ``oversample``-times-refined grid.

    Args:
        stack: Shape ``(chi_steps, phi_steps, H, W)``. Detector frame of the
            dislocated rocking sweep.
        phi_range, phi_steps: Half-range (degrees) and step count of φ.
        chi_range, chi_steps: Half-range (degrees) and step count of χ.
        chi_shift: Additive shift to the χ axis (degrees), as returned by
            :func:`compute_chi_shift`.
        oversample: High-resolution refinement factor for the (φ, χ) grids.

    Returns:
        ``(phi_list, chi_list)`` mosaicity maps, both shape ``(H, W)`` and in
        radians.
    """
    phi_high = np.deg2rad(np.linspace(-phi_range, phi_range, phi_steps * oversample))
    chi_high = np.deg2rad(
        np.linspace(-chi_range + chi_shift, chi_range + chi_shift, chi_steps * oversample)
    )

    H, W = stack.shape[2], stack.shape[3]
    phi_list = np.zeros((H, W))
    chi_list = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            chi_idx, phi_idx = center_of_mass(stack[:, :, i, j])
            phi_list[i, j] = phi_high[int(round(phi_idx * oversample))]
            chi_list[i, j] = chi_high[int(round(chi_idx * oversample))]

    return phi_list, chi_list
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
$PY -m pytest tests/test_mosaicity.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_mosaicity.py src/dfxm_geo/analysis/mosaicity.py
git commit -m "feat(analysis): add compute_com_maps port of init_forward.py:153-163"
```

---

### Task 3: Add `plot_mosaicity_maps` to `viz/mosaicity.py`

**Files:**
- Modify: `tests/test_mosaicity.py`
- Create: `src/dfxm_geo/viz/mosaicity.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mosaicity.py`:

```python
from pathlib import Path

from dfxm_geo.viz.mosaicity import plot_mosaicity_maps


class TestPlotMosaicityMaps:
    def test_writes_valid_svg(self, tmp_path: Path) -> None:
        """Smoke test: the plot function produces a non-empty SVG file."""
        H, W = 4, 4
        phi_list = np.random.default_rng(0).normal(0, 1e-5, size=(H, W))
        chi_list = np.random.default_rng(1).normal(0, 1e-5, size=(H, W))
        out = tmp_path / "mosaicity_maps.svg"

        plot_mosaicity_maps(
            phi_list, chi_list,
            xl_start=-1e-5, yl_start=-1e-5,
            out_path=out,
        )

        assert out.exists()
        content = out.read_text()
        assert len(content) > 0
        assert content.lstrip().startswith("<?xml") or "<svg" in content
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
$PY -m pytest tests/test_mosaicity.py::TestPlotMosaicityMaps -v
```

Expected: `ModuleNotFoundError: No module named 'dfxm_geo.viz.mosaicity'`.

- [ ] **Step 3: Implement `plot_mosaicity_maps`**

Create `src/dfxm_geo/viz/mosaicity.py`:

```python
"""Figure-making for DFXM mosaicity and qi-field outputs.

Wraps matplotlib in functions that save SVG files (no ``plt.show()``). Use
together with :mod:`dfxm_geo.analysis.mosaicity` to produce the SVGs that the
original ``init_forward.py`` script saved as ``extrem_phi+chi2.svg`` and
``qi1+qi2_fields1.svg``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402


def plot_mosaicity_maps(
    phi_list: np.ndarray,
    chi_list: np.ndarray,
    xl_start: float,
    yl_start: float,
    out_path: Path | str,
    *,
    vmin: float = -1e-4,
    vmax: float = 1e-4,
) -> None:
    """Save the two-panel "Extreme Phi" / "Extreme Chi" mosaicity SVG.

    Port of ``init_forward.py:167-214``. Both panels are negated and
    transposed relative to the array layout (matching the original script's
    convention).
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    panels = [
        (phi_list, "Extreme Phi", axs[0]),
        (chi_list, "Extreme Chi", axs[1]),
    ]
    for data, title, ax in panels:
        im = ax.imshow(
            (data * -1).T,
            interpolation="none",
            extent=[xl_start, -xl_start, yl_start, -yl_start],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            origin="lower",
        )
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(r"$y_{\ell}$ ($\mu$m)", fontsize=12)
        ax.set_ylabel(r"$x_{\ell}$ ($\mu$m)", fontsize=12)
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, format=ScalarFormatter(useMathText=True))
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
$PY -m pytest tests/test_mosaicity.py::TestPlotMosaicityMaps -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_mosaicity.py src/dfxm_geo/viz/mosaicity.py
git commit -m "feat(viz): add plot_mosaicity_maps port of init_forward.py:167-214"
```

---

### Task 4: Add `plot_qi_cross_section` to `viz/mosaicity.py`

**Files:**
- Modify: `tests/test_mosaicity.py`
- Modify: `src/dfxm_geo/viz/mosaicity.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mosaicity.py`:

```python
from dfxm_geo.viz.mosaicity import plot_qi_cross_section


class TestPlotQiCrossSection:
    def test_writes_valid_svg(self, tmp_path: Path) -> None:
        xl_steps, yl_steps, zl_steps = 8, 8, 4
        qi_field = np.random.default_rng(2).normal(
            0, 1e-5, size=(3, xl_steps, yl_steps, zl_steps)
        )
        out = tmp_path / "qi_cross_section.svg"

        plot_qi_cross_section(
            qi_field,
            xl_start=-1e-5, yl_start=-1e-5,
            xl_steps=xl_steps, yl_steps=yl_steps, zl_steps=zl_steps,
            out_path=out,
        )

        assert out.exists()
        content = out.read_text()
        assert len(content) > 0
        assert content.lstrip().startswith("<?xml") or "<svg" in content
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
$PY -m pytest tests/test_mosaicity.py::TestPlotQiCrossSection -v
```

Expected: `ImportError: cannot import name 'plot_qi_cross_section'`.

- [ ] **Step 3: Implement `plot_qi_cross_section`**

Append to `src/dfxm_geo/viz/mosaicity.py`:

```python
def plot_qi_cross_section(
    qi_field: np.ndarray,
    xl_start: float,
    yl_start: float,
    xl_steps: int,
    yl_steps: int,
    zl_steps: int,
    out_path: Path | str,
    *,
    vmin: float = -1e-4,
    vmax: float = 1e-4,
) -> None:
    """Save the two-panel qi_1 / qi_2 cross-section SVG at z = 0.

    Port of ``init_forward.py:217-269``. The qi field is sliced at
    ``zl_steps // 2`` (the z = 0 plane in symmetric coordinates).
    """
    X = np.linspace(-xl_start, xl_start, xl_steps) * 1e6  # µm rulers
    Y = np.linspace(-yl_start, yl_start, yl_steps) * 1e6

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    panels = [
        (0, "qi_1 for (x, y) plane, z=0", axs[0]),
        (1, "qi_2 for (x, y) plane, z=0", axs[1]),
    ]
    for idx, title, ax in panels:
        im = ax.imshow(
            qi_field[idx, :, :, zl_steps // 2].squeeze(),
            extent=[Y.min(), Y.max(), X.min(), X.max()],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            origin="lower",
        )
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(r"$y_{\ell}$ ($\mu$m)", fontsize=12)
        ax.set_ylabel(r"$x_{\ell}$ ($\mu$m)", fontsize=12)
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, format=ScalarFormatter(useMathText=True))
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
$PY -m pytest tests/test_mosaicity.py::TestPlotQiCrossSection -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_mosaicity.py src/dfxm_geo/viz/mosaicity.py
git commit -m "feat(viz): add plot_qi_cross_section port of init_forward.py:217-269"
```

---

### Task 5: Add `PostprocessConfig` dataclass + TOML round-trip

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `src/dfxm_geo/pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
from dfxm_geo.pipeline import PostprocessConfig


class TestPostprocessConfigDefaults:
    def test_default_values(self) -> None:
        pc = PostprocessConfig()
        assert pc.enabled is True
        assert pc.chi_oversample == 20
        assert pc.phi_oversample == 20
        assert pc.chi_oversample_for_shift == 100
        assert pc.figures_dirname == "figures"
        assert pc.data_dirname == "analysis"

    def test_simulation_config_includes_postprocess_field(self) -> None:
        cfg = SimulationConfig()
        assert cfg.postprocess == PostprocessConfig()


class TestPostprocessConfigFromToml:
    def test_section_present(self, tmp_path: Path) -> None:
        p = tmp_path / "with_pp.toml"
        p.write_text(
            "[scan]\nphi_range = 0.1\nphi_steps = 10\nchi_range = 0.2\nchi_steps = 20\n"
            "\n[postprocess]\nenabled = false\nchi_oversample = 5\n"
        )
        cfg = SimulationConfig.from_toml(p)
        assert cfg.postprocess.enabled is False
        assert cfg.postprocess.chi_oversample == 5
        # Unspecified keys retain their default
        assert cfg.postprocess.phi_oversample == 20

    def test_section_absent_uses_defaults(self, tmp_path: Path) -> None:
        p = tmp_path / "no_pp.toml"
        p.write_text(
            "[scan]\nphi_range = 0.1\nphi_steps = 10\nchi_range = 0.2\nchi_steps = 20\n"
        )
        cfg = SimulationConfig.from_toml(p)
        assert cfg.postprocess == PostprocessConfig()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
$PY -m pytest tests/test_pipeline.py::TestPostprocessConfigDefaults tests/test_pipeline.py::TestPostprocessConfigFromToml -v
```

Expected: `ImportError: cannot import name 'PostprocessConfig'`.

- [ ] **Step 3: Add `PostprocessConfig` to `pipeline.py`**

In `src/dfxm_geo/pipeline.py`, add this dataclass alongside the others (after `IOConfig`, before `SimulationConfig`):

```python
@dataclass
class PostprocessConfig:
    """Knobs for the post-processing stage (Phase 9.2).

    See ``docs/superpowers/specs/2026-05-12-phase-9-2-postprocessing-design.md``.
    """
    enabled: bool = True
    chi_oversample: int = 20
    phi_oversample: int = 20
    chi_oversample_for_shift: int = 100
    figures_dirname: str = "figures"
    data_dirname: str = "analysis"
```

Extend `SimulationConfig`:

```python
@dataclass
class SimulationConfig:
    crystal: CrystalConfig = field(default_factory=CrystalConfig)
    scan: ScanConfig = field(
        default_factory=lambda: ScanConfig(
            phi_range=0.0006 * 180 / np.pi,
            phi_steps=61,
            chi_range=0.002 * 180 / np.pi,
            chi_steps=61,
        )
    )
    io: IOConfig = field(default_factory=IOConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)

    @classmethod
    def from_toml(cls, path: Path) -> SimulationConfig:
        with path.open("rb") as f:
            raw = tomllib.load(f)
        crystal = CrystalConfig(**raw.get("crystal", {}))
        scan = ScanConfig(**raw["scan"])
        io = IOConfig(**raw.get("io", {}))
        postprocess = PostprocessConfig(**raw.get("postprocess", {}))
        return cls(crystal=crystal, scan=scan, io=io, postprocess=postprocess)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
$PY -m pytest tests/test_pipeline.py -v
```

Expected: All previously-passing tests still pass + 4 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline.py src/dfxm_geo/pipeline.py
git commit -m "feat(pipeline): add PostprocessConfig dataclass + TOML round-trip"
```

---

### Task 6: Implement `run_postprocess` with mocked `forward()`

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `src/dfxm_geo/pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
from dfxm_geo.pipeline import run_postprocess


@pytest.fixture
def tiny_simulation_output(tmp_path: Path) -> tuple[Path, SimulationConfig]:
    """Write tiny synthetic stacks that mimic save_images_parallel output.

    Names match the (j, i) convention in save_image: ``<prefix><i:04d>_<j:04d>.npy``.
    """
    chi_steps, phi_steps = 5, 5
    H, W = 4, 4
    config = SimulationConfig(
        crystal=CrystalConfig(dis=1.0, ndis=2),
        scan=ScanConfig(phi_range=0.05, phi_steps=phi_steps,
                        chi_range=0.05, chi_steps=chi_steps),
        io=IOConfig(),
    )
    output_dir = tmp_path / "out"
    dislocs_dir = output_dir / config.io.dislocs_dirname
    perfect_dir = output_dir / config.io.perfect_dirname
    dislocs_dir.mkdir(parents=True)
    perfect_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    for i in range(chi_steps):
        for j in range(phi_steps):
            suffix = f"{i:04d}_{j:04d}.npy"
            np.save(dislocs_dir / f"{config.io.fn_prefix.lstrip('/')}{suffix}",
                    rng.normal(1.0, 0.01, size=(H, W)))
            np.save(perfect_dir / f"{config.io.fn_prefix.lstrip('/')}{suffix}",
                    rng.normal(1.0, 0.01, size=(H, W)))
    return output_dir, config


class TestRunPostprocess:
    def test_golden_path(
        self, tiny_simulation_output: tuple[Path, SimulationConfig],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        output_dir, config = tiny_simulation_output

        # Mock forward() to avoid needing the kernel pickle.
        fake_qi = np.zeros((3, 4, 4, 4))
        fake_im = np.zeros((4, 4))
        monkeypatch.setattr(
            "dfxm_geo.pipeline.fm.forward",
            lambda Hg, phi=0, chi=0, qi_return=False: (
                (fake_im, fake_qi) if qi_return else (fake_im, None)
            ),
        )
        # Bypass the kernel preflight.
        monkeypatch.setattr("dfxm_geo.pipeline._ensure_kernel_loaded", lambda: None)
        # Sidestep the geometry globals — provide defaults that match the fake qi shape.
        monkeypatch.setattr("dfxm_geo.pipeline.fm.xl_start", 1e-5)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.yl_start", 1e-5)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.xl_steps", 4)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.yl_steps", 4)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.zl_steps", 4)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.Hg",
                            np.zeros((3, 3, 4, 4, 4)))

        result = run_postprocess(output_dir, config)

        # Data products
        data_dir = output_dir / "analysis"
        assert (data_dir / "phi_list.npy").exists()
        assert (data_dir / "chi_list.npy").exists()
        assert (data_dir / "qi_field.npy").exists()
        assert (data_dir / "chi_shift.txt").exists()
        # Figures
        fig_dir = output_dir / "figures"
        assert (fig_dir / "mosaicity_maps.svg").exists()
        assert (fig_dir / "qi_cross_section.svg").exists()
        # Return dict carries the arrays
        assert "phi_list" in result
        assert "chi_list" in result
        assert "chi_shift" in result

    def test_missing_dislocs_dir_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("dfxm_geo.pipeline._ensure_kernel_loaded", lambda: None)
        cfg = SimulationConfig()
        with pytest.raises(FileNotFoundError, match="dislocs"):
            run_postprocess(tmp_path, cfg)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
$PY -m pytest tests/test_pipeline.py::TestRunPostprocess -v
```

Expected: `ImportError: cannot import name 'run_postprocess'`.

- [ ] **Step 3: Implement `run_postprocess`**

Update imports at the top of `src/dfxm_geo/pipeline.py`:

```python
from dfxm_geo.analysis.mosaicity import compute_chi_shift, compute_com_maps
from dfxm_geo.io.images import load_images, save_images_parallel
from dfxm_geo.viz.mosaicity import plot_mosaicity_maps, plot_qi_cross_section
```

(`load_images` joins the existing `save_images_parallel` import.)

Add `run_postprocess` to `src/dfxm_geo/pipeline.py` after `run_simulation`:

```python
def run_postprocess(output_dir: Path, config: SimulationConfig) -> dict[str, Any]:
    """Stages 2-4 of init_forward.py against an existing output_dir.

    Reads the perfect and dislocated stacks from disk, computes the χ-shift
    correction, computes per-pixel COM maps, calls forward() for the qi field
    at z=0, then writes data products and SVGs under output_dir.

    Raises:
        FileNotFoundError: if either expected stack directory is absent.
        RuntimeError: from :func:`_ensure_kernel_loaded` if the reciprocal-
            space kernel is missing.
    """
    _ensure_kernel_loaded()

    dislocs_path = output_dir / config.io.dislocs_dirname
    perfect_path = output_dir / config.io.perfect_dirname

    if not dislocs_path.is_dir():
        raise FileNotFoundError(
            f"Expected dislocs stack at {dislocs_path}; run dfxm-forward "
            "without --postprocess-only first."
        )
    if not perfect_path.is_dir():
        raise FileNotFoundError(
            f"Expected perfect-crystal stack at {perfect_path}; run "
            "dfxm-forward without --postprocess-only first."
        )

    _, dis_reshape, _, _ = load_images(
        str(dislocs_path), config.scan.phi_steps, config.scan.chi_steps,
        file_ext=config.io.ftype,
    )
    _, perf_reshape, _, _ = load_images(
        str(perfect_path), config.scan.phi_steps, config.scan.chi_steps,
        file_ext=config.io.ftype,
    )

    # Stage 2: χ-shift
    chi_shift = compute_chi_shift(
        perf_reshape,
        config.scan.chi_steps,
        config.scan.chi_range,
        oversample=config.postprocess.chi_oversample_for_shift,
    )

    # Stage 3: per-pixel COM maps. The original script uses the same oversample
    # factor for both axes; we keep them independent in the API but they
    # default to the same value.
    phi_list, chi_list = compute_com_maps(
        dis_reshape,
        config.scan.phi_range,
        config.scan.phi_steps,
        config.scan.chi_range,
        config.scan.chi_steps,
        chi_shift=chi_shift,
        oversample=config.postprocess.phi_oversample,
    )

    # Stage 4: qi field at z=0 via a single forward() call. Uses module-level
    # Hg as left by run_simulation (or the default load).
    _, qi_field = fm.forward(fm.Hg, phi=0, qi_return=True)

    # Persist data products.
    data_dir = output_dir / config.postprocess.data_dirname
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "phi_list.npy", phi_list)
    np.save(data_dir / "chi_list.npy", chi_list)
    np.save(data_dir / "qi_field.npy", qi_field)
    (data_dir / "chi_shift.txt").write_text(f"{chi_shift}\n")

    # Render figures.
    fig_dir = output_dir / config.postprocess.figures_dirname
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_mosaicity_maps(
        phi_list, chi_list, fm.xl_start, fm.yl_start,
        fig_dir / "mosaicity_maps.svg",
    )
    plot_qi_cross_section(
        qi_field, fm.xl_start, fm.yl_start,
        fm.xl_steps, fm.yl_steps, fm.zl_steps,
        fig_dir / "qi_cross_section.svg",
    )

    return {
        "phi_list": phi_list,
        "chi_list": chi_list,
        "qi_field": qi_field,
        "chi_shift": chi_shift,
        "data_dir": data_dir,
        "figures_dir": fig_dir,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
$PY -m pytest tests/test_pipeline.py -v
```

Expected: all tests pass (existing + new).

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline.py src/dfxm_geo/pipeline.py
git commit -m "feat(pipeline): add run_postprocess for stages 2-4 of init_forward.py"
```

---

### Task 7: Wire CLI flags `--no-postprocess` and `--postprocess-only`

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `src/dfxm_geo/pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
from dfxm_geo.pipeline import cli_main


class TestCliMainFlags:
    def test_default_runs_sim_then_postprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_path = tmp_path / "cfg.toml"
        config_path.write_text(
            "[scan]\nphi_range=0.05\nphi_steps=5\nchi_range=0.05\nchi_steps=5\n"
        )
        calls: list[str] = []
        monkeypatch.setattr(
            "dfxm_geo.pipeline.run_simulation",
            lambda cfg, out: calls.append("sim") or {},
        )
        monkeypatch.setattr(
            "dfxm_geo.pipeline.run_postprocess",
            lambda out, cfg: calls.append("pp") or {},
        )

        cli_main(["--config", str(config_path), "--output", str(tmp_path / "out")])
        assert calls == ["sim", "pp"]

    def test_no_postprocess_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_path = tmp_path / "cfg.toml"
        config_path.write_text(
            "[scan]\nphi_range=0.05\nphi_steps=5\nchi_range=0.05\nchi_steps=5\n"
        )
        calls: list[str] = []
        monkeypatch.setattr(
            "dfxm_geo.pipeline.run_simulation",
            lambda cfg, out: calls.append("sim") or {},
        )
        monkeypatch.setattr(
            "dfxm_geo.pipeline.run_postprocess",
            lambda out, cfg: calls.append("pp") or {},
        )

        cli_main([
            "--config", str(config_path), "--output", str(tmp_path / "out"),
            "--no-postprocess",
        ])
        assert calls == ["sim"]

    def test_postprocess_only_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_path = tmp_path / "cfg.toml"
        config_path.write_text(
            "[scan]\nphi_range=0.05\nphi_steps=5\nchi_range=0.05\nchi_steps=5\n"
        )
        calls: list[str] = []
        monkeypatch.setattr(
            "dfxm_geo.pipeline.run_simulation",
            lambda cfg, out: calls.append("sim") or {},
        )
        monkeypatch.setattr(
            "dfxm_geo.pipeline.run_postprocess",
            lambda out, cfg: calls.append("pp") or {},
        )

        cli_main([
            "--config", str(config_path), "--output", str(tmp_path / "out"),
            "--postprocess-only",
        ])
        assert calls == ["pp"]

    def test_postprocess_disabled_in_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """[postprocess].enabled = false skips the stage even without --no-postprocess."""
        config_path = tmp_path / "cfg.toml"
        config_path.write_text(
            "[scan]\nphi_range=0.05\nphi_steps=5\nchi_range=0.05\nchi_steps=5\n"
            "\n[postprocess]\nenabled = false\n"
        )
        calls: list[str] = []
        monkeypatch.setattr(
            "dfxm_geo.pipeline.run_simulation",
            lambda cfg, out: calls.append("sim") or {},
        )
        monkeypatch.setattr(
            "dfxm_geo.pipeline.run_postprocess",
            lambda out, cfg: calls.append("pp") or {},
        )

        cli_main(["--config", str(config_path), "--output", str(tmp_path / "out")])
        assert calls == ["sim"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
$PY -m pytest tests/test_pipeline.py::TestCliMainFlags -v
```

Expected: 4 failures with `error: unrecognized arguments: --no-postprocess` (or `--postprocess-only`).

- [ ] **Step 3: Replace `cli_main` in `src/dfxm_geo/pipeline.py`**

Replace the existing `cli_main` function with:

```python
def cli_main(argv: list[str] | None = None) -> int:
    """Entry point for ``dfxm-forward`` and ``python scripts/run_forward.py``.

    Default behavior: run simulation, then post-processing.
    """
    parser = argparse.ArgumentParser(description="DFXM forward simulation")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Run simulation only; skip post-processing (Phase 6 behavior).",
    )
    mode.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Skip simulation; run post-processing against an existing output dir.",
    )
    args = parser.parse_args(argv)

    config = SimulationConfig.from_toml(args.config)

    if args.postprocess_only:
        run_postprocess(args.output, config)
    else:
        run_simulation(config, args.output)
        if config.postprocess.enabled and not args.no_postprocess:
            run_postprocess(args.output, config)
    return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
$PY -m pytest tests/test_pipeline.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline.py src/dfxm_geo/pipeline.py
git commit -m "feat(pipeline): add --no-postprocess and --postprocess-only CLI flags"
```

---

### Task 8: Update shipped configs with `[postprocess]` section

**Files:**
- Modify: `configs/default.toml`

- [ ] **Step 1: Add the `[postprocess]` section to `configs/default.toml`**

Append to `configs/default.toml`:

```toml

[postprocess]
enabled = true
chi_oversample = 20
phi_oversample = 20
chi_oversample_for_shift = 100
figures_dirname = "figures"
data_dirname = "analysis"
```

- [ ] **Step 2: Confirm existing all-shipped-variants test still passes**

```bash
$PY -m pytest tests/test_pipeline.py::TestSimulationConfigFromToml::test_all_shipped_variants_parse tests/test_pipeline.py::TestSimulationConfigFromToml::test_round_trip_default -v
```

Expected: both pass. The `test_all_shipped_variants_parse` test already iterates every TOML in `configs/` so it exercises the new section without code changes.

- [ ] **Step 3: Add an explicit assertion that `default.toml` has the new section**

In `tests/test_pipeline.py`, extend `TestSimulationConfigFromToml::test_round_trip_default` with:

```python
        assert cfg.postprocess.enabled is True
        assert cfg.postprocess.chi_oversample == 20
```

- [ ] **Step 4: Run the full pipeline test suite to confirm**

```bash
$PY -m pytest tests/test_pipeline.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add configs/default.toml tests/test_pipeline.py
git commit -m "feat(configs): add [postprocess] section to default.toml"
```

(Variants `dis_0p25.toml` / `dis_0p5.toml` / `dis_1.toml` / `dis_2.toml` intentionally inherit defaults — they only override `[crystal].dis`.)

---

### Task 9: Migrate `init_forward.py` to `legacy/`

**Files:**
- Move: `init_forward.py` → `legacy/init_forward.py`
- Modify: `README.md`

- [ ] **Step 1: Create the `legacy/` directory and move the file**

```bash
mkdir -p legacy
git mv init_forward.py legacy/init_forward.py
```

- [ ] **Step 2: Add a `legacy/README.md` note**

Create `legacy/README.md`:

```markdown
# Legacy entry points

Files in this directory are preserved verbatim from before the Phase 6 / 9.2
refactor. They are not maintained — use them as historical reference, not as
recommended workflows.

- `init_forward.py` — the single-file demo script that was the original
  entry point. Reproduce the same paper figures via:

    dfxm-forward --config configs/default.toml --output output/

  See `docs/reproducibility.md` for the supported flow.
```

- [ ] **Step 3: Update the top-level `README.md`**

Find the section that documents `init_forward.py` (search for `init_forward` in `README.md`). Replace any reference that recommends running `init_forward.py` with the `dfxm-forward` command, and add a single line at the end of the section:

> The original `init_forward.py` is preserved under `legacy/` as a single-file reference of the pre-cleanup workflow.

Use:

```bash
grep -n "init_forward" README.md
```

to find references. Update each in place via `Edit`.

- [ ] **Step 4: Run the test suite to confirm nothing imports the old path**

```bash
$PY -m pytest -q
```

Expected: all tests pass. If anything fails with `ModuleNotFoundError` for `init_forward`, that's a leftover import; remove it.

- [ ] **Step 5: Commit**

```bash
git add legacy/ README.md
git commit -m "refactor: move init_forward.py to legacy/ (superseded by dfxm-forward)"
```

---

### Task 10: Update `docs/reproducibility.md` for the new flow

**Files:**
- Modify: `docs/reproducibility.md`

- [ ] **Step 1: Read the current doc**

```bash
$PY -c "print(open('docs/reproducibility.md').read())"
```

Identify the section labeled "what's intentionally not yet configurable" (it mentions post-processing among the deferred items). That sentence should be removed.

- [ ] **Step 2: Edit the doc**

Make these edits:

1. Remove the line that says post-processing is deferred.
2. Add a new section after the existing "Quick start" or equivalent, titled `## Post-processing outputs`, with this body:

```markdown
## Post-processing outputs

`dfxm-forward` runs post-processing by default after the simulation. Outputs land under:

- `<output>/analysis/phi_list.npy` — mosaicity-in-φ map (radians)
- `<output>/analysis/chi_list.npy` — mosaicity-in-χ map (radians)
- `<output>/analysis/qi_field.npy` — qi field at the (x, y) plane, z = 0 included
- `<output>/analysis/chi_shift.txt` — scalar χ-axis correction (degrees)
- `<output>/figures/mosaicity_maps.svg` — "Extreme Phi" / "Extreme Chi" 2-panel figure
- `<output>/figures/qi_cross_section.svg` — qi_1 / qi_2 (x, y) cross-section

Skip post-processing for a sim-only run:

```bash
dfxm-forward --config configs/default.toml --output output/ --no-postprocess
```

Re-run post-processing against an existing simulation directory without
re-simulating (useful after tweaking `[postprocess]` parameters):

```bash
dfxm-forward --config configs/default.toml --output output/ --postprocess-only
```

The post-processing stage requires the reciprocal-space resolution kernel
pickle (same as the simulation stage) to compute the qi field.
```

3. Make sure the existing edge-effect `(dis, ndis)` table and variant configs section is unchanged.

- [ ] **Step 3: Lint and format check**

```bash
$PY -m ruff check . && $PY -m ruff format --check .
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add docs/reproducibility.md
git commit -m "docs: reproducibility.md — document Phase 9.2 post-processing outputs"
```

---

### Task 11: Final verification

**Files:** none modified

- [ ] **Step 1: Full pytest run**

```bash
$PY -m pytest -v --cov=src/dfxm_geo --cov-report=term-missing
```

Expected:
- All tests pass (previous 58 + new tests from Tasks 1-7, which adds ~13).
- `dfxm_geo/analysis/mosaicity.py` and `dfxm_geo/viz/mosaicity.py` at near 100% coverage.
- Total package coverage strictly above the 52% baseline (target ≥ 60%).

- [ ] **Step 2: mypy**

```bash
$PY -m mypy src/dfxm_geo/
```

Expected: 0 errors. If mypy complains about matplotlib types, add `matplotlib` to the `ignore_missing_imports` list in `pyproject.toml` under `[tool.mypy.overrides]`.

- [ ] **Step 3: ruff + pre-commit**

```bash
$PY -m ruff check . && $PY -m ruff format --check . && pre-commit run --all-files
```

Expected: all clean.

- [ ] **Step 4: Smoke import test from a fresh interpreter**

```bash
$PY -c "from dfxm_geo.pipeline import run_postprocess, PostprocessConfig, cli_main; from dfxm_geo.analysis.mosaicity import compute_chi_shift, compute_com_maps; from dfxm_geo.viz.mosaicity import plot_mosaicity_maps, plot_qi_cross_section; print('OK')"
```

Expected: `OK` printed; no import errors. (Imports the new surface from an interpreter that did not run the test suite first.)

- [ ] **Step 5: Mark spec + plan + tasks as complete**

Update `docs/superpowers/specs/2026-05-12-phase-9-2-postprocessing-design.md` status header to `**Status:** implemented YYYY-MM-DD on cleanup/main-modernization`.

Update the cleanup session-state memory entry to record what Phase 9.2 produced (new commit hashes, test count, coverage delta).

- [ ] **Step 6: Final commit**

```bash
git add docs/superpowers/specs/2026-05-12-phase-9-2-postprocessing-design.md
git commit -m "docs: mark Phase 9.2 spec as implemented"
```

No push. User reviews and decides when to push to origin.

---

## Notes for the implementer

- **`fm.Hg` semantics.** `run_postprocess` reads `fm.Hg` directly. When `run_simulation` runs first in `cli_main`, that global is set to the computed value before post-processing runs. When `--postprocess-only` is used, `fm.Hg` is whatever `_load_default_kernel` set at import (with `compute_Hg=True`). Both paths produce a defined `fm.Hg`; this is intentional, not a bug.
- **Reused `_ensure_kernel_loaded`.** Already present in `pipeline.py` from Phase 6. Called twice in the default path (once from `run_simulation`, once from `run_postprocess`); idempotent and cheap.
- **The COM loop is `O(H × W)` over `scipy.ndimage.center_of_mass`.** For the default 512×512 image at 61×61 rocking grid that's slow. Out of scope here — Phase 8 vectorizes.
- **Test isolation.** Every test that touches `fm.*` uses `monkeypatch` to restore globals. The `Agg` backend selection in `viz/mosaicity.py` is process-global but harmless under pytest.
- **CLAUDE.md rules.** Do not push without explicit user consent. All work is on `cleanup/main-modernization`. Run mypy + ruff on every commit; pre-commit will catch most issues but verify manually before the final commit.
