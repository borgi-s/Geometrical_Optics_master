# Config-driven detector geometry + counts_scale re-measurement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the single-reflection forward path's detector/ray-grid geometry config-driven via a `[detector_geometry]` block (`pixel_size`+`magnification`→object pitch, `Npixels`, `Nsub`), default-omitted path byte-identical, then re-run `derive_counts_scale.py` at the experiment's true 10× geometry and report whether `counts_scale` is still a v3.0.0 blocker.

**Architecture:** Reuse the existing `InstrumentContext` seam. Add a config dataclass + a from-config instrument builder; the forward orchestrator builds the instrument once from config and threads it through fov, `Find_Hg`, and `build_forward_context`. Fix `build_geometry_context` to take the y/z grid extents from the instrument (not module globals) so an overridden pitch produces a consistent grid. Defaults resolve to exactly `(40e-9, 510, 1)`, so all determinism/golden gates stay green.

**Tech Stack:** Python, dataclasses, tomllib, numpy, h5py, pytest, mypy.

**Environment:** venv python `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe`; working dir `C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master`; branch `feature/config-detector-geometry` (off `main` `9846fd6`) already exists.

**Spec:** `docs/superpowers/specs/2026-06-15-config-driven-detector-geometry-design.md`

---

## File structure

| File | Responsibility |
|---|---|
| `src/dfxm_geo/config.py` | **Modify.** Add `DetectorGeometryConfig` (frozen, with `from_dict` resolving `object_psize = pixel_size/magnification`); nest in `SimulationConfig`; parse in `from_toml`. |
| `src/dfxm_geo/direct_space/forward_model.py` | **Modify.** Add `build_instrument_context_from_config(...)`; fix `build_geometry_context` to derive `yl_range`/`zl_range` from the instrument. |
| `src/dfxm_geo/orchestrator.py` | **Modify.** Build the instrument from `config.detector_geometry`; use it for fov (`:359`), the effective-config print (`:377`), `build_forward_context` single (`:395`), and the wall `Find_Hg` (`:431`). |
| `tests/test_detector_geometry_config.py` | **Create.** Schema parse/resolve/validation + byte-identical-default context-equality + a forward run honoring an `Npixels` override. |
| `docs/calibration/derive_counts_scale.py` | **Modify.** Run the calibration at both the default (40 nm) and the matched 10× (37.6 nm) geometry; print both. |
| `docs/m4-validation-report.md` | **Modify.** Add a "counts_scale re-measurement (2026-06-15)" section with the before/after numbers + a "still a blocker?" verdict. |

**Out of scope (documented follow-ups):** the identify path, multi-reflection forward (`_context_for_run`), and forward z-scan (`Z_shift`) keep reading module globals — byte-identical for default configs but not yet config-driven.

---

### Task 1: `DetectorGeometryConfig` schema

**Files:**
- Modify: `src/dfxm_geo/config.py` (add class near `DetectorConfig` ~line 690; add field + parse in `SimulationConfig` ~lines 729, 746)
- Test: `tests/test_detector_geometry_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_detector_geometry_config.py
"""[detector_geometry] config block: pixel_size/magnification -> object pitch."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from dfxm_geo.config import DetectorGeometryConfig, SimulationConfig


def test_omitted_block_resolves_to_module_defaults():
    g = DetectorGeometryConfig.from_dict(None)
    assert g.object_psize == 40e-9
    assert g.Npixels == 510
    assert g.Nsub == 1


def test_pixel_size_over_magnification_resolves_object_psize():
    g = DetectorGeometryConfig.from_dict(
        {"pixel_size": 0.65e-6, "magnification": 17.31, "Npixels": 510, "Nsub": 1}
    )
    assert math.isclose(g.object_psize, 0.65e-6 / 17.31, rel_tol=0, abs_tol=0)
    assert g.pixel_size == 0.65e-6 and g.magnification == 17.31


def test_pixel_size_without_magnification_raises():
    with pytest.raises(ValueError, match="together"):
        DetectorGeometryConfig.from_dict({"pixel_size": 0.65e-6})


def test_nonpositive_values_raise():
    with pytest.raises(ValueError):
        DetectorGeometryConfig.from_dict({"pixel_size": 0.65e-6, "magnification": -1.0})
    with pytest.raises(ValueError):
        DetectorGeometryConfig.from_dict({"Npixels": 0})


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown"):
        DetectorGeometryConfig.from_dict({"psize": 40e-9})


def test_from_toml_parses_block(tmp_path: Path):
    cfg = tmp_path / "g.toml"
    cfg.write_text(
        '[detector_geometry]\npixel_size = 0.65e-6\nmagnification = 17.31\nNpixels = 120\n',
        encoding="utf-8",
    )
    sc = SimulationConfig.from_toml(cfg)
    assert sc.detector_geometry.Npixels == 120
    assert math.isclose(sc.detector_geometry.object_psize, 0.65e-6 / 17.31)


def test_from_toml_default_when_block_absent(tmp_path: Path):
    cfg = tmp_path / "g.toml"
    cfg.write_text('mode = "single"\n', encoding="utf-8")
    sc = SimulationConfig.from_toml(cfg)
    assert sc.detector_geometry.object_psize == 40e-9
    assert sc.detector_geometry.Npixels == 510
```

- [ ] **Step 2: Run to verify it fails**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_detector_geometry_config.py -q`
Expected: FAIL — `ImportError: cannot import name 'DetectorGeometryConfig'`.

- [ ] **Step 3: Add the dataclass**

In `config.py`, after `DetectorConfig` (~line 712), add:

```python
@dataclass(frozen=True, kw_only=True)
class DetectorGeometryConfig:
    """[detector_geometry] block: object-plane ray-grid geometry.

    ``pixel_size`` is the detector-plane effective pixel (camera pixel / visible
    objective, m) and ``magnification`` the X-ray objective magnification M; the
    object-plane pitch the forward model uses is ``object_psize = pixel_size /
    magnification``. When the block is omitted the defaults reproduce the
    forward_model module globals exactly (40 nm / 510 / 1), so existing configs
    and the byte-identity gates are unchanged.
    """

    object_psize: float = 40e-9
    Npixels: int = 510
    Nsub: int = 1
    pixel_size: float | None = None      # detector-plane pixel (m); provenance
    magnification: float | None = None   # X-ray objective M; provenance

    def __post_init__(self) -> None:
        if self.object_psize <= 0:
            raise ValueError(f"object_psize must be > 0, got {self.object_psize}")
        if self.Npixels <= 0:
            raise ValueError(f"Npixels must be > 0, got {self.Npixels}")
        if self.Nsub <= 0:
            raise ValueError(f"Nsub must be > 0, got {self.Nsub}")

    @classmethod
    def from_dict(cls, d: "dict | None") -> "DetectorGeometryConfig":
        if not d:
            return cls()
        d = dict(d)
        pixel_size = d.pop("pixel_size", None)
        magnification = d.pop("magnification", None)
        Npixels = int(d.pop("Npixels", 510))
        Nsub = int(d.pop("Nsub", 1))
        if d:
            raise ValueError(f"[detector_geometry] unknown keys: {sorted(d)}")
        if (pixel_size is None) != (magnification is None):
            raise ValueError(
                "[detector_geometry] pixel_size and magnification must be set "
                "together (object_psize = pixel_size / magnification)"
            )
        if pixel_size is None:
            object_psize = 40e-9
        else:
            pixel_size = float(pixel_size)
            magnification = float(magnification)
            if pixel_size <= 0 or magnification <= 0:
                raise ValueError(
                    f"[detector_geometry] pixel_size ({pixel_size}) and "
                    f"magnification ({magnification}) must be > 0"
                )
            object_psize = pixel_size / magnification
        return cls(
            object_psize=object_psize,
            Npixels=Npixels,
            Nsub=Nsub,
            pixel_size=pixel_size,
            magnification=magnification,
        )
```

- [ ] **Step 4: Nest it in `SimulationConfig` + parse in `from_toml`**

In `SimulationConfig` (after the `detector` field, ~line 729) add:

```python
    # v3 (config-driven detector geometry): object-plane pitch + grid.
    detector_geometry: DetectorGeometryConfig = field(default_factory=DetectorGeometryConfig)
```

In `SimulationConfig.from_toml`, after `detector = DetectorConfig(**raw.get("detector", {}))` (~line 746) add:

```python
        detector_geometry = DetectorGeometryConfig.from_dict(raw.get("detector_geometry"))
```

and add `detector_geometry=detector_geometry,` to the `return cls(...)` kwargs (~line 768).

- [ ] **Step 5: Run the test to verify it passes**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_detector_geometry_config.py -q`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/config.py tests/test_detector_geometry_config.py
git commit -m "feat(config): [detector_geometry] block (pixel_size/magnification -> object pitch)"
```

---

### Task 2: from-config instrument builder + geometry-context extent fix

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` (add builder after `build_instrument_context` ~line 682; fix `build_geometry_context` ~lines 704-713)
- Test: append to `tests/test_detector_geometry_config.py`

- [ ] **Step 1: Write the failing test (append)**

```python
def test_from_config_default_matches_module_globals():
    import numpy as np
    import dfxm_geo.direct_space.forward_model as fm

    a = fm.build_instrument_context()  # module globals
    b = fm.build_instrument_context_from_config(
        psize=40e-9, zl_rms=fm.zl_rms, Npixels=510, Nsub=1
    )
    for name in ("psize", "zl_rms", "Npixels", "Nsub", "NN1", "NN2", "NN3",
                 "yl_start", "xl_steps", "yl_steps", "zl_steps"):
        assert getattr(a, name) == getattr(b, name), name
    assert np.array_equal(a.flat_indices, b.flat_indices)
    assert np.array_equal(a.Ud, b.Ud) and np.array_equal(a.Us, b.Us)


def test_from_config_changed_pitch_changes_grid():
    import dfxm_geo.direct_space.forward_model as fm

    b = fm.build_instrument_context_from_config(
        psize=37.6e-9, zl_rms=fm.zl_rms, Npixels=120, Nsub=1
    )
    assert b.Npixels == 120 and b.NN1 == 40 and b.NN2 == 120  # 120//3, 120
    assert b.psize == 37.6e-9
    # yl_start scales with psize*Npixels: -37.6e-9*120/2 + 37.6e-9/2
    assert abs(b.yl_start - (-37.6e-9 * 120 / 2 + 37.6e-9 / 2)) < 1e-20


def test_geometry_context_y_extent_follows_instrument():
    import numpy as np
    import dfxm_geo.direct_space.forward_model as fm

    instr = fm.build_instrument_context_from_config(
        psize=37.6e-9, zl_rms=fm.zl_rms, Npixels=120, Nsub=1
    )
    geo = fm.build_geometry_context(0.2, instr)
    # The y-extent of the ray grid must reflect the overridden instrument,
    # not the module default (510*40nm). Max |yl| ~ -instr.yl_start.
    assert abs(float(np.abs(geo.rl[1]).max()) - (-instr.yl_start)) < 1e-12
```

- [ ] **Step 2: Run to verify it fails**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_detector_geometry_config.py -q -k "from_config or y_extent"`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_instrument_context_from_config'`.

- [ ] **Step 3: Add `build_instrument_context_from_config`**

In `forward_model.py`, immediately after `build_instrument_context()` (~line 682) add:

```python
def build_instrument_context_from_config(
    *, psize: float, zl_rms: float, Npixels: int, Nsub: int
) -> InstrumentContext:
    """Build an InstrumentContext from explicit detector geometry (config-driven).

    Recomputes the derived ray-grid constants from the given psize/Npixels/Nsub,
    using the SAME expressions as the module-level derivation, so a call with the
    module defaults (40e-9, zl_rms, 510, 1) returns a context byte-identical to
    build_instrument_context(). The geometry-independent Ud/Us are reused.
    """
    nn1 = int(Npixels // 3 * Nsub)
    nn2 = int(Npixels * Nsub)
    nn3 = int(Npixels // 30 * Nsub)
    yl_start_ = -psize * Npixels / 2 + psize / (2 * Nsub)
    yi = (np.arange(nn1) // Nsub).repeat(nn3 * nn2)
    zi = np.tile((np.arange(nn2) // Nsub).repeat(nn3), nn1)
    indices_ = np.vstack((zi, yi)).T
    flat_indices_ = indices_[:, 0].astype(np.int64) * (nn1 // Nsub) + indices_[:, 1].astype(
        np.int64
    )
    return InstrumentContext(
        psize=psize,
        zl_rms=zl_rms,
        Npixels=Npixels,
        Nsub=Nsub,
        NN1=nn1,
        NN2=nn2,
        NN3=nn3,
        Ud=Ud,
        Us=Us,
        flat_indices=flat_indices_,
        yl_start=yl_start_,
        xl_steps=nn1,
        yl_steps=nn2,
        zl_steps=nn3,
    )
```

- [ ] **Step 4: Fix `build_geometry_context` to take y/z extents from the instrument**

In `build_geometry_context` (~lines 702-713), replace the body that uses the module globals `yl_range`/`zl_range` so the grid extents follow the instrument. Change:

```python
    th = float(theta_0_)
    Theta_ = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    xl_start_ = instrument.yl_start / np.tan(2 * th) / 3
    xl_range_ = -xl_start_
    rl_ = np.vstack(  # type: ignore[call-overload]
        np.mgrid[
            -xl_range_ : xl_range_ : complex(instrument.xl_steps),
            -yl_range : yl_range : complex(instrument.yl_steps),
            -zl_range : zl_range : complex(instrument.zl_steps),
        ]
    ).reshape(3, -1)
    prob_z_ = np.exp(-0.5 * (rl_[2] / instrument.zl_rms) ** 2)
```

to:

```python
    th = float(theta_0_)
    Theta_ = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    xl_start_ = instrument.yl_start / np.tan(2 * th) / 3
    xl_range_ = -xl_start_
    # y/z grid extents follow the instrument (config-driven psize/Npixels), not
    # module globals. For the default instrument these equal the module-level
    # yl_range/zl_range exactly (same expressions), so the default path is
    # byte-identical.
    yl_range_ = -instrument.yl_start
    zl_start_ = -0.5 * instrument.zl_rms * 6
    zl_range_ = -zl_start_
    rl_ = np.vstack(  # type: ignore[call-overload]
        np.mgrid[
            -xl_range_ : xl_range_ : complex(instrument.xl_steps),
            -yl_range_ : yl_range_ : complex(instrument.yl_steps),
            -zl_range_ : zl_range_ : complex(instrument.zl_steps),
        ]
    ).reshape(3, -1)
    prob_z_ = np.exp(-0.5 * (rl_[2] / instrument.zl_rms) ** 2)
```

- [ ] **Step 5: Run the new tests + a determinism smoke**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_detector_geometry_config.py -q`
Expected: all pass (10 total).

Run the forward-sampling-invariance + find-hg parity tests that build `InstrumentContext`/geometry directly, to confirm the `build_geometry_context` edit is byte-safe:
`& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_forward_sampling_invariance.py tests/test_find_hg_kernel_parity.py tests/test_find_hg_oblique_zshift.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/direct_space/forward_model.py tests/test_detector_geometry_config.py
git commit -m "feat(forward): build_instrument_context_from_config + instrument-driven grid extents"
```

---

### Task 3: Wire the forward orchestrator

**Files:**
- Modify: `src/dfxm_geo/orchestrator.py` (~lines 359, 377, 395-400, 431-432)
- Test: append a forward-override test to `tests/test_detector_geometry_config.py`

- [ ] **Step 1: Write the failing test (append)**

```python
@pytest.mark.slow
def test_forward_run_honors_npixels_override(tmp_path: Path):
    """A [detector_geometry] Npixels override changes the output image dims."""
    import h5py

    from dfxm_geo.io.hdf5 import DETECTOR_INTERNAL_PATH
    from dfxm_geo.pipeline import SimulationConfig, run_simulation

    toml = (
        "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n"
        'backend = "analytic"\nbeamstop = false\naperture = false\n'
        "zeta_v_fwhm = 5.3e-4\nzeta_h_fwhm = 0.0\n"
        "NA_rms = 3.1106382978723403e-4\neps_rms = 6.0e-5\n\n"
        '[geometry]\nmode = "simplified"\n\n'
        '[crystal]\nmode = "centered"\nlattice = "cubic"\na = 4.05e-10\n\n'
        "[scan.phi]\nvalue = 1.75e-4\n\n"
        '[detector]\nmodel = "ideal"\n\n'
        "[detector_geometry]\npixel_size = 0.65e-6\nmagnification = 17.31\nNpixels = 120\n\n"
        "[io]\ninclude_perfect_crystal = false\nwrite_strain_provenance = false\n"
    )
    cfg_path = tmp_path / "ov.toml"
    cfg_path.write_text(toml, encoding="utf-8")
    cfg = SimulationConfig.from_toml(cfg_path)
    out = tmp_path / "out"
    run_simulation(cfg, out)
    det = out / "scan0001" / "dfxm_sim_detector_0000.h5"
    with h5py.File(det, "r") as f:
        img = f[DETECTOR_INTERNAL_PATH][0]
    # Npixels=120 -> (NN2, NN1) = (120, 40); default would be (510, 170).
    assert img.shape == (120, 40), img.shape
```

- [ ] **Step 2: Run to verify it fails**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_detector_geometry_config.py -q -m slow -k npixels_override`
Expected: FAIL — image shape is `(510, 170)` (override ignored).

- [ ] **Step 3: Build the instrument from config and thread it**

In `orchestrator.py`, replace the fov line (~359):

```python
    fov_lateral_um = fm.Npixels * fm.psize * 1e6  # m -> um
```

with:

```python
    # Config-driven detector geometry: build the run's instrument once from
    # [detector_geometry] (defaults reproduce the module globals exactly) and
    # thread it through fov, build_forward_context, and the wall Find_Hg.
    instr = fm.build_instrument_context_from_config(
        psize=config.detector_geometry.object_psize,
        zl_rms=fm.zl_rms,
        Npixels=config.detector_geometry.Npixels,
        Nsub=config.detector_geometry.Nsub,
    )
    fov_lateral_um = instr.Npixels * instr.psize * 1e6  # m -> um
```

Replace the effective-config print line (~377):

```python
        f"  Nsub={fm.Nsub}  Npixels={fm.Npixels}  NN1={fm.NN1}  NN2={fm.NN2}\n"
```

with:

```python
        f"  Nsub={instr.Nsub}  Npixels={instr.Npixels}  NN1={instr.NN1}  NN2={instr.NN2}\n"
```

Pass the instrument to the single-reflection `build_forward_context` (~395):

```python
        ctx = fm.build_forward_context(
            run_theta(config),
            res,
            config.reciprocal.hkl,
            instrument=instr,
            cell=_mount_cell(config),
        )
```

Replace the wall `Find_Hg` psize/zl_rms (~431-432):

```python
                fm.psize,
                fm.zl_rms,
```

with:

```python
                instr.psize,
                instr.zl_rms,
```

- [ ] **Step 4: Run the override test**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_detector_geometry_config.py -q -m slow -k npixels_override`
Expected: PASS — image shape `(120, 40)`.

- [ ] **Step 5: Run mypy on the touched modules**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/orchestrator.py tests/test_detector_geometry_config.py
git commit -m "feat(orchestrator): single-reflection forward consumes [detector_geometry]"
```

---

### Task 4: Byte-identity gate (defaults unchanged)

**Files:** none (verification only).

- [ ] **Step 1: Clean Fg caches**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -c "import glob,os;[os.remove(p) for p in glob.glob('direct_space/deformation_gradient_tensors/Fg_*.npy')];print('cleaned')"`

- [ ] **Step 2: Run the determinism + golden gates**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_fcc_bit_identity.py tests/test_cubic_bit_identity.py tests/test_structure_goldens.py -q -m slow`
Expected: all pass (FCC/BCC determinism, FCC wall, structure goldens unchanged — proves the default path is byte-identical).

- [ ] **Step 3: Run the default suite**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q`
Expected: green (≈1129 passed given the new non-slow config tests, 2 skipped, 1 xfailed). If any unexpected failure, STOP and investigate before continuing.

- [ ] **Step 4: Commit (only if any incidental fixups were needed; otherwise skip)**

No code change expected in this task. If the gates are green, proceed.

---

### Task 5: counts_scale re-measurement

**Files:**
- Modify: `docs/calibration/derive_counts_scale.py`
- Modify: `docs/m4-validation-report.md`

- [ ] **Step 1: Parametrize the sim geometry in the calibration script**

In `derive_counts_scale.py`, add near the other parameters (~line 102):

```python
# Matched-geometry re-measurement (2026-06-15): the data set is 10x focusing,
# so the data's object-plane pitch is (camera 6.5 um / 10x) / M(17.31) = 37.6 nm,
# vs the forward default 40 nm. We report counts_scale at BOTH geometries.
MATCHED_PIXEL_SIZE = 6.5e-6 / 10.0   # detector-plane effective pixel (m)
MATCHED_MAGNIFICATION = 17.31        # X-ray objective M (MAIN_X/OBX - 1)
```

Change `simulate_weak_beam` to accept an optional geometry and build the config with it:

```python
def simulate_weak_beam(
    detector_geometry: "DetectorGeometryConfig | None" = None,
) -> tuple[np.ndarray, float, float, float, int]:
```

and inside, add `detector_geometry` to the `SimulationConfig(...)` kwargs:

```python
        cfg = SimulationConfig(
            crystal=CrystalConfig(mode="centered", centered=CenteredCrystalConfig()),
            scan=ScanConfig(phi=AxisScanConfig(value=WEAK_BEAM_PHI)),
            io=IOConfig(include_perfect_crystal=False, write_strain_provenance=False),
            detector=DetectorConfig(model="ideal"),
            reciprocal=ReciprocalConfig(hkl=REFLECTION_HKL, keV=ENERGY_KEV, backend="mc"),
            detector_geometry=detector_geometry or DetectorGeometryConfig(),
        )
```

Add `DetectorGeometryConfig` to the imports from `dfxm_geo.pipeline`.

- [ ] **Step 2: Run both geometries in `main` and print a comparison**

In `main`, after the existing single derivation, wrap the sim + counts_scale + guards into a small helper and call it twice (default + matched). Minimal change: after the existing block that prints `counts_scale` and Guards, add:

```python
    print("=" * 70)
    print("MATCHED 10x geometry re-measurement (object_psize ~ 37.6 nm)")
    print("=" * 70)
    matched = DetectorGeometryConfig.from_dict(
        {"pixel_size": MATCHED_PIXEL_SIZE, "magnification": MATCHED_MAGNIFICATION}
    )
    print(f"object_psize = {matched.object_psize * 1e9:.2f} nm (default 40.00 nm)")
    _img2, sim_integral2, sim_peak2, fov_fraction2, sim_npix2 = simulate_weak_beam(matched)
    counts_scale2 = adu_integral / (sim_integral2 * EXPOSURE_TIME)
    core_peak_adu2 = sim_peak2 * counts_scale2 * EXPOSURE_TIME
    guard_a2 = GUARD_A_LO <= core_peak_adu2 <= GUARD_A_HI
    print(f"sim_integral (matched) : {sim_integral2:.6g}  (feature {sim_npix2} px)")
    print(f"counts_scale (matched) : {counts_scale2:.6g}")
    print(
        f"Guard A (matched): core-peak {core_peak_adu2:.1f} ADU "
        f"(target {GUARD_A_LO:.0f}-{GUARD_A_HI:.0f}) -> {'PASS' if guard_a2 else 'FAIL'}"
    )
    print(
        f"delta vs default: counts_scale {counts_scale2 / counts_scale:.3f}x, "
        f"feature px {sim_npix2}/{sim_npix} = {sim_npix2 / sim_npix:.3f}x"
    )
```

- [ ] **Step 3: Run the calibration script**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" docs/calibration/derive_counts_scale.py`
Expected: prints the default and matched `counts_scale`, both Guard A results, and the deltas. Capture the full output.

> If the script fails because the local ID03 data or the MC kernel is absent (`FileNotFoundError` / kernel lookup error), STOP and report — the re-measurement cannot be completed in this environment; record that fact in the report instead of numbers.

- [ ] **Step 4: Record the result in the validation report**

Append a section to `docs/m4-validation-report.md`:

```markdown
## counts_scale re-measurement at matched 10x geometry (2026-06-15)

The forward detector geometry is now config-driven (`[detector_geometry]`,
`pixel_size`/`magnification`). `derive_counts_scale.py` was re-run at the data's
true 10x object-plane pitch (camera 6.5 µm / 10x / M=17.31 ≈ 37.6 nm) alongside
the default 40 nm.

| Geometry | object_psize | sim feature px | counts_scale | Guard A (core-peak ADU) |
|---|---|---|---|---|
| default | 40.0 nm | <fill> | <fill> | <fill> PASS/FAIL |
| matched 10x | 37.6 nm | <fill> | <fill> | <fill> PASS/FAIL |

**Verdict:** <one sentence — is counts_scale still a blocker, and is the residual
pitch or FOV-fraction?> The shipped `DetectorConfig.counts_scale` default is
UNCHANGED in this pass (Sina reviews + pins separately).
```

Fill the `<...>` cells from Step 3's output.

- [ ] **Step 5: Commit**

```bash
git add docs/calibration/derive_counts_scale.py docs/m4-validation-report.md
git commit -m "calib: re-measure counts_scale at matched 10x geometry (config-driven psize)"
```

---

### Task 6: Wrap-up

- [ ] **Step 1: Clean Fg caches**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -c "import glob,os;[os.remove(p) for p in glob.glob('direct_space/deformation_gradient_tensors/Fg_*.npy')];print('cleaned')"`

- [ ] **Step 2: Confirm branch state**

Run: `git log --oneline main..HEAD` and `git status`.
Expected: 4 commits (Tasks 1, 2, 3, 5), clean tree apart from pre-existing untracked scratch. No push, no tag — awaits Sina's merge call.

- [ ] **Step 3: Report** the counts_scale before/after + the blocker verdict, and that `DetectorConfig.counts_scale` is unchanged.

---

## Self-review notes

- **Spec coverage:** schema (spec §1) = Task 1; from-config builder + the `build_geometry_context` extent fix (§2) = Task 2; orchestrator seam (§3) = Task 3; byte-identity (§Testing/Risks) = Task 4; counts_scale re-measurement (§4) = Task 5; no-re-pin honored (counts_scale default untouched). The spec's three read-site fixes: fov (`:359`) and wall `Find_Hg` (`:431`) are in Task 3; the HDF5 provenance `psize` (`hdf5.py:1047`) and the debug print are handled — the print in Task 3, the HDF5 provenance left reading the module global (cosmetic; noted as a follow-up below).
- **Deviations from the spec, documented:** (a) only the single-reflection forward path is config-driven; multi-reflection (`_context_for_run`), identify, and forward z-scan (`Z_shift`) keep the module-global grid (byte-identical for default configs). (b) HDF5 provenance `psize` still records the module default, not the run's config pitch — a cosmetic mismatch for overridden runs. Both are recorded in the validation report's follow-ups.
- **Byte-identity:** Task 2's context-equality test proves `from_config(40e-9,…,510,1)` == globals snapshot; Task 4 proves the default forward/identify/golden path is unchanged. The `build_geometry_context` extent edit uses the same expressions fed by `instrument.zl_rms`/`yl_start` (== module values for default).
- **Type/name consistency:** `DetectorGeometryConfig.{object_psize, Npixels, Nsub, pixel_size, magnification}`, `build_instrument_context_from_config(*, psize, zl_rms, Npixels, Nsub)`, `config.detector_geometry` are defined in Task 1/2 and used identically in Tasks 3/5.
```
