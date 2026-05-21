# Scan modes + crystal layouts (sub-projects B + C) implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the forward + identify pipelines so the scan dimension is any subset of motor axes (phi, chi, two_dtheta, z) configured independently, and the crystal layout is one of three mutually-exclusive modes (`centered`, `wall`, `random_dislocations`). One combined PR; hard cut-over of the flat `ScanConfig` / `CrystalConfig` shapes.

**Architecture:** New per-axis `AxisScanConfig` + per-mode crystal sub-block dataclasses in `pipeline.py`; new `build_scan_grid` + `build_dislocation_population` dispatchers in `direct_space/forward_model.py`; new `io/sidecar.py` writes random_dislocations realized params as JSON; HDF5 `/N.1` gains three new attrs (`scan_mode`, `scanned_axes`, `crystal_mode`); 9 TOML configs and all inline test fixtures migrated in the same PR.

**Tech Stack:** Python 3.11+, `numpy`, `numpy.random.Generator`, `tomllib` (stdlib), `json` (stdlib), `pytest` with `monkeypatch` + `tmp_path`, `mypy --strict`.

**Spec:** `docs/superpowers/specs/2026-05-21-scan-modes-crystal-layouts-design.md` (commit `6666390`).

---

## Parallel-dispatch hints (for subagent-driven execution)

Per `[[feedback-parallelize-subagents]]`:

- **Tasks 1–5 are sequential** — all touch `src/dfxm_geo/pipeline.py` (same file → must sequence). T1 (AxisScanConfig), T2 (ScanConfig + from_dict), T3 (derived_mode_name), T4 (crystal-mode sub-block dataclasses), T5 (CrystalConfig union + from_dict).
- **Tasks 6, 7, 8 are sequential** — all touch `src/dfxm_geo/direct_space/forward_model.py`.
- **Task 9** (new file `src/dfxm_geo/io/sidecar.py`) can run **in parallel with Tasks 1–8** — completely independent file.
- **Task 10 (rewire `run_simulation`)** depends on Tasks 1–9.
- **Task 11 (`_dataclass_to_toml_str`)** depends on Task 10 (must see final dataclass shapes).
- **Task 12 (rewire identification)** depends on Tasks 1–5 (needs new `ScanConfig`) but is otherwise independent — can run **in parallel with Tasks 6–11** if test files don't collide. T12 touches `pipeline.py` for identification-only edits; if T10/T11 are still in flight, sequence after them.
- **Task 13 (HDF5 attrs)** can run **in parallel with T10–T12** — touches only `src/dfxm_geo/io/hdf5.py`.
- **Task 14 (migrate 9 TOML configs)** depends on Tasks 1–5 + Task 10 + Task 12 (final TOML schema must be locked). Can split into parallel sub-tasks per config file.
- **Task 15 (carry-forward inline TOML fixtures in tests)** depends on Tasks 10 + 12 — same file collision rules as D's Task 6: split per test file for parallelism.
- **Task 16 (final whole-PR review)** depends on everything.

---

## File Structure

**Modify:**
- `src/dfxm_geo/pipeline.py` — replace `ScanConfig` (lines 65–70) + `CrystalConfig` (lines 51–62) with new shapes; replace `IdentificationScanConfig` (lines 182–189); rewire `from_toml` parsers; rewire `run_simulation` (lines 330–404) + `run_identification` to call new dispatchers; update `_dataclass_to_toml_str` (lines 407+). Net ~+250 LOC after deletions.
- `src/dfxm_geo/direct_space/forward_model.py` — add `build_scan_grid` (~30 LOC); add `build_dislocation_population` with three-mode dispatch (~120 LOC); preserve the existing `Find_Hg` / `Fd_find_*` calls behind the wall-mode branch. Net ~+150 LOC.
- `src/dfxm_geo/io/hdf5.py` — `write_simulation_h5` gains three new attrs on `/N.1` (`scan_mode`, `scanned_axes`, `crystal_mode`); signature accepts the new `ScanConfig` + `CrystalConfig` rather than flat phi_range/phi_steps args (~40 LOC delta).
- `configs/default.toml` — fully rewritten to new schema. Active mode = `centered`; commented-out `[crystal.wall]` and `[crystal.random_dislocations]` blocks underneath. `[scan]` populated with `[scan.phi]` + `[scan.chi]` preserving the current 2D mosa shape.
- `configs/identification_single.toml`, `identification_multi.toml`, `identification_zscan.toml` — migrate `[scan]` to per-axis primitives; move `poisson_noise`, `rng_seed`, `intensity_scale` to new `[noise]` block; `[zscan]` z_offsets_um stays (out-of-scope per spec).
- `configs/variants/dis_0p25.toml`, `dis_0p5.toml`, `dis_1.toml`, `dis_2.toml`, `sample_remount_S2.toml` — migrate `[crystal]` to `mode = "wall"` + `[crystal.wall]` sub-block; migrate `[scan]` to per-axis (radians instead of degrees).

**Create:**
- `src/dfxm_geo/io/sidecar.py` (~60 LOC) — `write_random_dislocations_sidecar` JSON writer.
- `tests/test_pipeline_scan_modes.py` (~250 LOC) — `AxisScanConfig`, `ScanConfig.from_dict`, `derived_mode_name`, `scanned_axes`, `is_scanned`, error paths.
- `tests/test_pipeline_crystal_modes.py` (~250 LOC) — `CrystalConfig.from_dict` for each of the three modes; sibling-sub-block rejection; missing-required-field; mode-validation cases.
- `tests/test_forward_dispatch.py` (~200 LOC) — `build_scan_grid` shape/order tests; `build_dislocation_population` for centered + wall (preserves existing Fd_find golden) + random_dislocations (seeded determinism + sidecar metadata + min_distance enforcement + retry-budget exhaustion).
- `tests/test_sidecar.py` (~80 LOC) — JSON write + round-trip via `json.load`.
- `tests/test_configs_load_under_new_schema.py` (~80 LOC) — smoke loader for all 9 migrated TOML configs.

**Update:**
- `tests/test_pipeline.py` — rewrite all inline TOML fixtures to the new schema (~20 fixtures). Same carry-forward sweep pattern as D's Task 6.
- `tests/test_pipeline_identification.py` — rewrite inline TOML fixtures; replace `IdentificationScanConfig` usages with `ScanConfig` + `IdentificationNoiseConfig`.
- `tests/test_pipeline_multi_reflection.py` — update fixtures (post-D file).
- `tests/test_hdf5_provenance.py` — add assertions for the three new attrs.

**Delete (Python types — replaced by new shapes):**
- Old flat `ScanConfig` (lines 65–70 of `pipeline.py`).
- Old flat `CrystalConfig` (lines 51–62 of `pipeline.py`).
- `IdentificationScanConfig` (lines 182–189 of `pipeline.py`).

**Total**: ~1300–1500 LOC (~600 production, ~900 tests).

**Working directory:** `C:/Users/borgi/Documents/GM-reworked/Geometrical_Optics_master/`
**Branch:** `chore/spec-scan-crystal-modes` (current). Code commits land on this branch; spec commit `6666390` already on it.
**Python interpreter:** `C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe` (bash `python` is 2.7 — do NOT use it).
**Run tests:** `& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest -q` from the working directory.

---

## Task 1: `AxisScanConfig` dataclass

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (insert new dataclass before existing `ScanConfig` at line 65)
- Test: `tests/test_pipeline_scan_modes.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline_scan_modes.py`:

```python
"""Unit tests for the per-axis scan primitives (sub-project B)."""

from __future__ import annotations

import pytest

from dfxm_geo.pipeline import AxisScanConfig


class TestAxisScanConfig:
    def test_default_is_fixed_at_zero(self) -> None:
        axis = AxisScanConfig()
        assert axis.value == 0.0
        assert axis.range is None
        assert axis.steps is None
        assert not axis.is_scanned

    def test_fixed_with_nonzero_value(self) -> None:
        axis = AxisScanConfig(value=1.5e-4)
        assert axis.value == 1.5e-4
        assert not axis.is_scanned

    def test_scanned_centered_on_zero(self) -> None:
        axis = AxisScanConfig(range=1e-3, steps=61)
        assert axis.is_scanned
        assert axis.value == 0.0
        assert axis.range == 1e-3
        assert axis.steps == 61

    def test_scanned_with_offset(self) -> None:
        axis = AxisScanConfig(value=1.5e-4, range=1e-3, steps=61)
        assert axis.is_scanned
        assert axis.value == 1.5e-4

    def test_range_without_steps_rejected(self) -> None:
        with pytest.raises(ValueError, match="both `range` and `steps`"):
            AxisScanConfig(range=1e-3)

    def test_steps_without_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="both `range` and `steps`"):
            AxisScanConfig(steps=61)

    def test_zero_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="`range` must be > 0"):
            AxisScanConfig(range=0.0, steps=61)

    def test_negative_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="`range` must be > 0"):
            AxisScanConfig(range=-1e-3, steps=61)

    def test_steps_below_two_rejected(self) -> None:
        with pytest.raises(ValueError, match="`steps` must be >= 2"):
            AxisScanConfig(range=1e-3, steps=1)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline_scan_modes.py -v
```

Expected: ImportError — `AxisScanConfig` does not exist in `dfxm_geo.pipeline`.

- [ ] **Step 3: Implement `AxisScanConfig`**

In `src/dfxm_geo/pipeline.py`, insert before the existing `ScanConfig` definition:

```python
@dataclass
class AxisScanConfig:
    """Per-motor-axis scan primitive (sub-project B).

    Each motor axis (phi, chi, two_dtheta, z) is independently fixed
    at `value` or scanned over `[value-range, value+range]` with `steps`
    samples (linspace). Both `range` and `steps` must be present
    together for the axis to be scanned, or both absent for fixed.
    """

    value: float = 0.0
    range: float | None = None
    steps: int | None = None

    def __post_init__(self) -> None:
        if (self.range is None) != (self.steps is None):
            raise ValueError(
                "AxisScanConfig must specify both `range` and `steps`, or neither "
                "(fixed at `value`). Got range=%r, steps=%r" % (self.range, self.steps)
            )
        if self.range is not None:
            if self.range <= 0:
                raise ValueError(f"`range` must be > 0; got {self.range!r}")
            if self.steps is None or self.steps < 2:
                raise ValueError(f"`steps` must be >= 2 when range is set; got {self.steps!r}")

    @property
    def is_scanned(self) -> bool:
        return self.range is not None and self.steps is not None
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline_scan_modes.py::TestAxisScanConfig -v
```

Expected: 9/9 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_pipeline_scan_modes.py
git commit -m "feat(pipeline): add AxisScanConfig per-axis scan primitive (B Task 1)"
```

---

## Task 2: `ScanConfig` dataclass + `from_dict` parser

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (replace old `ScanConfig` at lines 65–70 with the new one)
- Test: `tests/test_pipeline_scan_modes.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_pipeline_scan_modes.py`:

```python
from dfxm_geo.pipeline import ScanConfig


class TestScanConfigFromDict:
    def test_empty_dict_all_axes_fixed_at_zero(self) -> None:
        cfg = ScanConfig.from_dict({})
        assert not cfg.phi.is_scanned
        assert not cfg.chi.is_scanned
        assert not cfg.two_dtheta.is_scanned
        assert not cfg.z.is_scanned

    def test_none_dict_all_axes_fixed_at_zero(self) -> None:
        cfg = ScanConfig.from_dict(None)
        assert not cfg.phi.is_scanned

    def test_mosa_grid(self) -> None:
        cfg = ScanConfig.from_dict(
            {
                "phi": {"range": 6e-4, "steps": 61},
                "chi": {"range": 2e-3, "steps": 61},
            }
        )
        assert cfg.phi.is_scanned and cfg.phi.range == 6e-4
        assert cfg.chi.is_scanned and cfg.chi.range == 2e-3
        assert not cfg.two_dtheta.is_scanned
        assert not cfg.z.is_scanned

    def test_single_image_with_phi_offset(self) -> None:
        cfg = ScanConfig.from_dict({"phi": {"value": 1.5e-4}})
        assert not cfg.phi.is_scanned
        assert cfg.phi.value == 1.5e-4

    def test_rocking_strain(self) -> None:
        cfg = ScanConfig.from_dict(
            {
                "phi": {"range": 6e-4, "steps": 61},
                "two_dtheta": {"range": 5e-4, "steps": 41},
            }
        )
        assert cfg.phi.is_scanned
        assert cfg.two_dtheta.is_scanned
        assert not cfg.chi.is_scanned

    def test_unknown_axis_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown scan axis 'omega'"):
            ScanConfig.from_dict({"omega": {"range": 1e-3, "steps": 21}})

    def test_axis_value_propagated_to_axis_config(self) -> None:
        cfg = ScanConfig.from_dict({"chi": {"value": 1e-5, "range": 2e-3, "steps": 41}})
        assert cfg.chi.value == 1e-5
        assert cfg.chi.is_scanned

    def test_invalid_axis_data_propagates_post_init(self) -> None:
        with pytest.raises(ValueError, match="`range` must be > 0"):
            ScanConfig.from_dict({"phi": {"range": 0.0, "steps": 21}})
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline_scan_modes.py::TestScanConfigFromDict -v
```

Expected: ImportError — the new `ScanConfig` shape doesn't exist yet (old one has flat phi_range/phi_steps, not the new per-axis structure).

- [ ] **Step 3: Replace old `ScanConfig` with new shape**

In `src/dfxm_geo/pipeline.py`, **delete** the old flat `ScanConfig` (lines 65–70 in current main):

```python
# DELETE this old shape:
# @dataclass
# class ScanConfig:
#     phi_range: float
#     phi_steps: int
#     chi_range: float
#     chi_steps: int
```

Insert the new shape after `AxisScanConfig`:

```python
_CANONICAL_AXES = ("phi", "chi", "two_dtheta", "z")


@dataclass
class ScanConfig:
    """Per-axis scan primitives (sub-project B).

    Each motor axis is independently fixed or scanned. The "scan mode"
    label is derived from which axes carry range+steps — see
    `derived_mode_name`.
    """

    phi: AxisScanConfig = field(default_factory=AxisScanConfig)
    chi: AxisScanConfig = field(default_factory=AxisScanConfig)
    two_dtheta: AxisScanConfig = field(default_factory=AxisScanConfig)
    z: AxisScanConfig = field(default_factory=AxisScanConfig)

    @classmethod
    def from_dict(cls, data: dict | None) -> ScanConfig:
        if not data:
            return cls()
        unknown = set(data.keys()) - set(_CANONICAL_AXES)
        if unknown:
            raise ValueError(
                f"unknown scan axis {sorted(unknown)[0]!r}; "
                f"expected one of {_CANONICAL_AXES}"
            )
        kwargs = {
            axis: AxisScanConfig(**data[axis]) for axis in _CANONICAL_AXES if axis in data
        }
        return cls(**kwargs)
```

Note: `from __future__ import annotations` is already imported at line 15, so the forward reference works.

- [ ] **Step 4: Run tests to verify they pass**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline_scan_modes.py::TestScanConfigFromDict -v
```

Expected: 8/8 PASS. **But existing tests using the old `ScanConfig(phi_range=..., phi_steps=..., chi_range=..., chi_steps=...)` will now fail.** Those are fixed in Task 15. Run only the new test class for now.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_pipeline_scan_modes.py
git commit -m "feat(pipeline): replace flat ScanConfig with per-axis primitives (B Task 2)"
```

---

## Task 3: `ScanConfig.derived_mode_name` + introspection helpers

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (extend `ScanConfig`)
- Test: `tests/test_pipeline_scan_modes.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_pipeline_scan_modes.py`:

```python
class TestDerivedModeName:
    def _scanned(self, **axes_with_range: tuple[float, int]) -> ScanConfig:
        """Build a ScanConfig where named axes are scanned, others fixed."""
        data = {name: {"range": r, "steps": s} for name, (r, s) in axes_with_range.items()}
        return ScanConfig.from_dict(data)

    def test_no_axes_scanned_is_single(self) -> None:
        assert ScanConfig().derived_mode_name() == "single"

    def test_phi_only_is_rocking(self) -> None:
        assert self._scanned(phi=(6e-4, 61)).derived_mode_name() == "rocking"

    def test_chi_only_is_rolling(self) -> None:
        assert self._scanned(chi=(2e-3, 61)).derived_mode_name() == "rolling"

    def test_two_dtheta_only_is_strain(self) -> None:
        assert self._scanned(two_dtheta=(5e-4, 41)).derived_mode_name() == "strain"

    def test_z_only_is_layer(self) -> None:
        assert self._scanned(z=(1e-6, 5)).derived_mode_name() == "layer"

    def test_phi_chi_is_mosa(self) -> None:
        assert self._scanned(phi=(6e-4, 61), chi=(2e-3, 61)).derived_mode_name() == "mosa"

    def test_phi_chi_two_dtheta_is_mosa_strain(self) -> None:
        assert (
            self._scanned(phi=(6e-4, 61), chi=(2e-3, 61), two_dtheta=(5e-4, 41)).derived_mode_name()
            == "mosa_strain"
        )

    def test_phi_chi_z_is_mosa_layer(self) -> None:
        assert (
            self._scanned(phi=(6e-4, 61), chi=(2e-3, 61), z=(1e-6, 5)).derived_mode_name()
            == "mosa_layer"
        )

    def test_phi_chi_two_dtheta_z_is_mosa_strain_layer(self) -> None:
        assert (
            self._scanned(
                phi=(6e-4, 61), chi=(2e-3, 61), two_dtheta=(5e-4, 41), z=(1e-6, 5)
            ).derived_mode_name()
            == "mosa_strain_layer"
        )

    def test_non_canonical_combo_concatenates_in_axis_order(self) -> None:
        # phi + two_dtheta = "rocking_strain"  (chi missing → not mosa)
        assert (
            self._scanned(phi=(6e-4, 61), two_dtheta=(5e-4, 41)).derived_mode_name()
            == "rocking_strain"
        )
        # chi + z = "rolling_layer"
        assert (
            self._scanned(chi=(2e-3, 61), z=(1e-6, 5)).derived_mode_name() == "rolling_layer"
        )
        # phi + chi + z + two_dtheta hits the pre-canonized order, not raw concat
        # but phi + z (no chi, no two_dtheta) is "rocking_layer"
        assert self._scanned(phi=(6e-4, 61), z=(1e-6, 5)).derived_mode_name() == "rocking_layer"


class TestScannedAxesAndIsScanned:
    def test_scanned_axes_empty(self) -> None:
        assert ScanConfig().scanned_axes() == ()

    def test_scanned_axes_ordered_canonically(self) -> None:
        cfg = ScanConfig.from_dict(
            {
                # Insertion order: z, phi, two_dtheta (deliberately not canonical)
                "z": {"range": 1e-6, "steps": 5},
                "phi": {"range": 6e-4, "steps": 61},
                "two_dtheta": {"range": 5e-4, "steps": 41},
            }
        )
        assert cfg.scanned_axes() == ("phi", "two_dtheta", "z")

    def test_is_scanned_per_axis(self) -> None:
        cfg = ScanConfig.from_dict({"phi": {"range": 6e-4, "steps": 61}})
        assert cfg.is_scanned("phi")
        assert not cfg.is_scanned("chi")
        assert not cfg.is_scanned("two_dtheta")
        assert not cfg.is_scanned("z")

    def test_is_scanned_unknown_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown axis 'omega'"):
            ScanConfig().is_scanned("omega")
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline_scan_modes.py::TestDerivedModeName tests/test_pipeline_scan_modes.py::TestScannedAxesAndIsScanned -v
```

Expected: AttributeError — `ScanConfig.derived_mode_name`, `.scanned_axes`, `.is_scanned` not defined.

- [ ] **Step 3: Implement the introspection methods**

Add to `ScanConfig` in `src/dfxm_geo/pipeline.py`:

```python
_AXIS_TO_LABEL = {
    "phi": "rocking",
    "chi": "rolling",
    "two_dtheta": "strain",
    "z": "layer",
}

_PRE_CANONIZED_MODE_NAMES: dict[frozenset[str], str] = {
    frozenset(): "single",
    frozenset({"phi", "chi"}): "mosa",
    frozenset({"phi", "chi", "two_dtheta"}): "mosa_strain",
    frozenset({"phi", "chi", "z"}): "mosa_layer",
    frozenset({"phi", "chi", "two_dtheta", "z"}): "mosa_strain_layer",
}


# Inside ScanConfig:
    def scanned_axes(self) -> tuple[str, ...]:
        """Names of motor axes that carry a range+steps (in canonical order)."""
        return tuple(a for a in _CANONICAL_AXES if getattr(self, a).is_scanned)

    def is_scanned(self, axis: str) -> bool:
        if axis not in _CANONICAL_AXES:
            raise ValueError(f"unknown axis {axis!r}; expected one of {_CANONICAL_AXES}")
        return getattr(self, axis).is_scanned

    def derived_mode_name(self) -> str:
        """Derive the scan-mode label from which axes are scanned.

        Pre-canonized: single, rocking, rolling, strain, layer, mosa,
        mosa_strain, mosa_layer, mosa_strain_layer. All other combos
        are the 1D labels concatenated in canonical axis order.
        """
        scanned = self.scanned_axes()
        key = frozenset(scanned)
        if key in _PRE_CANONIZED_MODE_NAMES:
            return _PRE_CANONIZED_MODE_NAMES[key]
        if len(scanned) == 1:
            return _AXIS_TO_LABEL[scanned[0]]
        return "_".join(_AXIS_TO_LABEL[a] for a in scanned)
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline_scan_modes.py -v
```

Expected: all classes PASS (TestAxisScanConfig + TestScanConfigFromDict + TestDerivedModeName + TestScannedAxesAndIsScanned).

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_pipeline_scan_modes.py
git commit -m "feat(pipeline): derive scan-mode name from scanned axes (B Task 3)"
```

---

## Task 4: Crystal-mode sub-block dataclasses

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (insert before existing `CrystalConfig` at line 51)
- Test: `tests/test_pipeline_crystal_modes.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline_crystal_modes.py`:

```python
"""Unit tests for the per-mode crystal sub-block dataclasses (sub-project C)."""

from __future__ import annotations

import pytest

from dfxm_geo.pipeline import (
    CenteredCrystalConfig,
    RandomDislocationsConfig,
    WallCrystalConfig,
)


class TestCenteredCrystalConfig:
    def test_constructs_with_valid_111_b_n_t(self) -> None:
        # b = [1, -1, 0]; n = [1, 1, 1]; b·n = 0 → valid.
        # t = [1, 1, -2] is the canonical line direction for this slip system.
        cfg = CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 1, -2))
        assert cfg.b == (1, -1, 0)
        assert cfg.n == (1, 1, 1)
        assert cfg.t == (1, 1, -2)

    def test_b_not_perpendicular_to_n_rejected(self) -> None:
        # b·n = 1 + 1 + 1 = 3 ≠ 0
        with pytest.raises(ValueError, match="Burgers vector .* must be perpendicular"):
            CenteredCrystalConfig(b=(1, 1, 1), n=(1, 1, 1), t=(0, 1, -1))

    def test_t_not_consistent_with_n_cross_b_rejected(self) -> None:
        # n × b is perpendicular to both n and b; if t isn't parallel, reject.
        # For n=(1,1,1), b=(1,-1,0): n × b = (1, 1, -2). t=(1,0,0) is not parallel.
        with pytest.raises(ValueError, match="line direction .* must be parallel"):
            CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 0, 0))


class TestWallCrystalConfig:
    def test_constructs_with_defaults(self) -> None:
        cfg = WallCrystalConfig()
        assert cfg.dis == 4.0
        assert cfg.ndis == 151
        assert cfg.sample_remount == "S1"

    def test_custom_remount(self) -> None:
        cfg = WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S2")
        assert cfg.sample_remount == "S2"

    def test_invalid_remount_rejected(self) -> None:
        with pytest.raises(ValueError, match="sample_remount must be one of"):
            WallCrystalConfig(sample_remount="S9")


class TestRandomDislocationsConfig:
    def test_minimum_ndis_is_one(self) -> None:
        cfg = RandomDislocationsConfig(ndis=1)
        assert cfg.ndis == 1
        assert cfg.sigma is None
        assert cfg.min_distance is None
        assert cfg.seed is None

    def test_ndis_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="`ndis` must be >= 1"):
            RandomDislocationsConfig(ndis=0)

    def test_negative_sigma_rejected(self) -> None:
        with pytest.raises(ValueError, match="`sigma` must be > 0"):
            RandomDislocationsConfig(ndis=4, sigma=-1.0)

    def test_negative_min_distance_rejected(self) -> None:
        with pytest.raises(ValueError, match="`min_distance` must be >= 0"):
            RandomDislocationsConfig(ndis=4, min_distance=-1.0)

    def test_all_fields_set(self) -> None:
        cfg = RandomDislocationsConfig(ndis=4, sigma=5.0, min_distance=4.0, seed=42)
        assert cfg.ndis == 4
        assert cfg.sigma == 5.0
        assert cfg.min_distance == 4.0
        assert cfg.seed == 42
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline_crystal_modes.py -v
```

Expected: ImportError — none of the three classes exist.

- [ ] **Step 3: Implement the three sub-block dataclasses**

In `src/dfxm_geo/pipeline.py`, **replace** the old flat `CrystalConfig` (lines 51–62) with:

```python
@dataclass
class CenteredCrystalConfig:
    """Single dislocation at the origin (sub-project C, mode='centered').

    The Ud rotation matrix is built from (b, n, t):
      - b = Burgers vector indices
      - n = slip-plane normal indices
      - t = dislocation line direction indices

    Geometric constraints (validated):
      - b · n = 0   (Burgers vector lies in slip plane)
      - t parallel to (n × b)  (line direction consistent with slip system)
    """

    b: tuple[int, int, int]
    n: tuple[int, int, int]
    t: tuple[int, int, int]

    def __post_init__(self) -> None:
        b = self.b
        n = self.n
        t = self.t
        # b · n == 0 (exact, since these are integer crystallographic indices)
        if b[0] * n[0] + b[1] * n[1] + b[2] * n[2] != 0:
            raise ValueError(
                f"Burgers vector b={b} must be perpendicular to slip plane normal n={n} "
                "(integer dot product must be 0)"
            )
        # t parallel to (n × b) — both vectors in integer indices; parallel ⇔ cross == 0
        nxb = (
            n[1] * b[2] - n[2] * b[1],
            n[2] * b[0] - n[0] * b[2],
            n[0] * b[1] - n[1] * b[0],
        )
        # Cross product of t and nxb should be zero if they are parallel.
        cross = (
            t[1] * nxb[2] - t[2] * nxb[1],
            t[2] * nxb[0] - t[0] * nxb[2],
            t[0] * nxb[1] - t[1] * nxb[0],
        )
        if cross != (0, 0, 0):
            raise ValueError(
                f"line direction t={t} must be parallel to (n x b)={nxb} for the "
                "slip system to be self-consistent (cross product must be zero)"
            )


@dataclass
class WallCrystalConfig:
    """Dis-spaced grid of dislocations (sub-project C, mode='wall').

    The current Borgi/Purdue IUCrJ 2024 layout. Preserved unchanged from
    the legacy flat `CrystalConfig`.
    """

    dis: float = 4.0
    ndis: int = 151
    sample_remount: str = "S1"

    def __post_init__(self) -> None:
        if self.sample_remount not in SAMPLE_REMOUNT_OPTIONS:
            valid = ", ".join(SAMPLE_REMOUNT_OPTIONS.keys())
            raise ValueError(
                f"sample_remount must be one of: {valid} (got {self.sample_remount!r})"
            )


@dataclass
class RandomDislocationsConfig:
    """N random dislocations placed by 2D Gaussian (sub-project C).

    `sigma=None` → resolved at draw time from the FOV
    (sigma = FOV_lateral_half / 2).
    `min_distance=None` → no inter-dislocation distance constraint.
    `seed=None` → fresh entropy-pool seed drawn at run time; the realized
    seed value is logged into the sidecar for reproducibility.
    """

    ndis: int
    sigma: float | None = None
    min_distance: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.ndis < 1:
            raise ValueError(f"`ndis` must be >= 1 for random_dislocations; got {self.ndis}")
        if self.sigma is not None and self.sigma <= 0:
            raise ValueError(f"`sigma` must be > 0 when set; got {self.sigma}")
        if self.min_distance is not None and self.min_distance < 0:
            raise ValueError(f"`min_distance` must be >= 0 when set; got {self.min_distance}")
```

Note: the OLD flat `CrystalConfig` (lines 51–62) is **deleted** in Task 5 once the new union supersedes it. For now, leave both in place to keep existing tests compiling.

- [ ] **Step 4: Run tests to verify they pass**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline_crystal_modes.py -v
```

Expected: 11/11 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_pipeline_crystal_modes.py
git commit -m "feat(pipeline): add Centered/Wall/RandomDislocations sub-block configs (C Task 4)"
```

---

## Task 5: `CrystalConfig` discriminated union + `from_dict`

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (replace old flat `CrystalConfig`, update `SimulationConfig.from_toml`)
- Test: `tests/test_pipeline_crystal_modes.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_pipeline_crystal_modes.py`:

```python
from dfxm_geo.pipeline import CrystalConfig


class TestCrystalConfigFromDict:
    def test_centered_mode_parses(self) -> None:
        cfg = CrystalConfig.from_dict(
            {
                "mode": "centered",
                "centered": {"b": [1, -1, 0], "n": [1, 1, 1], "t": [1, 1, -2]},
            }
        )
        assert cfg.mode == "centered"
        assert cfg.centered is not None
        assert cfg.centered.b == (1, -1, 0)
        assert cfg.wall is None
        assert cfg.random_dislocations is None

    def test_wall_mode_parses(self) -> None:
        cfg = CrystalConfig.from_dict(
            {
                "mode": "wall",
                "wall": {"dis": 4.0, "ndis": 151, "sample_remount": "S1"},
            }
        )
        assert cfg.mode == "wall"
        assert cfg.wall is not None
        assert cfg.wall.ndis == 151
        assert cfg.centered is None
        assert cfg.random_dislocations is None

    def test_random_dislocations_mode_parses(self) -> None:
        cfg = CrystalConfig.from_dict(
            {
                "mode": "random_dislocations",
                "random_dislocations": {"ndis": 4, "sigma": 5.0, "seed": 42},
            }
        )
        assert cfg.mode == "random_dislocations"
        assert cfg.random_dislocations is not None
        assert cfg.random_dislocations.ndis == 4
        assert cfg.centered is None
        assert cfg.wall is None

    def test_none_dict_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing \\[crystal\\] block"):
            CrystalConfig.from_dict(None)

    def test_missing_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing `mode` in \\[crystal\\]"):
            CrystalConfig.from_dict({"centered": {"b": [1, -1, 0], "n": [1, 1, 1], "t": [1, 1, -2]}})

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown crystal mode 'bicrystal'"):
            CrystalConfig.from_dict({"mode": "bicrystal"})

    def test_missing_required_subblock_rejected(self) -> None:
        with pytest.raises(ValueError, match="\\[crystal.centered\\] sub-block is required"):
            CrystalConfig.from_dict({"mode": "centered"})

    def test_sibling_subblock_rejected(self) -> None:
        with pytest.raises(ValueError, match="extra sub-block.*\\['wall'\\]"):
            CrystalConfig.from_dict(
                {
                    "mode": "centered",
                    "centered": {"b": [1, -1, 0], "n": [1, 1, 1], "t": [1, 1, -2]},
                    "wall": {"dis": 4.0, "ndis": 151, "sample_remount": "S1"},
                }
            )

    def test_multiple_sibling_subblocks_rejected(self) -> None:
        with pytest.raises(ValueError, match="extra sub-block.*\\['random_dislocations', 'wall'\\]"):
            CrystalConfig.from_dict(
                {
                    "mode": "centered",
                    "centered": {"b": [1, -1, 0], "n": [1, 1, 1], "t": [1, 1, -2]},
                    "wall": {"dis": 4.0, "ndis": 151, "sample_remount": "S1"},
                    "random_dislocations": {"ndis": 4},
                }
            )

    def test_tuple_conversion_for_b_n_t(self) -> None:
        # TOML arrays come in as lists; CrystalConfig.from_dict must convert.
        cfg = CrystalConfig.from_dict(
            {
                "mode": "centered",
                "centered": {"b": [1, -1, 0], "n": [1, 1, 1], "t": [1, 1, -2]},
            }
        )
        assert isinstance(cfg.centered.b, tuple)
        assert isinstance(cfg.centered.n, tuple)
        assert isinstance(cfg.centered.t, tuple)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline_crystal_modes.py::TestCrystalConfigFromDict -v
```

Expected: ImportError — the new `CrystalConfig` shape doesn't exist yet.

- [ ] **Step 3: Replace old `CrystalConfig` with the new discriminated union**

In `src/dfxm_geo/pipeline.py`:

1. **Delete** the old flat `CrystalConfig` (lines 51–62 in current main).
2. Insert the new `CrystalConfig` after `RandomDislocationsConfig`:

```python
_CRYSTAL_MODE_NAMES = ("centered", "wall", "random_dislocations")


@dataclass
class CrystalConfig:
    """Discriminated union over the three crystal-layout modes (sub-project C).

    Exactly one of `centered`/`wall`/`random_dislocations` is non-None and
    matches `mode`. Constructed via `CrystalConfig.from_dict` from a TOML
    `[crystal]` table.
    """

    mode: Literal["centered", "wall", "random_dislocations"]
    centered: CenteredCrystalConfig | None = None
    wall: WallCrystalConfig | None = None
    random_dislocations: RandomDislocationsConfig | None = None

    def __post_init__(self) -> None:
        if self.mode not in _CRYSTAL_MODE_NAMES:
            raise ValueError(
                f"unknown crystal mode {self.mode!r}; expected one of {_CRYSTAL_MODE_NAMES}"
            )
        for mode in _CRYSTAL_MODE_NAMES:
            sub = getattr(self, mode)
            if mode == self.mode and sub is None:
                raise ValueError(
                    f"crystal mode={self.mode!r}: [crystal.{mode}] sub-block is required"
                )
            if mode != self.mode and sub is not None:
                raise ValueError(
                    f"crystal mode={self.mode!r}: extra sub-block {sorted([m for m in _CRYSTAL_MODE_NAMES if m != self.mode and getattr(self, m) is not None])} "
                    f"present; only [crystal.{self.mode}] is valid"
                )

    @classmethod
    def from_dict(cls, data: dict | None) -> CrystalConfig:
        if data is None:
            raise ValueError(
                "missing [crystal] block — forward/identify require explicit "
                "crystal layout; see configs/default.toml."
            )
        if "mode" not in data:
            raise ValueError("missing `mode` in [crystal] — required to pick a layout.")
        mode = data["mode"]
        if mode not in _CRYSTAL_MODE_NAMES:
            raise ValueError(
                f"unknown crystal mode {mode!r}; expected one of {_CRYSTAL_MODE_NAMES}"
            )

        # Check for extras BEFORE building sub-blocks — keeps the error precise.
        siblings = [m for m in _CRYSTAL_MODE_NAMES if m != mode and m in data]
        if siblings:
            raise ValueError(
                f"crystal mode={mode!r}: extra sub-block {siblings} present; "
                f"only [crystal.{mode}] is valid"
            )
        if mode not in data:
            raise ValueError(
                f"crystal mode={mode!r}: [crystal.{mode}] sub-block is required"
            )

        sub_data = data[mode]
        kwargs: dict = {"mode": mode}
        if mode == "centered":
            kwargs["centered"] = CenteredCrystalConfig(
                b=tuple(sub_data["b"]),
                n=tuple(sub_data["n"]),
                t=tuple(sub_data["t"]),
            )
        elif mode == "wall":
            kwargs["wall"] = WallCrystalConfig(**sub_data)
        elif mode == "random_dislocations":
            kwargs["random_dislocations"] = RandomDislocationsConfig(**sub_data)
        return cls(**kwargs)
```

3. Update `SimulationConfig` to use the new shape. Old field `crystal: CrystalConfig = field(default_factory=CrystalConfig)` won't work — the union has no useful default. Change to:

```python
@dataclass
class SimulationConfig:
    crystal: CrystalConfig  # required, no default
    scan: ScanConfig = field(default_factory=ScanConfig)
    io: IOConfig = field(default_factory=IOConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    reciprocal: ReciprocalConfig | None = None
```

4. Update `SimulationConfig.from_toml` to call the new parsers:

```python
@classmethod
def from_toml(cls, path: Path) -> SimulationConfig:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    crystal = CrystalConfig.from_dict(raw.get("crystal"))
    scan = ScanConfig.from_dict(raw.get("scan"))
    io = IOConfig(**raw.get("io", {}))
    postprocess = PostprocessConfig(**raw.get("postprocess", {}))
    reciprocal = ReciprocalConfig.from_dict(raw.get("reciprocal"))
    return cls(
        crystal=crystal, scan=scan, io=io, postprocess=postprocess, reciprocal=reciprocal
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline_crystal_modes.py -v
```

Expected: all 21 tests PASS (10 from TestCrystalConfigFromDict + 11 from earlier classes). Old `tests/test_pipeline.py` will be **broken** at this point because the inline fixtures still use the old flat schema; that's expected and fixed in Task 15.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_pipeline_crystal_modes.py
git commit -m "feat(pipeline): replace flat CrystalConfig with discriminated union (C Task 5)"
```

---

## Task 6: `build_scan_grid` in forward_model

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` (add helper near end of file)
- Test: `tests/test_forward_dispatch.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_forward_dispatch.py`:

```python
"""Tests for the scan-grid + dislocation-population dispatch helpers (sub-projects B+C)."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.direct_space.forward_model import ScanGrid, build_scan_grid
from dfxm_geo.pipeline import ScanConfig


class TestBuildScanGrid:
    def test_single_mode_all_axes_singleton(self) -> None:
        grid = build_scan_grid(ScanConfig())
        assert isinstance(grid, ScanGrid)
        assert grid.axes == ("phi", "chi", "two_dtheta", "z")
        for samples in grid.samples:
            assert samples.shape == (1,)
            assert samples[0] == 0.0

    def test_rocking_mode_phi_has_steps_samples(self) -> None:
        cfg = ScanConfig.from_dict({"phi": {"range": 1e-3, "steps": 21}})
        grid = build_scan_grid(cfg)
        assert grid.samples[0].shape == (21,)
        # First, middle, last are linspace(-range, +range, steps)
        np.testing.assert_allclose(grid.samples[0][0], -1e-3)
        np.testing.assert_allclose(grid.samples[0][-1], +1e-3)
        np.testing.assert_allclose(grid.samples[0][10], 0.0, atol=1e-12)
        # Other axes are singletons
        assert grid.samples[1].shape == (1,)
        assert grid.samples[2].shape == (1,)
        assert grid.samples[3].shape == (1,)

    def test_scan_centered_on_value_offset(self) -> None:
        cfg = ScanConfig.from_dict(
            {"phi": {"value": 1.5e-4, "range": 1e-3, "steps": 11}}
        )
        grid = build_scan_grid(cfg)
        np.testing.assert_allclose(grid.samples[0][0], 1.5e-4 - 1e-3)
        np.testing.assert_allclose(grid.samples[0][-1], 1.5e-4 + 1e-3)
        np.testing.assert_allclose(grid.samples[0][5], 1.5e-4, atol=1e-12)

    def test_fixed_axis_uses_value_singleton(self) -> None:
        cfg = ScanConfig.from_dict({"chi": {"value": 2e-5}})
        grid = build_scan_grid(cfg)
        # chi is index 1 in canonical order
        assert grid.samples[1].shape == (1,)
        np.testing.assert_allclose(grid.samples[1][0], 2e-5)

    def test_mosa_strain_mode(self) -> None:
        cfg = ScanConfig.from_dict(
            {
                "phi": {"range": 6e-4, "steps": 21},
                "chi": {"range": 2e-3, "steps": 21},
                "two_dtheta": {"range": 5e-4, "steps": 11},
            }
        )
        grid = build_scan_grid(cfg)
        assert grid.samples[0].shape == (21,)
        assert grid.samples[1].shape == (21,)
        assert grid.samples[2].shape == (11,)
        assert grid.samples[3].shape == (1,)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_forward_dispatch.py::TestBuildScanGrid -v
```

Expected: ImportError — `ScanGrid` + `build_scan_grid` don't exist.

- [ ] **Step 3: Implement `build_scan_grid`**

Append to `src/dfxm_geo/direct_space/forward_model.py`:

```python
from dataclasses import dataclass as _dataclass


@_dataclass
class ScanGrid:
    """Realized trajectory for a ScanConfig.

    `axes` is the canonical 4-tuple ("phi", "chi", "two_dtheta", "z").
    `samples` is parallel: per-axis 1-D arrays of position values
    (units: radians for angular axes, micrometers for z). Fixed axes
    have shape (1,); scanned axes have shape (steps,).

    The forward kernel iterates the Cartesian product over `samples`.
    """

    axes: tuple[str, str, str, str]
    samples: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def build_scan_grid(scan: "ScanConfig") -> ScanGrid:
    """Build a ScanGrid from a ScanConfig.

    For each canonical axis, returns either:
      - linspace(value-range, value+range, steps) if scanned
      - np.array([value]) if fixed (singleton)
    """
    from dfxm_geo.pipeline import _CANONICAL_AXES  # local import: avoid cycle

    samples = []
    for axis_name in _CANONICAL_AXES:
        axis = getattr(scan, axis_name)
        if axis.is_scanned:
            arr = np.linspace(
                axis.value - axis.range,
                axis.value + axis.range,
                axis.steps,
                dtype=np.float64,
            )
        else:
            arr = np.array([axis.value], dtype=np.float64)
        samples.append(arr)
    return ScanGrid(
        axes=("phi", "chi", "two_dtheta", "z"),
        samples=(samples[0], samples[1], samples[2], samples[3]),
    )
```

Also export `_CANONICAL_AXES` from `pipeline.py` (move from module-private to module-public — drop the underscore? No: keep the leading underscore but make it accessible via from-import. The local import in `build_scan_grid` already works since Python doesn't enforce the convention on imports). Verify by running the test.

- [ ] **Step 4: Run tests to verify they pass**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_forward_dispatch.py::TestBuildScanGrid -v
```

Expected: 5/5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/direct_space/forward_model.py tests/test_forward_dispatch.py
git commit -m "feat(forward): add build_scan_grid for N-D scan trajectories (B Task 6)"
```

---

## Task 7: `build_dislocation_population` for centered + wall modes

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py`
- Test: `tests/test_forward_dispatch.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_forward_dispatch.py`:

```python
from dfxm_geo.direct_space.forward_model import (
    DislocationPopulation,
    build_dislocation_population,
)
from dfxm_geo.pipeline import (
    CenteredCrystalConfig,
    CrystalConfig,
    WallCrystalConfig,
)


class TestBuildDislocationPopulationCentered:
    def test_returns_single_dislocation_at_origin(self) -> None:
        crystal = CrystalConfig(
            mode="centered",
            centered=CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 1, -2)),
        )
        pop = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=None)
        assert isinstance(pop, DislocationPopulation)
        assert pop.positions_um.shape == (1, 3)
        np.testing.assert_allclose(pop.positions_um[0], [0.0, 0.0, 0.0])
        assert pop.Ud.shape == (1, 3, 3)
        assert pop.sidecar is None  # no sidecar for centered mode

    def test_Ud_built_from_b_n_t(self) -> None:
        crystal = CrystalConfig(
            mode="centered",
            centered=CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 1, -2)),
        )
        pop = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=None)
        # Columns of Ud are (normalized) b, n, t.
        b_norm = np.array([1, -1, 0]) / np.linalg.norm([1, -1, 0])
        n_norm = np.array([1, 1, 1]) / np.linalg.norm([1, 1, 1])
        t_norm = np.array([1, 1, -2]) / np.linalg.norm([1, 1, -2])
        np.testing.assert_allclose(pop.Ud[0, :, 0], b_norm, atol=1e-12)
        np.testing.assert_allclose(pop.Ud[0, :, 1], n_norm, atol=1e-12)
        np.testing.assert_allclose(pop.Ud[0, :, 2], t_norm, atol=1e-12)


class TestBuildDislocationPopulationWall:
    def test_returns_ndis_dislocations(self) -> None:
        crystal = CrystalConfig(
            mode="wall",
            wall=WallCrystalConfig(dis=4.0, ndis=11, sample_remount="S1"),
        )
        pop = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=None)
        # Wall mode returns ndis positions along the wall.
        assert pop.positions_um.shape == (11, 3)
        assert pop.Ud.shape == (11, 3, 3)
        assert pop.sidecar is None

    def test_wall_population_is_deterministic(self) -> None:
        crystal = CrystalConfig(
            mode="wall",
            wall=WallCrystalConfig(dis=4.0, ndis=11, sample_remount="S1"),
        )
        pop_a = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=None)
        pop_b = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=None)
        np.testing.assert_array_equal(pop_a.positions_um, pop_b.positions_um)
        np.testing.assert_array_equal(pop_a.Ud, pop_b.Ud)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_forward_dispatch.py::TestBuildDislocationPopulationCentered tests/test_forward_dispatch.py::TestBuildDislocationPopulationWall -v
```

Expected: ImportError — `DislocationPopulation` and `build_dislocation_population` don't exist.

- [ ] **Step 3: Implement centered + wall branches**

Append to `src/dfxm_geo/direct_space/forward_model.py`:

```python
@_dataclass
class DislocationPopulation:
    """A realized set of dislocations.

    positions_um: shape (N, 3) — (x, y, z) sample-frame coordinates.
    Ud: shape (N, 3, 3) — column-stacked (b̂, n̂, t̂) rotation matrices.
    sidecar: dict to be written as JSON, or None if no sidecar needed.
    """

    positions_um: np.ndarray
    Ud: np.ndarray
    sidecar: dict | None


def _ud_matrix_from_bnt(
    b: tuple[int, int, int],
    n: tuple[int, int, int],
    t: tuple[int, int, int],
) -> np.ndarray:
    """Build a 3×3 column-stacked rotation matrix [b̂ | n̂ | t̂].

    Input vectors are crystallographic integer indices; output columns
    are unit-normalized.
    """
    arr = np.asarray([b, n, t], dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / norms).T  # columns = b̂, n̂, t̂


def build_dislocation_population(
    crystal: "CrystalConfig",
    fov_lateral_um: float,
    rng: np.random.Generator | None,
) -> DislocationPopulation:
    """Dispatch on crystal.mode and realize the dislocation population.

    Centered: 1 dislocation at origin with explicit (b, n, t).
    Wall: existing dis-spaced grid via the legacy Find_Hg path.
    Random_dislocations: 2D Gaussian placement with rejection-sampled
    min_distance and uniform draws over the {111} slip system family.
    """
    if crystal.mode == "centered":
        c = crystal.centered
        assert c is not None  # __post_init__ guarantees
        positions = np.zeros((1, 3), dtype=np.float64)
        Ud = _ud_matrix_from_bnt(c.b, c.n, c.t)[np.newaxis, :, :]  # (1, 3, 3)
        return DislocationPopulation(positions_um=positions, Ud=Ud, sidecar=None)

    if crystal.mode == "wall":
        w = crystal.wall
        assert w is not None
        # Preserve the current behavior: positions + Ud come from the legacy
        # Find_Hg-driven path. The legacy code generates these inside the
        # forward kernel itself; here we replicate the placement formula for
        # the dispatch return value but defer kernel-side Ud handling to
        # the existing path (run_simulation in Task 10 still calls Find_Hg
        # when mode='wall'). The positions + Ud arrays from this branch are
        # informational; the live forward path uses the Find_Hg outputs.
        #
        # Positions: ndis dislocations along a wall at z=0, x=0, y evenly
        # spaced by `dis` µm centered at 0.
        ys = (np.arange(w.ndis) - (w.ndis - 1) / 2.0) * w.dis
        positions = np.zeros((w.ndis, 3), dtype=np.float64)
        positions[:, 1] = ys
        # All dislocations in a wall share the same (b, n, t) — the canonical
        # {111}/<-110>/<11-2> slip system for the Borgi/Purdue layout.
        Ud_single = _ud_matrix_from_bnt((1, -1, 0), (1, 1, 1), (1, 1, -2))
        Ud = np.broadcast_to(Ud_single, (w.ndis, 3, 3)).copy()
        return DislocationPopulation(positions_um=positions, Ud=Ud, sidecar=None)

    if crystal.mode == "random_dislocations":
        raise NotImplementedError("random_dislocations branch implemented in Task 8")

    raise AssertionError(f"unreachable crystal.mode={crystal.mode!r}")  # pragma: no cover
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_forward_dispatch.py::TestBuildDislocationPopulationCentered tests/test_forward_dispatch.py::TestBuildDislocationPopulationWall -v
```

Expected: 5/5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/direct_space/forward_model.py tests/test_forward_dispatch.py
git commit -m "feat(forward): dispatch dislocation population for centered + wall modes (C Task 7)"
```

---

## Task 8: `build_dislocation_population` for `random_dislocations`

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py`
- Test: `tests/test_forward_dispatch.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_forward_dispatch.py`:

```python
from dfxm_geo.pipeline import RandomDislocationsConfig


class TestBuildDislocationPopulationRandomDislocations:
    def _config(self, **kwargs) -> CrystalConfig:
        return CrystalConfig(
            mode="random_dislocations",
            random_dislocations=RandomDislocationsConfig(**kwargs),
        )

    def test_seeded_run_is_deterministic(self) -> None:
        crystal = self._config(ndis=4, sigma=5.0, seed=42)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        pop_a = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=rng_a)
        pop_b = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=rng_b)
        np.testing.assert_array_equal(pop_a.positions_um, pop_b.positions_um)
        np.testing.assert_array_equal(pop_a.Ud, pop_b.Ud)

    def test_ndis_dislocations_returned(self) -> None:
        crystal = self._config(ndis=7, sigma=5.0, seed=1)
        pop = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=np.random.default_rng(1))
        assert pop.positions_um.shape == (7, 3)
        assert pop.Ud.shape == (7, 3, 3)

    def test_sigma_default_uses_fov(self) -> None:
        crystal = self._config(ndis=4, sigma=None, seed=1)
        pop = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=np.random.default_rng(1))
        # sidecar records the resolved sigma. FOV_lateral=20.4 → sigma = 20.4/2/2 = 5.1
        assert pop.sidecar is not None
        assert pop.sidecar["sigma_um"] == pytest.approx(5.1)
        assert pop.sidecar["sigma_source"] == "default-fov"

    def test_sigma_override_is_recorded(self) -> None:
        crystal = self._config(ndis=4, sigma=3.0, seed=1)
        pop = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=np.random.default_rng(1))
        assert pop.sidecar["sigma_um"] == 3.0
        assert pop.sidecar["sigma_source"] == "user"

    def test_min_distance_enforced(self) -> None:
        crystal = self._config(ndis=5, sigma=10.0, min_distance=2.0, seed=42)
        pop = build_dislocation_population(crystal, fov_lateral_um=40.0, rng=np.random.default_rng(42))
        # Verify all pairwise (x, y) distances ≥ min_distance.
        xy = pop.positions_um[:, :2]
        for i in range(len(xy)):
            for j in range(i + 1, len(xy)):
                d = float(np.linalg.norm(xy[i] - xy[j]))
                assert d >= 2.0, f"pair ({i},{j}) too close: {d}"

    def test_impossible_min_distance_raises_runtime_error(self) -> None:
        # 100 dislocations in a 0.1 µm radius with min_distance=10 µm — impossible.
        crystal = self._config(ndis=100, sigma=0.05, min_distance=10.0, seed=1)
        with pytest.raises(RuntimeError, match="exceeded retry budget"):
            build_dislocation_population(crystal, fov_lateral_um=20.4, rng=np.random.default_rng(1))

    def test_sidecar_lists_realized_b_n_t_per_dislocation(self) -> None:
        crystal = self._config(ndis=3, sigma=5.0, seed=42)
        pop = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=np.random.default_rng(42))
        assert pop.sidecar is not None
        assert len(pop.sidecar["dislocations"]) == 3
        for entry in pop.sidecar["dislocations"]:
            assert "x_um" in entry and "y_um" in entry and "z_um" in entry
            assert "b" in entry and "n" in entry and "t" in entry
            # b should be an integer triple from the {111} family
            assert len(entry["b"]) == 3

    def test_seed_source_user_when_explicit(self) -> None:
        crystal = self._config(ndis=2, sigma=5.0, seed=42)
        pop = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=np.random.default_rng(42))
        assert pop.sidecar["seed_source"] == "user"
        assert pop.sidecar["seed"] == 42

    def test_seed_source_entropy_when_absent(self) -> None:
        # When seed=None, build_dislocation_population draws a fresh seed and
        # logs it. We pass rng=None to let the function create the rng.
        crystal = self._config(ndis=2, sigma=5.0, seed=None)
        pop = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=None)
        assert pop.sidecar["seed_source"] == "entropy"
        assert isinstance(pop.sidecar["seed"], int)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_forward_dispatch.py::TestBuildDislocationPopulationRandomDislocations -v
```

Expected: 9 failures with `NotImplementedError("random_dislocations branch implemented in Task 8")`.

- [ ] **Step 3: Implement random_dislocations branch**

Add at module level in `src/dfxm_geo/direct_space/forward_model.py`:

```python
_MAX_REJECTION_TRIES = 10_000

# The {111} slip-system family for FCC Al. Each entry is (b, n, t) with
# b·n = 0 and t parallel to n × b. Six variants in {<110>, {111}, <11-2>}.
_SLIP_SYSTEM_111: tuple[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]], ...] = (
    ((1, -1, 0), (1, 1, 1), (1, 1, -2)),
    ((-1, 0, 1), (1, 1, 1), (-1, 2, -1)),
    ((0, 1, -1), (1, 1, 1), (-2, 1, 1)),
    ((1, 1, 0), (1, -1, 1), (1, -1, -2)),
    ((-1, 0, 1), (1, -1, 1), (-1, -2, -1)),
    ((0, -1, -1), (1, -1, 1), (-2, -1, 1)),
)
```

Then extend `build_dislocation_population`. Replace the `if crystal.mode == "random_dislocations": raise NotImplementedError(...)` block with:

```python
    if crystal.mode == "random_dislocations":
        rd = crystal.random_dislocations
        assert rd is not None

        # Resolve sigma: FOV-derived default if user didn't supply.
        if rd.sigma is None:
            sigma_um = (fov_lateral_um / 2.0) / 2.0
            sigma_source = "default-fov"
        else:
            sigma_um = rd.sigma
            sigma_source = "user"

        # Resolve seed + rng. If the caller passed a rng, use it (test
        # injection); otherwise fall back to rd.seed or fresh entropy.
        if rng is None:
            if rd.seed is None:
                # Draw a fresh seed and remember it for the sidecar.
                resolved_seed = int(np.random.SeedSequence().entropy)
                seed_source = "entropy"
            else:
                resolved_seed = rd.seed
                seed_source = "user"
            rng = np.random.default_rng(resolved_seed)
        else:
            # Caller injected an rng; treat seed-source as user-given if rd.seed
            # set, entropy otherwise. Resolved seed isn't recoverable from a
            # caller-supplied rng, so record rd.seed (or None) directly.
            resolved_seed = rd.seed if rd.seed is not None else -1
            seed_source = "user" if rd.seed is not None else "entropy"

        # Draw positions with optional min_distance rejection sampling.
        positions = np.zeros((rd.ndis, 3), dtype=np.float64)
        for i in range(rd.ndis):
            for _ in range(_MAX_REJECTION_TRIES):
                cand = rng.normal(loc=0.0, scale=sigma_um, size=2)  # (x, y)
                if rd.min_distance is None or i == 0:
                    positions[i, 0] = cand[0]
                    positions[i, 1] = cand[1]
                    break
                # Check distance to all already-placed dislocations.
                diffs = positions[:i, :2] - cand
                dists = np.linalg.norm(diffs, axis=1)
                if np.all(dists >= rd.min_distance):
                    positions[i, 0] = cand[0]
                    positions[i, 1] = cand[1]
                    break
            else:
                raise RuntimeError(
                    f"random_dislocations placement exceeded retry budget "
                    f"({_MAX_REJECTION_TRIES}) at dislocation {i}/{rd.ndis}; "
                    f"check min_distance={rd.min_distance}, sigma={sigma_um}, "
                    f"ndis={rd.ndis} — configuration may be impossible."
                )

        # Draw slip-system per dislocation (uniform over the {111} family).
        slip_indices = rng.integers(0, len(_SLIP_SYSTEM_111), size=rd.ndis)
        Ud = np.zeros((rd.ndis, 3, 3), dtype=np.float64)
        sidecar_dislocations: list[dict] = []
        for i in range(rd.ndis):
            b, n, t = _SLIP_SYSTEM_111[slip_indices[i]]
            Ud[i] = _ud_matrix_from_bnt(b, n, t)
            sidecar_dislocations.append(
                {
                    "index": i,
                    "x_um": float(positions[i, 0]),
                    "y_um": float(positions[i, 1]),
                    "z_um": float(positions[i, 2]),
                    "b": list(b),
                    "n": list(n),
                    "t": list(t),
                }
            )

        sidecar = {
            "ndis": rd.ndis,
            "sigma_um": float(sigma_um),
            "sigma_source": sigma_source,
            "min_distance_um": rd.min_distance,
            "seed": resolved_seed,
            "seed_source": seed_source,
            "dislocations": sidecar_dislocations,
        }
        return DislocationPopulation(positions_um=positions, Ud=Ud, sidecar=sidecar)
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_forward_dispatch.py -v
```

Expected: full file PASS (TestBuildScanGrid + TestBuildDislocationPopulationCentered + TestBuildDislocationPopulationWall + TestBuildDislocationPopulationRandomDislocations).

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/direct_space/forward_model.py tests/test_forward_dispatch.py
git commit -m "feat(forward): random_dislocations placement with sidecar metadata (C Task 8)"
```

---

## Task 9: `io/sidecar.py` JSON writer

**Files:**
- Create: `src/dfxm_geo/io/sidecar.py`
- Test: `tests/test_sidecar.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_sidecar.py`:

```python
"""Tests for the random_dislocations sidecar JSON writer (sub-project C)."""

from __future__ import annotations

import json
from pathlib import Path

from dfxm_geo.io.sidecar import write_random_dislocations_sidecar


def test_sidecar_written_next_to_stem(tmp_path: Path) -> None:
    metadata = {
        "ndis": 2,
        "sigma_um": 5.1,
        "sigma_source": "default-fov",
        "min_distance_um": None,
        "seed": 42,
        "seed_source": "user",
        "dislocations": [
            {"index": 0, "x_um": 1.0, "y_um": 2.0, "z_um": 0.0,
             "b": [1, -1, 0], "n": [1, 1, 1], "t": [1, 1, -2]},
            {"index": 1, "x_um": -3.0, "y_um": 4.0, "z_um": 0.0,
             "b": [0, 1, -1], "n": [1, 1, 1], "t": [-2, 1, 1]},
        ],
    }
    stem = tmp_path / "dfxm_geo"
    out_path = write_random_dislocations_sidecar(stem, metadata)
    assert out_path == stem.with_name("dfxm_geo_random_dislocations.json")
    assert out_path.exists()


def test_sidecar_round_trips_via_json(tmp_path: Path) -> None:
    metadata = {
        "ndis": 1,
        "sigma_um": 5.0,
        "sigma_source": "user",
        "min_distance_um": 2.0,
        "seed": 42,
        "seed_source": "user",
        "dislocations": [
            {"index": 0, "x_um": 0.5, "y_um": -0.5, "z_um": 0.0,
             "b": [1, -1, 0], "n": [1, 1, 1], "t": [1, 1, -2]},
        ],
    }
    stem = tmp_path / "run"
    out_path = write_random_dislocations_sidecar(stem, metadata)
    with open(out_path) as fh:
        loaded = json.load(fh)
    assert loaded == metadata


def test_sidecar_pretty_printed(tmp_path: Path) -> None:
    metadata = {"ndis": 1, "dislocations": []}
    stem = tmp_path / "run"
    out_path = write_random_dislocations_sidecar(stem, metadata)
    text = out_path.read_text()
    assert "\n" in text  # not single-line
    assert "  " in text  # indented
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_sidecar.py -v
```

Expected: ImportError — `dfxm_geo.io.sidecar` doesn't exist.

- [ ] **Step 3: Implement the writer**

Create `src/dfxm_geo/io/sidecar.py`:

```python
"""Sidecar JSON writer for random_dislocations realized parameters.

Sub-project C: when a forward run uses `mode="random_dislocations"`,
the realized (positions, Ud) per dislocation are written to a sidecar
JSON file next to the HDF5 output so users (notably ML-training
consumers) can recover the random draw without re-running.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_random_dislocations_sidecar(
    output_stem: Path,
    metadata: dict,
) -> Path:
    """Serialize realized random_dislocations params to JSON.

    Args:
        output_stem: Path stem (no `_random_dislocations.json` suffix).
            E.g. `/runs/2026-05-21/dfxm_geo` → writes
            `/runs/2026-05-21/dfxm_geo_random_dislocations.json`.
        metadata: Dict produced by `build_dislocation_population` for
            `mode="random_dislocations"`. See the spec for schema.

    Returns:
        The path the JSON was written to.
    """
    out_path = output_stem.with_name(output_stem.name + "_random_dislocations.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=False)
    return out_path
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_sidecar.py -v
```

Expected: 3/3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/io/sidecar.py tests/test_sidecar.py
git commit -m "feat(io): add random_dislocations sidecar JSON writer (C Task 9)"
```

---

## Task 10: Rewire `run_simulation` to new dataclasses + dispatch + sidecar

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (`run_simulation` body, ~330–404)
- Test: `tests/test_pipeline_multi_reflection.py` (update existing fixtures + add new mode-coverage tests)

- [ ] **Step 1: Write failing tests**

In `tests/test_pipeline_multi_reflection.py`, add a new class:

```python
class TestRunSimulationCrystalModes:
    """Smoke tests covering all 3 crystal modes via run_simulation."""

    def _base_toml(self, mode_block: str) -> str:
        return (
            '[reciprocal]\n'
            'hkl = [-1, 1, -1]\n'
            'keV = 17.0\n'
            '\n'
            '[scan.phi]\n'
            'range = 6e-4\n'
            'steps = 3\n'        # tiny grid for smoke
            '[scan.chi]\n'
            'range = 2e-3\n'
            'steps = 3\n'
            '\n'
            f'{mode_block}\n'
            '\n'
            '[io]\n'
            'include_perfect_crystal = false\n'
            '\n'
            '[postprocess]\n'
            'enabled = false\n'
        )

    def test_centered_mode_writes_h5(self, tmp_path: Path) -> None:
        toml_text = self._base_toml(
            '[crystal]\n'
            'mode = "centered"\n'
            '[crystal.centered]\n'
            'b = [1, -1, 0]\n'
            'n = [1, 1, 1]\n'
            't = [1, 1, -2]\n'
        )
        cfg_path = tmp_path / "centered.toml"
        cfg_path.write_text(toml_text)
        cfg = SimulationConfig.from_toml(cfg_path)
        out_dir = tmp_path / "out"
        result = run_simulation(cfg, out_dir)
        assert result["h5_path"].exists()
        # No sidecar for centered mode
        assert not (out_dir / "dfxm_geo_random_dislocations.json").exists()

    def test_wall_mode_preserves_legacy_behavior(self, tmp_path: Path) -> None:
        toml_text = self._base_toml(
            '[crystal]\n'
            'mode = "wall"\n'
            '[crystal.wall]\n'
            'dis = 4.0\n'
            'ndis = 151\n'
            'sample_remount = "S1"\n'
        )
        cfg_path = tmp_path / "wall.toml"
        cfg_path.write_text(toml_text)
        cfg = SimulationConfig.from_toml(cfg_path)
        out_dir = tmp_path / "out"
        result = run_simulation(cfg, out_dir)
        assert result["h5_path"].exists()
        assert not (out_dir / "dfxm_geo_random_dislocations.json").exists()

    def test_random_dislocations_mode_writes_sidecar(self, tmp_path: Path) -> None:
        toml_text = self._base_toml(
            '[crystal]\n'
            'mode = "random_dislocations"\n'
            '[crystal.random_dislocations]\n'
            'ndis = 2\n'
            'sigma = 3.0\n'
            'seed = 42\n'
        )
        cfg_path = tmp_path / "rd.toml"
        cfg_path.write_text(toml_text)
        cfg = SimulationConfig.from_toml(cfg_path)
        out_dir = tmp_path / "out"
        result = run_simulation(cfg, out_dir)
        sidecar = out_dir / "dfxm_geo_random_dislocations.json"
        assert sidecar.exists()
        import json
        sidecar_data = json.loads(sidecar.read_text())
        assert sidecar_data["ndis"] == 2
        assert sidecar_data["seed"] == 42
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline_multi_reflection.py::TestRunSimulationCrystalModes -v
```

Expected: failures around `config.scan.phi_range` / `config.crystal.dis` being missing (the old run_simulation still reads them).

- [ ] **Step 3: Rewire `run_simulation`**

In `src/dfxm_geo/pipeline.py`, replace the body of `run_simulation` (current lines 330–404):

```python
def run_simulation(config: SimulationConfig, output_dir: Path) -> dict[str, Any]:
    """Execute a DFXM forward-simulation run from a config object.

    Writes one `<output_dir>/dfxm_geo.h5` containing BLISS scan `/1.1`
    (dislocations) and, if `io.include_perfect_crystal=True`, `/2.1`
    (Hg=0 reference). For `crystal.mode="random_dislocations"`, also
    writes a `<output_dir>/dfxm_geo_random_dislocations.json` sidecar.
    """
    if config.reciprocal is None:
        raise ValueError(
            "SimulationConfig.reciprocal is None — must specify [reciprocal] "
            "block in TOML or set it programmatically before calling run_simulation."
        )
    _lookup_and_load_kernel(config.reciprocal.hkl, config.reciprocal.keV)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dislocation population (dispatches on crystal.mode).
    fov_lateral_um = fm.Npixels * fm.psize * 1e6  # m → µm
    population = fm.build_dislocation_population(
        config.crystal, fov_lateral_um=fov_lateral_um, rng=None
    )

    # Write sidecar BEFORE forward kernel so a forward crash still
    # leaves the realized draw recoverable.
    if population.sidecar is not None:
        from dfxm_geo.io.sidecar import write_random_dislocations_sidecar
        sidecar_path = write_random_dislocations_sidecar(
            output_dir / "dfxm_geo", population.sidecar
        )
        print(f"[dfxm-forward] sidecar: {sidecar_path}", flush=True)

    # Effective-config print. Wall mode still uses Find_Hg internally;
    # centered + random_dislocations use the population directly.
    print(
        f"[dfxm-forward] effective config:\n"
        f"  Nsub={fm.Nsub}  Npixels={fm.Npixels}  NN1={fm.NN1}  NN2={fm.NN2}\n"
        f"  kernel={fm._loaded_kernel_path}\n"
        f"  crystal.mode={config.crystal.mode}  ndis={len(population.positions_um)}\n"
        f"  scan.mode={config.scan.derived_mode_name()}  axes_scanned={config.scan.scanned_axes()}",
        flush=True,
    )

    # Wall mode preserves legacy Find_Hg path.
    if config.crystal.mode == "wall":
        w = config.crystal.wall
        assert w is not None
        S = SAMPLE_REMOUNT_OPTIONS[w.sample_remount]
        Hg, q_hkl = fm.Find_Hg(
            w.dis, w.ndis, fm.psize, fm.zl_rms, S=S, remount_name=w.sample_remount
        )
        fm.Hg = Hg
        fm.q_hkl = q_hkl
        sample_dis = w.dis
        sample_ndis = w.ndis
        sample_remount = w.sample_remount
    else:
        # Centered / random_dislocations: Hg + q_hkl come from the population.
        # For now, the forward kernel still needs Hg-shaped arrays; build them
        # from the population's positions + Ud.
        # NOTE: this path uses Find_Hg_from_population — added in the same task.
        Hg, q_hkl = fm.Find_Hg_from_population(population, fm.psize, fm.zl_rms)
        fm.Hg = Hg
        fm.q_hkl = q_hkl
        sample_dis = -1.0  # sentinel: not applicable
        sample_ndis = len(population.positions_um)
        sample_remount = "N/A"

    config_toml = _dataclass_to_toml_str(config)

    h5_path = output_dir / "dfxm_geo.h5"
    write_simulation_h5(
        h5_path,
        Hg=Hg,
        q_hkl=q_hkl,
        scan=config.scan,                          # NEW: pass full ScanConfig
        crystal_mode=config.crystal.mode,          # NEW
        scan_mode=config.scan.derived_mode_name(), # NEW
        scanned_axes=list(config.scan.scanned_axes()),  # NEW
        include_perfect_crystal=config.io.include_perfect_crystal,
        sample_dis=sample_dis,
        sample_ndis=sample_ndis,
        sample_remount=sample_remount,
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
```

Add to `direct_space/forward_model.py`:

```python
def Find_Hg_from_population(
    population: DislocationPopulation,
    psize: float,
    zl_rms: float,
    h: int = -1,
    k: int = 1,
    l: int = -1,
    *,
    S: np.ndarray = _S_IDENTITY,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Hg + q_hkl from an arbitrary DislocationPopulation.

    For mode='centered' (1 dislocation) and 'random_dislocations' (N drawn
    dislocations). Mirrors `Find_Hg` but takes explicit positions + Ud
    matrices instead of the wall-layout (dis, ndis) parameters.
    """
    Q_norm = np.sqrt(h * h + k * k + l * l)
    q_hkl = np.asarray([h, k, l]) / Q_norm

    # Build MixedDislocSpec per dislocation. The existing Fd_find_multi_dislocs_mixed
    # already supports per-crystal Ud + lab-frame offset.
    crystals = [
        MixedDislocSpec(
            Ud_mix=population.Ud[i],
            rotation_deg=0.0,  # rotation_deg=0 = pure-edge; mixed character
                               # comes from the Ud matrix itself for these modes.
                               # For random_dislocations with explicit (b, n, t)
                               # already encoded in Ud, rotation_deg=0 is correct.
            position_lab_um=tuple(float(x) for x in population.positions_um[i]),
        )
        for i in range(len(population.positions_um))
    ]

    # Fg has shape (X, 3, 3) — identity already added once at the end.
    # Hg is the same Fg minus identity (the kernel expects deformation, not deformation gradient).
    # Confirm Hg vs Fg convention by reading load_or_generate_Hg's docstring.
    Fg = Fd_find_multi_dislocs_mixed(rl, Us, crystals, Theta, S=S)
    Hg = Fg - np.identity(3)
    return Hg, q_hkl
```

**Complexity note for the implementer:** This function is the bridge between the new dispatcher (centered + random_dislocations) and the existing forward kernel. Several things to verify before finalizing:

1. **Hg vs Fg convention.** `Find_Hg` returns `Hg` from `load_or_generate_Hg`, which internally calls `Fd_find_mixed`-family functions and returns *some* shape (X, 3, 3) array. Confirm whether `Hg` is `Fg - I` or `Fg` itself by reading `io/strain_cache.py:load_or_generate_Hg`. If `Hg == Fg`, drop the `- I` subtraction above.
2. **Module-level `rl`, `Us`, `Theta`.** These are built at module-import time in `forward_model.py` (lines 50–110 in current main). Reusing them for centered/random modes is fine since they describe the detector ray grid (independent of dislocation layout).
3. **No disk cache.** Unlike `Find_Hg` (which caches `Fg` by `(dis, ndis, psize, zl_rms, Npixels, Nsub, remount_name)`), `Find_Hg_from_population` deliberately does NOT cache: every random draw is unique, and a centered single-dislocation run is cheap enough that caching adds complexity for no benefit. If profiling later shows the centered mode is slow enough to warrant a cache, hash on `(positions, Ud)` content.
4. **`MixedDislocSpec.rotation_deg=0` correctness.** The existing wall path uses `rotation_deg=0` for pure-edge dislocations; centered + random_dislocations encode their character (mixed/edge/screw) via the Ud matrix columns. If the implementer finds the resulting Hg doesn't reproduce a known reference (e.g. running centered with the wall's standard (b, n, t) gives a different field than the wall's first dislocation), the rotation_deg interpretation needs revisiting — refer to `Fd_find_mixed` lines 224–290.
5. **Smoke test sanity check.** After this task, run the centered mode end-to-end via the Task 10 test `test_centered_mode_writes_h5` and visually inspect one image from the HDF5 output (`silx view <h5_path>`). The image should show a recognizable dislocation strain field, not noise.

- [ ] **Step 4: Run tests to verify they pass**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline_multi_reflection.py::TestRunSimulationCrystalModes -v
```

Expected: 3/3 PASS. **Other tests in `test_pipeline.py` and `test_pipeline_multi_reflection.py` will still be broken** because their fixtures use the old flat schema; those get fixed in Task 15.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/pipeline.py src/dfxm_geo/direct_space/forward_model.py tests/test_pipeline_multi_reflection.py
git commit -m "feat(pipeline): dispatch run_simulation on crystal.mode + write sidecar (B+C Task 10)"
```

---

## Task 11: Update `_dataclass_to_toml_str` for new nested types

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (function at line 407+)
- Test: `tests/test_pipeline.py` (add round-trip test)

- [ ] **Step 1: Write failing test**

Add to `tests/test_pipeline.py` (or a new dedicated file `tests/test_dataclass_to_toml.py`):

```python
class TestDataclassToTomlRoundTrip:
    """Round-trip: SimulationConfig → TOML string → SimulationConfig (Task 11)."""

    def test_centered_mode_round_trip(self, tmp_path: Path) -> None:
        toml_text = (
            '[reciprocal]\n'
            'hkl = [-1, 1, -1]\n'
            'keV = 17.0\n'
            '\n'
            '[scan.phi]\n'
            'value = 0.0\n'
            'range = 6e-4\n'
            'steps = 21\n'
            '\n'
            '[crystal]\n'
            'mode = "centered"\n'
            '[crystal.centered]\n'
            'b = [1, -1, 0]\n'
            'n = [1, 1, 1]\n'
            't = [1, 1, -2]\n'
        )
        cfg_path = tmp_path / "cfg.toml"
        cfg_path.write_text(toml_text)
        cfg = SimulationConfig.from_toml(cfg_path)

        round_tripped_toml = _dataclass_to_toml_str(cfg)
        cfg_path_2 = tmp_path / "cfg2.toml"
        cfg_path_2.write_text(round_tripped_toml)
        cfg_2 = SimulationConfig.from_toml(cfg_path_2)
        assert cfg_2 == cfg

    def test_wall_mode_round_trip(self, tmp_path: Path) -> None:
        toml_text = (
            '[reciprocal]\n'
            'hkl = [-1, 1, -1]\n'
            'keV = 17.0\n'
            '\n'
            '[scan.phi]\n'
            'range = 6e-4\n'
            'steps = 21\n'
            '[scan.chi]\n'
            'range = 2e-3\n'
            'steps = 21\n'
            '\n'
            '[crystal]\n'
            'mode = "wall"\n'
            '[crystal.wall]\n'
            'dis = 4.0\n'
            'ndis = 151\n'
            'sample_remount = "S1"\n'
        )
        cfg_path = tmp_path / "cfg.toml"
        cfg_path.write_text(toml_text)
        cfg = SimulationConfig.from_toml(cfg_path)
        round_tripped = _dataclass_to_toml_str(cfg)
        cfg_2 = SimulationConfig.from_toml(_write_temp(tmp_path, round_tripped))
        assert cfg_2 == cfg

    def test_random_dislocations_mode_round_trip(self, tmp_path: Path) -> None:
        toml_text = (
            '[reciprocal]\n'
            'hkl = [-1, 1, -1]\n'
            'keV = 17.0\n'
            '\n'
            '[scan.phi]\n'
            'range = 6e-4\n'
            'steps = 21\n'
            '\n'
            '[crystal]\n'
            'mode = "random_dislocations"\n'
            '[crystal.random_dislocations]\n'
            'ndis = 4\n'
            'sigma = 5.0\n'
            'min_distance = 2.0\n'
            'seed = 42\n'
        )
        cfg_path = tmp_path / "cfg.toml"
        cfg_path.write_text(toml_text)
        cfg = SimulationConfig.from_toml(cfg_path)
        round_tripped = _dataclass_to_toml_str(cfg)
        cfg_2 = SimulationConfig.from_toml(_write_temp(tmp_path, round_tripped))
        assert cfg_2 == cfg


def _write_temp(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "rt.toml"
    p.write_text(text)
    return p
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline.py::TestDataclassToTomlRoundTrip -v
```

Expected: failures because `_dataclass_to_toml_str` still uses the old `asdict`-then-flat approach which doesn't render the new nested per-axis or per-mode structure correctly.

- [ ] **Step 3: Rewrite `_dataclass_to_toml_str`**

In `src/dfxm_geo/pipeline.py`, replace the function body (line 407+):

```python
def _dataclass_to_toml_str(config: SimulationConfig) -> str:
    """Serialize a SimulationConfig back to TOML-formatted text.

    Renders:
      [reciprocal] (hkl, keV)
      [scan.<axis>] for each axis whose value/range/steps are non-default
      [crystal] mode + matching [crystal.<mode>] sub-block
      [io], [postprocess]
    """
    lines: list[str] = []

    # [reciprocal]
    if config.reciprocal is not None:
        lines.append("[reciprocal]")
        h, k, l = config.reciprocal.hkl
        lines.append(f"hkl = [{h}, {k}, {l}]")
        lines.append(f"keV = {config.reciprocal.keV}")
        lines.append("")

    # [scan.<axis>] — only render axes that differ from default
    for axis_name in _CANONICAL_AXES:
        axis = getattr(config.scan, axis_name)
        if axis.value == 0.0 and not axis.is_scanned:
            continue  # default — skip
        lines.append(f"[scan.{axis_name}]")
        if axis.value != 0.0:
            lines.append(f"value = {axis.value}")
        if axis.is_scanned:
            lines.append(f"range = {axis.range}")
            lines.append(f"steps = {axis.steps}")
        lines.append("")

    # [crystal] + matching sub-block
    lines.append("[crystal]")
    lines.append(f'mode = "{config.crystal.mode}"')
    lines.append(f"[crystal.{config.crystal.mode}]")
    if config.crystal.mode == "centered":
        c = config.crystal.centered
        assert c is not None
        lines.append(f"b = [{c.b[0]}, {c.b[1]}, {c.b[2]}]")
        lines.append(f"n = [{c.n[0]}, {c.n[1]}, {c.n[2]}]")
        lines.append(f"t = [{c.t[0]}, {c.t[1]}, {c.t[2]}]")
    elif config.crystal.mode == "wall":
        w = config.crystal.wall
        assert w is not None
        lines.append(f"dis = {w.dis}")
        lines.append(f"ndis = {w.ndis}")
        lines.append(f'sample_remount = "{w.sample_remount}"')
    elif config.crystal.mode == "random_dislocations":
        rd = config.crystal.random_dislocations
        assert rd is not None
        lines.append(f"ndis = {rd.ndis}")
        if rd.sigma is not None:
            lines.append(f"sigma = {rd.sigma}")
        if rd.min_distance is not None:
            lines.append(f"min_distance = {rd.min_distance}")
        if rd.seed is not None:
            lines.append(f"seed = {rd.seed}")
    lines.append("")

    # [io]
    from dataclasses import asdict as _asdict
    io_dict = _asdict(config.io)
    if any(v != _IO_DEFAULTS.get(k) for k, v in io_dict.items()):
        lines.append("[io]")
        for k, v in io_dict.items():
            if v is None:
                continue
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, bool):
                lines.append(f"{k} = {str(v).lower()}")
            else:
                lines.append(f"{k} = {v}")
        lines.append("")

    # [postprocess]
    pp_dict = _asdict(config.postprocess)
    lines.append("[postprocess]")
    for k, v in pp_dict.items():
        if isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        elif isinstance(v, bool):
            lines.append(f"{k} = {str(v).lower()}")
        else:
            lines.append(f"{k} = {v}")

    return "\n".join(lines) + "\n"


# Defaults for terse [io] rendering — skip writing defaults to keep TOML tidy.
_IO_DEFAULTS = {
    "fn_prefix": "/mosa_test_0000_",
    "ftype": ".npy",
    "dislocs_dirname": "images10",
    "perfect_dirname": "images10_perf_crystal",
    "include_perfect_crystal": True,
    "max_workers": None,
}
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline.py::TestDataclassToTomlRoundTrip -v
```

Expected: 3/3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): _dataclass_to_toml_str renders new nested shapes (B+C Task 11)"
```

---

## Task 12: Rewire identification — shared ScanConfig + IdentificationNoiseConfig

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (`IdentificationScanConfig` deletion, `IdentificationConfig.scan` field, `load_identification_config`)
- Test: `tests/test_pipeline_identification.py` (update existing tests + add new IdentificationNoiseConfig tests)

- [ ] **Step 1: Write failing tests**

In `tests/test_pipeline_identification.py`, add:

```python
class TestIdentificationConfigScanReusesSharedShape:
    def test_phi_value_from_scan_phi_value(self, tmp_path: Path) -> None:
        toml_text = (
            'mode = "single"\n'
            '\n'
            '[reciprocal]\n'
            'hkl = [-1, 1, -1]\n'
            'keV = 17.0\n'
            '\n'
            '[crystal]\n'
            'slip_plane_normal = [1, 1, 1]\n'
            'sweep_all_slip_planes = true\n'
            'exclude_invisibility = true\n'
            '\n'
            '[scan.phi]\n'
            'value = 0.00015\n'
            '\n'
            '[noise]\n'
            'poisson_noise = true\n'
            'rng_seed = 0\n'
            'intensity_scale = 7.0\n'
            '\n'
            '[io]\n'
            'fn_prefix = "/mosa_test_0000_"\n'
            'ftype = ".npy"\n'
            'dislocs_dirname = "identify"\n'
            'perfect_dirname = "ignored"\n'
            'include_perfect_crystal = false\n'
        )
        cfg_path = tmp_path / "id.toml"
        cfg_path.write_text(toml_text)
        cfg = load_identification_config(cfg_path)
        # New shape: scan is the shared ScanConfig
        assert isinstance(cfg.scan, ScanConfig)
        assert cfg.scan.phi.value == 1.5e-4
        # Noise lives in its own block
        assert cfg.noise.poisson_noise is True
        assert cfg.noise.intensity_scale == 7.0


class TestIdentificationConfigZScanForbidsScanZ:
    def test_zscan_mode_with_scan_z_rejected(self, tmp_path: Path) -> None:
        # When mode='z-scan', the [zscan].z_offsets_um drives z; [scan.z] is forbidden.
        toml_text = (
            'mode = "z-scan"\n'
            '\n'
            '[reciprocal]\n'
            'hkl = [-1, 1, -1]\n'
            'keV = 17.0\n'
            '\n'
            '[crystal]\n'
            'slip_plane_normal = [1, 1, 1]\n'
            '\n'
            '[scan.phi]\n'
            'value = 0.00015\n'
            '[scan.z]\n'                    # forbidden in z-scan mode
            'range = 1e-6\n'
            'steps = 3\n'
            '\n'
            '[zscan]\n'
            'z_offsets_um = [-1.0, 0.0, 1.0]\n'
            'phi_range_deg = 0.03\n'
            'phi_steps = 5\n'
            'chi_range_deg = 0.1\n'
            'chi_steps = 5\n'
            '\n'
            '[io]\n'
            'include_perfect_crystal = false\n'
        )
        cfg_path = tmp_path / "id.toml"
        cfg_path.write_text(toml_text)
        with pytest.raises(ValueError, match="mode='z-scan' .* \\[scan.z\\] is forbidden"):
            load_identification_config(cfg_path)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline_identification.py::TestIdentificationConfigScanReusesSharedShape tests/test_pipeline_identification.py::TestIdentificationConfigZScanForbidsScanZ -v
```

Expected: failures — `IdentificationConfig.scan` is still `IdentificationScanConfig`; `cfg.noise` doesn't exist.

- [ ] **Step 3: Replace `IdentificationScanConfig` with new shape + add `IdentificationNoiseConfig`**

In `src/dfxm_geo/pipeline.py`:

1. **Delete** `IdentificationScanConfig` (current lines 182–189).
2. Insert `IdentificationNoiseConfig`:

```python
@dataclass(frozen=True, kw_only=True)
class IdentificationNoiseConfig:
    """Noise + intensity parameters for dfxm-identify forward calls.

    Sub-project B carry-out: these moved out of the old
    IdentificationScanConfig (now deleted) into their own block since
    they describe noise/detector, not the scan trajectory.
    """

    poisson_noise: bool = True
    rng_seed: int = 0
    intensity_scale: float = 7.0
```

3. Update `IdentificationConfig`:

```python
@dataclass(frozen=True, kw_only=True)
class IdentificationConfig:
    mode: Literal["single", "multi", "z-scan"]
    crystal: IdentificationCrystalConfig
    scan: ScanConfig                              # NEW: shared with forward
    noise: IdentificationNoiseConfig              # NEW
    io: IOConfig
    multi: IdentificationMonteCarloConfig | None = None
    zscan: IdentificationZScanConfig | None = None
    reciprocal: ReciprocalConfig | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("single", "multi", "z-scan"):
            raise ValueError(f"mode must be 'single', 'multi', or 'z-scan', got {self.mode!r}")
        if self.mode == "multi" and self.multi is None:
            raise ValueError("mode='multi' requires a `multi` config block")
        if self.mode == "z-scan" and self.zscan is None:
            raise ValueError("mode='z-scan' requires a `zscan` config block")
        if self.mode in ("single", "multi") and self.zscan is not None:
            raise ValueError(
                f"mode={self.mode!r}: zscan config block is only valid in mode='z-scan'"
            )
        # z-scan mode owns the z dimension via z_offsets_um; [scan.z] would conflict.
        if self.mode == "z-scan" and self.scan.is_scanned("z"):
            raise ValueError(
                f"mode='z-scan' uses [zscan].z_offsets_um for the z dimension; [scan.z] is forbidden"
            )
        try:
            _burgers_vectors(self.crystal.slip_plane_normal)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
```

4. Update `load_identification_config`:

```python
def load_identification_config(path: Path) -> IdentificationConfig:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    if "mode" not in data:
        raise ValueError(f"{path}: missing top-level 'mode' field")

    crystal_data = data.get("crystal", {})
    if "slip_plane_normal" in crystal_data:
        crystal_data = {
            **crystal_data,
            "slip_plane_normal": tuple(crystal_data["slip_plane_normal"]),
        }
    crystal = IdentificationCrystalConfig(**crystal_data)
    scan = ScanConfig.from_dict(data.get("scan"))                       # NEW
    noise = IdentificationNoiseConfig(**data.get("noise", {}))          # NEW
    io = IOConfig(**data.get("io", {}))
    multi = (
        IdentificationMonteCarloConfig(**data["multi"]) if data.get("multi") is not None else None
    )
    zscan = IdentificationZScanConfig(**data["zscan"]) if data.get("zscan") is not None else None
    reciprocal = ReciprocalConfig.from_dict(data.get("reciprocal"))

    return IdentificationConfig(
        mode=data["mode"],
        crystal=crystal,
        scan=scan,
        noise=noise,
        io=io,
        multi=multi,
        zscan=zscan,
        reciprocal=reciprocal,
    )
```

5. Update `run_identification` (existing body, similar to the forward rewire in Task 10) to read `config.scan.phi.value` instead of `config.scan.phi_rad`, and `config.noise.poisson_noise` / `.rng_seed` / `.intensity_scale` instead of `config.scan.poisson_noise` / etc. Also: identification's inner forward calls should construct a `SimulationConfig` with `crystal.mode="centered"` for each hypothesis being tested.

- [ ] **Step 4: Run tests to verify they pass**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline_identification.py::TestIdentificationConfigScanReusesSharedShape tests/test_pipeline_identification.py::TestIdentificationConfigZScanForbidsScanZ -v
```

Expected: 2/2 PASS. Other tests in `test_pipeline_identification.py` will still be broken; fixed in Task 15.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_pipeline_identification.py
git commit -m "feat(pipeline): identification reuses shared ScanConfig + IdentificationNoiseConfig (B Task 12)"
```

---

## Task 13: HDF5 attrs — `scan_mode`, `scanned_axes`, `crystal_mode`

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py` (`write_simulation_h5` signature + attr writes)
- Test: `tests/test_hdf5_provenance.py` (add attr assertions)

- [ ] **Step 1: Inspect current signature**

Read `src/dfxm_geo/io/hdf5.py` around the `write_simulation_h5` function. Note the current parameter list (phi_range, phi_steps, chi_range, chi_steps, sample_dis, sample_ndis, sample_remount, config_toml, cli, max_workers). Plan: replace the four scan-related scalars with a single `scan: ScanConfig` parameter; add `scan_mode`, `scanned_axes`, `crystal_mode`.

- [ ] **Step 2: Write failing test**

In `tests/test_hdf5_provenance.py`, add:

```python
class TestHdf5NewAttrs:
    def test_scan_mode_attr_written(self, tmp_path: Path) -> None:
        # Construct a SimulationConfig in mosa mode, run pipeline, open h5, check attr.
        cfg = SimulationConfig(
            crystal=CrystalConfig(
                mode="centered",
                centered=CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 1, -2)),
            ),
            scan=ScanConfig.from_dict(
                {"phi": {"range": 6e-4, "steps": 3}, "chi": {"range": 2e-3, "steps": 3}}
            ),
            io=IOConfig(include_perfect_crystal=False),
            postprocess=PostprocessConfig(enabled=False),
            reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
        )
        out_dir = tmp_path / "out"
        run_simulation(cfg, out_dir)
        h5_path = out_dir / "dfxm_geo.h5"
        with h5py.File(h5_path, "r") as f:
            assert f["/1.1"].attrs["scan_mode"] == "mosa"
            assert list(f["/1.1"].attrs["scanned_axes"]) == [b"phi", b"chi"]
            assert f["/1.1"].attrs["crystal_mode"] == "centered"
```

- [ ] **Step 3: Run test to verify it fails**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_provenance.py::TestHdf5NewAttrs -v
```

Expected: KeyError on `f["/1.1"].attrs["scan_mode"]` — attrs not written yet.

- [ ] **Step 4: Implement**

In `src/dfxm_geo/io/hdf5.py`:

1. Update the signature of `write_simulation_h5` per the Task 10 call site (it now passes `scan=config.scan`, `scan_mode=...`, `scanned_axes=...`, `crystal_mode=...` instead of phi_range/phi_steps/chi_range/chi_steps).
2. Use `scan` internally for the per-axis sample arrays. Replace any `phi_range_deg = phi_range` style reads with `scan.phi.range`, etc. **Units**: confirm whether the existing code expects degrees or radians. The new schema is radians; if `write_simulation_h5` internally converts, leave the conversion at the call boundary — the function should receive radians.
3. Where attrs are written to the `/1.1` group (look for existing `attrs["..."] = ...` lines, around the dataset creation), add:

```python
h5_group = f.require_group("/1.1")
h5_group.attrs["scan_mode"] = scan_mode
h5_group.attrs["scanned_axes"] = [a.encode("utf-8") for a in scanned_axes]
h5_group.attrs["crystal_mode"] = crystal_mode
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_provenance.py::TestHdf5NewAttrs -v
```

Expected: 1/1 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/io/hdf5.py tests/test_hdf5_provenance.py
git commit -m "feat(hdf5): embed scan_mode + scanned_axes + crystal_mode on /N.1 (B+C Task 13)"
```

---

## Task 14: Migrate 9 TOML configs

**Files:**
- Modify: `configs/default.toml`, `configs/identification_single.toml`, `configs/identification_multi.toml`, `configs/identification_zscan.toml`, `configs/variants/dis_0p25.toml`, `configs/variants/dis_0p5.toml`, `configs/variants/dis_1.toml`, `configs/variants/dis_2.toml`, `configs/variants/sample_remount_S2.toml`
- Test: `tests/test_configs_load_under_new_schema.py` (create)

- [ ] **Step 1: Write failing smoke loader test**

Create `tests/test_configs_load_under_new_schema.py`:

```python
"""All shipped TOML configs load under the new B+C schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from dfxm_geo.pipeline import SimulationConfig, load_identification_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


@pytest.mark.parametrize(
    "config_name",
    [
        "default.toml",
        "variants/dis_0p25.toml",
        "variants/dis_0p5.toml",
        "variants/dis_1.toml",
        "variants/dis_2.toml",
        "variants/sample_remount_S2.toml",
    ],
)
def test_forward_config_loads(config_name: str) -> None:
    path = CONFIGS_DIR / config_name
    cfg = SimulationConfig.from_toml(path)
    assert cfg.crystal.mode in ("centered", "wall", "random_dislocations")


@pytest.mark.parametrize(
    "config_name",
    [
        "identification_single.toml",
        "identification_multi.toml",
        "identification_zscan.toml",
    ],
)
def test_identification_config_loads(config_name: str) -> None:
    path = CONFIGS_DIR / config_name
    cfg = load_identification_config(path)
    assert cfg.mode in ("single", "multi", "z-scan")
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_configs_load_under_new_schema.py -v
```

Expected: 9/9 FAIL with "missing `mode` in [crystal]" or similar — the existing TOML files still use the old flat schema.

- [ ] **Step 3: Rewrite `configs/default.toml`**

```toml
# Default DFXM forward-simulation config (post-B+C).
# Active mode is the simple-case `centered` single dislocation. Switch to
# `wall` or `random_dislocations` by uncommenting the matching block below
# and changing `mode` accordingly.

[reciprocal]
hkl        = [-1, 1, -1]   # Al 111 reflection (default)
keV        = 17.0          # beam energy in keV
Nrays      = 100_000_000
npoints1   = 400
npoints2   = 200
npoints3   = 200
qi1_range  = 5e-4
qi2_range  = 7.5e-3
qi3_range  = 7.5e-3
zeta_v_fwhm = 5.3e-4
zeta_h_fwhm = 0.0
NA_rms      = 3.1106382978723403e-4
eps_rms     = 6.0e-5
D  = 0.000565685424949238
d1 = 0.274
beamstop   = true
bs_height  = 25e-3
aperture   = true
knife_edge = false
dphi_range = 0.0

# ─── Scan ────────────────────────────────────────────────────────
# Per-axis primitives. Each axis is independently fixed (value, default 0)
# or scanned (value + range + steps). Mode name is derived.

[scan.phi]
range = 6e-4               # rad; half-range (= 0.034° in old degrees)
steps = 61

[scan.chi]
range = 2e-3               # rad; half-range (= 0.115° in old degrees)
steps = 61

# [scan.two_dtheta]        # omitted → fixed at 0 (no strain scan)
# [scan.z]                 # omitted → fixed at 0 (no z translation)
# Derived scan-mode label: "mosa" (phi + chi).

# ─── Crystal ─────────────────────────────────────────────────────
# Active mode = centered. Commented-out blocks show the other two modes.

[crystal]
mode = "centered"

[crystal.centered]
b = [1, -1, 0]   # Burgers vector indices
n = [1, 1, 1]    # slip plane normal indices
t = [1, 1, -2]   # dislocation line direction

# [crystal.wall]
# dis = 4.0              # inter-dislocation distance (µm)
# ndis = 151             # number of dislocations
# sample_remount = "S1"  # one of S1/S2/S3/S4 — Purdue 2024 paper

# [crystal.random_dislocations]
# ndis = 4               # >= 1
# sigma = 5.0            # µm; omit → default = FOV_lateral_half / 2 ≈ 5.1 µm
# min_distance = 4.0     # µm, optional
# seed = 42              # optional; sidecar JSON written regardless

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "images10"
perfect_dirname = "images10_perf_crystal"
include_perfect_crystal = true

[postprocess]
enabled = true
chi_oversample = 20
phi_oversample = 20
chi_oversample_for_shift = 100
figures_dirname = "figures"
data_dirname = "analysis"
```

- [ ] **Step 4: Rewrite 4 variant `dis_*.toml` configs**

Each follows the same pattern. Example `configs/variants/dis_2.toml`:

```toml
# Variant: dis = 2 µm (half the default inter-dislocation distance).
# Used for sweeps over the dis parameter; otherwise identical to default.

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0

[scan.phi]
range = 6e-4
steps = 61
[scan.chi]
range = 2e-3
steps = 61

[crystal]
mode = "wall"
[crystal.wall]
dis = 2.0
ndis = 151
sample_remount = "S1"

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "images10"
perfect_dirname = "images10_perf_crystal"
include_perfect_crystal = true

[postprocess]
enabled = true
chi_oversample = 20
phi_oversample = 20
chi_oversample_for_shift = 100
figures_dirname = "figures"
data_dirname = "analysis"
```

Repeat the pattern for `dis_0p25.toml` (dis=0.25), `dis_0p5.toml` (dis=0.5), `dis_1.toml` (dis=1.0). Each `[crystal.wall].dis` matches the filename.

- [ ] **Step 5: Rewrite `configs/variants/sample_remount_S2.toml`**

```toml
# Variant: sample remounted in orientation S2 (Purdue 2024 paper).
# Equivalent to default.toml except for [crystal.wall].sample_remount.

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0

[scan.phi]
range = 6e-4
steps = 61
[scan.chi]
range = 2e-3
steps = 61

[crystal]
mode = "wall"
[crystal.wall]
dis = 4.0
ndis = 151
sample_remount = "S2"

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "images10"
perfect_dirname = "images10_perf_crystal"
include_perfect_crystal = true

[postprocess]
enabled = true
chi_oversample = 20
phi_oversample = 20
chi_oversample_for_shift = 100
figures_dirname = "figures"
data_dirname = "analysis"
```

- [ ] **Step 6: Rewrite `configs/identification_single.toml`**

```toml
# dfxm-identify: deterministic single-dislocation sweep
# Reproduces the test set of Borgi 2025 (J. Appl. Cryst. 58, 813-821).

mode = "single"

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0

[crystal]
slip_plane_normal = [1, 1, 1]
angle_start_deg = 0.0
angle_stop_deg = 350.0
angle_step_deg = 10.0
sweep_all_slip_planes = true
exclude_invisibility = true
invisibility_threshold_deg = 10.0

[scan.phi]
value = 1.5e-4   # 8e-4° ≈ 1.4e-4 rad in Borgi 2025; this run uses 1.5e-4 rad

[noise]
poisson_noise = true
rng_seed = 0
intensity_scale = 7.0

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify"
perfect_dirname = "ignored"
include_perfect_crystal = false
```

- [ ] **Step 7: Rewrite `configs/identification_multi.toml` and `configs/identification_zscan.toml`**

`identification_multi.toml` mirrors `identification_single.toml` with `mode = "multi"` plus the existing `[multi]` block (n_samples, pos_std_um, n_png_previews).

`identification_zscan.toml`:

```toml
# dfxm-identify: z-scan mode (4D scan — depth × Burgers × angle × rocking-curve).

mode = "z-scan"

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0

[crystal]
slip_plane_normal = [1, 1, 1]
angle_start_deg = 0.0
angle_stop_deg = 350.0
angle_step_deg = 10.0
sweep_all_slip_planes = true
exclude_invisibility = true
invisibility_threshold_deg = 10.0

[scan.phi]
value = 1.5e-4    # unused in z-scan but ScanConfig requires a phi block

[noise]
poisson_noise = false
rng_seed = 0
intensity_scale = 7.0

[zscan]
z_offsets_um = [-2.0, -1.0, 0.0, 1.0, 2.0]
phi_range_deg = 0.034377467707849395
phi_steps = 21
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

Note: `[zscan].phi_range_deg` etc. stay in degrees for now — they're internal to the z-scan path, not the shared `ScanConfig`. Out-of-scope for B+C.

- [ ] **Step 8: Run tests to verify they pass**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_configs_load_under_new_schema.py -v
```

Expected: 9/9 PASS.

- [ ] **Step 9: Commit**

```bash
git add configs/ tests/test_configs_load_under_new_schema.py
git commit -m "chore(configs): migrate 9 TOML configs to new B+C schema (Task 14)"
```

---

## Task 15: Carry-forward sweep of inline TOML fixtures in tests

**Files:**
- Modify: `tests/test_pipeline.py`, `tests/test_pipeline_identification.py`, `tests/test_pipeline_multi_reflection.py`, `tests/test_hdf5_provenance.py`, any other test file with inline TOML strings or `ScanConfig(...)` / `CrystalConfig(...)` constructor calls.

- [ ] **Step 1: Inventory remaining failures**

Run the full suite and collect every failure:

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/ -q --tb=line 2>&1 | tee task15_failures.txt
```

Each failure will be one of:
- `TypeError: __init__() got an unexpected keyword argument 'phi_range'` (old `ScanConfig(phi_range=..., phi_steps=..., chi_range=..., chi_steps=...)` calls)
- `TypeError: __init__() got an unexpected keyword argument 'dis'` (old `CrystalConfig(dis=..., ndis=..., sample_remount=...)` calls)
- `ValueError: missing [crystal] block` (inline TOML strings without the new schema)
- `AttributeError: 'ScanConfig' object has no attribute 'phi_range'` (callers reading the old fields)

- [ ] **Step 2: Migration rules**

For each failure category:

**`ScanConfig(phi_range=X, phi_steps=N, chi_range=Y, chi_steps=M)` constructor calls →**
```python
ScanConfig(
    phi=AxisScanConfig(range=X, steps=N),
    chi=AxisScanConfig(range=Y, steps=M),
)
```

**`CrystalConfig(dis=D, ndis=N, sample_remount="S1")` constructor calls →**
```python
CrystalConfig(
    mode="wall",
    wall=WallCrystalConfig(dis=D, ndis=N, sample_remount="S1"),
)
```

**Inline TOML `[scan] phi_range = X; phi_steps = N` →**
```toml
[scan.phi]
range = X
steps = N
```

**Inline TOML `[crystal] dis = D; ndis = N; sample_remount = "S1"` →**
```toml
[crystal]
mode = "wall"
[crystal.wall]
dis = D
ndis = N
sample_remount = "S1"
```

**Inline TOML for identification `[scan] phi_rad = X; poisson_noise = ...; rng_seed = ...; intensity_scale = ...` →**
```toml
[scan.phi]
value = X
[noise]
poisson_noise = ...
rng_seed = ...
intensity_scale = ...
```

**Code accessing `config.scan.phi_range` →** `config.scan.phi.range`
**Code accessing `config.scan.phi_steps` →** `config.scan.phi.steps`
**Code accessing `config.crystal.dis` →** `config.crystal.wall.dis` (assert `config.crystal.mode == "wall"` first if mode is variable)

- [ ] **Step 3: Apply migration**

Walk through `task15_failures.txt` and fix each test file. Re-run after each file:

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline.py -v
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline_identification.py -v
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_pipeline_multi_reflection.py -v
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/test_hdf5_provenance.py -v
```

- [ ] **Step 4: Verify whole suite green**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/ -q
```

Expected: all PASS, 1 xfail (pre-existing HDF5 bit-equiv from baseline), 6+ skipped (from D's deselected slow tests). Same shape as the post-D baseline + ~30 new tests for B+C.

mypy check:
```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m mypy src/dfxm_geo/
```
Expected: 0 errors.

Pre-commit:
```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pre_commit run --all-files
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: carry-forward inline fixtures to new B+C schema (Task 15)"
rm task15_failures.txt 2>/dev/null
```

---

## Task 16: Final whole-PR review + cleanup

**Files:**
- Review only — no edits unless review surfaces an issue.

- [ ] **Step 1: Confirm spec coverage**

Walk through `docs/superpowers/specs/2026-05-21-scan-modes-crystal-layouts-design.md` section by section. Each "Decisions" item Q1–Q9 must have a corresponding test:
- Q1 (independent dimensions) → no test needed; absence of cross-validation is structural.
- Q2/Q3 (per-axis primitives + derived mode) → `TestScanConfigFromDict`, `TestDerivedModeName`.
- Q4 (`value` + optional `range`+`steps`) → `TestAxisScanConfig`.
- Q5 (apply to both forward + identify) → `TestRunSimulationCrystalModes` + `TestIdentificationConfigScanReusesSharedShape`.
- Q6/Q8 (discriminated union) → `TestCrystalConfigFromDict`.
- Q7 (no multi-layer) → no test needed (absence of `[[crystal.layer]]`).
- Q9 (`two_dtheta` naming) → `TestScanConfigFromDict.test_rocking_strain` uses it.
- random_dislocations sub-decisions → `TestBuildDislocationPopulationRandomDislocations`.
- HDF5 attrs → `TestHdf5NewAttrs`.
- Migration of 9 TOML configs → `test_configs_load_under_new_schema.py`.

- [ ] **Step 2: Manual smoke run**

End-to-end CLI confirmation that the default config still produces an HDF5 output:

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m dfxm_geo.pipeline `
    --config configs/default.toml `
    --output /tmp/dfxm-b-c-smoke
```

Expected: writes `/tmp/dfxm-b-c-smoke/dfxm_geo.h5` for the centered single dislocation in mosa-mode. Inspect attrs:

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -c "import h5py; f = h5py.File('/tmp/dfxm-b-c-smoke/dfxm_geo.h5'); print(dict(f['/1.1'].attrs))"
```

Expected: includes `scan_mode = "mosa"`, `crystal_mode = "centered"`.

Then test a random_dislocations run:

```powershell
# Quick edit configs/default.toml or use an inline TOML to set mode="random_dislocations"
```

Confirm sidecar exists:
```powershell
ls /tmp/dfxm-b-c-smoke/dfxm_geo_random_dislocations.json
```

- [ ] **Step 3: Review test count**

```powershell
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest tests/ -q --collect-only 2>&1 | tail -3
```

Expected: ~370 tests collected (post-D was 337; B+C adds ~30+).

- [ ] **Step 4: Push branch and open PR**

Per CLAUDE.md, confirm with Sina before pushing.

```bash
git push -u origin chore/spec-scan-crystal-modes
```

Then open the PR (using `gh pr create`) with a body summarizing:
- B: per-axis scan primitives, mode derived
- C: three-mode crystal discriminated union
- Sidecar JSON for random_dislocations
- HDF5 attrs
- 9 configs + ~30 inline fixtures migrated

- [ ] **Step 5: Merge after Sina's review**

Local merge into main (matching A+D pattern):
```bash
git checkout main
git merge --no-ff chore/spec-scan-crystal-modes
git push origin main
```

After merge confirmed:
```bash
git branch -d chore/spec-scan-crystal-modes
git push origin --delete chore/spec-scan-crystal-modes  # explicit consent required per CLAUDE.md
```
