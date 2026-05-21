---
title: Scan modes + crystal layouts (sub-projects B + C)
status: draft
date: 2026-05-21
authors: Sina Borgi (decisions), Claude Code (synthesis)
inputs:
  - docs/superpowers/specs/2026-05-20-bootstrap-multi-reflection-design.md (sub-project A — Bragg validity + per-reflection kernel files)
  - docs/superpowers/specs/2026-05-20-multi-reflection-lookup-design.md (sub-project D — config-driven kernel lookup, the precedent for "apply to forward + identify")
  - src/dfxm_geo/pipeline.py (current CrystalConfig, ScanConfig, SimulationConfig, IdentificationConfig)
  - src/dfxm_geo/direct_space/forward_model.py (module-level FOV/beam constants; Npixels=510, psize=40nm, zl_rms≈64nm)
  - configs/default.toml, configs/identification_*.toml, configs/dis_*.toml, configs/sample_remount_S2.toml (8 TOML configs requiring migration)
---

# Scan modes + crystal layouts (sub-projects B + C)

## Purpose

After sub-projects A + D (merged 2026-05-20 on `main`, untagged post-v1.1.0),
the forward and identification pipelines carry a single fixed scan shape
(2D phi×chi rocking grid) and a single fixed crystal layout (the
Borgi/Purdue dis-spaced wall). B + C remove both of those constraints in
one combined PR, generalizing the scan dimension to **any subset of
motor axes** (phi, chi, two_dtheta, z) scanned independently, and the
crystal layout to one of three mutually-exclusive modes (`centered`,
`wall`, `random_dislocations`).

Together with sub-project E (identification → HDF5 output), B + C
complete the v1.2.0 release. Sub-project F (default config flip to
"simple") is a separate v2.0.0 candidate and is explicitly **not** part
of this work — though the TOML template ships with `mode = "centered"`
active as a non-binding default starting point.

## Decisions (Q1–Q9)

Captured via brainstorming (`superpowers:brainstorming`) over Discord/
terminal on 2026-05-21.

- **Q1 — B and C are independent dimensions.** All 3×N combinations of
  scan-mode and crystal-mode are valid; no cross-validation between
  `[scan]` and `[crystal]` blocks.
- **Q2/Q3 — Scan: per-axis primitives, mode name derived.** No
  discriminated union over scan-mode names. Each motor axis is
  configured independently in TOML under `[scan.<axis>]`. The mode name
  (`single`, `rocking`, `rolling`, `strain`, `layer`, `mosa`,
  `mosa_strain_layer`, …) is computed once from which axes carry
  `range`+`steps` and used only for output labels (HDF5 attrs,
  filenames). Future translational axes (x, y) plug in as new
  `[scan.x]`, `[scan.y]` blocks with no enum updates anywhere in code.
- **Q4 — Axis block shape: `value` + optional `range`+`steps`.** Each
  axis has a center value (default 0) and is *fixed* when `range`/
  `steps` are absent, *scanned* (centered on `value`, symmetric sweep)
  when both are present. `value` non-zero with no scan = a fixed
  offset (e.g. single image at phi=150 µrad). `value` non-zero with a
  scan = a scan offset from 0 (e.g. rocking centered on a Bragg-peak
  offset).
- **Q5 — Apply to both forward + identify** in v1.2.0. The precedent
  is A + D. Identification's `IdentificationScanConfig` is replaced by
  the shared `ScanConfig`; `IdentificationCrystalConfig` (which
  describes the hypothesis space, not the actual crystal layout) is
  unchanged, but identification's inner forward calls now use the new
  `CrystalConfig` modes.
- **Q6/Q8 — Crystal: discriminated union, three mutually-exclusive
  modes.** `mode = "centered" | "wall" | "random_dislocations"`. Each
  mode's parameters live in a `[crystal.<mode>]` sub-block; sibling
  sub-blocks for other modes are rejected at parse time. Within-layer
  composition (e.g. wall + random in one crystal) is a possible future
  extension, **not** in scope for v1.2.0.
- **Q7 — No multi-layer concept.** Earlier turns of brainstorming
  explored an `[[crystal.layer]]` array-of-tables, but a clarifying
  pass collapsed it: the only target experiments for v1.2.0 are (a)
  one centered dislocation, (b) the current dis-spaced wall, (c) N
  random dislocations placed by 2D Gaussian. None require composing
  multiple layouts. The "layer" physical concept (a beam-illuminated
  z-slab of thickness `zl_rms`) is unrelated to the TOML schema —
  layer-by-layer z-resolution comes from `[scan.z]` scans.
- **Q9 — Axis name `two_dtheta`** (TOML bare keys can't start with a
  digit). Spelled this way in TOML, in dataclass field names, in
  derived mode names, and in HDF5 attrs.

### `random_dislocations` sub-decisions

- **Name** is `random_dislocations`, not `random_pairs` (despite
  earlier brainstorm history). `ndis` counts dislocations, not pairs;
  minimum value `ndis = 1`.
- **`sigma`** (2D-Gaussian width in the field-of-view plane, µm)
  defaults to FOV-derived: `sigma_default = FOV_lateral_half / 2`,
  where `FOV_lateral = Npixels * psize` from
  `direct_space/forward_model.py`. With current values (Npixels=510,
  psize=40 nm) that's `sigma_default = 0.5 * 510 * 40e-9 / 2 = 5.1 µm`.
  Users can override; if the override is too large the placement may
  draw dislocations outside the FOV, with no error (this is a
  documented soft-constraint, not a hard one).
- **`min_distance`** (µm) is optional. If present, the placement
  algorithm rejection-samples: each new dislocation must be at least
  `min_distance` from every prior one; reject and re-draw on
  violation. Hard `RuntimeError` after a fixed retry budget
  (`MAX_REJECTION_TRIES = 10_000` per dislocation) — guards against
  impossible configurations (e.g. `ndis=100, min_distance=20µm` in
  a 20µm FOV).
- **`seed`** is optional. Absent → fresh `numpy.random.default_rng()`
  seed drawn from the OS entropy pool; the **realized** seed value is
  logged into the sidecar so re-running with that seed reproduces the
  draw.
- **Sidecar file** is the canonical recovery artifact. Written next to
  the HDF5 output, format JSON for machine + human readability,
  filename = `<output_stem>_random_dislocations.json`. Schema:

  ```json
  {
    "ndis": 4,
    "sigma_um": 5.1,
    "sigma_source": "default-fov" | "user",
    "min_distance_um": 4.0,
    "seed": 1234567890,
    "seed_source": "user" | "entropy",
    "dislocations": [
      {"index": 0, "x_um": 1.2, "y_um": -3.4, "z_um": 0.0,
       "b": [1, -1, 0], "n": [1, 1, 1], "t": [1, 1, -2]},
      ...
    ]
  }
  ```

  The `[b, n, t]` triples are the *realized* slip-plane / Burgers /
  line-direction integer indices for each dislocation, drawn uniformly
  from the {111} slip-system family. ML-training consumers load the
  JSON, read `dislocations[].{x_um, y_um, b, n, t}` as features.

## Approach

Single PR, hard cut-over. Per `[[dfxm-no-backcompat-constraint]]`, the
old flat `ScanConfig` / `CrystalConfig` shapes are deleted outright —
no compat shim, no two-cycle deprecation. All 8 existing TOML configs
are migrated in the same PR. All test fixtures using the old shapes are
rewritten.

Alternatives considered and rejected:
- **Compat shim cycle.** Accept both old (`phi_range`, `phi_steps`,
  `dis`, `ndis`, …) and new shapes for v1.2.0; warn on old. Rejected:
  doubles dataclass complexity, no users on legacy shape since Sina is
  sole consumer.
- **Phased rollout (B in one PR, C in the next).** Rejected: forward
  and identify both touch `pipeline.py` heavily for either; two PRs =
  two rounds of test migration. Combined PR is half the diff churn.

## Architecture

### Files modified

- **`src/dfxm_geo/pipeline.py`** — main work:
  - **Replace `ScanConfig`** with new per-axis hierarchy:
    - New `AxisScanConfig` dataclass: `value: float = 0.0`,
      `range: float | None = None`, `steps: int | None = None`. Self-
      validating `__post_init__` (both `range`+`steps` or neither;
      `range > 0`, `steps >= 2` when scanned).
    - New `ScanConfig` dataclass holding one `AxisScanConfig` per
      canonical axis (`phi`, `chi`, `two_dtheta`, `z`); all default
      to fixed-at-0.
    - New `ScanConfig.from_dict(data: dict)` parses `[scan.<axis>]`
      sub-tables; unknown axis names → `ValueError`.
    - New `ScanConfig.derived_mode_name() -> str` computes the label
      (see "Mode-name derivation" below).
    - New `ScanConfig.scanned_axes() -> tuple[str, ...]` and
      `ScanConfig.is_scanned(axis: str) -> bool` for downstream code.
  - **Replace `CrystalConfig`** with discriminated union:
    - New `CenteredCrystalConfig`: `b: tuple[int, int, int]`,
      `n: tuple[int, int, int]`, `t: tuple[int, int, int]`. Validates
      that `b · n = 0` (Burgers vector lies in slip plane) and that
      `n × b ∝ t` (line direction is consistent), with a tolerance.
    - New `WallCrystalConfig`: `dis: float`, `ndis: int`,
      `sample_remount: str` — the current `CrystalConfig` fields,
      same validation against `SAMPLE_REMOUNT_OPTIONS`.
    - New `RandomDislocationsConfig`: `ndis: int`, `sigma: float | None
      = None`, `min_distance: float | None = None`, `seed: int | None
      = None`. Self-validating (`ndis >= 1`, `sigma > 0` when given,
      `min_distance >= 0` when given). `sigma=None` resolved at
      generate time using FOV.
    - New `CrystalConfig`: `mode: Literal["centered", "wall",
      "random_dislocations"]` + exactly one of `centered:
      CenteredCrystalConfig | None`, `wall: WallCrystalConfig | None`,
      `random_dislocations: RandomDislocationsConfig | None`.
      `__post_init__` enforces "exactly the matching sub-block is
      non-None".
    - New `CrystalConfig.from_dict(data: dict)` parses `[crystal]
      mode = "..."` + the matching `[crystal.<mode>]` sub-table;
      raises `ValueError` if sibling sub-blocks for other modes are
      present (lists which extra blocks were found).
  - **Delete current flat `CrystalConfig`** (fields: `dis`, `ndis`,
    `sample_remount`). Replaced by `WallCrystalConfig` under the new
    discriminated union.
  - **Delete current flat `ScanConfig`** (fields: `phi_range`,
    `phi_steps`, `chi_range`, `chi_steps`). Replaced by per-axis
    hierarchy.
  - **`SimulationConfig`** — no shape change at top level; `crystal`
    and `scan` fields now hold the new types. `from_toml` calls the
    new `from_dict` parsers.
  - **`IdentificationConfig`** —
    - `scan: IdentificationScanConfig` → `scan: ScanConfig` (same
      shared type as forward). The single-`phi_rad` legacy default
      becomes `[scan.phi]` with `value = 150e-6` in
      `configs/identification_*.toml`.
    - `IdentificationScanConfig` dataclass deleted. Its `phi_rad`,
      `poisson_noise`, `rng_seed`, `intensity_scale` fields move
      somewhere appropriate: `phi_rad` is now `scan.phi.value`;
      `poisson_noise`/`rng_seed`/`intensity_scale` move into a new
      `IdentificationNoiseConfig` (`noise: IdentificationNoiseConfig`
      on `IdentificationConfig`).
    - `IdentificationZScanConfig.z_offsets_um` (a list of arbitrary z
      positions) is **kept**: it's not a centered-range/steps scan
      and doesn't fit the per-axis primitive cleanly. Identification's
      `mode="z-scan"` still drives its own z-stack logic via this
      list; the `[scan.z]` block on `IdentificationConfig` is then
      **forbidden** when `mode="z-scan"` (raises `ValueError` in
      `__post_init__`).
    - `IdentificationCrystalConfig` is unchanged — it describes the
      hypothesis space (which Ud's to TEST), orthogonal to the
      crystal-mode of the inner forward call.
  - **Add `_dataclass_to_toml_str` support** for the new nested types
    so HDF5-embedded `config_toml` round-trips through `from_toml`.
    Tuples render as TOML arrays; `None`-valued sub-block fields are
    omitted; only the active `[crystal.<mode>]` sub-block is rendered.
- **`src/dfxm_geo/direct_space/forward_model.py`** —
  - Hardcoded scan-grid construction (currently using `phi_range_deg`,
    `phi_steps`, `chi_range_deg`, `chi_steps` via positional args to
    the forward kernel) gains a new helper
    `build_scan_grid(scan: ScanConfig) -> ScanGrid` that returns a
    structured representation of the (possibly N-D) scan trajectory:
    axis names + per-axis sample arrays. The forward kernel iterates
    over the Cartesian product.
  - For each axis, the sample positions are
    `value + linspace(-range, +range, steps)` when scanned, or
    `[value]` (singleton array) when fixed. Single mode = all four
    axes singleton.
  - New `build_dislocation_population(crystal: CrystalConfig,
    fov_lateral_um: float, rng: np.random.Generator | None) ->
    DislocationPopulation` dispatches on `crystal.mode`:
    - `"centered"` → 1 dislocation at origin with the given `b, n, t`.
    - `"wall"` → existing `dis`-spaced-grid generator (preserved
      unchanged, since this is the current behavior).
    - `"random_dislocations"` → Gaussian-placed N dislocations with
      rejection-sampled `min_distance` and random uniform draws over
      the {111} slip family. Returns the population AND the metadata
      block needed for the sidecar.
  - Module-level FOV constants (`Npixels`, `psize`, `zl_rms`) stay
    where they are for now; `build_dislocation_population` reads
    `Npixels * psize` directly. Migrating these to config is **out
    of scope** for B+C (tracked as a follow-up for v1.3.0+).
- **`src/dfxm_geo/pipeline.py` — runtime sites**:
  - `run_simulation`: after kernel lookup, call
    `build_dislocation_population(config.crystal, fov_lateral_um,
    rng)` and pass the result into the forward kernel. If the result
    carries random_dislocations metadata, write the sidecar before
    the HDF5 output.
  - `run_identification`: same. Identification's inner forward calls
    construct a `SimulationConfig` with `mode="centered"` for each
    hypothesis being tested — explicit, not implicit as today.
- **`src/dfxm_geo/io/hdf5.py`** —
  - HDF5 BLISS `/N.1` per-scan group gains new attrs:
    - `scan_mode` = the derived mode name (e.g. `"mosa"`,
      `"rocking_strain"`).
    - `scanned_axes` = list of canonical axis names actually scanned.
    - `crystal_mode` = `"centered" | "wall" | "random_dislocations"`.
  - The full TOML round-trip via `config_toml` (already embedded) is
    sufficient to reconstruct the run parameters; the explicit attrs
    are for fast inspection by silx/darfix/darling and for users
    reading the file without parsing TOML.
- **`src/dfxm_geo/io/sidecar.py`** (NEW file) —
  - `write_random_dislocations_sidecar(output_stem: Path, metadata:
    dict) -> Path`. Writes the JSON file, returns the path. No
    dependencies beyond stdlib (`json`).
- **`configs/default.toml`** — migrated to the new schema. Active mode
  = `centered` with example `[crystal.wall]` and
  `[crystal.random_dislocations]` blocks commented-out underneath.
  `[scan]` populated with `[scan.phi]` + `[scan.chi]` (preserving the
  current 2D rocking-grid shape) to keep the default config running a
  recognizable forward simulation out-of-the-box.
- **`configs/identification_single.toml`**,
  **`identification_multi.toml`**, **`identification_zscan.toml`** —
  migrated. `phi_rad = 150e-6` → `[scan.phi] value = 0.00015`.
  `poisson_noise`, `rng_seed`, `intensity_scale` move under
  `[noise]` block.
- **`configs/dis_*.toml`** (5 variant configs) and
  **`configs/sample_remount_S2.toml`** — migrated. Old `dis`, `ndis`,
  `sample_remount` flat fields move under `[crystal.wall]`. Scan
  fields move under `[scan.phi]` + `[scan.chi]`.

### New API surface (Python)

```python
# pipeline.py

@dataclass
class AxisScanConfig:
    value: float = 0.0
    range: float | None = None
    steps: int | None = None

    @property
    def is_scanned(self) -> bool:
        return self.range is not None and self.steps is not None

    def __post_init__(self) -> None: ...

@dataclass
class ScanConfig:
    phi: AxisScanConfig = field(default_factory=AxisScanConfig)
    chi: AxisScanConfig = field(default_factory=AxisScanConfig)
    two_dtheta: AxisScanConfig = field(default_factory=AxisScanConfig)
    z: AxisScanConfig = field(default_factory=AxisScanConfig)

    @classmethod
    def from_dict(cls, data: dict | None) -> ScanConfig: ...

    def derived_mode_name(self) -> str: ...
    def scanned_axes(self) -> tuple[str, ...]: ...
    def is_scanned(self, axis: str) -> bool: ...

# Three crystal-mode dataclasses + their union:

@dataclass
class CenteredCrystalConfig:
    b: tuple[int, int, int]
    n: tuple[int, int, int]
    t: tuple[int, int, int]
    def __post_init__(self) -> None: ...  # b·n=0, n×b∝t

@dataclass
class WallCrystalConfig:
    dis: float = 4.0
    ndis: int = 151
    sample_remount: str = "S1"
    def __post_init__(self) -> None: ...

@dataclass
class RandomDislocationsConfig:
    ndis: int
    sigma: float | None = None  # µm; None → FOV-derived at draw time
    min_distance: float | None = None
    seed: int | None = None
    def __post_init__(self) -> None: ...

@dataclass
class CrystalConfig:
    mode: Literal["centered", "wall", "random_dislocations"]
    centered: CenteredCrystalConfig | None = None
    wall: WallCrystalConfig | None = None
    random_dislocations: RandomDislocationsConfig | None = None
    def __post_init__(self) -> None: ...  # exactly-one-sub-block-set

    @classmethod
    def from_dict(cls, data: dict | None) -> CrystalConfig: ...
```

```python
# direct_space/forward_model.py

@dataclass
class ScanGrid:
    """Trajectory representation produced from a ScanConfig.

    `axes` lists the scanned axis names (subset of {phi, chi, two_dtheta, z}).
    `samples` is parallel: per-axis arrays of position values.
    Fixed axes contribute singleton arrays. The forward kernel iterates
    the Cartesian product.
    """
    axes: tuple[str, ...]
    samples: tuple[np.ndarray, ...]

def build_scan_grid(scan: ScanConfig) -> ScanGrid: ...

@dataclass
class DislocationPopulation:
    """A realized set of dislocations + optional sidecar metadata."""
    positions_um: np.ndarray   # shape (N, 3), (x, y, z)
    Ud: np.ndarray             # shape (N, 3, 3) — rotation matrices
    sidecar: dict | None       # non-None iff mode == "random_dislocations"

def build_dislocation_population(
    crystal: CrystalConfig,
    fov_lateral_um: float,
    rng: np.random.Generator | None = None,
) -> DislocationPopulation: ...
```

```python
# io/sidecar.py (NEW)

def write_random_dislocations_sidecar(
    output_stem: Path,
    metadata: dict,
) -> Path:
    """Serialize realized random_dislocations params to JSON beside the HDF5 output.

    Filename: `<output_stem>_random_dislocations.json`.
    Returns the written path.
    """
```

### Removed API

- `pipeline.ScanConfig` (flat shape with phi_range/phi_steps/chi_range/chi_steps).
- `pipeline.CrystalConfig` (flat shape with dis/ndis/sample_remount).
  Conceptually replaced by `WallCrystalConfig` as one branch of the
  new union.
- `pipeline.IdentificationScanConfig` (single-phi flat shape). Fields
  redistributed: `phi_rad` → `scan.phi.value`; noise fields →
  new `IdentificationNoiseConfig`.

### Key invariants

- `CrystalConfig.<mode>_sub_block is not None ⇔ CrystalConfig.mode == <mode>`.
  Enforced in `__post_init__` and in `from_dict`. No silent ignores.
- `ScanConfig.derived_mode_name()` is pure (same inputs always produce
  the same string). It's the **only** source of mode-name labels;
  forward + identify + HDF5 attrs + sidecar all call this method.
- `build_dislocation_population` is the **only** code path that
  realizes a dislocation population. Forward and identify both call
  it; tests verify the random_dislocations sidecar matches the
  forward kernel's actual inputs.
- The sidecar file is written **before** the HDF5 output, so a crash
  during forward kernel execution still leaves the realized draw
  recoverable.

## Mode-name derivation

```python
# Pseudocode for ScanConfig.derived_mode_name():
axes_scanned = self.scanned_axes()  # e.g. ("phi", "chi")

# Canonical 1D names per axis:
_AXIS_TO_LABEL = {
    "phi": "rocking",
    "chi": "rolling",
    "two_dtheta": "strain",
    "z": "layer",
}

# Pre-canonized multi-axis names (set membership of scanned axes,
# checked in order of decreasing specificity):
_PRE_CANONIZED = {
    frozenset(): "single",
    frozenset({"phi", "chi"}): "mosa",
    frozenset({"phi", "chi", "two_dtheta"}): "mosa_strain",
    frozenset({"phi", "chi", "z"}): "mosa_layer",
    frozenset({"phi", "chi", "two_dtheta", "z"}): "mosa_strain_layer",
}

key = frozenset(axes_scanned)
if key in _PRE_CANONIZED:
    return _PRE_CANONIZED[key]

# Otherwise: concatenate 1D names in canonical axis order
# (phi, chi, two_dtheta, z), joined by "_". E.g.:
#   {"phi", "two_dtheta"} → "rocking_strain"
#   {"chi", "z"} → "rolling_layer"
return "_".join(
    _AXIS_TO_LABEL[a] for a in ("phi", "chi", "two_dtheta", "z") if a in key
)
```

This is the **only** code path that names a scan mode. The lookup
table makes adding new pre-canonized names (e.g. when translational x,
y axes arrive) a one-line change.

## Data flow

```
User invokes: dfxm-forward --config myconfig.toml
    │
    ▼
SimulationConfig.from_toml(path)
    ├── [reciprocal] → ReciprocalConfig            (unchanged from D)
    ├── [scan] → ScanConfig                        (NEW shape)
    │     ├── [scan.phi] → AxisScanConfig(value, range, steps)
    │     ├── [scan.chi] → AxisScanConfig(...)
    │     ├── [scan.two_dtheta] → AxisScanConfig(...)
    │     └── [scan.z] → AxisScanConfig(...)
    └── [crystal] → CrystalConfig                  (NEW shape)
          ├── mode = "centered" | "wall" | "random_dislocations"
          ├── one of [crystal.centered], [crystal.wall],
          │   [crystal.random_dislocations] present
          └── sibling sub-blocks absent → ValueError otherwise
    │
    ▼
run_simulation(config, output_dir):
    │
    ├── _lookup_and_load_kernel(config.reciprocal.hkl, .keV)  (D)
    │
    ├── build_dislocation_population(config.crystal, fov_um, rng)
    │     ├── if mode == "centered": one dislocation at origin
    │     ├── if mode == "wall":     dis-spaced grid (preserved)
    │     └── if mode == "random_dislocations":
    │         ├── draw N positions from N(0, σ²) until min_distance OK
    │         ├── draw N (b, n, t) triples from {111} slip system
    │         └── return DislocationPopulation(..., sidecar=metadata)
    │
    ├── if population.sidecar is not None:
    │     write_random_dislocations_sidecar(output_stem, sidecar)
    │
    ├── grid = build_scan_grid(config.scan)
    ├── for each (phi_i, chi_j, two_dtheta_k, z_l) in grid:
    │     image = forward(population, kernel, motor_positions)
    │     save to HDF5 /1.1 (and /2.1 if perfect-crystal reference enabled)
    │
    └── write HDF5 attrs:
          scan_mode = config.scan.derived_mode_name()
          scanned_axes = config.scan.scanned_axes()
          crystal_mode = config.crystal.mode
          (+ existing config_toml round-trip)
```

## Migration plan

All in the same PR.

1. **New dataclasses** added to `pipeline.py` alongside the old ones.
   Old shapes still in place. mypy clean, smoke tests still green.
2. **Per-mode `build_dislocation_population` dispatch** added to
   `forward_model.py`. New `build_scan_grid` helper. Forward kernel
   internals not yet rewired.
3. **TDD task per crystal mode**: tests for centered, wall, and
   random_dislocations population builders — each test asserts both
   the realized dislocation array AND the metadata block (for random).
4. **Wire forward**: `run_simulation` switches from old-flat config
   reads to new shape. All forward-touching tests updated.
5. **Wire identify**: `IdentificationConfig` gains `scan: ScanConfig`,
   `noise: IdentificationNoiseConfig`. Identification tests updated.
   `IdentificationScanConfig` deleted.
6. **Delete old `CrystalConfig` and `ScanConfig`**. mypy will surface
   any missed call sites.
7. **Config-file migration**: 8 TOML configs rewritten + verified by
   loading each through `from_toml` in a regression test
   (`test_configs_load_under_new_schema.py`).
8. **HDF5 attrs**: `write_simulation_h5` gains the three new attrs.
   The existing HDF5 round-trip test (currently xfailed for the Find_Hg
   seed reason — see [[baseline-test-failures-2026-05-20]]) is
   extended to also verify the new attrs.
9. **`_dataclass_to_toml_str` update**: round-trip the new nested
   shapes. Test that `from_toml(_dataclass_to_toml_str(config))` ==
   `config` for representative configs of each mode.

### Test surface plan

- **`tests/test_pipeline_scan_modes.py`** (NEW): unit tests for
  `ScanConfig.from_dict`, `derived_mode_name`, `is_scanned`,
  `scanned_axes`. Cover all single-axis modes, mosa, mosa_strain,
  rocking_strain (non-pre-canonized), and the error cases (range
  without steps, unknown axis, range ≤ 0, steps < 2).
- **`tests/test_pipeline_crystal_modes.py`** (NEW): unit tests for
  `CrystalConfig.from_dict`. Each of the three modes parses correctly;
  sibling sub-block present → `ValueError` with which-extra-block in
  message; missing required field per mode → `ValueError`;
  CenteredCrystalConfig validates `b·n=0`.
- **`tests/test_random_dislocations_generator.py`** (NEW): tests for
  `build_dislocation_population` with `mode="random_dislocations"`.
  Verifies (a) deterministic output given a fixed seed, (b)
  min_distance enforcement, (c) sidecar metadata matches the
  realized array, (d) `MAX_REJECTION_TRIES` exhausts cleanly on an
  impossible configuration.
- **`tests/test_sidecar.py`** (NEW): JSON write + schema validation;
  round-trip a sidecar through json.load and confirm fields.
- **`tests/test_pipeline.py`** and
  **`tests/test_pipeline_identification.py`**: existing inline TOML
  fixtures (~20+ in total) rewritten to the new schema. Carry-forward
  fixture sweep, same as D's Task 6.
- **`tests/test_configs_load_under_new_schema.py`** (NEW): smoke
  loader for all 8 migrated TOML configs. One assertion per config:
  `from_toml(<path>)` does not raise.

## HDF5 schema impact

Minimal. Three new BLISS `/N.1` group attrs:

```
/1.1/scan_mode        = "mosa"               # str
/1.1/scanned_axes     = ["phi", "chi"]       # str list
/1.1/crystal_mode     = "centered"           # str
```

The existing `/1.1/config_toml` (full round-trippable TOML) is
authoritative; the attrs are inspection-friendly redundancy. Sub-
project E's BLISS-schema work consumes these directly.

The sidecar file is **not** an HDF5 entry — it sits beside the .h5,
keyed by output stem. Rationale: random_dislocations' realized data
is large (N × {position, Ud}) and primarily ML-training input, not
darfix/darling input; keeping it out of the HDF5 keeps the BLISS
schema clean.

## Out of scope (v1.2.0)

- **Translational x, y axes.** Schema accommodates them (`[scan.x]`,
  `[scan.y]` would just work), but no code path uses them yet.
- **Within-layer composition** (e.g. a `wall` + a `random_dislocations`
  population in the same crystal volume). Possible future extension;
  schema would allow it with an array-of-tables, but Q7 explicitly
  scoped it out.
- **Scan trajectories that aren't centered ranges with equal steps**
  (e.g. arbitrary list of positions like
  `IdentificationZScanConfig.z_offsets_um`). Identification's z-scan
  mode keeps its existing `z_offsets_um` list outside the
  per-axis primitive. If the need spreads, a future schema extension
  could add `[scan.<axis>] positions = [...]` as an alternative to
  `range`+`steps`.
- **Migrating module-level FOV constants** (`Npixels`, `psize`,
  `zl_rms` in `forward_model.py`) to a `[detector]` config block.
  Read directly by `build_dislocation_population` for sigma defaulting,
  no other refactor needed for B+C.
- **Sub-project F (default config flip to "simple").** TOML template
  ships with `mode = "centered"` as a starting point, but the dataclass
  defaults for `WallCrystalConfig` remain at the IUCrJ 2024 values
  (`dis=4.0`, `ndis=151`, `sample_remount="S1"`) for users
  constructing `WallCrystalConfig()` directly in code. F flips the
  conceptual default; v1.2.0 just makes the alternatives reachable.
- **Bit-equivalence with pre-B+C golden files** for any mode other
  than `wall` (which preserves current `dis`-grid behavior). The
  Fd_find golden in `tests/data/golden/Fd_find_smoke.npy` covers the
  wall path and stays the safety net there.

## Open questions / TBD

- **`IdentificationNoiseConfig` field name** — `poisson_noise`,
  `rng_seed`, `intensity_scale` move into a new block. Block name
  could be `noise` (matches current adjective) or `detector` (matches
  physics origin) or something else. Defer to implementation; check
  with Sina if any tests need explicit name awareness.
- **Sigma resolution timing.** `sigma=None` is resolved at draw time
  inside `build_dislocation_population` using FOV. Alternative: resolve
  at config-construction time (would require `CrystalConfig.from_dict`
  to know FOV, coupling config to forward_model). Chosen path: lazy
  resolution at draw time; sidecar records both the resolved value
  AND `sigma_source` ("default-fov" vs. "user").
- **JSON sidecar vs. npz sidecar.** JSON is human-readable, stdlib-
  only, easy for ML loaders. npz is faster for N >> 100 dislocations
  but adds numpy serialization. Chosen path: JSON. Revisit if a
  use-case needs N > 10,000.
