# Config-driven detector geometry + counts_scale re-measurement (design)

**Date:** 2026-06-15
**Branch:** `feature/config-detector-geometry` (off `main` `9846fd6`)
**Context:** Toward v3.0.0. The `counts_scale` realism decision is the one
substantive v3.0.0 blocker; its derivation runs at a hardcoded detector
geometry, so the detector geometry must become config-driven before the
calibration can be matched to the experiment.

## Goal

1. Make the forward model's detector/ray-grid geometry (object-plane pixel
   pitch, `Npixels`, `Nsub`) **config-driven** via a `[detector_geometry]`
   TOML block, replacing the hardcoded module globals — with the
   block-omitted path **byte-identical** to today.
2. Use that to **re-measure `counts_scale`** at the experiment's true 10×
   geometry (`derive_counts_scale.py`) and report whether it is still a
   v3.0.0 blocker. **Do not re-pin** the shipped default in this pass.

## Out of scope

- Re-pinning `DetectorConfig.counts_scale` (the shipped noise-model default).
  Decided: stop at report; Sina reviews the number and pins separately.
- Eliminating the forward-model module globals wholesale (the `#16` refactor
  deliberately left the instrument constants global). We add a config-driven
  builder beside them; we do not remove `build_instrument_context()`.
- Any change to the resolution kernel, slip systems, or noise model.

## Background (verified)

- `forward_model.psize = 40e-9` is the **object-plane** pixel pitch (m);
  `Npixels = 510`, `Nsub = 1` are module globals computed at import, driving
  9 derived grid constants (`NN1/2/3`, `yl_start`, `YI`/`ZI`, `flat_indices`,
  the step/range pairs).
- The `#16` refactor already provides `InstrumentContext` (frozen dataclass,
  `forward_model.py:46`), `build_instrument_context()` (reads the globals,
  `:661`), and `build_forward_context(..., instrument=None)` (`:738`) which
  snapshots the globals only when no instrument is passed. The forward path
  (`forward`, `forward_from_static`, `precompute_forward_static`, `Find_Hg`,
  `Z_shift`) already consumes the grid through `InstrumentContext`/
  `GeometryContext`. **Only three sites still read the globals directly:**
  `orchestrator.py:359` (`fov_lateral_um = fm.Npixels * fm.psize`), the debug
  print (`:377`), and HDF5 provenance (`hdf5.py:1047`).
- The Hg cache key already includes `Npixels`/`Nsub` (`Find_Hg`), so changing
  geometry invalidates stale caches correctly.
- `derive_counts_scale.py` builds its own `SimulationConfig` with **no**
  `[detector_geometry]`, so it runs at px510/40 nm. It sets
  `counts_scale = ADU_integral / (sim_integral · t)` over the largest
  background-subtracted feature; Guard A checks
  `sim_core_peak · counts_scale · t ∈ [2000, 6000]` ADU; Guard B is a flux
  cross-check. Code comment (`:220`) already flags the data feature fraction
  as "~35× larger due to the 10× optics" — i.e. the mismatch is frame /
  FOV-fraction, not pitch.
- The experimental data is present locally
  (`experimental_data/111_individual_dislocations_10x_focusing_2/...`); the
  detector pixel is **6.5 µm** (`.../detector_information/pixel_size/xsize`).
  So the data's true object-plane pitch is
  `6.5 µm / (10× objective × M) = 6.5 / (10 · 17.31) = 37.6 nm` — within ~6 %
  of the code's 40 nm.

## Honest expectation

Matching the pitch is a ~6 % move (40 → 37.6 nm). The documented ~35×
discrepancy is frame-size / FOV-fraction, not pitch. So this wiring *enables*
the measurement and is worth doing for its own sake (removes a hardcoded
global, supports 2×/10×), but the re-run may well show `counts_scale` and
Guard A essentially unchanged — in which case the report states plainly that
the real fix is the FOV-fraction/normalization, not the object-plane pitch.
We measure, then decide. No outcome is assumed.

## Components

### 1. `[detector_geometry]` schema + resolution (`config.py`)

New frozen `DetectorGeometryConfig`, nested in `SimulationConfig` as
`detector_geometry` (default-constructed when the block is absent):

- TOML inputs: `pixel_size` (detector-plane effective pixel = camera pixel ÷
  visible objective, m), `magnification` (the X-ray objective magnification
  M), `Npixels` (int), `Nsub` (int).
- Resolved attribute `object_psize = pixel_size / magnification`.
- Resolution + validation in `SimulationConfig.from_toml` (and the
  identification loader):
  - Block **absent** → `object_psize = 40e-9`, `Npixels = 510`, `Nsub = 1`
    (exactly the module globals → byte-identical).
  - `pixel_size` and `magnification` must be supplied **together**; supplying
    one without the other raises a clear `ValueError`. `magnification > 0`
    and `pixel_size > 0` required; `Npixels`, `Nsub` positive ints.
  - `pixel_size`/`magnification` retained on the dataclass for provenance.

The dataclass exposes `object_psize`, `Npixels`, `Nsub` (the three values the
instrument builder needs) plus the raw `pixel_size`/`magnification`.

### 2. Config-driven instrument builder (`forward_model.py`)

Add:

```python
def build_instrument_context_from_config(
    *, psize: float, zl_rms: float, Npixels: int, Nsub: int
) -> InstrumentContext:
    """InstrumentContext from explicit geometry (config-driven), recomputing
    the derived grid constants. The module-global build_instrument_context()
    is unchanged (back-compat / tests that read globals)."""
```

It recomputes `NN1/2/3`, `yl_start`, `YI`/`ZI`, `flat_indices`, the
`xl/yl/zl` steps from the params, reuses the geometry-independent module
constants `Ud`/`Us`/`zl_rms`-derived `zl_start`, and returns the
`InstrumentContext`. The arithmetic is lifted verbatim from the module-level
derivation so a config of `(40e-9, zl_rms, 510, 1)` yields a context
bit-identical to `build_instrument_context()`.

### 3. Orchestrator seam + provenance (`orchestrator.py`, `hdf5.py`)

- Build the instrument from `config.detector_geometry` and pass it to
  `build_forward_context(..., instrument=instr)` (forward and identify
  orchestrators).
- `fov_lateral_um` (`:359`) uses `config.detector_geometry.{Npixels,
  object_psize}` instead of `fm.Npixels * fm.psize`.
- The debug print (`:377`) and HDF5 provenance (`hdf5.py:1047`) read from the
  built `InstrumentContext` (`ctx.instrument.psize`/`Npixels`/`Nsub`) so the
  recorded geometry matches the run, not the global.

### 4. `counts_scale` re-measurement (`derive_counts_scale.py`)

- Add a `[detector_geometry]`-equivalent to the script's `SimulationConfig`:
  `pixel_size = 6.5e-6 / 10` (the data's 6.5 µm camera pixel through the 10×
  objective), `magnification = 17.31` → `object_psize ≈ 37.6 nm`, matching the
  data. `Npixels`/`Nsub` start at the current defaults (510/1; the feature
  fits the ~19 µm FOV).
- Re-run; capture `counts_scale`, Guard A, Guard B for **both** the old
  (40 nm) and the matched (37.6 nm) geometry.
- Record the before/after table + verdict ("still a blocker?" + whether the
  residual is pitch or FOV-fraction) in `docs/m4-validation-report.md` under a
  new "counts_scale re-measurement (2026-06-15)" section. **No change to the
  shipped `DetectorConfig.counts_scale` default.**

## Testing

- `tests/test_detector_geometry_config.py` (new):
  - `[detector_geometry]` with `pixel_size`+`magnification` resolves
    `object_psize = pixel_size / magnification`.
  - block omitted → `object_psize == 40e-9`, `Npixels == 510`, `Nsub == 1`.
  - `pixel_size` without `magnification` (and vice-versa) raises; non-positive
    values raise.
  - `build_instrument_context_from_config(psize=40e-9, zl_rms=…, Npixels=510,
    Nsub=1)` returns a context whose `NN1/NN2/NN3/yl_start/flat_indices/psize`
    equal the module-global `build_instrument_context()` (byte-identical
    default proof).
  - an overridden `pixel_size`/`magnification` (e.g. → 37.6 nm) actually
    changes `ctx.instrument.psize` and `fov_lateral_um`.
- Regression gates that MUST stay green (default path unchanged):
  `tests/test_fcc_bit_identity.py`, `tests/test_cubic_bit_identity.py`,
  `tests/test_structure_goldens.py` (slow), and the default suite + mypy.

## Risks / decisions

- **Byte-identity.** The whole design hinges on the omitted-block default
  resolving to exactly `(40e-9, 510, 1)` and the from-config builder matching
  the module-global derivation bit-for-bit. Proven by a direct
  context-equality test, then by the determinism/golden gates.
- **`psize` naming collision.** `config.detector_geometry.pixel_size` is the
  detector-plane pixel; `forward_model.psize` / `InstrumentContext.psize` is
  the object-plane pitch (`= pixel_size / magnification`). The config field is
  named `pixel_size` (not `psize`) to keep them distinct; the resolved value
  is `object_psize`.
- **Outcome uncertainty.** The re-run may not move `counts_scale` (pitch is
  ~matched). That is an acceptable, reportable result — the deliverable is the
  measurement + an honest verdict, not a guaranteed fix.
- **Data dependency.** `derive_counts_scale.py` needs the local ID03 frames +
  `hdf5plugin`; both are present on this machine. If a future runner lacks
  them, the script fails loudly (unchanged behavior).

## Definition of done

- [ ] `[detector_geometry]` (`pixel_size`+`magnification`+`Npixels`+`Nsub`)
      parses, resolves `object_psize`, and drives the forward `InstrumentContext`.
- [ ] Block-omitted path byte-identical: context-equality test + FCC/cubic
      determinism + structure goldens green; default suite + mypy green.
- [ ] `derive_counts_scale.py` runs at the matched 37.6 nm geometry; the
      before/after `counts_scale` + Guard A/B recorded in
      `docs/m4-validation-report.md` with a plain "still a blocker?" verdict.
- [ ] `DetectorConfig.counts_scale` default unchanged.
- [ ] Branch ready for Sina's merge call (no push, no tag).
