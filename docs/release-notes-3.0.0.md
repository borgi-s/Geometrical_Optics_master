# DFXM v3.0.0 — Arbitrary crystal structures + realistic detector

Released: 2026-06-16.

`dfxm-geo` is a geometrical-optics forward and identification simulator for Dark
Field X-ray Microscopy (DFXM). v3.0.0 is a **major release**: it generalizes the
simulator from FCC-only to **arbitrary crystal structures via CIF**, adds **BCC and
HCP slip-system physics**, and replaces the old additive-noise model with a
**measured, realistic uint16-ADU detector model**. This note is the full picture
of what v3.0.0 can do.

---

## Headline

- **Any crystal structure.** Load a `.cif`, or declare FCC / BCC / HCP directly.
  Slip systems, Burgers vectors, structure-factor extinction, and the
  general (triclinic-capable) cell geometry are all derived for you.
- **Realistic detector.** A measured ID03 `pco_ff` PCO-edge model turns the
  simulation's normalized intensity into 16-bit ADU frames (gain, dark offset,
  read noise, Poisson, full-well clamp). The old `[noise]` block is gone.
- One forward engine, two paths: **`dfxm-forward`** (render dislocation images)
  and **`dfxm-identify`** (deterministic hypothesis sweeps for dislocation
  identification), both writing BLISS-schema HDF5.

---

## ⚠ Breaking changes

1. **`[noise]` → `[detector]`.** `IdentificationNoiseConfig` is replaced by
   `DetectorConfig`. Configs that used `[noise]` must migrate to `[detector]`:

   ```toml
   [detector]
   model = "pco_edge_4.2_id03"   # realistic uint16 ADU; or "ideal" for raw float32
   exposure_time = 1.0           # seconds (user-tunable brightness/SNR knob)
   rng_seed = 0
   ```

   Set `model = "ideal"` to recover the previous noiseless normalized-intensity
   float output.

2. **Non-cubic cells require material context.** A non-FCC structure that sets
   neither `material` nor `poisson_ratio` in `[crystal]` now raises (so strain is
   never silently computed with Al's elastic constants). FCC default runs are
   unaffected and remain byte-identical.

3. Major version: pin `dfxm-geo==3.0.0` if you depend on the v2.x `[noise]`
   schema or FCC-only assumptions.

---

## Everything v3.0.0 can do

### Crystal structures & slip systems
- **CIF ingestion** (`[crystal] cif = "..."`, lazy `gemmi` backend, optional
  `pip install dfxm-geo[cif]`): space group → structure type, lattice parameters,
  and structure-factor **extinction rules** (systematic absences) are resolved
  automatically. Cubic, hexagonal, and lower-symmetry cells are supported.
- **FCC / BCC / HCP** out of the box via a data-driven slip-system registry
  (`crystal/slip_systems.py`): FCC `{111}<110>` (12), BCC `{110}<111>` + `{112}<111>`
  (24 systems), HCP basal / prismatic / pyramidal `<a>` and 1st/2nd-order
  pyramidal `<c+a>` (30 systems, Miller–Bravais 4-index). Per-dislocation Burgers
  magnitudes (`<a>` = a, `<c+a>` = √(a²+c²)) thread through the numba kernels.
- **General cell geometry**: a Bravais cell builder turns any lattice into the
  real/reciprocal frames used by the forward model; the FCC cubic path stays the
  exact legacy code (determinism gates green).
- **Per-material elasticity** (`crystal/elasticity.py`): cited Poisson ratios per
  material; strain fields use the right ν.

### Reflections
- **Multi-reflection** runs via `[[reflections]]`: one manifest renders or
  identifies across several hkl reflections, with per-reflection ω geometry,
  shared-parameter RNG, and g·b visibility labels.
- **`dfxm-find-reflections`**: enumerate Laue-reachable reflections for a given
  structure, energy, and mount; reports ω/η and Bragg validity so you can pick a
  reachable reflection before simulating.

### Geometry & scan modes
- **Symmetric and oblique** diffraction geometry (`[geometry]`): the oblique mode
  carries an arbitrary surface mount (required for non-cubic reflections, where a
  reflection's η is computed, not assumed).
- **Scan trajectories** as independent per-axis primitives: `[scan.phi]`,
  `[scan.chi]` (the 2D "mosa" rocking grid), `[scan.two_dtheta]` (within-scan
  strain axis), `[scan.z]` (z translation). Any axis is fixed (default 0) or
  scanned (`value` + `range` + `steps`); the scan-mode label is derived.
- **Crystal layouts**: `centered` (single dislocation), `random_dislocations`
  (seeded population), and `wall` (dislocation wall with `sample_remount`).

### Optics / resolution backends
- **Two interchangeable backends** (`[reciprocal] backend`):
  - `mc` — the fused `@njit` Monte-Carlo resolution kernel (default; required
    when a beamstop is on).
  - `analytic` — the PentaGauss analytic resolution function (no kernel build;
    smooth; not compatible with a beamstop, which has no closed form).
- **Beam-stop and apertures in the back focal plane (BFP)**: square-aperture
  beam stop (`bs_height`), objective aperture (`aperture`), and knife edge
  (`knife_edge`) are modelled at the BFP; the optional `[beamstop-wire]` extra
  (`xraylib`) adds wire-absorption physics.
- Beam/objective parameters (NA, energy spread, condenser aperture) are all
  configurable in `[reciprocal]`.

### Forward simulation (`dfxm-forward`)
- Renders dislocation strain-contrast detector frames for the configured crystal,
  reflection(s), scan trajectory, and detector model.
- **Config-driven detector geometry** (`[detector_geometry]`, single-reflection
  forward): set the camera `pixel_size` and `magnification` to get the right
  object-plane pixel pitch; the default path is byte-identical.
- Optional perfect-crystal `/2.1` output alongside the dislocation `/1.1`.

### Identification (`dfxm-identify`)
- Deterministic single / multi / z-scan hypothesis sweeps reproducing the
  Borgi-2025 test design (slip planes × Burgers × character angles, minus g·b
  near-invisibility), each candidate labelled with its slip system, Burgers
  vector, rotation, and g·b visibility.
- **`dfxm_geo.scoring`**: FFT cross-correlation of candidate libraries — an
  all-pairs identifiability study and single-target ranking over an identify
  HDF5, with physical resampling onto the library grid (numpy default, optional
  torch). Identification is independent of the absolute `counts_scale`.

### Detector model
- Measured **PCO-edge 4.2 (ID03 `pco_ff`)** model: gain ≈ 2.14, exposure-dependent
  dark offset, read noise, Poisson statistics, and a 65,535-ADU full-well clamp,
  producing **uint16 ADU** frames. `model = "ideal"` emits raw float32 physics.
- **`exposure_time` is a first-class user knob** (`[detector] exposure_time`,
  default 1.0 s): simulated brightness = normalized-intensity × `counts_scale` ×
  `exposure_time`, so you dial brightness / SNR up or down to match your
  detector and acquisition. See *Known limitations* for the `counts_scale` status.

### Outputs & migration
- **BLISS-schema HDF5** master + per-scan layout for both forward and identify;
  detector data as float32 (raw) or uint16 (realistic).
- `dfxm-migrate-output` (legacy `.npy` → HDF5) and `dfxm-migrate-h5`
  (v1.1 → v1.2 layout) migration tools.
- `[io] write_strain_provenance = false` drops the per-scan Hg dump
  (106 MB → ~32 KB per config) for large ML/batch runs.

### Performance & cluster
- Fused `@njit` `Find_Hg` population/scene kernels (numpy oracle retained for
  parity); persistent fan-out worker pool.
- `scripts/fanout.py` in-node launcher (configs × threads via `DFXM_MAX_WORKERS`,
  pinned BLAS, batch-resilient) and **drafted LSF cluster scripts**
  (`lsf/fanout.bsub`) for DTU-HPC-style fan-out toward large (100k+) image sets.

### Reproducibility
- The structure showcase (`scripts/render_structure_showcase.py`: FCC Al, BCC W,
  HCP Ti) is locked by regression-snapshot goldens
  (`tests/test_structure_goldens.py`), and FCC determinism is bit-pinned.

---

## CLI commands

| command | purpose |
|---|---|
| `dfxm-forward` | render dislocation images for a config |
| `dfxm-identify` | deterministic identification hypothesis sweeps |
| `dfxm-find-reflections` | enumerate Laue-reachable reflections (ω/η, Bragg validity) |
| `dfxm-bootstrap` | (re)generate the MC resolution kernel |
| `dfxm-init` | scaffold a starter config |
| `dfxm-migrate-output` | legacy `.npy` → HDF5 |
| `dfxm-migrate-h5` | v1.1 → v1.2 HDF5 layout |

---

## Known limitations

- **`counts_scale` is provisional (`1.0e4`).** The absolute normalized-intensity
  → ADU constant is not yet pinned to a single value. A 2026-06-16 rocking-curve
  study (`docs/calibration/counts_scale_rocking_study_2026-06-16.md`) establishes
  that the true value is **O(10²–10³)** (weak-beam-regime convergence ~400–800)
  and that the provisional `1.0e4` is on the high side, but a real crystal is
  mosaic so the constant carries an inherent ~2–3× uncertainty. **In practice,
  tune `exposure_time` to set brightness;** identification is unaffected
  (it uses normalized cross-correlation).
- The simulated perfect-crystal rocking curve is somewhat more compact than a
  real one (a resolution-function fidelity item, tracked separately).
- Isotropic elasticity only (no full `C_ijkl`) — a v3.x item.

## Upgrade

```bash
pip install --upgrade dfxm-geo==3.0.0
# CIF support: pip install "dfxm-geo[cif]"
```

Migrate any `[noise]` block to `[detector]` (see Breaking changes) before
upgrading.
