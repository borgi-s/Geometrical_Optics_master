# Realistic detector model: measured noise + absolute ADU calibration

**Date**: 2026-06-12
**Status**: draft for review
**Decided with Sina**: pipeline-wide (HDF5 frames carry noise), data-anchored
hybrid calibration, uint16 ADU storage, ON everywhere by default, sensor
artifacts included in v1.

## 1. Goal

Simulated detector frames written to HDF5 should be statistically
indistinguishable from raw ID03 `pco_ff` frames: integer ADU on a measured
noise floor, photon statistics at the right absolute count level, exposure
time as a physically meaningful knob. Weak-beam images are then displayed
(not stored) with the experimental reduction step: subtract
`mean + 2·std`, floor at 0.

This replaces the current arbitrary-units float32 frames + optional bare
Poisson pass (`[noise] poisson_noise`), whose SNR was meaningless because
the intensity scale was arbitrary.

## 2. Measured noise model (calibration provenance)

Fitted 2026-06-12 from Sina's ID03 data (PCO Edge 4.2 bi + scintillator,
`pco_ff`, 2048×2048, 6.5 µm pixels). Datasets (local only, not in repo,
`C:\Users\borgi\Documents\GM-reworked\experimental_data\`):

- `bicrystal_111_layer_rocking_scans_darks/scan0001` — 10 × 0.5 s true darks
  (shutter closed) → offset, readout σ, FPN, hot pixels.
- `bicrystal_111_find_x_edge` scans 0054/0058 (0.1 s, near-dark) and
  scan0002 (1 s, near-dark) → offset vs exposure cross-check.
- `111_individual_dislocations_10x_focusing_2/scan0001` — 61 × 1 s focus
  sweep → photon-transfer curve via adjacent-pair differences with drift
  normalization and a boxcar scene-change mask.

Fitting scripts are archived under `docs/calibration/` (see §3.8); they
require the raw data and `hdf5plugin`, and are provenance, not CI material.

| parameter | value | source |
|---|---|---|
| offset | `102.5 + 7.5·t` ADU (t = exposure in s) | darks + near-darks at 0.1/0.5/1 s, all consistent |
| readout noise σ_R | 2.5–3.0 ADU, Gaussian | photon-transfer intercept (RN² = 6.3) + dark temporal σ |
| dark-accumulation noise | folded into σ(t): var ≈ 6.3 + 11·t ADU² | dark temporal σ at 0.1 s vs 0.5 s |
| gain g | **2.14 ADU per photon-equivalent** | photon-transfer slope, linear 30–5000 ADU |
| quantization | integer ADU, uint16, clamp 65535 | raw LIMA frames |
| FPN | σ ≈ 1.8 ADU (robust), Gaussian-ish | per-pixel mean of darks |
| warm pixels | 0.57 % above median+10 ADU | darks |
| hot pixels | ~1.1e-4 fraction above median+50 ADU | darks |
| row structure | first/last ~16 rows elevated (up to ~35 ADU p2p) | dark row profile |
| image persistence ghost | faint, present in darks | NOT modeled (v2 candidate) |

Noise composition for one pixel with ideal signal `s` ADU above offset:

```
adu = round( g · Poisson(s / g) + offset(t) + FPN_px + Normal(0, σ(t)) )
adu = clip(adu, 0, 65535) as uint16
```

`g · Poisson(s/g)` reproduces the measured variance `g·s` exactly; the
scintillator excess-noise factor is inside the fitted `g` by construction.

## 3. Design

### 3.1 `DetectorModel` (new module `src/dfxm_geo/detector.py`)

Frozen dataclass + pure application function:

- Fields: `offset_base`, `dark_rate`, `read_noise_var_base`,
  `read_noise_var_rate`, `gain`, `fpn_sigma`, `warm_fraction`,
  `warm_amplitude`, `hot_fraction`, `hot_amplitude`, `edge_rows`,
  `edge_row_boost`, `full_well = 65535`.
- Preset constant `PCO_EDGE_4P2_ID03` with the table above and a docstring
  pointing at this spec.
- `model.apply(ideal: float64 array (n, ny, nx), exposure_time: float,
  rng: Generator, sensor: SensorMap) -> uint16 array`.
- `make_sensor_map(shape, model, rng) -> SensorMap`: synthesizes the
  per-pixel FPN offset map + warm/hot pixel map + edge-row profile by
  sampling the measured distributions. Synthetic, not the literal 2048²
  calibration map: scales to any simulated detector size, keeps the
  package light, and avoids cloning one physical sensor's defects.
  Warm/hot amplitude tail shapes (exponential-ish, from the dark-map
  histogram) are fixed as preset constants during implementation, sourced
  from the archived darks-characterization script output.
- The sensor map is generated ONCE per run from a dedicated RNG child and
  reused for every scan/frame in that run (it is "fixed"-pattern), and is
  reproducible for a given config seed.

### 3.2 Absolute calibration (`counts_scale` + `exposure_time`)

- `ideal_adu = intensity_sim · counts_scale · exposure_time`
- `counts_scale` [ADU·s⁻¹ per simulation intensity unit] is the data
  anchor. Default is derived once during implementation by matching the
  simulated single-dislocation weak-beam scene (M5 notebook 03 settings)
  to the measured feature levels in the dislocations dataset
  (cores ≈ 2000–6000 ADU at 1 s); the derivation run is recorded in
  `docs/detector-noise-model.md`.
- `exposure_time` default 1.0 s. Linear knob: doubling it doubles signal
  ADU above offset and noise follows physically.
- `reciprocal_space/exposure.py` (HFP transmission MC) stays a standalone
  cross-check; cited in docs, not wired into the pipeline.

### 3.3 Config schema (new `[detector]` block)

```toml
[detector]
model = "pco_edge_4.2_id03"   # or "ideal"
exposure_time = 1.0            # seconds
counts_scale = <derived>       # ADU/s per sim intensity unit
seed_sensor = true             # sensor map from config seed chain
```

- Default when block absent: **model ON** (`pco_edge_4.2_id03`) — per
  decision "on everywhere". `model = "ideal"` is the escape hatch and
  reproduces today's float32 ideal frames exactly (no offset, no noise,
  no quantization).
- `[noise] poisson_noise` and the bare intensity-scale knob are REMOVED
  (no-backcompat constraint; one noise path only). `[noise]` block
  disappears; config validation rejects it with a pointer to `[detector]`.
- All shipped config templates (`data/configs/*.toml`, `configs/*.toml`)
  drop `[noise]` and gain an explicit `[detector]` block (self-documenting
  even though absence would default the same way).

### 3.4 Pipeline seam

- The post-write pass `_maybe_apply_poisson_noise` (orchestrator) becomes
  `_apply_detector_model`: reads the combined ideal detector data, applies
  scale → model → uint16, rewrites the dataset. Same RNG child ([1] from
  spawn(2)) as today, so identification RNG layout is unchanged.
- Applied to forward AND identification outputs (all modes, incl.
  multi-reflection). Ground-truth label/sidecar datasets
  (`render_per_dislocation`, instance labels) stay noiseless — the labels
  path already bypasses the Poisson pass; that bypass is preserved.

### 3.5 HDF5 / `io` changes

- `DETECTOR_DTYPE` logic: uint16 when a real model is active, float32 for
  `model = "ideal"`.
- `[detector]` parameters + model name + spec reference written as
  provenance attrs next to the detector dataset.
- `docs/output-format.md` updated; `dfxm-migrate-h5` NOT extended (no
  back-compat constraint; old files stay readable as-is).

### 3.6 Weak-beam display helper

- `viz/detector.py::subtract_background(img, k=2.0)`:
  `clip(img - (mean + k·std), 0, None)` — Sina's reduction, validated
  against the real weak-beam frames (subtracts ≈117 ADU at 1 s).
- Used in example notebooks when displaying weak-beam images; stored data
  stays raw. Known limitation (by design): in strong-beam condition it
  zeros nearly the whole image; the GO model is not trusted there anyway.

### 3.7 Tests

- Unit: model statistics (mean/var vs spec at several signal levels and
  exposures), determinism for fixed seed, uint16 clamp, sensor-map
  reproducibility + fixedness across scans, `ideal` passthrough.
- e2e: forward + identification single/multi/zscan with model on →
  uint16 dataset, floor ≈ offset(t), SNR sanity; labels stay noiseless.
- Existing suite: test configs asserting ideal physics get explicit
  `model = "ideal"`; goldens (`tests/data/golden/`) untouched.
- mypy stays 0 errors.

### 3.8 Docs + provenance archive

- `docs/detector-noise-model.md`: the §2 table, photon-transfer figure,
  fit method, counts_scale derivation, model equation.
- `docs/calibration/`: the three fitting scripts (darks characterization,
  photon transfer, quick floor) moved from scratch, headers stating data
  requirements.
- Notebook 03 gains a short "realistic detector" display section
  (production-Nrays rule applies to any shipped images).

## 4. Out of scope (v2 candidates)

- Image-persistence ghost; scintillator PRNU rings/dust shadows (visible
  in pink-beam frames); per-pixel gain variation; camera-specific presets
  beyond the ID03 pco_ff.
- Strong-beam physics (dislocation as intensity *deficit* against the
  diffracting background. The floor now exists, so deficits clamp at the
  floor like reality, but the GO model itself remains weak-beam-only).
- darfix/darling-driven end-to-end reduction parity.

## 5. Release note

Breaking output-format change (uint16 + noise on by default + `[noise]`
block removed). Next minor/major version per Sina's call at release time;
no PyPI publish currently planned (standing decision).
