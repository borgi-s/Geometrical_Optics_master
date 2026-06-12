# Detector noise model

The forward and identification pipelines render a noiseless, sampling-normalized
intensity image, then (when `[detector] model != "ideal"`) convert it to a
realistic uint16 ADU frame statistically indistinguishable from a raw ID03 PCO
Edge 4.2 bi frame. The conversion is

```
adu = ideal_intensity * counts_scale * exposure_time      # then noise
```

with the noise composition (gain-scaled Poisson + exposure-dependent offset +
fixed-pattern map + read/dark-shot Gaussian) implemented in
`src/dfxm_geo/detector.py` (`DetectorModel.apply`). The fitted `pco_edge_4.2_id03`
parameters (gain 2.14 ADU/photon, offset 102.5 + 7.5·t ADU, read noise 2.5–3 ADU)
come from Sina's ID03 beamtime calibration data; the full fit method and figures
are the subject of the rest of this document (Task 13).

<!-- Task 13 completes the fit-method / figures sections. The section below is
     the Task 12 data-anchor record for the `counts_scale` calibration. -->

## counts_scale derivation (data anchor)

`counts_scale` is the absolute calibration constant (ADU/s per normalized-intensity
unit) that ties the simulation's arbitrary intensity scale to measured ADU. It was
derived (script: [`docs/calibration/derive_counts_scale.py`](calibration/derive_counts_scale.py))
by matching a simulated weak-beam dislocation feature to a measured ID03 frame.
The experimental data are local-only (not in the repo) and require `hdf5plugin`
to read; the script is **provenance, not CI**.

### Scene & method

| | |
|---|---|
| Data frame | `experimental_data/111_individual_dislocations_10x_focusing_2/scan0001/pco_ff_0000.h5`, frame 30 (61×1 s focus sweep, in-focus middle) |
| Detector | PCO Edge 4.2 bi, 2048², 6.5 µm pixels, **10× focusing optics** → ~0.65 µm object-plane pixel |
| Offset subtracted | 110 ADU (`DetectorModel.offset(1 s)` = 102.5 + 7.5) |
| Sim scene | Al hkl=(−1,1,−1) @ 17 keV, single centered dislocation, weak-beam φ = 1.25×10⁻⁴ rad, `model="ideal"`, MC backend, px510 (40 nm object-plane pixel) |
| Feature select | `subtract_background(img, k=2.0) > 0`, largest connected component (`scipy.ndimage.label`) |
| Formula | `counts_scale = ADU_integral / (sim_integral · t)` |

### Results (run 2026-06-13, venv python)

| quantity | value |
|---|---|
| `ADU_integral` (data, largest feature: 62 784 px, peak 4285 ADU) | 5.006×10⁷ ADU |
| `sim_integral` (sim, largest feature: 112 px, core peak 0.7353) | 17.25 (norm. intensity) |
| FOV fraction (sim feature / frame) | 1.29×10⁻³ |
| **derived counts_scale** | **2.90×10⁶ ADU/s per norm. intensity** |
| Guard A: sim_core_peak × counts_scale × t | **2.13×10⁶ ADU** (target 2 000–6 000) → **FAIL** |
| Guard B physics: flux(1×10¹² ph/s, assumed) × FOV_frac × transmission(0.0527) × gain(2.14) | expected 1.46×10⁸ ADU/s vs derived 5.01×10⁷ ADU/s, ratio 2.91 (within 10×) → **PASS** |

Physics cross-check uses `run_exposure_simulation()` (single-objective transmission
≈ 5.3 %) and an **assumed** `flux_ID03 = 1×10¹² ph/s` (ESRF ID03 microdiffraction;
order-of-magnitude only — **to be confirmed at the beamline**, record number + URL
pending from the domain expert).

### Decision: NOT PINNED — left at the provisional `counts_scale = 1.0e4`

**Guard A fails by ~1000×; Guard B passes.** That signature is diagnostic: the
absolute *photon budget* is right (Guard B), but the *per-pixel* scale is not — a
**spatial-sampling / optics mismatch, not a units bug**.

- The data images one dislocation through **10× optics onto 6.5 µm pixels**
  (~0.65 µm object-plane pixel → one dislocation spans ~62 784 px).
- The simulation uses a **40 nm object-plane pixel on a 510² grid** (~112 px for
  the same dislocation). Linear ratio 16.25×, **area ratio ~264×**.
- The integral is sampling-invariant (Task 5 normalization), so matching
  *integrals* across the two pitches is self-consistent for total flux (Guard B),
  but then writing per-pixel ADU bakes the ~264× area ratio (× the feature-shape
  difference) into the per-pixel peak — hence the ~10⁶ ADU core, ~32× full-well.

Independent corroboration that the per-pixel anchor belongs near **~5×10³–1×10⁴**,
not 10⁶: the existing provisional `counts_scale = 1.0e4` puts the simulated core
peak at **7353 ADU** (just above the 2 000–6 000 band); the value that centers
Guard A is **~5440**. So `1.0e4` is already within ~2× of a per-pixel-correct
anchor and is a far better default than the integral-derived 2.90×10⁶.

**What the domain expert (Sina) should decide before pinning** — the right
anchoring for a *per-pixel* constant under mismatched sampling:

1. **Resample to a common object-plane pitch.** Bin/downsample the simulation to
   the data's ~0.65 µm object-plane pixel (or upsample the data feature to 40 nm)
   *before* taking the integral — removes the ~264× area factor.
2. **Match the per-pixel peak (per object-area), not the integral.** A peak-per-
   area match gives counts_scale ≈ 22 (Guard A core ≈ 16 ADU — now *too low*),
   bracketing the true value between the integral and peak-area estimates and
   pointing at the ~5×10³ range Guard A wants.
3. **Confirm the 10× optic's true demagnification and collection efficiency** for
   this beamtime — the factor folding object-plane intensity into detector ADU is
   optic-specific and is the missing physical lever here.
4. **Confirm `flux_ID03`** against the ID03 logbook (the Guard B assumption).

Until that decision, `DetectorConfig.counts_scale` stays at the provisional
`1.0e4`. Re-run `docs/calibration/derive_counts_scale.py` after adopting a
resampling/peak-area method to land both guards, then pin.
