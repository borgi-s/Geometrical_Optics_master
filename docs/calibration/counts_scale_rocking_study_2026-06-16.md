# counts_scale rocking-curve study (2026-06-16)

Investigation report: how the detector absolute-intensity constant `counts_scale`
relates to measured ID03 data, what the rocking-curve comparison against the
simulation tells us, and the decisions taken. This is **provenance**, not a CI
test. The experimental data are Sina's local ID03 beamtime files (not in the
repo); reading the LIMA HDF5 frames needs `import hdf5plugin`.

## 0. The model and the question

The detector model converts the simulation's arbitrary normalized intensity into
measured ADU (`orchestrator._apply_detector_model`):

```
adu = ideal_intensity * counts_scale * exposure_time      # then offset + noise
```

`counts_scale` (ADU/s per normalized-intensity unit) is the one free absolute
constant. It had a provisional value `1.0e4`; the goal of this study was to pin
it against real data for v3.0.0.

## 1. The starting bug (resolved understanding)

`derive_counts_scale._largest_feature_integral` anchored on the **largest
connected above-background component** of a measured frame. In the 10x focusing
frame that component is a **62,784-px NON-dislocation structure**, not a
dislocation. Anchoring on its integral gave `counts_scale ~ 1e6`, which drives
the simulated detector to ~2e6 ADU at the core, **32x over the 65,535 full well**
(saturation). The "~35x FOV mismatch" recorded earlier in
`docs/m4-validation-report.md` is an artifact of selecting this wrong feature.

Real individual dislocations are ~300 px (comparable to the simulation's ~190 px),
so a correct anchor gives `counts_scale ~ O(10^2 - 10^3)` and **no saturation**.

## 2. Methods tried and what each gave

All values are far-field (pco_ff) unless noted. Different datasets/beamtimes have
different flux and optics, so absolute values differ between them.

| Method | counts_scale | Notes |
|---|---|---|
| largest-component integral (the bug) | ~1e6 | saturates; wrong feature |
| per-dislocation library matching (Borgi-2025, `dfxm_geo.scoring`) | ~420-880 | clean singles; integral vs peak basis differ ~2x |
| 2x rocking, whole-crystal plateau | ~314 | diluted by mosaic/dim regions |
| 2x rocking, field-of-interest 25x25 patch | ~700 | representative single-crystal patch |
| 9-4N fine diffry rocking, 50x50 clean patches | ~815 (median) | best far-field single-crystal rocking dataset |
| 9-4N, 250x250 patches across whole field | ~438 (median) | representative; scales linearly with patch brightness |
| pristine Al, whole-crystal mask mean | ~159 | low flux beamtime; scan not centred |

**Convergent reading:** anchored in the **weak-beam regime** (where GO is valid
and where DFXM actually images dislocations), the clean single-crystal far-field
datasets cluster at **~400-800**. The provisional `1.0e4` is **~15-30x too high**.

## 3. The rocking-curve physics we learned

### 3.1 Intensity AND shape/area are strongly rocking-curve (phi) dependent

Re-rendering the 432-candidate library at phi 150 -> 100 urad (50 urad closer to
the strong beam): per-candidate **area x21, integral x10, peak x0.64** (median).
So `counts_scale` cannot be anchored at an arbitrary phi; the simulation must sit
at the same rocking condition as the data, and the dislocation's shape/area
itself encodes its rocking offset. (`tmp/phi_rocking_compare.py`.)

### 3.2 GO is valid in the weak beam (tails), not the strong beam (peak)

Tail-fitting the rocking curve (instead of peak-normalizing) shows GO
**over-predicts the strong-beam peak by ~2.4x** relative to the weak-beam
shoulder. This is the expected failure of geometrical optics at the exact Bragg
condition. The physically meaningful anchor is therefore the **weak-beam (tail)**
regime, not the peak. (`tmp/tail_fit.py`.)

### 3.3 The simulated perfect-crystal rocking curve is too compact

The simulated perfect-crystal rocking curve (MC backend, beamstop on) is a broad
**flat-top box, FWHM ~170 urad, that drops to exactly 0 beyond +/-130 urad** (the
beamstop aperture function). Real rocking curves are **sharply peaked with tails**:

- A small (50x50) single-mosaic-block patch: FWHM ~87 urad (sharper than the sim).
- A large (250x250) patch: FWHM ~305 urad (mosaic averaging broadens it past the sim).
- The **pristine** crystal (no dislocations): more compact than the dislocation
  sample (so dislocations/mosaic add the big tails), but still has tails out to
  +300 urad where the sim is already 0.

So two distinct fidelity points:
1. **The sim resolution function is too compact** (its beamstop-carved flat-top
   lacks the smooth tails seen even in a pristine crystal). A resolution-model
   follow-up, independent of `counts_scale`.
2. **Dislocations/mosaic add the larger weak-beam tails** on top, which a
   perfect-crystal sim cannot reproduce by construction.

### 3.4 counts_scale is region- and scale-dependent

`counts_scale` per patch is essentially linear in patch brightness, and the
rocking width grows with patch size. A real mosaic crystal has **no single
"perfect-crystal intensity"**, so `counts_scale` carries an inherent **~2-3x
uncertainty** from crystal heterogeneity, the spatial averaging scale, and which
region is chosen. This is physics, not a measurement error to be removed.

## 4. Dataset survey (what is and isn't usable)

Only **far-field (pco_ff)**, **single-crystal**, **rocking (mu / diffry) scans**
are valid for anchoring against the simulated rocking curve.

| Dataset | Scan | Camera | Usable? |
|---|---|---|---|
| Z_local_rocking_2x (x11) | diffry rocking, 1 s | far field | yes (one-tailed) |
| 111_..._9-4N..._fscanfine | diffry rocking, fine, 0.2 s | far field | **yes, best** (dislocation sample) |
| Al_sample_5_pristine | chi x diffry mosa, 1 s | far field | yes but **not centred** on the peak |
| 111_..._10x_focusing | obx/obz3 focus | far field | dislocation matching only (not a rocking scan) |
| bicrystal_..._find_x_edge | omega single shots / 2D | far field | no (omega curve, single frames) |
| bicrystal_..._layer_rocking | dmmpi (mono) | **near field** | no (wrong camera + strong-beam regime) |
| pink_beam | dmmpi, broadband | near field | no |

**Terminology (Sina):** never mix scan terminology - a **rocking scan (mu motor)**
gives a **rocking curve**; an **omega scan** gives an **omega curve**. They are
different degrees of freedom. The simulation's `[scan.phi]` is the rocking angle,
so its rocking-curve comparison must use a mu/diffry rocking scan, not an omega scan.

## 5. Decisions taken

- **`counts_scale` is NOT yet pinned.** All evidence puts it at **O(10^2 - 10^3)**
  (weak-beam-regime convergence ~400-800), and confirms the provisional `1.0e4`
  is ~15-30x too high with no saturation at the real level. The exact pin remains
  open pending a centred pristine rocking scan and/or Sina's choice of reference
  scale; provisional `1.0e4` stays until then.
- **Exposure time: leave the current default as-is** (`[detector] exposure_time`,
  default 1.0 s). Because `adu = ideal_intensity * counts_scale * exposure_time`,
  the absolute brightness is the product of `counts_scale` and `exposure_time`;
  rather than over-tuning the absolute constant, **we keep `exposure_time` as a
  user-facing knob** so users can dial simulated brightness / SNR **up or down**
  to match their target detector and acquisition. No code change is needed -
  `exposure_time` is already threaded through `detector.py` (offset, noise, ADU
  scaling) and configurable per run in the `[detector]` block.

## 6. Open items / next steps

1. **Pin `counts_scale`** (~500-700, weak-beam regime) OR analyse a **centred
   pristine rocking scan** (both tails) to fix the perfect-crystal shape + scale
   cleanly. (More datasets were being collected.)
2. **Resolution-function follow-up:** the simulated perfect-crystal rocking curve
   is ~2x too broad and a flat-top box vs the real peaked shape; the beamstop
   model may be carving the tails too hard. Separate from `counts_scale`.
3. **Correct the stale conclusions** in `docs/detector-noise-model.md` Sec.5 and
   `docs/m4-validation-report.md` (the "FOV-fraction / 35x mismatch" framing is
   the wrong-feature artifact) once the pin is settled.
4. Identification is **independent of `counts_scale`** (it uses std-normalized
   cross-correlation), so it can ship regardless of the pin.

## 7. Session scratch (figures + scripts, in `GM-reworked/tmp/`)

`pin_counts_scale.py` (per-dislocation matching), `phi_rocking_compare.py`
(phi dependence), `anchor_rocking.py` (whole-crystal plateau),
`patch_anchor.py` (field-of-interest patch), `rocking_survey.py` (dataset survey),
`rocking_vs_sim.py` (9-4N shape+scale), `rocking_largepatch.py` (250x250 sweep),
`tail_fit.py` (weak-beam tail fit), `pristine_rocking.py` / `pristine_patches.py`
(pristine sample). Figures share those names with a `.png` extension. These are
scratch and may be cleaned; this report is the durable record.
