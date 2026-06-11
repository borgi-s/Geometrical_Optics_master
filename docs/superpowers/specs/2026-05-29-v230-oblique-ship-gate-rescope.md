# v2.3.0 oblique-angle ship-gate — re-scope recommendation

**Date:** 2026-05-29
**Amends:** `plans/2026-05-28-oblique-angle-phase-a.md` (Tasks 21–22)
**Status:** recommendation, pending approval

## TL;DR
Drop the plan's **pixel-faithful Fig 3B gate** (Task 21 vetted golden + Task 22
`RMS ≤ 5e-3`). Replace it with a **resolution-parity + qualitative-contrast**
gate. The paper's Fig 3 was produced by **darkmod**, a full 3D ray-trace forward
model — not by dfxm_geo, which is a deliberate small-angle approximation. dfxm_geo
cannot reproduce darkmod's detector images pixel-for-pixel *by design*. The shared,
verifiable physics is the **reciprocal resolution function**, and dfxm's oblique
implementation already matches the paper's to ~4%.

## Evidence (2026-05-28/29 investigation)
- Paper `arXiv:2503.22022` was made with **darkmod** (Henningsson; cloned to
  `C:\Users\borgi\tmp\darkmod`). darkmod = UB crystallography + full ω/χ/ϕ rotation
  matrices + 3D CRL ray-projection (Astra, **GPU-only**) + PentaGauss resolution.
  dfxm_geo = small-angle q-offset + simplified pixel↔column mapping.
- **Oblique resolution parity confirmed:** dfxm `_build_M(θ,η)` vs darkmod
  `PentaGauss._get_M(θ,η)` for −1−13 @ 19.1 keV — covariance-ellipsoid **shape
  matches to 0.04%** (normalized eigenvalues). Only discrepancy: dfxm applies η as a
  post-hoc `R_x(η)` rigid rotation, darkmod weaves η into the geometry → ~**3.9%** on
  the thinnest (rocking) eigenvalue. (`tmp/res_parity_oblique.py`.)
- The "dots, not full field" symptom is the **spatial forward-model gap**, not the
  resolution (Fig 3C single-pixel response = broad smooth blob; covariance check above).
  Faithful Fig 3 detector images require darkmod + a CUDA GPU (not available here).

## Recommended v2.3.0 ship-gate (replaces Tasks 21–22)
**Gate A — eta=0 bit-identity (keep; this is plan Task 16).**
Default/simplified configs produce byte-identical forward + identify output vs the
v2.2.0 goldens. Guarantees the oblique work is non-breaking.

**Gate B — oblique resolution parity vs darkmod (new; replaces Task 21).**
Assert dfxm `_build_M(θ,η)` matches darkmod `PentaGauss._get_M(θ,η)` for the paper
geometry: normalized covariance eigenvalues within **1%**, full covariance within
**~5%** (allows the post-hoc-R_x vs woven-η ~3.9%). darkmod's `_get_M` is ~15 lines —
**vendor it into the test as the oracle**; no darkmod/Astra/GPU dependency. This
validates the actual new physics (η in the resolution) against the paper's own code.

**Gate C — qualitative oblique contrast (new; replaces Task 22).**
Run `dfxm-forward` on the corrected oblique config near the rocking peak; assert:
(i) a non-trivial, mostly-illuminated field (not a 1-byte / all-zero file);
(ii) a recognizable single-dislocation contrast feature near image centre (e.g. a
connected low-intensity "bowtie" region); (iii) HDF5 carries `geometry_mode=oblique`
+ `eta` + validated θ provenance. This is "produces physically sensible oblique
output", not pixel-fidelity to the paper.

## Config fixes to land alongside (`al_oblique_figure3.toml` errors found)
- **Axes per the Fig 3B caption:** `phi=+0.46e-3`, `chi=+0.067e-3`,
  `two_dtheta=-0.42e-3` (∆θ). The original config had these scrambled onto the wrong
  axes (−0.42 on φ, the narrow rocking axis → off-peak → empty).
- **Resolution params:** `zeta_v_fwhm=0.027e-3` (was 5.3e-4, ~20× too wide);
  `NA_rms=0.556e-3/2.35=2.366e-4` (was 3.112e-4).
- **psize note:** the `[detector] psize=0.75µm` is the *detector* pixel. dfxm's
  object-plane `psize` = detector_psize / M(15.1) ≈ 50 nm ≈ the default 40 nm. The
  pipeline currently uses module-default detector geometry (the `[detector]` block is
  not consumed) — document this; full wiring is deferred (below).

## Separate small task: fix `forward_model.theta_0` staleness
`theta_0` is hardcoded to 8.98° (Al 111 @ 17 keV) and never set to the run's
reflection θ → wrong for any non-default reflection (affects `Theta`, `ang2`,
`xl_start`). Thread the run's validated θ into the forward model, guarded by Gate A
(eta=0 bit-identity). Real latent bug, independent of the oblique arc.

## Deferred (explicitly OUT of v2.3.0)
- **η-weave refinement** (match darkmod's exact η geometry; the ~3.9%): optional.
- **Pixel-faithful Fig 3 reproduction:** needs darkmod + CUDA GPU — run darkmod's
  `tests/end_to_end/defrec/paper_plots.ipynb` on a GPU box as a separate validation.
- **`[detector]` block wiring** (psize/Npixels → re-derive ray grid `rl` at runtime):
  if/when dfxm must image at arbitrary detector geometries.

## Why this is the right bar
dfxm_geo's value is a fast simplified forward model (Borgi 2024 lineage), not a
darkmod replacement. The oblique arc's real deliverables are: (1) don't break
simplified mode, (2) get the η physics right where dfxm actually models it — the
resolution, (3) produce sensible oblique images. Gates A/B/C test exactly those.
Pixel-matching darkmod's Fig 3 is neither achievable with this model nor the point.

## CLOSING NOTE (2026-06-11): pixel-level eta DECIDED - accepted approximation

Roadmap M2 gaps G1.4/G1.5 are CLOSED as **Option B: accept + document**
(Sina, 2026-06-11; full options analysis in
`2026-06-11-m2-pixel-eta-decision.md`). The ~3.9 % residual on the thinnest
covariance eigenvalue of the resolution function (eta entering ONLY via the
resolution kernel, no pixel-level R_x(eta) in the direct-space ray grid) is
an ACCEPTED approximation of dfxm_geo's forward model. Rationale: contrast
statistics (the ML-training-data consumer) are unaffected; M3's B-prime
omega model is deliberately first-order at the same level; the gap is
confined to the resolution function's narrowest axis.

**Revisit triggers** (would reopen as G1.4 implement, 3-5 d + darkmod
parity re-validation; +2-3 d for the G1.5 eta-weave): any quantitative
pixel-level oblique comparison against darkmod or measured Fig-3-class
data (e.g. a paper figure), or M4/CIF-era non-cubic reflections where the
approximation is unvalidated.

With this, the M2 "decision recorded" DoD box is ticked and M2 oblique
full compatibility is 100 % closed.
