# DFXM Forward-Model Taxonomy — two orthogonal axes

**Date:** 2026-06-02
**Owner:** Sina Borgi (`dfxm_geo`)
**Status:** Reference. Cross-references
[`docs/recovery-plan-darkmod-vs-go.md`](recovery-plan-darkmod-vs-go.md).

This note exists to stop a recurring mistake: conflating `darkmod`'s
*spatial-rendering* choice (3D-voxel tomographic projection via ASTRA) with a
*different diffraction physics* (wave / dynamical optics). It is **not** — both
`darkmod` and `dfxm_geo` are the same kinematic geometrical-optics (GO) family.
A prior session misread this and concluded a faithful reproduction of
arXiv:2503.22022 Figure 3 was architecturally impossible with `dfxm_geo`. That
conclusion was wrong; see the recovery plan for the full refutation and primary
sources.

---

## Why two axes

A DFXM forward model makes two **independent** modelling choices. Describing a
model by only one axis (e.g. "GO vs ray-trace vs wave optics") collapses these
and produces category errors.

### Axis 1 — diffraction physics

How the X-ray scattering itself is computed.

- **Kinematic (geometrical optics).** Bragg scattering is evaluated on
  infinitesimal sub-volumes of the crystal; the imaging instrument's acceptance
  is applied as a **reciprocal-space resolution function** that gates which
  local scattering vectors `q` reach the detector. No phase propagation, no
  interference, no Pendellösung. The shared kernel is the Gaussian reciprocal
  resolution function of Poulsen et al. (2017). `dfxm_geo` (Borgi 2024) and
  `darkmod` (arXiv:2503.22022) both live here.
- **Dynamical (wave / wavefront).** The full X-ray wavefield is propagated
  through the deformed lattice via the Takagi–Taupin equations (TTE);
  coherence, interference, and Pendellösung are captured. This is a genuinely
  different physics. Carlsen et al. (2022, arXiv:2201.07549) is the one true
  wave-optics DFXM model.

### Axis 2 — spatial rendering

How sample-space points are mapped onto detector pixels (the *geometry* of
image formation, independent of Axis 1).

- **Thin 2D ray-grid.** A grid of rays is scattered/depth-integrated over the
  thin illuminated slab (the beam thickness). `dfxm_geo` sums `NN3 = 17`
  `prob_z`-weighted depth rays per pixel along the 2θ diffracted-beam direction
  — a depth-integrated projection over its ~0.45 µm beam-thickness slab.
- **3D voxel tomographic projection.** A full 3D scalar-intensity volume is
  computed, then projected onto the detector by a **CT line-integral operator**
  along straight diffracted rays (with CRL magnification + inversion).
  `darkmod` uses ASTRA (`parallel3d_vec` / `create_sino3d_gpu`) for exactly
  this. ASTRA is a tomographic line-integrator — **not** a wavefront propagator
  and **not** a diffraction ray-tracer.
- **Full wavefield.** The propagated complex wavefield is sampled at the
  detector (Carlsen 2022).

---

## Placement of the three models

| Model | Axis 1 (physics) | Axis 2 (rendering) | Notes |
|---|---|---|---|
| **`dfxm_geo`** (Borgi 2024, IUCrJ) | kinematic GO | thin 2D ray-grid (~17-point depth integration) | analytic + MC-LUT resolution backends; forward-only |
| **`darkmod`** (arXiv:2503.22022) | kinematic GO — **same family as `dfxm_geo`** | 3D voxel ASTRA tomographic projection | small-angle approximations removed (arbitrary η, ω → oblique/full-ω); adds multi-reflection F-tensor **inverse** reconstruction |
| **Carlsen 2022** (arXiv:2201.07549) | **dynamical (TTE / wave optics)** | full wavefield propagation | the one genuine wave-optics DFXM model |

`dfxm_geo` and `darkmod` sit on the **same** side of Axis 1 (kinematic GO,
shared Poulsen-2017 resolution function). They differ on Axis 2 (thin ray-grid
vs 3D-voxel projection) and in scope (`darkmod` removes small-angle
approximations and adds an inverse problem). **`darkmod` is not wave optics.**

### Primary-source citations (arXiv:2503.22022)

- **§4 (line 490):** *"Similar to Poulsen et al. (2021) we adapt a geometrical
  optics approach … utilize the same Gaussian reciprocal resolution function
  described in Poulsen et al. (2017)."* → same kinematic GO physics as
  `dfxm_geo`.
- **§4.2 (line 584):** ASTRA enters only as the tomographic line-integral
  projection operator (CT-style), applied to an already-computed scalar
  intensity volume.
- **§7 (line 1263):** *"a kinematical scattering approximation … Borgi et al.
  (2024) verifies that such approximations are well justified in the weak beam
  limit."*

(arXiv:2503.22022 is published as: Henningsson, Borgi, Winther, El-Azab &
Poulsen, *J. Mech. Phys. Solids* **204**, 106277, 2025.)

`darkmod` source confirms the absence of wave-optics primitives (no
fresnel/fft/takagi/taupin/wavefront/`exp(1j)`/dynamical): `resolution.py` is
documented as *"Vectorization of Poulsen 2017"*; `crystal.py:590` computes
intensity as `p_Q * beam_weight * density` (a kinematic product, not a
propagated field).

---

## GO ↔ wavefront convergence (the cross-validation)

Borgi 2024 (IUCrJ) is the load-bearing cross-validation between the two Axis-1
families: GO and full wavefront propagation *"become nearly identical"* in the
**weak-beam limit** — precisely the regime in which dislocations are imaged in
DFXM. This is why a kinematic GO model is physically justified for dislocation
imaging, and why `darkmod` (arXiv:2503.22022 §7) explicitly cites Borgi 2024 to
license its kinematic approximation. The two Axis-1 families are not in tension
for DFXM dislocation contrast; they agree where it matters.

---

## Figure 3: panel B (image) vs panel C (angular scan-grid "dots")

A specific reading hazard in arXiv:2503.22022 Figure 3 that seeded the earlier
misdiagnosis:

- **Panel B is a full-field 272×272 spatial image.** GO produces full-field,
  spatially extended dislocation contrast at and near the Bragg condition
  (empirically verified: an on-Bragg single dislocation fills 100% of the field
  with an extended butterfly + dark core). Near-point features appear only in
  the deep weak-beam regime, which is physically correct.
- **Panel C is the per-pixel ANGULAR scan grid** for one detector pixel `u*` —
  a *"point-wise supported … discrete and non-uniform grid in Δθ, φ, χ"*. The
  "dots" in panel C are the **angular scan-grid samples**, not the spatial
  image and not a model deficiency.

**Do not read panel-C dots as a `dfxm_geo` model gap.** The earlier "GO renders
dislocations as sparse dots / faithful Fig-3 repro is architecturally
impossible" framing has been **retracted**. The observed "dots" in the
`dfxm_geo` Fig-3 runs trace to a config/regime effect — a φ ≈ 0.46 mrad
setpoint that pushes the rocking component ~1.78× outside the on-disk kernel's
±0.25 mrad rocking-LUT acceptance, plus a secondary (fixable) thin-slab depth
contribution — not to a difference in diffraction physics. See
[`docs/recovery-plan-darkmod-vs-go.md`](recovery-plan-darkmod-vs-go.md) §Q3 and
the recovery phases for the evidence and the corrective plan.
