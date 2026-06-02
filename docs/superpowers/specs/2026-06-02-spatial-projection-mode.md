# Design Spec — Spatial-Projection "Full-Field" Rendering Mode

**Date:** 2026-06-02
**Author/owner:** Sina Borgi (`dfxm_geo`)
**Status:** DESIGN ONLY — no code in this change. Weighs two implementation
options and recommends one; implementation is a follow-on arc.
**Depends on / supersedes context:**
`docs/recovery-plan-darkmod-vs-go.md` (Phases 3–4),
`docs/superpowers/specs/2026-05-28-multi-reflection-oblique-angle-design.md`
(oblique reciprocal-space geometry, shipped as the unreleased v2.3.0 arc),
auto-memory `fig3_repro_darkmod_is_the_paper_model.md` (corrected taxonomy).

---

## 0. The one-paragraph problem statement

`dfxm_geo` is a **thin 2D ray-grid kinematic geometrical-optics (GO)** forward
model: each detector pixel is a depth-integrated projection over a **fixed
`NN3 = Npixels // 30 ≈ 17`-point** column whose physical half-extent is tied
to the **beam thickness** (`zl_rms = 150 nm / 2.35`, full column ≈ **0.38 µm**;
verified at px510/Nsub=1). `darkmod` (Henningsson, Borgi, Winther, El-Azab &
Poulsen, *JMPS* **204**, 106277, 2025; arXiv:2503.22022) shares the **exact same
kinematic GO physics and Poulsen-2017 resolution function**, but renders the
detector field from a **true 3D voxel volume** (`Crystal.discretize(X,Y,Z,
defgrad, density)` → per-voxel scalar intensity `p_Q · beam_weight · density`)
**projected via a tomographic line-integral operator** (`GpuProjector` →
ASTRA `parallel3d_vec` / `create_sino3d_gpu`, CUDA GPU). That projection
integrates the diffracted intensity along the **2θ diffracted-ray direction
across the full ~25 µm sample column**, which is what fills the detector field
for the paper's Fig-3-like images. This spec proposes adding that capability to
`dfxm_geo` as a **separate, explicitly-named rendering mode** — never a silent
"fix" of the GO model, and never wave-optics code.

> **Non-goal / red line (restated three times because it is the whole point):**
> This mode MUST NOT change the default geometrical-optics path. It MUST NOT be
> presented or implemented as a bug-fix to the existing thin-slab model. It MUST
> NOT introduce wave-optics (Takagi–Taupin / wavefront propagation / FFT /
> `exp(1j·…)`); the genuine wave model is the *separate* Carlsen 2022
> (arXiv:2201.07549), explicitly out of scope. The new mode is **Axis-2 spatial
> rendering only** — same Axis-1 kinematic GO diffraction physics.

---

## 1. Background: what `dfxm_geo` does today (code-verified)

### 1.1 The thin ray-grid and depth integration

`src/dfxm_geo/direct_space/forward_model.py`:

- The sample is sampled on a **ray grid `rl`** of shape `(3, NN1·NN2·NN3)`,
  built from `np.mgrid` over `(xl, yl, zl)` (`:128-134`). Counts:
  - `NN1 = Npixels // 3 · Nsub` (the `/3` is `1/sin(2·~18°) ≈ 3.24`, see §3.2),
  - `NN2 = Npixels · Nsub`,
  - `NN3 = Npixels // 30 · Nsub` = **17** at px510/Nsub=1 (`:77-79`).
- Each ray's strain → scattering vector:
  `qi = Theta @ (Us @ Hg @ q_hkl + goniometer)` (`precompute_forward_static`
  `:567`, fused in `_mc_lut_forward` `:626-638`).
- A pixel is bright where its column's `qi` lands inside the resolution-function
  LUT `Resq_i` (top-hat-limit accepted-path-length; Poulsen 2021 Eq.57), summed
  over the column's `NN3` rays, each weighted by the **beam profile**
  `prob_z = exp(-½ (rl[2]/zl_rms)²)` (`:136`).
- The `(ZI, YI)` → flat-pixel mapping `_flat_indices` (`:116-123`) collapses each
  `NN3`-ray column onto one detector pixel — i.e. the **depth integration is a
  bincount scatter**, not a geometric line integral through a volume.

### 1.2 The depth extent is the *beam thickness*, not the sample column

This is the crux for Option (a). The `zl` axis spans
`zl_start = -0.5 · zl_rms · 6` (`:97`), i.e. ±3σ of the **150 nm-FWHM Gaussian
beam profile**. Measured: full column ≈ **0.38 µm**, `NN3 = 17` rays.
`darkmod` projects the **full ~25 µm illuminated sample column** along the
diffracted ray. So `dfxm_geo`'s depth integration is **~65× too short** and is
*physically a beam-thickness convolution*, not a sample-depth projection.

Borgi 2024 (IUCrJ; doi:10.1107/S1600576724001183) is the calibration anchor:
contrast **streak length scales with beam thickness** (1.5 µm @ Δz_l = 600 nm;
2.5 µm @ 1200 nm). That scaling law is the parity oracle for any depth upgrade
(§6).

### 1.3 Why this matters for Fig-3 (the recovery-plan finding)

Per `recovery-plan-darkmod-vs-go.md` Q3/Phase 3, the dominant cause of the
"dots" was a **config/regime** effect (φ = 0.46 mrad ≈ 1.78× outside the
±0.25 mrad rocking LUT half-range), NOT the depth model. The **thin-slab depth
is the secondary, architectural contributor** to the deep-weak-beam butterfly
*extent* at off-Bragg setpoints. This spec addresses *that secondary
contributor* — and the broader goal of matching `darkmod`/Fig-3 full-field
spatial contrast — while Phase 3 (config + wider `qi1_range` kernel) handles the
primary cause separately.

---

## 2. The eta-tilted real-space oblique geometry gap (must be addressed regardless)

Code-verified incompleteness in the **oblique** mode (v2.3.0 arc), independent
of which depth option is chosen but **load-bearing** for any darkmod-parity
attempt at the Fig-3 oblique geometry:

| Quantity | Where | Current state | Correct (darkmod / removed small-angle) |
|---|---|---|---|
| `R_x(η)` azimuthal tilt | `reciprocal_space/resolution.py:263-269` | Applied to the **reciprocal** resolution function `(qroll, q2th)` only | OK in reciprocal space — but the **real-space ray grid is NOT tilted** to match |
| Imaging rotation `Theta` | `forward_model.py:114`, `:163-166` | `Theta = R_y(θ)` only — **no η tilt** | Real-space map must carry the same `R_x(η)` so the diffracted-ray direction and the ray grid are consistent |
| `xl_start` depth-skew factor | `forward_model.py:96`, `:166` | `yl_start / tan(2θ) / 3` | `yl_start / sin(2θ)` — at θ=15.42° the current factor is **0.199** vs the correct **1.95** (≈9.8× off); the `/3` is a coincidental fit at the calibrated 8.98° (`1/sin(2·8.98°)=3.24`) that **breaks at oblique θ** |
| Paraxial `ang2` | `forward_model.py:682` | `(2Δθ/2)/tan(θ₀)` | Compare to the removed-small-angle `ang2` in arXiv:2503.22022 §4 |

**Implication.** Today `R_x(η)` lives **only** in the reciprocal resolution
function. The real-space ray grid `rl` (which the strain field is sampled on)
and the imaging rotation `Theta` are still `R_y(θ)`-only and `tan(2θ)/3`-skewed.
A faithful spatial-projection mode at an oblique reflection therefore requires
**completing the real-space oblique geometry first** (tilt `rl`/`Theta` by
`R_x(η)`; fix `xl_start` to `1/sin(2θ)`), validated so that **η=0 reduces
*exactly* (bit-identical) to simplified mode**. This work is shared by both
depth options and is the recovery-plan Phase-4 prerequisite.

> Both options below assume the real-space oblique fix lands as a **gating
> sub-task** (§7, Step 0), because the depth column is built from `xl/zl_start`
> and `Theta`, which are exactly the quantities that are wrong at oblique θ.

---

## 3. The two options

### Option (a) — Cheap: extend `NN3` / the `zl` column toward the true ~25 µm depth, within the existing ray-grid

**What it is.** Keep the thin 2D-ray-grid scatter architecture. Decouple the
**depth-integration column** from the **beam-profile width**:

- Introduce an explicit **probed sample-depth** parameter (e.g. `depth_um`,
  default `None` → today's beam-tied 0.38 µm column, preserving v2.2.0
  numerics). When set, build the `zl` column over `±depth_um/2` with
  `NN3_depth` steps, while `prob_z` continues to weight by the **beam profile**
  `zl_rms` (so a thin beam over a deep column down-weights deep rays correctly,
  rather than truncating the column at ±3σ-beam).
- The geometric depth→detector skew (a ray at depth `zl` lands at lateral
  offset `zl / tan(2θ)` or `zl · …`) is **already implicit** in the
  `xl`-vs-`zl` grid construction and `Theta`; raising `NN3`/extending `zl`
  lengthens the projected streak the same way Borgi 2024 shows beam thickness
  does. The `xl_start` fix (§2) makes that skew correct at oblique θ.

**What it buys.**
- Lengthens the depth-projected streak toward the ~25 µm `darkmod` column →
  recovers the *extent* of the deep-weak-beam butterfly that the 0.38 µm slab
  truncates (recovery-plan Q3 part 3).
- **Minimal new architecture.** Reuses `rl`, `_flat_indices`, `prob_z`,
  `_mc_lut_forward`, the analytic backend, the whole HDF5/scan/fanout pipeline.
- **CPU-only, no CUDA, no new heavy dependency.** Runs on the laptop.
- Cost scales **linearly in `NN3`** — the per-frame fused kernel already loops
  `n_rays = NN1·NN2·NN3` once; a 65× deeper column is ~65× the rays
  (mitigate: keep `NN3_depth ≪ 65·17` — adaptive step so the column is sampled
  at ≥ beam-profile Nyquist, not at full 17-per-0.38µm density × 65).

**What it costs / its limits.**
- It is **still a single-column scatter onto one pixel** — it does **not** model
  the true geometric fact that a ray entering the sample at depth `z` and
  diffracting at 2θ exits through a **laterally displaced** detector pixel
  governed by the full 3D ray path + CRL magnification/inversion. The
  `xl`/`zl` skew is a *paraxial* surrogate, calibrated (the `/3`) at 8.98°.
  At large depth + oblique η this surrogate accumulates error.
- No CRL magnification/inversion, no true ray-divergence through the volume.
- The depth column reuses the **same `Hg` field sampled on the laser-thin ray
  grid** — extending `zl` means sampling `Hg` deeper, which requires the strain
  field (`find_hg_population` / `Fd_find`) to be valid over the deeper column
  (it is, for the analytic 1/r edge field; verify the population z-extent).
- **Parity ceiling:** expected to match Borgi-2024 streak-scaling and recover
  full-field *extent*, but **not** to reproduce `darkmod`'s exact tomographic
  projection geometry to ~4%. Good enough for ML training data and qualitative
  Fig-3 full-field contrast; not a metrological darkmod replacement.

**Config/mode surface (Option a):**
```toml
[render]
mode = "ray_grid"          # default; today's behaviour exactly when depth_um unset
depth_um = 25.0            # optional; probed sample-depth column (None => beam-tied 0.38 um)
depth_steps = 51           # optional NN3 override for the depth column
```
`mode = "ray_grid"` + `depth_um` unset MUST be **bit-identical** to v2.2.0
(regression-gated against the existing forward goldens — do NOT touch
`tests/data/golden/`).

### Option (b) — Faithful: true 3D voxel volume + tomographic line-integral projector (the darkmod/ASTRA pattern), or vendor/interface `darkmod`

**What it is.** Add a genuinely separate Axis-2 renderer:

1. **Build a 3D scalar intensity volume.** On a voxel grid `(m,n,o)` spanning
   the illuminated gauge volume, compute per-voxel the **same kinematic GO
   intensity** `dfxm_geo` already computes per ray:
   `I(voxel) = Res_q(qi(voxel)) · beam_weight(voxel) · density(voxel)`,
   with `qi(voxel) = Theta @ (Us @ Hg(voxel) @ q_hkl + goniometer)`. This is
   *exactly* `darkmod.crystal`'s `p_Q · beam_weight · density` (crystal.py:590)
   — and identical physics to `dfxm_geo`'s current per-ray gather, just
   evaluated on a dense 3D voxel grid instead of a thin ray column.
2. **Project the volume along the diffracted-ray direction** with a tomographic
   line-integral operator (the CT pattern): for each detector pixel, integrate
   `I(voxel)` along the straight 2θ diffracted ray through the volume, with CRL
   magnification + image inversion. This is `darkmod.projector.GpuProjector`
   (ASTRA `parallel3d_vec`, `create_sino3d_gpu`).

Two sub-paths:
- **(b1) Vendor/interface `darkmod`.** A clone exists at
  `C:\Users\borgi\tmp\darkmod` (package `darkmod/`: `crystal.py`,
  `projector.py`, `resolution.py`, `reconstruct.py`, …). Wire `dfxm_geo`'s
  `Hg`/population field + reflection geometry into `darkmod.Crystal.discretize`
  and call its projector. **Blocker:** `projector.py` `import astra` at top →
  **ASTRA needs CUDA, absent on this laptop.** Usable only on the DTU GPU
  cluster.
- **(b2) Write a CPU line-integral projector in `dfxm_geo`.** Reimplement the
  `parallel3d_vec` line-integral on CPU (Siddon/Joseph ray-marching, or a
  shear-warp). No CUDA; slower but laptop-runnable for small volumes. This is
  the path that keeps `dfxm_geo` self-contained and `noarch`-clean.

**What it buys.**
- **Metrologically faithful** to `darkmod`/Fig-3: full 3D depth, correct
  ray-path lateral displacement, CRL magnification/inversion. The real
  full-field spatial contrast the paper renders.
- Shares the **identical kinematic GO intensity** (`Res_q · beam · density`) —
  same Axis-1 physics, by construction; the only new thing is the Axis-2
  projector.
- A clean home for future `darkmod`-parity work (multi-reflection F-tensor
  reconstruction lives in the same volume representation, `darkmod.reconstruct`).

**What it costs / its limits.**
- **Heavy.** A new voxel-volume data structure + projector; large memory
  (a 272³ volume × float32 ≈ 80 MB before super-sampling, far more at darkmod
  resolution). Antithetical to the ML-throughput arc's per-frame budget.
- **(b1) needs CUDA** (cluster-only; breaks the laptop dev loop and the
  `noarch: python` conda-forge packaging — ASTRA is a compiled CUDA extension,
  see CLAUDE.md "Keep it `noarch`").
- **(b2)** a CPU line-integral projector is a non-trivial, slow new subsystem;
  per-image cost ≫ the current fused kernel; hard to keep bit-stable across
  platforms.
- **Vendoring `darkmod`** pulls its dependency tree (astra, its `crl.py`,
  `goniometer.py`) and licensing into `dfxm_geo`. Interfacing (not vendoring)
  is cleaner but couples release cadence to an external research package.
- Either way it is a **months-scale arc**, not a follow-on commit.

**Config/mode surface (Option b):**
```toml
[render]
mode = "voxel_projection"  # explicit, separate renderer
voxel_size_um = 0.1
volume_extent_um = [Lx, Ly, Lz]   # gauge volume; Lz ~ 25 um
projector = "astra_gpu"           # (b1) requires CUDA; or "cpu_siddon" (b2)
super_sampling = 1
```
`mode = "voxel_projection"` is a **distinct dispatch branch** from
`"ray_grid"`; selecting it on a CUDA-less box with `projector="astra_gpu"` MUST
fail loudly with a clear "ASTRA/CUDA unavailable; use projector='cpu_siddon' or
run on the GPU cluster" message — never silently fall back to the ray grid.

---

## 4. The mode flag / config surface (unified)

A single new `[render]` block selects the Axis-2 renderer. **The default is
unchanged behaviour.**

```toml
[render]
# "ray_grid"         -> today's thin 2D ray-grid GO scatter (DEFAULT).
#                       With depth_um unset == bit-identical to v2.2.0.
# "voxel_projection" -> Option (b): 3D voxel volume + tomographic line integral.
mode = "ray_grid"

# --- Option (a) knobs (only consulted when mode = "ray_grid") ---
depth_um = ""        # probed sample-depth column; unset => beam-tied 0.38 um (v2.2.0)
depth_steps = ""     # optional NN3 override for the depth column

# --- Option (b) knobs (only consulted when mode = "voxel_projection") ---
voxel_size_um = 0.1
volume_extent_um = [5.0, 5.0, 25.0]
projector = "astra_gpu"   # "astra_gpu" (GPU/cluster) | "cpu_siddon" (laptop, slow)
super_sampling = 1
```

Rules:
- **Absent `[render]` block ⇒ `mode="ray_grid"`, `depth_um` unset ⇒ v2.2.0
  numerics, bit-identical.** Gated by existing forward goldens.
- The renderer is selected once at config-parse time and threaded through
  `pipeline.py` forward + identify dispatch (the same place the `[geometry]`
  mode / `reflection_theta_if_oblique` is wired). **Never** auto-switch based
  on data or angle.
- `mode="voxel_projection"` is **orthogonal** to `[geometry] mode`
  (simplified/oblique): the oblique reciprocal+real-space geometry feeds the
  voxel intensities; the projector is the spatial-rendering axis.
- The chosen mode is recorded in the HDF5 provenance (`/N.1/dfxm_geo` group) so
  ML datasets are self-describing about which renderer produced them.

---

## 5. What each option buys vs costs — decision table

| Criterion | (a) Extend depth in ray-grid | (b1) Interface/vendor darkmod (ASTRA) | (b2) CPU voxel projector |
|---|---|---|---|
| Fidelity to darkmod/Fig-3 spatial contrast | Medium (extent ✓, exact geom ✗) | **High** | High |
| New architecture | Minimal (reuse fused kernel) | Large (external coupling) | Large (new subsystem) |
| CUDA required | No | **Yes (cluster only)** | No |
| Laptop dev loop | ✓ | ✗ | ✓ (small vols, slow) |
| `noarch: python` / conda-forge safe | ✓ | **✗ (ASTRA compiled CUDA)** | ✓ |
| Per-image cost vs ML-throughput arc | ~linear in NN3 (acceptable) | N/A on laptop | ≫ current (bad for 100k) |
| Reuses HDF5/scan/fanout/analytic backend | ✓ | partial | partial |
| Effort | Days | Weeks (cluster integration) | Months |
| Bit-identical default preserved | ✓ (depth_um unset) | ✓ (separate branch) | ✓ (separate branch) |

---

## 6. Parity-validation plan

Shared oracle hierarchy (cheap → authoritative):

1. **Regression floor (both options):** `mode="ray_grid"`, `depth_um` unset MUST
   be bit-identical to the current forward goldens. Run the targeted forward
   bit-equiv test only; **never modify `tests/data/golden/`** — if a change
   would alter a `*bit_equiv*` / `*snapshot*` / `*pickle_era*` golden, revert
   that change.
2. **Borgi-2024 depth-streak scaling (primary oracle for Option a):** vary the
   depth column (`depth_um` / beam thickness analogue) and confirm the
   single-edge-dislocation contrast streak length scales as Borgi 2024 reports
   (1.5 µm @ Δz_l=600 nm; 2.5 µm @ 1200 nm). This is the law the current model
   already obeys at 0.38 µm; the upgrade must extend it linearly toward 25 µm
   without breaking the slope. Validate against the rocking-COM parity scratch
   (`C:\Users\borgi\tmp\rocking_com_compare\`, RMS Δφ 3.4e-8 rad reference).
3. **darkmod cross-check (authoritative, Option b / final Option-a sign-off):**
   matched Fig-3 config (energy, hkl, θ/η, Δθ/φ/χ ranges from paper Table 1 /
   §6.2). On the GPU cluster, run `darkmod` and `dfxm_geo` `voxel_projection`
   on the same `Hg`/population; target **~4% oblique parity** (the resolution
   parity already achieved per auto-memory `darkmod_analytic_resolution_parity`).
   For Option (a), compare the ray-grid full-field image to the darkmod
   projection — expect extent/qualitative match, quantify the residual to
   decide if (a) is *sufficient* (recovery-plan Phase-3 exit criterion).
4. **η=0 exact-reduction gate (real-space oblique fix, §2/§7-Step-0):** with the
   `R_x(η)` real-space tilt + `xl_start = 1/sin(2θ)` landed, `eta=0.0` MUST
   reproduce simplified-mode output bit-identically (the same gate the v2.3.0
   reciprocal arc used). This protects the default path while completing the
   oblique geometry both options need.
5. **Full-field lit-fraction metric (Fig-3 repro):** at the corrected Fig-3
   geometry + a setpoint inside the rocking acceptance, the single-dislocation
   image must be full-field extended (≫ the ~6-lit-pixel "dots"), with a
   reported lit-fraction + bbox-fill (recovery-plan Phase-3 exit).

---

## 7. Recommendation

**Land Option (a), gated behind the explicit `[render] mode="ray_grid"` +
`depth_um` knob, AFTER first completing the real-space oblique geometry fix
(§2). Treat Option (b) as a separately-named `mode="voxel_projection"` future
arc, prototyped as (b2 CPU) for the laptop and validated against (b1 darkmod)
on the GPU cluster — built only if Option (a) fails the §6.3 darkmod parity
check.**

Rationale:
- The recovery plan establishes that thin-slab depth is the **secondary**
  contributor; the dominant Fig-3 cause (config + narrow `qi1_range`) is handled
  separately. So the cheap depth extension is the **proportionate** fix for the
  architectural part, and it preserves the whole ML-throughput arc that the
  100k-image goal depends on.
- Option (a) is CPU-only, `noarch`-safe, reuses the fused kernel + analytic
  backend + HDF5/fanout pipeline, and keeps the laptop dev loop. ASTRA's CUDA
  dependency makes (b1) cluster-only and would break `noarch: python`
  conda-forge packaging (CLAUDE.md red line).
- The real-space oblique fix (`R_x(η)` on `Theta`/`rl`, `xl_start = 1/sin(2θ)`,
  η=0 exact reduction) is **required regardless** and is the recovery-plan
  Phase-4 prerequisite — do it as **Step 0** before any depth change, since the
  depth column is built from exactly the quantities (`xl_start`, `Theta`) that
  are wrong at oblique θ.

Suggested ordering for the follow-on implementation arc (TDD, behind the flag,
default bit-identical at every step):
- **Step 0 — real-space oblique geometry (§2).** Tilt `rl`/`Theta` by `R_x(η)`;
  fix `xl_start` → `1/sin(2θ)`; gate η=0 exact reduction; address paraxial
  `ang2`. (Recovery-plan Phase 4, shared prerequisite.)
- **Step 1 — `[render]` config surface + `mode="ray_grid"` dispatch**, default
  unchanged, bit-identical goldens.
- **Step 2 — `depth_um` / `depth_steps` decoupling** the depth column from
  `zl_rms`; Borgi-2024 streak-scaling validation (§6.2).
- **Step 3 — `mode="voxel_projection"` scaffold** (loud CUDA-absent error;
  `projector="cpu_siddon"` stub), only if §6.3 shows Option (a) insufficient.

**Open decision deferred to the user (recovery-plan "Build vs adopt"):** whether
the eventual faithful path is (b1) interface `darkmod` on the cluster or (b2)
vendor a CPU projector into `dfxm_geo`; §6.3 darkmod parity decides whether (b)
is even needed. Also defer: acceptable parity tolerance vs `darkmod` (~4%?), and
whether the depth/oblique fix ships in v2.3.0 or a follow-on (per recovery-plan
"Release plan" open decision).

---

## 8. Explicit non-goals / guardrails (the red lines, restated)

1. **No silent GO-model change.** The default (`[render]` absent or
   `mode="ray_grid"` with `depth_um` unset) is bit-identical to v2.2.0; any
   spatial-projection behaviour is reachable only via an explicit, named mode.
2. **No wave optics.** No Takagi–Taupin, no wavefront/FFT propagation, no
   `exp(1j·…)`. Carlsen 2022 is the separate wave model and is out of scope.
   Both options stay on Axis-1 kinematic GO (same Poulsen-2017 resolution fn).
3. **No golden tampering.** Never edit `tests/data/golden/`; if a depth/oblique
   change would alter a `*bit_equiv*`/`*snapshot*`/`*pickle_era*` golden, that
   indicates the default path moved — revert and re-gate.
4. **No `noarch`-breaking hard dependency in the default install.** ASTRA/CUDA
   (Option b1) is an optional, cluster-only extra — never a base requirement.
5. **Loud failure, never silent fallback.** Requesting `voxel_projection` with
   `astra_gpu` on a CUDA-less host errors clearly; it must not quietly render
   the ray grid instead.
