# DFXM Forward-Model Recovery Plan — Corrected Model Taxonomy & Get-Back-On-Track

**Date:** 2026-05-29
**Author/owner:** Sina Borgi (dfxm_geo)
**Trigger:** A prior session concluded that the geometrical-optics (GO) DFXM forward model renders dislocations as sparse "dots," that arXiv:2503.22022 used a fundamentally different "full 3D ray-trace (Astra GPU)" model (`darkmod`), and that a faithful reproduction of that paper's Figure 3 is therefore impossible with `dfxm_geo` — i.e. an *architectural* gap. This document refutes those conclusions against primary sources (papers + `darkmod` source + `dfxm_geo` code + empirical runs) and lays out the recovery.

**Provenance:** Produced by an unbiased multi-agent investigation (6 independent investigation streams + 4 adversarial verifiers + synthesis). The headline empirical result was independently re-verified by the lead session: the saved arrays in `C:/Users/borgi/tmp/go_evidence/` were re-loaded (`im_on_bragg_zero.npy` → 100% nonzero; `im_fig3B_setpoint.npy` → 0.2% nonzero) and the PNGs viewed directly. Evidence images: `on_bragg_zero.png` (full field + dark dislocation), `weakbeam_phi_150urad.png` (full-field butterfly), `fig3B_setpoint.png` (single dot).

---

## Corrected understanding

### Q1 — Does GO produce full-field or sparse contrast, and by what mechanism?
GO (Poulsen 2017/2021; Borgi 2024 = `dfxm_geo`) produces **full-field, spatially extended** dislocation contrast at and near the Bragg condition. Architecturally each detector pixel is a **depth-integrated projection** along the 2θ diffracted-beam direction (Poulsen 2021: *"the image is projected along the 2theta angle"*; *"the resolution function imposes a 1D integration of the deformation field in the sample along a direction normal to the object plane"*; gauge volume *"75 × 204 × 600 nm³"*). `dfxm_geo` realizes exactly this — **code-verified** that each pixel sums `NN3=17` depth rays, `prob_z`-weighted (`forward_model.py:128-136`, `:626-638`; reproduced rays/pixel min=max=mean=17).

**Mechanism (kinematic reciprocal-space gating):** the local displacement-gradient field `Hg(r)` (1/r edge field) maps each sample point to a scattering vector `qi = Theta @ (Us@Hg@q_hkl + goniometer)`; a pixel is **bright** where its column's `qi` lands inside the resolution function `Res_q` and **dark** where it falls outside (top-hat limit: intensity = accepted path length, Poulsen 2021 Eq.57). A single edge dislocation renders as an extended structured feature: Poulsen 2021 Fig.10 forward-simulates one edge dislocation matching experiment; Borgi 2024 reports 1.5–2.5 µm contrast streaks. Empirically (oblique Fig-3 kernel, 272 px) an on-Bragg single dislocation fills 100% of the field with an extended butterfly + dark core; a rocking-scan COM-φ map is fully extended. Near-point features (~8–14 nm radii, Borgi 2024) appear **only** in the deep weak-beam regime — physically correct (Res_q integrates out Hg-components below ~0.0011) and exactly where GO matches full wavefront propagation.

### Q2 — What physics does arXiv:2503.22022 / `darkmod` actually use? What is ASTRA for?
**Geometrical optics under the kinematic approximation — the same family as `dfxm_geo`.** arXiv:2503.22022 §4 (line 490): *"Similar to Poulsen et al. (2021) we adapt a geometrical optics approach … utilize the same Gaussian reciprocal resolution function described in Poulsen et al. (2017)."* §7 (line 1263): *"a kinematical scattering approximation … Borgi et al. (2024) verifies that such approximations are well justified in the weak beam limit."* It is **not** wave optics; it names dynamical-diffraction wave propagation (Carlsen et al. 2022, Takagi-Taupin) as a *separate, out-of-scope* approach. `darkmod` source confirms: zero wave-optics primitives (grep for fresnel/fft/takagi/taupin/wavefront/exp(1j)/dynamical → no matches); `resolution.py` = *"Vectorization of Poulsen 2017"*; `crystal.py:590` intensity = `p_Q * beam_weight * density`.

**ASTRA is only a tomographic line-integral projection operator P** (arXiv §4.2 line 584; `projector.py` `parallel3d_vec`, `create_sino3d_gpu` + adjoint `create_backprojection3d_gpu`): it integrates an *already-computed* scalar intensity volume along straight diffracted rays, exactly as in CT. ASTRA is **not** a wavefront propagator and **not** a diffraction ray-tracer.

(arXiv:2503.22022 is now published: Henningsson, Borgi, Winther, El-Azab & Poulsen, *J. Mech. Phys. Solids* **204**, 106277, 2025.)

### Q3 — Root cause of the "dots"
A **config/regime** effect, not architecture, in three evidence-backed parts:
1. **Setpoint (dominant):** the Fig-3B setpoint φ=+0.46 mrad pushes the perfect-lattice rocking component `qi1` to +0.4438 mrad — **~1.78× outside** the on-disk kernel's rocking LUT half-range of ±0.25 mrad (`qi1_range=5e-4`). ~99.99% of rays exit resolution support → ~6 lit pixels. The collapse is *progressive* in φ (0.05 mrad → 99.7% lit; 0.15 → ~95%; 0.20 → 8.5%; 0.46 → dots), proving a continuous acceptance effect. χ and Δθ land inside their wide ±3.75 mrad windows and are harmless.
2. **Rocking-window width:** `qi1_range=5e-4` may be too narrow for the Fig-3 field (single-dislocation `qi1` spreads to ±7.8e-4 near the core); widening it or correcting φ restores full-field contrast.
3. **Secondary architectural contributor:** the deep-weak-beam butterfly *extent* at φ=0.46 mrad is partly built by depth projection; `dfxm_geo` integrates only over its ~0.45 µm beam-thickness slab (`NN3=17`) vs `darkmod`'s ~25 µm full-3D-volume projection. Borgi 2024 confirms streak length scales with beam thickness (1.5 µm @ Δz_l=600 nm; 2.5 µm @ 1200 nm).

Full-field GO images are achievable; the immediate cause is the off-acceptance φ + narrow rocking LUT, with thin-slab depth secondary.

### Q4 — Real differences `dfxm_geo` vs `darkmod`
**Scope and spatial-rendering, not a GO-vs-wave split.** Both are kinematic GO with the same Poulsen-2017 resolution function. `darkmod` adds: (1) removal of several small-angle approximations (η, ω arbitrary → oblique/full-ω); (2) a 3D-voxel-volume model projected via ASTRA with CRL magnification + inversion, vs `dfxm_geo`'s thin ~17-point ray-grid scatter; (3) detector PSF/noise (though `dfxm_geo` also has these); (4) a multi-reflection **inverse** problem — per-pixel least-squares of all 9 deformation-gradient components from ≥3 reflections (`reconstruct.deformation`) — `dfxm_geo` is forward-only; (5) closed-form analytic resolution (PentaGauss) vs `dfxm_geo`'s MC LUT + analytic backend. `dfxm_geo` additionally has documented small-angle incompleteness in oblique mode (η only in the reciprocal resolution function via `R_x(η)`, not the real-space map `Theta = R_y(θ)`; paraxial `ang2 = (2dθ/2)/tan(θ0)`; `xl_start` hardcoded `/3` calibrated for θ0=9°).

### Correct taxonomy (two orthogonal axes)
- **Axis 1 — diffraction physics:** *kinematic* (GO; Bragg on infinitesimal sub-volumes; resolution-function gating) vs *dynamical* (wave/wavefront propagation; Takagi-Taupin; coherence, Pendellösung).
- **Axis 2 — spatial rendering:** thin 2D ray-grid vs full 3D voxel tomographic projection vs full wavefield.

| Model | Axis 1 | Axis 2 | Extra |
|---|---|---|---|
| `dfxm_geo` (Borgi 2024) | kinematic GO | thin 2D ray-grid | analytic + MC-LUT backends; forward-only |
| `darkmod` (arXiv:2503.22022) | kinematic GO | 3D voxel ASTRA projection | small-angle removed; multi-reflection F-tensor reconstruction |
| Carlsen 2022 (arXiv:2201.07549) | **dynamical (TTE)** | wavefront propagation | the one genuine wave-optics DFXM model |

`dfxm_geo` and `darkmod` are on the **same** side of Axis 1. Borgi 2024 is the load-bearing cross-validation: GO and wavefront propagation *"become nearly identical"* in the weak-beam limit where dislocations are imaged.

---

## What the previous session got wrong

1. **"GO renders dislocations as sparse dots."** — Wrong. GO is a depth-integrated projection producing full-field contrast; code-verified `NN3=17` depth rays/pixel; empirically 100%-lit fields on-Bragg. Near-point features are the physically-correct deep-weak-beam regime (Borgi 2024 ~8–14 nm).
2. **"`darkmod` is a fundamentally different full 3D ray-trace (Astra GPU) wave-optics model."** — Wrong. Same kinematic GO family, same Poulsen-2017 resolution function; ASTRA is a CT line-integrator, not a wave/diffraction engine (arXiv §4 line 490, §4.2 line 584; `darkmod` `resolution.py`/`projector.py`/`crystal.py:590`). The real wave model is the separate Carlsen 2022.
3. **"The gap is architectural; a faithful Fig-3 repro is impossible with `dfxm_geo`."** — Wrong. The dots are reproducibly the φ=0.46 mrad ~1.78× off-acceptance setpoint + narrow rocking LUT; full-field butterfly recovers on-Bragg / at φ inside acceptance. The only genuine architectural contributor (thin-slab depth) is secondary and fixable.
4. **Taxonomy framing "GO vs ray-trace vs wave optics."** — Wrong axis. Two orthogonal axes; `darkmod`'s "ray-trace" is geometric CT projection on Axis 2, not wave physics on Axis 1.
5. **Likely figure misreading that seeded the error.** — Fig 3 **panel B** is a full-field 272×272 image; the dots are **panel C**, the *angular* scan-grid scatter for one pixel u* (*"point-wise supported … discrete and non-uniform grid in Δθ, φ, χ"*). The dots are the scan grid, not the spatial image.

---

## Phase 1 — Correctness fixes (identification rl-units + oblique wiring)
**Goal:** Eliminate the two code-verified identification-path defects: the rl metres-vs-µm bug and missing oblique kernel dispatch.

- [ ] Fix rl-units in `pipeline.py`: every `Fd_find_mixed` call in `_build_single_scan_specs` (~`:1460`), `_run_identification_single` (~`:1650`, `:1658`), `_run_identification_multi` (~`:1867`, `:1877`, `:1892`) passes `rl_eff` (= `fm.rl` / `fm.Z_shift(z)`, both **metres**), but `Fd_find_mixed` expects **micrometres** (docstring + `dislocations.py:287` comment). Forward path already does `find_hg_population(rl_eff * 1e6, …)` (`forward_model.py:1139`) and `Fd_find(rl * 1e6, …)` (`strain_cache.py:61`). Apply `* 1e6` at each identification site (or centralize in `rl_eff`/`Z_shift`). Effect: corrects |Hg| inflated ~1e6×.
- [ ] Add a regression test: identification single+multi |Hg|/1-over-r field must match the forward-mode population field (the correct oracle) for the same geometry — catches the 1e6× factor.
- [ ] Wire `[geometry]`/oblique into the identification CLI + kernel dispatch (the Phase-A "Task 16.5" ~50 LOC gap); rebuild θ-dependent globals via `reflection_theta_if_oblique`.
- [ ] `python -m pytest -q` (venv python); subtract the 17 pre-existing failures; no NEW failures; `mypy src/dfxm_geo/` = 0.

**Rationale:** The rl-units bug is independently verified this session and silently corrupts every identification strain field ~1e6× — collapsing weak-beam contrast to the core, the exact artifact that fed the "dots" misdiagnosis. Must precede any Fig-3 repro or ML-data run through identification. Oblique wiring is a prerequisite for using the Fig-3 geometry.

**Exit criteria:** All identification `Fd_find_mixed` sites pass µm-scaled rl; parity test catches the 1e6× factor; identification CLI dispatches the oblique kernel; no new test failures vs the 17-baseline; mypy clean.

---

## Phase 2 — Document the corrected model taxonomy (make it non-recurrable)
**Goal:** Put the corrected taxonomy + the `darkmod`-is-GO fact into auto-loaded memory and in-repo docs.

- [ ] Rewrite auto-memory `fig3_repro_darkmod_is_the_paper_model.md`: replace "darkmod IS a full 3D ray-trace (Astra GPU) … fundamentally unlike dfxm_geo" and "faithful Fig-3 repro not achievable / dots = model gap" with the corrected, cited statement (arXiv §4 line 490, §4.2 line 584, §7 line 1263; `darkmod` `resolution.py`/`projector.py`/`crystal.py:590`). **[DONE 2026-05-29]**
- [ ] Add `docs/model-taxonomy.md`: the two-axis taxonomy with `dfxm_geo`/`darkmod`/Carlsen 2022 placed; Borgi 2024 weak-beam GO↔wavefront convergence as cross-validation; the Fig-3 panel-B (image) vs panel-C (angular scan-grid dots) distinction.
- [ ] Update `CLAUDE.md` + `MEMORY.md` index to point at the corrected note and flag the superseded "dots/architectural" framing as **retracted** with a one-line correction. **[MEMORY.md DONE 2026-05-29]**
- [ ] Add a contrast-formation docstring at the `dfxm_geo` forward site (depth-integrated projection; near-point = deep-weak-beam regime; paper citations).

**Rationale:** The prior conclusions live in auto-memory and re-bias every future session. Correcting them with primary-source citations is the cheapest high-leverage anti-recurrence step and clarifies the v2.3.0 oblique arc's true role.

**Exit criteria:** The memory note no longer asserts darkmod=wave-optics/repro-impossible; `docs/model-taxonomy.md` exists and is referenced; MEMORY.md flags the retraction; a fresh session would answer "darkmod = same kinematic GO" from memory.

---

## Phase 3 — Reproduce Fig-3 full-field contrast with `dfxm_geo`
**Goal:** Show `dfxm_geo` rendering full-field extended contrast at the Fig-3 geometry; decide config-fix vs wider rocking LUT vs both.

- [ ] Read the paper's exact Fig-3 parameters (Table 1 / §6.2: energy, hkl, θ/η, per-reflection Δθ/φ/χ ranges + steps); reconcile against `al_oblique_figure3.toml`; correct the scrambled φ/χ/Δθ axes and zeta_v/NA flagged in the Fig-3 memory note.
- [ ] Determine the needed bootstrap `qi1_range` (field spreads `qi1` to ±7.8e-4 vs current ±2.5e-4); bootstrap a wider-`qi1` kernel; re-run the wide-`qi1` forward test that previously crashed and capture clean output.
- [ ] Forward-run at the corrected Fig-3B setpoint + on-Bragg + a φ inside acceptance; quantify field-lit fraction and bbox fill; confirm butterfly recovery.
- [ ] Isolate depth: hold `qi1_range` fixed, vary `NN3`/beam-thickness to measure depth's share of the φ=0.46 mrad butterfly extent — this gates Phase 4.

**Rationale:** The dots cause (config + narrow LUT) is established but its decomposition vs thin-slab depth was not isolated (wide-`qi1` run crashed; `darkmod` not renderable locally — ASTRA needs CUDA). Pinning this decides "improve `dfxm_geo`" vs "adopt `darkmod`."

**Exit criteria:** Full-field extended single-dislocation image at the corrected Fig-3 geometry (PNG + lit-fraction metric); required `qi1_range` determined + kernel bootstrapped; rocking-LUT-vs-depth weight quantified → clear yes/no on Phase 4.

---

## Phase 4 (conditional) — Depth/projection completeness for Fig-3-faithful images
**Goal:** If Phase 3 shows thin-slab depth materially limits faithfulness, extend the spatial forward model toward `darkmod`'s full-3D projection, keeping the shared kinematic GO physics.

- [ ] Choose the minimal sufficient upgrade: (a) raise `NN3`/extend `zl` to the true ~25 µm column (cheap, within the ray-grid) — try first; (b) true 3D voxel-volume + tomographic line-integral (the `darkmod`/ASTRA pattern) — only if (a) is insufficient.
- [ ] Complete oblique real-space geometry: currently η only in the reciprocal resolution function (`resolution.py`/`_build_M`) while `Theta = R_y(θ)` (`forward_model.py:163-166`) and `xl_start` keeps `/3`. Tilt the real-space ray grid by η; correct `xl_start` to `1/sin(2θ)` (=1.95 @ 15.42°, not `tan(2θ)/3`). Validate η=0 reduces exactly to simplified mode.
- [ ] Address the paraxial `ang2 = (2dθ/2)/tan(θ0)` at oblique θ vs the removed-small-angle formulation in arXiv:2503.22022.
- [ ] Parity-check upgraded `dfxm_geo` vs `darkmod` on a matched Fig-3 config (GPU box, or published Fig-3B + Borgi-2024 depth-streak scaling as reference); target ~4% oblique parity.

**Rationale:** The v2.3.0 oblique arc (removing small-angle approximations) is the *same direction* arXiv:2503.22022 took and is necessary; Phase 3 decides whether it is *sufficient* or whether a depth/projection operator is also needed. The real-space oblique incompleteness (η reciprocal-only; `xl_start` sin→tan double-counting) is code-verified and must be fixed regardless.

**Exit criteria:** Oblique real-space geometry complete (η-tilted grid + corrected `xl_start`; η=0 exact reduction); chosen depth upgrade implemented; Fig-3 image matches `darkmod`/published reference within a stated tolerance; smoke + mypy clean.

---

## Phase 5 — ML-training-data generation on the corrected model
**Goal:** Resume the 100k+ DFXM-image ML goal on a physics-, units-, and geometry-verified forward model.

- [ ] Re-validate the forward-throughput arc (Find_Hg fusion ~14.7×; float32 detector + `fanout.py` + `write_strain_provenance`) against the corrected identification path + oblique geometry.
- [ ] Define the data parameter space (populations/structures, hkl, setpoints) with setpoints **inside** the kernel rocking acceptance (Phase-3 lesson) — or deliberately span the weak-beam regime with correctly-sized `qi1_range` kernels.
- [ ] Cluster fan-out (config-level, ~8 configs × 16 threads/node; `write_strain_provenance=false` → 106 MB→32 KB/config, 0.51 TB→~3 GB for 100k).
- [ ] Spot-check generated images for full-field contrast + correct |Hg| (the Phase-1 rl-units guard as a per-batch check).

**Rationale:** Generating 100k images on a model with a 1e6× units bug or off-acceptance setpoints yields a corrupted/dotty dataset. With Phases 1–4 closed, the already-built throughput arc runs on a trustworthy model.

**Exit criteria:** A validated batch (e.g. 1k images) shows full-field contrast + correct magnitudes; cluster fan-out reproduces ~3 node-hr / ~3 GB-per-100k; rl-units guard passes per batch; dataset spec documented.

---

## Open decisions
- **Build vs adopt for Fig-3 faithfulness:** improve `dfxm_geo` (extend depth / add 3D-voxel projection + finish oblique real-space geometry) vs adopt/interface `darkmod` (already full-3D, but GPU-gated)? Phase 3 informs; the call is yours.
- **φ=0.46 mrad:** config/units error to correct, or the paper's deliberate deep-weak-beam setpoint demanding a wider-`qi1` kernel? Needs your read of the exact Table-1 scan range.
- **Release plan:** v2.3.0 oblique arc is implemented but UNPUSHED / ON HOLD (HEAD `18d3fd8`). Ship the Phase-1 rl-units fix as a hotfix *before* v2.3.0 (it's a correctness bug in already-released identification mode), or fold into v2.3.0? Does the oblique real-space fix (Phase 4) belong in v2.3.0 or a follow-on?
- **Depth-upgrade scope (if Phase 4 triggers):** cheap (raise `NN3`/extend `zl`) vs faithful (3D voxel + projection). Acceptable parity tolerance vs `darkmod` (~4%?).
- **Reconstruction scope:** is multi-reflection F-tensor reconstruction (the paused Phase-B brainstorm, `darkmod`'s inverse) in scope for `dfxm_geo`, or delegated to `darkmod` (keeping `dfxm_geo` forward-only)?
- **ML regime:** generate inside the rocking acceptance (clean full-field) only, or deliberately span the weak-beam/off-Bragg regime (closer to real data; needs correctly-sized `qi1_range` kernels)?
