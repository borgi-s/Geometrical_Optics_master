# GNB / oblique forward — implement the FULL-OMEGA path (handoff)

## UPDATE 2026-06-21 (later) — FULL-OMEGA IMPLEMENTED + VALIDATED (uncommitted)

Done on branch `fix/gnb-placement-bugs` (worktree `wt-gnb-walls`), **NOT committed**.

- **Fix (one guarded seam):** `build_geometry_context` now counter-rotates the ray
  grid `rl -> R_z(omega).T @ rl` when `omega != 0` (z preserved -> `prob_z`
  byte-identical). The q-path keeps `R_z(omega) @ Us` in
  `precompute_forward_static`, so omega is applied once on each side. Adding the
  rl rotation flips the old B' into full-omega. Guarded so `omega == 0.0` stays
  bit-identical.
- **Convention settled EMPIRICALLY (oracle, not argument):** rotate `rl`, KEEP the
  offset (it lives in the omega0 lab frame); the field-singular `|Hg|` ridge then
  lands in the detector at `R_z(omega) @ offset` to grid resolution. So the
  overlay/core formula is `det = R_z(omega) @ (offset + t*xi)`. (The earlier
  "counter-rotate offsets / 1.57um mismatch" worry was wrong.)
- **Handoff diagnosis corrected on two points:** (1) the dominant defect for the
  COM-map builders was that the SINGLE-reflection oblique path *drops* omega
  (orchestrator ~387 calls `build_forward_context` with no omega, so omega=0 even
  though `config.geometry.omega` is resolved) — that is a deliberate M2/M3
  convention, LEFT AS-IS (changing it churns BCC/HCP/M2/gnb oblique goldens).
  (2) "off-Bragg at omega=0 -> dark" is FALSE for this GO model: a perfect crystal
  is bright at every omega (resolution centered at qi=0); omega only orients the
  LOCAL strain projection. The real test is whether the dislocation contrast lands
  where the rotated wall predicts (the overlay), which it now does.
- **Tests:** new `tests/test_full_omega_geometry.py` (rl rotation; z/prob_z
  preserved; omega=0 plain-mgrid guard; e2e core-at-`R_z(omega)*offset`).
  `tests/test_bprime_projection.py` docstrings updated (q-path half of full-omega).
  **Gate: 1107 passed / 70 skipped / 0 failed; mypy 0/52.** omega=0 bit-identical;
  single-reflection oblique goldens (gnb/bcc/hcp) unchanged; multi-reflection
  `[[reflections]]` path upgraded B'->full-omega (its physics tests still pass).
- **Deliverables (in `_gnb_render/`):** `_verifyomega_offset.py` oracle (|Hg| ridge
  on `R_z(omega)*offset`); `_gnb_grid_standard.py` -> `gnb_leds_eq11_fullomega_4x3.png`
  + `gnb_leds_eq14_fullomega_4x3.png` (standard centred 4x3 COM grids, overlays);
  `_gnb_fullomega_grid.py` (3x4 hi-DPI, 4x-pixel FOV, in-plane shift demo);
  `_gnb_growth_movie.py` (growth movies eq11+eq14, schedule 1..40 by1 / 43..70 by3 /
  80..400 by10, round-robin centre-outward; -> `gnb_leds_eq{11,14}_growth_fullomega.mp4`).
- **SINGLE-REFLECTION OMEGA FIXED (2026-06-21):** the single-reflection branch of
  `_run_simulation_inner` (orchestrator ~387) now threads `omega=config.geometry.omega`,
  so oblique single-reflection runs render full-omega at the resolved omega
  (simplified mode resolves omega=0 -> byte-identical). Test:
  `tests/test_single_reflection_omega.py` (spies the threaded omega). Only the two
  gnb byte-goldens changed (`tests/data/golden/gnb/{leds_eq11,frankus}_oblique.npy`)
  -> REGENERATED to the eta/omega-consistent full-omega render via
  `tests/test_gnb_e2e._regen_goldens()` (the old goldens paired the omega=124 eta
  with geometry omega=0 -- inconsistent). BCC/HCP/oblique-provenance/identify tests
  are smoke -> unaffected. **Gate after fix: 1108 passed / 70 skipped / 0 failed,
  mypy 0/52.**
- **STILL OPEN:** (b) full-omega +
  z-scan: `Z_shift` path does NOT counter-rotate rl (only `ctx.geometry.rl` does)
  — handle if scanning z at oblique reflections; (c) commit/merge decision (the
  fix needs a physics review — Sina is the author); (d) multi-reflection byte
  goldens, if any are added later, will reflect full-omega not B'.

---

Date 2026-06-21. Handoff for a NEW session.
Worktree `C:\Users\borgi\Documents\GM-reworked\wt-gnb-walls`, branch
`fix/gnb-placement-bugs` (the GNB placement-bug fix is committed `6a9b0c0` and
`--no-ff`-merged to LOCAL `main` = `d0f9251`, UNPUSHED, no tag; branch + worktree
retained). venv `.venv-wt\Scripts\python.exe` (bash `python` is Py2.7 — don't use
it). Run from the worktree root. Windows console is cp1252 — ASCII-only prints.

## TL;DR of the task
Implement the **full-omega forward path** so a single crystal mounting can be
imaged at MULTIPLE reflections correctly. Today the goniometer rotation `omega`
is applied only via the **B' approximation** (rotate the diffraction vector,
keep the probed volume fixed), which is crude at the large `omega` (56-100 deg)
that off-scattering-plane reflections need. The exact treatment — rotate the
probed volume (ray grid `rl`) AND the wall placement by `omega` per reflection —
is an explicitly-deferred "upgrade path" and is what this task delivers.

## How we got here (context)
This came out of a long visualization session imaging an eq14 GNB wall (a (111)
boundary, 3 dislocation sets) as 4x3 COM-map grids and growth animations over the
reflections `[(-1,1,-1), (2,2,0), (2,0,0), (1,1,1)]`. Sina (DFXM expert) kept
sensing "the geometry is wrong." An Opus subagent + direct verification pinned it.

## The geometry finding (verified this session)
The forward model's `omega` seam is in `precompute_forward_static`
(`src/dfxm_geo/direct_space/forward_model.py:843-851`):

```
base_qc = (R_z(omega) @ Us) @ Hg @ q_hkl     # B' approximation
```

i.e. omega rotates ONLY the diffraction vector about lab z; the ray grid `rl`,
the beam profile `prob_z`, and the strain field `Hg` stay shared (same probed
volume). The docstring (`:834-840`) calls this the B' approximation and says
**"Full-ω (rotating rl itself) is the documented upgrade path, isolated behind
this one seam."** That upgrade is this task.

For multi-reflection imaging of ONE mounting, each reflection needs its
goniometer `omega` (from `compute_omega_eta`, `crystal/oblique.py:345`). For the
(111) GNB at the identity FCC Al mount, 17 keV (verified via `_verify_omega.py`):

| reflection | `Us·q` (lab)            | y-comp | omega needed | status |
|---|---|---|---|---|
| (-1,1,-1)  | [0, 0, 1]              | 0.000  | 124.0 (FREE) | EXACT at any omega |
| (2,2,0)    | [0.5, -0.866, 0]      | -0.866 | 59.8         | needs omega; B' crude |
| (2,0,0)    | [0.707,-0.408,-0.577] | -0.408 | 100.4        | needs omega; B' crude |
| (1,1,1)    | [0,-0.943,-0.333]     | -0.943 | 56.0         | needs omega; B' crude |

KEY POINTS:
- **(-1,1,-1) is exact regardless of omega** because `Us·q ∥ lab z` = the omega
  rotation axis, so omega is a FREE azimuth (it just rotates the image). This is
  why the shipped single-reflection gnb (omega=0) and the (-1,1,-1) golden are
  VALID. Verified: disloc-image mean 0.099 at omega=0 vs 0.682 at omega=124 — a
  pure image rotation, both at Bragg.
- **The other three reflections have `Us·q` out of the scattering plane**, so
  omega is FIXED (56-100 deg). At omega=0 they are off-Bragg for the dislocation
  contrast. **The multi-reflection COM grids/animations from this session
  rendered those 3 rows at omega=0 → physically wrong.** (The PERFECT crystal
  still looked bright because the analytic backend's `eta` centers the acceptance
  window regardless of omega — that is what masked the error. The DISLOCATION
  contrast is what's wrong.)
- The multi-reflection `[[reflections]]` path DOES thread omega
  (`orchestrator.py:276` `_context_for_run` -> `build_forward_context(omega=run.omega)`
  -> the B' seam), so it is "less wrong" than the omega=0 diagnostics — but it is
  still only B', which is a poor approximation at omega = 56-100 deg.

Net: accurate multi-reflection imaging of THIS wall at THESE reflections is not
supported today. Full-omega fixes it.

## The implementation task: full-omega
Physically, the goniometer rotates the SAMPLE (crystal + its dislocations) by
`omega` about lab z to bring each reflection to Bragg. Everything fixed in the
sample frame (dislocation positions, Hg field, crystal axes) rotates with it; the
lab beam + detector are fixed. So a correct render must rotate **the probed
volume and the wall placement by omega**, consistently with the q-path — not just
the q-vector.

What must change together (derive carefully, then implement behind the existing
B' seam, guarded so `omega == 0.0` stays bit-identical to v2.5.1):
1. **The ray grid / probed volume.** The lab ray grid `rl` (built in
   `build_geometry_context`, `forward_model.py:727-771`) must be expressed in the
   omega-rotated sample frame (e.g. sample-frame `rl_s = R_z(omega).T @ rl_lab`),
   OR equivalently rotate the sample into lab. Today `rl` ignores omega.
2. **The wall placement / dislocation offsets.** `build_dislocation_population`
   (gnb branch, `forward_model.py:~1436`) places cores with
   `crystal_to_lab = Theta.T @ Us` and NO omega. Under full-omega the offsets are
   in the rotating sample, so they must pick up the SAME omega rotation as `rl`
   (currently consistent with B' precisely because neither rotates). The
   orchestrator already has `ctx.geometry.omega` available to thread in alongside
   the existing `theta=ctx.geometry.Theta` kwarg (`orchestrator.py:~395`). Decide
   whether to rotate positions or `rl`; they must be consistent so
   `rd = Ud.T·Us.T·S.T·Theta·(rl - offset)` is evaluated in one frame.
3. **The q-path.** Replace/augment the B' `R_z(omega) @ Us` in
   `precompute_forward_static` so the full rotation is applied once and only once
   (don't double-count omega between the rl path and the q path). Work out the
   exact algebra: B' currently rotates `base_qc`; full-omega rotates `rl` (hence
   `Hg` is sampled in the rotated volume) — confirm what the q-path then needs.
4. **Find_Hg.** `Find_Hg_from_population` (`forward_model.py:1656`) subtracts
   `position_lab_um` from `rl` in the lab frame; ensure the offset and `rl` are in
   the same (rotated) frame.

Recommended approach: a Phase-0 spike (no rendering) that derives the exact
full-omega chain symbolically/numerically and proves: (a) for (-1,1,-1) the
result equals the current omega=0 render up to a pure image rotation, and (b) for
(2,2,0) the perfect crystal stays at-Bragg AND the dislocation cores' slab
intersection sits where the rotated wall geometry predicts. Then TDD the
implementation.

### Validation gates for the implementation
- `omega == 0.0` path BIT-IDENTICAL to current (guard it; the whole suite + the
  gnb goldens must stay green: default ~1103 passed, mypy 0/52).
- Perfect crystal at-Bragg (bright) at each reflection's true omega.
- g·b = 0 extinction preserved (it is omega-independent — a good invariant check).
- A new e2e: render the SAME centered dislocation at two reflections that share a
  reflection family but need different omega; confirm the contrast transforms as
  the rotated geometry predicts (not as B').
- Cross-check against `compute_omega_eta` and the M2 spec.

### Read these first
- B' decision + oblique geometry rationale:
  `docs/superpowers/specs/2026-06-10-m3-multi-reflection-sweeps-design.md`,
  `docs/superpowers/specs/2026-06-11-m2-pixel-eta-decision.md`.
- `forward_model.py`: `build_geometry_context` (727), `precompute_forward_static`
  (819, the B' seam), `forward`/`forward_from_static` (1039/924),
  `Find_Hg_from_population` (1656), gnb branch of `build_dislocation_population`
  (~1406-1436).
- `crystal/dislocations.py:290-300` (the `rd = Ud.T Us.T S.T Theta (rl-offset)`
  chain; `Hg = transpose(Fg^-1) - I`, NONLINEAR in dislocations — no incremental
  shortcut).
- `crystal/oblique.py:345` `compute_omega_eta` (omega/eta from paper Appendix A;
  rotation axis = lab z).
- `crystal/reflections.py` `resolve_reflections` ([[reflections]] schema: each
  entry needs `hkl`; omega/eta/theta auto-computed; raises if Laue unsatisfiable).
- `orchestrator.py:261-276` `_context_for_run`/`_resolution_for_run` (threads
  `run.omega`), and `_run_simulation_inner` (~330-540) where the population + ctx
  are built per reflection.

## Reproducers (untracked scratch in the worktree, written this session)
- `_verify_omega.py` — omega/eta per reflection + `Us·q` y-component (the table above).
- `_verify_multirefl.py` — renders all 4 reflections via `[[reflections]]`
  (omega threaded, B') vs single-reflection omega=0; prints perfect + disloc
  mean/std. Shows omega changes the disloc image; perfect stays bright.
- `_diag_plane.py` — coplanarity of cores (1e-15) + wall-plane ∩ slab is a single
  line (the cores ARE collinear; that part is correct).
- `_diag_overlay.py`, `_gnb_anim_overlay.py` — overlay dislocation slab-crossings
  on COM maps (markers on one line; the X is the field). NOTE these loop over the
  4 reflections and so inherit the omega=0 invalidity for 3 of them.
- COM/animation builders: `_gnb_com_maps.py`, `_gnb_com_figure.py`,
  `_gnb_com_special.py`, `_gnb_anim.py`, `_gnb_anim_encode.py`. Outputs in
  `C:\Users\borgi\Documents\GM-reworked\_gnb_render`. MP4 needs `imageio` +
  `imageio-ffmpeg` (pip-installed into `.venv-wt` this session).

## Status of this session's deliverables
- VALID: anything at `(-1,1,-1)` only — the 3D interactive HTML walls (eq11,
  eq14), the eq11/frankus oblique goldens, single-reflection renders.
- INVALID (omega dropped, off-Bragg for the disloc contrast): the multi-reflection
  4x3 COM grids and the growth animations, for the `(2,2,0)/(2,0,0)/(1,1,1)`
  rows. The `(-1,1,-1)` row in each is fine. REDO after full-omega lands.

## Open questions for Sina (physics review needed)
- Confirm the full-omega convention: rotate `rl` (probed volume) vs rotate the
  sample positions — and the exact q-path so omega is applied once. This is a
  forward-model physics change; have Sina review the derivation before merge.
- Decide whether to keep B' as a fast/approximate mode (small-omega) alongside
  full-omega, or replace it.

## Not in scope / leave alone
- The placement bug fix (interleave + `Theta.T @ Us`) is correct and merged — do
  NOT revert it. The Opus subagent proved `Theta.T @ Us` is the unique
  self-consistent placement (placed line == rendered field line to 1e-16).
- Cores ARE coplanar and DO intersect the slab as a single line — that is correct
  physics, not a bug. The apparent "X" is the long-range strain field.
