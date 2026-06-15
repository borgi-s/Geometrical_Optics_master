# M4 Stage 4.4 — Validation (design)

**Date:** 2026-06-15
**Branch:** `feature/m4-stage44-validation` (off `abaf614`)
**Milestone:** M4 (CIF crystal structures), final stage before v3.0.0
**Roadmap:** `roadmap202606audit.md` §5, "Stage 4.4 — Validation (~1 week)"

## Goal

Formally close out the M4 crystal-structures milestone on current `main`
(`abaf614`). Stages 4.1–4.3b are merged; their tests already cover all four
M4 definition-of-done boxes. Stage 4.4 adds the **regression locks**, the
**per-structure figures**, and a **validation record**, and confirms the
suite is green on `abaf614`.

This is a **validation-only** stage. The v3.0.0 release (version bump, the
detector `counts_scale` decision, conda-forge recipe sync, tag) is
deliberately deferred to a separate, user-triggered step.

## Out of scope (explicitly deferred)

- v3.0.0 version bump / tag / PyPI / conda-forge sync.
- The detector `counts_scale` realism decision (still provisional `1.0e4`).
- The dislocation cross-correlation scorer identifiability re-validation run
  (tracked separately in the scorer kickoff; depends on the same screw-fixed
  physics but is a paper deliverable, not crystal-structure validation).
- Any new physics, new slip families, or anisotropic elasticity.

## Background: what already exists

The roadmap's Stage 4.4 list ("golden tests: Al/Fe-or-W/Ti-or-Mg + one CIF
round-trip per structure; one forward image per structure as a tutorial
figure") was written before 4.3a/4.3b landed. As of `abaf614`:

- **FCC/BCC byte-identity** gates exist (`tests/test_fcc_bit_identity.py`,
  `tests/test_cubic_bit_identity.py`) — but they are 2-run determinism
  comparisons, not stored-pixel baselines.
- **BCC e2e** (`tests/test_bcc_e2e.py`): forward + identify, explicit
  `structure_type` and Fe-`Im-3m`-CIF routes, ν gating, cell-derived |b|.
- **HCP e2e** (`tests/test_hcp_e2e.py`): Ti + Mg forward + identify, ⟨a⟩/⟨c+a⟩
  labels, Ti-`P6_3/mmc`-CIF route, c/a provenance.
- **Extinction / forbidden-reflection rejection**
  (`tests/test_extinction_rules.py`, `tests/test_reflections_extinct.py`,
  `tests/test_ceramic_acceptance_al2o3.py`).
- **CIF round-trip per structure** already satisfied: FCC Al
  (`tests/data/cif/al_fm3m.cif`), BCC Fe (inline CIF in `test_bcc_via_fe_cif`),
  HCP Ti (inline CIF in `test_hcp_via_ti_cif`).
- **Isotropic-elasticity limitation** documented (`docs/crystal-structures.md`).
- **Screw g·b = 0 extinction** across all structures
  (`tests/test_screw_gb_extinction.py`).

The **genuine gaps** Stage 4.4 fills:

1. No locked, stored-pixel golden image per structure (only same-machine
   determinism gates).
2. No per-structure forward figure committed in-repo for the tutorial / M5
   (one exists only in the paper folder).
3. No single document mapping the four M4 DoD boxes to the tests that prove
   them, and no recorded current-`main` gate numbers.

## Canonical recipe: one source of truth

The deterministic showcase recipe already exists in
`papers/2026-06-dfxm-geo-software-paper/scripts/make_showcase.py`: one
**pure-edge** dislocation (`angle_start_deg = angle_stop_deg = 0`, no slip-plane
sweep, no RNG draw), analytic resolution backend (kernel-free),
`[detector] model = "ideal"` (raw float32), imaged at a weak-beam phi offset
(`[scan.phi] value = 1.75e-4`), rendered via `dfxm-identify --mode single`.

| Structure | Material | Mode | Reflection | Mount | ν |
|---|---|---|---|---|---|
| FCC | Al | simplified | (-1, 1, -1) | default (none) | 0.334 |
| BCC | W | oblique | (2, 0, 0) | (100)/(010)/(001) | 0.28 |
| HCP | Ti | oblique | (1, 0, -1) | (2,-1,0)/(0,1,0)/(0,0,1) | 0.32 |

Because there is no random draw, the **raw detector image is deterministic**
and reproducible bit-for-bit on a machine (the same property the existing
`test_bcc_forward_deterministic` / `test_fcc_wall_forward_deterministic`
gates rely on), so it can be a stored golden.

These three recipes become the single source of truth shared by the golden
tests, the in-repo figure, and the paper figure. The recipe builders are
moved into an in-repo module; `make_showcase.py` is refactored to import them
so nothing can drift.

## Components

### 1. `scripts/render_structure_showcase.py` (recipe + renderer)

Holds the three recipe builders (`fcc_toml`, `bcc_toml(eta)`, `hcp_toml(eta)`)
and the η computation (`compute_omega_eta`), with a small CLI:

- `--figures` — full-resolution render (default 510 px, like the paper) →
  `docs/img/showcase_fcc_bcc_hcp.png` (+ `.pdf`) and per-panel PNGs
  `docs/img/showcase_{fcc,bcc,hcp}.png`. Run once, committed.
- `--golden` — small-grid render (the e2e-test grid size, scaled via
  `fm.Npixels`) writing raw `.npy` arrays to
  `tests/data/golden/structure_showcase/{fcc,bcc,hcp}.npy`. Used to
  (re)generate the golden baselines.

A `RECIPES` mapping (`tag -> builder`) and a `render_raw(tag, grid)` helper
let both the figure mode, the golden-generation mode, and the test import the
exact same recipe text. Rendering is **in-process** (via
`dfxm_geo.pipeline.{load_identification_config, run_identification}`), not a
subprocess, so the test stays fast and exe-independent. The full-res figure
mode may still shell out to match the paper exactly; the golden path must be
in-process.

`papers/.../make_showcase.py` is refactored to `from … import fcc_toml,
bcc_toml, hcp_toml` (or an equivalent thin shim) so the published figure and
the validation goldens are byte-for-byte the same recipe.

### 2. `tests/test_structure_goldens.py` (locked regression)

Parametrized over the three structures. For each:

- Render the small-grid raw float32 detector image in-process via the shared
  recipe.
- Assert it matches the stored golden
  `tests/data/golden/structure_showcase/{tag}.npy` via `np.allclose` with a
  tight tolerance (analytic backend is pure-numpy deterministic; the tolerance
  guards only against platform float noise, not physics drift —
  `rtol=1e-6, atol` scaled to the image dynamic range).
- Assert 2-run **bit identity** (`np.array_equal`) to prove same-machine
  determinism independent of the stored baseline.

Marked `@pytest.mark.slow`. The goldens are small (the e2e grid, not 510 px)
so they stay well under the 10 MB cleanup threshold and are safe to commit.

### 3. `docs/img/` figures + doc reference

The committed `showcase_fcc_bcc_hcp.png` plus per-panel PNGs. Referenced from
`docs/crystal-structures.md` (a short "Forward contrast across crystal
systems" figure block) and reserved for M5's future `06_other_crystals`
notebook. The figure is the same render that anchors the golden, at full
resolution.

### 4. `docs/m4-validation-report.md` (the validation record)

The artifact that closes the milestone. Contents:

- A DoD table: each of the four M4 definition-of-done boxes →
  the exact test(s) / doc that proves it (file:test name).
  1. `[crystal] cif=…` + BCC reflection → forward image + identify library
     with BCC slip labels → `test_bcc_e2e.py::test_bcc_via_fe_cif`,
     `::test_bcc_identify_has_bcc_slip_labels`.
  2. Cubic/FCC bit-identical to v2.4.0 → `test_fcc_bit_identity.py`,
     `test_cubic_bit_identity.py` (slip-order + wall + determinism gates).
  3. Forbidden reflections rejected with explanatory error →
     `test_extinction_rules.py`, `test_reflections_extinct.py`,
     `test_ceramic_acceptance_al2o3.py`.
  4. Isotropic-elasticity limitation documented →
     `docs/crystal-structures.md` (limitations section).
- A "golden regression" row pointing at `test_structure_goldens.py` and the
  committed figure.
- The recorded **gate numbers** on `abaf614`: full default suite, full slow
  suite, and `mypy` (counts + the known pre-existing slow failures named as
  non-regressions).
- A short "deferred to v3.0.0 release" checklist (counts_scale, version bump,
  conda-forge sync) so the release step has a ready punch list.

### 5. Baseline gate run (recorded, not a code artifact)

Before recording numbers, `rm direct_space/deformation_gradient_tensors/Fg_*.npy`
(the documented stale-`Fg`-cache hazard). Run with the venv python:

- `python -m pytest -q` (default suite).
- `python -m pytest -q -m slow` (slow suite).
- `mypy src/dfxm_geo/`.

Confirm the three known pre-existing slow failures (the two cumulative-RAM
`test_analytic_backend_integration` and the detector-OOM
`test_pipeline::…sample_remount_S2`, now mitigated by the `apply()` chunking
in `9d5705d` — re-verify) are understood as environmental / pre-existing, not
4.4 regressions. Record the resulting numbers in the report.

## Testing strategy

- The golden test is itself the new regression test; no test needs a test.
- New golden `.npy` baselines are generated once via
  `render_structure_showcase.py --golden`, eyeballed against the committed
  figure for sanity (non-empty, has contrast), then committed.
- Determinism is asserted in-test (2-run `array_equal`) so a future
  non-deterministic regression is caught even on a machine where the stored
  golden's tolerance would otherwise mask it.
- FCC byte-identity to v2.x is *not* re-litigated here — it is already locked
  by `test_fcc_bit_identity.py` / `test_cubic_bit_identity.py`; the report
  cites them.

## Risks / decisions

- **Golden tolerance vs. drift.** Too loose hides physics changes; too tight
  flakes across platforms. Mitigation: pure-numpy analytic backend is
  effectively deterministic, so `rtol=1e-6` is both safe and strict; the
  same-machine `array_equal` assertion is the real determinism guard.
- **Refactoring the paper's `make_showcase.py`.** Touching the paper folder
  is acceptable (approved); the refactor is import-only, the rendered figure
  is unchanged. If the import path proves awkward (the paper script lives
  outside the package), fall back to a single shared recipe module under
  `scripts/` that both import.
- **Grid-size knob.** The golden uses the same small-grid mechanism as the
  existing e2e tests (`fm.Npixels`), not a new knob.

## Definition of done (Stage 4.4)

- [ ] `scripts/render_structure_showcase.py` renders all three structures in
      both `--figures` and `--golden` modes; `make_showcase.py` shares its
      recipe.
- [ ] `tests/test_structure_goldens.py` passes (golden allclose + 2-run
      determinism) for FCC Al / BCC W / HCP Ti.
- [ ] `docs/img/showcase_fcc_bcc_hcp.png` (+ per-panel) committed and
      referenced from `docs/crystal-structures.md`.
- [ ] `docs/m4-validation-report.md` maps all four DoD boxes to tests and
      records the green-gate numbers on `abaf614`.
- [ ] Full default + slow suites + mypy recorded green (pre-existing failures
      named as non-regressions); `Fg_*.npy` caches cleaned at wrap-up.
- [ ] Branch ready for Sina's merge call (no push, no tag).
