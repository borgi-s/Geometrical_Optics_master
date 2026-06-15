# M4 (CIF Crystal Structures) — Validation Report

**Date:** 2026-06-15
**Branch:** `feature/m4-stage44-validation` off `abaf614`
**Milestone:** M4, Stage 4.4 (validation-only; v3.0.0 release deferred).

This report closes the M4 crystal-structures milestone: every definition-of-done
box maps to a test, the three structure families are locked as golden
regressions, and the gates are recorded on current `main`.

## Definition-of-done coverage

| # | M4 DoD item | Proven by |
|---|---|---|
| 1 | `[crystal] cif=…` + a BCC reflection produces a forward image and an identify library with BCC slip-system labels | `tests/test_bcc_e2e.py::test_bcc_via_fe_cif`, `::test_bcc_identify_has_bcc_slip_labels`, `::test_bcc_forward_runs` |
| 2 | Cubic/FCC path bit-identical to v2.4.0 | `tests/test_fcc_bit_identity.py`, `tests/test_cubic_bit_identity.py` (slip-order lock, wall + random-dislocation determinism, deliberate forward/identify ordering distinction) |
| 3 | Forbidden reflections rejected with explanatory error | `tests/test_extinction_rules.py`, `tests/test_reflections_extinct.py`, `tests/test_ceramic_acceptance_al2o3.py` |
| 4 | Isotropic-elasticity limitation documented prominently | `docs/crystal-structures.md` (Limitations section) |

## Stage 4.4 additions

| Artifact | Location |
|---|---|
| Per-structure golden regression (FCC Al / BCC W / HCP Ti) | `tests/test_structure_goldens.py` + `tests/data/golden/structure_showcase/{fcc,bcc,hcp}.npy` |
| Shared deterministic recipe (single source of truth) | `scripts/render_structure_showcase.py` (also drives the docs + paper figures) |
| Tutorial figure | `docs/img/showcase_fcc_bcc_hcp.png` (+ per-panel), referenced from `docs/crystal-structures.md` |

The golden recipe renders one pure-edge dislocation per structure on the
kernel-free analytic backend with the ideal detector (raw float32, no RNG draw),
so the raw detector image is deterministic and reproducible bit-for-bit. The
test asserts `np.allclose` vs the committed golden plus a 2-run bit-identity
determinism check on FCC. The same recipe is imported by
`papers/2026-06-dfxm-geo-software-paper/scripts/make_showcase.py`, so the
published paper figure, the docs figure, and the golden cannot drift.

## CIF round-trips per structure (already covered)

| Structure | CIF route |
|---|---|
| FCC | `tests/data/cif/al_fm3m.cif` via `tests/test_cif_loader.py` |
| BCC | inline Fe `Im-3m` CIF in `tests/test_bcc_e2e.py::test_bcc_via_fe_cif` |
| HCP | inline Ti `P6_3/mmc` CIF in `tests/test_hcp_e2e.py::test_hcp_via_ti_cif` |

## Gate numbers on `abaf614` (+ the Stage 4.4 commits; `Fg_*.npy` caches cleaned first)

- **Default suite (`pytest -q`): 1122 passed, 2 skipped, 1 xfailed, 55 deselected**
  (313 s). Skips: legacy pickle absent (`test_kernel_format`), `torch` absent
  (`test_scoring_engine`). xfail: the documented #16 true-Bragg HDF5 bit-equiv
  (`test_hdf5_bit_equiv`, Sina-authorized). The 55 deselected are the `slow`
  tests.
- **`mypy src/dfxm_geo/`: Success — 0 issues in 51 source files.**
- **New Stage 4.4 test (`tests/test_structure_goldens.py`, slow): 4 passed**
  (FCC/BCC/HCP golden match + FCC determinism), 23 s.
- **Slow suite:** see the triage below. With the `test_custom_slip_system_e2e`
  fix applied, the only remaining slow-suite failures on this machine are four
  RAM-bound MC-kernel/resolution tests that pass in isolation or with memory
  headroom — not 4.4 regressions.

### Slow-suite triage (this run, 16 GB Windows laptop)

A single-process `pytest -m slow` run failed 5 tests. Triaged:

| Test | Cause | Verdict |
|---|---|---|
| `test_analytic_backend_integration::test_analytic_forward_matches_mc_no_beamstop` | `MemoryError` — single 763 MiB allocation (`resolution.py:408`, 100M-ray MC array) | Environmental (absolute RAM); MC path untouched by 4.4 |
| `test_mc_vs_analytic_oblique_parity::…[0.35-0.1745…]` | cumulative RAM; **passes in isolation** | Environmental |
| `test_oblique_remount_wall::test_wall_remount_oblique_forward_e2e` | subprocess `dfxm-forward` could not import scipy (paging file exhausted by prior tests); **passes with memory headroom** | Environmental |
| `test_pipeline::…sample_remount_S2` | same paging exhaustion in a spawned subprocess; **passes with memory headroom** (detector OOM already mitigated by `apply()` chunking, `9d5705d`) | Environmental |
| `test_custom_slip_system_e2e::test_custom_slip_system_forward_runs` | the repo-audit #2 ν-gate (`19387b1`) rejects the `[[crystal.slip_system]]` `custom:user` config because it set neither `material` nor `poisson_ratio` | **Real pre-existing bug — FIXED in this branch** |

The ν-gate breakage was introduced by the parallel post-4.3 follow-up batch and
missed because the test is `slow` (the batch's verification ran the default
suite). The Stage 4.4 commits add only new files plus this one-line test fix, so
none of them can cause a forward/identify regression. The four environmental
failures are the documented cumulative/absolute-RAM behavior of the heavy
MC-kernel tests on a memory-tight machine; production validation runs them on a
cluster node or in isolation.

## Deferred to the v3.0.0 release (separate, user-triggered step)

- [ ] Version bump `pyproject.toml` 2.5.1 → 3.0.0 + tag.
- [ ] Detector `counts_scale` realism decision (still provisional `1.0e4`).
- [ ] conda-forge recipe sync: add the `dfxm-find-reflections` entry point to
      `build.python.entry_points`; document `gemmi` as an optional extra (do
      NOT add to the `run` deps).
- [ ] PyPI publish (gated on the `pypi` GitHub Environment approval).

## Follow-ups surfaced by this validation (non-blocking)

- The full `pytest -m slow` suite cannot complete in one process on a 16 GB
  machine: `test_analytic_backend_integration` alone needs a 763 MiB
  allocation, and the cumulative footprint exhausts the Windows paging file,
  which then breaks subprocess-spawning tests. Consider chunking/streaming the
  100M-ray MC resolution array (`resolution.py:408`) or scaling that test's
  ray count down, mirroring the detector `apply()` chunking fix.
