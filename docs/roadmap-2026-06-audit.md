# Project Audit & Roadmap — June 2026

**Date:** 2026-06-09
**Baseline:** v2.4.0 (commit 4297017)
**Scope:** Full-project audit plus a phased roadmap to five goals:
oblique full compatibility, identify fan-out profiling/optimization,
multi-reflection sweeps, `.cif`-driven crystal structures, and a final
tutorial/walkthrough.

---

## 1. Audit summary

### Where the project stands

| Area | State | Evidence |
|---|---|---|
| Code quality | Strong — 0 TODO/FIXME markers in `src/`, ruff + mypy + pre-commit, legacy quarantined | `pyproject.toml`, `.pre-commit-config.yaml`, `legacy/README.md` |
| Tests | 97 files / ~650 tests, golden-data regression, bit-identity gates, CI on py3.11+3.12 | `tests/`, `.github/workflows/ci.yml` |
| Oblique (goal 1) | ~85 % complete (Phase A shipped in v2.3.0); known, documented gaps | §2 below |
| Identify fan-out (goal 2) | Works at cluster scale but subprocess-per-config; several clear hot spots, no identify-specific profile yet | §3 below |
| Multi-reflection (goal 3) | Designed (Phase B of the 2026-05-28 spec) but not implemented; Burgers/slip tables block non-{111} | §4 below |
| Crystal structures (goal 4) | Hard FCC/cubic: `CrystalMount` rejects non-cubic, slip systems hardcoded, no CIF dependency | §5 below |
| Docs/tutorial (goal 5) | ML tutorial notebook is excellent; no docs site, no oblique/multi-reflection examples, citations only in README | §6 below |

### Cross-cutting health issues (not goal-specific)

1. `pipeline.py` is 2 093 lines mixing config dataclasses, two orchestrators,
   and CLI parsing. Every goal below touches it. **Split it
   (`config.py` / `orchestrator.py` / `cli.py`) before Milestones 3–4 land**,
   or the merge cost compounds.
2. `forward_model.py` module-level globals (`psize`, `zl_rms`, ray grid `rl`,
   `theta_0`) are ID06-hardcoded and mutated via the
   `reflection_theta_if_oblique()` context manager. This global-state pattern
   is the single biggest structural risk for goals 1, 3 and 4 — each of them
   needs per-run geometry. Plan: migrate to an explicit `GeometryState`
   object passed through `forward()` (v3.0.0, see Milestone 4).
3. `[detector]` TOML block is parsed-but-ignored (documentation-only);
   surprising for users. Either consume it or warn loudly.
4. No coverage artifact/badge; `environment.yml` forces all extras while pip
   keeps them optional.

---

## 2. Goal 1 — Oblique angle full compatibility

### What already works (v2.3.0 Phase A, verified in this audit)

- `CrystalMount`, `compute_omega_eta()` (Appendix-A solver),
  `find_reflections()` — `crystal/oblique.py`
- η threaded through **both** resolution backends:
  analytic (`analytic_resolution.py:27-56,96`) and MC LUT
  (`resolution.py:263-269`)
- Kernel bootstrap validates η against `compute_omega_eta()` (tolerance
  1e-3 rad) and embeds θ/η/ω in kernel metadata (`kernel.py:74-110`)
- Forward **and** identify orchestrators wrap in
  `reflection_theta_if_oblique()`; η=0 is bit-identical to v2.2.0
- HDF5 provenance records `geometry_mode`/η/θ/mount

### Remaining gaps (ordered by user impact)

| # | Gap | Where | Effort |
|---|---|---|---|
| G1.1 | No identify-pipeline oblique integration test (only forward has Gate C) | `tests/` | 0.5 d |
| G1.2 | No z-scan + oblique combination test | `tests/` | 0.5 d |
| G1.3 | `dfxm-find-reflections` CLI not wired (function exists + tested) | `crystal/oblique.py:241-288`, `pyproject.toml` | 1 d |
| G1.4 | Pixel-level R_x(η) absent from direct-space `forward()` — η only enters via the resolution kernel. Intentional Phase-A scope, but it is the remaining physics gap vs darkmod (~3.9 % on the thinnest covariance eigenvalue) | `forward_model.py:769-825` | 3–5 d incl. parity re-validation |
| G1.5 | η-weave refinement (weave η through geometry instead of post-hoc R_x) | resolution backends | 2–3 d, optional |
| G1.6 | Verify `crystal/remount.py` S-matrix and `viz/mosaicity.py` behave correctly under oblique | both files | 1 d |

### Definition of done ("full compatibility")

- [ ] All three identify modes (single / multi / z-scan) pass an oblique
      integration test with provenance round-trip.
- [ ] `dfxm-find-reflections --config X.toml` prints the accessible-reflection
      table (paper Table A.2 reproduction as the acceptance test).
- [ ] Decision recorded on G1.4/G1.5: either implement pixel-level η or
      document the ~4 % approximation as accepted (spec update).
- [ ] Remount + oblique interaction tested or explicitly forbidden with a
      clear error.

**Estimated total: ~1.5–2 weeks** (1 week if G1.4/G1.5 are documented-as-accepted
rather than implemented).

---

## 3. Goal 2 — Profile & optimize identify fan-out

### Audit findings: where the time goes

Per-config cost is dominated by **fixed overhead, not physics**. Today
`scripts/fanout.py:175-181` spawns a *fresh Python subprocess per config*
(`dfxm-identify --config …`); each one pays:

| Phase | Est. cost | Source |
|---|---|---|
| Interpreter start + imports | 100–500 ms | `fanout.py:36-45` |
| Numba JIT (cold cache) | 1–5 s | `dislocations.py` / `forward_model.py` kernels |
| Kernel `.npz` load (50–300 MB) | 100–500 ms | `forward_model.py:330`, `io/hdf5.py:293` |
| `Fd_find_mixed()` per (plane, b, angle) — **never cached in identify mode** | 50–200 ms each | `pipeline.py:1534,1727,1735,1947,1957,1972` |
| Frame loop (`@njit`, nogil, threaded) | 100–2000 ms | `io/hdf5.py:247-252` |
| HDF5 write | 50–500 ms | `io/hdf5.py:202-253` |

Additional waste: `render_per_dislocation=True` recomputes all frames from
scratch for each of the 3 detector files (`io/hdf5.py:634`), and identify
mode bypasses the existing `strain_cache` that forward mode uses
(`io/strain_cache.py:13-69`).

### Phase 2a — Measure first (don't optimize blind)

1. Add `scripts/profile_identify.py` mirroring `profile_forward.py`:
   cProfile + per-stage `perf_counter` for one config of each mode
   (single / multi / z-scan), reporting **startup vs Hg vs frames vs I/O**
   as separate numbers. (~1 d)
2. Add a `--timing-json` flag to `fanout.py` that records per-config wall
   time + phase breakdown into the manifest, so cluster runs produce
   throughput data (configs/hour, images/sec) for free. (~0.5 d)
3. Run the 8-worker laptop benchmark and one LSF array task; commit the
   numbers to `docs/cluster-profiling.md` as the baseline. (~0.5 d + queue)

### Phase 2b — Optimization backlog (apply in this order, re-measure between)

| Pri | Change | Where | Expected win |
|---|---|---|---|
| P0 | **Persistent worker pool**: replace subprocess-per-config with `multiprocessing` workers that import once, JIT once, load the kernel once, then consume configs from a queue. Keep `--isolate` flag for the old behavior (debugging). | `fanout.py:144-206` | Removes 1.5–6 s fixed cost/config; biggest single win at 5 000-config scale |
| P0 | **In-process Hg cache** keyed by (plane, b_idx, angle, z, position): a dict at sweep level. Single mode's Cartesian sweep re-derives geometry that repeats across z-layers and seeds. | `pipeline.py:1514-1542` | 10–50× on Hg stage for library sweeps |
| P1 | **Frame reuse for `render_per_dislocation`**: compute combined + per-dislocation frames once, write 3 files from memory. | `pipeline.py:1724-1746`, `io/hdf5.py:634` | ~3× when the flag is on |
| P1 | Fuse `Fd_find_mixed` into a single `@njit` kernel (pattern already exists: `_population_hg_kernel`, `dislocations.py:400`). | `dislocations.py:224-326` | 2–5× on Hg compute |
| P2 | Precompute unique `Z_shift(z)` grids and Ud matrices before the sweep loop. | `pipeline.py:1504,1524-1525,1881` | 5–10 % |
| P2 | Ship the numba cache: run a warmup in `dfxm-bootstrap` so array jobs never cold-compile (stampede noted in 2026-05-27 profiling spec). | `init_cmd.py` / bsub templates | removes compile stampede |

### Definition of done

- [ ] Baseline + post-optimization numbers in `docs/cluster-profiling.md`
      (configs/hour on laptop-8-worker and one LSF node).
- [ ] Target: **≥5× throughput** on a 500-config identify sweep (the P0 pair
      alone should reach this if the audit estimates hold).
- [ ] Determinism preserved: seeded sweeps produce bit-identical HDF5 vs
      v2.4.0 (extend `test_fanout.py`).

**Estimated total: ~2 weeks** (2a: 2 d; P0s: 4–5 d; P1s: 3–4 d).

---

## 4. Goal 3 — Multi-reflection sweeps

This is "Phase B" of the existing design
(`docs/superpowers/specs/2026-05-28-multi-reflection-oblique-angle-design.md`)
plus sweep tooling. It **depends on Goal 1** (oblique θ/η machinery) and
**benefits from Goal 2** (sweeps multiply the config count by N_reflections).

### Current blockers found in the audit

- `[reciprocal] hkl` is a single tuple; no `[[reflections]]` array parsing
  (`pipeline.py`, `kernel.py`).
- Burgers vectors hard-restricted to the {111} family
  (`crystal/burgers.py:34-57`); slip systems fixed to the 12 FCC
  {111}/⟨110⟩ entries (`forward_model.py:54-73`);
  `_ALL_111_PLANES` (`pipeline.py:1603-1608`).
- Sweep generators hardcode Al 111 @ 17 keV
  (`gen_sweep_configs.py:40`, `gen_identify_sweep_configs.py:58-59`).
- One kernel `.npz` per (θ, η, keV) — multi-reflection runs need per-reflection
  kernel lookup/bootstrap.

### Plan

1. **Config schema** — parse `[[reflections]]` (each entry: hkl, optional η
   target/ω choice), validated through `compute_omega_eta()`; reflections
   sharing (θ, η, keV) within tolerance may share a kernel. (2–3 d)
2. **Kernel management** — `dfxm-bootstrap` loops the reflection list,
   generating/looking-up one kernel per unique (θ, η, keV); manifest of
   kernels written next to the config. (2 d)
3. **Orchestrator loop** — forward and identify iterate reflections; HDF5
   layout gains a per-reflection scan group + provenance (hkl, θ, η, ω).
   Decide and document: q_hkl direction & Burgers projection change per
   reflection — this is the physics review step. (3–4 d)
4. **Per-reflection geometry correctness** — for cubic this is mostly done
   (`q_hkl` from hkl), but visibility/extinction (g·b = 0 invisibility) per
   reflection is the scientifically interesting output; add it as a computed
   label in identify HDF5. (2 d)
5. **Sweep tooling** — `--hkl-list` in both gen-sweep scripts; `fanout.py`
   treats reflection as one more sweep axis. (1–2 d)
6. **Tests** — extend `test_pipeline_multi_reflection.py` from scaffold to:
   2-reflection forward run, kernel sharing, g·b=0 invisibility check
   (a strong physics regression test), identify sweep across reflections.
   (2 d)

Note: within FCC, sweeping 111-family reflections is fully consistent with
the existing Burgers machinery. Sweeping to 200/220 etc. only changes q_hkl
(diffraction vector), **not** the slip systems — so this goal does *not*
require Goal 4. Keep that boundary explicit in the spec.

### Definition of done

- [ ] One TOML with `[[reflections]]` (e.g. 111 + 200 + 220 on Al) runs
      forward and identify end-to-end, one kernel per unique geometry.
- [ ] g·b invisibility criterion demonstrably reproduced across reflections
      (this doubles as a tutorial figure for Goal 5).
- [ ] Sweep generators + fanout support the reflection axis.

**Estimated total: ~2.5–3 weeks.**

---

## 5. Goal 4 — `.cif` files: all crystal structures, not just FCC

Largest item; already earmarked as v3.0.0 in
`crystal/oblique.py:33` (`[[followups-cif-crystal-structures]]`). The audit
confirms FCC/Al is hardcoded in ~8 places (constants, slip tables, Burgers
basis, d-spacing formula, Poisson ratio, cell matrix).

### Staged plan (each stage independently shippable)

**Stage 4.1 — General cell geometry (no CIF yet, ~1 week)**
- Replace `d = a/√(h²+k²+l²)` with the metric-tensor form
  `d = 1/|B·(hkl)|` from `(a, b, c, α, β, γ)` (`kernel.py:113-118`,
  `exposure.py:23`).
- Widen `CrystalMount` beyond `Literal["cubic"]`: build `C_s` from the six
  cell parameters; recompute `U_mount` for non-orthogonal axes.
- Generalize `compute_omega_eta()` / `find_reflections()` to the reciprocal
  metric. Cubic results must stay bit-identical (regression gate).

**Stage 4.2 — CIF ingestion (~1 week)**
- Dependency choice: **gemmi** (lightweight, no heavy stack, robust CIF
  parser) as an `[cif]` optional extra; ASE/pymatgen rejected for dependency
  weight. Record as an ADR in `docs/superpowers/specs/`.
- `[crystal] cif = "path/to/file.cif"` populates cell parameters + space
  group; explicit TOML keys override CIF values.
- Space-group extinction rules → systematically-absent reflections filtered
  out of `find_reflections()` and rejected at config validation with a clear
  error.

**Stage 4.3 — Slip systems & Burgers beyond FCC (~1.5–2 weeks)**
- Replace `_SLIP_SYSTEM_111` / `_BASIS_TABLE` with a data-driven registry:
  per structure-type tables (FCC {111}⟨110⟩, BCC {110}/{112}⟨111⟩,
  HCP basal/prismatic/pyramidal ⟨a⟩ and ⟨c+a⟩) selected by space group or
  explicit `[crystal] structure_type`. User-defined slip systems via TOML
  as the escape hatch for anything exotic.
- Burgers magnitude from the cell (e.g. a√3/2 for BCC ⟨111⟩/2) instead of
  the Al constant (`constants.py:20-21`).
- Per-material elastic input: keep **isotropic elasticity** (ν per material,
  from TOML/CIF metadata) for v3.0.0. Full anisotropic `C_ijkl` displacement
  fields are a research-grade rewrite of `crystal/dislocations.py` —
  explicitly out of scope; document as v3.x follow-up.

**Stage 4.4 — Validation (~1 week)**
- Golden tests: Al (FCC, regression), Fe or W (BCC), Ti or Mg (HCP, tests
  the non-cubic metric), one CIF round-trip per structure.
- One forward image per structure becomes a tutorial figure (feeds Goal 5).

### Definition of done

- [ ] `[crystal] cif = "Fe.cif"` + a BCC reflection produces a forward image
      and an identify library with BCC slip-system labels.
- [ ] Cubic/FCC path bit-identical to v2.4.0.
- [ ] Forbidden reflections rejected with explanatory error.
- [ ] Isotropic-elasticity limitation documented prominently.

**Estimated total: ~4–5 weeks.** Prerequisite: the `pipeline.py` split and
`forward_model.py` global-state cleanup (§1) — do them at the start of this
milestone if not already done.

---

## 6. Goal 5 — Final tutorial / walkthrough

### What exists

- `examples/identification_ml_tutorial/` — excellent, self-contained, but
  identify/ML-focused.
- `examples/cluster_showcase/` — needs cluster assets; template-grade.
- `docs/physics.md`, `docs/architecture.md` — good prose, no rendered site.
- Citations: the two Borgi papers + Poulsen 2021 in README only.

### Plan (style: minimal text, figure-led, citation-anchored)

1. **Notebook series** under `examples/` — each section ≤3 sentences of
   prose, then code + figure (1–1.5 weeks total, staged after each
   milestone so features land with their example):
   - `01_quickstart.ipynb` — empty-TOML → first image in 5 lines; the
     two-stage model in one diagram (reuse `docs/img/`). Cite Borgi 2024 §2.
   - `02_reciprocal_space.ipynb` — kernel point cloud, MC vs analytic
     backend comparison figure. Cite Poulsen 2021; Borgi 2024 §3.
   - `03_dislocations_and_contrast.ipynb` — edge/screw/mixed, weak-beam,
     COM ≈ −qi reproduction (README figure, now executable). Cite
     Borgi 2024 Figs. 3–5; Hirth & Lothe for the displacement field.
   - `04_oblique_and_reflections.ipynb` — *new features showcase*: oblique
     mount, `dfxm-find-reflections` table, g·b invisibility across a
     reflection sweep. Cite Borgi 2024 Appendix A. (After Milestones 1+3.)
   - `05_identification_at_scale.ipynb` — slim re-cut of the ML tutorial +
     fanout throughput plot from Goal 2's profiling. Cite Borgi 2025.
   - `06_other_crystals.ipynb` — CIF → BCC/HCP gallery. (After Milestone 4.)
2. **Docs site** — MkDocs Material (faster to stand up than Sphinx, good
   notebook embedding via mkdocs-jupyter) publishing `docs/` + executed
   notebooks to GitHub Pages; API reference via mkdocstrings. (2–3 d)
3. **References page** — `docs/references.md` with full BibTeX for: Borgi
   2024 (forward model), Borgi 2025 (identification), Poulsen 2021 (original
   MATLAB GO model), Poulsen 2017 (DFXM optics), Hirth & Lothe (dislocation
   fields), plus the darkmod comparison source. Every notebook links here. (0.5 d)
4. **CI executes notebooks 01–03** (papermill/nbmake, smoke-size configs) so
   the tutorial can't rot. (1 d)
5. README: swap the prose Roadmap section for a link to this document +
   notebook index. (0.5 d)

### Definition of done

- [ ] Six executed notebooks rendered on a public docs site.
- [ ] Each figure reproducible from a fresh `pip install dfxm-geo[dev]`.
- [ ] Every physics claim carries a citation to the references page.

**Estimated total: ~2 weeks**, deliberately spread across milestones.

---

## 7. Sequenced roadmap

Ordering rationale: optimize the fan-out *first* (every later goal runs
sweeps on top of it), close oblique second (multi-reflection builds on it),
then multi-reflection, then the CIF generalization, with tutorial chapters
landing alongside the features they demonstrate.

| Milestone | Version | Content | Est. effort | Depends on |
|---|---|---|---|---|
| **M1: Measure & speed up identify fan-out** | v2.5.0 | §3 — profile_identify, timing-json, persistent worker pool, Hg cache, frame reuse | ~2 wks | — |
| **M2: Oblique full compatibility** | v2.5.x | §2 — identify/z-scan oblique tests, find-reflections CLI, G1.4/G1.5 decision, remount audit | ~1.5–2 wks | — (parallelizable with M1) |
| **M3: Multi-reflection sweeps** | v2.6.0 | §4 — `[[reflections]]`, per-reflection kernels, g·b invisibility labels, sweep/fanout reflection axis | ~2.5–3 wks | M1, M2 |
| **Refactor gate** | v2.6.x | §1 — split `pipeline.py`; replace forward_model globals with explicit geometry state | ~1.5 wks | before M4 |
| **M4: CIF crystal structures** | v3.0.0 | §5 — metric-tensor geometry, gemmi CIF, extinction rules, BCC/HCP slip registries, golden tests | ~4–5 wks | refactor gate, M3 |
| **M5: Final tutorial & docs site** | v3.0.x | §6 — notebooks 01–06, MkDocs site, references page, notebook CI | ~2 wks (staged: 01–03 after M1/M2; 04 after M3; 06 after M4) | rolling |

**Critical path: M1 → M3 → refactor → M4 ≈ 11–13 working weeks**; M2 and the
early tutorial chapters run in parallel. Calendar estimate for a single
developer at research pace: **~4–5 months to v3.0.0 + final tutorial**.

### Suggested immediate next steps (this week)

1. M1 Phase 2a: write `scripts/profile_identify.py` and get the baseline
   numbers — every later perf decision hangs on them.
2. M2 G1.1/G1.2: the two missing oblique integration tests are half-day
   items that de-risk everything downstream of oblique.
3. File the G1.4/G1.5 decision (pixel-level η: implement vs accept ~4 % gap)
   as a short spec — it changes M2's scope by a week.
