# DFXM v2.5.0 — ForwardContext: the end of forward-model global state

Released: 2026-06-10.

The headline is internal but load-bearing: the forward model no longer keeps
any per-reflection state in module globals. Every public forward/identify
entry point now threads an explicit `ForwardContext`. On top of that ride a
user-visible bug fix for multi-mode instance labels, memory-bounded kernel
generation, and the identify-throughput profiling toolkit (roadmap M1
Phase 2a) with its measured baseline.

## ForwardContext refactor (#16)

`direct_space/forward_model.py` previously held the loaded resolution kernel,
ray grid, Bragg angle and reflection identity as module globals, mutated by
the kernel loaders and the `reflection_theta_if_oblique()` context manager.
All of it is gone:

- Deleted globals: `theta_0/theta/Theta`, `xl_start/xl_range`, `rl`,
  `prob_z`, `Resq_i` + the `qi*` grid family, `Hg`, `q_hkl`,
  `_analytic_eval`, `_loaded_kernel_path`; deleted machinery:
  `reflection_theta_if_oblique`, `_context_from_globals`,
  `_forward_state_guard`.
- Loaders return a `ResolutionContext`; per-process instrument constants
  (pixel grid, `Ud`/`Us`, …) remain module-level.
- `forward`, `forward_from_static`, `precompute_forward_static`, `Find_Hg`,
  `Find_Hg_from_population` and the HDF5 writers
  (`write_simulation_h5` / `write_identification_h5`) **require `ctx`**.
  `Z_shift(offset_um, *, xl_range)` takes its range explicitly.
- Kernel-load idempotency lives in a module `_KERNEL_CTX_CACHE` (cleared
  between tests by an autouse fixture; bounded by construction in
  production).

Why it matters beyond hygiene: cross-config state leakage on a long-lived
worker process is now impossible by construction — the prerequisite for the
persistent worker pool planned in the next throughput release.

**Behavioral change (intentional):** simplified-geometry runs now use each
reflection's *own* Bragg angle via `run_theta` (validated through
`_validate_reflection`) instead of the import-time `theta_0` constant. The
pickle-era bit-equivalence golden is now a documented `xfail`; `io/migrate.py`
records true-Bragg θ (~58 µrad shift vs the old constant).

## Fixed: per-dislocation instance labels rendered at the origin

In `mode = "multi"` with `render_per_dislocation = true` (the ML
instance-label path), the combined scene placed each dislocation at its drawn
position but the solo `dis0`/`dis1` renders forgot `position_lab_um` — both
instance labels were rendered at the origin and could not overlay the
combined image. Present since the flag shipped (v1.2.0-era) through v2.4.0.
The solo renders now receive the same position stored under
`/N.1/sample/dislocations/<k>/position_um`; a regression test pins it.

## Memory-bounded kernel generation (`batch_size`)

`generate_kernel(..., batch_size=N)` / `reciprocal_res_func` accumulate the
Monte Carlo rays into the histogram in batches of `N`, so peak memory is
~O(batch) instead of ~O(Nrays) — the full `Nrays=1e8` bootstrap now runs on
small machines. A single batch is bit-identical to the unbatched path
(tested); diagnostic modes (`return_qs`, `plot_figs`) keep the single-shot
path. API-only for now (no CLI flag).

## Identify-throughput profiling (roadmap M1 Phase 2a)

- `scripts/profile_identify.py`: warm + measured single-thread pass with
  per-stage wall-time accumulators (kernel load / Hg geometry / frames /
  HDF5 / Poisson), one laptop-scale config per identify mode under
  `configs/profile_identify_*.toml`.
- `scripts/fanout.py --timing-json`: the child CLI invocation now reports an
  import-vs-run split (`DFXM_TIMING` log line); the flag writes a sweep
  manifest with per-config timings and throughput (configs/hour, images/s).
  Both LSF templates emit it by default.
- Measured baseline (in `docs/cluster-profiling.md`): Hg geometry is
  **61–70 %** of a warm identify run in all three modes; subprocess import
  is **~47 %** of per-config wall in an 8-worker sweep. The next throughput
  release targets exactly those two numbers (persistent worker pool +
  sweep-level Hg cache).

## Deferred / known-stale

- `compute_Hg` on the kernel loaders is accepted-but-ignored (kept for call
  compatibility; removal in a later major).
- `scripts/render_rocking_gif.py` (manual docs-figure script, not in CI;
  its README GIF was replaced by the 4-panel COM↔qi figure in v2.1.2) still
  pokes pre-#16 globals and will crash if run — fix or retire it when the
  figure is next regenerated.
- `#8 rotation_deg` in `build_dislocation_population` (needs a screw/edge
  distribution decision) and `#17 spatial-projection mode` (own spec) moved
  out of this release.
