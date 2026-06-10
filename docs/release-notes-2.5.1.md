# DFXM v2.5.1 — identify throughput: Fd dedup, fused Hg kernel, persistent worker pool

Released: 2026-06-10.

The M1 Phase 2b performance arc (roadmap workstreams W1/W2/W3), shipped as a
patch release: the ≥5× sweep-throughput target awaits cluster validation
(laptop measured 3.87×, see the benchmark below), so the version number stays
in the 2.5 line. No API removals; one default changed (the identify Hg
engine, output delta ~1e-15).

## W2 — one Hg seam, each dislocation's field computed once

All three identify orchestrators (single / multi / z-scan) now route their
Hg computation through a single seam, `find_hg_scene` in
`crystal/dislocations.py`. With `per_dislocation=True` the combined scene is
assembled as `Σ(Fg_i − I) + I` from per-dislocation fields computed **once**
— the pre-2.5.1 code recomputed every solo render from scratch (4 Fd
evaluations per multi sample instead of 2). The `engine="numpy"` path is
**bit-identical** to the legacy `Fd_find_mixed` / `Fd_find_multi_dislocs_mixed`
compositions (locked by `assert_array_equal` tests in `tests/test_hg_scene.py`).

## W3 — fused numba kernels are the default identify engine

`find_hg_scene(engine="numba")` — now the default — runs the Phase-1 fused
population kernel for combined-only scenes and a new fused
`_scene_perdis_hg_kernel` when per-dislocation solos are requested (combined
+ solos in one ray pass). Parity vs the NumPy oracle is ≤ atol 1e-14 /
rtol 1e-12 (measured ~1e-15 at the identify grid); no stored golden pins
identify output, and forward-mode goldens are untouched. Per-call Hg-stage
speedups at the flip: single 10.8×, multi 18.4×, zscan 14.4× — the Hg stage
drops from 61–70 % of a warm identify run (the v2.5.0 baseline) to ~13 %.
Pass `engine="numpy"` only to run the legacy bit-identical path for parity
verification.

## W1 — `scripts/fanout.py` runs a persistent worker pool by default

Pool workers (`dfxm_geo.fanout_worker.run_one` on a `ProcessPoolExecutor`)
import dfxm_geo once, JIT once, and keep the kernel LUT across configs —
amortizing the ~47 %-of-wall per-subprocess startup the old mode paid every
config. The pre-2.5.1 subprocess-per-config behavior remains behind
`--isolate` (use it to debug a config that kills workers). Hard worker death
is contained: finished results are kept, the batch retries on a fresh
executor, and a config that kills the pool twice is reported `rc=-2`.
**Pool and isolate outputs are bit-identical** (gated by a seeded
identify-multi test comparing every HDF5 dataset byte-for-byte).
`timing.json` gains one additive field, `sweep.isolate`; per-config rows are
unchanged (`import_s` is per-worker amortized in pool mode: real on a
worker's first config, 0.0 on warm ones). Both LSF templates document the
new default.

## Benchmark (laptop, 8 workers × 1 thread; `docs/cluster-profiling.md`)

| 16-config identify-multi sweep | configs/hour | vs v2.4.0 baseline |
|---|---|---|
| recorded baseline (isolate, pre-W2/W3) | 703 | — |
| v2.5.1 `--isolate` (W2+W3 only) | 1309 | 1.86× |
| v2.5.1 pool (full stack) | 2717 | 3.87× |

500-config production run (`--render-per-dislocation`): 2583 configs/hour,
15.8 images/s, 500/500 ok in 11.6 min; amortized import 0.20 s/config. The
≥5× DoD is **pending the LSF node run** (the 16-config shape cannot amortize
the pool's first-batch JIT; the cluster row follows `lsf/fanout.bsub`).

## Tooling

- `scripts/profile_identify.py` re-wired to time the `find_hg_scene` seam
  (the W2 routing had orphaned its old `Fd_find_mixed` patch points).

## Deferred / known-stale (unchanged from v2.5.0)

- `scripts/render_readme_examples.py` (bench-marked smoke test) has been
  broken since v2.5.0 removed `fm.xl_start`; untouched by this release —
  fix or retire alongside `render_rocking_gif.py`.
- Roadmap P2 levers (`Z_shift`/`Ud` precompute) are the next throughput
  candidates if the cluster row lands under 5×.
