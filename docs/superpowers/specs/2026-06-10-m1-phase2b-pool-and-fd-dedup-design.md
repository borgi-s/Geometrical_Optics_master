# M1 Phase 2b (v2.6.0): persistent worker pool + Fd dedup + Fd fusion

**Date:** 2026-06-10
**Status:** DRAFT — awaiting Sina's review. The LSF node baseline row
(`docs/cluster-profiling.md`) is being produced by a separate session; it is
the cluster denominator for the DoD but does not change this design.
**Baseline:** v2.5.0 (`origin/main = da891d9`), everything ctx-threaded
(`ForwardContext` everywhere, zero forward_model per-reflection globals).

## 1. Goal and measured starting point

Roadmap M1 Phase 2b (`roadmap202606audit.md` §3): **≥5× throughput on a
500-config identify sweep**. Laptop baseline (v2.4.0 ≈ v2.5.0 perf-wise, in
`docs/cluster-profiling.md`):

- Per-subprocess fixed cost: import 6.4–7.3 s uncontended, **16–20 s under
  8-way contention (~47 % of per-config wall)**; cold start (numba JIT +
  first kernel load) adds 10–14 s.
- **Hg geometry (`Fd_find*` + `fast_inverse2`) = 61–70 %** of the warm run
  in all three identify modes; frames ~17–23 %; HDF5 ~10 %; Poisson ~4 %.
- Throughput: 703 configs/hour, 4.3 images/s (16 multi-mode configs, 8×1).

## 2. Evidence that re-scopes the roadmap's second P0

The roadmap's P0 pair was (a) persistent worker pool and (b) an "in-process
Hg cache keyed (plane, b_idx, angle, z, position) … 10–50× on Hg stage".
Code investigation (2026-06-10, v2.5.0 source) kills (b) as written:

1. **Entry size.** One `Fg`/`Hg` is `(NN1·NN2·NN3, 3, 3)` float64 — at
   px510/Nsub=1 that is 1 474 200 rays × 72 B ≈ **106 MB per entry** (not
   the ~19 MB earlier notes assumed). A default single-mode sweep has
   4 planes × ≤6 b × 36 angles = up to 864 distinct entries ≈ 92 GB. No
   LRU budget rescues this: each config iterates the full Cartesian sweep
   in the same order, so any LRU smaller than the full sweep evicts every
   entry exactly before it would be reused (sequential-scan worst case).
2. **Multi mode — the production ML-sweep shape — has zero cross-config
   repetition.** `gen_identify_sweep_configs.py` varies only
   `noise.rng_seed`, but in multi mode the seed drives the *parameter*
   stream (`param_rng, _ = default_rng(seed).spawn(2)`, `pipeline.py:1835`)
   that draws each sample's `(Ud, alpha, position)` from continuous
   distributions. No two configs ever ask for the same Fd.
3. **Single mode is seed-independent up to the Poisson pass.** The sweep
   geometry (`pipeline.py:1644–1680`) consumes no RNG at all; the seed
   enters only in the post-write `_maybe_apply_poisson_noise`
   (`pipeline.py:1969`). A seed-only single-mode sweep recomputes byte-identical
   scans N times — but the right fix there is frame-level reuse
   (render once, re-noise N times), not an Hg cache, and that sweep shape
   is not what the ML pipeline currently runs. **Deferred** (§9).
4. **The real intra-config duplication:** with `render_per_dislocation=True`
   (which the ML mask workflow uses), every multi-mode sample evaluates each
   dislocation's `Fd_find_mixed` **twice** — once inside
   `Fd_find_multi_dislocs_mixed` for the combined scene
   (`pipeline.py:1872`, summed as `Σ(Fg_i − I) + I`,
   `dislocations.py:381–397`) and once for the solo render with the
   identical `(rl_eff, Us, Ud, alpha, Theta, position)`
   (`pipeline.py:1888–1904`). Z-scan does the same
   (`pipeline.py:2117–2148`). 4 Fd evaluations where 2 suffice.

**Re-scoped Phase 2b:** three workstreams that the evidence supports.

| WS | What | Expected win (laptop, multi sweep) |
|---|---|---|
| W1 | Persistent worker pool in `fanout.py` | ~2× (removes the 47 % import cost + JIT/kernel-load cold start) |
| W2 | Per-dislocation Fd dedup (multi + z-scan) | Hg calls 4→2 per sample → ~1.5× on the run when `render_per_dislocation=true` |
| W3 | Fuse `Fd_find_mixed`→Hg into one `@njit` kernel (roadmap P1, pulled in) | 2–5× on the Hg stage (Phase-1 `_population_hg_kernel` precedent was 14.7×) |

W1×W2 alone is ~3×; W3 is required to clear ≥5× honestly. (The roadmap's
"P0 pair alone should reach 5×" assumed the 10–50× Hg cache that doesn't
survive the evidence.)

## 3. W1 — persistent worker pool (`scripts/fanout.py`)

### Approach chosen

`concurrent.futures.ProcessPoolExecutor`, one submitted task per config,
workers calling the existing CLI entrypoints **in-process**:
`cli_main(argv)` / `cli_main_identify(argv)` already accept an explicit
argv list (`pipeline.py:1282`, `2243`) — the pool worker calls the same
function the `python -c` snippet imports today. Pool is the **default**;
`--isolate` keeps today's subprocess-per-config path verbatim (debugging +
hard-crash isolation).

Alternatives considered:
- *multiprocessing.Pool*: same semantics, weaker failure introspection than
  futures; no advantage.
- *Hand-rolled Process + Queue supervisor with worker respawn*: most robust
  against hard crashes (segfault/OOM-kill), but ~80 extra LOC; rejected for
  v2.6.0 — see failure handling below for the middle ground.
- *Threads in one process*: GIL + shared numba/frame-pool state, no
  isolation; rejected.

### Worker function lives in the package, not the script

Windows `spawn` (and the CI matrix) pickles the worker callable by module
path. `scripts/fanout.py` is loaded as a non-package module (tests use
`spec_from_file_location`), so the worker fn moves into the package:

```
src/dfxm_geo/fanout_worker.py
    def run_one(mode, config_path, output_dir, log_path) -> dict
```

`run_one` redirects `sys.stdout` **and** `sys.stderr` (tqdm writes to
stderr) to the per-config log file, times the run, calls
`cli_main`/`cli_main_identify` with the same flags the subprocess path
passes (`--no-postprocess` forward-only), catches every exception into
`rc != 0` + traceback-in-log (batch-resilience semantics preserved), and
**prints the same `DFXM_TIMING {json}` line into the log** so
`parse_timing_log` / `write_timing_json` work unchanged in both modes:

- `import_s`: real value on each worker's first config (measured around
  `import dfxm_geo.pipeline` in the worker, stashed in a module global),
  `0.0` on warm configs. The schema stays stable; the amortization is
  visible directly in the rows.
- `run_s`: per-config as today.

### Env pinning

`worker_env()` values (`DFXM_MAX_WORKERS`, BLAS/numba pins) must be set in
the **parent's** `os.environ` *before* the executor is created — spawn
re-imports numpy in the child and inherits the parent env either way
(fork on Linux/LSF, spawn on Windows). `run_manifest` sets/restores them
around the pool block. `_auto_max_workers` reads the env at call time
(`io/images.py:48`), so no further plumbing is needed.

### Failure handling

- Config raises → caught in `run_one`, `rc=-1`, traceback in log, sweep
  continues (same contract as today's `task()` wrapper).
- Worker dies hard (segfault/OOM-kill) → `BrokenProcessPool`. Recovery
  loop: catch it, mark in-flight config(s) failed, recreate the executor,
  resubmit the remainder (bounded retries: a config that breaks the pool
  twice is marked failed and skipped). ~30 LOC; preserves "one bad config
  doesn't kill a 100k sweep" without a hand-rolled supervisor.
- `--isolate` documented as the debugging fallback.

### What the pool retains per worker (the point of it)

Interpreter + imports (6–20 s), numba JIT (first call), and the kernel LUT:
`_KERNEL_CTX_CACHE` (`pipeline.py:95`, keyed by resolved kernel path) holds
the ~100–122 MB `ResolutionContext` for the worker's lifetime. Memory:
N_workers × (LUT + transient Fg arrays ~3 × 106 MB during a px510 multi
sample) ≈ 0.5 GB/worker peak — fine at 8 workers on the laptop and at 8–16
on a 32-core node. No per-worker warmup task needed: the first config per
worker pays JIT once, amortized over ~60+ configs/worker at 500-config
scale. (The roadmap-P2 `dfxm-bootstrap` warmup item becomes moot — note in
release notes.)

### `run_manifest` / CLI surface

- `run_manifest(..., isolate: bool = False)`; pool path branches internally.
  The existing `runner` test seam keeps exercising the isolate path; the
  pool path gets its own seam (injectable executor factory or worker-fn
  monkeypatch — decided in the plan).
- `--isolate` flag added to `main()`; `--n-workers`/`--threads-per-worker`
  semantics unchanged. LSF templates unchanged (pool becomes the default);
  `lsf/fanout.bsub` gains a comment documenting `--isolate`.
- `write_timing_json`: schema unchanged (sweep block + config rows);
  `sweep` gains `"isolate": bool`.

## 4. W2 — per-dislocation Fd dedup

`Fd_find_multi_dislocs_mixed` grows an opt-in components return:

```python
def Fd_find_multi_dislocs_mixed(..., return_components: bool = False):
    # combined = Σ(Fg_i − I) + I  (unchanged accumulation order)
    # return_components=True → (Fg_combined, [Fg_1, …, Fg_N])
```

Bit-identity by construction: the combined result is produced by the same
floating-point operations in the same order; the solo `Fg_i` are the very
arrays the solo renders would recompute. The orchestrators
(`_iter_identification_multi` `pipeline.py:1872–1905`,
`_iter_identification_zscan` `pipeline.py:2117–2148`) request components
only when `render_per_dislocation=True`, convert each part to its solo Hg
immediately and drop the `Fg` reference (bounds the transient peak: parts +
combined ≈ 3 × 106 MB at px510, comparable to today's sequential peaks —
verify with a memory check in the plan). When the flag is off, nothing
changes (no extra arrays materialized).

Z-scan additionally reuses the primary/secondary parts for its solo files
(`pipeline.py:2129–2148`); the solo detector files stay noiseless by design
(they bypass the Poisson pass) — unaffected.

## 5. W3 — fused `@njit` Hg kernel for `Fd_find_mixed`

Pattern proven by Phase 1 (`_population_hg_kernel`, `dislocations.py:400`,
14.7× on `Find_Hg_from_population`). Two design options, decided by a spike
benchmark at the start of the plan:

- **W3a (preferred if parity holds):** fuse the whole chain
  `Fd_find_mixed → fast_inverse2 → transpose − I` into one
  `hg_find_mixed(...)` `@njit` kernel returning **Hg directly** — never
  materializes the 106 MB `Fg`, halves peak memory of the Hg stage, and
  saves a full pass over the array. Call sites
  (`pipeline.py:1673–1680, 1872–1873, 1888–1905, 2117–2163`) collapse to
  one call. W2's components path then returns per-dislocation **Hg** parts
  plus the combined (the `Σ(Fg_i − I)` sum is formed inside the kernel).
- **W3b (fallback):** fuse only the `Fd_find_mixed` body (screw+edge
  displacement-gradient math per ray), keep `fast_inverse2` as is.

Either way the NumPy body is retained as `_fd_find_mixed_numpy` — the
parity oracle, same as Phase 1 (golden parity test at ~1e-15, plus the
existing `Fd_find_smoke` golden stays the safety net). `fastmath` choice
must match what the bit-identity gate needs: if fastmath breaks
pool-vs-isolate bit-identity or the v2.5.0 goldens, ship without it
(Phase 1 precedent: bit-exact and platform-deterministic was achievable).

**Bit-identity caveat (resolve in the plan's first task):** W3 changes the
floating-point instruction stream of the Hg stage, so outputs may not be
bit-identical to v2.5.0 (parity ~1e-15 instead). The determinism gate (§7)
is **pool-vs-isolate at the same code version** — that stays bit-exact.
Golden tests that assert bit-equality against v2.5.0-era arrays must either
keep passing (if the kernel reproduces the op order) or be regenerated in
the same commit with the regeneration documented — decide after the spike.

## 6. Timing-json continuity

- Both modes write per-config logs with `DFXM_TIMING` lines →
  `parse_timing_log` / count regexes unchanged.
- Pool mode: `import_s` is real on each worker's first config, `0.0` after;
  `wall_s` (launcher-side) minus `run_s` shrinks to queue/dispatch overhead
  instead of spawn cost. `sweep.isolate` records the mode.
- LSF templates keep passing `--timing-json`; the cluster row lands in
  `docs/cluster-profiling.md` (the "LSF node baseline: TODO" block —
  pending from the parallel session; the post-optimization run re-uses the
  same 16-config sweep for before/after comparability).

## 7. Determinism and gates

1. **Pool vs isolate bit-identity (new test, the headline gate):** run the
   same ≥2 seeded multi-mode configs (small grid per
   [[feedback-smoke-test-scale-down]]: e.g. 2 scenes × 5 frames, Npixels
   ≤128) through `run_manifest(isolate=True)` and the pool path; compare
   every detector dataset + positioners in the HDF5 outputs byte-for-byte
   (datasets, not raw files — master files embed timestamps).
   **On v2.5.0 semantics** (the #16 true-Bragg θ fix intentionally changed
   outputs vs v2.4.0 — the roadmap DoD line is amended accordingly).
2. **W2 bit-identity:** combined detector output with
   `render_per_dislocation` on/off unchanged by the dedup; solo files
   identical pre/post-dedup (same-version comparison).
3. **W3 parity:** fused-vs-NumPy oracle test ~1e-15; golden policy per §5.
4. **Full suite** `-m "slow or not slow"` AND a **kernel-less run** (rename
   `reciprocal_space/pkl_files` away — the v2.5.0 CI lesson: a local kernel
   masks CI-only failures). mypy 0 errors, ruff clean.
5. Existing `test_fanout.py` behavioural tests keep passing against the
   isolate path; new pool-path tests mirror them (concurrency cap, env
   restore, failure resilience incl. a BrokenProcessPool simulation).
6. Windows + Linux: the pool path must be exercised by CI (spawn) and at
   least one local Linux/WSL or cluster smoke (fork) before release.

## 8. Definition of done (kickoff + amendments)

- [ ] Baseline + post-optimization numbers in `docs/cluster-profiling.md`
      (laptop 8-worker; LSF node row when the parallel session delivers it).
- [ ] **≥5× throughput on a 500-config identify sweep** (multi-mode,
      `gen_identify_sweep_configs.py` shape, `render_per_dislocation=true`
      — the production ML-data configuration).
- [ ] Pool-vs-isolate bit-identical HDF5 on v2.5.0 semantics (§7.1).
- [ ] `--timing-json` works in both modes; LSF templates unchanged or
      updated in lockstep.
- [ ] Gates §7.4 green; CLAUDE.md + release notes updated at ship.

## 9. Deferred / out of scope (recorded so they don't get lost)

- **Cross-config Hg cache** (roadmap P0-B as written): rejected on
  evidence — see §2. Revisit only if a sweep shape with repeated discrete
  geometry appears.
- **Seed-replica rendering for single-mode noise sweeps** (render once,
  re-noise N times — ~10× for that shape): deferred until someone actually
  runs seed-only single-mode sweeps; file as a follow-up memory.
- **Frame reuse across the 3 detector files** (roadmap P1, `io/hdf5.py:634`):
  the 3 files render different physics (combined vs solo Hg), so the naive
  "compute once, write 3×" doesn't apply; verify during implementation
  whether any genuinely identical recomputation remains, and if so file it
  separately.
- Roadmap P2 items (Z_shift/Ud precompute, bootstrap JIT warmup): the pool
  makes the warmup moot; precompute is ~5–10 % — only if the 5× needs it.

## 10. Release mechanics (v2.6.0)

Bump + rename `tests/test_version_is_2_5_0.py`; `git fetch` + check
`origin/main` **before** tagging; merge `--no-ff`; publish.yml on tag
(TestPyPI auto, PyPI gated on the `pypi` Environment — transient
`upload.pypi.org` timeouts: `gh run rerun --failed` + re-approve).
conda-forge: deps/entry_points/requires-python unchanged → autotick bot
version-only bump suffices. Also merge the **v2.5.0** bot PR when it
appears (not yet open as of 2026-06-10).
