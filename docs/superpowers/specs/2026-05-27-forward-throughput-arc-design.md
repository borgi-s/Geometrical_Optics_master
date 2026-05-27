# Forward-throughput arc — design

**Date:** 2026-05-27
**Status:** design approved, pre-plan
**Goal:** Cut wall-clock per config so generating 100k+ DFXM images for ML
training is a ~3-node-hour, sub-TB job on a DTU HPC node. Targets the two
levers the 2026-05-27 cluster profiling identified (see
`cluster_profiling_results_2026-05-27` in auto-memory): the forward numba
kernel is *not* the bottleneck (~3-5% of a config), so this arc leaves it
alone and attacks `Find_Hg_from_population` + config-level throughput.

## Scope

One combined spec, **two sequential phases** so each lever's speedup is
independently attributable. Phase 1 is pure local work in `src/`; Phase 2's
cluster steps are run manually by the operator — this arc builds and
locally-validates the deliverables.

| Phase | Lever | Layer | Profiled share of a config |
|-------|-------|-------|----------------------------|
| 1 | `Find_Hg_from_population` | physics core (`src/dfxm_geo/`) | ~80% (random_dislocations, 4.2 s) |
| 2a | HDF5 detector dtype slim | I/O writer | 106 MB/config |
| 2b | Config-level fan-out | cluster harness (`scripts/`, `lsf/`) | throughput multiplier |

The phases ship sequentially: Phase 1 merges and is verified as a standalone
speedup before Phase 2 begins, so the two levers never confound each other.

---

## Phase 1 — `Find_Hg_from_population`

### Current path (the cost)

`Find_Hg_from_population` (`direct_space/forward_model.py:921`) →
`Fd_find_multi_dislocs_mixed` (`crystal/dislocations.py:348`) →
`Fd_find_mixed` per dislocation (`crystal/dislocations.py:224`):

- `Fd_find_multi_dislocs_mixed` runs a **pure-Python `for spec in crystals`
  loop**, summing `Fg_one - I` into an `(X, 3, 3)` accumulator.
- Each `Fd_find_mixed` call allocates a fresh `Fdd = np.zeros((X, 3, 3))`,
  does ~4 `(3,3)@(3,X)` coordinate-transform matmuls, the elementwise
  edge + screw field formulae, then a batched `Ud @ Fdd @ Ud.T`.
- Back in `Find_Hg_from_population`, `fast_inverse2(Fg)`
  (`crystal/rotations.py:63`, already a vectorized analytic 3×3 inverse with
  a "rewrite this" comment) runs over all X rays, then a transpose and
  `- I`.

At the profiled config (ndis=4, rocking scan) the per-dislocation Python loop
is cheap (4 iterations); the cost is the chain of **separate
bandwidth-bound NumPy passes over X rays**, each spilling temporaries —
exactly the shape the 2026-05-27 numba MC forward kernel beat ~6.6×.

### Approach (chosen)

A single `@njit` kernel with an **explicit double loop (dislocations × rays)**
that fuses, in one pass over X:

1. coordinate transform `r_l → r_d` per dislocation,
2. edge + screw field evaluation (`Fdd`),
3. `Ud @ Fdd @ Ud.T` rotation back to the grain frame,
4. accumulation across dislocations into `Fg`,
5. `Fg → Hg` via the analytic inverse + transpose + `- I`.

**Why the double loop over a NumPy broadcast-over-dislocations:** a broadcast
materialises an `(N, X, 3, 3)` intermediate and balloons memory at high ndis.
The dataset's ndis distribution is unsettled (could be 4 or 200), so the
explicit double loop — flat in memory and roughly linear in N×X regardless of
N — is the robust choice. It also matches the proven last-arc pattern.

The public signatures of `Find_Hg_from_population` / `Fd_find_*` stay
unchanged; the kernel is an internal `@njit` helper they delegate to. The
existing NumPy implementations are retained (or kept reconstructable) as the
reference for the parity test.

### Discipline

- **Profile-first.** Before writing the kernel, build a local repro of the
  ndis=4 rocking case and confirm with `cProfile` / line-level timing that the
  per-ray passes + inverse dominate and the dislocation loop is cheap. No
  guessing — mirrors the arc that "profiling killed the guesswork."
- **Correctness bar: 1e-12 rtol.** Assert `np.allclose(..., rtol=1e-12)`
  against a regenerated golden; allow tiny FMA/reassociation drift, document
  the tolerance inline. Regenerate the `Fd_find_smoke.npy` golden (or a new
  Find_Hg golden) and record why.
- **Suite green + mypy clean** (`python -m pytest -q`, `mypy src/dfxm_geo/`)
  with the venv python.

### Success criteria

- Local repro shows `Find_Hg_from_population` is no longer ~80% of a config —
  target a multiplicative cut comparable to the forward-kernel arc (numba
  fusion of bandwidth-bound passes typically several×). Exact target set after
  the profile.
- Parity within 1e-12 rtol; downstream forward image unchanged within the
  same tolerance.

---

## Phase 2 — fan-out + HDF5 slimming

### 2a. HDF5 detector dtype: float64 → float32 (lossless)

The detector `image` dataset (`io/hdf5.py:150-167`) is stored `float64` with
gzip-4 + shuffle, chunked per-frame — the bulk of the 106 MB/config. Switch
the stored dtype to **float32**, keep gzip/shuffle.

- ~2× smaller, **no quantization** — continuous diffracted intensities are
  preserved (chosen over lossy uint16 photon-count packing).
- Compute can stay float64; cast at write time.
- Update the writer + any dtype assertions; ensure `dfxm-migrate-*` and the
  bit/tolerance parity tests still pass (the golden comparison may need a
  float32 tolerance).
- Note the format change in `docs/output-format.md`.

### 2b. Config-level fan-out: in-node Python launcher

The threaded forward scan plateaus ~16 cores (memory-bandwidth bound), so a
128-core node is best used as **~8 concurrent configs × ~16 threads each**,
not 32+ threads on one scan.

Deliverable: `scripts/fanout.py` — an in-node launcher that

- consumes a **config manifest** (a list/dir of TOMLs; tie into the existing
  `scripts/scaling_sweep.py` to generate sweeps),
- spawns ~8 config-worker **processes** (process isolation, no shared-state /
  GIL contention),
- **caps each worker's thread pool** (~16) via a thread-budget knob: set
  numba / OMP thread env per worker and add an override to the ray-proportional
  `_auto_max_workers` so a worker can't grab the whole node,
- writes each config's output to its own HDF5 (Phase-2a float32),
- one bsub submits the whole batch (`lsf/` template calling `fanout.py`).

Worker/thread counts are configurable (default 8×16, tunable to the node).

### Validation

- Locally at **small scale** (2 workers × few threads, tiny scan grids per the
  smoke-test sizing rule) — confirm process isolation, thread capping, and
  per-config output correctness before any node run.
- Operator runs the real node batch manually.

### Success criteria

- Per-config HDF5 ~halved (float32).
- `fanout.py` runs N small configs concurrently locally with bounded threads,
  outputs one valid HDF5 per config, and the aggregate matches running them
  serially.
- Back-of-envelope confirms the 100k target stays ~3 node-hours / sub-TB with
  the slimmer HDF5 (re-checked against the profiling memory's numbers).

---

## Out of scope

- The forward numba kernel (already optimal per profiling; ~3-5% of a config).
- The analytic resolution backend as a speed lever (ruled out — 2.7× slower).
- Lossy intensity quantization (uint16) — explicitly declined in favour of
  float32 lossless.
- Multi-node LSF job arrays — in-node launcher only for this arc.

## Open follow-ups (not blocking)

- `Find_Hg_from_population` unused `psize` / `zl_rms` params — drop or consume
  while in the file (tracked in CLAUDE.md punch list).
- HDF5 codec sweep (lzf / blosc / zstd) — deferred; float32 + gzip is the
  conservative first cut.
