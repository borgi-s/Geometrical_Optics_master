# Find_Hg_from_population baseline (Phase 1 profile-first gate)

Date: 2026-05-27
Branch: `feature/forward-throughput-arc`
Repro: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" scripts/profile_find_hg.py`
Env: venv Python, NumPy 2.4.4, Windows laptop. Population =
`random_dislocations`, ndis=4, sigma=5.0, min_distance=4.0, seed=42
(the `configs/profile_rocking_random.toml` path). Call under test:
`Find_Hg_from_population(pop, h=-1, k=1, l=-1)`.

## Median wall time

```
median wall: 3302.5 ms  (n=5)
```

~3.3 s per call. Consistent with the cluster profiling result
(`cluster_profiling_results_2026-05-27`: 4.17 s random, cumulative within a
full config run; ~80% of a per-config wall, vs `forward_from_static` kernel at
0.11 s = ~3–5%). This is the ML-throughput bottleneck the perf arc now targets.

## cProfile cumulative breakdown (top rows, n=5 calls)

```
         411 function calls in 15.858 seconds

   Ordered by: cumulative time
   List reduced from 24 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        5    0.451    0.090   15.858    3.172 forward_model.py:921(Find_Hg_from_population)
        5    1.800    0.360   13.945    2.789 dislocations.py:348(Fd_find_multi_dislocs_mixed)
       20   12.142    0.607   12.144    0.607 dislocations.py:224(Fd_find_mixed)
        5    1.462    0.292    1.462    0.292 rotations.py:62(fast_inverse2)
       30    0.000    0.000    0.002    0.000 numpy/.../numeric.py:2242(identity)
       55    0.001    0.000    0.001    0.000 {built-in method numpy.zeros}
       30    0.001    0.000    0.001    0.000 numpy/.../_twodim_base_impl.py:176(eye)
       25    0.000    0.000    0.000    0.000 {built-in method numpy.asarray}
       30    0.000    0.000    0.000    0.000 <frozen importlib._bootstrap>:1207(_handle_fromlist)
        5    0.000    0.000    0.000    0.000 forward_model.py:960(<listcomp>)
       20    0.000    0.000    0.000    0.000 {method 'reshape' of 'numpy.ndarray' objects}
        5    0.000    0.000    0.000    0.000 numpy/.../fromnumeric.py:604(transpose)
```

(Times are over 5 profiled calls; divide by 5 for per-call. `tottime` is
self-time, `cumtime` includes callees.)

## What dominates

The per-ray NumPy strain-field passes dominate, **not** the dislocation loop:

- **`Fd_find_mixed`** (dislocations.py:224) — **12.14 s self-time / 12.14 s
  cumulative, ~77% of total wall**. Called 20× = 4 dislocations × 5 repeats,
  i.e. once per dislocation per call. Each call is the full per-ray finite-
  strain field over the whole lab-frame ray grid `rl`. This is the hotspot.
- **`fast_inverse2`** (rotations.py:62) — 1.46 s (~9%). The per-ray 3×3
  Fg-inverse done once per call to form `Hg = transpose(Fg^-1) - I`.
- **`Fd_find_multi_dislocs_mixed`** (dislocations.py:348) — 1.80 s self-time
  on top of its `Fd_find_mixed` callees; the orchestration/accumulation over
  the population (superposing per-dislocation fields).
- The **4-iteration dislocation loop itself is cheap** — there is no
  per-iteration Python overhead of note; cost is entirely inside the NumPy
  array passes of `Fd_find_mixed` (× ndis) plus the single `fast_inverse2`.

Confirmation: the NumPy per-ray passes (`Fd_find_mixed`) + `fast_inverse2`
account for ~96% of wall; the 4-iteration population loop is negligible. The
profile-first hypothesis holds — optimize the per-ray field math, not the loop.

## Concrete speedup target

`Fd_find_mixed` + `fast_inverse2` together = (12.14 + 1.46) / 15.858 ≈ **86%**
of `Find_Hg_from_population` wall (≈ 2.72 s of the 3.30 s per-call median; the
remaining ~0.45 s is `Find_Hg`'s own self-time + `Fd_find_multi_dislocs_mixed`
orchestration). These are the fusible per-ray array passes a single `@njit`
kernel can collapse (eliminating the large intermediate-array traffic that
makes NumPy memory-bandwidth bound here).

Per the cluster profiling note, the per-frame forward-scan kernel
(`forward_from_static`) costs **0.11 s** per config (~3–5% of a config wall).
The gate for this arc is to stop `Find_Hg_from_population` being the dominant
per-config stage:

> **Target: ≥10× speedup on `Find_Hg_from_population`** — drop the per-call
> median from ~3.3 s to **≤330 ms** (random, ndis=4). Justification: at 10×,
> the ~2.72 s of fusible per-ray work (`Fd_find_mixed` + `fast_inverse2`)
> collapses to ~0.27 s, bringing the whole call from ~3.3 s into the same
> order of magnitude as the 0.11 s forward kernel + ~0.19 s
> `precompute_forward_static` per config — i.e. Find_Hg stops being ~80% of
> per-config wall and becomes a minority stage alongside HDF5 write
> (the next-largest at ~3 s, addressed separately). A 10× target is realistic
> for fusing the NumPy passes into one `@njit(nogil)` kernel: the numba
> forward-kernel fusion in this codebase achieved ~6.6×/frame on similar
> per-ray array work (`session_handoff_2026-05-27_numba-mc-kernel`), and the
> Find_Hg passes carry even more intermediate-array traffic to eliminate.

Floor / stretch: an absolute floor is the HDF5-write stage (~3.07 s random) —
once Find_Hg is well below that, it is no longer the binding constraint and
further Find_Hg speedup yields diminishing per-config returns until HDF5 is
also addressed. Stretch goal ≥15× (≤220 ms) if the fused kernel parallelizes
cleanly over rays.
