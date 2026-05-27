# Forward-model static/dynamic split — hoist the loop-invariant `qs` out of the per-frame loop

**Date:** 2026-05-26
**Status:** Design proposed; pending Sina's review before plan-writing
**Release:** TBD (patch/minor — bit-exact performance change, no API break)
**Reference implementation:** `C:\Users\borgi\Documents\HFP_Book_Ch\direct_space\forward_model.py` (lines 146–207: `precompute_forward_static` + `forward_from_static`)

## Summary

`forward()` recomputes `qs = Us @ Hg @ q_hkl` — a batched 3×3 matmul over ~1.47M
rays — on **every** scan frame. Profiling shows this single line is **~73% of the
per-call cost** (~1.08 s/call at the default `Nsub=1, Npixels=510`), yet it depends
*only* on `Hg`: it is identical for every frame of a scan. The angles (`phi`, `chi`,
`TwoDeltaTheta`) don't enter until the next line.

This spec splits `forward()` into a **static** half (computed once per scan) and a
**dynamic** half (per frame), so the 73% cost is paid once instead of N times. The
change is **bit-exact** (same float operations, just not repeated), so all existing
goldens stay green. The reference implementation already exists in `HFP_Book_Ch` and
is ported — not invented — with two corrections (keep production's `bincount`, keep
the `qi_field` guard).

## Motivation

### Measured, not assumed

A `cProfile` + clean wall-clock profiling pass on a representative real run
(5×5 mosa scan at px510, centered single dislocation, MC-LUT backend) produced:

**Per-frame `forward()` ≈ 1.08 s**, internal breakdown:

| Step | Share |
|---|---|
| **`qs = Us @ Hg @ q_hkl`** (batched 3×3 matmul, 1.47M rays) | **~73%** |
| `np.floor` / index computation | ~11% |
| LUT gather `Resq_i[i1,i2,i3]` | ~10% |
| `qc`, `qi = Theta @ qc`, bounds mask | ~6% |
| **`np.bincount` scatter** | **~3%** |

**One-time startup ≈ 24 s** (≈21 s numba JIT warmup at import, ~1.7 s kernel load,
~1.3 s `Find_Hg` on Fg-cache hit). Dominates *small* runs; negligible for full scans.

**Parallelism today:** `ThreadPoolExecutor` over frames gives ~3× (NumPy releases the
GIL on the large array ops); memory-capped at ~1 GiB of intermediates per `forward()`
call (`_auto_max_workers`, `io/hdf5.py`).

### The key fact: the dominant cost is loop-invariant

In `forward()`:

```python
qs = Us @ Hg @ q_hkl              # depends only on Hg  ← 73% of the call
qc = qs.squeeze().T + ang_arr     # ang_arr = f(phi, chi, TwoDeltaTheta)
qi = Theta @ qc                   # Theta = f(theta_0, TwoDeltaTheta)
```

`qs` uses neither the angles nor `Theta`. For a 61×61 mosa scan
(3721 frames × 2 for the perfect crystal ≈ 7442 calls), `qs` is recomputed
identically ~7442 times. Hoisting it to once-per-scan is the highest-leverage change
available, and it is free of numerical risk.

### Why the obvious alternatives are wrong or moot

- **einsum reassociation** (`np.einsum('ij,njk,k->ni', Us, Hg, qv, optimize=True)`)
  gives a 3.7× speedup on the matmul *in isolation*, but (a) it drifts results by
  ~1.3e-18 (float reassociation), which **breaks the bit-equivalence goldens**, and
  (b) once `qs` is hoisted to once-per-scan, speeding up a one-time 0.4 s step is
  noise on a multi-thousand-frame scan. **Dropped.**
- **GPU (cupy/CUDA):** the dense batched matmul that would suit a GPU is exactly what
  leaves the per-frame loop. The residual per-frame hot path (`np.floor`/indexing +
  LUT gather) is memory-bound fancy-indexing, not GPU-friendly. Adding a CUDA
  dependency to a freshly-PyPI-published package for this is premature. **Deferred
  pending the post-hoist re-profile.**
- **More parallelism / higher worker cap:** already 3× via threads; the cap is a
  memory function. The hoist *frees* the ~270 MiB `qs`/`base_qc` allocation from each
  worker, so the cap can rise — but that is a separate, measured follow-up, not part
  of this change. **Deferred pending re-profile.**

### Premise correction

`Nsub = 1` and `Npixels = 510` are **already the code defaults**
(`forward_model.py:69-70`). No change is needed there unless config-layer enforcement
is later desired; that is out of scope here.

## The reference implementation

`HFP_Book_Ch/direct_space/forward_model.py` already contains the split:

```python
def precompute_forward_static(Hg, TwoDeltaTheta=0.0):
    # Theta depends only on theta_0 + TwoDeltaTheta
    Theta = ...
    qs = Us @ Hg @ q_hkl          # "expensive and independent of phi/chi"
    base_qc = qs.squeeze().T
    return Theta, base_qc

def forward_from_static(Theta, base_qc, phi=0.0, chi=0.0, TwoDeltaTheta=0.0, qi_return=False):
    ang_arr = np.asarray([[phi - TwoDeltaTheta/2], [chi], [(TwoDeltaTheta/2)/np.tan(theta_0)]])
    qc = base_qc + ang_arr
    qi = Theta @ qc
    # floor → mask → LUT gather → scatter
```

It is ported, not redesigned. **Two corrections on the way in:**

1. **Keep production's `np.bincount` scatter.** The HFP version still uses the slow
   `np.add.at` (its line 202). Production already replaced that (~10× on that step).
   The ported `forward_from_static` must use `np.bincount(_flat_indices[idx], ...)`.
2. **Keep the `qi_field` guard.** HFP builds `qi_field` unconditionally (its line 184).
   Production only builds it when `qi_return=True`. Preserve the guard.

**One correctness subtlety:** `base_qc` is `TwoDeltaTheta`-independent (it is just
`(Us @ Hg @ q_hkl).squeeze().T`); only `Theta` depends on `TwoDeltaTheta`. So `Theta`
must be (re)built **per frame** from each frame's `TwoDeltaTheta` — it must NOT be
baked into the static part, or 2θ scans break. `Theta` is a trivial 3×3 (~0.02 ms),
so per-frame rebuild is free.

## Design

### Function surface (Q2: keep `forward()` as a wrapper)

In `dfxm_geo/direct_space/forward_model.py`:

- **New** `precompute_forward_static(Hg) -> base_qc`. Computes
  `base_qc = (Us @ Hg @ q_hkl).squeeze().T` once. (Does **not** take `TwoDeltaTheta`;
  `Theta` is per-frame — this is the one deviation from the HFP signature, and the
  correct one.)
- **New** `forward_from_static(base_qc, phi=0, chi=0, TwoDeltaTheta=0, qi_return=False)`.
  The per-frame half: builds `ang_arr`, `Theta` (from `TwoDeltaTheta`), `qc`, `qi`,
  the floor/index, the bounds mask, the LUT gather, and the `bincount` scatter —
  byte-for-byte the current `forward()` tail. Supports both the MC-LUT and analytic
  backends (same branch structure as today).
- **Refactor** `forward(Hg, phi=0, chi=0, TwoDeltaTheta=0, qi_return=False)` into a
  thin wrapper: `forward_from_static(precompute_forward_static(Hg), phi, chi,
  TwoDeltaTheta, qi_return=qi_return)`. Public signature and output bytes unchanged.

`base_qc` is read-only after construction; `forward_from_static` allocates a fresh
`qc = base_qc + ang_arr` per call, so it is safe to share one `base_qc` across worker
threads with no copy.

### Wiring (Q3: precompute in `_scan_frames_args`)

`_scan_frames_args` (`pipeline.py:1123`) is the single seam: forward, identification
single/multi, and z-scan all funnel through it, and each call already holds **one
uniform `Hg`** for its whole `args_list` (z-scan slices per-z and calls it once per z).

- In `_scan_frames_args`: compute `base_qc = precompute_forward_static(Hg)` once.
- Change the per-frame tuple from `(idx, Hg, phi, chi, two_dtheta)` to
  `(idx, base_qc, phi, chi, two_dtheta)`. The element type changes but the arity does
  not.
- Update the workers `_compute_frame` (`io/hdf5.py:144`) and `save_image`
  (`io/images.py`) to unpack `base_qc` and call
  `fm.forward_from_static(base_qc, phi=..., chi=..., TwoDeltaTheta=...)` instead of
  `fm.forward(Hg, ...)`.
- The z-scan Hg-cache generator (`pipeline.py:1095`) is unaffected in structure: it
  still yields one `Hg` per unique z; `_scan_frames_args` turns each into a `base_qc`.

### Memory knock-on (deferred, noted only)

Sharing one read-only `base_qc` removes the ~270 MiB `qs` allocation from each worker.
`_auto_max_workers` may therefore safely allow more workers. **This change does not
touch the cap** — it is logged as a post-re-profile follow-up so the new per-worker
footprint can be measured first.

## Scope

### In scope

1. `precompute_forward_static` + `forward_from_static` in `forward_model.py`; both
   MC-LUT and analytic backends supported in the dynamic half.
2. `forward()` refactored to a thin bit-exact wrapper over the two.
3. `_scan_frames_args` precomputes `base_qc` once and carries it in the frame tuple.
4. `_compute_frame` and `save_image` workers call `forward_from_static`.
5. Characterization test: `forward(Hg, φ, χ, 2θ)` equals
   `forward_from_static(precompute_forward_static(Hg), φ, χ, 2θ)` **byte-for-byte**
   across several angle combinations including non-zero `TwoDeltaTheta`.
6. Before/after wall-clock measurement on a real scan, recorded in the commit message.

### Out of scope (deferred pending the post-hoist re-profile)

- einsum reassociation (dropped — moot + golden-breaking).
- GPU backend.
- `_auto_max_workers` cap increase.
- Optimizing the new hot path (`np.floor`/indexing, LUT gather).

## Acceptance criteria (Q5: bit-exact + green + measured)

1. **Bit-exact:** new characterization test passes (wrapper ≡ split, byte-for-byte).
2. **Green:** full `python -m pytest -q` passes with the same pass/skip/xfail profile
   as before the change — in particular the `Fd_find_smoke` golden and the forward
   bit-equivalence tests are **untouched** (the wrapper preserves their bytes).
3. **Clean:** `mypy src/dfxm_geo/` reports 0 errors.
4. **Measured:** before/after wall-clock on a real scan (e.g. the 5×5 px510 profiling
   config, and ideally a larger scan) confirms the per-frame speedup landed
   (~2.9× per-frame term expected).

## Risks

- **Accidental reassociation breaking goldens.** The dynamic half must reproduce the
  exact operation order of today's `forward()` tail (`qs.squeeze().T + ang_arr` then
  `Theta @ qc`, `bincount` on float32-cast `prob`). The characterization test guards
  this; if it fails byte-for-byte, the port is wrong, not the goldens.
- **Thread-safety of shared `base_qc`.** Mitigated by `forward_from_static` never
  mutating `base_qc` (always allocates a fresh `qc`). Verified by the existing parallel
  writers continuing to pass.
- **2θ-scan regression** from baking `TwoDeltaTheta` into the static part. Mitigated by
  keeping `Theta` per-frame and testing a non-zero `TwoDeltaTheta` case explicitly.
