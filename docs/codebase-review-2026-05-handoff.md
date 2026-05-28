# Codebase Review — Handoff (2026-05-28)

Working document for the next session. Branch: `claude/codebase-review-workflow-ZFCF1`
(do **not** merge yet). This captures a whole-codebase audit (bugs, inefficiency,
physics, feature extensions), what was fixed, and everything still open.

## TL;DR

- The **physics core is verified correct** — the Hirth & Lothe edge/screw fields,
  the frame-transform chain, `Hg = (Fg⁻¹)ᵀ − I`, the S-matrices, and the
  analytic-vs-Monte-Carlo resolution backends were all re-derived/checked. No
  critical physics errors.
- **Three batches of fixes shipped** to this branch (commits below). Full test
  suite green after each: **505 passed, 0 failed, 63 skipped** (the skips are
  environment-only — see "Environment gotchas").
- **Three larger items deferred** because they need validation on a machine with
  the MC kernel and a representative `ndis` (not available in the review
  container). Plus a backlog of smaller nits/doc-reconciliations listed below —
  none are blockers.

## Commits on this branch

| SHA | Batch | Summary |
|-----|-------|---------|
| `6865825` | 1 | `ndis` cache-key collision; `int16` bin overflow; remove debug `print(chunks)` |
| `5d17d7a` | 2 | stale-TOML `KeyError` paths; retire `compute_chi_shift`; `bincount`+guard+`ratio_outside`; `nogil` wall parallelism; psutil/git-provenance/.gitignore hygiene |
| `c3f0184` | 3 | moments/colormaps/burgers edge-case robustness (+4 regression tests) |

## How the review was run

Five parallel deep-audit agents, one per subsystem: (1) physics correctness,
(2) reciprocal-space/resolution, (3) performance/efficiency, (4) I/O+pipeline,
(5) analysis+viz. Each verified formulas symbolically/numerically where possible.
The full prioritized findings are reproduced below so nothing is lost.

---

## ✅ Fixed (shipped on this branch)

### Correctness
- **`ndis` missing from Fg cache key** (`forward_model.py` Find_Hg). `ndis` controls
  how many bipolar walls are summed but does not change array shape, so the shape
  guard couldn't catch a collision — two runs with same geometry / different `ndis`
  silently reused each other's strain field. Now `_ndis{ndis}` is in the filename.
- **`int16` bin-index overflow** (`resolution.py`). Indices past 32767 wrapped
  negative → rays silently mis-binned/dropped. Now `np.intp`.
- **Stale flat-TOML `KeyError` paths**: `load_h5_scan` (`hdf5.py`) and
  `dfxm-migrate-output --config` (`migrate.py`) read obsolete flat keys
  (`scan.phi_steps`, `crystal.dis`). Now parse the nested `[scan.phi]` /
  `[crystal.wall]` schema; migrate gives a clear error on missing keys.
- **All-zero kernel → silent all-NaN** (`resolution.py`): now raises with a clear
  message; also computes the documented `ratio_outside` and warns when the
  `qiN_range` clips the acceptance cloud.
- **`moments.calc_moments`**: raises on non-positive total intensity (was NaN/inf);
  removed the stray `m00 *= 2` MATLAB artifact.
- **`colormaps.inv_polefigure_colors`**: nearest-neighbour fallback for out-of-hull
  points (no more NaN RGBA past the `[0,1]` clip).
- **`viz/burgers.plot_slip_plane_3d`**: plane rendered from in-plane basis vectors
  instead of `z = .../n_z` — fixes inf/NaN blank render for `(110)`-type normals
  (`n_z=0`); raises on zero normal.

### Performance / hygiene
- **`np.add.at` → `np.bincount`** for the Resq_i histogram (`resolution.py`) —
  large kernel-gen speedup.
- **`nogil=True` on `_accumulate_bipolar_walls`** (`dislocations.py`) — the
  `ndis>100` wall-mode `ThreadPoolExecutor` now actually runs in parallel (the
  worker docstring already *claimed* the GIL was released; it wasn't).
- **Removed leftover `print(chunks)`** in the hot wall path.
- **Git provenance** (`hdf5.py`) now runs `git` with `cwd` pinned to the package
  repo root, not the process CWD (was recording an unrelated repo's SHA).
- **`psutil` tests** use `pytest.importorskip` (were hard-failing without the
  optional dep). **`.gitignore`** now covers the Fg-cache `*_vars.txt` sidecars and
  `direct_space/deformation_gradient_tensors/`.

### Retired
- **`analysis.mosaicity.compute_chi_shift`** removed entirely (function + tests +
  doc refs). It had an off-by-half-cell baseline bias (~100× the mosaicity signal)
  and an `abs()` sign-loss, and `run_postprocess` already bypassed it with
  `chi_shift=0.0`. The `chi_shift` keyword on `compute_com_maps` remains for
  externally-determined offsets.

---

## 🔭 Deferred — needs the MC kernel + a real `ndis` benchmark

These were intentionally **not** done in the review container: they are
performance refactors whose correctness/payoff must be validated against a real
kernel build and a representative large-`ndis` run, neither of which is available
here (no kernel npz, modest CPU/RAM).

### D1. Route wall mode through the fused `find_hg_population` kernel
- **Where:** `io/strain_cache.py` (`load_or_generate_Hg` → `Fd_find` → `fast_inverse2`),
  `direct_space/forward_model.py` (`Find_Hg`).
- **Problem:** centered/random modes use the fused nogil kernel
  (`_population_hg_kernel` / `find_hg_population`) that computes `rl→Fg→inv→Hg` in
  one pass. **Wall mode** still goes through the older path: `Fd_find` builds the
  full `(X,3,3)` Fdd, then `fast_inverse2` makes a second pass, materializing Fg
  twice. This is also the GIL-bound path D1 shares with the (now-fixed) nogil item.
- **Fix:** `build_dislocation_population(mode="wall")` already exists
  (`forward_model.py`); route wall mode through it + `Find_Hg_from_population`.
  The wall branch in `run_simulation` just doesn't use it yet.
- **Validate:** golden `forward` smoke output must stay bit-identical (or within
  documented tolerance); benchmark a cache-miss wall run (`ndis=151`, px510).

### D2. Chunk the `reciprocal_res_func` ray loop (memory)
- **Where:** `reciprocal_space/resolution.py`.
- **Problem:** ~10–14 full `Nrays` float64 arrays live at once → ~5–10 GB peak at
  `Nrays=1e8`. The `np.add.at`→`bincount` change (already shipped) helps speed but
  not peak memory; the q-arrays are the bulk.
- **Fix:** loop over ray chunks (the `scripts/compare_resolution_backends.py`
  already does this), accumulating into `Resq_i` per chunk. **Guard the chunking**
  so the `plot_figs` and `return_qs` paths (which need the full arrays) and the
  beamstop filtering still work — likely "chunk only when not plotting/returning".
- **Validate:** a full `Nrays=1e8` kernel build on a real node; confirm the
  resulting `normResq_i` is bit-identical (seeded) or statistically equivalent to
  the unchunked build, and that peak RSS drops.

### D3. Fg cache: float32 and/or compression
- **Where:** `io/strain_cache.py` (`np.save(file_path, Fg)`).
- **⚠️ Reproducibility hazard — read before attempting.** `test_hdf5_bit_equiv`
  derives `Hg` through the Fg disk cache and compares the rendered image
  **bit-for-bit** against a float64-era golden
  (`tests/data/golden/forward_legacy_writer_4frames_8x8.npy`). So Fg precision is
  load-bearing: downcasting to float32 only on save makes a cache-**hit** run
  (float32 reload) diverge from a cache-**miss** run (float64 in memory). Any
  float32 approach must downcast **consistently** in both paths *and* regenerate
  the golden — and that test **skips when no kernel is present** (as in this
  container), so it cannot be verified here.
- **Safer alternative:** keep float64, compress on disk (`np.savez_compressed`).
  But this migrates the cache from `.npy` to `.npz`, which ripples through the
  filename scheme (`forward_model.py`), the `_vars.txt` sidecar naming, the shape
  guard / load in `strain_cache.py`, and the `test_io.py` + `test_find_hg_z_cache`
  tests that glob `Fg_*.npy`. Treat as a deliberate cache-format migration.
- **Validate:** run `test_hdf5_bit_equiv` and the forward goldens on a machine
  with the kernel.

---

## 🟠 Open backlog (smaller, no kernel needed — safe next-session work)

Grouped by area. None are blockers. File refs are approximate (verify with grep).

### Correctness / robustness
- **z-offset cache filename collision** (`forward_model.py`, `z_suffix`).
  `f"_z{round(z_offset_um * 1000)}nm"` collapses sub-nm and sign: `z=+0.0004` and
  `z=−0.0004` both → `_z0nm` and overwrite each other. Fix: signed, higher
  precision, e.g. `f"_z{z_offset_um:+.6f}um"`. (Mirror of the `ndis` bug, same
  file.)
- **`compute_com_maps` negative-total guard** (`analysis/mosaicity.py`). Validity
  is `totals != 0`; a negative-sum pixel passes and yields a sign-flipped COM.
  Consider `totals > 0` (or clamp negatives to 0). Low risk; intensities should be
  ≥0 upstream.
- **`DFXM_MAX_WORKERS` invalid values silently ignored** (`io/images.py`). `0`,
  negatives, or non-int strings fall through to auto-detect with no warning. Add a
  one-line stderr warning on a present-but-unparseable value.
- **`viz/burgers.py` over-broad `except (ImportError, TypeError)`** on the plotly
  import — a genuinely-broken plotly would be masked as "not installed".

### Resolution / kernel niceties
- **Analytic backend silently ignores a configured beamstop**
  (`analytic_resolution.py`); docs/scripts claim it "raises if a beamstop is
  configured." Either implement the guard or fix the docs
  (`resolution-backend-comparison.md`, `scripts/compare_resolution_backends.py`).
- **Oversample factor `1.01` is hardcoded/fragile** (`resolution.py`). The
  `phys_aper` survival headroom is geometry-dependent; scale the factor with the
  expected survival fraction instead of a fixed 1%.
- **`mem_save` kwarg is dead** (`resolution.py`) — wire it to D2's chunking or
  remove.
- **No content hash on the kernel npz** (`kernel.py`). A `sha256(normResq_i)` would
  let consumers detect a corrupted/mismatched kernel (feature, not a bug).
- **Diagnostic plot x-axis 2× too wide** (`resolution.py`, `np.linspace(-qi1_range,
  qi1_range, ...)` vs bins over `±qi1_range/2`). Cosmetic — plots only.
- **`exposure.py` NA-acceptance rule differs** from the resolution model's
  (per-ray Gaussian threshold vs draw-and-box-clip). Confirm intentional.

### Single-sourcing / docs to reconcile
- **Duplicated geometry constants** (`forward_model.py` top vs `constants.py`):
  `psize`, `zl_rms`, `theta_0`, `Npixels`, `Nsub` are hardcoded again in
  `forward_model.py`. They agree today, but the documented override point is
  `constants.py` — import from there to single-source.
- **`migrate` "byte-identical" doc claim** (`output-format.md`) is false for
  float64 input (pixels are cast to float32 in `_create_detector_skeleton`).
  Soften the doc to "lossless for float32 source" or preserve source dtype.
- **`--postprocess-only` ignores `[postprocess].enabled=false`** (`pipeline.py`) —
  asymmetry with the normal path; document or honor consistently.
- **`/2.1` carries `scan_mode`/`scanned_axes`/`crystal_mode` attrs**
  (`hdf5.py`) — contradicts the `output-format.md` "Known gap (deferred to
  v1.3.0)" note. Code is arguably correct; reconcile the doc.
- **`analysis/__init__` and `moments.py` advertise an FWHM API that doesn't exist.**
  Remove the stale mention or add the estimator.
- **`rotatedU` rotation sense undocumented** (`crystal/rotations.py`) — it's a
  passive (transposed) Rodrigues matrix; document the convention so callers don't
  assume active rotation.
- **Commit `3b71b33`** referenced in `physics.md` (the `+2ν` Appendix-A sign
  correction) is **not in this repo's history**. The fix *is* present and correct;
  only the cross-reference is stale.

### Physics conventions to confirm against the paper (not suspected bugs)
- **Edge/screw axis pairing** (`dislocations.py`): the edge field uses line ∥ z_d
  (denom `(x²+y²)²`), the screw term uses line ∥ x_d (denom `z_d²+y_d²`), within the
  same `Ud` frame. Matches the cited Borgi 2025 Eq. 1 and is comment-flagged;
  confirm the mixed-character superposition is the intended parameterization.
- **`viz/mosaicity` sign-flip + transpose** (`(data*-1).T`) and the differing
  transpose handling between `plot_mosaicity_maps` and `plot_qi_cross_section` —
  documented and faithful to legacy `init_forward.py`, but eyeball against the
  published figures when reusing.

### Low-impact efficiency (only if they show up in a profile)
- **z-scan writes one `.npy` Fg per layer** (`strain_cache.py`) — consider a single
  chunked+compressed HDF5 Fg stack (ties into D3).
- **`crystal/burgers.py` nested Python loops** in `ud_matrices` /
  `rotated_t_vectors` — vectorizable with `np.cross` broadcasting; not on the
  forward hot path.

---

## 🚀 Feature-extension ideas (from the review)

1. General dislocation geometry — curved lines / loops, beyond straight parallel
   edge walls.
2. Anisotropic elasticity (Stroh) — replaces the isotropic H&L field; matters for
   non-Al cubics / HCP.
3. Surface & grain-boundary image forces (currently infinite-body).
4. Arbitrary strain-field input — read a displacement-gradient field from
   FEM/MD/DCT (the forward stage already consumes a generic `Hg`).
5. Inversion/fitting — fit dislocation params to measured rocking stacks
   (package is forward-only today).
6. GPU backend (`numba.cuda`) for the MC kernel and the per-frame forward.
7. Detector realism — PSF/MTF, dark current (Poisson noise already present).
8. Kernel content-hash provenance (see backlog item above).

---

## Environment gotchas (review container)

The 63 skipped tests are **environment-only**, not failures:
- **No MC kernel npz** under `reciprocal_space/pkl_files/` → all tests that need a
  real kernel skip (`test_hdf5_bit_equiv`, `test_hdf5_pipeline`,
  `test_analytic_backend_integration`, `test_find_hg_z_cache`, etc.). Generate one
  with `python -m dfxm_geo.reciprocal_space.kernel` (slow; or a small
  `generate_kernel(Nrays=...)` for smoke).
- **No `plotly`** (optional `[identification]` dep) → `test_viz_burgers` skips,
  including the new `(110)` vertical-plane regression test.
- **No `psutil`** (optional `[memory-aware]` dep) → the two `_auto_max_workers`
  tests skip (now via `importorskip`, previously hard-failed).
- **pytest** is not pre-installed; `pip install pytest` first. The full suite is
  ~2m45s.

## How to run

```bash
pip install -e ".[dev]"        # or at least: pip install pytest
pytest -q                      # full suite (~165s)
pytest tests/test_analysis.py -q
python -m dfxm_geo.reciprocal_space.kernel   # generate the MC kernel (un-skips many tests)
```

## Suggested next-session order

1. Knock out the **open backlog** items that need no kernel (z-offset filename
   collision, COM negative guard, `DFXM_MAX_WORKERS` warning, doc reconciliations)
   — small, testable, low-risk.
2. Then tackle **D1 / D2 / D3** on a machine with the kernel, one PR each, with
   before/after benchmarks and the bit-equivalence golden green.
3. Decide on merge once D1–D3 land or are explicitly deferred.
