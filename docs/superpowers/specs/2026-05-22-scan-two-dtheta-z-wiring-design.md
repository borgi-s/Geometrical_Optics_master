# Wire `[scan.two_dtheta]` / `[scan.z]` into forward + identification

**Status:** approved 2026-05-22 (brainstorming → spec).
**Owner:** Sina Borgi.
**Targets:** v1.3.0.
**Predecessors:**
- B + C (`docs/superpowers/specs/2026-05-21-scan-modes-crystal-layouts-design.md`) — established the `[scan.<axis>]` schema, `ScanGrid`, `build_scan_grid`, and the eager `ValueError` guard in `run_simulation` that this design lifts.
- E (`docs/superpowers/specs/2026-05-21-identification-hdf5-design.md`) — established the BLISS master+per-scan HDF5 layout and the `ScanSpec` ↔ `MasterWriter.add_scan` interface that this design extends to four positioner axes.

## Goal

Make `[scan.two_dtheta]` and `[scan.z]` first-class scan axes in `dfxm-forward` and `dfxm-identify` (single + multi modes). Today both axes can be configured in TOML and are accepted by `ScanConfig` / `build_scan_grid`, but the forward kernel only iterates `(phi, chi)` and `run_simulation` raises `ValueError` if either of the unwired axes is scanned.

After this change, the same TOML surface supports the full canonical 4-axis trajectory `(phi, chi, two_dtheta, z)`, with the existing `derived_mode_name()` labels (`mosa`, `mosa_strain`, `mosa_layer`, `mosa_strain_layer`, plus the 1-axis variants `strain`, `layer`) reflecting which axes are scanned. The BLISS HDF5 output gains per-frame positioner arrays for all 4 axes; no schema change.

## Non-goals

This spec does **not** address:

- Retiring the dedicated identification `mode = "z-scan"` (item B in the v1.3.0 follow-up list). Once `[scan.z]` is wired, `mode = "single"` + `[scan.z]` covers the same physics, but the migration story is out of scope here.
- `render_per_dislocation` analogue for z-scan (item C). Depends on the above.
- Pixel-level segmentation masks for `multi` mode (item D). Independent.
- Removing the temporary `mode = "z-scan"` identification iterator. It stays untouched.
- Caching `Find_Hg_from_population` outputs. Centered/random_dislocations remain uncached, matching v1.2.0 behaviour.
- Changing `forward()`'s inner math. `forward(Hg, phi, chi, TwoDeltaTheta)` already exists (`forward_model.py:402-543`) and is correct.

## Current state

What exists:

- `ScanConfig` (`pipeline.py:106-155`) — 4-axis canonical config with `derived_mode_name()` covering all 0–4-axis scan combos via `_PRE_CANONIZED_MODE_NAMES`.
- `ScanGrid` + `build_scan_grid` (`forward_model.py:548-594`) — emit per-axis sample arrays for all 4 canonical axes. Unit-tested in `tests/test_pipeline_scan_modes.py`. **Not consumed by the forward kernel.**
- `forward(Hg, phi, chi, TwoDeltaTheta=0)` (`forward_model.py:402-543`) — accepts `TwoDeltaTheta` with full per-frame `Theta` rotation and `ang_arr` adjustment. Inner math handles two_dtheta correctly.
- `Z_shift(offset_um)` (`forward_model.py:513-539`) — returns a z-translated `rl` grid. Currently used only inside `_iter_identification_zscan` (the dedicated identification z-scan mode).

Where the gap is:

- `_frame_grid_from_scan` (`pipeline.py:957-976`) — emits only `(Phi_rad, Chi_rad, n_frames)`. Drops two_dtheta + z.
- `_scan_frames_args` (`pipeline.py:979-997`) — emits only 4-tuples `(idx, Hg, phi, chi)`. No per-frame two_dtheta or z.
- `_positioners_for_scan` (`pipeline.py:1000-1011`) — writes only `phi` + `chi` into the positioners dict.
- `run_simulation` (`pipeline.py:616-622`) — eager `ValueError` rejecting two_dtheta/z scans.
- `Find_Hg` cache key (`forward_model.py:209-219`) — includes `dis/psize/zl_rms/Npixels/Nsub/remount_name` but not `z_offset`. A z-aware cache must extend the filename.

## Design (Approach A: extend in place)

### Section 1: `ScanFrames` + helpers

Add a new dataclass and helper in `pipeline.py`:

```python
@dataclass(frozen=True)
class ScanFrames:
    """Per-frame trajectory arrays for one scan, all parallel of length n_frames.

    Frame ordering: phi-innermost, chi, two_dtheta, z-outermost.
    Units: phi/chi/two_dtheta in radians; z in micrometers.
    """
    phi_pf: np.ndarray
    chi_pf: np.ndarray
    two_dtheta_pf: np.ndarray
    z_pf: np.ndarray
    n_frames: int


def _build_scan_frames(scan: ScanConfig) -> ScanFrames:
    """Flatten the Cartesian product of the canonical 4-axis ScanGrid."""
    grid = build_scan_grid(scan)
    phi, chi, two_dtheta, z = grid.samples
    # Cartesian product with phi-innermost, z-outermost.
    phi_pf, chi_pf, twodt_pf, z_pf = _flatten_4d(phi, chi, two_dtheta, z)
    return ScanFrames(phi_pf, chi_pf, twodt_pf, z_pf, n_frames=phi_pf.size)
```

`_flatten_4d` implementation:

```python
phi_g, chi_g, twodt_g, z_g = np.meshgrid(phi, chi, two_dtheta, z, indexing="ij")
# Ravel order: phi-innermost. Equivalent to nested loops
#   for z in z_arr: for twodt in twodt_arr: for chi in chi_arr: for phi in phi_arr: yield ...
# Use 'F' order on the meshgrid axes, or transpose+ravel-C.
return tuple(arr.ravel(order="F") for arr in (phi_g, chi_g, twodt_g, z_g))
```

`_frame_grid_from_scan` is deleted. Call sites migrate to `_build_scan_frames`.

`_scan_frames_args` is the single-Hg helper used by identification iterators (where Hg is constant across a `ScanSpec`'s inner frames):

```python
def _scan_frames_args(
    Hg: np.ndarray, frames: ScanFrames
) -> tuple[list[tuple[int, np.ndarray, float, float, float]], dict[str, np.ndarray | float]]:
    """Build (args_list, positioners) for one ScanSpec.

    Each args tuple is (frame_idx, Hg, phi_rad, chi_rad, two_dtheta_rad).
    """
```

For forward mode, where Hg varies with z, the orchestrator described in Section 2 handles per-z Hg recompute internally; identification iterators feed `_scan_frames_args` once per (z, plane, b, alpha) configuration.

`_positioners_for_scan` is extended to write all 4 axes; fixed axes still collapse to scalar:

```python
def _positioners_for_scan(frames: ScanFrames, scan: ScanConfig) -> dict[str, np.ndarray | float]:
    return {
        axis: getattr(frames, f"{axis}_pf") if getattr(scan, axis).is_scanned else float(getattr(scan, axis).value)
        for axis in _CANONICAL_AXES
    }
```

### Section 2: Forward path — `run_simulation`

1. Lift the `ValueError` guard at `pipeline.py:616-622`.
2. Build `frames = _build_scan_frames(config.scan)`.
3. Identify unique z values: `z_uniques = np.unique(frames.z_pf)`. Each unique z needs one Hg recompute.
4. Define `Hg_provider(z) -> (Hg, q_hkl)`:
   - **wall mode:** `Find_Hg(w.dis, w.ndis, fm.psize, fm.zl_rms, h, k, l, S=S, remount_name=w.sample_remount, z_offset_um=z)`. Inside `Find_Hg`, the `rl` argument is computed as `Z_shift(z)` when `z != 0`.
   - **centered/random:** `Find_Hg_from_population(population, h=, k=, l=, S=S, rl=Z_shift(z) if z != 0 else None)`. `rl` becomes an explicit optional kwarg defaulting to the module-level `fm.rl`.
5. New orchestrator (lives in `pipeline.py`; uses the existing `_compute_and_write_detector_file_parallel` for the inner parallel work):

```python
def _iterate_simulation_frames(
    frames: ScanFrames,
    Hg_provider: Callable[[float], np.ndarray],
    max_workers: int | None,
) -> Iterator[tuple[int, np.ndarray, float, float, float]]:
    """Yield (frame_idx, Hg_for_this_frame, phi, chi, two_dtheta) tuples.

    Outer loop over unique z values (Hg recompute happens once per z).
    Inner iteration walks the frames in canonical order.
    """
    z_to_Hg: dict[float, np.ndarray] = {}
    for k in range(frames.n_frames):
        z = float(frames.z_pf[k])
        if z not in z_to_Hg:
            z_to_Hg[z] = Hg_provider(z)
        yield (k, z_to_Hg[z], float(frames.phi_pf[k]),
               float(frames.chi_pf[k]), float(frames.two_dtheta_pf[k]))
```

The detector-file writer (`_compute_and_write_detector_file_parallel`) consumes 5-tuples instead of 4-tuples (added `two_dtheta`). `_compute_frame` (the worker) is updated:

```python
def _compute_frame(args: tuple[int, np.ndarray, float, float, float]) -> tuple[int, np.ndarray]:
    idx, Hg, phi, chi, two_dtheta = args
    return idx, fm.forward(Hg, phi, chi, TwoDeltaTheta=two_dtheta)
```

6. `write_simulation_h5` signature changes: drop `phi_range/phi_steps/chi_range/chi_steps`; add `frames: ScanFrames`. Internally it derives positioners + n_frames from `frames`. The two `master.add_scan(...)` calls (still inside the loop introduced in `cb53f76`) pull positioners from `_positioners_for_scan`.

### Section 3: Identification path

`_iter_identification_single` and `_iter_identification_multi` use the same `ScanFrames` data shape, but build a per-z inner-trajectory (each ScanSpec is at a single z) instead of the full 4-D flat trajectory that forward uses. A new helper:

```python
def _build_scan_frames_at_z(scan: ScanConfig, z_value: float) -> ScanFrames:
    """Inner (phi × chi × two_dtheta) trajectory with z_pf fixed to z_value.

    Returned ScanFrames has phi_pf/chi_pf/two_dtheta_pf walking their
    Cartesian product (phi-innermost), z_pf = full-length np.full(n, z_value).
    """
```

Iterator shape:

```python
z_samples = build_scan_grid(config.scan).samples[3]  # ScanGrid.z is index 3
for z in z_samples:
    frames_at_z = _build_scan_frames_at_z(config.scan, float(z))
    for plane in planes:
        for b_idx in b_indices:
            for alpha in angles_deg:
                Hg = compute_Hg_for_this_disloc(z, plane, b_idx, alpha)
                args_list, positioners = _scan_frames_args(Hg, frames_at_z)
                yield ScanSpec(positioners=positioners, detectors={"...": args_list}, ...)
```

`compute_Hg_for_this_disloc(z, plane, b_idx, alpha)` is the existing identification Hg recipe (`Fd_find_mixed` for single mode, `Fd_find_multi_dislocs_mixed` for multi mode), called with `rl = Z_shift(z) if z != 0 else fm.rl`.

Each `ScanSpec` covers one `(z, plane, b, alpha)` configuration with `(phi × chi × two_dtheta)` inner frames. The number of scans in the HDF5 master multiplies by `len(z_samples)`. Identification masters with z-scanning therefore have more `/N.1` groups (see Risk 3).

Identification multi follows the same pattern; the secondary-dislocation Monte Carlo loop sits inside the `(z, plane, b, alpha)` loop, unchanged.

### Section 4: `Find_Hg` z-cache (disk)

`forward_model.Find_Hg` gains an optional kwarg `z_offset_um: float = 0.0`:

```python
def Find_Hg(
    dis: float, ndis: int, psize: float, zl_rms: float,
    h: int = -1, k: int = 1, l: int = -1,
    *,
    S: np.ndarray = _S_IDENTITY,
    remount_name: str = "S1",
    z_offset_um: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
```

Cache filename construction:

```python
z_suffix = "" if z_offset_um == 0.0 else f"_z{round(z_offset_um * 1000)}nm"
Fg_path = Fg_dir / f"Fg_{dis}_{psize_nm}nm_{zl_rms_nm}nm_px{Npixels}_sub{Nsub}_remount{remount_name}{z_suffix}.npy"
```

When `z_offset_um == 0.0`, the filename is **identical** to today's — non-z scans hit the same cache. The Round 15 shape guard (`Fg.shape[0] == rl.shape[1]`) still applies.

Inside `Find_Hg`, when `z_offset_um != 0.0`, the rl grid passed to `load_or_generate_Hg` is `Z_shift(z_offset_um)` instead of the module-level `fm.rl`.

`Find_Hg_from_population` similarly gains an optional `rl` kwarg (default to `fm.rl`) so the caller can pass `Z_shift(z)`. No cache for that path; just thread rl through.

### Section 5: HDF5 output

Zero schema change. The `MasterWriter.add_scan` already iterates the `positioners` dict and writes each entry under `/N.1/instrument/positioners/<axis>`. We just populate all 4 axes instead of 2.

`/N.1` attrs:
- `scan_mode`: derived by `derived_mode_name()` — already correct for all combos. The pre-canonized labels cover the `phi+chi+two_dtheta` (`mosa_strain`), `phi+chi+z` (`mosa_layer`), `phi+chi+two_dtheta+z` (`mosa_strain_layer`) combos. Other combos (e.g., pure `two_dtheta` scan) fall through to the `_AXIS_TO_LABEL`-based concatenation.
- `scanned_axes`: `config.scan.scanned_axes()` — already correct.
- `crystal_mode`: unchanged.

Detector data: `(n_frames, H, W)` flat stack where `n_frames = nphi * nchi * ntwodt * nz`. Frame ordering matches the iteration: phi-innermost, z-outermost.

### Section 6: Validation

Drop the `unwired` `ValueError` guard at `pipeline.py:616-622`. No new guards needed:
- `ScanConfig.__post_init__` already validates per-axis `range`/`steps`.
- `derived_mode_name` already produces a sensible label for any combo.
- No interaction is forbidden between axes.

### Section 7: Frame ordering invariant

```
flat_idx = phi_idx + nphi * (chi_idx + nchi * (twodt_idx + ntwodt * z_idx))
```

Phi-innermost (stride 1), z-outermost (stride nphi · nchi · ntwodt). Reasons:
- Backward compatible with v1.2.0 phi/chi-only output (when ntwodt = nz = 1, the formula collapses to the existing `phi_idx + nphi * chi_idx`).
- Matches the cost model: Hg recompute happens once per z, so we want all frames sharing a z to be contiguous in flat_idx (they are — z-outermost gives them stride `nphi*nchi*ntwodt`).

## Risks

1. **Bit-equivalence regression on existing phi/chi scans.** The refactor of `_frame_grid_from_scan` → `_build_scan_frames` could perturb the per-frame ordering. Mitigation: when ntwodt = nz = 1, the new helper must produce the *exact* same `phi_pf` and `chi_pf` arrays as today's `_frame_grid_from_scan`. Add a regression test that compares the outputs at degenerate axis counts.

2. **Memory footprint of Hg dictionary.** `z_to_Hg: dict[float, np.ndarray]` keeps all per-z Hg arrays in memory simultaneously inside one `_iterate_simulation_frames` call. At default config that's ~720 MB per z. With e.g. 10 z values, 7.2 GB — out of laptop reach. Mitigation: pop the Hg array after the last frame at that z is yielded. Sketch:

   ```python
   z_pf = frames.z_pf
   for k in range(frames.n_frames):
       z = float(z_pf[k])
       Hg = z_to_Hg.setdefault(z, Hg_provider(z))
       yield (k, Hg, ...)
       if k == frames.n_frames - 1 or float(z_pf[k+1]) != z:
           del z_to_Hg[z]  # last frame at this z; free Hg before next z's recompute
   ```

   Because z is outermost in our ordering, the "last frame at this z" check is just "the next frame's z differs." This keeps at most one Hg in memory at a time during the inner walk.

3. **Identification scan-count explosion.** When `[scan.z]` is set in identification mode, each `(plane, b, alpha)` triple now produces `nz` `ScanSpec`s instead of 1, so the master HDF5 grows from O(angles · planes · b · 1) to O(angles · planes · b · nz) `/N.1` entries. Sina's typical identification configs (≤ 864 frames) stay well within HDF5 limits even with `nz = 10` (8640 entries). Document the scaling in `docs/output-format.md`.

4. **Cache dir growth.** `_z{nm}nm` filename suffix means the Fg cache dir grows linearly with distinct z values across all runs (wall mode only). Disk is cheap; the dir is human-pruneable. No automated eviction.

5. **`Find_Hg_from_population` rl kwarg backward compat.** Adding an optional `rl=None` kwarg is back-compat (callers passing no rl continue to use module-level `fm.rl`). Confirmed no test or external code passes `rl` today.

## Backward compatibility

- Phi/chi-only scans (the v1.2.0 default config): **bit-equivalent output**. The new helpers degenerate to the same flat-frame ordering when ntwodt = nz = 1. Detector data, positioners, and HDF5 attrs all unchanged.
- TOML schema: unchanged. The same `[scan.<axis>]` blocks that triggered the `ValueError` today now produce real output.
- Fg cache: existing cached files (no `_z…` suffix) are still picked up for z=0 scans (the new suffix is only added when `z_offset_um != 0.0`).
- CLI signatures: unchanged.
- HDF5 schema: unchanged.

The only externally-visible change is the lifted `ValueError`. Per the `dfxm_no_backcompat_constraint` (Sina is sole user since cleanup started), even minor semantic shifts are acceptable, but in practice this design preserves them anyway.

## Testing

Unit tests:

- `tests/test_scan_frames.py` (new file):
  - `_build_scan_frames` with 0, 1, 2, 3, 4 scanned axes.
  - Cartesian product flattening: `n_frames == nphi * nchi * ntwodt * nz`.
  - Frame ordering: phi-innermost, z-outermost. Explicit per-frame index ↔ (i_phi, i_chi, i_twodt, i_z) round-trip.
  - Bit-equivalence with `_frame_grid_from_scan` at ntwodt = nz = 1.

- `tests/test_find_hg_z_cache.py` (new file):
  - Cache filename gains `_z{nm}nm` only when `z_offset_um != 0.0`.
  - `z_offset_um=0.0` hits the same file as today's filename.
  - Shape guard fires when stale cache is loaded.

- `tests/test_pipeline_scan_modes.py` (extend):
  - End-to-end `run_simulation` with `[scan.two_dtheta].range/steps` only — produces `mosa_strain`-labelled output with correct positioner array.
  - End-to-end `run_simulation` with `[scan.z].range/steps` only — produces `mosa_layer` (actually just `layer`) output; Hg recompute confirmed per z (via fixture monkeypatching).
  - End-to-end `mosa_strain_layer` with 2×2×2×2 = 16 frames — confirm positioner dict has all 4 arrays + flat detector stack shape.

- `tests/test_identification_scan_modes.py` (extend):
  - Same axis combinations on `mode = "single"` and `mode = "multi"`.

Bit-equivalence guard:

- `tests/test_pipeline_scan_modes.py::test_v120_phi_chi_output_bit_equivalent` — run a 5×5 phi/chi simulation, compare detector + positioners + attrs against a checked-in golden from v1.2.0. Catches accidental ordering changes in the helper refactor.

Manual:

- Reuse the v1.2.0 smoke configs at `C:\Users\borgi\tmp\v120_smoke_configs\` to spot-check 4-axis output in silx + the h5py dump. No darling test (it's externally blind per `followups_darling_external_link_blind.md`).

## Open questions

- Cache filename precision for `z_offset_um`: `round(z * 1000)` (1-nm precision) — sufficient for any practical sample-depth scan. Confirm at implementation time if Sina ever uses sub-nm z steps.
- `_compute_frame` signature change ripples to any external callers? Grep confirms only the pipeline path uses it; no tests monkey-patch it. Safe.
- Should `_iterate_simulation_frames` live in `pipeline.py` or `forward_model.py`? Putting it in `pipeline.py` keeps the orchestration close to `run_simulation`; putting it in `forward_model.py` would let identification + forward share it more naturally. Defer to writing-plans phase.

## Touch list (estimated)

| File | Change |
|---|---|
| `src/dfxm_geo/pipeline.py` | `ScanFrames` + `_build_scan_frames` + `_build_scan_frames_at_z`; replace `_frame_grid_from_scan`; extend `_scan_frames_args` (now 5-tuples with two_dtheta) + `_positioners_for_scan` (4 axes); new `_iterate_simulation_frames` orchestrator; lift ValueError in `run_simulation`; thread z into `_iter_identification_single`/`_multi` via the z-outer loop pattern. |
| `src/dfxm_geo/direct_space/forward_model.py` | `Find_Hg`: new `z_offset_um` kwarg + cache filename extension; internal `rl=Z_shift(z)` plumbing. `Find_Hg_from_population`: new `rl` kwarg. `_compute_frame`: accept 5-tuple. `_compute_and_write_detector_file_parallel`: pass 5-tuples through. |
| `src/dfxm_geo/io/hdf5.py` | `write_simulation_h5` signature: drop phi/chi range+steps; accept `ScanFrames`. Adjust internal positioner construction. |
| `tests/` | New: `test_scan_frames.py`, `test_find_hg_z_cache.py`. Extend: `test_pipeline_scan_modes.py`, `test_identification_scan_modes.py`. |
| `docs/output-format.md` | Document 4-axis positioners + identification scan-count scaling. |
| `configs/` | Optional: ship a small `variants/forward_strain_scan.toml` and `variants/forward_z_scan.toml` showcasing the new axes. |
