# Lift the `[scan.two_dtheta]` Eager Guard in Identification (v1.3.0-B / v1.3.1)

**Goal:** Remove the `ValueError` that today rejects `[scan.two_dtheta]` in `run_identification`, so identification single / multi / zscan configs can scan two_dtheta as a within-scan frame axis (matching the v1.3.0-A forward-mode architecture).

**Spec date:** 2026-05-22

**Tracking:** v1.3.1 (point release on top of v1.3.0).

---

## Background

v1.3.0-A wired `[scan.two_dtheta]` into the forward path (`run_simulation`) and `[scan.z]` into both forward and identification single/multi. The eager guard in `run_identification` (lines 1730-1735 of `src/dfxm_geo/pipeline.py`) was narrowed from rejecting both `("two_dtheta", "z")` to rejecting only `"two_dtheta"`:

```python
if config.scan.is_scanned("two_dtheta"):
    raise ValueError(
        "scan axis two_dtheta is configured but not yet wired into "
        "identification. For now, set range+steps only on "
        "[scan.phi], [scan.chi], and/or [scan.z]."
    )
```

The justification at the time was: "two_dtheta lifting is tracked as a future follow-up." This spec is that follow-up.

## Architecture finding: the plumbing already works

A code audit during v1.3.0-A handoff verification confirmed that **the entire frame-generation pipeline for `two_dtheta` is already in place** in identification:

1. `_scan_frames_args(Hg, frames_at_z, scan)` in `pipeline.py` builds args from a `ScanFrames` (which carries `two_dtheta_pf` per frame), emitting 5-tuples `(frame_idx, Hg, phi, chi, two_dtheta)`.
2. `_compute_frame(args)` in `io/hdf5.py` unpacks the 5-tuple and calls `_fm.forward(Hg, phi=phi, chi=chi, TwoDeltaTheta=two_dtheta)` per frame.
3. `_iter_identification_single` / `_iter_identification_multi` already pass `frames_at_z = _build_scan_frames_at_z(config.scan, z_float)` through `_scan_frames_args`. `frames_at_z.two_dtheta_pf` walks the configured two_dtheta values transparently.
4. `_iter_identification_zscan` uses the same `_build_scan_frames_at_z` + `_scan_frames_args` pattern internally — also already two_dtheta-ready.
5. `forward()` accepts `TwoDeltaTheta: float = 0` (parameter at `forward_model.py:420`) and shifts the Bragg angle internally via the rotation matrix at line 464. `Hg` is independent of `TwoDeltaTheta` by construction.

The eager guard is therefore the **sole** blocker.

## Design

### Within-scan, not between-scan

`two_dtheta` is a within-scan frame axis in identification — multiplies `n_frames` per `/N.1/`, does NOT multiply the number of scan groups. Justification:

- **Physics:** `Hg` is the deformation-gradient tensor at the dislocation; `TwoDeltaTheta` is a Bragg-angle offset applied by `forward()` via a rotation about the Bragg vector. The two are orthogonal: scanning `two_dtheta` does not require recomputing `Hg`.
- **Consistency with forward:** In `run_simulation`, all four axes (phi, chi, two_dtheta, z) walk inside a single `/1.1/` entry. The same flat-trajectory shape applies to identification.
- **Avoiding wasted compute:** Making two_dtheta between-scan (outer loop like z) would mean `Fd_find_mixed` / `Fd_find_multi_dislocs_mixed` calls per two_dtheta value — but each would produce the same `Hg`. Pure waste.
- **Compatibility with z:** A user scanning both `[scan.two_dtheta]` and `[scan.z]` in identification gets `n_z × n_planes × n_b × n_alpha` scan groups (z multiplies, two_dtheta does not), each with `phi_steps × chi_steps × two_dtheta_steps` frames.

### Tasks

1. **Remove the eager guard.** Delete the `if config.scan.is_scanned("two_dtheta")` block in `run_identification` (`pipeline.py:1730-1735`). Add a smoke test verifying that a single-mode config with `[scan.two_dtheta]` configured runs without raising.

2. **Test: single + two_dtheta scan count.** Add `test_single_with_two_dtheta_scanned_keeps_one_scan_per_plane` to `tests/test_identification_scan_modes.py`. Verifies:
   - 1 scan group per `(plane, b, alpha)` (NOT multiplied by `n_two_dtheta`).
   - `n_frames` per scan = `n_two_dtheta` (with phi/chi fixed).
   - `/N.1/instrument/positioners/two_dtheta` is a length-`n_two_dtheta` array.

3. **Test: multi + two_dtheta.** Add `test_multi_with_two_dtheta_scanned` to the same file. Verifies:
   - 1 scan group per `(z, mc_sample)` (NOT multiplied by `n_two_dtheta`).
   - `n_frames` per scan = `n_two_dtheta` (with phi/chi fixed).

4. **Test: 4-axis identification.** Add `test_single_with_all_four_axes_scanned` (single mode, phi+chi+two_dtheta+z all scanned). Verifies:
   - `n_z` scan groups (z is the only between-scan axis; phi/chi/two_dtheta are within-scan).
   - `n_frames` per scan = `n_phi × n_chi × n_two_dtheta`.

5. **Remove stale config-changes test.** The parametrized `test_run_identification_eager_guards_unwired_axes` in `tests/test_identification_config_changes.py` is `@pytest.mark.parametrize("axis_name", ["two_dtheta"])` — once the guard is gone, this test has nothing to assert. Delete it (the test function entirely) along with the dangling note. The eager-guard story is over.

6. **Docs.** Update `docs/output-format.md`:
   - In the "Identification scan-count scaling" paragraph (the one updated by v1.3.0-A), append a clarifying sentence: `two_dtheta` is within-scan in identification (multiplies `n_frames`-per-scan, not the scan count). Distinguishes it from `z`.
   - In the `N_frames` table, no row changes — identification was already documented per-mode.

7. **Release.** Bump `pyproject.toml` 1.3.0 → 1.3.1; rename `tests/test_version_is_1_3_0.py` → `tests/test_version_is_1_3_1.py` and update the literal. Commit. Tag `v1.3.1` (annotated). Merge feature branch via `--no-ff` to main. Push main + tag (fires `publish.yml`).

### File touch list

| Path | Change |
|---|---|
| `src/dfxm_geo/pipeline.py` | Delete guard block (`pipeline.py:1727-1735` or thereabouts; verify line numbers at task time). |
| `tests/test_identification_scan_modes.py` | Add 3 tests (single, multi, 4-axis). |
| `tests/test_identification_config_changes.py` | Delete `test_run_identification_eager_guards_unwired_axes` + its parametrize + the explanatory comment. |
| `docs/output-format.md` | Append clarifying sentence to the identification scan-count paragraph. |
| `pyproject.toml` | 1.3.0 → 1.3.1. |
| `tests/test_version_is_1_3_0.py` → `tests/test_version_is_1_3_1.py` | Rename + update literal. |

No new files. No new helpers. No production-code changes beyond the guard removal.

### Tech stack

Same as v1.3.0-A: numpy, h5py, existing `dfxm_geo` machinery.

### Verification gates

- pytest full suite: must pass with the same 2 pre-existing xfails as v1.3.0-A; no new failures.
- mypy `src/dfxm_geo/`: 0 errors.
- New tests must actually exercise the two_dtheta path (assert per-frame two_dtheta values, not just shapes).

### Risks

- **R1 (forward_model.forward correctness):** `forward(Hg, phi=phi, chi=chi, TwoDeltaTheta=two_dtheta)` must produce a correctly-shifted detector image. This is a v1.0+ path with thorough forward-mode coverage; not a new risk. Mitigation: existing forward-mode two_dtheta tests stay green.
- **R2 (scan-mode label):** With phi+chi+two_dtheta scanned in identification, `derived_mode_name()` returns `mosa_strain` — same string the forward path produces. Document that this label now also appears under identification masters.
- **R3 (memory):** Each `/N.1/` now has `n_frames = phi × chi × two_dtheta` instead of `phi × chi`. For typical configs (phi=61, chi=61, two_dtheta=5) this is 5× the per-scan stack — confirm pytest-of-borgi headroom before the full-suite run. Mitigation: keep new test scan counts small (e.g. 2×2×2 not 61×61×5).

### Out of scope

- Identification using a per-frame `Fd_find_mixed` call (would let `Hg` vary by two_dtheta). Physically unmotivated; not pursued.
- Frame-ordering changes. Phi-innermost, z-outermost stays.
- `_PRE_CANONIZED_MODE_NAMES` additions. The existing `mosa_strain` / `mosa_strain_layer` / single-axis fallbacks already cover every identification configuration.
- Silx smoke. Deferred to user.

## Acceptance criteria

1. A user config with `[scan.two_dtheta]` (and any combination of phi / chi / z) in identification single/multi mode runs to completion and produces a valid master HDF5.
2. The number of scan groups in the master HDF5 is governed by `n_z × n_planes × n_b × n_alpha` (single) or `n_z × n_samples` (multi). The `n_two_dtheta` factor multiplies `n_frames`-per-scan, not the scan count.
3. The full test suite passes. `tests/test_identification_config_changes.py::test_run_identification_eager_guards_unwired_axes` is removed (it has nothing left to assert).
4. mypy clean.
5. Tag `v1.3.1` exists on origin/main with the annotated message describing this change.
