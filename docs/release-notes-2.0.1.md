# DFXM v2.0.1 — Patch: population strain-path units fix

Released: 2026-05-23.

## Bug fix (correctness): `rl` units in the population strain path

`Find_Hg_from_population` passed the lab-frame ray grid `rl` to
`Fd_find_multi_dislocs_mixed` in **metres**, but the edge/screw
displacement-gradient formula in `Fd_find_mixed` expects **micrometres**
(the Burgers magnitude `b = 2.862e-4` is in µm). The conversion that the
wall path applies via `Fd_find(rl * 1e6, ...)` — and that the reference
`disloc_identify.py` applies via `Fd_find_mixed(rl * 1e6, ...)` — was
missing.

**Symptom:** for `mode="centered"`, `mode="random_dislocations"`, and
identification `mode="multi"`, the displacement-gradient field was ~10⁶×
too large, so weak-beam contrast collapsed onto the singular dislocation
core (a small blob at the origin) instead of the extended dislocation
field. **`mode="wall"` was unaffected** — it uses the legacy `Fd_find`
path with the correct `* 1e6`, which is why the published IUCrJ-2024 wall
results and the `Fd_find_smoke.npy` golden never exhibited the bug. Latent
since v1.2.0 (sub-project C); first user-visible in v2.0.0.

**Fix:**
- `Find_Hg_from_population` now calls
  `Fd_find_multi_dislocs_mixed(rl_eff * 1e6, ...)`.
- `Fd_find_mixed`'s `position_lab_um` offset is subtracted in micrometres
  (was metres) to stay consistent with `rl` now in micrometres.

## Validation

Cross-checked against the original reference implementation
(`HFP_Book_Ch`) with all parameters matched (hkl=(-1,1,-1), 17 keV, θ,
Npixels=510, Nsub=1, b=2.862e-4, ν=0.334, Ud convention):

- Reciprocal-space MC kernels match to ~4 significant figures.
- `Fd_find_mixed` produces bit-identical `Fg` (rotation 0°/30°/90°).
- Wall `Fd_find` bit-identical (ndis 1/10/50); 3-dislocation superposition
  bit-identical.
- `disloc_identify` Ud construction matches to 2e-16.
- The single-dislocation forward image reproduces the reference's
  lobe/fringe pattern.

## Tests

- New `tests/test_population_rl_units.py`: asserts the centered-mode field
  matches a micrometre oracle and is physically scaled (fails on the v2.0.0
  code, passes here).
- `tests/test_dislocations_mixed.py::test_..._position_offset_shifts_singularity`
  updated from metre-scale to micrometre-scale `rl` (it had encoded the
  buggy offset convention).

Suite: 500 passed / 10 deselected / 2 xfailed; mypy clean.

## Upgrade

```bash
pip install --upgrade dfxm-geo==2.0.1
```

No API or config changes. If you ran `centered`, `random_dislocations`, or
identification `multi` modes under v1.2.0–v2.0.0, regenerate those outputs
— their contrast was incorrect.
