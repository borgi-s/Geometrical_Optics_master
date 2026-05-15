# Purdue_Paper Port (Design)

**Date:** 2026-05-14
**Branch under port:** `origin/Purdue_Paper` (one commit, `cb254d2`)
**Target:** `cleanup/main-modernization` (HEAD `4f2e4c4` at port start)
**Status:** Brainstorming complete; awaiting user review of this spec.

**Related work:**
- Round 16: ported `origin/dislocation_identification` as `dfxm-identify` with `mode={single, multi}` and a {111}-family slip-system sweep. Conceptually distinct from this port — that work selects which slip system to *generate* dislocations on; this port rotates the already-generated sample as if it were physically remounted.
- Round 17: ported `origin/ESRF_DTU` (z-scan mode, `Z_shift(offset_um)`, sample-frame viz). Round 17's `Z_shift` is the canonical path for z-offsetting the rl grid — this port reuses it rather than introducing a new `offset` argument on `Find_Hg` (see R1 in Risks).

## Goal

Port the canonical content of `origin/Purdue_Paper` into the cleanup. The substantive addition is a **sample-remount rotation matrix `S`** inserted between the sample frame and the crystal frame in the rotation chain used by `Fd_find` / `Fd_find_mixed` / `load_or_generate_Hg`. Operationally: `S` rotates the entire sample relative to the goniometer, simulating physical remounting at a symmetry-equivalent orientation. The branch ships four constants `S1, S2, S3, S4`: S1 is identity (no remount); S2, S3, S4 are three specific cubic-symmetry proper rotations from the Purdue 2024 paper that map the reference mount to other symmetry-equivalent sample orientations. (Their numerical traces are not all equal — S2 and S4 have trace 1/3 (≈109.47° rotation); S3 has trace 5/3 (≈70.53° rotation) — so they are not three rotations about a single axis; they are independent group elements selected for the paper's specific remount scenarios. Cleanup ports them verbatim and does not re-derive their geometric meaning.)

## Non-goals

- **Not** porting `Find_Hg`'s `offset` kwarg from the Purdue branch. Round 17's `Z_shift(offset_um)` is the documented path for z-offsetting the `rl` grid; adding a second mechanism would split the workflow.
- **Not** porting the 450-line paper-figure plotting code from the Purdue `init_forward.py` (3×3 H_ij component grids per depth slice etc). That goes verbatim into `legacy/init_forward_purdue.py` with a header marking it as a frozen, unmaintained reference. Not lint-clean, not tested, not imported by anything.
- **Not** porting the externally-precomputed Hg loading workflow (`Hg = np.load('Test_S1_Hg_rg.npy')` followed by `Hg_rot = Theta @ Us.T @ Hg @ Us @ Theta.T`). That's a CDD-coupling / ML-data pathway tangential to the remount port; defer until a real use case surfaces.
- **Not** wiring `sample_remount` into `dfxm-identify`. That mode already sweeps slip planes via `_ALL_111_PLANES` (Round 16) and depth via `IdentificationZScanConfig` (Round 17). Adding remount there would duplicate the same physical axis (a sample remount under [111]-rotation is symmetry-equivalent to a slip-plane swap from the detector's POV). YAGNI; revisit if a real workflow needs it.
- **Not** porting the Purdue branch's scan / instrument parameter changes (`psize=50nm`, `Npixels=410`, `Nsub=1`, `phi_range=0.0004 rad`, `chi_range=0.001 rad`, `Resq_i_20231107_0952.pkl`). All of these are already TOML-configurable via `configs/default.toml` and the `[forward_model]` / `[scan]` / `[io]` sections.
- **Not** introducing free-form rotation matrix specification (Euler angles, raw 3×3 list) in the TOML. Per user call (2026-05-14), the surface is the four-value enum `{S1, S2, S3, S4}` only. Add free-form later if a real need surfaces.

## What gets ported

| Branch artefact | Lands as | Notes |
|---|---|---|
| `S1, S2, S3, S4` constants (defined inline in Purdue `direct_space/forward_model.py`) | `dfxm_geo.crystal.remount.{S1,S2,S3,S4}` + `SAMPLE_REMOUNT_OPTIONS` dict | New module. S1 = identity; S2/S3/S4 = 120°-around-[111] rotations. Verified as proper rotations (`det=1`, `R.T @ R = I`) by construction. |
| `Fd_find(..., S, ...)` signature change | `Fd_find(..., S=np.identity(3), ...)` in `dfxm_geo.crystal.dislocations` | New kwarg, default identity → backward-compatible. Applied in the rotation chain at `rs → rgon = S.T @ rs → rc = Us.T @ rgon`. |
| `Fd_find_mixed(..., S, ...)` (implied — Purdue branch predates Round 16; mirror the change here) | Same kwarg shape on `Fd_find_mixed` and `Fd_find_multi_dislocs_mixed` | Same math placement (between Theta and Us.T). |
| `load_or_generate_Hg(..., S, ...)` signature change | `load_or_generate_Hg(..., S=np.identity(3), ...)` in `dfxm_geo.io.strain_cache` | S threaded through to `Fd_find`. Fg cache filename gains `_remount{name}` suffix so different S's don't share cache entries. |
| `Find_Hg(..., S, ...)` signature change | `Find_Hg(..., S=np.identity(3), ...)` in `dfxm_geo.direct_space.forward_model` | Resolves the S kwarg, threads it down to `load_or_generate_Hg`. |
| Purdue `init_forward.py` (paper-figure 3×3 viz, custom Hg load, etc.) | `legacy/init_forward_purdue.py` (verbatim + header) | Not maintained, not tested. Header notes "frozen reference for Purdue paper figures; do not import." |

## Architecture

### New module: `src/dfxm_geo/crystal/remount.py`

```python
"""Sample-remount rotation matrices (Purdue 2024 paper).

S_i is applied between the sample frame (after Theta) and the crystal frame
(before Us.T) in the Fd_find rotation chain. Operationally: S rotates the
entire sample relative to the goniometer, simulating physical remounting.
S1 = identity (no remount); S2, S3, S4 are three specific cubic-symmetry
proper rotations from the Purdue 2024 paper. They are ported verbatim from
the branch source; values are not re-derived from a symmetry argument here.
"""

import numpy as np

S1: np.ndarray = np.identity(3)

# S2, S3, S4 from Purdue branch direct_space/forward_model.py
S2: np.ndarray = np.array([[1/3, -2/3, -2/3],
                            [2/3, -1/3,  2/3],
                            [-2/3, -2/3,  1/3]])

S3: np.ndarray = np.array([[1/3, -2/3,  2/3],
                            [2/3,  2/3,  1/3],
                            [-2/3,  1/3,  2/3]])

S4: np.ndarray = np.array([[1/3,  2/3,  2/3],
                            [2/3,  1/3, -2/3],
                            [-2/3, 2/3, -1/3]])

SAMPLE_REMOUNT_OPTIONS: dict[str, np.ndarray] = {
    "S1": S1,
    "S2": S2,
    "S3": S3,
    "S4": S4,
}
```

Rationale for a new module rather than appending to `crystal/rotations.py`: `rotations.py` holds defect-physics helpers (`rotatedU`, `fast_inverse2`, Round 17's `rotate_matrix_z_axis`, `is_valid_rotation_matrix`); `remount.py` is conceptually about lab-frame sample mounting, a different abstraction layer. Keeping them separate avoids a misleading import.

### Modifications to `src/dfxm_geo/crystal/dislocations.py`

Three function signatures gain `S: np.ndarray = np.identity(3)`:

```python
def Fd_find(rl, Ud, Us, Theta, dis=1, ndis=1, b=2.862e-4, ny=0.334,
            misorientation=False, t_vec=None, *,
            S: np.ndarray = np.identity(3)) -> np.ndarray:
    ...
    rs = Theta @ rl       # sample frame
    rgon = S.T @ rs       # NEW: goniometer frame (after remount)
    rc = Us.T @ rgon      # crystal frame
    rd = Ud.T @ rc        # dislocation frame
    ...

def Fd_find_mixed(rl, Ud, Us, Theta, spec: MixedDislocSpec, ndis=1, *,
                  S: np.ndarray = np.identity(3)) -> np.ndarray: ...

def Fd_find_multi_dislocs_mixed(rl, Ud, Us, Theta, specs, *,
                                 S: np.ndarray = np.identity(3)) -> np.ndarray: ...
```

S is applied in the Python wrapper before the numba JIT helper `_accumulate_bipolar_walls` receives `rd`. No JIT-signature change; no recompile.

### Modifications to `src/dfxm_geo/io/strain_cache.py`

```python
def load_or_generate_Hg(rl, Ud, Us, Theta, dis, ndis, file_path=None, *,
                         S: np.ndarray = np.identity(3),
                         remount_name: str = "S1") -> np.ndarray:
    ...
```

Cache filename format (Round 15 key + remount):
```
Fg_{dis}_{psize_nm}nm_{zl_rms_nm}nm_px{Npixels}_sub{Nsub}_remount{remount_name}.npy
```

Decision: cache filename **always** includes the `_remount{name}` suffix, even at S=I (`remount_name="S1"`). Two cache namespaces would be uglier than one extra suffix. Pre-port caches need a one-time regen — cheap, one-shot per `(dis, Nsub, Npixels)` combo.

### Modifications to `src/dfxm_geo/direct_space/forward_model.py`

```python
def Find_Hg(dis, ndis, psize, zl_rms, I=np.identity(3),
             h=-1, k=1, l=-1, *,
             S: np.ndarray = np.identity(3),
             remount_name: str = "S1") -> tuple[np.ndarray, np.ndarray]:
    ...
```

Resolved S and its name are passed straight through to `load_or_generate_Hg`. No other changes to `Find_Hg`'s body.

### Modifications to `src/dfxm_geo/pipeline.py`

```python
@dataclass(frozen=True)
class CrystalConfig:
    dis: float = 4.0
    ndis: int = 151
    sample_remount: str = "S1"   # NEW
```

`load_simulation_config()` validates `cfg.crystal.sample_remount in SAMPLE_REMOUNT_OPTIONS` and raises `ValueError("sample_remount must be one of: S1, S2, S3, S4")` on bad input.

`run_simulation()` resolves `S = SAMPLE_REMOUNT_OPTIONS[cfg.crystal.sample_remount]` once at the top, then threads `S=S, remount_name=cfg.crystal.sample_remount` through `_ensure_kernel_loaded` / `Find_Hg`.

### TOML schema

`configs/default.toml`:
```toml
[crystal]
dis = 4.0
ndis = 151
sample_remount = "S1"   # NEW — one of "S1", "S2", "S3", "S4". Default "S1" (identity).
```

`configs/variants/sample_remount_S2.toml` (new): identical to default plus `sample_remount = "S2"`. Smallest reproducer for "same defect, remounted" and the seed for the CLI integration smoke test.

## Output layout

No change. `run_simulation` writes the same image stack and analysis products it does today; the only behavioral difference is the strain field used to produce them. The output dir name is whatever the TOML specifies; we don't auto-suffix with the remount name (keep paper-figure runs orthogonal — users can put each S in its own output dir explicitly).

## Math placement of S (sanity check vs the Purdue branch)

Purdue branch `functions.py` `Fd_find`:
```python
rs = Theta @ rl              # sample
rgon = S.T @ rs              # goniometer (introduced by Purdue)
rc = Us.T @ rgon             # crystal/grain
rd = Ud.T @ rc               # dislocation
```

Cleanup mirrors this exactly. At S=I, `rgon = rs` and the chain collapses to the existing `rc = Us.T @ Theta @ rl`. Backward-compat is bit-exact (pinned by `test_Fd_find_S_identity_matches_existing_golden`).

## Testing

### `tests/test_remount.py` (new, 4 tests)

- `test_constants_are_proper_rotations` — each `S_i` satisfies `S.T @ S ≈ I` and `det(S) ≈ +1` (within `1e-10`).
- `test_S1_is_identity` — `np.allclose(S1, np.identity(3))`.
- `test_S2_S3_S4_match_purdue_source_verbatim` — assert each constant equals the literal 3×3 array copied from `origin/Purdue_Paper` `direct_space/forward_model.py` (regression-style; catches accidental edits to the constants).
- `test_sample_remount_options_map_is_complete` — keys are exactly `{"S1","S2","S3","S4"}`; each value is the corresponding constant (not a copy).

### Extensions to `tests/test_dislocations.py` (~3 tests)

- `test_Fd_find_S_identity_matches_existing_golden` — `Fd_find(..., S=np.identity(3))` is bit-equal to `tests/data/golden/Fd_find_smoke.npy`. The backward-compat pin.
- `test_Fd_find_distinct_S_yields_distinct_output` — S1 output ≠ S2 output on a non-trivial rl grid. Asserts S actually wires into the math.
- `test_Fd_find_S_kwarg_is_keyword_only` — calling `Fd_find(rl, Ud, Us, Theta, dis, ndis, S=np.identity(3))` works; positional `Fd_find(rl, Ud, Us, Theta, dis, ndis, np.identity(3))` is treated as `b=...` (existing positional) — asserts the kwarg-only marker is honoured.

### Extensions to `tests/test_dislocations_mixed.py` (~2 tests)

- `test_Fd_find_mixed_S_identity_matches_no_S` — mixed-case backward-compat pin.
- `test_Fd_find_mixed_distinct_S_yields_distinct_output` — same shape as above.

### Extensions to `tests/test_pipeline.py` (~3 tests)

- `test_crystal_config_sample_remount_defaults_to_S1` — TOML without the field still parses; `cfg.crystal.sample_remount == "S1"`.
- `test_load_simulation_config_rejects_unknown_sample_remount` — `sample_remount = "S99"` raises `ValueError` mentioning the four valid names.
- `test_run_simulation_passes_resolved_S_into_find_hg` — monkeypatch `Find_Hg`, run with `sample_remount = "S2"`, assert `S` kwarg arrives as `SAMPLE_REMOUNT_OPTIONS["S2"]` and `remount_name == "S2"`.

### Extensions to `tests/test_io.py` (~1 test)

- `test_load_or_generate_Hg_keys_cache_by_remount_name` — call with S1 then S2 (same `dis,ndis,psize,zl_rms,Npixels,Nsub`); assert two distinct cache files appear with the expected `_remountS1` / `_remountS2` suffixes, and a third call with S1 hits cache (no recompute). Mirrors the Round 15 shape-mismatch test pattern.

### CLI smoke (~1 test)

- `test_dfxm_forward_with_sample_remount_S2_runs_end_to_end` — subprocess call to `dfxm-forward` with `configs/variants/sample_remount_S2.toml`. Uses the real `Resq_i_*.pkl` kernel; skip with clear marker if absent (same pattern as Round 16's CLI smoke).

**Total: ~14 new tests** (4 remount + 3 dislocations + 2 mixed + 3 pipeline + 1 io + 1 CLI). No bench updates needed (Fg generation per-S is one-shot and cached).

## Risks

### R1 — `Z_shift` over `offset` (decided)

Purdue's `Find_Hg(offset=...)` and Round 17's `Z_shift(offset_um)` are operationally the same. We do **not** add `offset` to `Find_Hg`; `Z_shift` is the documented path for z-offsetting the rl grid. Doc'd in `pipeline.py` module docstring.

### R2 — Fg cache key always includes `_remount{name}` (decided)

Even at S=I (`remount_name="S1"`), the cache filename includes the suffix. Two cache namespaces would be uglier than one extra suffix. Pre-port caches need a one-time regen — cheap.

### R3 — Backward-compat is bit-exact at S=I (decided)

Pinned by `test_Fd_find_S_identity_matches_existing_golden` against the existing golden `tests/data/golden/Fd_find_smoke.npy`.

### R4 — Reflection lock unchanged (decided)

The existing reflection lock (`dfxm-forward` bound to `(-1, 1, -1)`) is preserved. S is applied to the sample frame *before* the slip-system geometry encoded in `Us`, so the S/reflection interaction is the same one the Purdue paper exercised. We do not separately verify that the four S matrices produce a "symmetry-equivalent" image for the cleanup's specific reflection — that is a paper-level physics claim, not a code-level invariant, and is out of scope for this port. The bit-exact-at-S=I test pins backward compatibility; distinctness tests pin the wiring.

### R5 — Numba JIT recompile risk (mitigated)

S is applied in the Python wrapper before `rd` reaches `_accumulate_bipolar_walls`. The JIT function's signature does not change; no recompile per remount choice.

### R6 — Cache name collision when user changes other params (not in scope)

`Z_shift`-ed runs currently share Fg cache files with non-shifted runs because the cache key doesn't include z-offset. Pre-existing latent issue, not introduced by this port. Filed as a follow-up; mentioned here for completeness.

## Implementation order (preview for writing-plans handoff)

1. `src/dfxm_geo/crystal/remount.py` + `tests/test_remount.py`.
2. `Fd_find` / `Fd_find_mixed` / `Fd_find_multi_dislocs_mixed` signature → S kwarg + bit-equal-at-I tests + distinct-output tests.
3. `load_or_generate_Hg` signature + cache-key change + `test_load_or_generate_Hg_keys_cache_by_remount_name`.
4. `Find_Hg` signature + smoke test (no Hg, just signature plumbing).
5. `CrystalConfig.sample_remount` + validation + the three `test_pipeline.py` tests.
6. `run_simulation` wiring + monkeypatch assertion test.
7. `configs/default.toml` updated; new `configs/variants/sample_remount_S2.toml`.
8. CLI integration smoke (kernel-gated, same pattern as Round 16).
9. `legacy/init_forward_purdue.py` — verbatim copy of Purdue branch `init_forward.py` + header.
10. `docs/architecture.md` — note on the goniometer frame in the rotation chain.
11. `docs/physics.md` — note on the sample-remount symmetry and S1-S4 constants.

## Out-of-scope follow-ups filed

- (FU1) External Hg loading (`np.load(...)` + post-load `Us.T @ Hg @ Us` rotation) for CDD coupling / ML data. Defer until a use case lands.
- (FU2) `Z_shift` + Fg cache key interaction (Round 17 z-offset not in the cache key). Pre-existing.
- (FU3) Free-form rotation matrix / Euler angles in TOML. YAGNI; revisit if enum coverage is insufficient.
- (FU4) `sample_remount` in `dfxm-identify`. Out of scope per Section 2; revisit if a real workflow needs it.
