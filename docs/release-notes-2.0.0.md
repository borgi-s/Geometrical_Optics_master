# DFXM v2.0.0 — Default config flip to "simple"

Released: 2026-05-23.

## Headline: BREAKING CHANGE — `WallCrystalConfig` requires explicit fields

`WallCrystalConfig` no longer ships with the IUCrJ-2024 publication-grade
defaults (`dis=4.0`, `ndis=151`, `sample_remount="S1"`). Calling
`WallCrystalConfig()` bare — or with fewer than three keyword arguments —
now raises `TypeError`.

**Migration:** specify all three fields explicitly.

```python
# Before (v1.3.1 and earlier):
from dfxm_geo.pipeline import WallCrystalConfig
cfg = WallCrystalConfig()                          # silently IUCrJ-default

# After (v2.0.0+):
from dfxm_geo.pipeline import WallCrystalConfig
cfg = WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S1")
```

**TOML configs are unaffected.** All wall-mode configs shipped in
`configs/variants/` already specify `dis`/`ndis`/`sample_remount`
explicitly.

## New: empty TOML now produces a valid run

A literally-empty `.toml` file is now a valid input to both
`dfxm-forward` and `dfxm-identify`. The empty case resolves to:

- **`dfxm-forward`**: `mode="centered"` single dislocation at origin
  with canonical FCC primary slip system `(b, n, t) = ((1, 0, -1),
  (1, 1, 1), (1, -2, 1))`, scan trajectory `"single"` (no axes scanned,
  one detector image), Al 111 reflection at 17 keV.
- **`dfxm-identify`**: `mode="single"` + canonical {111} hypothesis sweep
  + default noise (Poisson on, `intensity_scale=7.0`) + Al 111 reciprocal.

Every dataclass in the config hierarchy that previously had no default
now does: `CenteredCrystalConfig`, `ReciprocalConfig`, `CrystalConfig`,
`SimulationConfig`, `IdentificationCrystalConfig`, `IdentificationConfig`.

## New: partial `[reciprocal]` overrides

`ReciprocalConfig.from_dict` now accepts partial dicts. The following
TOML works (keeps default `hkl`, overrides only `keV`):

```toml
[reciprocal]
keV = 21.0
```

Symmetric for `hkl`-only.

## New: identification `mode` is optional

`load_identification_config` no longer requires a top-level `mode = "..."`
field. Missing → defaults to `"single"`. The `[multi]` and `[zscan]`
blocks remain required when `mode = "multi"` or `mode = "z-scan"` is
declared (existing gating preserved).

## Migrated / updated

- `configs/default.toml`: header rewritten with override-only framing.
  Active blocks still produce the recognizable IUCrJ mosa grid for
  `dfxm-forward --config configs/default.toml`; only an explicitly
  empty TOML drops to single-image behavior.
- `configs/identification_{single,multi,zscan}.toml`: same header
  treatment.
- 13 test call sites updated to add explicit `sample_remount="S1"` (7 Python
  `WallCrystalConfig(...)` constructor calls + 6 inline TOML strings).

## Unchanged

- Bit-equivalence safety net (`tests/data/golden/Fd_find_smoke.npy`)
  — covers the wall path, unaffected by F.
- All 7 `configs/variants/*.toml` — already explicit, untouched.
- HDF5 schema (B+C's `scan_mode` / `scanned_axes` / `crystal_mode`
  attrs populate identically; empty-TOML runs write `scan_mode="single"`,
  `scanned_axes=[]`, `crystal_mode="centered"`).
- Collaborator branches (`Beam_Stop`, `CDD_inc`, `ESRF_DTU`,
  `Purdue_Paper`, `dislocation_identification`) — ground-truth
  references, never touched.

## Out of scope (deferred)

- Module-level FOV constants (`Npixels`, `psize`, `zl_rms`) migrating to
  a `[detector]` config block — still tracked as a v2.1+ follow-up.
- Find_Hg seeding (the 2 long-standing xfailed bit-equivalence tests
  stay xfailed).
- darling 2.0.0 external-link traversal — separate follow-up.

## Upgrade

```bash
pip install --upgrade dfxm-geo==2.0.0
```

If your code constructs `WallCrystalConfig()` bare or with partial args,
audit and fix per the migration snippet above before upgrading.
