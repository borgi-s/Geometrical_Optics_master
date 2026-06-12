# M4 Stage 4.2 — CIF ingestion + space-group extinction rules (design)

**Status: approved by Sina 2026-06-12; implementation on
`feature/m4-stage42-cif-ingestion`.**

Roadmap §5 Stage 4.2. Builds on Stage 4.1 (general cell geometry,
`docs/superpowers/specs/2026-06-11-m4-stage41-general-cell-geometry.md`),
whose ADR fixed **gemmi** as the CIF/symmetry library. A kickoff spike
(gemmi 0.7.5) confirmed the ergonomics; findings are baked in below.

## Goals

1. `[crystal] cif = "path/to/file.cif"` populates cell parameters and
   space group; explicit TOML keys override CIF values.
2. Space-group systematic absences are filtered out of
   `find_reflections()` and rejected at config validation with a clear
   error.
3. Ceramics acceptance: an Al₂O₃ (corundum, R-3c) CIF drives
   `dfxm-find-reflections` end-to-end with correct extinction filtering.

## Decisions (with Sina, 2026-06-12)

- **Space-group sources**: a CIF *or* a plain `[crystal] space_group`
  TOML key (usable without any CIF). TOML overrides CIF, consistent
  with the cell-parameter override rule.
- **Opt-in filtering**: no space group anywhere → behavior identical to
  v2.x/4.1 (no extinction knowledge). The built-in Al default mount
  stays space-group-less. No warnings when absent.
- **Hard error, no escape hatch**: an explicitly-listed extinct hkl is
  a `ValueError` at every site where a concrete hkl enters (config
  load both modes, each `[[reflections]]` entry, `dfxm-bootstrap`).
  The escape hatch is removing the `space_group` key.
- **Ceramic acceptance case**: Al₂O₃ CIF fixture + e2e
  `dfxm-find-reflections` test (Sina asked whether 4.2 enables
  ceramics — answer: planning yes, dislocation simulation in 4.3).

## Architecture (Approach A)

### New module `crystal/cif.py` — the only gemmi boundary

gemmi is imported **lazily** inside this module so the `[cif]` extra
stays optional. No gemmi objects escape it. Public surface:

- `CifCell` dataclass: `lattice` (inferred system string), `a/b/c` in
  **metres** (converted from CIF Å), `alpha_deg/beta_deg/gamma_deg`,
  `space_group: str | None` (canonical H-M symbol), `source` path.
- `load_cif(path) -> CifCell` via `gemmi.read_small_structure`.
  **Spike finding: `read_structure` (macromolecular) silently returns
  an empty cell on small-molecule CIFs — `read_small_structure` is the
  required entry point.** Lattice system inferred from the space
  group's crystal system, falling back to cell-parameter inspection
  when the CIF has no SG tag. Hexagonal-setting trigonal →
  `lattice="hexagonal"` (4.1 convention).
- `validate_space_group(name) -> str` → canonical H-M symbol.
  gemmi accepts forgiving spellings (`Fm-3m`, `F m -3 m`, `P63/mmc`,
  `P6_3/mmc`); unknown names raise a clean `ValueError`.
- `is_systematically_absent(space_group, hkl) -> bool` wrapping
  `gemmi.SpaceGroup(...).operations().is_systematically_absent`.
  Covers centering *and* glide/screw absences (spike: FCC 100/110/210
  absent, 111/200/220/311 present; P6₃/mmc 00l-odd absent; both
  textbook-correct). Callers that loop hkl (e.g. `find_reflections`)
  get a build-once-`GroupOps` helper so the symbol isn't re-parsed
  1331 times.

### `CrystalMount` gains `space_group: str | None = None`

Validated in `__post_init__`: symbol resolvable, and its crystal
system must match `lattice` (`space_group="Fm-3m"` +
`lattice="hexagonal"` → `ValueError`). **`UnitCell` is untouched** —
the Stage 4.1 cubic bit-identity fast-paths never see this change.

## Config semantics

`[crystal]` accepts two new keys: `cif` and `space_group`.

- **Resolution order** in `_crystal_mount_from_toml`: CIF values are
  the base; any explicit TOML key (`lattice`, `a`…`gamma_deg`,
  `space_group`) overrides per-key.
- `mount_x/y/z` stay **required and TOML-only** — they describe the
  experimental mounting, which no CIF knows.
- Relative `cif` paths resolve against the TOML file's directory.
- **Simplified mode** (Bragg θ driven by `[reciprocal] lattice_a`, not
  the mount): cubic CIF + no explicit `lattice_a` → `lattice_a`
  inherits the CIF's `a`; explicit `lattice_a` wins. Non-cubic CIF in
  simplified mode hits the existing 4.1 rejection — the mount-guard
  trigger extends from `"lattice" in crystal_raw` to also fire on
  `"cif" in crystal_raw`.
- **`[reflections_auto]`** inherits filtering via `find_reflections`;
  `[[reflections]]` entries are individually validated.

## Data flow

```
TOML [crystal] cif ──► load_cif() ──► CifCell (base values)
TOML explicit keys ──────────────────► override per-key
                                          │
                                          ▼
                          _crystal_mount_from_toml() ──► CrystalMount(space_group=…)
                                          │
            ┌─────────────────────────────┼──────────────────────────┐
            ▼                             ▼                          ▼
   find_reflections()            config validation             dfxm-bootstrap
   (skip absent rows;          (single hkl + [[reflections]]   (reject extinct hkl
   GroupOps built once           entries → hard error)          before kernel build)
   per call)
```

The mount already travels to all three consumers (Stage 4.1 plumbing);
the space group rides along — no new threading.

## Provenance

Serializers emit the **resolved** cell values plus `space_group` — not
the `cif` path — so archived provenance never depends on a file that
moved. Cubic mounts without a space group serialize byte-identically
to today (regression-gated).

## Error handling

All config-time failures are `ValueError` with actionable messages
(the one exception: a missing optional dependency is `ImportError`):

- gemmi missing but `cif`/`space_group` used → `ImportError`:
  "CIF support requires gemmi — `pip install dfxm-geo[cif]` (or
  `conda install -c conda-forge gemmi`)". Raised lazily, only when the
  keys are used.
- CIF unreadable or no cell parameters → error naming the path.
  (Spike: a CIF without cell tags yields gemmi's 1 Å sentinel cell —
  the loader checks for it explicitly.)
- Unknown space-group symbol → wrapped `ValueError` naming the key.
- SG/lattice crystal-system mismatch → `ValueError` from
  `CrystalMount.__post_init__`.
- Extinct explicit hkl → "(1,0,0) is systematically absent in space
  group F m -3 m — run 'dfxm-find-reflections' to list allowed
  reflections." Same message shape at all three sites.
- CIF without an SG tag → **not** an error; `space_group=None`, no
  filtering (opt-in decision).

## Packaging & release notes

- `pyproject.toml`: `[project.optional-dependencies] cif = ["gemmi>=0.7"]`;
  also appended to `dev` so the suite exercises it. CIF tests use
  `pytest.importorskip("gemmi")` so a gemmi-less install still passes.
- mypy: `ignore_missing_imports` for gemmi if it ships no stubs.
- ⚠ Release checklist (next PyPI publish / conda-forge sync):
  conda recipes have no optional deps — install docs say
  `conda install -c conda-forge gemmi`; do NOT add gemmi to the
  recipe's `run` requirements. (The pending `dfxm-find-reflections`
  entry-point mirror from M3 still applies too.)

## Testing

Fixtures under `tests/data/cif/`: Al (Fm-3m), Mg (P6₃/mmc),
**Al₂O₃ corundum (R-3c, hexagonal setting)**.

- **Unit**: Å→m conversion; lattice inference with/without SG tag;
  per-key override precedence; absence wrapper asserted against
  *textbook* extinction tables (FCC, BCC, HCP, R-3c) — not against
  gemmi itself.
- **Config**: simplified-mode `lattice_a` inheritance + override;
  extinct-hkl rejection at all three sites; gemmi-missing error path
  (monkeypatched import).
- **Ceramic acceptance (e2e)**: alumina config → `dfxm-find-reflections`
  filters R-centering (−h+k+l ≠ 3n) and c-glide absences; known-allowed
  reflections (e.g. 006, 113) survive with correct θ.
- **Gates**: full suite + slow markers green; mypy 0 errors;
  byte-identical provenance for cubic no-SG configs.

## Out of scope (Stage 4.3+)

- Non-cubic forward/identify (the three `q_hkl = hkl/√(h²+k²+l²)`
  sites) and slip systems beyond FCC — Stage 4.3.
- Atomic positions / structure factors from the CIF (basis-forbidden
  or weak reflections beyond space-group absences are not modeled).
- HDF5 provenance of cell params beyond the TOML echo — lands with 4.3.
