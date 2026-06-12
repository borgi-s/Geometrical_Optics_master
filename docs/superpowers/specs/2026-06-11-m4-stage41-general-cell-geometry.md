# M4 Stage 4.1 — General Cell Geometry (design record)

**Status: shipped on `feature/m4-stage41-cell-geometry`, 2026-06-12.**

## What

`UnitCell` (`crystal/cell.py`): six-parameter triclinic cell; real-space
matrix `A` (a∥x̂, b in x-y plane, IUCr/gemmi setting), reciprocal
`B = 2π·A⁻ᵀ`, metric d-spacing `d = 2π/|B·G|`. `UnitCell.from_lattice`
fills constrained parameters for the seven crystal systems (trigonal =
rhombohedral setting; hexagonal-setting trigonal → `lattice="hexagonal"`).
Cell angles are **degrees** (`*_deg` fields, CIF convention) — the
repo-wide radians rule covers scan/geometry angles only.

`CrystalMount` accepts any system; `C_s = cell.A`; mount Miller indices
are plane normals, Cartesian directions `B@m`, orthogonality checked in
Cartesian (tol 1e-9 on normalized dots). `compute_omega_eta` /
`find_reflections` needed **zero changes** — they were already written
against `C_s`/`U_mount` (paper eq 24); hexagonal-Mg tests prove it.
`_validate_reflection(hkl, keV, cell)` replaces the scalar-`a` signature.

The TOML `[crystal]` block accepts `b`/`c`/`alpha_deg`/`beta_deg`/`gamma_deg`
alongside `a`, and the provenance serializers emit those five keys for
non-cubic mounts only (cubic provenance output is byte-identical to v2.x).
`run_exposure_simulation` takes `d0_angstrom` (default Al d_111) instead of
a hardcoded spacing.

## Bit-identity guarantee (the regression gate)

Cubic cells take verbatim legacy fast-paths: `A = a·np.eye(3)`,
`B = (2π/a)·np.eye(3)`, `d = a/np.sqrt(h²+k²+l²)`, legacy raw-Miller
`U_mount` normalization and orthogonality check. Asserted with exact
`==`/`np.array_equal` in tests; the untouched cubic suite + goldens gate
the rest. Do not merge the fast-paths into the general branch.

## What is deliberately NOT in 4.1

- Forward/identify for non-cubic cells — **rejected with an explanatory
  error** in `_build_geometry_config` (both modes) because
  `forward_model.py` `q_hkl = hkl/√(h²+k²+l²)` (3 sites: ~268/1266/1343,
  "We have assumed B_0 = I") and the FCC dislocation/Burgers math are
  Stage 4.3 scope. `dfxm-bootstrap` (oblique) and `dfxm-find-reflections`
  DO support non-cubic — that is the 4.1 deliverable.
- Simplified mode stays cubic-only (guarded, both bootstrap- and
  pipeline-side).
- CIF parsing (gemmi) — Stage 4.2. Space-group extinction rules — 4.2.
  Slip-system registries / Burgers magnitudes — 4.3. HDF5 provenance of
  cell params beyond the TOML echo — lands with 4.3 when non-cubic can
  actually reach the writers.
- Handedness of the mount triple is not validated (pre-existing behavior,
  cubic included); a left-handed triple flips η sign. Revisit in 4.3.

## Stage 4.2 CIF library decision (ADR, 2026-06-11)

**gemmi reaffirmed over cifkit** (Sina asked 2026-06-11; decided keep
gemmi). gemmi's space-group symmetry-operation machinery is exactly what
the Stage 4.2 extinction rules need; cifkit's strengths are
materials-science analysis (coordination environments, bond lengths),
which is out of scope here. Optional: a half-day library spike as Stage
4.2's first task to confirm extinction-rule ergonomics, recorded as the
ADR — but the default is gemmi. Do not re-litigate without new evidence.

## Gates at completion

Full suite: 899 passed / 0 failures / 1 skipped / 1 xfailed (baseline 851
at branch point `b0a2300`; the skip = legacy pickle absent and the xfail =
`test_forward_output_matches_pickle_era_snapshot` are both pre-existing).
mypy: 0 errors across 40 source files.
