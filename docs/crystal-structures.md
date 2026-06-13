# Crystal structures

How `dfxm-geo` models dislocation physics in FCC and BCC crystals —
slip-system selection, Burgers vector magnitudes, isotropic Poisson ratios,
and the configuration TOML keys that control them. HCP support (basal,
prismatic, and pyramidal families with hexagonal Miller-Bravais indices) is
planned for Stage 4.3b and is **not** implemented yet.

## Structure resolution

Before the simulation runs, the code resolves a single *structure type* string
(`"fcc"` or `"bcc"`) that determines which slip-system table to load. The
precedence is:

| Source | When present | Overrides |
|---|---|---|
| Space group | `[crystal] space_group = "..."` or CIF file | everything — if an explicit `structure_type` *contradicts* the space group, the run aborts with an error |
| Explicit `structure_type` | `[crystal] structure_type = "bcc"` | the default |
| Default | nothing set | falls back to `"fcc"` (back-compat with v2.x) |

Space-group-to-structure mapping:

| Bravais lattice | Representative space groups | Derived type |
|---|---|---|
| F-cubic | Fm-3m, F4/mmm, … | `"fcc"` |
| I-cubic | Im-3m, I4/mmm, … | `"bcc"` |
| P-hexagonal | P6₃/mmc, P6/mmm, … | `"hcp"` (4.3b — raises if used) |

Contradicting the space group is an error:

```toml
[crystal]
cif = "Fe.cif"        # space group Im-3m → bcc
structure_type = "fcc"  # Error: contradicts Im-3m
```

## Supported structures and slip systems

### FCC — {111}⟨110⟩

The classic 12 slip systems on the four {111} planes:

| Plane family | Burgers family | Unique planes | Systems per plane | Total |
|---|---|---|---|---|
| {111} | ⟨110⟩ | 4 | 3 | 12 |

The order of the 12 systems is fixed to the v2.x `_SLIP_SYSTEM_111` sequence
for bit-identity: `random_dislocations` draws `rng.integers(0, 12)` into this
ordered list, and `wall` mode uses entry [0]. A symmetry-completeness assertion
runs at import time to guarantee the set is unchanged.

Only one slip family is registered for FCC and `slip_families` has no
effect.

### BCC — {110}⟨111⟩ + {112}⟨111⟩

By default, both principal slip families are active, giving 24 systems across
18 distinct planes:

| Family | Unique planes | Systems | Systems per plane |
|---|---|---|---|
| {110}⟨111⟩ | 6 | 12 | 2 |
| {112}⟨111⟩ | 12 | 12 | 1 |
| **Both (default)** | 18 | **24** | — |

The {123}⟨111⟩ "pencil glide" family is **not** in the built-in registry.
If you need it, use the `[[crystal.slip_system]]` escape hatch (see below).

#### Narrowing to one BCC family

```toml
[crystal]
structure_type = "bcc"
slip_families = ["{110}<111>"]   # only the 12 primary systems
```

Valid names for `slip_families` are `"{110}<111>"` and `"{112}<111>"`.
Supplying an unknown name raises an error listing the available options.

## TOML configuration examples

### Explicit structure type

```toml
[crystal]
structure_type = "bcc"
material = "Fe"            # sets ν from the Poisson table (0.29, [KL])
lattice = "cubic"
a = 2.87                   # Å — α-iron lattice parameter
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
```

### CIF-driven (space group inferred automatically)

```toml
[crystal]
cif = "Fe.cif"             # space group Im-3m → bcc (authoritative)
material = "Fe"
# lattice + a are populated from the CIF; mount_x/y/z remain TOML-only
mount_x = [1, 1, 0]
mount_y = [-1, 1, 0]
mount_z = [0, 0, 1]
```

`cif` requires `pip install "dfxm-geo[cif]"` (or `conda install -c conda-forge gemmi`).
The path is resolved relative to the config file's directory.

### Custom slip systems (escape hatch)

For non-standard families such as {123}⟨111⟩ BCC pencil glide, or
for any crystal structure not in the built-in registry, use the
`[[crystal.slip_system]]` array-of-tables block:

```toml
[crystal]
# structure_type must NOT be set when using [[crystal.slip_system]]
material = "Fe"
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]

[[crystal.slip_system]]
plane = [1, 2, 3]
burgers = [1, 1, 1]    # b·n must equal zero — [1,1,1]·[1,2,3] = 1+2+3 = 6 → error

[[crystal.slip_system]]
plane = [1, 2, 3]
burgers = [1, -1, 0]   # 1·1 + 2·(-1) + 3·0 = 0 ✓ — valid glide system
```

Each entry is stored as a *literal* slip system (no symmetry-orbit
expansion). The validator enforces **b·n = 0** (glide condition); providing
a non-glide pair raises `ValueError`. Entries are enumerated in the order
they appear in the TOML file.

`structure_type` and `[[crystal.slip_system]]` are mutually exclusive —
setting both raises an error.

### BCC oblique-η landmine

A single-reflection BCC oblique run requires an **exact** η value (the
azimuthal angle at which the reflection satisfies the diffraction condition
for the given mount). The `[geometry]` validator cross-checks the supplied
`eta` against `compute_omega_eta` and raises if it does not match.
Supplying `eta = 0.0` will be rejected unless 0 happens to be the correct
value for your mount:

```toml
# ❌ likely to fail: eta=0.0 is rarely correct for a BCC oblique run
[geometry]
mode = "oblique"
eta = 0.0
```

**Workaround:** use the multi-reflection `[[reflections]]` syntax, which
permits omitting η (the solver finds all valid (ω, η) pairs internally):

```toml
[[reflections]]
hkl = [1, 1, 0]
# eta omitted — the solver finds valid geometry automatically
```

Or supply the correct η from `dfxm_geo.reciprocal_space.kernel.compute_omega_eta`
before writing the TOML:

```python
from dfxm_geo.reciprocal_space.kernel import compute_omega_eta
theta, omega, eta = compute_omega_eta(mount, hkl=(1, 1, 0), keV=17.0)
print(f"eta = {eta:.6f}")   # put this value in [geometry] eta
```

## Burgers vector magnitudes

The Burgers magnitude |b| used in the displacement-field formula depends on
the crystal structure.

### FCC (Al, Cu, Ni, …)

For backward-compatibility with v2.x, the FCC magnitude is the **historical
constant**:

```
BURGERS_VECTOR = 2.862e-4 µm   (hard-coded in dfxm_geo.constants)
```

This value matches the Al (−1, 1, −1) reference from the Borgi et al. (2024)
paper. All FCC simulations use this constant regardless of the actual lattice
parameter in the TOML — changing `[crystal] a` for FCC does **not** alter |b|.

### Non-FCC (BCC, custom)

For BCC and custom structures, |b| is derived from the cell matrix **A** and
the integer Burgers direction **b_int**:

```
|b| = fraction × |A · b_int|   (in µm, A in metres)
```

where the lattice-translation fraction is 1/2 for both ⟨110⟩_fcc and ⟨111⟩_bcc
(centered-lattice primitive translations). For the two BCC families:

| Family | Burgers direction | Formula | Example (α-Fe, a = 2.87 Å) |
|---|---|---|---|
| {110}⟨111⟩ | ½⟨111⟩ | a√3/2 | 2.483 × 10⁻⁴ µm |
| {112}⟨111⟩ | ½⟨111⟩ | a√3/2 | 2.483 × 10⁻⁴ µm |

Both BCC families share the same Burgers direction family ⟨111⟩, so they give
the same |b|.

## Poisson ratio

The displacement-field formula (Hirth & Lothe isotropic elasticity) requires a
Poisson ratio ν. Resolution precedence:

1. `[crystal] poisson_ratio = <value>` — explicit override (takes any float).
2. `[crystal] material = "<element>"` — looked up in the built-in table.
3. Default — falls back to Al: ν = 0.334 (source: [SW]).

### Built-in Poisson table

| Material | Symbol | Structure | ν | Source |
|---|---|---|---|---|
| Aluminium | Al | FCC | 0.334 | [SW] |
| Copper | Cu | FCC | 0.34 | [SW] |
| Nickel | Ni | FCC | 0.31 | [KL] |
| Iron (α) | Fe | BCC | 0.29 | [KL] |
| Tungsten | W | BCC | 0.28 | [KL] |
| Titanium (α) | Ti | HCP | 0.32 | [KL] |
| Magnesium | Mg | HCP | 0.29 | [SW] |

Values are polycrystalline Voigt-Reuss-Hill (VRH) aggregate averages.

**Full references:**

- **[KL]** Kaye, G. W. C. & Laby, T. H. *Tables of Physical and Chemical
  Constants*, 16th ed. Longman, 1995. Sec. 2.3.4 (elastic properties of
  polycrystalline solids).
- **[SW]** Simmons, G. & Wang, H. *Single Crystal Elastic Constants and
  Calculated Aggregate Properties: A Handbook*, 2nd ed. MIT Press, 1971
  (VRH aggregate ν).

The Ti and Mg entries appear in the table for completeness — they will become
relevant for HCP simulations in Stage 4.3b. A forward or identify run that
sets `material = "Ti"` today will still raise an error later when the HCP
slip-system registry is absent; the Poisson lookup itself will succeed.

## Structure provenance in HDF5 output

For **structure-aware runs** (oblique geometry or any run with a non-trivial
crystal mount), the `/N.1/` scan entry in the master HDF5 carries the
following additional attributes:

| Attr | Type | Example | Notes |
|---|---|---|---|
| `structure_type` | string | `"bcc"` | Resolved structure family |
| `poisson_ratio` | float64 | `0.29` | Resolved ν |
| `poisson_source` | string | `"KL"` | `"KL"`, `"SW"`, or `"override"` |
| `burgers_magnitude_um` | float64 | `2.483e-4` | \|b\| in µm for the primary slip family |
| `material` | string | `"Fe"` | Only present if `[crystal] material` was set |
| `slip_families` | list[string] | `["{110}<111>"]` | Only present if `slip_families` was set |
| `space_group` | string | `"Im-3m"` | Only present if derived from a space group or CIF |

These attrs are written by `dfxm_geo.io.hdf5.build_structure_provenance_attrs`
and are appended to the same `/N.1/` attrs that carry `scan_mode`,
`scanned_axes`, and `crystal_mode` (see `docs/output-format.md`).

**FCC simplified path:** when `mount` is `None` (the default for a symmetric
FCC run with no `[crystal] cif` or `structure_type` set), **none** of these
attrs are written. The forward and identify outputs remain byte-identical to
v2.5.x in this case.

## Limitations

**Isotropic elasticity only.** The displacement field implemented in
`crystal/dislocations.py` uses the Hirth & Lothe isotropic solution with a
single scalar ν. Anisotropic single-crystal elasticity (the full C_ijkl tensor
— e.g. C₁₁, C₁₂, C₄₄ for cubic) is **not** implemented and is out of scope
for v3.0.0. For materials with strong elastic anisotropy (W, Ni), the isotropic
VRH average is a reasonable first approximation for bulk polycrystalline
aggregates, but per-grain anisotropic contrast will differ from the isotropic
prediction.

**HCP is 4.3b.** The `derive_structure_type` function returns `"hcp"` for
P-hexagonal space groups, and the Poisson table includes Ti and Mg. However,
the slip-system registry has no HCP entry, so any simulation that reaches the
slip-system lookup will raise a clear error. Full HCP support — basal
{0001}⟨11-20⟩, prismatic {10-10}⟨11-20⟩, and pyramidal ⟨c+a⟩ families with
Miller-Bravais → Cartesian conversion — is Stage 4.3b.

**{123}⟨111⟩ BCC (pencil glide) only via the escape hatch.** The built-in
registry covers {110} and {112} BCC families. The {123} family is physically
active in BCC metals at elevated temperatures; it can be added via
`[[crystal.slip_system]]` entries (one per system, no symmetry expansion).

**Cubic only in the symmetry enumerator.** The `_variants` / `_enumerate_orbit`
functions that expand a (plane, Burgers) family representative into a full
symmetry orbit treat Miller indices as Cartesian components, which is only
exact for cubic crystals. Non-cubic structures must supply explicit systems
via the escape hatch.
