# GNB walls — geometrically-necessary boundaries

`dfxm-geo` can simulate a **geometrically-necessary boundary (GNB)** — a planar
dislocation wall whose dislocation content is fixed by the Frank equation for a
prescribed misorientation angle θ.  This is controlled by setting
`[crystal] mode = "gnb"` and filling in a `[crystal.gnb]` block.

The feature targets DFXM experiments where a GNB is visible as a row of
parallel dislocation lines and the experimenter wants to forward-model the
expected contrast, visibility under different reflections, or the g·b
invisibility condition for individual sets within the wall.

## Quick start

```toml
[reciprocal]
hkl     = [-1, 1, -1]
keV     = 17.0
backend = "analytic"    # kernel-free: no dfxm-bootstrap needed
beamstop = false

[geometry]
mode = "oblique"
eta  = 2.1950624059     # radians — from compute_omega_eta (see below)

[crystal]
lattice  = "cubic"
a        = 4.05e-10     # Al lattice parameter, metres
structure_type = "fcc"
material = "Al"
mount_x  = [1, 0, 0]
mount_y  = [0, 1, 0]
mount_z  = [0, 0, 1]
mode     = "gnb"

[crystal.gnb]
recipe    = "leds_eq11" # (111) two-set pure-twist
theta_deg = 0.05        # misorientation angle (degrees)
extent_um = 25.0        # in-plane half-width rendered per family (µm)
```

Full runnable configs are provided in
`src/dfxm_geo/data/configs/gnb_leds_eq11.toml` and
`src/dfxm_geo/data/configs/gnb_frankus.toml`.

## `[crystal.gnb]` keys

| Key | Type | Required | Description |
|---|---|---|---|
| `recipe` | string | yes | `"leds_eq11"`, `"leds_eq14"`, `"frankus"`, or `"custom"` |
| `theta_deg` | float (> 0) | yes | Misorientation angle across the boundary, degrees |
| `extent_um` | float (> 0) | yes | In-plane half-width rendered per dislocation family, µm |
| `max_dislocations` | int | no | Safety cap; raises if the total dislocation count exceeds this |

For `recipe = "custom"` an additional `[crystal.gnb.custom]` block is required
(see the [Custom recipe](#custom-recipe) section below).

## Built-in recipes

Three Frank-equation recipes are registered.  Each recipe specifies a boundary
normal **n**, a misorientation axis **a**, and a set of dislocation families
with their integer Burgers vectors **b**, line directions **ξ**, slip-plane
normals, and relative density ratios.

| Recipe | Boundary normal **n** | Misorientation axis **a** | Sets | Relative densities | Frank residual |
|---|---|---|---|---|---|
| `leds_eq11` | (1, 1, 1) | (1, 1, 1) | 2 | 1 : 1 | exact (< 1e-6) |
| `leds_eq14` | (1, 1, 1) | (−1, 3, 1) | 3 | 1 : 1 : 3 | exact (< 1e-6) |
| `frankus` | (0, 1, 0) | (0, 1, 0) | 3 | 1 : 1 : 1 | approximate (≤ 2%) |

### `leds_eq11` — (111) pure-twist, two-set

Boundary normal **n = a = (1, 1, 1)**.  Two in-plane {111}⟨110⟩ mixed
dislocations at equal density:

| Set | **b** | **ξ** | Slip plane | Relative density |
|---|---|---|---|---|
| 1 | [1, 0, −1] | [2, −1, −1] | (1, 1, 1) | 1 |
| 2 | [0, 1, −1] | [−1, 2, −1] | (1, 1, 1) | 1 |

The two Burgers vectors lie at 60° to each other in the (111) plane.  Their
tilt components cancel and the net Burgers content is a pure twist about (111).
The Frank equation is satisfied exactly (residual < 1 × 10⁻⁶).

### `leds_eq14` — (111) mixed tilt/twist, three-set

Boundary normal **n = (1, 1, 1)**, misorientation axis **a = (−1, 3, 1)**.
Three {111}⟨110⟩ sets at density ratio 1 : 1 : 3:

| Set | **b** | **ξ** | Slip plane | Relative density |
|---|---|---|---|---|
| 1 | [1, 0, −1] | [1, 0, −1] (screw) | (1, 1, 1) | 1 |
| 2 | [0, 1, −1] | [1, 0, −1] | (1, 1, 1) | 1 |
| 3 | [1, 0, 1] | [1, −1, 0] | (1, 1, −1) | 3 |

The Frank equation is satisfied exactly (residual < 1 × 10⁻⁶).

### `frankus` — (010) twist, three-set (approximate)

Boundary normal **n = a = (0, 1, 0)**.  Two in-plane B2/B5-type {111}⟨110⟩
dislocations plus a collinear screw, all at equal density 1 : 1 : 1:

| Set | **b** | **ξ** | Slip plane | Relative density |
|---|---|---|---|---|
| B5 | [1, −1, 0] | [1, 0, −1] | (1, 1, 1) | 1 |
| B2 | [0, 1, −1] | [1, 0, −1] | (1, 1, 1) | 1 |
| screw | [1, 0, 1] | [1, 0, 1] | (1, 1, −1) | 1 |

The 1 : 1 : 1 ratio gives a directional Frank residual of 0% (exact).

#### The frankus approximation and the Eq. 2 discrepancy — flag for G. Winther

The recipe is built at 1 : 1 : 1 (Sina's decision), which matches the paper's
own relaxed DDD densities.  However, the paper's Equation 2 highlights the ratio
**2 : 2 : 1** — and that ratio is **not** stress-free: the directional Frank
residual for 2 : 2 : 1 is approximately 40%, and the angle between the effective
Burgers content and the Frank target vector is ~18°.

Additionally, the literal Burgers vectors printed in the paper inset fail the
Frank equation entirely (~111% directional residual): `b_B2 = [0, 1, 1]` has a
component along (111) (`b·n = 2 ≠ 0`), so it is not a glide vector in the (111)
plane.

**This discrepancy is documented and flagged for co-author G. Winther to
confirm.**  The implementation uses 1 : 1 : 1 with the corrected in-plane
vectors; the registry gates the recipe with a loose tolerance (2%) to make the
approximation explicit.

## The θ knob and dislocation spacing

The misorientation angle θ (`theta_deg`) controls the density of dislocations
in the wall.  The approximate spacing between parallel lines of the same set is:

```
d ≈ |b| / (2 sin(θ/2))
```

This is an **approximation**.  The exact Frank spacing carries a recipe-specific
geometric factor that depends on the angle between the Burgers vectors in the
wall plane.  For `leds_eq11`, the two in-plane Burgers vectors are 60° apart,
which introduces an exact factor of **√3/2 ≈ 0.866** relative to the simple
formula.  The exact value comes from the Frank solver (`solve_density_scale`).

**Larger θ → smaller spacing → more dislocations.**  Because `d ∝ 1/sin(θ/2)`,
increasing θ increases the dislocation density.  The `max_dislocations` cap
therefore protects against **large θ** (a very dense wall), not tiny θ.

### Resolvability

With a ~40 nm object pixel (`psize` default):

| θ (degrees) | Approx. spacing d | Spacing in pixels | Resolvable? |
|---|---|---|---|
| 0.05 | ~320 nm | ~8 px | Yes — recommended default |
| 0.5 | ~33 nm | < 1 px | Sub-pixel (wall unresolved) |

A misorientation of 0.05° produces individually resolvable lines spaced ~8
object pixels apart — a useful default for DFXM forward modelling.  At 0.5°
the spacing is sub-pixel and the wall appears as a diffuse contrast feature.
Smaller θ produces fewer, more widely spaced lines.

## `extent_um` — in-plane width

`extent_um` sets the half-width of the rendered wall in micrometres.  The
builder places dislocation lines symmetrically over `[−extent_um/2, +extent_um/2]`
in the in-plane direction perpendicular to each family's line direction.
This should be set **larger than the imaging field of view** (typically ~20 µm)
so the wall fills the entire frame.  Setting it too small produces a truncated
wall.

## Simplified vs. oblique geometry

GNB mode works in both geometry modes.

**Oblique geometry** (`[geometry] mode = "oblique"`) is recommended for
physically-grounded simulations.  It gives exact control over the crystal
mount and enables the kernel-free **analytic resolution backend**
(`backend = "analytic"`), so no `dfxm-bootstrap` run is needed.  For FCC Al
at 17 keV, compute the exact η from `compute_omega_eta`:

```python
from dfxm_geo.crystal.oblique import CrystalMount, compute_omega_eta
import numpy as np

mount = CrystalMount(
    lattice="cubic", a=4.05e-10,
    structure_type="fcc", material="Al",
    mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1),
)
geom = compute_omega_eta(mount, hkl=(-1, 1, -1), keV=17.0)
eta = geom.eta_1 if not np.isnan(geom.eta_1) else geom.eta_2
print(f"eta = {eta:.10f}")  # put this value in [geometry] eta
```

**Simplified geometry** (`[geometry] mode = "simplified"`, or omit the block)
synthesizes a cubic FCC Al cell automatically (a = 4.05 Å, ν = 0.334).
Simplified mode does not require a crystal mount and can run with an MC
kernel.  It is useful for quick sanity checks but does not support the analytic
backend.

## Frame model

Dislocation positions are placed in the **lab frame** via the sample→grain
rotation matrix `Us` (the module-level constant in `dfxm_geo.direct_space.
forward_model`).  This is the only placement that keeps the dislocation field
lines coplanar with the boundary plane the position comb occupies.  Using the
identity matrix or `Us.T` instead causes the field lines to pierce the boundary
by 18–60° — an incoherent wall (verified in the Task-1 spike,
`docs/superpowers/notes/2026-06-20-gnb-spike-findings.md`).  The crystal mount
(`mount_x/y/z`) controls only the reflection/Q path and does not appear in
the displacement-field placement.

## Schmid–Boas sign reconciliation

Each dislocation set in a built-in recipe satisfies three geometric constraints:

- **Glide condition:** b · slip_plane = 0
- **Line in glide plane:** ξ · slip_plane = 0
- **Line in boundary plane:** ξ · n = 0

The Frank residual gate (`frank_residual`) verifies these automatically at build
time.  `leds_eq11` and `leds_eq14` have an exact residual below 1 × 10⁻⁶;
`frankus` uses a loose but documented tolerance (≤ 2%).

## Custom recipe

If none of the built-in recipes match your boundary, supply a full recipe via
the TOML escape hatch.  Set `recipe = "custom"` and add a
`[crystal.gnb.custom]` block with the boundary normal, misorientation axis,
and one `[[crystal.gnb.custom.set]]` entry per dislocation family:

```toml
[crystal.gnb]
recipe    = "custom"
theta_deg = 0.05
extent_um = 25.0

[crystal.gnb.custom]
n = [1, 1, 1]        # boundary-plane normal (Miller indices)
a = [1, 1, 1]        # misorientation axis (Miller indices)

[[crystal.gnb.custom.set]]
b           = [1, 0, -1]       # Burgers vector (Miller indices)
xi          = [2, -1, -1]      # line direction (Miller indices)
slip_plane  = [1, 1, 1]        # slip-plane normal (Miller indices)
rel_density = 1.0              # relative density weight

[[crystal.gnb.custom.set]]
b           = [0, 1, -1]
xi          = [-1, 2, -1]
slip_plane  = [1, 1, 1]
rel_density = 1.0
```

Each entry must satisfy the glide condition (b · slip_plane = 0) and the line
directions must lie in both the glide plane and the boundary plane.  The
validator enforces these constraints at build time.  There is no default
`frank_tol` for custom recipes (the builder uses the standard residual gate and
reports the measured residual in the population sidecar).

## Scope and limitations

**Reactions excluded.** The built-in recipes do not model Lomer-lock or other
dislocation reaction products (e.g. b₃/b₆ stair-rod partials).  The dislocation
content is the primary glide set per recipe.

**Forward only.** GNB mode generates a `DislocationPopulation` for the forward
simulation.  There is no dedicated identification path for GNB walls; the
standard `dfxm-identify` pipeline can be used on the resulting images if the
individual slip systems are known.

**Infinite straight dislocations.** The builder places infinitely straight
dislocation lines (the standard `dfxm-geo` model).  Curvature, jogs, and
partial dislocations are not modelled.

**Isotropic elasticity.** Like all other crystal modes, GNB uses the
Hirth & Lothe isotropic solution with a single scalar ν.  See the
[Limitations](crystal-structures.md#limitations) section of the crystal
structures page.

**FCC recipes only in the built-in registry.** The three built-in recipes are
written for FCC {111}⟨110⟩ slip.  The `custom` hatch accepts any slip system,
including BCC and HCP families, but the recipe vectors must be supplied by the
user.
