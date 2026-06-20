# GNB Frank-equation wall builder (`gnb` crystal mode) — design

Date: 2026-06-20
Branch: `feature/gnb-walls` (worktree `wt-gnb-walls`, off `main` = `e60247e`)
Status: design approved + Opus-reviewed (2026-06-20). `leds_eq11`/`leds_eq14`
verified exact (Frank residual ~1e-16); `frankus` built **approximate per the
paper** (Sina's decision) — see its section. Frame model corrected per code
review (positions are lab-frame; mount does not rotate the dislocation field).
→ implementation plan next.

## Goal

Add a new crystal-layout mode, `mode = "gnb"`, that constructs a **2-D
dislocation network representing a geometrically necessary boundary (GNB)** from
a **Frank-equation recipe**, and feeds it through the existing forward model to
produce DFXM images. The misorientation angle **θ** is the primary user knob;
the network's per-set densities (hence spacings) are derived from θ via the
Frank equation.

Three literature walls are the first built-in recipes (all FCC):

1. **`leds_eq11`** — two sets, pure twist. Winther, Hong & Huang, *Phil. Mag.*
   95(13) (2015), Eq. 11.
2. **`leds_eq14`** — three sets, mixed tilt/twist (two in the boundary plane, one
   from another slip plane). Same paper, Eq. 14.
3. **`frankus`** — (010) twist boundary, three effective families. Frankus,
   Pachaury, El-Azab, Devincre, Poulsen & Winther, *JMPS* 199 (2025) 106069,
   Eq. 2.

Plus a **`custom`** hatch so any `(n, a, sets)` recipe builds a wall — this is
the "general builder" requested.

Primary use: **research now, paper figures later.** Aim for physical
faithfulness, not calibration to specific measured numbers.

## Out of scope (explicitly deferred)

- **Identification-side support.** Forward only. The design must not *preclude*
  later identify support, but we build none now.
- **Dislocation reactions.** No b₃ / b₆ Lomer-lock products (LEDS Eqs. 12–13,
  16–19; Frankus reacted states). Per Grethe: build only the un-reacted nets.
- **Quantitative DDD/TEM matching.** No calibration to Frankus's θ ≈ 22–31 mdeg
  or measured densities.
- **Finite segments / nodes.** Each dislocation stays an infinite straight line,
  exactly as the current forward model treats them.
- **Built-in BCC/HCP recipes.** The framework stays structure-agnostic (the 3
  recipes carry `structure = "fcc"`), but no non-FCC built-ins ship now.
- **Anisotropic elasticity.** Isotropic ν only, as today.

## Background: what already exists (with file refs)

Frames and the forward engine are reused unchanged. Key facts established during
brainstorming (see the two Explore maps in the session):

- **Crystal-layout modes** are a discriminated union in
  `src/dfxm_geo/config.py` (`CrystalConfig`, `mode ∈ {centered, wall,
  random_dislocations}` ~lines 309–397), each with a sub-block
  (`CenteredCrystalConfig`, `WallCrystalConfig`, `RandomDislocationsConfig`).
- **The population builder** `build_dislocation_population()` in
  `src/dfxm_geo/direct_space/forward_model.py` (~lines 1336–1550) dispatches on
  `crystal.mode` and returns a `DislocationPopulation`.
- **`DislocationPopulation`** (`forward_model.py` ~lines 1201–1236) already
  carries everything a heterogeneous network needs:
  `positions_um (N,3)`, `Ud (N,3,3)` = column-stacked `[b̂, n̂, t̂]`,
  `rotation_deg (N,)` (edge↔screw character), `b_um` (scalar) **or**
  `b_um_per (N,)` (per-dislocation |b|), `ny` (Poisson ν), `sidecar` (provenance).
- **The current `wall`** (`forward_model.py` ~lines 1386–1412) is the degenerate
  case for us: one family of *identical* lines along sample-**y** (`x=z=0`), all
  sharing the structure's first slip system; spatial direction **decoupled** from
  the crystal plane. We do **not** modify it.
- **The forward engine** `Find_Hg_from_population()` (`forward_model.py`
  ~lines 1650–1724) → numba `find_hg_population()` (`crystal/dislocations.py`
  ~lines 544–588) consumes the population directly. **No engine change is
  required**; the new work is a builder that populates the existing struct.
- **Frame chain** (`crystal/dislocations.py`, kernel transform
  `M = Ud.T @ Us.T @ S.T @ Theta`): lab →(Theta)→ sample →(S, remount)→
  →(Us)→ grain/crystal →(Ud.T)→ dislocation frame. `Ud` is built from crystal
  Miller vectors (`_ud_matrix_from_bnt` / `_ud_matrix_from_bnt_cell`,
  `forward_model.py` ~1238–1290). The crystal mount (`crystal/oblique.py`,
  `CrystalMount.U_mount`) orients crystal→lab in oblique mode; simplified mode
  uses the fixed module-level `Us`.
- **Slip-system registry** `crystal/slip_systems.py` provides `(b, n, t)` per
  system and `burgers_magnitude_of(...)`; **bit-identity landmines** there
  (`_FCC_111_110_ORDERED`, FCC |b| = `BURGERS_VECTOR`) must not be touched.

> ✅ **Frame model (resolved by the Opus code review, was the #1 risk):**
> `positions_um` / `offset` is consumed in the **lab frame** — the kernel
> subtracts `offset` from the lab-frame ray grid `rl` *before* applying `M`
> (`forward_model.py:1664–1687`, `dislocations.py:468–474`, ray grid built lab-
> frame at `forward_model.py:755–761`). Crucially, the **dislocation field
> orientation is governed only by the fixed module-level `Us`**, the same in
> simplified and oblique geometry; `U_mount` is used *only* in the reflection /
> Q-vector path (`oblique.py:364`), **never** in the dislocation-field path. So:
> (1) `Ud` stays in the crystal frame (consistent with `q_hkl`, mapped via `Us`
> exactly like `wall`/`centered`), giving correct g·b contrast; (2) the recipe
> `n`/`a` set the network's *internal* geometry; (3) to make the boundary plane
> land at a chosen lab orientation we rotate the line **positions** (and only
> positions) by a crystal→lab placement matrix. The exact placement matrix is
> still pinned by the Phase-0 spike + a permanent round-trip test (Component 5).

## Physics: the recipes (source of truth)

A dislocation boundary free of long-range stress satisfies the **Frank equation**
(both papers, Eq. 1):

```
Σ_i  ρ̂_i b_i { (n × ξ_i) · V } = 2 sin(θ/2) (V × a)      ∀ V in the boundary plane
```

- `b_i` Burgers vector, `ξ_i` line direction, `ρ̂_i` planar density (line length
  per unit area, units m⁻¹) of set `i`; `n` boundary-plane normal; `a`
  misorientation axis; `θ` misorientation angle.

Each recipe fixes, per set, `(b_i, ξ_i, slip_plane_i, rel_density c_i)` plus the
boundary `(n, a)`. *Relative* densities are fixed by the recipe; the *absolute*
scale `ρ_0` (with `ρ̂_i = c_i ρ_0`) comes from θ. Spacing `d_i = 1/ρ̂_i ≈
|b_i| / (2 sin(θ/2))`.

All vectors below are **reconciled signed Miller indices** (the papers' Schmid–
Boas labels carry a sign convention; these signs are chosen so that, per set,
`b_i · slip_plane_i = 0`, `ξ_i · slip_plane_i = 0`, `ξ_i · n = 0`). **`leds_eq11`
and `leds_eq14` are numerically verified exact** (independent Opus check: Frank
residual 1.8e-16 and 3.3e-16; density ratios 1:1 and 1:1:3 match the paper). The
literal `frankus` transcription does **not** satisfy the Frank equation
(residual ≈690%); `frankus` is therefore built **approximate per the paper** —
see its section below.

### `leds_eq11` — two sets, pure twist
- `n = (1, 1, 1)`, `a = [1, 1, 1]`
- set 1: `b=[1,0,-1]`, `ξ=[2,-1,-1]`, `slip_plane=(1,1,1)`, `c=1`
- set 2: `b=[0,1,-1]`, `ξ=[-1,2,-1]`, `slip_plane=(1,1,1)`, `c=1`
- ρ̂₁ = ρ̂₂. Both Burgers in the (111) plane parallel to the wall.

Hand-verification: with unit n, `n×ξ₁ ∝ [0,1,-1]`, `n×ξ₂ ∝ [-1,0,1]`, so for any
in-plane `V`, LHS ∝ `b₁(V_y−V_z) + b₂(−V_x+V_z) = [V_y−V_z, V_z−V_x, V_x−V_y] =
V × [1,1,1] = V × a`. Frank equation holds in direction. ✔

### `leds_eq14` — three sets, mixed tilt/twist
- `n = (1, 1, 1)`, `a = [-1, 3, 1]`
- set 1: `b=[1,0,-1]`, `ξ=[1,0,-1]`, `slip_plane=(1,1,1)`, `c=1`  (screw, b∥ξ)
- set 2: `b=[0,1,-1]`, `ξ=[1,0,-1]`, `slip_plane=(1,1,1)`, `c=1`  (60° mixed)
- set 4: `b=[1,0,1]`,  `ξ=[1,-1,0]`, `slip_plane=(1,1,-1)`, `c=3` (from the other plane)
- ρ̂₁ = ρ̂₂ = ρ̂₄/3  ⇒ relative densities (1 : 1 : 3).
- Checks: b·slip_plane = 0 and ξ·slip_plane = 0 for each set; all ξ lie in the
  boundary (111) (`[1,0,-1]·(111)=0`, `[1,-1,0]·(111)=0`). Note set 4's slip
  plane is **(1,1,-1)**, distinct from the boundary normal — the SB label
  "(11̄1)" reconciled to signed Miller.

### `frankus` — (010) twist, three effective families (APPROXIMATE per the paper)
- `n = (0, 1, 0)`, `a = [0, 1, 0]` (pure twist, a∥n).
- Systems (Schmid–Boas): B2, B5 coplanar on (111); A3, C3 collinear (shared
  Burgers, screw), merged to one family. Line directions (paper p.9):
  `ξ_{B5,B2}=[1,0,-1]`, `ξ_{A3,C3}=[1,0,1]` — orthogonal, both in (010) →
  90° crossed net.
- ⚠ **The literal inset vectors `b_B2=[0,1,1]`, `b_B5=[1,1,0]` are NOT in (111)**
  (`[0,1,1]·[111]=2≠0`) and the network **fails the exact Frank equation**
  (independent Opus check + hand calc: residual ≈690%, b_eff 64.7° off `V×a`
  because the out-of-plane tilt components *add* instead of cancelling). They are
  **not** used verbatim.
- **Approximate-per-the-paper recipe (Sina's decision):** use physically-valid
  in-plane Burgers for B2/B5 (genuine (111) glide directions with *opposite*
  `b·n` so tilt cancels) + collinear screw `b=ξ=[1,0,1]`, with the paper's
  density structure. The exact signed B2/B5 + the density ratio are pinned
  **numerically in Phase 0** to best reproduce the paper's reported near-in-plane
  effective Burgers vector (`b_eff=[3.83,0,-0.03]×10⁻⁴`, `∠(b_eff,V×a)=0.47°` at
  the relaxed densities), accepting that small documented residual. `frankus`
  uses a **loose, documented** Frank tolerance (Component 4), not the strict
  `1e-6` gate of `leds_eq11`/`leds_eq14`.
- **Known discrepancy (flag for Grethe).** An *exact* stress-free (010) twist net
  is reproducible at ratio **1:1:1** — candidate `b_B2=[1,-1,0]`, `b_B5=[0,1,-1]`,
  collinear `[1,0,1]` gives `b_eff ∝ V×a` (residual ~0). This differs from the
  paper's highlighted **Eq. 2 ratio 2:2:1**, while matching the paper's own
  *relaxed DDD* densities (≈1:1:1). We do not resolve this here; it is documented
  in `sidecar`/docs as a candidate to confirm with the co-author (G. Winther).
  It does not block the approximate recipe.

## Components

### 1. `crystal/frank_walls.py` — recipe types + registry + builder (new module)

Pure-physics, numpy-only. It depends on a few existing helpers —
`_ud_matrix_from_bnt` / `_ud_matrix_from_bnt_cell` and `DislocationPopulation`
(currently in `forward_model.py`) and `burgers_magnitude_of`
(`slip_systems.py`). **Circular-import avoidance:** `forward_model`'s `gnb`
branch imports `build_wall_population` with a *function-local* import, so
`frank_walls` may import the Ud helpers + `DislocationPopulation` from
`forward_model` at module level without a load-time cycle. **Preferred
(reviewer's recommendation):** extract `DislocationPopulation` + the two Ud
helpers into a tiny shared module (e.g. `direct_space/population.py`) so neither
side imports the other — removes the ordering foot-gun entirely. Either works; do
NOT module-level-import `frank_walls` from `forward_model`. Public surface:

```python
@dataclass(frozen=True)
class DislocationSet:
    b: tuple[int, int, int]            # Burgers (Miller, crystal frame)
    xi: tuple[int, int, int]           # line direction (Miller)
    slip_plane: tuple[int, int, int]   # the set's slip-plane normal (Miller)
    rel_density: float                  # c_i (relative); absolute = c_i * rho_0

@dataclass(frozen=True)
class WallRecipe:
    name: str
    n: tuple[int, int, int]            # boundary-plane normal (Miller)
    a: tuple[int, int, int]            # misorientation axis (Miller)
    sets: tuple[DislocationSet, ...]
    structure: str = "fcc"
    def validate(self, cell) -> None:  # raises on bad geometry (see Component 5)

RECIPES: dict[str, WallRecipe]         # "leds_eq11", "leds_eq14", "frankus"

def build_wall_population(
    recipe: WallRecipe,
    *,
    theta_deg: float,
    extent_um: float,
    cell,                              # UnitCell for Cartesian + |b|
    ny: float,                         # Poisson ratio (caller resolves from mount/default)
    crystal_to_lab: np.ndarray,        # (3,3) position placement; positions are lab-frame (Phase 0)
    max_dislocations: int | None = None,
) -> DislocationPopulation: ...
```

`build_wall_population` steps:
1. `recipe.validate(cell)`.
2. **Solve θ → densities** (Component 3): `rho_hat_i` (m⁻¹), `d_i = 1/rho_hat_i`.
3. For each set: Cartesian (via `cell`) unit `ξ̂`, `n̂_boundary`; in-plane
   perpendicular `û_i = normalize(n_boundary × ξ_i)`; line count
   `N_i = floor(extent_um*1e-6 / d_i) + 1` (centered); positions in the **crystal
   Cartesian frame** at `(k − (N_i−1)/2) · d_i · û_i`.
4. Enforce `Σ N_i ≤ max_dislocations` (clear error otherwise).
5. Map all positions crystal→lab via `crystal_to_lab`; store in `positions_um`
   (µm) — consumed in the **lab frame**. `Ud` stays crystal-frame (mapped by the
   fixed `Us`, like `wall`/`centered`).
6. Per dislocation: `Ud = [b̂, slip_plane_normal̂, t̂₀]` with edge
   `t̂₀ = normalize(slip_plane × b)` (cubic: `_ud_matrix_from_bnt`; non-cubic:
   `_ud_matrix_from_bnt_cell`); `rotation_deg = signed_angle(t̂₀ → ξ̂)` about the
   slip-plane normal; per-dislocation `|b|` from `burgers_magnitude_of`.
7. Assemble `DislocationPopulation` (per-dislocation `Ud`, `rotation_deg`,
   `b_um_per`; `ny` from the passed `ny`). `sidecar` = provenance dict (recipe
   name, θ, per-set ρ̂/d/N, Frank residual).
8. **Frank residual self-check** (Component 4) before returning.

### 2. Config: `GnbCrystalConfig` + union wiring (`config.py`)

New dataclass + a new `mode = "gnb"` arm of `CrystalConfig`. `centered` / `wall`
/ `random_dislocations` untouched (byte-identical).

```toml
[crystal]
mode = "gnb"

[crystal.gnb]
recipe    = "frankus"      # leds_eq11 | leds_eq14 | frankus | custom
theta_deg = 0.05           # primary knob (deg). d≈|b|/(2 sin θ/2): θ=0.5°→33 nm < 40 nm
                           #   pixel (unresolved!); θ=0.05°→~320 nm ≈ 8 px (resolvable default)
extent_um = 25.0           # in-plane width per family (must exceed the FOV ≈ 20 µm)
# max_dislocations = 20000 # optional cap; clear error at tiny θ

# recipe = "custom" → the general hatch:
# [crystal.gnb.custom]
# n = [1, 1, 1]
# a = [1, 1, 1]
# [[crystal.gnb.custom.set]]
#   b = [1, 0, -1]; xi = [2, -1, -1]; slip_plane = [1, 1, 1]; rel_density = 1.0
# [[crystal.gnb.custom.set]]
#   b = [0, 1, -1]; xi = [-1, 2, -1]; slip_plane = [1, 1, 1]; rel_density = 1.0
```

Fields: `recipe: Literal["leds_eq11","leds_eq14","frankus","custom"]`,
`theta_deg: float` (>0), `extent_um: float` (>0), `max_dislocations: int | None`,
`custom: GnbCustomConfig | None` (required iff `recipe == "custom"`, forbidden
otherwise). `GnbCustomConfig` holds `n`, `a`, and `set: list[GnbSetConfig]`
(each `b`, `xi`, `slip_plane`, `rel_density`). Resolution builds a `WallRecipe`
(named → `RECIPES[...]`; custom → from the sub-block). Works in both
`simplified` and `oblique` geometry. **New parsing surface:** the
`[[crystal.gnb.custom.set]]` list-of-tables is not used by any existing mode, so
`from_dict` needs a small new nested-list parser (with validation). ν is resolved
exactly like the other modes (FCC simplified → `POISSON_RATIO` 0.334; oblique →
`mount.resolved_poisson_ratio`); a test pins the FCC `gnb` path at ν=0.334.

### 3. θ → density solver (inside `frank_walls.py`)

Write the Frank equation as a tensor on in-plane `V`:
`G·V = −2 sin(θ/2) [a]_× V`, with `G = Σ_i ρ̂_i b_i (n×ξ_i)ᵀ = ρ_0 · G_0`,
`G_0 = Σ_i c_i b_i (n×ξ_i)ᵀ` (Cartesian; `b_i` carry physical |b| in metres, `n`
and `ξ_i` unit). Restrict both sides to the 2-D in-plane subspace (basis of two
orthonormal in-plane vectors) and solve the scalar `ρ_0` by least squares:
`ρ_0 = argmin ‖ ρ_0 (G_0 restricted) − (−2 sin(θ/2)[a]_× restricted) ‖`. Then
`ρ̂_i = c_i ρ_0`. The solver **returns its post-fit residual** (the rejected
non-parallel component); callers must not trust `ρ_0` when it is large (the
frankus case) — do not rely solely on the separate Component-4 gate. Units:
`[a]_×` and `(n×ξ)` dimensionless, `b` in metres ⇒ `G` dimensionless ⇒ `ρ_0` in
m⁻¹, so `d_i = 1/ρ̂_i` in metres. (Sanity: `d ≈ |b|/(2 sin θ/2)`.) Verified
non-degenerate for `a∥n` pure twist (eq11 solves cleanly: the frankus failure is
signs, not the `a∥n` geometry).

### 4. Frank residual gate (validation, used by builder + tests)

`frank_residual(recipe, rho_hat, theta_deg) -> float`: max over several random
in-plane `V` of `‖ G·V − (−2 sin(θ/2)[a]_× V) ‖ / ‖2 sin(θ/2)[a]_× V‖`. The
builder asserts against a **per-recipe tolerance** declared on the `WallRecipe`:
strict `< 1e-6` for `leds_eq11` / `leds_eq14` / exact `custom`; a **loose,
documented** tolerance for `frankus` (~ the paper's 0.47° ≈ 1e-2). Strict recipes
are proven stress-free; `frankus` is gated to stay within its documented
approximation (so it can't silently drift).

### 5. Frame placement (`crystal_to_lab`) — pinned by Phase 0

The review settled the frame model (see the ✅ box above): positions are
lab-frame, the field orientation is the fixed `Us`, and the mount does not rotate
the field. The `gnb` branch passes a crystal→lab **position-placement** matrix so
the boundary plane lands at the intended lab orientation. The exact matrix (and
the default placement convention — e.g. boundary normal along a chosen lab axis)
is pinned by the Phase-0 spike and locked by the permanent round-trip test
(rendered line directions ⊥ recipe `n`). `Ud` is unchanged crystal-frame (mapped
by `Us`), keeping g·b contrast consistent with the reflection path.

### 6. Docs + examples
- Short section in `docs/crystal-structures.md` (or a new `docs/gnb-walls.md`):
  the recipe table, θ knob, extent semantics, geometry notes, SB sign
  reconciliation, the "reactions excluded" note.
- Example config(s) under `src/dfxm_geo/data/configs/` (e.g. `gnb_frankus.toml`).

## Phasing (feeds the implementation plan)

- **Phase 0 — spike (pin placement + frankus).** A throwaway script that (a)
  builds the `leds_eq11` net for one θ, renders it, and pins the exact crystal→lab
  **position-placement** matrix so the boundary plane lands correctly (round-trip:
  rendered line directions ⊥ recipe `n`) — the frame model is already settled by
  the review, so this is a placement *decision + verification*, not a frame
  discovery; (b) numerically pins the `frankus` in-plane B2/B5 signs + density
  ratio to reproduce the paper's near-in-plane `b_eff` within its documented
  tolerance. Outputs encoded in Components 4/5 + permanent tests; not merged.
- **Phase 1 — `frank_walls.py`** (types, registry, solver, residual gate) under
  TDD: pure-physics, no forward-model coupling. Unit tests first.
- **Phase 2 — config** (`GnbCrystalConfig` + union + resolution) under TDD.
- **Phase 3 — wiring** (`build_dislocation_population` `gnb` branch using the
  Phase-0 frame map) + e2e forward render.
- **Phase 4 — docs + example configs + golden snapshots.**

## Testing strategy

- **Unit (Phase 1):** Frank residual strict (<1e-6) for `leds_eq11`/`leds_eq14`
  across a θ sweep, and within its documented loose tol for `frankus`; density
  ratios match each recipe (1:1; 1:1:3; frankus per Phase-0); per-set
  `b·slip_plane=0`, `ξ·slip_plane=0`, `ξ·n=0`; spacing scaling
  `d ∝ |b|/(2 sin θ/2)`; `max_dislocations` cap raises cleanly at tiny θ;
  **resolvability guard** (warn/recommend when `d < object_psize`).
- **Frame round-trip (Phase 0/3):** a built wall's effective line directions
  (from `Ud`+`rotation_deg`) are ⊥ the boundary normal after the frame chain —
  locks Component 5 and the `rotation_deg` sign convention.
- **e2e (Phase 3):** forward-render each recipe in **both** geometries
  (simplified + oblique) → finite, non-degenerate output; a small **golden
  snapshot** per recipe (downscaled grid) as a regression lock.
- **Contrast sanity (Phase 3):** choose a reflection with `g·b = 0` for one set
  and confirm that set's contrast drops (ties into the Angle-B g·b physics).
- **Byte-identity (regression):** `wall` / `centered` / `random_dislocations`
  determinism gates unchanged (`test_cubic_bit_identity.py` et al.).
- **Whole-suite + mypy** green before any merge.

## Risks / decisions

1. **Frame placement.** Was the #1 risk; the review settled the frame model
   (positions lab-frame; field via fixed `Us`; mount not in the field path). The
   residual unknown is the exact crystal→lab placement matrix. *Mitigation:*
   Phase-0 spike + permanent round-trip test before building the module.
2. **`frankus` is approximate and disagrees with Eq. 2.** Literal vectors fail
   Frank; an exact net exists at 1:1:1, not the paper's highlighted 2:2:1.
   *Mitigation:* build approximate per the paper (Sina's decision); document the
   residual + discrepancy in `sidecar`/docs; flag for G. Winther; loose
   per-recipe gate so it can't silently drift.
3. **Schmid–Boas sign reconciliation.** Paper labels vs signed Miller; relative
   signs / ξ orientation matter. *Mitigation:* the Frank residual gate is
   automatic and recipe-agnostic; eq11 + eq14 numerically verified as anchors.
4. **Tiny-θ blow-up / under-resolution.** `d ∝ 1/θ` ⇒ huge counts, and small θ
   can put `d` below the 40 nm object pixel. *Mitigation:* resolvable default
   (θ=0.05°), `max_dislocations` cap with a clear error, a resolvability warning,
   and the `d≈|b|/(2 sin θ/2)` formula in docs.
5. **Screw slip-plane ambiguity** (frankus collinear, `ξ = b`). The `n̂` column of
   `Ud` is arbitrary for a screw (field is axisymmetric); pick a fixed valid plane
   and document it; the round-trip test confirms it's inert.
6. **Decoupling from the old `wall`.** Decided: new mode (Approach A), `wall`
   untouched, to protect its byte-identity gates and keep responsibilities clean.

## Definition of done

- `mode = "gnb"` resolves the 3 named recipes + the `custom` hatch; θ-driven;
  physical-width extent; runs in simplified **and** oblique geometry.
- Frank residual gate passes: strict (<1e-6) for `leds_eq11`/`leds_eq14`, and
  within its documented tolerance for `frankus`, across a θ sweep.
- Frame round-trip test passes.
- e2e forward render of each recipe (both geometries) → finite output + golden
  snapshot.
- `g·b = 0` invisibility sanity check passes.
- `wall` / `centered` / `random_dislocations` byte-identity preserved.
- Whole test suite green; mypy clean.
- Docs page + at least one example config landed.
- (Stretch, non-blocking) a rendered figure of each wall in `docs/img/`.
