# GNB Phase-0 spike findings (Task 1)

Date: 2026-06-20
Branch: `feature/gnb-walls`
Scope: pin the crystal→lab position-placement matrix `_CRYSTAL_TO_LAB`, confirm
the `frankus` 1:1:1 net, and determine the `rotation_deg` sign convention.

These results are verified geometrically at the population level (the
displacement field is exact, so no image render is needed) against the real
`dfxm_geo` kernel code. They feed Tasks 4, 5, and 7.

---

## 1. `_CRYSTAL_TO_LAB` = `Us` (the module-level sample→grain rotation)

**Decision: `_CRYSTAL_TO_LAB = Us`** — the fixed module-level `Us` from
`src/dfxm_geo/direct_space/forward_model.py:159`, NOT `np.eye(3)` and NOT `Us.T`.

Exact value (orthonormal, det = +1):

```python
Us = np.array([
    [ 0.7071067812,  0.0         , -0.7071067812],
    [-0.4082482905, -0.8164965809, -0.4082482905],
    [-0.5773502692,  0.5773502692, -0.5773502692],
])
```

### Code-grounded derivation

The frame chain in the kernel (`crystal/dislocations.py:296-300`, and the fused
kernel `forward_model.py:1660-1687` / `dislocations.py:468-474`) is, per ray
`rl` (lab µm) and dislocation `d`:

```
rd = Ud.T @ Us.T @ S.T @ Theta @ (rl - offset)
```

`offset = positions_um` is subtracted in the **lab frame** before any rotation.
In simplified geometry `S = Theta = I`, so `rd = Ud.T @ Us.T @ (rl - offset)`.

Two consequences fix the placement matrix:

- **The field's line direction lives in the lab frame via `Us`.** We built `Ud`
  + `rotation_deg` so the dislocation's geometric line equals `ξ` expressed in
  the **crystal/grain frame** (verified in §3). `Us.T` maps lab→grain, so its
  inverse `Us` maps a grain-frame direction to the lab frame:
  `line_lab = Us @ ξ_crystal`. The crystal mount `U_mount` is **never** in this
  path — it appears only in the reflection/Q path (`oblique.py:364`,
  `compute_omega_eta`). Confirmed by reading both call sites.
- **The position comb must be mapped the same way as the field**, or the spatial
  arrangement of the parallel lines lives in a different lab frame than the
  strain field they carry — an incoherent (smeared) boundary. So the only
  self-consistent placement is `crystal_to_lab = Us` (the same map the field
  applies to orientations).

### Numerical evidence (the round-trip-under-constraint nuance, addressed)

With boundary normal `n = (1,1,1)` and `n_lab = R @ n̂`, requiring BOTH the line
directions and the position offsets to lie in the boundary plane:

| `R = crystal_to_lab` | `max\|positions · n_lab\|` | `max\|line_lab · n_lab\|` |
|----------------------|---------------------------:|--------------------------:|
| `np.eye(3)`          | 2.6e-17 (in-plane)         | **0.32 / 0.35 (lines pierce!)** |
| **`Us`**             | **1.55e-17**               | **6.7e-17 (lines in-plane)** |
| `Us.T`               | 3.5e-17 (in-plane)         | **0.30 / -0.88 (lines pierce!)** |

**This is exactly the nuance the brief warned about.** The permanent Task-7
round-trip test `max(|positions_um @ n_lab|) < 1e-6` with
`n_lab = _CRYSTAL_TO_LAB @ n̂` holds for **all three** orthogonal candidates,
because positions are built as `R @ (in-plane vector ⊥ n̂)` and stay ⊥ `(R @ n̂)`
for any orthogonal `R`. So the position test **alone does not pin the matrix** —
it only confirms orthogonality + in-plane.

The discriminator is the **field line direction**: only `Us` makes the
field-borne lines lie in the boundary plane (`line_lab · n_lab ≈ 0`).
`eye` and `Us.T` produce lines that pierce the boundary by 18–60°, i.e. an
incoherently-oriented wall. So `np.eye(3)` is **physically incorrect** here;
`Us` is required.

> ⚠ Task 7 must therefore strengthen the round-trip test beyond the positions
> inequality: also assert the per-set **field line direction**
> `line_lab = Us @ ξ_crystal` (equivalently the `Ud`+`rotation_deg`-reproduced
> line) is ⊥ `n_lab`. The positions-only inequality cannot catch a wrong
> placement matrix.

---

## 2. `frankus` confirmed at 1:1:1 (literal vectors rejected; 2:2:1 flagged)

Boundary `(010)` twist: `n = a = [0,1,0]`. Test direction `V = [0,0,-1]`, target
`V × a = [1,0,0]`. The effective Burgers content is
`b_eff(V) = Σ_i c_i (|b| b̂_i) ((n × ξ̂_i) · V)`.

### 1:1:1 candidate (Sina's decision) — EXACT

Sets: `b_B5=[1,-1,0]`, `b_B2=[0,1,-1]` (coplanar (111) glide, opposite `b·n` so
tilt cancels), collinear screw `b=ξ=[1,0,1]`; line dirs `ξ_{B5,B2}=[1,0,-1]`,
`ξ_screw=[1,0,1]`.

- `b_eff(V) = [1, 0, 0]`  →  **`b_eff ∝ V × a`**.
- **`∠(b_eff, V × a) = 0.0000°`**.
- **Directional Frank residual = 0.0%** (max over random in-plane `V`).

This is the exact in-plane stress-free net at ratio 1:1:1.

### Literal paper vectors — REJECTED

Sets `b_B2=[0,1,1]`, `b_B5=[1,1,0]` (with the same collinear screw):

- `b_B2 · [111] = 2 ≠ 0` — **not in the (111) glide plane** (out-of-plane).
- `b_eff(V) = [1,1,1]`, **`∠(b_eff, V × a) = 54.74°`**.
- **Directional Frank residual = 111%** (clearly fails; the out-of-plane tilt
  components add instead of cancelling).

> Note on the residual magnitude: the brief quoted "≈690%" for the literal
> transcription; that figure comes from a different normalization (the raw
> `‖G·V − rhs‖ / ‖rhs‖` with un-fitted `ρ_0`, which inflates because the wrong
> net also mis-scales). My number (111%) uses the **scale-free directional**
> residual (ρ_0 fitted by least squares, so only the directional mismatch
> remains). Both agree on the conclusion: the literal vectors fail badly and
> are rejected. Tasks 3/4 should pick ONE residual definition and document it;
> the directional one is recommended because it isolates the physics from the
> ρ_0 fit.

### 2:2:1 (paper Eq. 2 highlighted ratio) — NOT stress-free, flag for G. Winther

Using the valid 1:1:1 vectors but the paper's highlighted **2:2:1** densities:

- **Directional Frank residual = 40%**, `∠(b_eff, V × a) = 18.4°`.

So the paper's highlighted Eq. 2 ratio (2:2:1) is **not** stress-free with these
in-plane vectors, while **1:1:1 is exact** (and matches the paper's own relaxed
DDD densities ≈ 1:1:1). **Documented discrepancy — confirm with co-author
G. Winther.** Build 1:1:1 (Sina's decision); gate `frankus` with a loose,
documented Frank tolerance so it can't silently drift.

### Anchor recipes re-confirmed exact

- `leds_eq11` (1:1, `n=[111]`, `a=[111]`): directional residual **0.0%**.
- `leds_eq14` (1:1:3, `n=[111]`, `a=[-1,3,1]`): directional residual **0.0%**.

---

## 3. `rotation_deg` sign convention — use `Ud[:,2]`, NOT the plan's `slip_plane × b`

**Determination: the rotation_deg the kernel needs is**

```python
rotation_deg = _signed_angle(Ud[:, 2], xi_hat, slip_plane_normal)
#   _signed_angle(u, v, axis) = degrees(atan2(cross(u,v)·axis, u·v))
```

where `Ud[:, 2]` is the **actual post-flip third column** of the matrix returned
by `_ud_matrix_from_bnt(b, n, t)` (or `_ud_matrix_from_bnt_cell`).

### Why NOT the plan's literal formula

The plan (Task 5) defines `edge_t0 = slip_plane × b` (= `n × b`) and
`rotation_deg = _signed_angle(edge_t0 → ξ)`. **This is wrong by more than a
global sign.** The reason:

- `_ud_matrix_from_bnt` **flips the `t` column** whenever the raw column-stack
  has `det < 0` (to keep `Ud` a proper rotation; `forward_model.py:1247-1248`).
  For every leds set, this flip fires, so the **actual** edge reference axis is
  `Ud[:,2] = b × n = −(slip_plane × b)`, i.e. the negative of the plan's
  `edge_t0`.
- Because `n × b` is the *reflection* of `b × n` through the `n`-axis, simply
  negating the plan's angle does **not** recover the correct value. Verified:
  `−_signed_angle(slip_plane × b → ξ)` reproduces `ξ` only for the pure-screw
  set; for the mixed sets it lands on the wrong line (alignment 0.5 / −0.5).

### Evidence (geometric line model, exact for all 5 sets)

The kernel's combined edge+screw field reproduces the geometric line
`line = cos(α)·b̂ + sin(α)·t̂` with `α = 90° − rotation_deg`, `t̂ = Ud[:,2]`.
Brute-forcing the `rotation_deg` that makes `line == ξ` and comparing to the
candidate formulas:

| set | `b·ξ` angle α | best `rot` (line==ξ) | `_signed_angle(Ud[:,2]→ξ)` | `_signed_angle(n×b→ξ)` |
|-----|--------------:|---------------------:|---------------------------:|-----------------------:|
| eq11 s1 (60° mixed) | 30° | +60.00 | **+60.00 ✓** | −120.00 ✗ |
| eq11 s2 (60° mixed) | 30° | +120.00 | **+120.00 ✓** | −60.00 ✗ |
| eq14 s1 (screw)     | 0°  | +90.00 | **+90.00 ✓** | −90.00 |
| eq14 s2 (60° mixed) | 60° | +30.00 | **+30.00 ✓** | −150.00 ✗ |
| eq14 s4 (60° mixed) | 60° | +150.00 | **+150.00 ✓** | −30.00 ✗ |

`_signed_angle(Ud[:,2] → ξ)` reproduces `ξ` to align = **+1.00000** for **all
five** sets; `line == ξ` exactly (`np.allclose` True). The plan's `n × b`
formula matches only the screw.

### Answer to "does `_signed_angle` need negation?"

**Not a simple negation.** The robust fix is to **reference the post-flip
`Ud[:, 2]`** as the edge axis in `_signed_angle`, not `slip_plane × b`. If Task 5
keeps an explicit `edge_t0`, it must set `edge_t0 = Ud[:, 2]` (build `Ud` first,
then read its third column) so the det-flip is automatically honored. The
parameterization the kernel documents (`Fd_find_mixed` docstring,
`dislocations.py:251-258`): `rotation_deg = 0` ⇔ pure edge, `90` ⇔ pure screw,
`α_paper = 90° − rotation_deg`, with the edge reference being the **actual `Ud`
`t̂` column**.

> Note on the failed naive probe: a finite-difference "field invariance axis"
> probe is NOT a reliable way to recover a mixed dislocation's geometric line
> (the superposed edge+screw field's numerical null direction is not the
> geometric line axis). The **geometric line model** `cos(α)b̂ + sin(α)t̂` is the
> correct, exact framework and is what Task 8's g·b / golden tests will confirm
> end-to-end.

---

## 4. Caveats for Tasks 4 / 5 / 7

1. **(Task 5)** Compute `rotation_deg = _signed_angle(Ud[:, 2], ξ̂, n̂)` using the
   **post-flip** `Ud[:, 2]` from `_ud_matrix_from_bnt(_cell)`. Do NOT use
   `slip_plane × b` directly — the det-flip silently negates it and a plain sign
   flip does not recover the right line for mixed sets.
2. **(Task 5 / 7)** Pass `_CRYSTAL_TO_LAB = Us` (import the module-level `Us`
   from `forward_model`, or the shared population module if it is extracted).
   Map only **positions** by it; `Ud` stays crystal-frame (mapped by `Us` inside
   the kernel, exactly like `wall`/`centered`).
3. **(Task 7)** Strengthen the round-trip test: the positions-⊥-`n_lab`
   inequality is **necessary but not sufficient** (it passes for `eye`/`Us`/`Us.T`
   alike). Add an assertion that the field **line direction**
   `line_lab = Us @ ξ_crystal` (or the `Ud`+`rotation_deg` line) is ⊥ `n_lab`.
   That is the assertion that actually pins `_CRYSTAL_TO_LAB = Us`.
4. **(Task 4)** Pick a single Frank-residual definition and document it. The
   **directional** residual (ρ_0 fitted out) is recommended: 0% for
   `leds_eq11` / `leds_eq14` / `frankus@1:1:1`, ~40% for `frankus@2:2:1`, ~111%
   for the literal frankus vectors. The strict gate `<1e-6` applies to the first
   three; `frankus` uses a loose documented tolerance.
5. **(Task 4 / docs)** Record the `frankus` 2:2:1-vs-1:1:1 discrepancy in the
   recipe `sidecar` + docs and flag it for G. Winther (paper Eq. 2 highlights
   2:2:1, which is not stress-free; we build the exact 1:1:1 net, matching the
   paper's relaxed DDD densities).
6. **(Task 5)** Screw sets (`ξ = b`, e.g. eq14 s1, frankus collinear) have an
   arbitrary `n̂` column for `Ud`; pick the recipe's stated `slip_plane`, which is
   valid (`b·slip_plane = 0`). The round-trip confirms it is inert for the line
   geometry (the screw line equals `±b` regardless).

---

## Summary (one-liners for the registry / builder)

- `_CRYSTAL_TO_LAB = Us` (module-level rotation; positions-only map).
- `rotation_deg = _signed_angle(Ud[:, 2], ξ̂, slip_plane_normal)` (post-flip `Ud`
  col 2; reproduces `ξ` exactly for all leds sets).
- `frankus` = 1:1:1, exact (0% directional residual); literal rejected; 2:2:1
  flagged for G. Winther.
