# Physics

The model the code implements, and the conventions used. For the code
organization, see `docs/architecture.md`. For the full derivation, see
the published paper:

> Borgi, S. et al. *J. Appl. Cryst.* (2024).
> [DOI: 10.1107/S1600576724001183](https://doi.org/10.1107/S1600576724001183)

## What is DFXM

**Dark Field X-ray Microscopy (DFXM)** is a synchrotron technique that
images strain fields and crystal defects in 3D. A monochromatic
collimated X-ray beam illuminates a sample; a Bragg-reflected beam
passes through an objective lens, which selects rays around a chosen
direction in reciprocal space (the "dark field"); a 2D detector records
the resulting image.

The image intensity at a detector pixel is, to a good approximation, the
integral over a small reciprocal-space volume around the chosen
direction of the sample's reciprocal-space density. Different (phi, chi,
2θ) goniometer settings probe different volumes — the goal of a *forward
model* is to predict the image at each setting given the sample's strain
field.

The ID06 beamline at the ESRF is the reference setup (`Npixels=510`,
`psize=40 nm`, Bragg angle θ₀ ≈ 8.98° for the (-1, 1, -1) reflection at
17.0 keV); other beamlines are accommodated by overriding the constants
in `dfxm_geo.constants`.

## Coordinate frames

Three right-handed orthonormal frames are used; the rotation matrices
are stored module-level in `direct_space/forward_model.py`:

| Symbol | Frame      | Meaning                                                |
|--------|------------|--------------------------------------------------------|
| `rl`   | Lab        | Frame fixed to the beamline; x_l along the beam        |
| `rs`   | Sample     | Goniometer-mounted sample; defined by phi/chi/2θ       |
| `rc`   | Crystal    | The crystal's principal axes                           |
| `rd`   | Dislocation| Aligned with the dislocation's Burgers vector + line   |

The matrices that take you between them, as defined in the paper:

- `Theta`: lab ← sample (Bragg rotation by θ₀ around y_l)
- `Us`: sample ← crystal (depends on the specimen orientation)
- `Ud`: crystal ← dislocation (depends on which slip system is active)

A vector `r` in lab coords transforms to dislocation coords by

```
rd = Ud.T @ Us.T @ Theta @ rl
```

The composed transpose form appears throughout `Fd_find`. This is
*passive* rotation — the vector is unchanged, only its coordinate
representation changes.

### Sample remount (Purdue 2024)

In addition to the lab/sample/crystal/dislocation frames, `Fd_find` supports a
sample-remount rotation `S` inserted between sample and crystal frames. This
models a physical operation: the sample is removed from the goniometer and
remounted in a different (symmetry-equivalent) orientation. The four named
constants `S1` (identity), `S2`, `S3`, `S4` are ported verbatim from the
Purdue paper (`dfxm_geo.crystal.remount`). They are proper rotations from the
cubic point group; their numerical traces differ (S2 and S4 are ~109.47°
rotations; S3 is ~70.53°) so they are not three rotations about a single
axis. Cleanup ports them as-is and does not re-derive their geometric
interpretation.

## Edge dislocation displacement field

For an isolated edge dislocation in an isotropic linear-elastic medium
(Hirth & Lothe), the displacement gradient at point `(x_d, y_d, z_d)` in
dislocation coords is

```
F_d[0,0] = -y_d (3 x_d² + y_d² - 2ν (x_d² + y_d²)) / ((x_d² + y_d²)²) · b / (4π(1-ν))
F_d[0,1] =  x_d (3 x_d² + y_d² - 2ν (x_d² + y_d²)) / ((x_d² + y_d²)²) · b / (4π(1-ν))
F_d[1,0] = -x_d (3 y_d² + x_d² - 2ν (x_d² + y_d²)) / ((x_d² + y_d²)²) · b / (4π(1-ν))
F_d[1,1] =  y_d (x_d² - y_d² + 2ν (x_d² + y_d²)) / ((x_d² + y_d²)²) · b / (4π(1-ν))
```

with `b` the Burgers vector magnitude (default `BURGERS_VECTOR = 2.862e-4 µm`
for Al) and `ν` the Poisson ratio (`POISSON_RATIO = 0.334`). The full
gradient tensor is `F = I + F_d`. A small `α = 1e-20` is added in the
denominator to keep `F` finite at the dislocation core itself.

> **Sign convention on `F_d[1,1]`**: the `+2ν` on the `[1,1]` component
> (vs `-2ν` on the other three) follows the correction documented in
> Appendix A of [Borgi et al., *J. Appl. Cryst.* (2024)](https://doi.org/10.1107/S1600576724001183).
> Pre-correction versions of the code (including stale `main` snapshots)
> used `-2ν` on `[1,1]` — see commit `3b71b33` in this branch.

`Fd_find` builds this field on the lab-coordinate grid `rl`, optionally
summing contributions from multiple parallel dislocations stacked in a
wall along `y_d`. The wall is bipolar — for index `i ∈ [1, ndis)` the
offsets are `+1·dis, -1·dis, +2·dis, -2·dis, …` so the wall is centered
on the origin. For `ndis > 100` the loop is sharded across a
`ThreadPoolExecutor` (one chunk per CPU); for `≤100` it runs serially.
Both branches use the same deterministic per-`i` offset formula, so
results are identical (up to floating-point order-of-summation) across
the branch boundary.

The function returns `Fg = Ud @ F_d @ Ud.T`, the gradient field rotated
back into the *grain* frame (so downstream code can use the same
sample-frame transforms regardless of how many slip systems contribute).

## Strain → image

Given `Fg`, the forward model computes a per-ray scattering vector in
imaging space:

```
H_g = (F_g⁻¹)ᵀ - I               # in dfxm_geo.io.strain_cache
q_s = U_s · H_g · q_hkl          # sample-space scattering vector
q_c = q_s + (phi, chi, ΔθΔ_2θ)    # add goniometer offsets
q_i = Theta · q_c                # imaging-space scattering vector
```

Each ray's `q_i` is voxelized into the precomputed reciprocal-space
resolution kernel `Resq_i[i, j, k]`. Rays inside the kernel contribute
their probability (times the beam-profile weight `prob_z`) to the
detector pixel they hit; rays outside are discarded.

`Resq_i` is computed once per beamline geometry by
`dfxm_geo.reciprocal_space.resolution.reciprocal_res_func` (Monte Carlo
with ~10⁸ rays for paper-quality results) and saved as a pickle. Its
generation is a separate offline step — see
`dfxm_geo.reciprocal_space.kernel`.

## Approximations and limits

The model is **geometrical** — diffraction-limited resolution effects
beyond the objective NA + physical aperture are not modeled. In
particular:

- Wavefront curvature inside the sample is neglected.
- Multiple scattering is neglected (single-Bragg approximation).
- The crystal is assumed pseudo-perfect outside the dislocation cores;
  no plastic relaxation or work hardening is modeled.
- The dislocation displacement field uses the isotropic, infinite-body
  Hirth & Lothe solution. Surfaces and grain boundaries are ignored.

For samples with `ndis ≥ 7501` at `dis ≤ 0.25 µm` spacing the edge
effects become non-negligible — those configurations require the
parallel branch and quite a bit of memory.

## Default reflection

The code defaults to the `(-1, 1, -1)` reflection (set by
`HKL_DEFAULT` in `dfxm_geo.constants`). To use a different reflection,
pass `h, k, l` to `Find_Hg` explicitly:

```python
Hg, q_hkl = Find_Hg(dis=4, ndis=151, psize=40e-9, zl_rms=zl_rms,
                    h=0, k=2, l=0)
```

The reciprocal-space resolution kernel will need to be regenerated for
the new reflection — Resq_i is reflection-specific because the
out-of-plane resolution depends on the Bragg angle.

## Notation reference

| Symbol     | Code name    | Meaning                                    |
|------------|--------------|--------------------------------------------|
| b          | `b`          | Burgers vector magnitude (µm)              |
| ν          | `ny`         | Poisson ratio                              |
| φ          | `phi`        | Rotation around y_l (rad)                  |
| χ          | `chi`        | Rotation around x_l (rad)                  |
| 2Δθ        | `TwoDeltaTheta` | 2θ shift off Bragg (rad)                |
| θ₀         | `theta_0`    | Bragg angle (rad)                          |
| H_g        | `Hg`         | Displacement gradient field (grain frame)  |
| F_g        | `Fg`         | Strain gradient (grain frame); `Hg = (F_g⁻¹)ᵀ - I` |
| q_hkl      | `q_hkl`      | Reciprocal-lattice vector for the reflection |
| Resq_i     | `Resq_i`     | Voxelized reciprocal-space resolution function |
| ndis       | `ndis`       | Number of parallel dislocations in the wall |
| dis        | `dis`        | Dislocation spacing in y_d (µm)            |

## References

1. Borgi, S. et al. *J. Appl. Cryst.* (2024).
   DOI: 10.1107/S1600576724001183.
2. Poulsen, H. F. et al. *J. Appl. Cryst.* (2017) — the foundational
   DFXM resolution-function paper that `reciprocal_res_func` is based
   on. See the header comments in
   `src/dfxm_geo/reciprocal_space/resolution.py`.
3. Hirth, J. P. & Lothe, J. *Theory of Dislocations* (Wiley, 1982) —
   the canonical reference for the edge-dislocation displacement field
   used by `Fd_find`.
