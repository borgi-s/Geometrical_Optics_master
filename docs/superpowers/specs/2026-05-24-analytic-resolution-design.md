# Analytic reciprocal-space resolution function — design

**Date:** 2026-05-24
**Status:** Draft (math verified 2026-05-24; pending user review)
**Author:** Sina Borgi (with Claude Code)

## 1. Motivation

dfxm_geo currently builds its reciprocal-space resolution function (`Resq_i`)
by Monte-Carlo: `reciprocal_res_func` draws ~1e8 rays of 5 instrument
variables, pushes them through a linear diffraction transform, and histograms
the result onto a 3-D grid (`dfxm-bootstrap`, ~50 s). `forward()` then uses
that grid as a lookup table.

Two costs follow from the grid:

1. **Quantization artifacts.** The COM/mosaicity "banding" fixed in v2.0.2
   (`com_dequantize_v202`) and the `qi_range`-as-wrong-de-bin-lever trap both
   stem from discretizing the resolution function onto `Resq_i`. A grid-free
   resolution function removes this entire artifact class.
2. **Shot noise + bootstrap overhead** — a cached `.npz` per reflection, plus
   `qi_range`/`npoints` tuning knobs.

Axel Henningsson's `darkmod` (the `darling` author) showed the resolution
function can be **closed-form**: for *untruncated* Gaussian instrument
variables it is exactly `N(M·mu, M·Sigma·M^T)` ("vectorized Poulsen 2017").
A 2026-05-24 parity check (`C:\Users\borgi\tmp\darkmod_parity\`) confirmed this
to <0.03% against our own sampler — **but** our beam has a hard physical
truncation that `darkmod`'s closed form cannot represent:

> Carlsen et al. 2022 (Acta Cryst A78 482–490): the incident-beam vertical
> divergence is a truncated Gaussian — FWHM Δζ_v = 0.53 mrad from the condenser
> lens, **hard-clipped at ±140 µrad by the condenser's physical aperture**.

Our `zeta_v` clip at `±1.4e-4` (`resolution.py:187-188`) is exactly this. At
our FWHM it sits at ±0.62σ — it dominates the rocking-axis (`qrock'`) shape,
making it a near-rectangular box, not a Gaussian.

**Key realization:** the truncation is on a *single* variable and the map is
*linear*, so the resolution function is a convolution of one truncated-normal
with Gaussians — which has a **closed form** (an `erf` difference). We can keep
`darkmod`'s grid-free elegance *and* the real condenser physics.

## 2. Objective & scope

- **Primary objective:** grid-free fidelity. Eliminate the quantization /
  COM-banding artifact class. Speed (no bootstrap, no `.npz`) is a welcome
  bonus, not the driver.
- **Coverage:** no-beamstop configurations only (the v2.0.0 "simple" default).
  The wire/knife-edge/aperture beamstop is stochastic/geometric and stays on
  the MC path.
- **Reference & acceptance bar:** the existing MC kernel is the validated
  physics reference. The analytic backend must (a) reproduce MC forward images
  within the MC kernel's shot-noise envelope on a no-beamstop case, and
  (b) produce a demonstrably banding-free COM/rocking map.

## 3. Approaches considered

| | Approach | Verdict |
|---|---|---|
| **A** | Fully closed-form `p_Q` (`exp × erf-diff`), grid-free, evaluated per-ray in `forward()`. Drops the ±3.3σ NA square aperture. | **TARGET** — only option that fully delivers grid-free fidelity. |
| **B** | Semi-analytic: marginalize Gaussians analytically, integrate truncated variables (incl. NA aperture) by deterministic Gauss–Legendre quadrature. | **ORACLE + FALLBACK** — used to validate A; drop-in if A's aperture approximation ever exceeds tolerance. |
| **C** | Compute `p_Q` analytically once onto the existing Q-grid, keep the `Resq_i` LUT. | **REJECTED** — reintroduces the grid we are trying to remove. |

Build **A**, validate against **B** (and MC). Order: A first, then B.

## 4. The closed-form `p_Q` (Approach A)

### 4.1 Variables and the linear map

Instrument variables `x = [eps, zeta_v, zeta_h, delta_2theta, xi]`:

| var | distribution | parameter |
|---|---|---|
| `eps` | `N(0, σ_eps²)` | `σ_eps = eps_rms` |
| `zeta_v` | `TruncNormal(0, σ_zv², [−c, c])` | `σ_zv = zeta_v_fwhm/2.355`, `c = 1.4e-4` |
| `zeta_h` | `N(0, σ_zh²)` | `σ_zh = zeta_h_fwhm/2.35` |
| `delta_2theta` | `N(0, σ_NA²)` | `σ_NA = NA_rms` |
| `xi` | `N(0, σ_NA²)` | `σ_NA = NA_rms` |

The NA square aperture (`phys_aper` rejection) is **dropped** in A; its effect
is validated < tolerance (§6.3).

Linear transform (verbatim from `resolution.py:250-256`, `dphi=0`):

```
qrock       = -zeta_v/2 - delta_2theta/2
qroll       = -zeta_h/(2 sinθ) - xi/(2 sinθ)
qpar        =  eps + cotθ·(-zeta_v/2 + delta_2theta/2)
qrock_prime =  cosθ·qrock + sinθ·qpar
q2th        = -sinθ·qrock + cosθ·qpar
```

So `q = [qrock_prime, qroll, q2th] = M·x`, with `M` (3×5) built numerically as
`M[:,j] = transform(e_j)`.

### 4.2 Decomposition and the closed form

Split off the truncated scalar `u = zeta_v`; let `g = [eps, zeta_h,
delta_2theta, xi]` be the Gaussian 4-vector with `Σ_g = diag(σ_eps², σ_zh²,
σ_NA², σ_NA²)`. With `m_u = M[:, zeta_v]` (3-vector) and `M_g` the other four
columns (3×4):

```
q = m_u·u + M_g·g
```

Conditional on `u`: `q | u ~ N₃(m_u·u, C_rest)`, `C_rest = M_g Σ_g M_gᵀ`.
Marginalizing over the bounded `u`:

```
p_Q(q) = ∫_{−c}^{+c} f_u(u) · N₃(q; m_u·u, C_rest) du
```

`f_u(u) = exp(−u²/2σ_zv²)/Z_trunc`, `Z_trunc = σ_zv√(2π)[Φ(c/σ_zv) −
Φ(−c/σ_zv)]`. The integrand is Gaussian in `u`; completing the square with

```
A      = m_uᵀ C_rest⁻¹ m_u + 1/σ_zv²
B(q)   = m_uᵀ C_rest⁻¹ q
Q₀(q)  = qᵀ C_rest⁻¹ q
μ_u(q) = B(q)/A
```

gives the **closed form**:

```
p_Q(q) = N₀ · exp(−½[Q₀(q) − B(q)²/A]) · ½[ erf((c−μ_u)√(A/2)) − erf((−c−μ_u)√(A/2)) ]
N₀     = (1/Z_trunc) · (2π)^(−3/2) · |C_rest|^(−½) · √(2π/A)
```

> **VERIFIED 2026-05-24** (independent skeptical re-derivation + numerics,
> scratch in `C:\Users\borgi\tmp\math_check\`). Closed form vs deterministic
> 1-D quadrature agree to **2.3e-13** (machine precision); vs brute-force MC
> the full 3×3 covariance matches to **<0.1%** and box-probabilities to
> 0.01–0.4% (sampling noise); global normalization `∫_{R³} p_Q = 1` to
> **7e-15**. Every constant (`A`, `B`, `Q₀`, `μ_u=B/A`, the `½` erf prefactor,
> the `√(A/2)` argument, `N₀`) confirmed term-for-term. No sign or factor
> errors.

Everything is vectorized over the ray array `q`: `Q₀`, `B` are forms through a
single precomputed solve against `C_rest`; `erf` is elementwise.

**Precomputed once at load:** `M`, a Cholesky factor of `C_rest` (for stable
solves — see §7 on conditioning), `det C_rest`, `A`, `Z_trunc`, `N₀`. Per-call
cost is O(N_rays) matrix-vector + `exp` + `erf`.

**Normalization for image comparison:** peak-normalize `p_Q` to 1 (matching the
MC `Resq_i/Resq_i.max()` convention) using the analytic mode, so analytic and
MC forward images are directly comparable.

## 5. Architecture & integration

Five pieces; the MC path is untouched.

### 5.1 `AnalyticResolution` (new: `reciprocal_space/analytic_resolution.py`)

Constructed from the same instrument parameters the MC bootstrap consumes
(`theta`, `zeta_v_fwhm`, `zeta_h_fwhm`, `NA_rms`, `eps_rms`, clip `c`).
Precomputes `M`, `C_rest⁻¹`, `A`, `Z_trunc`, `N₀`, the peak normalizer, and the
conditioning guard. Exposes:

```python
class AnalyticResolution:
    def __init__(self, *, theta, zeta_v_fwhm, zeta_h_fwhm, NA_rms, eps_rms,
                 zeta_v_clip=1.4e-4): ...
    def __call__(self, qi: np.ndarray) -> np.ndarray:
        """qi: (3, N) imaging-space scattering vectors -> prob: (N,)."""
```

One clear purpose (evaluate the closed-form resolution density), a well-defined
interface (`qi → prob`), depends only on numpy/scipy + the instrument params.

### 5.2 Builder/loader

Analogous to `_load_default_kernel()` but with no file I/O: build the
`AnalyticResolution` from config and register it as the active resolution
evaluator for `forward()`. Sets the `theta_0` module state `forward()` already
reads.

### 5.3 `forward()` seam

One minimal branch at the current LUT line (`forward_model.py:498`):

```python
if _analytic_eval is not None:
    prob = _analytic_eval(qi) * prob_z          # all rays; no grid mask
else:
    # existing LUT path verbatim
    prob = Resq_i[index1[idx], index2[idx], index3[idx]] * prob_z[idx]
```

The analytic path skips the `index1/2/3` floors and the grid-bounds `idx` mask
(the closed form returns ≈0 in the tails). The `qs→qc→qi` build and the
`bincount` accumulation are unchanged.

### 5.4 Backend dispatch (config)

`[reciprocal] backend` with values `"auto"` (default), `"analytic"`, `"mc"`:

| backend | beamstop off | beamstop on |
|---|---|---|
| `auto` (default) | **analytic** | MC |
| `analytic` | analytic | **ValueError** (clear: switch to mc / disable beamstop) |
| `mc` | MC | MC |

So with the default `auto`, no-beamstop runs use the grid-free analytic path
and beamstop runs transparently use MC. This default-flip is a release-worthy
behavior change, **gated on the §6 validation suite passing**.

### 5.5 MC path

`reciprocal_res_func`, `kernel.py`, the `Resq_i` LUT, and the existing
`forward()` lookup are unchanged. Analytic is purely additive.

## 6. Validation (the gate for the default-flip)

Cheapest-first; e2e cases use scaled-down grids (5×5 scan, 64–128 px) per the
smoke-test rule. No external data required.

1. **Closed-form vs quadrature (Approach B as oracle).** B integrates the same
   `p_Q` by deterministic Gauss–Legendre over the `zeta_v` box. Assert A ≡ B
   (no NA aperture) to quadrature precision — validates the algebra with zero
   MC noise.
2. **Closed-form vs MC sampling.** Promote the parity prototype to a test:
   sample the 5 vars (with `zeta_v` truncation), histogram, compare to `p_Q`
   on a grid + check the 3×3 covariance — match within MC noise.
3. **NA-aperture tolerance check** (justifies A's one approximation). B-with-
   aperture vs A-without: assert the dropped ±3.3σ objective aperture changes
   the covariance and a forward image by **< 1%**. If it fails, A falls back to
   B (aperture kept via quadrature).
4. **Analytic vs MC forward image.** Small no-beamstop case: assert per-pixel
   agreement within the MC kernel's shot-noise envelope (reference = high-
   `Nrays` MC).
5. **COM cleanliness demo** (headline). Single-dislocation rocking case from
   `com_dequantize_v202`: the analytic COM/mosaicity map shows none of the grid
   banding. Asserted as reduced quantization vs a smooth reference; before/after
   figure saved.
6. **Determinism golden + housekeeping.** `p_Q` is RNG-free → small golden
   array for regression (à la the `Fd_find` golden). `mypy src/dfxm_geo/`
   clean; `python -m pytest -q` green.

**Performance** (bonus, non-gating): time analytic forward vs MC
bootstrap+forward and record.

## 7. Edge cases & error handling

- **`beamstop=True` + explicit `backend="analytic"`** → `ValueError` with a
  switch-to-mc message.
- **`C_rest` is intrinsically ill-conditioned** — `qrock'` gets Gaussian
  spread *only* from `eps` (the `qrock'` row of `M_g` is `[0.156, 0, 0, 0]`;
  `zeta_h`/`delta_2theta`/`xi` contribute nothing to it). At the nominal config
  the condition number is ≈1.7e4 (min eigenvalue 8.7e-11 = `0.156²·σ_eps²`, max
  1.5e-6). This is fine but demands a **numerically stable solve**: factor
  `C_rest` once via Cholesky and `solve` for `Q₀`, `B` — never form an explicit
  `inv()`.
- **`eps_rms → 0` is a genuine singularity, NOT something to regularize away.**
  As `σ_eps → 0`, `C_rest` drops to rank 2 (min eig ∝ `σ_eps²` → 0) and the true
  marginal collapses onto a 2-D plane — it has *no* density in ℝ³, so a finite
  `p_Q(q)` does not exist. The formula correctly blows up. **Ridge
  regularization (`C_rest + λI`) is unsound** — it injects fictitious variance
  into all three q-directions, including the degenerate `qrock'`, fabricating a
  broader, wrong distribution (and perturbing `N₀` via `|C|^{−½}`). Handling:
  guard with a conditioning check; if `C_rest` is genuinely degenerate (cond
  beyond a threshold, i.e. `σ_eps` pathologically small), **raise a clear error**
  directing the user to a finite energy bandwidth. A proper 2-D-projected
  resolution function (project out the `qrock'` null direction) is a possible
  future extension, not in this scope. Physically `eps_rms > 0` always, so the
  nominal path never hits this.
- **`zeta_v_fwhm = 0`** (no vertical divergence) → `σ_zv → 0`; truncation
  collapses to `u ≡ 0`; `p_Q` becomes a plain 3-D Gaussian `N₃(0, C_rest)`.
  Handle as a clean branch (no erf term).
- **No truncation at all** → `p_Q = N₃(0, M Σ Mᵀ)` exactly — `darkmod`'s
  `PentaGauss`, a free special case.
- **Per-reflection** — `AnalyticResolution` is built per `(hkl, keV)` via
  `theta`; divergences are reflection-independent. Multi-reflection builds
  several instances, no files.

## 8. Rollout

1. Implement A + B. Validate (§6). `mypy` + `pytest` green.
2. Flip the `auto` default to analytic for no-beamstop (release-worthy change;
   version bump). MC remains for beamstop and `backend="mc"`.
3. Defer: removing the bootstrap `.npz` machinery — keep it for beamstop and as
   a regression reference.

## 9. Open questions

- COM-cleanliness test: assert against a smooth analytic reference, or against
  a high-resolution grid? (Decide during implementation.)
- Version number for the default-flip release (v2.1 vs v3.0 — behavior change
  for existing no-beamstop configs).
- Conditioning threshold for the `σ_eps`-degeneracy error (§7) — pick a concrete
  cutoff during implementation (e.g. on `cond(C_rest)` or `min-eig/max-eig`).
