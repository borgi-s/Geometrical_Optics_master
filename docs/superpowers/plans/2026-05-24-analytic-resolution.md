# Analytic Reciprocal-Space Resolution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a grid-free, closed-form reciprocal-space resolution backend that replaces the Monte-Carlo `Resq_i` histogram for no-beamstop configs, eliminating the COM/`qi_range` quantization artifact class.

**Architecture:** A new `AnalyticResolution` evaluates the verified closed form `p_Q(q) = N₀·exp(−½[Q₀−B²/A])·½[erf(...)−erf(...)]` (truncated condenser `zeta_v` + 4 Gaussians). `forward()` gains a one-line branch that calls it instead of the LUT. The pipeline dispatches analytic-vs-MC by config (`backend="auto"` → analytic when beamstop off). MC path untouched; analytic is purely additive until the v2.1.0 default-flip.

**Tech Stack:** Python, numpy, scipy (`scipy.linalg.cho_factor`/`cho_solve`, `scipy.special.erf`), pytest, mypy. venv python: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe`.

**Spec:** `docs/superpowers/specs/2026-05-24-analytic-resolution-design.md`
**Verified math reference (cross-check against these):** `C:\Users\borgi\tmp\math_check\` (`verify.py`, `compare.py`, `norm_check.py`).

---

## File structure

| File | Responsibility |
|---|---|
| `src/dfxm_geo/reciprocal_space/analytic_resolution.py` (create) | `_build_M`, `_norm_cdf`, `AnalyticResolution` (closed form), `quadrature_pq` (Approach-B oracle/fallback) |
| `src/dfxm_geo/pipeline.py` (modify) | `ReciprocalConfig` gains `backend` + `beamstop` + instrument params; new `_load_resolution(config)` dispatch |
| `src/dfxm_geo/direct_space/forward_model.py` (modify) | `_analytic_eval` module global, `_load_analytic_resolution(...)`, `forward()` branch at line ~498 |
| `tests/reciprocal_space/test_analytic_resolution.py` (create) | unit tests for the math (M, closed form, quadrature equivalence, edge cases) |
| `tests/test_analytic_backend_integration.py` (create) | dispatch, forward()-vs-MC parity, COM cleanliness |
| `pyproject.toml` + `tests/test_version_is_*.py` (modify) | version bump to 2.1.0 |

**Convention notes:** the venv python is required (`python` is Py2.7). Run tests with `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest ...`. The repo uses pre-commit (ruff/format) on commit. All e2e tests use small grids (5×5 scan, ≤128 px) per the smoke-test rule.

---

## Task 1: `AnalyticResolution` — the verified closed form

**Files:**
- Create: `src/dfxm_geo/reciprocal_space/analytic_resolution.py`
- Test: `tests/reciprocal_space/test_analytic_resolution.py`

- [ ] **Step 1: Write the failing test for `_build_M`**

```python
# tests/reciprocal_space/test_analytic_resolution.py
import numpy as np
import pytest
from dfxm_geo.reciprocal_space.analytic_resolution import (
    _build_M, AnalyticResolution,
)

THETA = 0.156611  # Al 111 @ 17 keV

def test_build_M_matches_transform():
    M = _build_M(THETA)
    assert M.shape == (3, 5)
    # Columns = [eps, zeta_v, zeta_h, delta_2theta, xi]; values verified in
    # C:\Users\borgi\tmp\math_check\verify.py.
    expected = np.array([
        [0.155972, -0.987762, 0.0,       0.0,      0.0],
        [0.0,       0.0,     -3.205712,  0.0,     -3.205712],
        [0.987762, -3.049741, 0.0,       3.205712, 0.0],
    ])
    np.testing.assert_allclose(M, expected, atol=1e-5)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/reciprocal_space/test_analytic_resolution.py::test_build_M_matches_transform -v`
Expected: FAIL — `ModuleNotFoundError: dfxm_geo.reciprocal_space.analytic_resolution`.

- [ ] **Step 3: Implement `_build_M` + `_norm_cdf`**

```python
# src/dfxm_geo/reciprocal_space/analytic_resolution.py
"""Grid-free closed-form reciprocal-space resolution function.

Replaces the Monte-Carlo Resq_i histogram for no-beamstop configs. The 5
instrument variables [eps, zeta_v, zeta_h, delta_2theta, xi] map linearly to
imaging-space q = M @ x. zeta_v is hard-truncated at +-zeta_v_clip (the
condenser physical aperture, Carlsen 2022); the other 4 are Gaussian. The
resolution density p_Q(q) is the marginal over the truncated zeta_v of a
conditional 3D Gaussian, which is closed-form (an erf difference).

Math verified 2026-05-24 (quadrature 2.3e-13, MC <0.1% cov, norm 1 to 7e-15).
See docs/superpowers/specs/2026-05-24-analytic-resolution-design.md sec. 4.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
from scipy.special import erf


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF Phi(x) via erf (avoids a scipy.stats import)."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _build_M(theta: float) -> np.ndarray:
    """3x5 linear map from instrument variables to imaging-space q.

    Columns: [eps, zeta_v, zeta_h, delta_2theta, xi].
    Rows:    [qrock_prime, qroll, q2th].
    Built by applying the resolution.py:250-256 transform (dphi=0, no
    truncation) to the identity, since the map is linear: M[:, j] = transform(e_j).
    """
    s, c = np.sin(theta), np.cos(theta)
    cot = c / s

    def transform(x: np.ndarray) -> np.ndarray:  # x: (5, N) -> (3, N)
        eps, zeta_v, zeta_h, d2t, xi = x
        qrock = -zeta_v / 2 - d2t / 2
        qroll = -zeta_h / (2 * s) - xi / (2 * s)
        qpar = eps + cot * (-zeta_v / 2 + d2t / 2)
        qrock_prime = c * qrock + s * qpar
        q2th = -s * qrock + c * qpar
        return np.array([qrock_prime, qroll, q2th])

    return transform(np.eye(5))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/reciprocal_space/test_analytic_resolution.py::test_build_M_matches_transform -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test for `AnalyticResolution.__call__`**

```python
def _nominal_kwargs():
    return dict(
        theta=THETA,
        zeta_v_fwhm=5.3e-4,
        zeta_h_fwhm=5.3e-4,
        NA_rms=7.31e-4 / 2.35,
        eps_rms=1.41e-4 / 2.35,
        zeta_v_clip=1.4e-4,
    )

def test_call_is_peak_normalized_and_vectorized():
    res = AnalyticResolution(**_nominal_kwargs())
    # Peak is at q=0 (all inputs zero-mean); value there must be exactly 1.
    p0 = res(np.zeros((3, 1)))
    np.testing.assert_allclose(p0, [1.0], atol=1e-12)
    # Vectorized over many rays; finite, in [0, 1], decreasing away from 0.
    q = np.zeros((3, 4))
    q[0, 1] = 2e-4      # along qrock_prime
    q[1, 2] = 4e-3      # along qroll
    q[2, 3] = 4e-3      # along q2th
    p = res(q)
    assert p.shape == (4,)
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0) & (p <= 1.0 + 1e-12))
    assert p[0] == pytest.approx(1.0)
    assert p[1] < 1.0 and p[2] < 1.0 and p[3] < 1.0
```

- [ ] **Step 6: Run it to verify it fails**

Run: `& "...python.exe" -m pytest tests/reciprocal_space/test_analytic_resolution.py::test_call_is_peak_normalized_and_vectorized -v`
Expected: FAIL — `AttributeError`/`TypeError` (class not implemented).

- [ ] **Step 7: Implement `AnalyticResolution`**

```python
class AnalyticResolution:
    """Closed-form p_Q(q), peak-normalized to 1 (matches MC Resq_i/Resq_i.max()).

    Parameters mirror the MC bootstrap (reciprocal_space.kernel.generate_kernel):
    sigma_zv = zeta_v_fwhm/2.355, sigma_zh = zeta_h_fwhm/2.35, sigma_NA = NA_rms,
    sigma_eps = eps_rms. The objective NA square aperture (phys_aper) is dropped
    here (validated <1% vs MC); use the MC backend if it ever matters.

    `cond_max` guards the intrinsic ill-conditioning of C_rest: the qrock'
    direction gets Gaussian spread only from eps, so as eps_rms -> 0 the true
    density collapses onto a 2D plane and no finite p_Q exists. We raise rather
    than ridge-regularize (which would fabricate variance and bias the result).
    """

    def __init__(
        self,
        *,
        theta: float,
        zeta_v_fwhm: float,
        zeta_h_fwhm: float,
        NA_rms: float,
        eps_rms: float,
        zeta_v_clip: float = 1.4e-4,
        cond_max: float = 1e8,
    ) -> None:
        self.theta = float(theta)
        self.c = float(zeta_v_clip)
        sig_eps = float(eps_rms)
        sig_zv = zeta_v_fwhm / 2.355
        sig_zh = zeta_h_fwhm / 2.35
        sig_na = float(NA_rms)
        self.sig_zv = sig_zv

        M = _build_M(theta)
        self.M = M
        self.m_u = M[:, 1]                       # zeta_v column
        M_g = M[:, [0, 2, 3, 4]]                 # eps, zeta_h, d2t, xi
        Sigma_g = np.diag([sig_eps**2, sig_zh**2, sig_na**2, sig_na**2])
        C_rest = M_g @ Sigma_g @ M_g.T

        eigvals = np.linalg.eigvalsh(C_rest)
        cond = eigvals[-1] / eigvals[0] if eigvals[0] > 0 else np.inf
        if not np.isfinite(cond) or cond > cond_max:
            raise ValueError(
                f"Analytic resolution: C_rest is degenerate (cond={cond:.3e} > "
                f"{cond_max:.0e}). The rocking-direction Gaussian spread vanishes "
                "as eps_rms -> 0. Use a finite eps_rms, or backend='mc'."
            )

        self._cho = scipy.linalg.cho_factor(C_rest, lower=True)
        self._logdet_C = 2.0 * float(np.sum(np.log(np.diag(self._cho[0]))))
        self._gaussian_only = sig_zv == 0.0      # no vertical divergence

        if not self._gaussian_only:
            cinv_mu = scipy.linalg.cho_solve(self._cho, self.m_u)
            self.A = float(self.m_u @ cinv_mu) + 1.0 / sig_zv**2
            z_trunc = (
                sig_zv * np.sqrt(2 * np.pi)
                * (_norm_cdf(self.c / sig_zv) - _norm_cdf(-self.c / sig_zv))
            )
            self.N0 = (
                (1.0 / z_trunc)
                * (2 * np.pi) ** (-1.5)
                * np.exp(-0.5 * self._logdet_C)
                * np.sqrt(2 * np.pi / self.A)
            )
        else:
            self.A = np.inf
            self.N0 = (2 * np.pi) ** (-1.5) * np.exp(-0.5 * self._logdet_C)

        self._peak = float(self._raw_pq(np.zeros((3, 1)))[0])

    def _raw_pq(self, qi: np.ndarray) -> np.ndarray:
        """Unnormalized closed-form density. qi: (3, N) -> (N,)."""
        cinv_q = scipy.linalg.cho_solve(self._cho, qi)         # (3, N)
        q0 = np.einsum("in,in->n", qi, cinv_q)                 # q^T C^-1 q
        if self._gaussian_only:
            return self.N0 * np.exp(-0.5 * q0)
        b = self.m_u @ cinv_q                                  # (N,)
        mu = b / self.A
        scale = np.sqrt(self.A / 2.0)
        erf_term = 0.5 * (
            erf((self.c - mu) * scale) - erf((-self.c - mu) * scale)
        )
        return self.N0 * np.exp(-0.5 * (q0 - b**2 / self.A)) * erf_term

    def __call__(self, qi: np.ndarray) -> np.ndarray:
        """Peak-normalized p_Q. qi: (3, N) imaging-space q -> prob (N,)."""
        qi = np.asarray(qi, dtype=float)
        return self._raw_pq(qi) / self._peak
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `& "...python.exe" -m pytest tests/reciprocal_space/test_analytic_resolution.py -v`
Expected: PASS (both tests).

- [ ] **Step 9: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/analytic_resolution.py tests/reciprocal_space/test_analytic_resolution.py
git commit -m "feat: AnalyticResolution closed-form p_Q (Task 1)"
```

---

## Task 2: Quadrature oracle (`quadrature_pq`) + algebra-equivalence test

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/analytic_resolution.py`
- Test: `tests/reciprocal_space/test_analytic_resolution.py`

- [ ] **Step 1: Write the failing equivalence test**

```python
from dfxm_geo.reciprocal_space.analytic_resolution import quadrature_pq

def test_closed_form_matches_quadrature():
    kw = _nominal_kwargs()
    res = AnalyticResolution(**kw)
    rng = np.random.default_rng(0)
    # q points spanning a few sigma in each imaging axis.
    q = (rng.standard_normal((3, 200)).T * np.array([2e-4, 1.2e-3, 1.2e-3])).T
    closed = res._raw_pq(q)             # unnormalized, to compare to the integral
    quad = quadrature_pq(q, **kw)
    np.testing.assert_allclose(closed, quad, rtol=1e-9, atol=1e-12)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `& "...python.exe" -m pytest tests/reciprocal_space/test_analytic_resolution.py::test_closed_form_matches_quadrature -v`
Expected: FAIL — `ImportError: cannot import name 'quadrature_pq'`.

- [ ] **Step 3: Implement `quadrature_pq` (Approach B)**

```python
def quadrature_pq(
    qi: np.ndarray,
    *,
    theta: float,
    zeta_v_fwhm: float,
    zeta_h_fwhm: float,
    NA_rms: float,
    eps_rms: float,
    zeta_v_clip: float = 1.4e-4,
    n_nodes: int = 200,
) -> np.ndarray:
    """Approach B: marginalize the 4 Gaussians analytically (conditional 3D
    Gaussian), integrate the truncated zeta_v over [-c, c] by Gauss-Legendre
    quadrature. Returns the *unnormalized* density (same scale as
    AnalyticResolution._raw_pq). Serves as the algebra oracle and the fallback
    if the NA-aperture approximation ever exceeds tolerance.
    """
    qi = np.asarray(qi, dtype=float)
    sig_zv = zeta_v_fwhm / 2.355
    M = _build_M(theta)
    m_u = M[:, 1]
    M_g = M[:, [0, 2, 3, 4]]
    Sigma_g = np.diag(
        [eps_rms**2, (zeta_h_fwhm / 2.35) ** 2, NA_rms**2, NA_rms**2]
    )
    C_rest = M_g @ Sigma_g @ M_g.T
    cho = scipy.linalg.cho_factor(C_rest, lower=True)
    logdet = 2.0 * float(np.sum(np.log(np.diag(cho[0]))))
    norm3 = (2 * np.pi) ** (-1.5) * np.exp(-0.5 * logdet)

    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    u = zeta_v_clip * nodes               # map [-1,1] -> [-c,c]
    wu = zeta_v_clip * weights
    z_trunc = (
        sig_zv * np.sqrt(2 * np.pi)
        * (_norm_cdf(zeta_v_clip / sig_zv) - _norm_cdf(-zeta_v_clip / sig_zv))
    )
    fu = np.exp(-(u**2) / (2 * sig_zv**2)) / z_trunc

    out = np.zeros(qi.shape[1])
    for ui, wui, fui in zip(u, wu, fu):
        d = qi - m_u[:, None] * ui        # (3, N)
        sol = scipy.linalg.cho_solve(cho, d)
        quad = np.einsum("in,in->n", d, sol)
        out += wui * fui * norm3 * np.exp(-0.5 * quad)
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `& "...python.exe" -m pytest tests/reciprocal_space/test_analytic_resolution.py::test_closed_form_matches_quadrature -v`
Expected: PASS (agreement ~1e-12).

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/analytic_resolution.py tests/reciprocal_space/test_analytic_resolution.py
git commit -m "feat: quadrature_pq Approach-B oracle + equivalence test (Task 2)"
```

---

## Task 3: Edge-case tests (conditioning guard + zeta_v=0 Gaussian branch)

**Files:**
- Test: `tests/reciprocal_space/test_analytic_resolution.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_degenerate_eps_raises():
    kw = _nominal_kwargs()
    kw["eps_rms"] = 1e-12     # collapses qrock' Gaussian spread
    with pytest.raises(ValueError, match="degenerate"):
        AnalyticResolution(**kw)

def test_zero_vertical_divergence_is_pure_gaussian():
    kw = _nominal_kwargs()
    kw["zeta_v_fwhm"] = 0.0    # sigma_zv = 0 -> no truncation, pure 3D Gaussian
    res = AnalyticResolution(**kw)
    assert res._gaussian_only is True
    p0 = res(np.zeros((3, 1)))
    np.testing.assert_allclose(p0, [1.0], atol=1e-12)
    # Pure-Gaussian density integrates without erf; just confirm it is finite
    # and peaked at 0.
    q = np.zeros((3, 2)); q[2, 1] = 3e-3
    p = res(q)
    assert p[0] == pytest.approx(1.0) and 0.0 < p[1] < 1.0
```

- [ ] **Step 2: Run them to verify they fail or pass**

Run: `& "...python.exe" -m pytest tests/reciprocal_space/test_analytic_resolution.py -k "degenerate or pure_gaussian" -v`
Expected: PASS (the guard and the `_gaussian_only` branch were implemented in Task 1; these tests lock that behavior). If either fails, fix the Task-1 implementation to satisfy them.

- [ ] **Step 3: Commit**

```bash
git add tests/reciprocal_space/test_analytic_resolution.py
git commit -m "test: lock AnalyticResolution edge cases (Task 3)"
```

---

## Task 4: Extend `ReciprocalConfig` with backend + instrument params

**Files:**
- Modify: `src/dfxm_geo/pipeline.py:369-407` (`ReciprocalConfig`)
- Test: `tests/test_analytic_backend_integration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analytic_backend_integration.py
import pytest
from dfxm_geo.pipeline import ReciprocalConfig

def test_reciprocal_config_backend_defaults():
    cfg = ReciprocalConfig.from_dict(None)
    assert cfg.backend == "auto"
    assert cfg.beamstop is True            # matches generate_kernel default
    assert cfg.zeta_v_fwhm == pytest.approx(5.3e-4)
    assert cfg.eps_rms == pytest.approx(1.41e-4 / 2.35)

def test_reciprocal_config_parses_backend_and_beamstop():
    cfg = ReciprocalConfig.from_dict(
        {"hkl": [-1, 1, -1], "keV": 17.0, "backend": "analytic",
         "beamstop": False, "zeta_h_fwhm": 5.3e-4}
    )
    assert cfg.backend == "analytic"
    assert cfg.beamstop is False
    assert cfg.zeta_h_fwhm == pytest.approx(5.3e-4)

def test_reciprocal_config_rejects_bad_backend():
    with pytest.raises(ValueError, match="backend"):
        ReciprocalConfig.from_dict({"backend": "nonsense"})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `& "...python.exe" -m pytest tests/test_analytic_backend_integration.py -k reciprocal_config -v`
Expected: FAIL — `AttributeError: 'ReciprocalConfig' object has no attribute 'backend'`.

- [ ] **Step 3: Extend the dataclass**

Replace the `ReciprocalConfig` body (`pipeline.py:369-407`) field block and `from_dict` with:

```python
@dataclass
class ReciprocalConfig:
    """Reflection identity + resolution-backend selection.

    `backend`: "auto" (default; analytic when beamstop off, else MC),
    "analytic" (force closed-form; errors if beamstop on), or "mc".
    The instrument params mirror reciprocal_space.kernel.generate_kernel and
    feed the analytic backend (the MC backend reads them from the kernel .npz).
    """

    hkl: tuple[int, int, int] = (-1, 1, -1)
    keV: float = 17.0
    backend: str = "auto"
    beamstop: bool = True                       # matches generate_kernel default
    zeta_v_fwhm: float = 5.3e-4
    zeta_h_fwhm: float = 0.0
    NA_rms: float = 7.31e-4 / 2.35
    eps_rms: float = 1.41e-4 / 2.35
    zeta_v_clip: float = 1.4e-4

    _VALID_BACKENDS = ("auto", "analytic", "mc")

    def __post_init__(self) -> None:
        if not isinstance(self.hkl, tuple):
            self.hkl = tuple(self.hkl)
        if self.backend not in self._VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {self._VALID_BACKENDS}, got {self.backend!r}."
            )
        from dfxm_geo.reciprocal_space.kernel import _validate_reflection

        _validate_reflection(self.hkl, self.keV, 4.0495e-10)

    @classmethod
    def from_dict(cls, data: dict | None) -> ReciprocalConfig:
        if not data:
            return cls()
        kwargs: dict = {}
        if "hkl" in data:
            kwargs["hkl"] = tuple(data["hkl"])
        if "keV" in data:
            kwargs["keV"] = float(data["keV"])
        for key in ("backend",):
            if key in data:
                kwargs[key] = str(data[key])
        if "beamstop" in data:
            kwargs["beamstop"] = bool(data["beamstop"])
        for key in ("zeta_v_fwhm", "zeta_h_fwhm", "NA_rms", "eps_rms", "zeta_v_clip"):
            if key in data:
                kwargs[key] = float(data[key])
        return cls(**kwargs)
```

Note: the `[reciprocal]` block already permits these keys for `dfxm-bootstrap`; bootstrap's own key-allowlist (`kernel.py:324`) is unchanged. The `_validate_reflection` import path is unchanged from the original.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `& "...python.exe" -m pytest tests/test_analytic_backend_integration.py -k reciprocal_config -v`
Expected: PASS (all three).

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_analytic_backend_integration.py
git commit -m "feat: ReciprocalConfig backend + instrument params (Task 4)"
```

---

## Task 5: `forward()` analytic branch + loader

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` (add globals + `_load_analytic_resolution`; branch at line ~498)
- Test: `tests/test_analytic_backend_integration.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import dfxm_geo.direct_space.forward_model as fm

def test_forward_uses_analytic_when_registered():
    # Load any kernel for geometry/Hg/q_hkl, then register the analytic eval.
    from dfxm_geo.pipeline import _lookup_and_load_kernel, ReciprocalConfig
    cfg = ReciprocalConfig.from_dict(None)
    _lookup_and_load_kernel(cfg.hkl, cfg.keV)          # sets Hg, q_hkl, geometry
    try:
        fm._load_analytic_resolution(cfg)
        assert fm._analytic_eval is not None
        img = fm.forward(fm.Hg, phi=0.0, chi=0.0)
        assert img.shape == (fm.NN2 // fm.Nsub, fm.NN1 // fm.Nsub)
        assert np.all(np.isfinite(img))
        assert img.sum() > 0
    finally:
        fm._analytic_eval = None                       # restore LUT path
```

- [ ] **Step 2: Run it to verify it fails**

Run: `& "...python.exe" -m pytest tests/test_analytic_backend_integration.py::test_forward_uses_analytic_when_registered -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_analytic_eval'`.

- [ ] **Step 3: Add the global + loader**

Near the other module-level kernel state in `forward_model.py` (after the `Resq_i` global declaration block, around line 64), add:

```python
# Analytic resolution backend (Task 5). When set, forward() evaluates this
# closed-form p_Q(qi) instead of the Resq_i lookup. None => MC LUT path.
_analytic_eval = None  # type: ignore[var-annotated]
```

Then add the loader function (next to `_load_default_kernel`):

```python
def _load_analytic_resolution(config: "ReciprocalConfig") -> None:
    """Build the closed-form resolution evaluator and register it for forward().

    Derives theta from (hkl, keV) the same way the MC bootstrap does, then
    builds an AnalyticResolution from the config's instrument params. Also
    computes the default Hg/q_hkl if not already present (parity with
    _load_default_kernel(compute_Hg=True)).
    """
    global _analytic_eval, Hg, q_hkl
    from dfxm_geo.reciprocal_space.analytic_resolution import AnalyticResolution
    from dfxm_geo.reciprocal_space.kernel import _validate_reflection

    theta = _validate_reflection(config.hkl, config.keV, 4.0495e-10)
    _analytic_eval = AnalyticResolution(
        theta=theta,
        zeta_v_fwhm=config.zeta_v_fwhm,
        zeta_h_fwhm=config.zeta_h_fwhm,
        NA_rms=config.NA_rms,
        eps_rms=config.eps_rms,
        zeta_v_clip=config.zeta_v_clip,
    )
    if "Hg" not in globals() or globals().get("Hg") is None:
        Hg, q_hkl = Find_Hg(dis, ndis, psize, zl_rms)
```

- [ ] **Step 4: Add the `forward()` branch**

In `forward()`, replace the index/lookup block (`forward_model.py:480-499`) so the analytic path skips the grid indexing entirely:

```python
    if _analytic_eval is not None:
        # Grid-free: evaluate the closed-form p_Q at every ray's qi. No
        # grid-bounds mask — the closed form returns ~0 in the tails.
        prob = (_analytic_eval(qi) * prob_z).astype(np.float32)
        if not qi_return:
            del qi
        contribution = np.bincount(
            _flat_indices, weights=prob, minlength=im_1.size
        )
        del prob
        im_1 += contribution.reshape(im_1.shape)
        del contribution
        if qi_return:
            qi_field = qi.reshape(3, NN1, NN2, NN3)
            return im_1, qi_field
        return im_1

    # --- existing MC LUT path below (unchanged) ---
    index1 = np.floor((qi[0] - qi1_start) / qi1_step).astype(np.int16)
    # ... rest of the current implementation verbatim ...
```

Note: `prob_z` and `_flat_indices` are existing module globals already used by the LUT path; the analytic path uses the full (unmasked) `_flat_indices`. Keep the existing `RuntimeError` guard at the top of `forward()` but relax it so an analytic-only run (no `Resq_i`) is allowed:

```python
    if Resq_i is None and _analytic_eval is None:
        raise RuntimeError(
            "forward_model state is not initialized. Load a kernel "
            "(_lookup_and_load_kernel) or register the analytic backend "
            "(_load_analytic_resolution) before calling forward()."
        )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `& "...python.exe" -m pytest tests/test_analytic_backend_integration.py::test_forward_uses_analytic_when_registered -v`
Expected: PASS.

- [ ] **Step 6: Run mypy on the touched module**

Run: `& "...python.exe" -m mypy src/dfxm_geo/direct_space/forward_model.py src/dfxm_geo/reciprocal_space/analytic_resolution.py`
Expected: 0 errors. Fix any annotations (e.g. `_analytic_eval: AnalyticResolution | None`).

- [ ] **Step 7: Commit**

```bash
git add src/dfxm_geo/direct_space/forward_model.py tests/test_analytic_backend_integration.py
git commit -m "feat: forward() analytic-resolution branch + loader (Task 5)"
```

---

## Task 6: Pipeline dispatch (`_load_resolution`)

**Files:**
- Modify: `src/dfxm_geo/pipeline.py` (new `_load_resolution`; call it from `run_simulation:627` and the identification path `:1746`)
- Test: `tests/test_analytic_backend_integration.py`

- [ ] **Step 1: Write the failing dispatch tests**

```python
from dfxm_geo.pipeline import _load_resolution, ReciprocalConfig
import dfxm_geo.direct_space.forward_model as fm

def test_dispatch_auto_no_beamstop_selects_analytic():
    cfg = ReciprocalConfig.from_dict({"beamstop": False})   # auto + beamstop off
    fm._analytic_eval = None
    _load_resolution(cfg)
    assert fm._analytic_eval is not None
    fm._analytic_eval = None

def test_dispatch_auto_beamstop_selects_mc():
    cfg = ReciprocalConfig.from_dict({"beamstop": True})     # auto + beamstop on
    fm._analytic_eval = None
    _load_resolution(cfg)
    assert fm._analytic_eval is None        # MC path: kernel loaded, no analytic
    assert fm._loaded_kernel_path is not None

def test_dispatch_explicit_analytic_with_beamstop_errors():
    cfg = ReciprocalConfig.from_dict({"backend": "analytic", "beamstop": True})
    with pytest.raises(ValueError, match="beamstop"):
        _load_resolution(cfg)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `& "...python.exe" -m pytest tests/test_analytic_backend_integration.py -k dispatch -v`
Expected: FAIL — `ImportError: cannot import name '_load_resolution'`.

- [ ] **Step 3: Implement `_load_resolution`**

Add to `pipeline.py` (next to `_lookup_and_load_kernel`):

```python
def _load_resolution(config: ReciprocalConfig) -> None:
    """Select and load the resolution backend per config (spec sec. 5.4).

    auto     -> analytic if beamstop off, else MC
    analytic -> analytic; ValueError if beamstop on (cannot represent it)
    mc       -> MC
    """
    use_analytic = config.backend == "analytic" or (
        config.backend == "auto" and not config.beamstop
    )
    if config.backend == "analytic" and config.beamstop:
        raise ValueError(
            "backend='analytic' is incompatible with beamstop=True (the wire/"
            "knife-edge/aperture stop cannot be represented in closed form). "
            "Use backend='mc', or disable the beamstop."
        )
    if use_analytic:
        fm._analytic_eval = None          # clear any stale evaluator
        fm._load_analytic_resolution(config)
    else:
        fm._analytic_eval = None          # ensure forward() uses the LUT
        _lookup_and_load_kernel(config.hkl, config.keV)
```

Then change the two call sites from `_lookup_and_load_kernel(config.reciprocal.hkl, config.reciprocal.keV)` (`pipeline.py:627` and `:1746`) to:

```python
    _load_resolution(config.reciprocal)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `& "...python.exe" -m pytest tests/test_analytic_backend_integration.py -k dispatch -v`
Expected: PASS (all three).

- [ ] **Step 5: Run the existing pipeline smoke tests (no regression)**

Run: `& "...python.exe" -m pytest tests/ -k "pipeline or run_simulation" -q`
Expected: PASS (the default config has `beamstop=True` → still MC, unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/pipeline.py tests/test_analytic_backend_integration.py
git commit -m "feat: pipeline _load_resolution backend dispatch (Task 6)"
```

---

## Task 7: Validation — analytic vs MC forward image (no-beamstop)

**Files:**
- Test: `tests/test_analytic_backend_integration.py`

This is the §6.4 fidelity gate: a no-beamstop MC kernel (which *does* include the ±3.3σ NA aperture) vs the analytic image. Agreement within shot noise implicitly validates the dropped aperture (§6.3).

- [ ] **Step 1: Write the parity test (built around a small in-test MC kernel)**

```python
import numpy as np
import pytest
import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.reciprocal_space.kernel import _validate_reflection
from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func

@pytest.mark.slow
def test_analytic_forward_matches_mc_no_beamstop(tmp_path):
    hkl, keV = (-1, 1, -1), 17.0
    theta = _validate_reflection(hkl, keV, 4.0495e-10)
    # Build a no-beamstop MC kernel (aperture rejection ON, beamstop OFF) with
    # the SAME instrument params the analytic backend uses. Small but high-N to
    # keep shot noise low. Written into tmp_path/pkl_files.
    out = tmp_path / "pkl_files" / f"Resq_i_h-1_k1_l-1_17keV_test.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    reciprocal_res_func(
        Nrays=int(2e7), npoints1=400, npoints2=200, npoints3=200,
        qi1_range=5e-4, qi2_range=0.75e-2, qi3_range=0.75e-2,
        plot_figs=0, save_resqi=1,
        zeta_v_fwhm=5.3e-4, zeta_h_fwhm=5.3e-4,
        NA_rms=7.31e-4 / 2.35, eps_rms=1.41e-4 / 2.35,
        theta=theta, phys_aper=float(2 * np.sqrt(50e-6 * 1.6e-3)) / 0.274,
        date="test", beamstop=False, output_path=out,
        kernel_meta={"qi1_range": 5e-4, "qi2_range": 0.75e-2, "qi3_range": 0.75e-2,
                     "npoints1": 400, "npoints2": 200, "npoints3": 200,
                     "hkl": hkl, "keV": keV},
    )

    # MC image
    fm._analytic_eval = None
    fm._load_default_kernel(str(out), expected_hkl=hkl, expected_keV=keV)
    img_mc = fm.forward(fm.Hg, phi=0.0, chi=0.0)

    # Analytic image (same geometry / Hg already loaded)
    from dfxm_geo.pipeline import ReciprocalConfig
    cfg = ReciprocalConfig.from_dict({"beamstop": False, "zeta_h_fwhm": 5.3e-4})
    fm._load_analytic_resolution(cfg)
    img_an = fm.forward(fm.Hg, phi=0.0, chi=0.0)
    fm._analytic_eval = None

    # Compare normalized images: correlation high, mean abs diff within the MC
    # shot-noise envelope. Both peak-normalized first.
    a = img_an / img_an.max()
    m = img_mc / img_mc.max()
    corr = np.corrcoef(a.ravel(), m.ravel())[0, 1]
    assert corr > 0.99, f"analytic/MC correlation {corr:.4f} too low"
    assert np.mean(np.abs(a - m)) < 0.03, "analytic/MC mean abs diff too large"
```

- [ ] **Step 2: Run it**

Run: `& "...python.exe" -m pytest tests/test_analytic_backend_integration.py::test_analytic_forward_matches_mc_no_beamstop -v`
Expected: PASS. If correlation is high but the mean-abs-diff bound is slightly exceeded, raise `Nrays` (lower shot noise) before loosening the bound — do NOT loosen silently. If it fails structurally (corr < 0.9), STOP and use systematic-debugging; the NA-aperture drop may matter and Approach B (`quadrature_pq`) is the fallback.

- [ ] **Step 3: Commit**

```bash
git add tests/test_analytic_backend_integration.py
git commit -m "test: analytic-vs-MC forward parity, no beamstop (Task 7)"
```

---

## Task 8: Validation — COM cleanliness demo

**Files:**
- Test: `tests/test_analytic_backend_integration.py`
- Create (figure output): `docs/img/analytic_vs_mc_com.png` (committed artifact)

The §6.5 headline: the analytic COM/rocking map is free of the grid banding. We assert the analytic COM has fewer discrete levels than the MC/grid COM over a single-dislocation rocking scan.

- [ ] **Step 1: Write the COM-cleanliness test**

```python
@pytest.mark.slow
def test_analytic_com_has_no_grid_banding(tmp_path):
    """A phi rocking scan COM map from the analytic backend should be smooth
    (many distinct float levels), not quantized onto grid steps like the MC
    LUT. We compare the count of unique COM values: grid quantization collapses
    them onto a small discrete set.
    """
    import numpy as np
    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo.pipeline import _lookup_and_load_kernel, ReciprocalConfig

    cfg = ReciprocalConfig.from_dict({"beamstop": False, "zeta_h_fwhm": 5.3e-4})
    # Geometry + Hg from any kernel lookup, then analytic eval.
    _lookup_and_load_kernel(cfg.hkl, cfg.keV)
    Hg = fm.Hg

    phis = np.linspace(-3e-4, 3e-4, 21)   # small rocking scan (radians)

    def com_map(eval_setup):
        eval_setup()
        stack = np.stack([fm.forward(Hg, phi=p, chi=0.0) for p in phis], axis=0)
        weight = stack.sum(axis=0) + 1e-30
        com = (stack * phis[:, None, None]).sum(axis=0) / weight
        return com

    com_an = com_map(lambda: fm._load_analytic_resolution(cfg))
    fm._analytic_eval = None
    # MC reference (uses the on-disk default kernel; quantized).
    com_mc = com_map(lambda: _lookup_and_load_kernel(cfg.hkl, cfg.keV))

    # The analytic COM should resolve far more distinct levels than the grid one.
    uniq_an = np.unique(np.round(com_an[np.isfinite(com_an)], 9)).size
    uniq_mc = np.unique(np.round(com_mc[np.isfinite(com_mc)], 9)).size
    assert uniq_an > 5 * uniq_mc, (
        f"analytic COM not smoother than MC: {uniq_an} vs {uniq_mc} levels"
    )

    # Save a before/after figure (artifact for the spec/PR).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(com_mc); ax[0].set_title(f"MC COM ({uniq_mc} levels)")
    ax[1].imshow(com_an); ax[1].set_title(f"analytic COM ({uniq_an} levels)")
    fig.tight_layout()
    fig.savefig("docs/img/analytic_vs_mc_com.png", dpi=110)
```

- [ ] **Step 2: Run it**

Run: `& "...python.exe" -m pytest tests/test_analytic_backend_integration.py::test_analytic_com_has_no_grid_banding -v`
Expected: PASS, and `docs/img/analytic_vs_mc_com.png` written. If the unique-level heuristic is brittle (e.g. the MC kernel happens to be fine-grained), replace the assertion with a comparison against a smooth high-resolution reference — decide here, but keep the saved figure.

- [ ] **Step 3: Commit (including the figure)**

```bash
git add tests/test_analytic_backend_integration.py docs/img/analytic_vs_mc_com.png
git commit -m "test: analytic COM is banding-free vs MC grid (Task 8)"
```

---

## Task 9: Version bump to 2.1.0 + default-flip + housekeeping

**Files:**
- Modify: `pyproject.toml` (version), rename `tests/test_version_is_*.py`
- Modify: `configs/` example (add a commented `backend`/`beamstop` note) and release notes / `docs/output-format.md` or `CHANGELOG` if present
- Modify: `CLAUDE.md` working notes (state-of-cleanup)

- [ ] **Step 1: Find the current version test + pyproject version**

Run: `& "...python.exe" -m pytest tests/ -k version -v` and `Select-String -Path pyproject.toml -Pattern '^version'`
Expected: shows the current `test_version_is_2_0_X.py` and `version = "2.0.x"`.

- [ ] **Step 2: Bump version**

Edit `pyproject.toml`: set `version = "2.1.0"`. Rename the version test file to `tests/test_version_is_2_1_0.py` and update its asserted string to `"2.1.0"`.

- [ ] **Step 3: Run the version test**

Run: `& "...python.exe" -m pytest tests/test_version_is_2_1_0.py -v`
Expected: PASS.

- [ ] **Step 4: Document the default-flip**

In the release notes / changelog (follow the existing v2.0.0 notes location), add a "v2.1.0 — grid-free analytic resolution (breaking for no-beamstop)" entry: no-beamstop runs now default to the closed-form backend; set `[reciprocal] backend = "mc"` to restore the old grid behavior. Note beamstop runs are unaffected.

- [ ] **Step 5: Full suite + mypy gate**

Run: `& "...python.exe" -m pytest -q` and `& "...python.exe" -m mypy src/dfxm_geo/`
Expected: all pass; mypy 0 errors. Fix anything red before proceeding.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/test_version_is_2_1_0.py docs/ CLAUDE.md
git rm tests/test_version_is_2_0_*.py   # the old version test
git commit -m "release: v2.1.0 — analytic resolution default for no-beamstop"
```

---

## Self-review notes (author)

- **Spec coverage:** §4 closed form → Task 1; §3/Approach B oracle → Task 2; §7 edge cases → Tasks 1+3; §5.1/5.2 component+loader → Tasks 1+5; §5.3 forward seam → Task 5; §5.4 dispatch table → Task 6; §6.1 quadrature equivalence → Task 2; §6.2/6.4 MC parity → Task 7; §6.5 COM demo → Task 8; §6.6 mypy/pytest → Tasks 5+9; §8 default-flip+v2.1.0 → Tasks 6+9. All covered.
- **NA-aperture tolerance (§6.3):** validated implicitly in Task 7 (the MC reference includes the aperture; the analytic does not — agreement within shot noise proves the drop). If Task 7 fails structurally, `quadrature_pq` (Task 2) is the documented fallback.
- **Type/name consistency:** `_analytic_eval`, `AnalyticResolution`, `quadrature_pq`, `_load_analytic_resolution`, `_load_resolution`, `ReciprocalConfig.backend/beamstop` used consistently across tasks.
- **Open items deferred to implementation (spec §9):** COM-cleanliness assertion form (Task 8 Step 2 note), `cond_max` threshold (default 1e8 in Task 1), version locked at 2.1.0.
