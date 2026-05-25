# tests/reciprocal_space/test_analytic_resolution.py
import numpy as np
import pytest

from dfxm_geo.reciprocal_space.analytic_resolution import (
    AnalyticResolution,
    _build_M,
    quadrature_pq,
)

THETA = 0.156611  # Al 111 @ 17 keV


def test_build_M_matches_transform():
    M = _build_M(THETA)
    assert M.shape == (3, 5)
    # Columns = [eps, zeta_v, zeta_h, delta_2theta, xi]; values verified in
    # C:\Users\borgi\tmp\math_check\verify.py.
    expected = np.array(
        [
            [0.155972, -0.987762, 0.0, 0.0, 0.0],
            [0.0, 0.0, -3.205712, 0.0, -3.205712],
            [0.987762, -3.049741, 0.0, 3.205712, 0.0],
        ]
    )
    np.testing.assert_allclose(M, expected, atol=1e-5)


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
    q[0, 1] = 2e-4  # along qrock_prime
    q[1, 2] = 4e-3  # along qroll
    q[2, 3] = 4e-3  # along q2th
    p = res(q)
    assert p.shape == (4,)
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0) & (p <= 1.0 + 1e-12))
    assert p[0] == pytest.approx(1.0)
    assert p[1] < 1.0 and p[2] < 1.0 and p[3] < 1.0


def test_closed_form_matches_quadrature():
    kw = _nominal_kwargs()
    res = AnalyticResolution(**kw)
    rng = np.random.default_rng(0)
    # q points spanning a few sigma in each imaging axis.
    q = (rng.standard_normal((3, 200)).T * np.array([2e-4, 1.2e-3, 1.2e-3])).T
    closed = res._raw_pq(q)  # unnormalized, to compare to the integral
    quad = quadrature_pq(q, **kw)
    # Compare where the density is meaningfully nonzero. The wide q-draw above
    # scatters ~1/3 of the points many sigma into the tail, where the closed
    # form's exp() factor underflows to exactly 0.0 while the Gauss-Legendre
    # sum returns a denormal-range positive (~1e-30); a relative test there is
    # meaningless. On the bulk the two agree to ~1e-13 (verified against an
    # independent scipy.integrate.quad oracle in C:\Users\borgi\tmp\math_check).
    mask = quad > quad.max() * 1e-9
    np.testing.assert_allclose(closed[mask], quad[mask], rtol=1e-8, atol=1e-12)


def test_degenerate_eps_raises():
    kw = _nominal_kwargs()
    kw["eps_rms"] = 1e-12  # collapses qrock' Gaussian spread
    with pytest.raises(ValueError, match="degenerate"):
        AnalyticResolution(**kw)


def test_zero_vertical_divergence_is_pure_gaussian():
    kw = _nominal_kwargs()
    kw["zeta_v_fwhm"] = 0.0  # sigma_zv = 0 -> no truncation, pure 3D Gaussian
    res = AnalyticResolution(**kw)
    assert res._gaussian_only is True
    p0 = res(np.zeros((3, 1)))
    np.testing.assert_allclose(p0, [1.0], atol=1e-12)
    # Pure-Gaussian density integrates without erf; just confirm it is finite
    # and peaked at 0.
    q = np.zeros((3, 2))
    q[2, 1] = 3e-3
    p = res(q)
    assert p[0] == pytest.approx(1.0) and 0.0 < p[1] < 1.0
