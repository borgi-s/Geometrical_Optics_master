"""Tests for _solve_quadratic_in_tan_half (paper eq A.7 -> A.8)."""

import numpy as np

from dfxm_geo.crystal.oblique import _solve_quadratic_in_tan_half


def test_two_real_roots_known():
    """(α₂-α₀)s² + 2α₁s + (α₀+α₂) = 0 with two real roots: s=1, s=-1.
    Pick α₀=1, α₁=0, α₂=0: equation becomes -s²+1=0; roots s=±1; ω = ±π/2."""
    s1, s2 = _solve_quadratic_in_tan_half(α0=1.0, α1=0.0, α2=0.0)
    assert {round(s1, 12), round(s2, 12)} == {1.0, -1.0}


def test_special_case_alpha2_minus_alpha0_zero():
    """When α₂ - α₀ == 0, quadratic degenerates to linear: 2α₁ s + (α₀+α₂) = 0."""
    s1, s2 = _solve_quadratic_in_tan_half(α0=1.0, α1=1.0, α2=1.0)
    # 2·1·s + 2 = 0 → s = -1
    assert s1 == -1.0
    assert np.isnan(s2)


def test_no_real_solution_returns_nan():
    """Discriminant < 0 → both NaN."""
    s1, s2 = _solve_quadratic_in_tan_half(α0=1.0, α1=0.0, α2=2.0)
    # (α₂-α₀)=1, α₁=0, (α₀+α₂)=3 → 1·s²+0+3=0 → s² = -3, no real
    assert np.isnan(s1) and np.isnan(s2)
