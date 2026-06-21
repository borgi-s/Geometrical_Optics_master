"""Omega q-path seam: R_z(omega) @ Us at the projection step.

This is the q-path half of full-omega (2026-06-21). The matching rl
counter-rotation lives in build_geometry_context (see test_full_omega_geometry.py).
These tests pin the projection-step rotation and that Us itself stays unrotated
on the instrument context (so omega is applied exactly once on the q side)."""

from __future__ import annotations

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.crystal.oblique import _R_z


def _dummy_resolution() -> fm.ResolutionContext:
    # Fields per ResolutionContext dataclass; analytic_eval/Resq_i None is fine —
    # precompute_forward_static never touches resolution.
    return fm.ResolutionContext(
        Resq_i=None,
        qi1_start=0.0,
        qi1_step=1.0,
        qi2_start=0.0,
        qi2_step=1.0,
        qi3_start=0.0,
        qi3_step=1.0,
        npoints1=None,
        npoints2=None,
        npoints3=None,
        analytic_eval=None,
        loaded_kernel_path=None,
    )


@pytest.fixture
def small_hg() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.normal(scale=1e-4, size=(50, 3, 3))


def test_omega_zero_is_bit_identical(small_hg):
    ctx0 = fm.build_forward_context(0.3, _dummy_resolution(), (1, 1, 1))
    ctx0_explicit = fm.build_forward_context(0.3, _dummy_resolution(), (1, 1, 1), omega=0.0)
    a = fm.precompute_forward_static(small_hg, ctx0)
    b = fm.precompute_forward_static(small_hg, ctx0_explicit)
    assert np.array_equal(a, b)  # bit-identical, not approx


def test_omega_default_field_is_zero():
    ctx = fm.build_forward_context(0.3, _dummy_resolution(), (1, 1, 1))
    assert ctx.geometry.omega == 0.0


def test_omega_rotates_projection(small_hg):
    omega = 0.7
    ctx0 = fm.build_forward_context(0.3, _dummy_resolution(), (1, 1, 1))
    ctxw = fm.build_forward_context(0.3, _dummy_resolution(), (1, 1, 1), omega=omega)
    base0 = fm.precompute_forward_static(small_hg, ctx0)  # (3, N)
    basew = fm.precompute_forward_static(small_hg, ctxw)
    # B': base_qc(omega) == R_z(omega) applied to the un-rotated projection
    np.testing.assert_allclose(basew, _R_z(omega) @ base0, rtol=1e-12, atol=1e-18)


def test_instrument_us_stays_unrotated_under_omega():
    """ctx.instrument.Us must stay UN-rotated: the q-path applies R_z(omega) @ Us
    locally in precompute_forward_static, and Find_Hg reads the unrotated Us while
    the rl grid carries the rotation (full-omega). If Us itself were rotated here,
    omega would be double-counted on the q side."""
    ctxw = fm.build_forward_context(0.3, _dummy_resolution(), (1, 1, 1), omega=0.7)
    ctx0 = fm.build_forward_context(0.3, _dummy_resolution(), (1, 1, 1))
    assert np.array_equal(ctxw.instrument.Us, ctx0.instrument.Us)
