"""Structural smoke tests for direct_space.forward_model.

These guard the lazy-pickle-load refactor: the module must import on a
clean clone (no reciprocal-space kernel pickle present), and `forward()`
must raise a clear error when called before kernel state is loaded.

A future test (Phase 7) will add numerical pinning by generating a tiny
fixture kernel and parameterizing the module geometry to small grids.
"""

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm


def test_module_imports_with_required_symbols():
    """Module-level public surface remains stable across refactors."""
    assert hasattr(fm, "forward")
    assert hasattr(fm, "Find_Hg")
    assert hasattr(fm, "_load_default_kernel")
    # Numerical constants that init_forward.py and others depend on
    assert hasattr(fm, "psize")
    assert hasattr(fm, "zl_rms")
    assert hasattr(fm, "theta_0")
    assert hasattr(fm, "Npixels")
    assert hasattr(fm, "Ud")
    assert hasattr(fm, "Us")
    assert hasattr(fm, "Theta")
    assert hasattr(fm, "rl")
    assert hasattr(fm, "prob_z")
    assert hasattr(fm, "indices")


def test_forward_raises_when_kernel_not_loaded():
    """forward() must signal clearly when state hasn't been bootstrapped.

    Regression guard for the lazy-pickle-load refactor (commit 610c445):
    if a future change re-introduces eager pickle loading at import time,
    this test won't break — but if someone removes the kernel-state guard
    from forward(), this test catches it.
    """
    saved = fm.Resq_i
    fm.Resq_i = None
    try:
        with pytest.raises(RuntimeError, match="not initialized"):
            fm.forward(Hg=None)
    finally:
        fm.Resq_i = saved


def test_Z_shift_zero_offset_matches_module_rl():
    """Z_shift(0.0) reproduces the module-level rl grid bit-for-bit."""
    import dfxm_geo.direct_space.forward_model as fm

    rl_shifted = fm.Z_shift(0.0)
    np.testing.assert_array_equal(rl_shifted, fm.rl)


def test_Z_shift_shifts_z_column_only():
    """Z_shift(5.0) shifts z by 5 µm; x and y unchanged."""
    import dfxm_geo.direct_space.forward_model as fm

    offset_um = 5.0
    rl_shifted = fm.Z_shift(offset_um)
    np.testing.assert_array_equal(rl_shifted[0], fm.rl[0])
    np.testing.assert_array_equal(rl_shifted[1], fm.rl[1])
    # The z column is shifted by -offset_um * 1e-6 m (Z_shift moves the
    # dislocation core *up* in lab z, equivalent to translating rl *down*).
    np.testing.assert_allclose(rl_shifted[2], fm.rl[2] - offset_um * 1e-6, atol=1e-18, rtol=1e-15)
