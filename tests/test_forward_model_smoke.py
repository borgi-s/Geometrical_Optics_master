"""Structural smoke tests for direct_space.forward_model.

These guard the lazy-pickle-load refactor: the module must import on a
clean clone (no reciprocal-space kernel pickle present), and `forward()`
must raise a clear error when called before kernel state is loaded.

A future test (Phase 7) will add numerical pinning by generating a tiny
fixture kernel and parameterizing the module geometry to small grids.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import direct_space.forward_model as fm  # noqa: E402


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
