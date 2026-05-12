"""Smoke test: pin Fd_find output to a baseline.

This guards against silent numerical regressions during refactors. Inputs
are deliberately small and deterministic; we are not validating physics here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add repo root to import path so the flat-layout modules resolve.
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from functions import Fd_find  # noqa: E402


def _build_inputs(n: int = 8):
    """Small, deterministic inputs for the smoke test."""
    lin = np.linspace(-1.0, 1.0, n)
    # Create 3D grid and flatten to shape (3, N) where N = n^3
    grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"))  # shape (3, n, n, n)
    rl = grid.reshape(3, -1)  # flatten to (3, n^3)
    Ud = np.eye(3)
    Us = np.eye(3)
    Theta = np.eye(3)
    return rl, Ud, Us, Theta


def test_Fd_find_matches_golden(golden_dir: Path):
    rl, Ud, Us, Theta = _build_inputs(n=8)
    out = Fd_find(rl, Ud, Us, Theta, dis=1, ndis=1)
    out = np.asarray(out)

    golden_path = golden_dir / "Fd_find_smoke.npy"
    if not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(golden_path, out)
        pytest.skip("Golden file created; rerun to enable comparison.")

    expected = np.load(golden_path)
    assert out.shape == expected.shape
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)
