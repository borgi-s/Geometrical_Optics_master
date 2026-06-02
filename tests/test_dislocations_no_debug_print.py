"""Regression guard: Fd_find's parallel branch must not spam stdout.

The `ndis > 100` parallel path in ``Fd_find`` previously contained a bare
``print(chunks)`` debug residue that fired on every large wall — spamming
batch / ML runs that generate thousands of forward images. This test pins
the silent behavior so the debug print cannot silently return.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.crystal.dislocations import Fd_find


def _make_grid(n: int, extent: float = 1.0) -> np.ndarray:
    """Build a flattened (3, n^3) coordinate grid spanning ±extent."""
    lin = np.linspace(-extent, extent, n)
    grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"))
    return grid.reshape(3, -1)


def test_fd_find_parallel_branch_is_silent(capsys: pytest.CaptureFixture[str]) -> None:
    """`ndis > 100` (parallel path) must produce no stdout output."""
    rl = _make_grid(n=4, extent=3.0)  # tiny grid; we only care about the print
    Fd_find(rl, np.eye(3), np.eye(3), np.eye(3), dis=1.0, ndis=101)
    captured = capsys.readouterr()
    assert captured.out == "", f"Fd_find parallel branch emitted stdout: {captured.out!r}"
