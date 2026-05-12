"""Unit tests for dfxm_geo.crystal.dislocations."""

import numpy as np
import pytest

from dfxm_geo.crystal.dislocations import Fd_find


def _make_grid(n: int, extent: float = 1.0) -> np.ndarray:
    """Build a flattened (3, n^3) coordinate grid spanning ±extent."""
    lin = np.linspace(-extent, extent, n)
    grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"))
    return grid.reshape(3, -1)


class TestFdFindShape:
    def test_returns_3x3_per_point(self):
        """Output shape is (N, 3, 3) — one tensor per coordinate."""
        rl = _make_grid(n=6)
        out = Fd_find(rl, np.eye(3), np.eye(3), np.eye(3), dis=1, ndis=1)
        assert out.shape == (rl.shape[1], 3, 3)

    def test_dtype_is_float(self):
        """Output is a floating-point array (not int)."""
        rl = _make_grid(n=4)
        out = Fd_find(rl, np.eye(3), np.eye(3), np.eye(3), dis=1, ndis=1)
        assert np.issubdtype(out.dtype, np.floating)

    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_scales_with_grid_size(self, n):
        """N output tensors for an N-point grid."""
        rl = _make_grid(n=n)
        out = Fd_find(rl, np.eye(3), np.eye(3), np.eye(3), dis=1, ndis=1)
        assert out.shape[0] == n**3


class TestFdFindFarField:
    def test_far_field_approaches_identity(self):
        """Far from the dislocation core, Fd ≈ I (the strain decays as 1/r²)."""
        rl = _make_grid(n=8, extent=200.0)  # µm, well outside core radius
        out = Fd_find(rl, np.eye(3), np.eye(3), np.eye(3), dis=1, ndis=1)
        # Take the eight corner tensors (maximum distance from origin).
        # Reshape to find corners: (3, 8, 8, 8) -> indices 0 and 7.
        out_grid = out.reshape(8, 8, 8, 3, 3)
        corners = np.stack(
            [
                out_grid[0, 0, 0],
                out_grid[0, 0, 7],
                out_grid[0, 7, 0],
                out_grid[7, 0, 0],
                out_grid[7, 7, 7],
            ]
        )
        for corner in corners:
            np.testing.assert_allclose(corner, np.eye(3), atol=1e-4)

    def test_finite_at_off_axis_points(self):
        """The 1e-20 regularization keeps Fd finite everywhere on a grid."""
        rl = _make_grid(n=8, extent=5.0)  # crosses near-origin region
        out = Fd_find(rl, np.eye(3), np.eye(3), np.eye(3), dis=1, ndis=1)
        assert np.isfinite(out).all()


class TestFdFindMultipleDislocations:
    def test_ndis_one_vs_many_differs(self):
        """Adding more dislocations changes the field.

        Fd_find only writes to the [0,0], [0,1], [1,0], [1,1] tensor components
        (the rest stay at identity), so we check the off-diagonal in-plane
        block specifically rather than the full 9-entry tensor.
        """
        rl = _make_grid(n=8, extent=5.0)
        out1 = Fd_find(rl, np.eye(3), np.eye(3), np.eye(3), dis=1, ndis=1)
        out5 = Fd_find(rl, np.eye(3), np.eye(3), np.eye(3), dis=1, ndis=5)
        # The active in-plane block should differ at most pixels.
        block_diff = ~np.isclose(out1[:, :2, :2], out5[:, :2, :2], atol=1e-12)
        assert block_diff.mean() > 0.5
        # Sanity: the arrays aren't identical at all.
        assert not np.allclose(out1, out5)

    def test_zero_burgers_vector_gives_identity(self):
        """b=0 means no displacement — Fd is just the identity field."""
        rl = _make_grid(n=4, extent=5.0)
        out = Fd_find(rl, np.eye(3), np.eye(3), np.eye(3), dis=1, ndis=1, b=0.0)
        expected = np.broadcast_to(np.eye(3), out.shape)
        np.testing.assert_allclose(out, expected, atol=1e-12)
