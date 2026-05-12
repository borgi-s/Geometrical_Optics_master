"""Unit tests for dfxm_geo.crystal.dislocations."""

import numpy as np
import pytest

from dfxm_geo.constants import BURGERS_VECTOR, POISSON_RATIO
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


class TestFdFindBipolarWall:
    """Regression tests guarding the sequential/parallel-branch parity.

    Pre-2026-05-12, `multi_dislocs_parallel` produced a monotone one-sided
    wall (offsets `-1, -2, -3, …`) while the sequential branch produced a
    bipolar centered wall (offsets `+1, -1, +2, -2, …`). The two branches
    disagreed by ~22% in Fdd Frobenius norm for the same `ndis`. These
    tests pin the post-fix behavior so the bug can't silently come back.
    """

    @staticmethod
    def _bipolar_reference_Fdd(
        rd: np.ndarray,
        dis: float,
        ndis: int,
        b: float = BURGERS_VECTOR,
        ny: float = POISSON_RATIO,
    ) -> np.ndarray:
        """Independent reference implementation of the bipolar wall sum.

        Mirrors Fd_find's per-point arithmetic exactly. Caller must pass
        identity Ud/Us/Theta into Fd_find so this reference (which skips
        coordinate transforms) is directly comparable. The sign convention
        for the per-i offset matches Fd_find:
            odd i  (1, 3, 5, …)  → rd_n[1] += ceil(i/2) * dis
            even i (2, 4, 6, …)  → rd_n[1] -= (i/2) * dis
        which is the deterministic, order-independent form of Fd_find's
        sequential `count1/count2` pattern.
        """
        Fdd = np.zeros((rd.shape[1], 3, 3))

        # Central wall (i=0): regularized denom with alpha=1e-20.
        sqx = rd[0] ** 2
        sqy = rd[1] ** 2
        denom = (sqx + sqy) ** 2 + 1e-20
        nyfactor = 2 * ny * (sqx + sqy)
        Fdd[:, 0, 0] = -rd[1] * (3 * sqx + sqy - nyfactor) / denom
        Fdd[:, 0, 1] = rd[0] * (3 * sqx + sqy - nyfactor) / denom
        Fdd[:, 1, 0] = -rd[0] * (3 * sqy + sqx - nyfactor) / denom
        Fdd[:, 1, 1] = rd[1] * (sqx - sqy + nyfactor) / denom

        # Bipolar wall contributions (no alpha regularization in the loop).
        for i in range(1, ndis):
            rd_n = np.copy(rd[:2])
            k = (i + 1) // 2
            if i % 2 == 1:
                rd_n[1] += k * dis
            else:
                rd_n[1] -= k * dis
            sqx = rd_n[0] ** 2
            sqy = rd_n[1] ** 2
            denom = (sqx + sqy) ** 2
            nyfactor = 2 * ny * (sqx + sqy)
            Fdd[:, 0, 0] += -rd_n[1] * (3 * sqx + sqy - nyfactor) / denom
            Fdd[:, 0, 1] += rd_n[0] * (3 * sqx + sqy - nyfactor) / denom
            Fdd[:, 1, 0] += -rd_n[0] * (3 * sqy + sqx - nyfactor) / denom
            Fdd[:, 1, 1] += rd_n[1] * (sqx - sqy + nyfactor) / denom

        Fdd *= b / (4 * np.pi * (1 - ny))
        Fdd += np.identity(3)
        return Fdd

    @pytest.mark.parametrize("ndis", [50, 100, 101, 200])
    def test_matches_bipolar_reference(self, ndis: int) -> None:
        """Fd_find with identity transforms agrees with the bipolar reference.

        Parametrized to cover both the sequential branch (ndis <= 100) and
        the parallel branch (ndis > 100) and the boundary on either side.
        """
        # Identity coordinate transforms so rd == rl in Fd_find and the
        # final `Ud @ Fdd @ Ud.T` is a no-op.
        rl = _make_grid(n=6, extent=3.0)
        out = Fd_find(rl, np.eye(3), np.eye(3), np.eye(3), dis=1.0, ndis=ndis)
        expected = self._bipolar_reference_Fdd(rl, dis=1.0, ndis=ndis)
        np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-12)

    def test_boundary_continuity_at_ndis_100_to_101(self) -> None:
        """ndis=101 (parallel path) ≈ ndis=100 (sequential) + one extra wall.

        Across the ndis>100 boundary, Fd_find switches code paths. With the
        fix in place, the only physical difference between ndis=100 and
        ndis=101 should be the single additional bipolar wall (i=100 →
        offset -50*dis). Diff norm should be much smaller than the full
        signal — pre-fix it was ~22% of ||Fd_find(100)||.
        """
        rl = _make_grid(n=6, extent=3.0)
        out_100 = Fd_find(rl, np.eye(3), np.eye(3), np.eye(3), dis=1.0, ndis=100)
        out_101 = Fd_find(rl, np.eye(3), np.eye(3), np.eye(3), dis=1.0, ndis=101)
        diff_norm = np.linalg.norm(out_101 - out_100)
        signal_norm = np.linalg.norm(out_100)
        # A single dislocation at offset -50*dis is a tiny perturbation at
        # this scale. Pre-fix the diff was ~0.22 * signal; post-fix it's
        # ~1e-3 * signal or less. The 0.05 ceiling gives margin for grids
        # closer to the wall while still trapping the broken branch.
        assert diff_norm / signal_norm < 0.05, (
            f"Discontinuity at ndis>100 boundary: ||delta|| / ||signal|| = "
            f"{diff_norm / signal_norm:.4f}. Did multi_dislocs_parallel "
            f"regress to the monotone wall pattern?"
        )
