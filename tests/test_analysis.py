"""Unit tests for dfxm_geo.analysis.moments and dfxm_geo.analysis.colormaps."""

import numpy as np
import pytest

from dfxm_geo.analysis.colormaps import inv_polefigure_colors
from dfxm_geo.analysis.moments import calc_moments


class TestCalcMoments:
    def test_uniform_image_has_zero_mean(self):
        """A uniform image is symmetric → its centroid in (u, v) is the origin."""
        image = np.ones((32, 32))
        u_range = v_range = 1.0
        u_steps = v_steps = 32
        m = calc_moments(image, u_range, v_range, u_steps, v_steps)
        # On a symmetric grid the mean should be at the origin.
        assert abs(m["mean_u"]) < 1e-12
        assert abs(m["mean_v"]) < 1e-12

    def test_returns_expected_keys(self):
        """All advertised moment names are present in the result dict."""
        image = np.ones((16, 16))
        m = calc_moments(image, 1.0, 1.0, 16, 16)
        expected_keys = {
            "mean_u",
            "mean_v",
            "m00",
            "m01",
            "m10",
            "m02",
            "m20",
            "mu02",
            "mu20",
            "mu03",
            "mu30",
            "nu03",
            "nu30",
        }
        assert expected_keys.issubset(m.keys())

    def test_off_center_gaussian_has_nonzero_mean(self):
        """A Gaussian peak shifted from the origin produces a nonzero centroid."""
        # Build a Gaussian centered at (0.4, 0.0).
        u_range = v_range = 1.0
        n = 64
        u_grid, v_grid = np.mgrid[-u_range : u_range : complex(n), -v_range : v_range : complex(n)]
        sigma = 0.1
        image = np.exp(-((u_grid - 0.4) ** 2 + v_grid**2) / (2 * sigma**2))
        m = calc_moments(image, u_range, v_range, n, n)
        # mean_u corresponds to the u-axis offset (0.4) — confirm sign and rough magnitude.
        assert 0.3 < m["mean_u"] < 0.5
        assert abs(m["mean_v"]) < 0.05

    def test_m00_is_total_intensity(self):
        """m00 (zeroth raw moment) equals the summed intensity — no stray factor."""
        image = np.full((10, 10), 3.0)
        m = calc_moments(image, 1.0, 1.0, 10, 10)
        assert m["m00"] == pytest.approx(np.sum(image))

    def test_zero_intensity_image_raises(self):
        """An all-zero image has no defined moments → ValueError, not silent NaN."""
        with pytest.raises(ValueError, match="positive total intensity"):
            calc_moments(np.zeros((8, 8)), 1.0, 1.0, 8, 8)

    def test_negative_total_intensity_raises(self):
        """A net-negative image is non-physical and would flip moment signs."""
        with pytest.raises(ValueError, match="positive total intensity"):
            calc_moments(np.full((8, 8), -1.0), 1.0, 1.0, 8, 8)


class TestInvPolefigureColors:
    def test_returns_array_shapes(self):
        """The function returns (colors, xydata) with matching point counts."""
        o_grid = (np.linspace(-0.5, 0.5, 8), np.linspace(-0.5, 0.5, 8))
        test_grid = (np.linspace(-1.0, 1.0, 16), np.linspace(-1.0, 1.0, 16))
        colors, xydata = inv_polefigure_colors(o_grid, test_grid)
        assert colors.ndim == 2 and colors.shape[1] == 4  # RGBA
        assert xydata.shape == (colors.shape[0], 2)
        assert colors.shape[0] == o_grid[0].size * o_grid[1].size

    def test_colors_clipped_to_unit_interval(self):
        """All R/G/B/A values are in [0, 1] (the function clamps explicitly)."""
        o_grid = (np.linspace(-0.5, 0.5, 4), np.linspace(-0.5, 0.5, 4))
        test_grid = (np.linspace(-1.0, 1.0, 4), np.linspace(-1.0, 1.0, 4))
        colors, _ = inv_polefigure_colors(o_grid, test_grid)
        assert (colors >= 0).all()
        assert (colors <= 1).all()

    def test_no_nan_outside_anchor_hull(self):
        """Sample points outside the 7-anchor hull fall back to nearest, not NaN.

        With o_grid wider than test_grid, many points lie outside the convex
        hull of the colour anchors — linear griddata returns NaN there, which
        previously leaked past the [0, 1] clip. The nearest-neighbour fallback
        must leave no NaN in the output.
        """
        o_grid = (np.linspace(-2.0, 2.0, 9), np.linspace(-2.0, 2.0, 9))
        test_grid = (np.linspace(-1.0, 1.0, 5), np.linspace(-1.0, 1.0, 5))
        colors, _ = inv_polefigure_colors(o_grid, test_grid)
        assert not np.isnan(colors).any()
