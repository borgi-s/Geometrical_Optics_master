"""Unit tests for dfxm_geo.analysis.moments and dfxm_geo.analysis.colormaps."""

import numpy as np

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
        # NaNs may appear outside the convex hull of the key points; ignore those.
        finite = colors[np.isfinite(colors)]
        assert (finite >= 0).all()
        assert (finite <= 1).all()
