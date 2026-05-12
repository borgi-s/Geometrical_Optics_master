"""Unit tests for dfxm_geo.analysis.moments and dfxm_geo.analysis.colormaps."""

import numpy as np

from dfxm_geo.analysis.colormaps import inv_polefigure_colors
from dfxm_geo.analysis.moments import calc_moments, fastgrainplot


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


class TestFastgrainplot:
    """Tests for the per-pixel moment/FWHM map computation.

    Note: `fastgrainplot` mutates its input stack in place
    (`img[img < 0] = 0`). Tests build a fresh stack each time.
    """

    def _square_uniform_stack(self, n: int = 8, h: int = 4, w: int = 4) -> np.ndarray:
        """Stack of (n*n) identical positive images on a square (n, n) grid."""
        return np.ones((n * n, h, w), dtype=float)

    def test_uniform_intensity_zero_mean_for_symmetric_motors(self) -> None:
        """For symmetric motor ranges, identical images → per-pixel mean = 0."""
        n = 8
        stack = self._square_uniform_stack(n=n, h=4, w=4)
        # vlist/ulist symmetric about 0 so the centroid lands on 0.
        vlist = np.linspace(-1, 1, n)
        ulist = np.linspace(-1, 1, n)
        unorm, vnorm, ufwhm, vfwhm = fastgrainplot(stack, vlist, ulist)
        assert unorm.shape == (4, 4) == vnorm.shape == ufwhm.shape == vfwhm.shape
        assert np.allclose(unorm, 0, atol=1e-12)
        assert np.allclose(vnorm, 0, atol=1e-12)

    def test_uniform_intensity_shifted_motor_gives_nonzero_mean(self) -> None:
        """An offset motor range shifts the per-pixel centroid."""
        n = 8
        stack = self._square_uniform_stack(n=n, h=2, w=2)
        # Both motors centered on +0.5, so per-pixel mean should be ~+0.25
        # (the /2 in the function is a legacy weighting; we just check sign + sanity).
        vlist = np.linspace(0, 1, n)
        ulist = np.linspace(0, 1, n)
        unorm, vnorm, _, _ = fastgrainplot(stack, vlist, ulist)
        assert (unorm > 0).all()
        assert (vnorm > 0).all()

    def test_negative_intensities_are_clipped(self) -> None:
        """`fastgrainplot` zero-clips negative pixel values in place."""
        n = 4
        stack = np.full((n * n, 2, 2), -1.0)  # all negative
        # After clipping, inttot=0 everywhere → moments produce NaN/inf; we just
        # care that the function doesn't crash and that the input was modified.
        vlist = np.linspace(-1, 1, n)
        ulist = np.linspace(-1, 1, n)
        # Suppress the expected divide-by-zero warning from the all-zero inttot.
        with np.errstate(divide="ignore", invalid="ignore"):
            fastgrainplot(stack, vlist, ulist)
        assert (stack >= 0).all(), "fastgrainplot should clip negatives to 0 in place"

    def test_returns_four_maps_of_matching_shape(self) -> None:
        """Each returned map has the per-pixel shape (H, W)."""
        n = 4
        stack = self._square_uniform_stack(n=n, h=3, w=5)
        vlist = np.linspace(-1, 1, n)
        ulist = np.linspace(-1, 1, n)
        results = fastgrainplot(stack, vlist, ulist)
        assert len(results) == 4
        for arr in results:
            assert arr.shape == (3, 5)


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
