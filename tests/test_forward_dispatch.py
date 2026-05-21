"""Tests for the scan-grid + dislocation-population dispatch helpers (sub-projects B+C)."""

from __future__ import annotations

import numpy as np

from dfxm_geo.direct_space.forward_model import ScanGrid, build_scan_grid
from dfxm_geo.pipeline import ScanConfig


class TestBuildScanGrid:
    def test_single_mode_all_axes_singleton(self) -> None:
        grid = build_scan_grid(ScanConfig())
        assert isinstance(grid, ScanGrid)
        assert grid.axes == ("phi", "chi", "two_dtheta", "z")
        for samples in grid.samples:
            assert samples.shape == (1,)
            assert samples[0] == 0.0

    def test_rocking_mode_phi_has_steps_samples(self) -> None:
        cfg = ScanConfig.from_dict({"phi": {"range": 1e-3, "steps": 21}})
        grid = build_scan_grid(cfg)
        assert grid.samples[0].shape == (21,)
        # First, middle, last are linspace(-range, +range, steps)
        np.testing.assert_allclose(grid.samples[0][0], -1e-3)
        np.testing.assert_allclose(grid.samples[0][-1], +1e-3)
        np.testing.assert_allclose(grid.samples[0][10], 0.0, atol=1e-12)
        # Other axes are singletons
        assert grid.samples[1].shape == (1,)
        assert grid.samples[2].shape == (1,)
        assert grid.samples[3].shape == (1,)

    def test_scan_centered_on_value_offset(self) -> None:
        cfg = ScanConfig.from_dict({"phi": {"value": 1.5e-4, "range": 1e-3, "steps": 11}})
        grid = build_scan_grid(cfg)
        np.testing.assert_allclose(grid.samples[0][0], 1.5e-4 - 1e-3)
        np.testing.assert_allclose(grid.samples[0][-1], 1.5e-4 + 1e-3)
        np.testing.assert_allclose(grid.samples[0][5], 1.5e-4, atol=1e-12)

    def test_fixed_axis_uses_value_singleton(self) -> None:
        cfg = ScanConfig.from_dict({"chi": {"value": 2e-5}})
        grid = build_scan_grid(cfg)
        # chi is index 1 in canonical order
        assert grid.samples[1].shape == (1,)
        np.testing.assert_allclose(grid.samples[1][0], 2e-5)

    def test_mosa_strain_mode(self) -> None:
        cfg = ScanConfig.from_dict(
            {
                "phi": {"range": 6e-4, "steps": 21},
                "chi": {"range": 2e-3, "steps": 21},
                "two_dtheta": {"range": 5e-4, "steps": 11},
            }
        )
        grid = build_scan_grid(cfg)
        assert grid.samples[0].shape == (21,)
        assert grid.samples[1].shape == (21,)
        assert grid.samples[2].shape == (11,)
        assert grid.samples[3].shape == (1,)
