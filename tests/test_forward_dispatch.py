"""Tests for the scan-grid + dislocation-population dispatch helpers (sub-projects B+C)."""

from __future__ import annotations

import numpy as np

from dfxm_geo.direct_space.forward_model import (
    DislocationPopulation,
    ScanGrid,
    build_dislocation_population,
    build_scan_grid,
)
from dfxm_geo.pipeline import (
    CenteredCrystalConfig,
    CrystalConfig,
    ScanConfig,
    WallCrystalConfig,
)


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


class TestBuildDislocationPopulationCentered:
    def test_returns_single_dislocation_at_origin(self) -> None:
        crystal = CrystalConfig(
            mode="centered",
            centered=CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 1, -2)),
        )
        pop = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=None)
        assert isinstance(pop, DislocationPopulation)
        assert pop.positions_um.shape == (1, 3)
        np.testing.assert_allclose(pop.positions_um[0], [0.0, 0.0, 0.0])
        assert pop.Ud.shape == (1, 3, 3)
        assert pop.sidecar is None  # no sidecar for centered mode

    def test_Ud_built_from_b_n_t(self) -> None:
        crystal = CrystalConfig(
            mode="centered",
            centered=CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 1, -2)),
        )
        pop = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=None)
        # Columns of Ud are (normalized) b, n, t.
        b_norm = np.array([1, -1, 0]) / np.linalg.norm([1, -1, 0])
        n_norm = np.array([1, 1, 1]) / np.linalg.norm([1, 1, 1])
        t_norm = np.array([1, 1, -2]) / np.linalg.norm([1, 1, -2])
        np.testing.assert_allclose(pop.Ud[0, :, 0], b_norm, atol=1e-12)
        np.testing.assert_allclose(pop.Ud[0, :, 1], n_norm, atol=1e-12)
        np.testing.assert_allclose(pop.Ud[0, :, 2], t_norm, atol=1e-12)


class TestBuildDislocationPopulationWall:
    def test_returns_ndis_dislocations(self) -> None:
        crystal = CrystalConfig(
            mode="wall",
            wall=WallCrystalConfig(dis=4.0, ndis=11, sample_remount="S1"),
        )
        pop = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=None)
        assert pop.positions_um.shape == (11, 3)
        assert pop.Ud.shape == (11, 3, 3)
        assert pop.sidecar is None

    def test_wall_population_is_deterministic(self) -> None:
        crystal = CrystalConfig(
            mode="wall",
            wall=WallCrystalConfig(dis=4.0, ndis=11, sample_remount="S1"),
        )
        pop_a = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=None)
        pop_b = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=None)
        np.testing.assert_array_equal(pop_a.positions_um, pop_b.positions_um)
        np.testing.assert_array_equal(pop_a.Ud, pop_b.Ud)
