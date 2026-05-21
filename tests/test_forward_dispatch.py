"""Tests for the scan-grid + dislocation-population dispatch helpers (sub-projects B+C)."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.direct_space.forward_model import (
    DislocationPopulation,
    ScanGrid,
    build_dislocation_population,
    build_scan_grid,
)
from dfxm_geo.pipeline import (
    CenteredCrystalConfig,
    CrystalConfig,
    RandomDislocationsConfig,
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
        # Columns of Ud are (normalized) b, n, and t (with t possibly
        # flipped so det=+1 — the legacy IUCrJ 2024 right-handed convention).
        # For b=(1,-1,0), n=(1,1,1), t=(1,1,-2): the raw column stack has
        # det=-1, so _ud_matrix_from_bnt flips t.
        b_norm = np.array([1, -1, 0]) / np.linalg.norm([1, -1, 0])
        n_norm = np.array([1, 1, 1]) / np.linalg.norm([1, 1, 1])
        t_norm_flipped = -np.array([1, 1, -2]) / np.linalg.norm([1, 1, -2])
        np.testing.assert_allclose(pop.Ud[0, :, 0], b_norm, atol=1e-12)
        np.testing.assert_allclose(pop.Ud[0, :, 1], n_norm, atol=1e-12)
        np.testing.assert_allclose(pop.Ud[0, :, 2], t_norm_flipped, atol=1e-12)
        # And the result is a proper rotation (det=+1).
        np.testing.assert_allclose(np.linalg.det(pop.Ud[0]), 1.0, atol=1e-12)


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


class TestBuildDislocationPopulationRandomDislocations:
    def _config(self, **kwargs) -> CrystalConfig:
        return CrystalConfig(
            mode="random_dislocations",
            random_dislocations=RandomDislocationsConfig(**kwargs),
        )

    def test_seeded_run_is_deterministic(self) -> None:
        crystal = self._config(ndis=4, sigma=5.0, seed=42)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        pop_a = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=rng_a)
        pop_b = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=rng_b)
        np.testing.assert_array_equal(pop_a.positions_um, pop_b.positions_um)
        np.testing.assert_array_equal(pop_a.Ud, pop_b.Ud)

    def test_ndis_dislocations_returned(self) -> None:
        crystal = self._config(ndis=7, sigma=5.0, seed=1)
        pop = build_dislocation_population(
            crystal, fov_lateral_um=20.4, rng=np.random.default_rng(1)
        )
        assert pop.positions_um.shape == (7, 3)
        assert pop.Ud.shape == (7, 3, 3)

    def test_sigma_default_uses_fov(self) -> None:
        crystal = self._config(ndis=4, sigma=None, seed=1)
        pop = build_dislocation_population(
            crystal, fov_lateral_um=20.4, rng=np.random.default_rng(1)
        )
        assert pop.sidecar is not None
        assert pop.sidecar["sigma_um"] == pytest.approx(5.1)
        assert pop.sidecar["sigma_source"] == "default-fov"

    def test_sigma_override_is_recorded(self) -> None:
        crystal = self._config(ndis=4, sigma=3.0, seed=1)
        pop = build_dislocation_population(
            crystal, fov_lateral_um=20.4, rng=np.random.default_rng(1)
        )
        assert pop.sidecar["sigma_um"] == 3.0
        assert pop.sidecar["sigma_source"] == "user"

    def test_min_distance_enforced(self) -> None:
        crystal = self._config(ndis=5, sigma=10.0, min_distance=2.0, seed=42)
        pop = build_dislocation_population(
            crystal, fov_lateral_um=40.0, rng=np.random.default_rng(42)
        )
        xy = pop.positions_um[:, :2]
        for i in range(len(xy)):
            for j in range(i + 1, len(xy)):
                d = float(np.linalg.norm(xy[i] - xy[j]))
                assert d >= 2.0, f"pair ({i},{j}) too close: {d}"

    def test_impossible_min_distance_raises_runtime_error(self) -> None:
        crystal = self._config(ndis=100, sigma=0.05, min_distance=10.0, seed=1)
        with pytest.raises(RuntimeError, match="exceeded retry budget"):
            build_dislocation_population(crystal, fov_lateral_um=20.4, rng=np.random.default_rng(1))

    def test_sidecar_lists_realized_b_n_t_per_dislocation(self) -> None:
        crystal = self._config(ndis=3, sigma=5.0, seed=42)
        pop = build_dislocation_population(
            crystal, fov_lateral_um=20.4, rng=np.random.default_rng(42)
        )
        assert pop.sidecar is not None
        assert len(pop.sidecar["dislocations"]) == 3
        for entry in pop.sidecar["dislocations"]:
            assert "x_um" in entry and "y_um" in entry and "z_um" in entry
            assert "b" in entry and "n" in entry and "t" in entry
            assert len(entry["b"]) == 3

    def test_seed_source_user_when_explicit(self) -> None:
        crystal = self._config(ndis=2, sigma=5.0, seed=42)
        pop = build_dislocation_population(
            crystal, fov_lateral_um=20.4, rng=np.random.default_rng(42)
        )
        assert pop.sidecar["seed_source"] == "user"
        assert pop.sidecar["seed"] == 42

    def test_seed_source_entropy_when_absent(self) -> None:
        crystal = self._config(ndis=2, sigma=5.0, seed=None)
        pop = build_dislocation_population(crystal, fov_lateral_um=20.4, rng=None)
        assert pop.sidecar["seed_source"] == "entropy"
        assert isinstance(pop.sidecar["seed"], int)
