"""Unit tests for the per-mode crystal sub-block dataclasses (sub-project C)."""

from __future__ import annotations

import pytest

from dfxm_geo.pipeline import (
    CenteredCrystalConfig,
    CrystalConfig,
    RandomDislocationsConfig,
    WallCrystalConfig,
)


class TestCenteredCrystalConfig:
    def test_constructs_with_valid_111_b_n_t(self) -> None:
        # b = [1, -1, 0]; n = [1, 1, 1]; b·n = 0 → valid.
        # t = [1, 1, -2] is the canonical line direction for this slip system.
        cfg = CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 1, -2))
        assert cfg.b == (1, -1, 0)
        assert cfg.n == (1, 1, 1)
        assert cfg.t == (1, 1, -2)

    def test_b_not_perpendicular_to_n_rejected(self) -> None:
        # b·n = 1 + 1 + 1 = 3 ≠ 0
        with pytest.raises(ValueError, match="Burgers vector .* must be perpendicular"):
            CenteredCrystalConfig(b=(1, 1, 1), n=(1, 1, 1), t=(0, 1, -1))

    def test_t_not_consistent_with_n_cross_b_rejected(self) -> None:
        # n × b is perpendicular to both n and b; if t isn't parallel, reject.
        # For n=(1,1,1), b=(1,-1,0): n × b = (1, 1, -2). t=(1,0,0) is not parallel.
        with pytest.raises(ValueError, match="line direction .* must be parallel"):
            CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 0, 0))


class TestWallCrystalConfig:
    def test_explicit_construction_succeeds(self) -> None:
        cfg = WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S1")
        assert cfg.dis == 4.0
        assert cfg.ndis == 151
        assert cfg.sample_remount == "S1"

    def test_custom_remount(self) -> None:
        cfg = WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S2")
        assert cfg.sample_remount == "S2"

    def test_invalid_remount_rejected(self) -> None:
        with pytest.raises(ValueError, match="sample_remount must be one of"):
            WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S9")


class TestRandomDislocationsConfig:
    def test_minimum_ndis_is_one(self) -> None:
        cfg = RandomDislocationsConfig(ndis=1)
        assert cfg.ndis == 1
        assert cfg.sigma is None
        assert cfg.min_distance is None
        assert cfg.seed is None

    def test_ndis_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="`ndis` must be >= 1"):
            RandomDislocationsConfig(ndis=0)

    def test_negative_sigma_rejected(self) -> None:
        with pytest.raises(ValueError, match="`sigma` must be > 0"):
            RandomDislocationsConfig(ndis=4, sigma=-1.0)

    def test_negative_min_distance_rejected(self) -> None:
        with pytest.raises(ValueError, match="`min_distance` must be >= 0"):
            RandomDislocationsConfig(ndis=4, min_distance=-1.0)

    def test_all_fields_set(self) -> None:
        cfg = RandomDislocationsConfig(ndis=4, sigma=5.0, min_distance=4.0, seed=42)
        assert cfg.ndis == 4
        assert cfg.sigma == 5.0
        assert cfg.min_distance == 4.0
        assert cfg.seed == 42


class TestCrystalConfigFromDict:
    def test_centered_mode_parses(self) -> None:
        cfg = CrystalConfig.from_dict(
            {
                "mode": "centered",
                "centered": {"b": [1, -1, 0], "n": [1, 1, 1], "t": [1, 1, -2]},
            }
        )
        assert cfg.mode == "centered"
        assert cfg.centered is not None
        assert cfg.centered.b == (1, -1, 0)
        assert cfg.wall is None
        assert cfg.random_dislocations is None

    def test_wall_mode_parses(self) -> None:
        cfg = CrystalConfig.from_dict(
            {
                "mode": "wall",
                "wall": {"dis": 4.0, "ndis": 151, "sample_remount": "S1"},
            }
        )
        assert cfg.mode == "wall"
        assert cfg.wall is not None
        assert cfg.wall.ndis == 151
        assert cfg.centered is None
        assert cfg.random_dislocations is None

    def test_random_dislocations_mode_parses(self) -> None:
        cfg = CrystalConfig.from_dict(
            {
                "mode": "random_dislocations",
                "random_dislocations": {"ndis": 4, "sigma": 5.0, "seed": 42},
            }
        )
        assert cfg.mode == "random_dislocations"
        assert cfg.random_dislocations is not None
        assert cfg.random_dislocations.ndis == 4
        assert cfg.centered is None
        assert cfg.wall is None

    def test_none_dict_returns_default(self) -> None:
        # Sub-project F: from_dict(None) now returns the canonical centered default.
        cfg = CrystalConfig.from_dict(None)
        assert cfg.mode == "centered"
        assert cfg.centered is not None

    def test_missing_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"missing `mode` in \[crystal\]"):
            CrystalConfig.from_dict(
                {"centered": {"b": [1, -1, 0], "n": [1, 1, 1], "t": [1, 1, -2]}}
            )

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown crystal mode 'bicrystal'"):
            CrystalConfig.from_dict({"mode": "bicrystal"})

    def test_missing_required_subblock_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[crystal.centered\] sub-block is required"):
            CrystalConfig.from_dict({"mode": "centered"})

    def test_sibling_subblock_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"extra sub-block.*'wall'"):
            CrystalConfig.from_dict(
                {
                    "mode": "centered",
                    "centered": {"b": [1, -1, 0], "n": [1, 1, 1], "t": [1, 1, -2]},
                    "wall": {"dis": 4.0, "ndis": 151, "sample_remount": "S1"},
                }
            )

    def test_multiple_sibling_subblocks_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"extra sub-block.*'random_dislocations'.*'wall'"):
            CrystalConfig.from_dict(
                {
                    "mode": "centered",
                    "centered": {"b": [1, -1, 0], "n": [1, 1, 1], "t": [1, 1, -2]},
                    "wall": {"dis": 4.0, "ndis": 151, "sample_remount": "S1"},
                    "random_dislocations": {"ndis": 4},
                }
            )

    def test_tuple_conversion_for_b_n_t(self) -> None:
        cfg = CrystalConfig.from_dict(
            {
                "mode": "centered",
                "centered": {"b": [1, -1, 0], "n": [1, 1, 1], "t": [1, 1, -2]},
            }
        )
        assert isinstance(cfg.centered.b, tuple)
        assert isinstance(cfg.centered.n, tuple)
        assert isinstance(cfg.centered.t, tuple)
