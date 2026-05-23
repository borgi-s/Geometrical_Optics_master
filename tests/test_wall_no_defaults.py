"""Sub-project F: WallCrystalConfig has no defaults (breaking change for v2.0.0)."""

from __future__ import annotations

import pytest

from dfxm_geo.pipeline import WallCrystalConfig


class TestWallNoDefaults:
    def test_bare_construction_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="missing 3 required"):
            WallCrystalConfig()  # type: ignore[call-arg]

    def test_missing_two_fields_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="missing 2 required"):
            WallCrystalConfig(dis=4.0)  # type: ignore[call-arg]

    def test_missing_one_field_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="missing 1 required"):
            WallCrystalConfig(dis=4.0, ndis=151)  # type: ignore[call-arg]

    def test_all_three_provided_succeeds(self) -> None:
        cfg = WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S1")
        assert cfg.dis == 4.0
        assert cfg.ndis == 151
        assert cfg.sample_remount == "S1"
