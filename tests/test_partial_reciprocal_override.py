"""Sub-project F: ReciprocalConfig.from_dict accepts partial overrides."""

from __future__ import annotations

from dfxm_geo.pipeline import ReciprocalConfig


class TestPartialReciprocalOverride:
    def test_only_keV_provided_keeps_default_hkl(self) -> None:
        cfg = ReciprocalConfig.from_dict({"keV": 21.0})
        assert cfg.hkl == (-1, 1, -1)
        assert cfg.keV == 21.0

    def test_only_hkl_provided_keeps_default_keV(self) -> None:
        cfg = ReciprocalConfig.from_dict({"hkl": [1, 1, 1]})
        assert cfg.hkl == (1, 1, 1)
        assert cfg.keV == 17.0

    def test_both_provided_uses_both(self) -> None:
        cfg = ReciprocalConfig.from_dict({"hkl": [2, 0, 0], "keV": 19.5})
        assert cfg.hkl == (2, 0, 0)
        assert cfg.keV == 19.5
