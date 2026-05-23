"""Sub-project F: dataclass-level defaults for the empty-TOML path."""

from __future__ import annotations

import pytest

from dfxm_geo.pipeline import (  # noqa: F401
    CenteredCrystalConfig,
    CrystalConfig,
    IdentificationConfig,
    IdentificationCrystalConfig,
    ReciprocalConfig,
    SimulationConfig,
)


class TestCenteredCrystalDefaults:
    def test_bare_construction_uses_canonical_fcc_primary(self) -> None:
        cfg = CenteredCrystalConfig()
        assert cfg.b == (1, 0, -1)
        assert cfg.n == (1, 1, 1)
        assert cfg.t == (1, -2, 1)

    def test_canonical_defaults_satisfy_validators(self) -> None:
        # b · n == 0 and t ∥ (n × b) — the __post_init__ checks.
        # Construction passing without ValueError is the assertion.
        CenteredCrystalConfig()


class TestReciprocalDefaults:
    def test_bare_construction_uses_al_111_17kev(self) -> None:
        cfg = ReciprocalConfig()
        assert cfg.hkl == (-1, 1, -1)
        assert cfg.keV == 17.0

    def test_from_dict_none_returns_default(self) -> None:
        cfg = ReciprocalConfig.from_dict(None)
        assert cfg.hkl == (-1, 1, -1)
        assert cfg.keV == 17.0

    def test_from_dict_empty_returns_default(self) -> None:
        cfg = ReciprocalConfig.from_dict({})
        assert cfg.hkl == (-1, 1, -1)
        assert cfg.keV == 17.0


class TestCrystalConfigDefault:
    def test_default_classmethod_returns_centered_canonical(self) -> None:
        cfg = CrystalConfig.default()
        assert cfg.mode == "centered"
        assert cfg.centered is not None
        assert cfg.wall is None
        assert cfg.random_dislocations is None
        assert cfg.centered.b == (1, 0, -1)
        assert cfg.centered.n == (1, 1, 1)
        assert cfg.centered.t == (1, -2, 1)

    def test_from_dict_empty_returns_default(self) -> None:
        # The None-path equivalent is covered in
        # test_pipeline_crystal_modes.py::TestCrystalConfigFromDict.test_none_dict_returns_default,
        # which exercises the same code path in context with the rest of the
        # parser. Here we own the empty-dict {} case.
        cfg = CrystalConfig.from_dict({})
        assert cfg.mode == "centered"
        assert cfg.centered is not None
        assert cfg.centered.b == (1, 0, -1)

    def test_from_dict_mode_only_still_requires_sub_block(self) -> None:
        # Explicit `mode = "centered"` with no `[crystal.centered]` still raises:
        # if the user wrote `[crystal]`, they intended to specify something.
        with pytest.raises(ValueError, match=r"\[crystal\.centered\] sub-block is required"):
            CrystalConfig.from_dict({"mode": "centered"})

        with pytest.raises(ValueError, match=r"\[crystal\.wall\] sub-block is required"):
            CrystalConfig.from_dict({"mode": "wall"})


class TestSimulationConfigDefaults:
    def test_bare_construction_succeeds(self) -> None:
        cfg = SimulationConfig()
        # Crystal cascades to default
        assert cfg.crystal.mode == "centered"
        assert cfg.crystal.centered is not None
        # Reciprocal cascades to default
        assert cfg.reciprocal is not None
        assert cfg.reciprocal.hkl == (-1, 1, -1)
        assert cfg.reciprocal.keV == 17.0
        # Scan defaults to all-axes-fixed (single mode)
        assert cfg.scan.scanned_axes() == ()
        assert cfg.scan.derived_mode_name() == "single"


class TestIdentificationCrystalDefaults:
    def test_bare_construction_uses_slip_plane_111(self) -> None:
        cfg = IdentificationCrystalConfig()
        assert cfg.slip_plane_normal == (1, 1, 1)
        # Other fields already had defaults — spot-check:
        assert cfg.sweep_all_slip_planes is True
        assert cfg.exclude_invisibility is True
