"""Sub-project F: dataclass-level defaults for the empty-TOML path."""

from __future__ import annotations

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
