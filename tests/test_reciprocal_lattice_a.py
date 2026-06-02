"""#11 end-to-end: ReciprocalConfig carries a `lattice_a` field.

forward_model._load_analytic_resolution already reads `config.lattice_a`
(getattr, Al default) when deriving the Bragg theta. This pins the producing
side: ReciprocalConfig exposes a `lattice_a` field (default Al 4.0495e-10),
parses it from the `[reciprocal]` TOML block, and feeds it to the
__post_init__ reflection-validity check (which previously hardcoded the Al
value). Together they let a non-Al lattice flow from TOML end-to-end.
"""

from __future__ import annotations

import pytest

from dfxm_geo.pipeline import ReciprocalConfig


def test_reciprocal_config_defaults_to_al_lattice() -> None:
    assert ReciprocalConfig().lattice_a == pytest.approx(4.0495e-10)


def test_reciprocal_config_accepts_lattice_a() -> None:
    cfg = ReciprocalConfig(lattice_a=5.0e-10)
    assert cfg.lattice_a == pytest.approx(5.0e-10)


def test_reciprocal_config_from_dict_parses_lattice_a() -> None:
    cfg = ReciprocalConfig.from_dict({"hkl": [-1, 1, -1], "keV": 17.0, "lattice_a": 4.2e-10})
    assert cfg.lattice_a == pytest.approx(4.2e-10)
