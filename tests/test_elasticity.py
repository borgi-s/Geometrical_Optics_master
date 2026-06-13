"""Per-material Poisson ratio resolution."""

import pytest

from dfxm_geo.crystal.elasticity import poisson_ratio, poisson_source


def test_override_wins():
    assert poisson_ratio(override=0.41, material="Fe") == 0.41


def test_material_lookup():
    assert poisson_ratio(override=None, material="Fe") == pytest.approx(0.29, abs=0.005)
    assert poisson_ratio(override=None, material="W") == pytest.approx(0.28, abs=0.005)


def test_default_is_al():
    assert poisson_ratio(override=None, material=None) == pytest.approx(0.334)


def test_unknown_material_raises():
    with pytest.raises(ValueError, match="unknown material"):
        poisson_ratio(override=None, material="Unobtainium")


def test_poisson_source_returns_citation_tag():
    assert poisson_source("Fe") in ("KL", "SW")
    assert poisson_source(None) == "SW"  # default Al value source
