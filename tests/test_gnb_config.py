"""Tests for GnbCrystalConfig + CrystalConfig.from_dict gnb branch (Task 6)."""

import pytest

from dfxm_geo.config import CrystalConfig


def test_gnb_named_recipe_resolves():
    c = CrystalConfig.from_dict(
        {"mode": "gnb", "gnb": {"recipe": "frankus", "theta_deg": 0.05, "extent_um": 25.0}}
    )
    assert c.mode == "gnb"
    assert c.gnb.recipe == "frankus"
    assert c.gnb.to_recipe().name == "frankus"


def test_gnb_custom_recipe_parses_list_of_tables():
    c = CrystalConfig.from_dict(
        {
            "mode": "gnb",
            "gnb": {
                "recipe": "custom",
                "theta_deg": 0.05,
                "extent_um": 25.0,
                "custom": {
                    "n": [1, 1, 1],
                    "a": [1, 1, 1],
                    "set": [
                        {
                            "b": [1, 0, -1],
                            "xi": [2, -1, -1],
                            "slip_plane": [1, 1, 1],
                            "rel_density": 1.0,
                        },
                        {
                            "b": [0, 1, -1],
                            "xi": [-1, 2, -1],
                            "slip_plane": [1, 1, 1],
                            "rel_density": 1.0,
                        },
                    ],
                },
            },
        }
    )
    r = c.gnb.to_recipe()
    assert len(r.sets) == 2 and r.n == (1, 1, 1)


def test_gnb_custom_required_when_recipe_custom():
    with pytest.raises(ValueError, match="custom"):
        CrystalConfig.from_dict(
            {"mode": "gnb", "gnb": {"recipe": "custom", "theta_deg": 0.05, "extent_um": 25.0}}
        )


def test_gnb_theta_must_be_positive():
    with pytest.raises(ValueError):
        CrystalConfig.from_dict(
            {"mode": "gnb", "gnb": {"recipe": "frankus", "theta_deg": 0.0, "extent_um": 25.0}}
        )
