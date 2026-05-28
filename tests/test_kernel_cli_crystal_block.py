"""cli_main parses [crystal] block into a CrystalMount and uses it for Bragg θ."""

import tomllib

import pytest

from dfxm_geo.reciprocal_space.kernel import _crystal_mount_from_toml


def test_paper_al_crystal_block_parses() -> None:
    toml_str = """
    [crystal]
    lattice  = "cubic"
    a        = 4.0493e-10
    mount_x  = [1, 0, 0]
    mount_y  = [0, 1, 0]
    mount_z  = [0, 0, 1]
    """
    data = tomllib.loads(toml_str)
    mount = _crystal_mount_from_toml(data["crystal"])
    assert mount.lattice == "cubic"
    assert mount.a == 4.0493e-10
    assert mount.mount_x == (1, 0, 0)


def test_crystal_block_default_omitted_uses_paper_al() -> None:
    """When [crystal] is absent, default to paper Al setup (CLAUDE.md note)."""
    mount = _crystal_mount_from_toml(None)
    assert mount.lattice == "cubic"
    assert mount.mount_x == (1, 0, 0)


def test_invalid_crystal_block_propagates_ValueError() -> None:
    bad = {
        "lattice": "cubic",
        "a": 4.0e-10,
        "mount_x": [1, 0, 0],
        "mount_y": [1, 1, 0],
        "mount_z": [0, 0, 1],  # not orthogonal
    }
    with pytest.raises(ValueError, match="mutually orthogonal"):
        _crystal_mount_from_toml(bad)
