"""[crystal] structure-key parsing + slip-system hatch."""

import pytest

from dfxm_geo.crystal.slip_systems import register_custom, slip_systems
from dfxm_geo.reciprocal_space.kernel import _crystal_mount_from_toml


def test_crystal_block_parses_structure_keys(tmp_path):
    block = {
        "lattice": "cubic",
        "a": 2.8665e-10,
        "structure_type": "bcc",
        "material": "Fe",
        "poisson_ratio": 0.30,
        "slip_families": ["{110}<111>"],
    }
    mount = _crystal_mount_from_toml(block, base_dir=tmp_path)
    assert mount.resolved_structure_type == "bcc"
    assert mount.resolved_poisson_ratio == 0.30
    assert mount.slip_families == ("{110}<111>",)


def test_custom_slip_systems_registration():
    register_custom("mycustom", [{"plane": (1, 0, 0), "burgers": (0, 1, 0)}])
    sys = slip_systems("mycustom")
    assert len(sys) == 1
    assert sys[0].n == (1, 0, 0) and sys[0].b == (0, 1, 0)


def test_custom_slip_system_rejects_non_glide():
    with pytest.raises(ValueError, match="b.n"):
        register_custom("bad", [{"plane": (1, 0, 0), "burgers": (1, 0, 0)}])


def test_structure_type_and_slip_system_mutually_exclusive(tmp_path):
    block = {
        "lattice": "cubic",
        "a": 3e-10,
        "structure_type": "bcc",
        "slip_system": [{"plane": (1, 0, 0), "burgers": (0, 1, 0)}],
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        _crystal_mount_from_toml(block, base_dir=tmp_path)
