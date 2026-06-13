"""[crystal] structure-key parsing + slip-system hatch."""

import pytest

from dfxm_geo.crystal.slip_systems import register_custom, slip_systems
from dfxm_geo.pipeline import SimulationConfig
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


# --- I1: simplified mode must reject structure-aware [crystal] keys ----------

_SIMPLIFIED_HEAD = (
    '[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\nbackend = "analytic"\nbeamstop = false\n\n'
)


@pytest.mark.parametrize(
    "extra",
    [
        'structure_type = "bcc"\n',
        'material = "Fe"\n',
        "poisson_ratio = 0.29\n",
        'slip_families = ["{110}<111>"]\n',
        "[[crystal.slip_system]]\nplane = [1, 1, 0]\nburgers = [1, -1, 1]\n",
    ],
)
def test_simplified_mode_rejects_structure_keys(tmp_path, extra):
    """Simplified geometry discards the mount, so a structure key there would
    silently resolve to FCC. The parser must raise loudly instead (M4 4.3a I1)."""
    cfg_toml = (
        _SIMPLIFIED_HEAD
        + "[crystal]\n"
        + 'mode = "centered"\n'
        + "\n"
        + extra
        + "\n"
        + "[crystal.centered]\n"
        + "b = [1, -1, 0]\n"
        + "n = [1, 1, 1]\n"
        + "t = [1, 1, -2]\n"
        + "\n"
        + "[scan.phi]\n"
        + "value = 0.0\n"
    )
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(cfg_toml, encoding="utf-8")
    with pytest.raises(ValueError, match='mode = "oblique"'):
        SimulationConfig.from_toml(cfg_path)


def test_simplified_mode_plain_cell_keys_still_work(tmp_path):
    """Back-compat: simplified mode with ONLY plain cubic cell keys is accepted
    (no structure keys) and resolves to the FCC mount=None default path."""
    cfg_toml = (
        _SIMPLIFIED_HEAD
        + "[crystal]\n"
        + 'lattice = "cubic"\n'
        + "a = 4.0495e-10\n"
        + 'mode = "centered"\n'
        + "\n"
        + "[crystal.centered]\n"
        + "b = [1, -1, 0]\n"
        + "n = [1, 1, 1]\n"
        + "t = [1, 1, -2]\n"
        + "\n"
        + "[scan.phi]\n"
        + "value = 0.0\n"
    )
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(cfg_toml, encoding="utf-8")
    cfg = SimulationConfig.from_toml(cfg_path)
    assert cfg.geometry.mode == "simplified"
    assert cfg.geometry.mount is None
