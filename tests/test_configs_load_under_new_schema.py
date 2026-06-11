"""All shipped TOML configs load under the new B+C schema."""

from __future__ import annotations

import pytest

from dfxm_geo.data import configs_root
from dfxm_geo.pipeline import SimulationConfig, load_identification_config

HEX_CRYSTAL_LINES = """
[crystal]
lattice = "hexagonal"
a       = 3.2094e-10
c       = 5.2108e-10
mount_x = [2, -1, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
mode = "centered"
[crystal.centered]
b = [1, 0, -1]
n = [1, 1, 1]
t = [1, -2, 1]
"""

CONFIGS_DIR = configs_root()


@pytest.mark.parametrize(
    "config_name",
    [
        "default.toml",
        "variants/dis_0p25.toml",
        "variants/dis_0p5.toml",
        "variants/dis_1.toml",
        "variants/dis_2.toml",
        "variants/sample_remount_S2.toml",
    ],
)
def test_forward_config_loads(config_name: str) -> None:
    path = CONFIGS_DIR / config_name
    cfg = SimulationConfig.from_toml(path)
    assert cfg.crystal.mode in ("centered", "wall", "random_dislocations")


@pytest.mark.parametrize(
    "config_name",
    [
        "identification_single.toml",
        "identification_multi.toml",
        "identification_zscan.toml",
    ],
)
def test_identification_config_loads(config_name: str) -> None:
    path = CONFIGS_DIR / config_name
    cfg = load_identification_config(path)
    assert cfg.mode in ("single", "multi", "z-scan")


def test_noncubic_simplified_mode_rejected(tmp_path):
    cfg = (
        HEX_CRYSTAL_LINES
        + """
[geometry]
mode = "simplified"
"""
    )
    p = tmp_path / "hex_simplified.toml"
    p.write_text(cfg, encoding="utf-8")
    with pytest.raises(ValueError, match="non-cubic .* require .*oblique"):
        SimulationConfig.from_toml(p)


def test_cubic_simplified_mode_with_lattice_key_loads(tmp_path):
    cfg = """
[crystal]
lattice = "cubic"
a       = 4.0495e-10
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
mode = "centered"
[crystal.centered]
b = [1, 0, -1]
n = [1, 1, 1]
t = [1, -2, 1]

[geometry]
mode = "simplified"
"""
    p = tmp_path / "cubic_simplified.toml"
    p.write_text(cfg, encoding="utf-8")
    config = SimulationConfig.from_toml(p)
    assert config.geometry.mode == "simplified"


def test_noncubic_oblique_forward_rejected_until_stage43(tmp_path):
    cfg = (
        HEX_CRYSTAL_LINES
        + """
[geometry]
mode = "oblique"
eta = 0.0
"""
    )
    p = tmp_path / "hex_oblique.toml"
    p.write_text(cfg, encoding="utf-8")
    with pytest.raises(ValueError, match="Stage 4.3"):
        SimulationConfig.from_toml(p)
