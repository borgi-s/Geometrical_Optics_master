"""All shipped TOML configs load under the new B+C schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from dfxm_geo.pipeline import SimulationConfig, load_identification_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


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
