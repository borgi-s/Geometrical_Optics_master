"""Stage 4.2: provenance TOML emits space_group; cubic no-SG output byte-identical."""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.config import SimulationConfig, _dataclass_to_toml_str

DATA = Path(__file__).parent / "data" / "cif"

MOUNT_LINES = "mount_x = [1, 0, 0]\nmount_y = [0, 1, 0]\nmount_z = [0, 0, 1]\n"


def _load(tmp_path: Path, crystal_extra: str) -> SimulationConfig:
    shutil.copy(DATA / "al_fm3m.cif", tmp_path / "al.cif")
    p = tmp_path / "config.toml"
    # hkl=(1,1,1) at 17 keV → eta1 ≈ -0.9465 rad (all-odd, so allowed in Fm-3m).
    p.write_text(
        f"[crystal]\n{crystal_extra}{MOUNT_LINES}"
        '[geometry]\nmode = "oblique"\neta = -0.9465\n'
        "[reciprocal]\nhkl = [1, 1, 1]\nkeV = 17.0\n",
        encoding="utf-8",
    )
    return SimulationConfig.from_toml(p)


def test_space_group_emitted(tmp_path: Path) -> None:
    cfg = _load(tmp_path, 'cif = "al.cif"\n')
    toml_str = _dataclass_to_toml_str(cfg)
    assert 'space_group = "F m -3 m"' in toml_str


def test_cif_path_never_emitted(tmp_path: Path) -> None:
    cfg = _load(tmp_path, 'cif = "al.cif"\n')
    crystal_block = _dataclass_to_toml_str(cfg).split("[crystal]")[1].split("\n[")[0]
    assert "cif" not in crystal_block  # resolved values only; no raw file path


def test_no_space_group_line_when_none(tmp_path: Path) -> None:
    cfg = _load(tmp_path, 'lattice = "cubic"\na = 4.0495e-10\n')
    assert "space_group" not in _dataclass_to_toml_str(cfg)


def test_provenance_round_trips_space_group(tmp_path: Path) -> None:
    cfg = _load(tmp_path, 'cif = "al.cif"\n')
    echo = tmp_path / "echo.toml"
    echo.write_text(_dataclass_to_toml_str(cfg), encoding="utf-8")
    cfg2 = SimulationConfig.from_toml(echo)
    assert cfg2.geometry.mount is not None
    assert cfg2.geometry.mount.space_group == "F m -3 m"
