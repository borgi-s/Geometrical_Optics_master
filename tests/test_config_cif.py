"""Stage 4.2 config semantics: cif trigger, lattice_a inheritance, extinct rejection."""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.config import SimulationConfig, load_identification_config

DATA = Path(__file__).parent / "data" / "cif"


def _write(tmp_path: Path, body: str) -> Path:
    shutil.copy(DATA / "al_fm3m.cif", tmp_path / "al.cif")
    shutil.copy(DATA / "mg_p63mmc.cif", tmp_path / "mg.cif")
    p = tmp_path / "config.toml"
    p.write_text(body, encoding="utf-8")
    return p


MOUNT_LINES = """
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
"""


class TestSimplifiedMode:
    def test_cubic_cif_inherits_lattice_a(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            f'[crystal]\ncif = "al.cif"\n{MOUNT_LINES}\n'
            '[geometry]\nmode = "simplified"\n'
            "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n",
        )
        cfg = SimulationConfig.from_toml(p)
        assert cfg.reciprocal.lattice_a == pytest.approx(4.0495e-10)

    def test_explicit_lattice_a_wins_over_cif(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            f'[crystal]\ncif = "al.cif"\n{MOUNT_LINES}\n'
            '[geometry]\nmode = "simplified"\n'
            "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\nlattice_a = 4.05e-10\n",
        )
        cfg = SimulationConfig.from_toml(p)
        assert cfg.reciprocal.lattice_a == 4.05e-10

    def test_noncubic_cif_rejected_in_simplified(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            '[crystal]\ncif = "mg.cif"\n'
            "mount_x = [1, 0, 0]\nmount_y = [1, -2, 0]\nmount_z = [0, 0, 1]\n"
            '[geometry]\nmode = "simplified"\n',
        )
        with pytest.raises(ValueError, match="non-cubic"):
            SimulationConfig.from_toml(p)

    def test_extinct_hkl_rejected_simplified(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            f'[crystal]\ncif = "al.cif"\n{MOUNT_LINES}\n'
            '[geometry]\nmode = "simplified"\n'
            "[reciprocal]\nhkl = [1, 0, 0]\nkeV = 17.0\n",
        )
        with pytest.raises(ValueError, match="systematically absent"):
            SimulationConfig.from_toml(p)

    def test_bare_space_group_key_rejects_extinct(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            '[crystal]\nspace_group = "Fm-3m"\n[reciprocal]\nhkl = [1, 0, 0]\nkeV = 17.0\n',
        )
        with pytest.raises(ValueError, match="systematically absent"):
            SimulationConfig.from_toml(p)

    def test_bare_space_group_allowed_hkl_passes(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            '[crystal]\nspace_group = "Fm-3m"\n[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n',
        )
        cfg = SimulationConfig.from_toml(p)
        assert cfg.geometry.mode == "simplified"


class TestObliqueMode:
    def test_extinct_hkl_rejected_oblique(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            f'[crystal]\ncif = "al.cif"\n{MOUNT_LINES}\n'
            '[geometry]\nmode = "oblique"\neta = 0.0\n'
            "[reciprocal]\nhkl = [1, 0, 0]\nkeV = 17.0\n",
        )
        with pytest.raises(ValueError, match="systematically absent"):
            SimulationConfig.from_toml(p)

    def test_mount_carries_space_group(self, tmp_path: Path) -> None:
        # eta/hkl is a valid identity-mount oblique pair ((-1,-1,3) @ 17 keV,
        # eta=0.324555 = the solver's eta_1); the point of this test is that
        # the parsed mount surfaces the CIF space group.
        p = _write(
            tmp_path,
            f'[crystal]\ncif = "al.cif"\n{MOUNT_LINES}\n'
            '[geometry]\nmode = "oblique"\neta = 0.324555\n'
            "[reciprocal]\nhkl = [-1, -1, 3]\nkeV = 17.0\n",
        )
        cfg = SimulationConfig.from_toml(p)
        assert cfg.geometry.mount is not None
        assert cfg.geometry.mount.space_group == "F m -3 m"


class TestIdentificationConfig:
    def test_identify_loader_strips_new_keys_and_checks_extinct(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            f'[crystal]\ncif = "al.cif"\n{MOUNT_LINES}\n'
            '[geometry]\nmode = "oblique"\neta = 0.0\n'
            "[reciprocal]\nhkl = [1, 0, 0]\nkeV = 17.0\n",
        )
        with pytest.raises(ValueError, match="systematically absent"):
            load_identification_config(p)

    def test_identify_loader_happy_path(self, tmp_path: Path) -> None:
        # Same valid identity-mount oblique pair as TestObliqueMode
        # (eta=0.0 does not validate for (-1,1,-1) with the identity mount).
        p = _write(
            tmp_path,
            f'[crystal]\ncif = "al.cif"\n{MOUNT_LINES}\n'
            '[geometry]\nmode = "oblique"\neta = 0.324555\n'
            "[reciprocal]\nhkl = [-1, -1, 3]\nkeV = 17.0\n",
        )
        cfg = load_identification_config(p)
        assert cfg.geometry.mount is not None
        assert cfg.geometry.mount.space_group == "F m -3 m"
