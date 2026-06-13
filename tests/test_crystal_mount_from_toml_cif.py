"""Stage 4.2: [crystal] cif + space_group parsing with per-key TOML override."""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.reciprocal_space.kernel import _crystal_mount_from_toml

DATA = Path(__file__).parent / "data" / "cif"
MOUNTS = {"mount_x": [1, 0, 0], "mount_y": [0, 1, 0], "mount_z": [0, 0, 1]}


def test_cif_populates_cell_and_space_group() -> None:
    m = _crystal_mount_from_toml({"cif": str(DATA / "al_fm3m.cif"), **MOUNTS})
    assert m.lattice == "cubic"
    assert m.a == pytest.approx(4.0495e-10)
    assert m.space_group == "F m -3 m"


def test_explicit_toml_key_overrides_cif() -> None:
    m = _crystal_mount_from_toml({"cif": str(DATA / "al_fm3m.cif"), "a": 4.05e-10, **MOUNTS})
    assert m.a == 4.05e-10  # TOML wins
    assert m.space_group == "F m -3 m"  # untouched keys still come from the CIF


def test_space_group_toml_overrides_cif() -> None:
    m = _crystal_mount_from_toml(
        {"cif": str(DATA / "al_fm3m.cif"), "space_group": "Pm-3m", **MOUNTS}
    )
    assert m.space_group == "P m -3 m"


def test_space_group_without_cif() -> None:
    m = _crystal_mount_from_toml(
        {"lattice": "cubic", "a": 4.0495e-10, "space_group": "Fm-3m", **MOUNTS}
    )
    assert m.space_group == "F m -3 m"


def test_relative_cif_path_resolves_against_base_dir(tmp_path: Path) -> None:
    shutil.copy(DATA / "al_fm3m.cif", tmp_path / "al.cif")
    m = _crystal_mount_from_toml({"cif": "al.cif", **MOUNTS}, base_dir=tmp_path)
    assert m.a == pytest.approx(4.0495e-10)


def test_cif_without_mount_keys_uses_identity_defaults() -> None:
    # M4 Stage 4.3a: mount_x/y/z are now optional; absent → identity (1,0,0)/(0,1,0)/(0,0,1).
    # Stage 4.2 required them explicitly to avoid silently mis-orienting non-trivial crystals,
    # but the CrystalMount defaults are already identity and the task test block omits them.
    m = _crystal_mount_from_toml({"cif": str(DATA / "al_fm3m.cif")})
    assert m.mount_x == (1, 0, 0)
    assert m.mount_y == (0, 1, 0)
    assert m.mount_z == (0, 0, 1)


def test_no_cif_no_lattice_still_errors() -> None:
    with pytest.raises(ValueError, match="missing key: lattice"):
        _crystal_mount_from_toml({"a": 4.0495e-10, **MOUNTS})


def test_hexagonal_cif_end_to_end() -> None:
    # Hexagonal needs an orthogonal plane-normal triple: a*, a*-2b* (.L c*).
    m = _crystal_mount_from_toml(
        {
            "cif": str(DATA / "mg_p63mmc.cif"),
            "mount_x": [1, 0, 0],
            "mount_y": [1, -2, 0],
            "mount_z": [0, 0, 1],
        }
    )
    assert m.lattice == "hexagonal"
    assert m.c == pytest.approx(5.2108e-10)
    assert m.space_group == "P 63/m m c"
