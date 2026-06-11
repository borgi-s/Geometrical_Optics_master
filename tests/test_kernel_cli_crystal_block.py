"""cli_main parses [crystal] block into a CrystalMount and uses it for Bragg θ."""

import tomllib

import pytest

from dfxm_geo.reciprocal_space.kernel import _crystal_mount_from_toml, cli_main


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


class TestNonCubicCrystalBlock:
    def test_hexagonal_block_parses(self):
        mount = _crystal_mount_from_toml(
            {
                "lattice": "hexagonal",
                "a": 3.2094e-10,
                "c": 5.2108e-10,
                "mount_x": [2, -1, 0],
                "mount_y": [0, 1, 0],
                "mount_z": [0, 0, 1],
            }
        )
        assert mount.lattice == "hexagonal"
        assert mount.cell.c == 5.2108e-10
        assert mount.cell.gamma_deg == 120.0

    def test_triclinic_block_with_nonorthogonal_mount_rejected(self):
        # (1,0,0)/(0,1,0)/(0,0,1) plane normals are NOT Cartesian-orthogonal
        # in this triclinic cell -> CrystalMount must reject the triple.
        with pytest.raises(ValueError, match="orthogonal"):
            _crystal_mount_from_toml(
                {
                    "lattice": "triclinic",
                    "a": 5.1e-10,
                    "b": 6.2e-10,
                    "c": 7.3e-10,
                    "alpha_deg": 81.0,
                    "beta_deg": 98.5,
                    "gamma_deg": 105.2,
                    "mount_x": [1, 0, 0],
                    "mount_y": [0, 1, 0],
                    "mount_z": [0, 0, 1],
                }
            )


def test_bootstrap_noncubic_simplified_rejected(tmp_path, capsys):
    cfg_text = """
[reciprocal]
hkl = [1, 0, 0]
keV = 17.0

[geometry]
mode = "simplified"

[crystal]
lattice = "hexagonal"
a       = 3.2094e-10
c       = 5.2108e-10
mount_x = [2, -1, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
"""
    p = tmp_path / "hex.toml"
    p.write_text(cfg_text, encoding="utf-8")
    rc = cli_main(["--config", str(p)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "non-cubic" in err, f"expected 'non-cubic' in stderr; got: {err!r}"
    assert "oblique" in err, f"expected 'oblique' in stderr; got: {err!r}"
