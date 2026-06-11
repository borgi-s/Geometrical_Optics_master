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


# ---------------------------------------------------------------------------
# Task 9: TOML serializers emit cell parameters for non-cubic mounts
# ---------------------------------------------------------------------------


def _make_hex_mount():
    """Hexagonal Ti mount for serializer tests (Ti: a=3.2094 Å, c=5.2108 Å)."""
    from dfxm_geo.crystal.oblique import CrystalMount

    return CrystalMount(
        lattice="hexagonal",
        a=3.2094e-10,
        c=5.2108e-10,
        mount_x=(2, -1, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )


def _make_cubic_mount():
    """Standard cubic Al mount."""
    from dfxm_geo.crystal.oblique import CrystalMount

    return CrystalMount(
        lattice="cubic",
        a=4.0495e-10,
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )


def _make_oblique_sim_config(mount):
    """Build a SimulationConfig directly (bypassing the Task 8 Stage 4.3 guard)
    wrapping the given mount (any crystal system) in oblique geometry."""
    from dfxm_geo.config import GeometryConfig
    from dfxm_geo.pipeline import SimulationConfig

    geo = GeometryConfig(
        mode="oblique",
        eta=0.0,
        theta_validated=0.1,
        omega=0.0,
        mount=mount,
    )
    return SimulationConfig(geometry=geo)


def test_toml_serializer_emits_cell_params_for_noncubic():
    """_dataclass_to_toml_str must emit b/c/alpha_deg/beta_deg/gamma_deg for hexagonal mounts."""
    from dfxm_geo.pipeline import _dataclass_to_toml_str

    mount = _make_hex_mount()
    cfg = _make_oblique_sim_config(mount)
    toml_str = _dataclass_to_toml_str(cfg)

    assert 'lattice = "hexagonal"' in toml_str
    # c is free for hexagonal; b == a (constrained), so only c is strictly
    # non-trivial — but ALL five lines must be present.
    assert f"c = {mount.cell.c}" in toml_str
    assert f"gamma_deg = {mount.cell.gamma_deg}" in toml_str
    assert f"alpha_deg = {mount.cell.alpha_deg}" in toml_str
    assert f"beta_deg = {mount.cell.beta_deg}" in toml_str
    assert f"b = {mount.cell.b}" in toml_str


def test_toml_serializer_cubic_output_unchanged():
    """Cubic configs must NOT gain new cell-parameter lines (provenance stability)."""
    from dfxm_geo.pipeline import _dataclass_to_toml_str

    mount = _make_cubic_mount()
    cfg = _make_oblique_sim_config(mount)
    toml_str = _dataclass_to_toml_str(cfg)

    assert "alpha_deg" not in toml_str
    assert "beta_deg" not in toml_str
    assert "gamma_deg" not in toml_str
    # No scalar cell 'b ='/'c =' lines may appear in the [crystal] mount section
    # (the text between "[crystal]" and the "[crystal.<mode>]" sub-table). The
    # 'b = [...]' Burgers vector inside [crystal.centered] is unrelated and out
    # of scope of this scan.
    mount_section = toml_str.split("[crystal]", 1)[1].split("[crystal.", 1)[0]
    for line in mount_section.splitlines():
        stripped = line.strip()
        assert not stripped.startswith(("b =", "c =")), (
            f"Unexpected cell-parameter line in cubic [crystal] mount section: {line!r}"
        )


def test_toml_serializer_noncubic_round_trips_cell():
    """The serialized non-cubic [crystal] block round-trips through the Task 7 TOML parser.

    We can't go all the way through SimulationConfig.from_toml (Stage 4.3 guard
    blocks non-cubic oblique), but we can verify that the emitted crystal block
    text re-parses correctly using the low-level _crystal_mount_from_toml helper.
    """
    import tomllib

    from dfxm_geo.pipeline import _dataclass_to_toml_str
    from dfxm_geo.reciprocal_space.kernel import _crystal_mount_from_toml

    mount = _make_hex_mount()
    cfg = _make_oblique_sim_config(mount)
    toml_str = _dataclass_to_toml_str(cfg)

    # Parse the full serialized TOML to get the [crystal] section.
    parsed = tomllib.loads(toml_str)
    assert "crystal" in parsed

    # Re-build the mount from the crystal section.
    reparsed_mount = _crystal_mount_from_toml(parsed["crystal"])

    assert reparsed_mount.lattice == mount.lattice
    assert reparsed_mount.cell.a == pytest.approx(mount.cell.a, rel=1e-12)
    assert reparsed_mount.cell.b == pytest.approx(mount.cell.b, rel=1e-12)
    assert reparsed_mount.cell.c == pytest.approx(mount.cell.c, rel=1e-12)
    assert reparsed_mount.cell.alpha_deg == pytest.approx(mount.cell.alpha_deg, rel=1e-12)
    assert reparsed_mount.cell.beta_deg == pytest.approx(mount.cell.beta_deg, rel=1e-12)
    assert reparsed_mount.cell.gamma_deg == pytest.approx(mount.cell.gamma_deg, rel=1e-12)


def test_identification_toml_serializer_emits_cell_params_for_noncubic():
    """_identification_config_to_toml_str emits cell params for non-cubic mounts."""
    from dfxm_geo.config import (
        GeometryConfig,
        IdentificationConfig,
        _identification_config_to_toml_str,
    )

    mount = _make_hex_mount()
    geo = GeometryConfig(
        mode="oblique",
        eta=0.0,
        theta_validated=0.1,
        omega=0.0,
        mount=mount,
    )
    cfg = IdentificationConfig(geometry=geo)
    toml_str = _identification_config_to_toml_str(cfg)

    assert 'lattice = "hexagonal"' in toml_str
    assert f"b = {mount.cell.b}" in toml_str
    assert f"c = {mount.cell.c}" in toml_str
    assert f"alpha_deg = {mount.cell.alpha_deg}" in toml_str
    assert f"beta_deg = {mount.cell.beta_deg}" in toml_str
    assert f"gamma_deg = {mount.cell.gamma_deg}" in toml_str


def test_identification_toml_serializer_noncubic_round_trips_cell():
    """The identification serializer's [crystal] block re-parses through the Task 7 parser.

    The identification TOML render is documented best-effort, but the crystal
    mount block specifically must round-trip (oblique identify provenance).
    """
    import tomllib

    from dfxm_geo.config import (
        GeometryConfig,
        IdentificationConfig,
        _identification_config_to_toml_str,
    )
    from dfxm_geo.reciprocal_space.kernel import _crystal_mount_from_toml

    mount = _make_hex_mount()
    geo = GeometryConfig(
        mode="oblique",
        eta=0.0,
        theta_validated=0.1,
        omega=0.0,
        mount=mount,
    )
    cfg = IdentificationConfig(geometry=geo)
    toml_str = _identification_config_to_toml_str(cfg)

    parsed = tomllib.loads(toml_str)
    assert "crystal" in parsed
    reparsed_mount = _crystal_mount_from_toml(parsed["crystal"])

    assert reparsed_mount.lattice == mount.lattice
    assert reparsed_mount.cell.a == pytest.approx(mount.cell.a, rel=1e-12)
    assert reparsed_mount.cell.b == pytest.approx(mount.cell.b, rel=1e-12)
    assert reparsed_mount.cell.c == pytest.approx(mount.cell.c, rel=1e-12)
    assert reparsed_mount.cell.alpha_deg == pytest.approx(mount.cell.alpha_deg, rel=1e-12)
    assert reparsed_mount.cell.beta_deg == pytest.approx(mount.cell.beta_deg, rel=1e-12)
    assert reparsed_mount.cell.gamma_deg == pytest.approx(mount.cell.gamma_deg, rel=1e-12)


def test_identification_toml_serializer_cubic_output_unchanged():
    """Cubic identification configs must NOT gain cell-parameter lines."""
    from dfxm_geo.config import (
        GeometryConfig,
        IdentificationConfig,
        _identification_config_to_toml_str,
    )

    mount = _make_cubic_mount()
    geo = GeometryConfig(
        mode="oblique",
        eta=0.0,
        theta_validated=0.1,
        omega=0.0,
        mount=mount,
    )
    cfg = IdentificationConfig(geometry=geo)
    toml_str = _identification_config_to_toml_str(cfg)

    assert "alpha_deg" not in toml_str
    assert "beta_deg" not in toml_str
    assert "gamma_deg" not in toml_str
