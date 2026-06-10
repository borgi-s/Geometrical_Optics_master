"""[[reflections]] config-loader wiring: parse, validate, refuse-at-run."""

from __future__ import annotations

from pathlib import Path

import pytest

from dfxm_geo.pipeline import SimulationConfig, load_identification_config


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(body, encoding="utf-8")
    return p


def _multi_toml(extra_reciprocal: str = "", geometry_eta: str = "") -> str:
    """Build a valid multi-reflection TOML.

    [crystal] carries both the dislocation-layout schema (mode=centered +
    [crystal.centered]) required by CrystalConfig.from_dict AND the mount
    keys (lattice/a/mount_x/y/z) required by _crystal_mount_from_toml.
    """
    return f"""
[reciprocal]
keV = 19.1
{extra_reciprocal}

[geometry]
mode = "oblique"
{geometry_eta}

[crystal]
lattice = "cubic"
a       = 4.0493e-10
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
mode    = "centered"

[crystal.centered]
b = [1, -1, 0]
n = [1,  1, -1]
t = [1,  1,  2]

[[reflections]]
hkl = [1, 1, 3]
[[reflections]]
hkl = [-1, -1, 3]
"""


def test_reflections_parse_to_runs(tmp_path):
    cfg = SimulationConfig.from_toml(_write(tmp_path, _multi_toml()))
    assert len(cfg.reflections) == 2
    assert cfg.reflections[0].hkl == (1, 1, 3)
    assert cfg.reflections[0].theta == pytest.approx(cfg.reflections[1].theta)


def test_no_reflections_block_is_empty_list(tmp_path):
    cfg = SimulationConfig.from_toml(_write(tmp_path, "[reciprocal]\nkeV = 17.0\n"))
    assert cfg.reflections == []


def test_reflections_with_reciprocal_hkl_rejected(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        SimulationConfig.from_toml(
            _write(tmp_path, _multi_toml(extra_reciprocal="hkl = [1, 1, 1]"))
        )


def test_reflections_require_oblique(tmp_path):
    body = _multi_toml().replace('mode = "oblique"', 'mode = "simplified"')
    with pytest.raises(ValueError, match="oblique"):
        SimulationConfig.from_toml(_write(tmp_path, body))


def test_reflections_and_auto_mutually_exclusive(tmp_path):
    body = _multi_toml() + "\n[reflections_auto]\neta_target = 0.3531\n"
    with pytest.raises(ValueError, match="reflections_auto"):
        SimulationConfig.from_toml(_write(tmp_path, body))


def test_geometry_eta_optional_with_reflections(tmp_path):
    # No [geometry] eta — per-entry solution-1 defaults apply.
    from dfxm_geo.crystal.oblique import CrystalMount, compute_omega_eta

    cfg = SimulationConfig.from_toml(_write(tmp_path, _multi_toml()))
    mount = CrystalMount(
        lattice="cubic",
        a=4.0493e-10,
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )
    kev = 19.1
    for r in cfg.reflections:
        geom = compute_omega_eta(mount, r.hkl, kev)
        assert r.eta == pytest.approx(geom.eta_1)


def test_geometry_eta_acts_as_default(tmp_path):
    cfg = SimulationConfig.from_toml(_write(tmp_path, _multi_toml(geometry_eta="eta = 0.3531")))
    for r in cfg.reflections:
        assert r.eta == pytest.approx(0.3531, abs=1e-3)


def test_run_simulation_refuses_multi_reflection(tmp_path):
    cfg = SimulationConfig.from_toml(_write(tmp_path, _multi_toml()))
    from dfxm_geo.pipeline import run_simulation

    with pytest.raises(NotImplementedError, match="multi-reflection"):
        run_simulation(cfg, tmp_path)


_IDENTIFY_MULTI_REFLECTION_TOML = """
mode = "single"

[reciprocal]
keV = 19.1

[geometry]
mode = "oblique"

[crystal]
# load_identification_config strips lattice/a/mount_x/y/z before passing to
# IdentificationCrystalConfig, so only mount keys are needed here (no mode sub-block).
lattice = "cubic"
a       = 4.0493e-10
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]

[noise]
poisson_noise = false

[[reflections]]
hkl = [1, 1, 3]
[[reflections]]
hkl = [-1, -1, 3]
"""


def test_identification_loader_parses_reflections(tmp_path):
    cfg = load_identification_config(_write(tmp_path, _IDENTIFY_MULTI_REFLECTION_TOML))
    assert len(cfg.reflections) == 2


def test_run_identification_refuses_multi_reflection(tmp_path):
    cfg = load_identification_config(_write(tmp_path, _IDENTIFY_MULTI_REFLECTION_TOML))
    from dfxm_geo.pipeline import run_identification

    with pytest.raises(NotImplementedError, match="multi-reflection"):
        run_identification(cfg, tmp_path)


def test_toml_round_trip_serializer_omits_placeholder_eta(tmp_path):
    """Multi-reflection configs must not serialize the eta=0.0 placeholder (Plan-2 trap)."""
    from dfxm_geo.pipeline import _dataclass_to_toml_str

    cfg = SimulationConfig.from_toml(_write(tmp_path, _multi_toml()))
    out = _dataclass_to_toml_str(cfg)
    assert "eta = 0.0" not in out
