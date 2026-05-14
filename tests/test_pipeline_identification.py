"""Unit tests for dfxm-identify pipeline shape."""

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    IdentificationConfig,
    IdentificationCrystalConfig,
    IdentificationMonteCarloConfig,
    IdentificationScanConfig,
    IOConfig,
    _run_identification_single,
    load_identification_config,
)


def test_identification_crystal_config_defaults():
    cfg = IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1))
    assert cfg.slip_plane_normal == (1, 1, 1)
    assert cfg.angle_start_deg == 0.0
    assert cfg.angle_stop_deg == 350.0
    assert cfg.angle_step_deg == 10.0
    assert cfg.b_vector_indices is None
    assert cfg.sweep_all_slip_planes is True
    assert cfg.exclude_invisibility is True
    assert cfg.invisibility_threshold_deg == 10.0


def test_identification_scan_config_defaults():
    cfg = IdentificationScanConfig()
    assert cfg.phi_rad == pytest.approx(150e-6)
    assert cfg.poisson_noise is True
    assert cfg.rng_seed == 0
    assert cfg.intensity_scale == 7.0


def test_identification_montecarlo_config_defaults():
    cfg = IdentificationMonteCarloConfig()
    assert cfg.n_samples == 1000
    assert cfg.pos_std_um == 5.0
    assert cfg.n_png_previews == 50


def test_identification_crystal_config_is_frozen():
    from dataclasses import FrozenInstanceError

    cfg = IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1))
    with pytest.raises(FrozenInstanceError):
        cfg.angle_step_deg = 5.0  # type: ignore[misc]


def _make_io_config():
    return IOConfig(
        fn_prefix="/mosa_test_0000_",
        ftype=".npy",
        dislocs_dirname="identify",
        perfect_dirname="ignored",
        include_perfect_crystal=False,
    )


def test_identification_config_mode_single_ok():
    cfg = IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=IdentificationScanConfig(),
        io=_make_io_config(),
    )
    assert cfg.mode == "single"
    assert cfg.multi is None


def test_identification_config_mode_multi_ok():
    cfg = IdentificationConfig(
        mode="multi",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=IdentificationScanConfig(),
        multi=IdentificationMonteCarloConfig(),
        io=_make_io_config(),
    )
    assert cfg.mode == "multi"
    assert cfg.multi is not None


def test_identification_config_mode_multi_requires_multi_block():
    with pytest.raises(ValueError, match="mode='multi'"):
        IdentificationConfig(
            mode="multi",
            crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
            scan=IdentificationScanConfig(),
            multi=None,
            io=_make_io_config(),
        )


def test_identification_config_invalid_slip_plane_raises():
    with pytest.raises(ValueError, match="not one of the four"):
        IdentificationConfig(
            mode="single",
            crystal=IdentificationCrystalConfig(slip_plane_normal=(2, 0, 0)),
            scan=IdentificationScanConfig(),
            io=_make_io_config(),
        )


def test_identification_config_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        IdentificationConfig(
            mode="bogus",  # type: ignore[arg-type]
            crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
            scan=IdentificationScanConfig(),
            io=_make_io_config(),
        )


def test_load_identification_config_single(tmp_path):
    """Round-trip a single-mode TOML file → IdentificationConfig."""
    toml_text = """
mode = "single"

[crystal]
slip_plane_normal = [1, 1, 1]
angle_start_deg = 0.0
angle_stop_deg = 350.0
angle_step_deg = 10.0
sweep_all_slip_planes = true
exclude_invisibility = true
invisibility_threshold_deg = 10.0

[scan]
phi_rad = 1.5e-4
poisson_noise = true
rng_seed = 0
intensity_scale = 7.0

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify"
perfect_dirname = "ignored"
include_perfect_crystal = false
"""
    cfg_path = tmp_path / "identification_single.toml"
    cfg_path.write_text(toml_text)

    cfg = load_identification_config(cfg_path)
    assert cfg.mode == "single"
    assert cfg.crystal.slip_plane_normal == (1, 1, 1)
    assert cfg.scan.phi_rad == pytest.approx(1.5e-4)
    assert cfg.multi is None


def test_load_identification_config_multi(tmp_path):
    """Multi-mode TOML round-trips, including the [multi] block."""
    toml_text = """
mode = "multi"

[crystal]
slip_plane_normal = [1, 1, 1]

[scan]
phi_rad = 1.5e-4
rng_seed = 42

[multi]
n_samples = 100
pos_std_um = 3.0
n_png_previews = 10

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify_multi"
perfect_dirname = "ignored"
include_perfect_crystal = false
"""
    cfg_path = tmp_path / "identification_multi.toml"
    cfg_path.write_text(toml_text)

    cfg = load_identification_config(cfg_path)
    assert cfg.mode == "multi"
    assert cfg.multi is not None
    assert cfg.multi.n_samples == 100
    assert cfg.scan.rng_seed == 42


def test_load_identification_config_missing_mode_raises(tmp_path):
    """A TOML missing the top-level `mode = ...` field raises."""
    cfg_path = tmp_path / "bad.toml"
    cfg_path.write_text("[crystal]\nslip_plane_normal = [1, 1, 1]\n")
    with pytest.raises(ValueError, match="missing top-level 'mode'"):
        load_identification_config(cfg_path)


def _tiny_single_config(tmp_path):
    return IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=(1, 1, 1),
            angle_start_deg=0.0,
            angle_stop_deg=90.0,
            angle_step_deg=90.0,  # only 2 angles: 0 and 90
            b_vector_indices=[0, 1],  # only 2 Burgers vectors
            sweep_all_slip_planes=False,  # just one plane
            exclude_invisibility=False,  # don't filter
        ),
        scan=IdentificationScanConfig(rng_seed=0, intensity_scale=1.0),
        io=_make_io_config(),
    )


def test_run_identification_single_writes_expected_count(tmp_path, monkeypatch):
    """Tiny sweep (1 slip plane × 2 b × 2 angles = 4 images) writes 4 .npy files
    and 4 PNGs, and one manifest.csv with 4 rows.
    """
    expected_image = np.ones((170, 510))
    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: expected_image)

    output_dir = tmp_path / "out"
    cfg = _tiny_single_config(tmp_path)

    result = _run_identification_single(cfg, output_dir)

    npys = sorted((output_dir / "n_1_1_1" / "im_data").glob("*.npy"))
    assert len(npys) == 4
    pngs = sorted((output_dir / "n_1_1_1" / "images").glob("*.png"))
    assert len(pngs) == 4
    manifest = output_dir / "manifest.csv"
    assert manifest.is_file()
    lines = manifest.read_text().strip().splitlines()
    assert len(lines) == 5  # 1 header + 4 rows
    assert result["n_images"] == 4
