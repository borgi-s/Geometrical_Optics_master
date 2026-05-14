"""Unit tests for dfxm-identify pipeline shape."""

import pytest

from dfxm_geo.pipeline import (
    IdentificationConfig,
    IdentificationCrystalConfig,
    IdentificationMonteCarloConfig,
    IdentificationScanConfig,
    IOConfig,
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
