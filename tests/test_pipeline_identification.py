"""Unit tests for dfxm-identify pipeline shape."""

import pytest

from dfxm_geo.pipeline import (
    IdentificationCrystalConfig,
    IdentificationMonteCarloConfig,
    IdentificationScanConfig,
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
