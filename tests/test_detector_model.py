"""Unit tests for the realistic detector model."""

import numpy as np
import pytest

from dfxm_geo.detector import (
    FULL_WELL,
    PCO_EDGE_4P2_ID03,
    SensorMap,
    resolve_model,
)


def test_preset_matches_measured_calibration():
    m = PCO_EDGE_4P2_ID03
    assert m.name == "pco_edge_4.2_id03"
    assert m.gain == pytest.approx(2.14)
    assert m.offset(0.0) == pytest.approx(102.5)
    assert m.offset(1.0) == pytest.approx(110.0)
    # noise sigma: sqrt(6.3) ~ 2.51 at t=0, sqrt(6.3 + 11*0.5) ~ 3.43 at 0.5 s
    assert m.noise_sigma(0.0) == pytest.approx(2.51, abs=0.01)
    assert m.noise_sigma(0.5) == pytest.approx(3.43, abs=0.01)


def test_resolve_model_registry():
    assert resolve_model("pco_edge_4.2_id03") is PCO_EDGE_4P2_ID03
    assert resolve_model("ideal") is None
    with pytest.raises(ValueError, match="unknown detector model"):
        resolve_model("pco_edge_99")


def test_full_well_is_uint16_max():
    assert np.iinfo(np.uint16).max == FULL_WELL


def test_sensor_map_statistics_match_dark_census():
    rng = np.random.default_rng(7)
    m = PCO_EDGE_4P2_ID03
    sm = m.make_sensor_map((512, 512), rng)
    fpn = sm.fpn_offset
    assert fpn.shape == (512, 512)
    interior = fpn[m.edge_rows : -m.edge_rows, :]
    # Gaussian core sigma ~ fpn_sigma (robust estimate, tail excluded)
    mad = np.median(np.abs(interior - np.median(interior)))
    assert 1.4826 * mad == pytest.approx(m.fpn_sigma, rel=0.2)
    # warm/hot census: ~0.57 % above +10 ADU, ~0.011 % above +50 ADU (spec §2)
    frac10 = (interior > 10.0).mean()
    frac50 = (interior > 50.0).mean()
    assert 0.003 < frac10 < 0.012
    assert 1e-5 < frac50 < 5e-4
    # edge rows elevated relative to interior
    assert fpn[0, :].mean() > interior.mean() + 0.5 * m.edge_peak


def test_sensor_map_is_reproducible_and_seed_sensitive():
    m = PCO_EDGE_4P2_ID03
    a = m.make_sensor_map((64, 64), np.random.default_rng(3)).fpn_offset
    b = m.make_sensor_map((64, 64), np.random.default_rng(3)).fpn_offset
    c = m.make_sensor_map((64, 64), np.random.default_rng(4)).fpn_offset
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_apply_statistics_match_photon_transfer():
    """Mean and variance must reproduce the measured model:
    mean = s + offset(t), var = gain*s + noise_sigma(t)^2 (+ tiny rounding var)."""
    m = PCO_EDGE_4P2_ID03
    rng = np.random.default_rng(11)
    flat = SensorMap(fpn_offset=np.zeros((200, 200)))
    for t, s in [(1.0, 0.0), (1.0, 50.0), (1.0, 1000.0), (0.1, 300.0)]:
        ideal = np.full((10, 200, 200), s)
        out = m.apply(ideal, exposure_time=t, rng=rng, sensor=flat)
        assert out.dtype == np.uint16
        vals = out.astype(np.float64)
        expected_mean = s + m.offset(t)
        expected_var = m.gain * s + m.noise_sigma(t) ** 2 + 1.0 / 12.0
        assert vals.mean() == pytest.approx(expected_mean, rel=0.01)
        assert vals.var() == pytest.approx(expected_var, rel=0.05)


def test_apply_clamps_at_full_well_and_zero():
    m = PCO_EDGE_4P2_ID03
    flat = SensorMap(fpn_offset=np.zeros((4, 4)))
    rng = np.random.default_rng(0)
    hot = m.apply(np.full((1, 4, 4), 1e9), 1.0, rng, flat)
    assert (hot == FULL_WELL).all()
    # negative ideal input (shouldn't happen, but) must not wrap below zero
    cold = m.apply(np.full((1, 4, 4), -1e6), 1.0, rng, flat)
    assert (cold <= 200).all()


def test_apply_is_deterministic_for_fixed_rng():
    m = PCO_EDGE_4P2_ID03
    flat = SensorMap(fpn_offset=np.zeros((8, 8)))
    ideal = np.full((2, 8, 8), 500.0)
    a = m.apply(ideal, 1.0, np.random.default_rng(5), flat)
    b = m.apply(ideal, 1.0, np.random.default_rng(5), flat)
    assert np.array_equal(a, b)


def test_apply_adds_sensor_map_offset():
    m = PCO_EDGE_4P2_ID03
    sm = SensorMap(fpn_offset=np.full((16, 16), 40.0))
    out = m.apply(np.zeros((50, 16, 16)), 1.0, np.random.default_rng(1), sm)
    assert out.astype(float).mean() == pytest.approx(m.offset(1.0) + 40.0, rel=0.02)
