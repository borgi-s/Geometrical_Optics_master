"""Unit tests for the realistic detector model."""

import numpy as np
import pytest

from dfxm_geo.detector import (
    FULL_WELL,
    PCO_EDGE_4P2_ID03,
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
