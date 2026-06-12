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
