# tests/test_analytic_backend_integration.py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import ReciprocalConfig


def test_reciprocal_config_backend_defaults():
    cfg = ReciprocalConfig.from_dict(None)
    assert cfg.backend == "auto"
    assert cfg.beamstop is True  # matches generate_kernel default
    assert cfg.zeta_v_fwhm == pytest.approx(5.3e-4)
    assert cfg.eps_rms == pytest.approx(1.41e-4 / 2.35)


def test_reciprocal_config_parses_backend_and_beamstop():
    cfg = ReciprocalConfig.from_dict(
        {
            "hkl": [-1, 1, -1],
            "keV": 17.0,
            "backend": "analytic",
            "beamstop": False,
            "zeta_h_fwhm": 5.3e-4,
        }
    )
    assert cfg.backend == "analytic"
    assert cfg.beamstop is False
    assert cfg.zeta_h_fwhm == pytest.approx(5.3e-4)


def test_reciprocal_config_rejects_bad_backend():
    with pytest.raises(ValueError, match="backend"):
        ReciprocalConfig.from_dict({"backend": "nonsense"})


def test_forward_uses_analytic_when_registered():
    # Load any kernel for geometry/Hg/q_hkl, then register the analytic eval.
    from dfxm_geo.pipeline import _lookup_and_load_kernel

    cfg = ReciprocalConfig.from_dict(None)
    _lookup_and_load_kernel(cfg.hkl, cfg.keV)  # sets Hg, q_hkl, geometry
    try:
        fm._load_analytic_resolution(cfg)
        assert fm._analytic_eval is not None
        img = fm.forward(fm.Hg, phi=0.0, chi=0.0)
        assert img.shape == (fm.NN2 // fm.Nsub, fm.NN1 // fm.Nsub)
        assert np.all(np.isfinite(img))
        assert img.sum() > 0
    finally:
        fm._analytic_eval = None  # restore LUT path
