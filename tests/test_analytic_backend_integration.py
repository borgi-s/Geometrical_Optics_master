# tests/test_analytic_backend_integration.py
import pytest

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
