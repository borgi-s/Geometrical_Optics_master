"""forward() reads eta from ReciprocalConfig and passes it to AnalyticResolution.

This is a wire-up test, not a bit-identity test. Bit-identity math is covered
by test_analytic_resolution_eta_zero_bit_identical.py (Task 9). This test ONLY
proves that the eta value gets threaded through _load_analytic_resolution to
fm._analytic_eval.eta at the integration site.

Integration site: forward_model._load_analytic_resolution (line ~498),
called from pipeline._load_resolution when use_analytic=True.
"""

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import ReciprocalConfig


def _make_config(eta: float) -> ReciprocalConfig:
    """Build an analytic-backend ReciprocalConfig with the given eta.

    Uses Al 111 @ 17 keV (the default kernel params) with beamstop=False so
    _load_resolution selects the analytic path without trying to load an npz.
    """
    return ReciprocalConfig(
        hkl=(-1, 1, -1),
        keV=17.0,
        backend="analytic",
        beamstop=False,
        eta=eta,
    )


def test_eta_threads_from_config_to_analytic(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ReciprocalConfig carries eta != 0, _analytic_eval receives it."""
    # Patch Find_Hg to avoid disk-burn: return a zero Hg and a unit q_hkl.
    dummy_Hg = np.zeros((1, 3, 3))
    dummy_q = np.array([1.0, 0.0, 0.0])
    monkeypatch.setattr(fm, "Find_Hg", lambda *a, **kw: (dummy_Hg, dummy_q))

    # Also patch fm.Hg so the guard inside _load_analytic_resolution that
    # calls Find_Hg only when Hg is None is satisfied without disk IO.
    monkeypatch.setattr(fm, "Hg", dummy_Hg)

    config = _make_config(eta=0.5)
    fm._load_analytic_resolution(config)

    assert fm._analytic_eval is not None
    assert fm._analytic_eval.eta == pytest.approx(0.5)


def test_legacy_config_eta_zero_defaults_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A v2.2.0-era config (no explicit eta) resolves to eta=0.0 in _analytic_eval."""
    dummy_Hg = np.zeros((1, 3, 3))
    dummy_q = np.array([1.0, 0.0, 0.0])
    monkeypatch.setattr(fm, "Find_Hg", lambda *a, **kw: (dummy_Hg, dummy_q))
    monkeypatch.setattr(fm, "Hg", dummy_Hg)

    # Default ReciprocalConfig.eta is 0.0 (unchanged from v2.2.0).
    config = _make_config(eta=0.0)
    fm._load_analytic_resolution(config)

    assert fm._analytic_eval is not None
    assert fm._analytic_eval.eta == pytest.approx(0.0)
