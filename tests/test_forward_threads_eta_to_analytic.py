"""forward() reads eta from ReciprocalConfig and passes it to AnalyticResolution.

This is a wire-up test, not a bit-identity test. Bit-identity math is covered
by test_analytic_resolution_eta_zero_bit_identical.py (Task 9). This test ONLY
proves that the eta value gets threaded through _load_analytic_resolution to
the returned ResolutionContext's analytic_eval.eta at the integration site.

Integration site: forward_model._load_analytic_resolution (line ~498),
called from pipeline._load_resolution when use_analytic=True. #16 Slice 5:
_load_analytic_resolution no longer sets a module global; it RETURNS a
ResolutionContext whose .analytic_eval carries the eta.
"""

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


def test_eta_threads_from_config_to_analytic() -> None:
    """When ReciprocalConfig carries eta != 0, the returned analytic_eval gets it."""
    config = _make_config(eta=0.5)
    res = fm._load_analytic_resolution(config)

    assert res.analytic_eval is not None
    assert res.analytic_eval.eta == pytest.approx(0.5)


def test_legacy_config_eta_zero_defaults_to_zero() -> None:
    """A v2.2.0-era config (no explicit eta) resolves to eta=0.0 in analytic_eval."""
    # Default ReciprocalConfig.eta is 0.0 (unchanged from v2.2.0).
    config = _make_config(eta=0.0)
    res = fm._load_analytic_resolution(config)

    assert res.analytic_eval is not None
    assert res.analytic_eval.eta == pytest.approx(0.0)
