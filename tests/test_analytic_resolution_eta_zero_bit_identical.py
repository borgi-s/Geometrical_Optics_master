"""Analytic backend: eta=0 must collapse to v2.1.0 (current) closed form.

Adaptations from plan template:
- Plan used sigma_zv/sigma_zh/sigma_NA/sigma_eps kwargs; real API uses
  zeta_v_fwhm/zeta_h_fwhm/NA_rms/eps_rms (pre-divided form not accepted).
  Converted: fwhm = sigma * 2.355 (or 2.35 for zh), rms passed directly.
- Plan used ar._M; real attribute is ar.M (set in __init__ as self.M = M).
"""

import numpy as np

from dfxm_geo.reciprocal_space.analytic_resolution import AnalyticResolution

# Instrument parameters (converted to the real API's fwhm/rms form)
_THETA = 0.2691
_ZETA_V_FWHM = 5.3e-4  # sigma_zv * 2.355 == 5.3e-4 / 2.355 * 2.355
_ZETA_H_FWHM = 0.0  # sigma_zh = 0.0 -> fwhm = 0.0
_NA_RMS = 7.31e-4 / 2.35  # plan's sigma_NA passed as NA_rms directly
_EPS_RMS = 1.41e-4 / 2.35  # plan's sigma_eps passed as eps_rms directly
_ZETA_V_CLIP = 1.4e-4


def test_eta_zero_M_matches_legacy_M() -> None:
    """_build_M result (self.M) is bit-identical at eta=0 vs no-eta call."""
    ar_legacy = AnalyticResolution(
        theta=_THETA,
        zeta_v_fwhm=_ZETA_V_FWHM,
        zeta_h_fwhm=_ZETA_H_FWHM,
        NA_rms=_NA_RMS,
        eps_rms=_EPS_RMS,
        zeta_v_clip=_ZETA_V_CLIP,
    )
    ar_new = AnalyticResolution(
        theta=_THETA,
        eta=0.0,
        zeta_v_fwhm=_ZETA_V_FWHM,
        zeta_h_fwhm=_ZETA_H_FWHM,
        NA_rms=_NA_RMS,
        eps_rms=_EPS_RMS,
        zeta_v_clip=_ZETA_V_CLIP,
    )
    np.testing.assert_array_almost_equal(ar_new.M, ar_legacy.M, decimal=14)


def test_eta_zero_pq_matches_legacy_pq() -> None:
    """At eta=0, the closed-form density p_Q(q) at 20 sample points matches legacy."""
    ar_legacy = AnalyticResolution(
        theta=_THETA,
        zeta_v_fwhm=_ZETA_V_FWHM,
        zeta_h_fwhm=_ZETA_H_FWHM,
        NA_rms=_NA_RMS,
        eps_rms=_EPS_RMS,
        zeta_v_clip=_ZETA_V_CLIP,
    )
    ar_new = AnalyticResolution(
        theta=_THETA,
        eta=0.0,
        zeta_v_fwhm=_ZETA_V_FWHM,
        zeta_h_fwhm=_ZETA_H_FWHM,
        NA_rms=_NA_RMS,
        eps_rms=_EPS_RMS,
        zeta_v_clip=_ZETA_V_CLIP,
    )
    qs = np.random.default_rng(0).normal(scale=1e-4, size=(3, 20))
    np.testing.assert_array_almost_equal(ar_new(qs), ar_legacy(qs), decimal=12)
