"""Regression and beamstop-physics tests for the reciprocal-space resolution kernel.

The Monte Carlo Nrays is kept tiny here (1e3-1e4) so the suite runs in seconds.
Statistical sanity checks use loose tolerances; structural checks (output
shape, masking direction, kwargs plumbed through) are tight.
"""

from __future__ import annotations

import numpy as np

from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func

# Canonical test parameters - CDD_inc-shaped but with tiny Nrays.
NRAYS = 10_000
NPOINTS = (40, 30, 30)
QI_RANGES = (5e-4, 7.5e-3, 7.5e-3)
ZETA_V_FWHM = 5.3e-04
ZETA_H_FWHM = 0.0
NA_RMS = 7.31e-4 / 2.35
EPS_RMS = 1.41e-4 / 2.35
THETA = 0.15662  # ~17 keV / Al 111
D = 2 * np.sqrt(50e-6 * 1.6e-3)
D1 = 0.274
PHYS_APER = D / D1


def _call(rng=None, **overrides):
    """Helper: run reciprocal_res_func with canonical params + overrides."""
    kwargs = dict(
        Nrays=NRAYS,
        npoints1=NPOINTS[0],
        npoints2=NPOINTS[1],
        npoints3=NPOINTS[2],
        qi1_range=QI_RANGES[0],
        qi2_range=QI_RANGES[1],
        qi3_range=QI_RANGES[2],
        plot_figs=False,
        save_resqi=False,
        zeta_v_fwhm=ZETA_V_FWHM,
        zeta_h_fwhm=ZETA_H_FWHM,
        NA_rms=NA_RMS,
        eps_rms=EPS_RMS,
        theta=THETA,
        phys_aper=PHYS_APER,
        date="golden",
        rng=rng,
    )
    kwargs.update(overrides)
    return reciprocal_res_func(**kwargs)


def test_seeded_rng_makes_output_reproducible():
    """Same seed -> identical output. Confirms rng kwarg is plumbed through."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    out1 = _call(rng=rng1, return_qs=True)
    out2 = _call(rng=rng2, return_qs=True)
    qrock1, _, _, _, _, _ = out1
    qrock2, _, _, _, _, _ = out2
    np.testing.assert_array_equal(qrock1, qrock2)


def test_no_beamstop_baseline_matches_golden(golden_dir):
    """Seeded no-beamstop run reproduces the pinned baseline to bit-equality."""
    rng = np.random.default_rng(20260513)
    result = _call(rng=rng, return_qs=True)
    assert result is not None
    qrock, qroll, qpar, qrock_prime, q2th, delta_2theta = result

    golden = np.load(golden_dir / "reciprocal_baseline.npz")
    np.testing.assert_array_equal(qrock, golden["qrock"])
    np.testing.assert_array_equal(qroll, golden["qroll"])
    np.testing.assert_array_equal(qpar, golden["qpar"])
    np.testing.assert_array_equal(qrock_prime, golden["qrock_prime"])
    np.testing.assert_array_equal(q2th, golden["q2th"])
    np.testing.assert_array_equal(delta_2theta, golden["delta_2theta"])
