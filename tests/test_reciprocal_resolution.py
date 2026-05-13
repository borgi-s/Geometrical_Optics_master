"""Regression and beamstop-physics tests for the reciprocal-space resolution kernel.

The Monte Carlo Nrays is kept tiny here (1e3-1e4) so the suite runs in seconds.
Statistical sanity checks use loose tolerances; structural checks (output
shape, masking direction, kwargs plumbed through) are tight.
"""

from __future__ import annotations

import numpy as np
import pytest

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


def test_dphi_range_zero_matches_baseline():
    """dphi_range=0 must reproduce the no-beamstop baseline exactly."""
    rng = np.random.default_rng(20260513)
    result = _call(rng=rng, return_qs=True, dphi_range=0.0)
    assert result is not None
    qrock, *_ = result
    rng_baseline = np.random.default_rng(20260513)
    baseline = _call(rng=rng_baseline, return_qs=True)
    assert baseline is not None
    qrock_baseline, *_ = baseline
    np.testing.assert_array_equal(qrock, qrock_baseline)


def test_dphi_range_positive_broadens_qrock():
    """Positive dphi_range adds a uniform offset, broadening qrock std."""
    rng1 = np.random.default_rng(7)
    out_narrow = _call(rng=rng1, return_qs=True, dphi_range=0.0)
    rng2 = np.random.default_rng(7)
    out_wide = _call(rng=rng2, return_qs=True, dphi_range=1e-3)
    assert out_narrow is not None and out_wide is not None
    qrock_narrow = out_narrow[0]
    qrock_wide = out_wide[0]
    # Adding U(-5e-4, 5e-4) adds variance (1e-3)^2/12 ~= 8.3e-8 to qrock.
    # Narrow std is dominated by zeta_v and delta_2theta (both ~few e-5 rad)
    # so the relative widening should be substantial.
    assert qrock_wide.std() > qrock_narrow.std() * 1.5


def test_aperture_beamstop_drops_rays_in_corners():
    """Square aperture absorbs rays whose |alpha_x|>bs/2 OR |alpha_y|>bs/2."""
    rng = np.random.default_rng(99)
    open_count = _call(rng=rng, save_resqi=False, return_qs=True)
    rng = np.random.default_rng(99)
    masked = _call(
        rng=rng,
        return_qs=True,
        beamstop=True,
        aperture=True,
        knife_edge=False,
        bs_height=25e-3,
    )
    assert open_count is not None and masked is not None
    # Masked output should have strictly fewer rays than unmasked.
    assert masked[0].size < open_count[0].size


def test_aperture_beamstop_requires_bs_height():
    """beamstop=True, aperture=True without bs_height should raise."""
    rng = np.random.default_rng(0)
    with pytest.raises((TypeError, ValueError)):
        _call(
            rng=rng,
            beamstop=True,
            aperture=True,
            knife_edge=False,
            bs_height=None,
        )


def test_knife_edge_beamstop_drops_rays_below_edge():
    """Knife-edge masks rays whose BFP x is below the edge position."""
    from dfxm_geo.reciprocal_space.resolution import _bfp_alpha_to_x

    rng = np.random.default_rng(11)
    open_count = _call(rng=rng, return_qs=True)
    rng = np.random.default_rng(11)
    masked = _call(
        rng=rng,
        return_qs=True,
        beamstop=True,
        aperture=False,
        knife_edge=True,
        bs_height=25e-3,
    )
    assert open_count is not None and masked is not None
    # Knife-edge removes ~half the rays on average.
    assert masked[0].size < open_count[0].size
    # Surviving rays should have BFP x of delta_2theta/2 at or above edge_pos.
    delta_2theta_passed = masked[5]  # index 5 = delta_2theta
    bfp_x = _bfp_alpha_to_x(delta_2theta_passed / 2)
    assert (bfp_x >= 25e-3 / 2 - 1e-12).all()


def test_wire_beamstop_drops_rays_through_wire():
    """Wire mode uses xraylib for Tungsten absorption; some rays must absorb."""
    pytest.importorskip("xraylib")
    rng = np.random.default_rng(13)
    open_count = _call(rng=rng, return_qs=True)
    rng = np.random.default_rng(13)
    masked = _call(
        rng=rng,
        return_qs=True,
        beamstop=True,
        aperture=False,
        knife_edge=False,
        bs_height=25e-3,
    )
    assert open_count is not None and masked is not None
    # Some absorption must happen.
    assert masked[0].size < open_count[0].size


def test_wire_beamstop_without_xraylib_raises_clear_error(monkeypatch):
    """If xraylib is not installed, wire mode raises a clear RuntimeError."""
    import sys

    # Simulate the import failing regardless of whether xraylib is installed.
    monkeypatch.setitem(sys.modules, "xraylib", None)
    rng = np.random.default_rng(0)
    with pytest.raises((RuntimeError, ImportError), match="xraylib"):
        _call(
            rng=rng,
            beamstop=True,
            aperture=False,
            knife_edge=False,
            bs_height=25e-3,
        )
