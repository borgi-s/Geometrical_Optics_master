"""Regression and beamstop-physics tests for the reciprocal-space resolution kernel.

The Monte Carlo Nrays is kept tiny here (1e3-1e4) so the suite runs in seconds.
Statistical sanity checks use loose tolerances; structural checks (output
shape, masking direction, kwargs plumbed through) are tight.
"""

from __future__ import annotations

from pathlib import Path

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
    """Seeded no-beamstop run reproduces the pinned baseline to FP-noise tolerance.

    The golden was generated on Windows; cross-platform FP variance
    (Windows MSVC vs Linux GCC builds of numpy/scipy) introduces
    differences with absolute magnitude bounded by ~1e-19 (one ULP for
    these scales). On values near zero the relative diff balloons —
    observed max relative ~2.97e-12 on 4/10000 elements. Use BOTH atol
    and rtol so the near-zero edge case is covered:
    - atol=1e-18 catches absolute diffs at FP-epsilon scale
    - rtol=1e-11 catches anything larger (still ~3 orders tighter than
      any plausible algorithmic drift)
    """
    rng = np.random.default_rng(20260513)
    result = _call(rng=rng, return_qs=True)
    assert result is not None
    qrock, qroll, qpar, qrock_prime, q2th, delta_2theta = result

    golden = np.load(golden_dir / "reciprocal_baseline.npz")
    kw = {"atol": 1e-18, "rtol": 1e-11}
    np.testing.assert_allclose(qrock, golden["qrock"], **kw)
    np.testing.assert_allclose(qroll, golden["qroll"], **kw)
    np.testing.assert_allclose(qpar, golden["qpar"], **kw)
    np.testing.assert_allclose(qrock_prime, golden["qrock_prime"], **kw)
    np.testing.assert_allclose(q2th, golden["q2th"], **kw)
    np.testing.assert_allclose(delta_2theta, golden["delta_2theta"], **kw)


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
    """Wire mode uses xraylib for Tungsten absorption; some rays must absorb.

    Uses a realistic 0.06 mm wire (CDD_inc's commented default) so the
    stochastic Beer-Lambert ``exp(-mu*thick*rho/10)`` branch is the
    dominant attenuation mechanism, not geometric occlusion at the wire
    boundary.
    """
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
        bs_height=0.06,  # ~realistic Tungsten wire diameter (mm)
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


def test_kernel_defaults_match_cdd_inc_generate_Resq_i_py():
    """generate_kernel defaults reproduce the CDD_inc canonical recipe.

    Reference: origin/CDD_inc:reciprocal_space/generate_Resq_i.py.
    """
    import inspect

    from dfxm_geo.reciprocal_space import kernel

    sig = inspect.signature(kernel.generate_kernel)
    defaults = {p.name: p.default for p in sig.parameters.values()}

    # Scalar params from CDD_inc generate_Resq_i.py
    assert defaults["Nrays"] == int(1e8)
    assert defaults["npoints1"] == 400
    assert defaults["npoints2"] == 200
    assert defaults["npoints3"] == 200
    assert defaults["qi1_range"] == 5e-4
    assert defaults["qi2_range"] == 0.75e-2
    assert defaults["qi3_range"] == 0.75e-2
    assert defaults["zeta_v_fwhm"] == 5.3e-04
    assert defaults["zeta_h_fwhm"] == 0
    # Beamstop / aperture switches
    assert defaults["beamstop"] is True
    assert defaults["aperture"] is True
    assert defaults["knife_edge"] is False
    assert defaults["bs_height"] == 25e-3
    # Theta derived from 17 keV / Al 111, not the hardcoded 17.953/2 deg
    expected_a = 4.0495e-10
    expected_wavelength = 1.239841984e-9 / 17
    expected_d_111 = expected_a / np.sqrt(3)
    expected_theta = np.arcsin(expected_wavelength / (2 * expected_d_111))
    assert defaults["theta"] == pytest.approx(expected_theta, rel=1e-12)
    # Remaining scalar params that complete the CDD_inc canonical recipe.
    assert defaults["NA_rms"] == pytest.approx(7.31e-4 / 2.35, rel=1e-12)
    assert defaults["eps_rms"] == pytest.approx(1.41e-4 / 2.35, rel=1e-12)
    assert defaults["D"] == pytest.approx(2 * np.sqrt(50e-6 * 1.6e-3), rel=1e-12)
    assert defaults["d1"] == 0.274
    assert defaults["dphi_range"] == 0.0


def test_beamstop_aperture_and_knife_edge_both_true_raises():
    """The dispatcher must reject aperture=True AND knife_edge=True as exclusive."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _call(
            rng=rng,
            beamstop=True,
            aperture=True,
            knife_edge=True,
            bs_height=25e-3,
        )


def test_truncnorm_chunked_matches_unchunked():
    """Chunking the truncnorm sampling preserves bit-equality.

    The helper feeds the same Generator into multiple smaller .rvs(size=...) calls.
    Because scipy.stats.truncnorm.rvs internally consumes rng via a single
    rng.uniform(size=...) call, chunked vs single-shot must produce identical
    output for the same seeded rng.
    """
    from dfxm_geo.reciprocal_space.resolution import _chunked_truncnorm_rvs

    rng1 = np.random.default_rng(42)
    out_chunked = _chunked_truncnorm_rvs(
        a=-1.0,
        b=1.0,
        loc=0.0,
        scale=1.0,
        size=10_000,
        random_state=rng1,
        chunk_size=1_000,
    )

    rng2 = np.random.default_rng(42)
    import scipy.stats

    out_single = scipy.stats.truncnorm.rvs(
        -1.0, 1.0, loc=0.0, scale=1.0, size=10_000, random_state=rng2
    )

    np.testing.assert_array_equal(out_chunked, out_single)


class TestNotEnoughSamplesGuards:
    """phys_aper too tight -> raise ValueError, do not call exit().

    Library code must not call exit(). These guards convert the original
    bare exit("...") shutdown into recoverable ValueErrors that the caller
    can catch (or that the test runner can report cleanly).
    """

    def test_raises_value_error_when_xi_filter_drops_too_many_samples(self) -> None:
        # phys_aper of 1e-12 m << NA_rms (~3e-4) means the |x| < phys_aper/2
        # filter keeps ~0 of the 1.01 * Nrays oversampled draws.
        with pytest.raises(ValueError, match="Not enough values for"):
            _call(phys_aper=1e-12)


class TestExplicitOutputPath:
    """The `output_path` kwarg lets callers pin the kernel npz to a specific file
    instead of writing under `pkl_files/Resq_i_<date>.npz` in CWD. Required
    by `dfxm-bootstrap`, which must write to the path stage 0 will read.
    """

    def test_writes_to_explicit_path(self, tmp_path: Path) -> None:
        """When `output_path` is provided, kernel npz goes there (not to CWD/pkl_files)."""
        from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func

        out = tmp_path / "kernel.npz"
        reciprocal_res_func(
            Nrays=1000,
            npoints1=20,
            npoints2=20,
            npoints3=20,
            qi1_range=5e-4,
            qi2_range=7.5e-3,
            qi3_range=7.5e-3,
            plot_figs=False,
            save_resqi=True,
            zeta_v_fwhm=5.3e-4,
            zeta_h_fwhm=0,
            NA_rms=7.31e-4 / 2.35,
            eps_rms=1.41e-4 / 2.35,
            theta=0.1566,
            phys_aper=2e-3 / 0.274,
            date="test",
            rng=np.random.default_rng(42),
            output_path=out,
        )
        assert out.is_file(), "expected kernel npz written to explicit output_path"
        # And the legacy default path was NOT created.
        assert not (tmp_path / "pkl_files").exists(), (
            "explicit output_path must not also create the legacy pkl_files/ dir"
        )

    def test_default_path_unchanged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no `output_path`, falls back to `pkl_files/Resq_i_<date>.npz` in CWD."""
        from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func

        monkeypatch.chdir(tmp_path)
        reciprocal_res_func(
            Nrays=1000,
            npoints1=20,
            npoints2=20,
            npoints3=20,
            qi1_range=5e-4,
            qi2_range=7.5e-3,
            qi3_range=7.5e-3,
            plot_figs=False,
            save_resqi=True,
            zeta_v_fwhm=5.3e-4,
            zeta_h_fwhm=0,
            NA_rms=7.31e-4 / 2.35,
            eps_rms=1.41e-4 / 2.35,
            theta=0.1566,
            phys_aper=2e-3 / 0.274,
            date="legacy",
            rng=np.random.default_rng(42),
        )
        assert (tmp_path / "pkl_files" / "Resq_i_legacy.npz").is_file()
