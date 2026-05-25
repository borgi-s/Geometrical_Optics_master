# tests/test_analytic_backend_integration.py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import ReciprocalConfig, _load_resolution
from dfxm_geo.reciprocal_space.kernel import _validate_reflection
from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func


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


def test_dispatch_auto_no_beamstop_selects_analytic():
    cfg = ReciprocalConfig.from_dict({"beamstop": False})  # auto + beamstop off
    fm._analytic_eval = None
    _load_resolution(cfg)
    assert fm._analytic_eval is not None
    fm._analytic_eval = None


def test_dispatch_auto_beamstop_selects_mc():
    cfg = ReciprocalConfig.from_dict({"beamstop": True})  # auto + beamstop on
    fm._analytic_eval = None
    _load_resolution(cfg)
    assert fm._analytic_eval is None  # MC path: kernel loaded, no analytic
    assert fm._loaded_kernel_path is not None


def test_dispatch_explicit_analytic_with_beamstop_errors():
    cfg = ReciprocalConfig.from_dict({"backend": "analytic", "beamstop": True})
    with pytest.raises(ValueError, match="beamstop"):
        _load_resolution(cfg)


@pytest.mark.slow
def test_analytic_forward_matches_mc_no_beamstop(tmp_path):
    """Forward parity: the closed-form backend reproduces the MC LUT forward
    image (no beamstop) within the MC shot-noise envelope.

    Kernel sampling note (verified empirically, see commit message): the MC
    LUT must be *well populated* for this comparison to be fair. The MC
    resolution function is a histogram; `Resq_i/Resq_i.max()` divides by the
    single most-populated bin, so an undersampled grid makes the peak (and the
    whole normalization) a shot-noise spike and tanks the agreement. A
    400x200x200 grid has 16M bins -- at any test-feasible Nrays that is ~1
    count/bin (pure noise). We therefore build a *coarse* (100x80x80 = 640k
    bins) grid at Nrays=1e8 -> ~156 counts/bin, ~8% per-bin shot noise. That
    resolves the resolution function adequately along all three qi axes (the
    qrock' core spans ~8 bins) while keeping the LUT smooth.

    Metrics: correlation gates structural agreement; mean-abs-diff is the tight
    quantitative gate. A perfect-1.0 correlation is not attainable here because
    this forward scene is a near-flat bright field plus a thin dark dislocation
    line -- ~98% of pixels sit on the resolution plateau, so Pearson is driven
    by the small varying fraction and floored by residual MC plateau noise. The
    physics that matters (the dislocation structure + plateau level) agrees:
    central row/column profiles of the two images overlay to within the MC
    wiggle. The smoothness *difference* that survives here is exactly what
    Task 8 turns into the COM headline.

    Normalization note: the mad gate compares *mean*-normalized images, not
    peak-normalized. Peak-normalization divides by `img.max()`, a single pixel
    whose value tracks the MC LUT's noisiest (most-populated) bin -- so
    peak-norm mad swings run-to-run (0.02-0.07 across RNG seeds) purely from
    that one spike, while the analytic max/mean is a stable 1.019. Mean-norm is
    the robust, RNG-stable shape comparison: mad ~0.007-0.016 (< 0.03) across
    seeds. Correlation is scale-invariant so the choice doesn't affect it.
    """
    hkl, keV = (-1, 1, -1), 17.0
    theta = _validate_reflection(hkl, keV, 4.0495e-10)
    # No-beamstop MC kernel (NA aperture ON via phys_aper, beamstop OFF) with
    # the SAME instrument params the analytic backend uses. Coarse + high-N so
    # each bin is well populated (see docstring). Written into tmp_path/pkl_files.
    out = tmp_path / "pkl_files" / "Resq_i_h-1_k1_l-1_17keV_test.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    reciprocal_res_func(
        Nrays=int(1e8),
        npoints1=100,
        npoints2=80,
        npoints3=80,
        qi1_range=5e-4,
        qi2_range=0.75e-2,
        qi3_range=0.75e-2,
        plot_figs=0,
        save_resqi=1,
        zeta_v_fwhm=5.3e-4,
        zeta_h_fwhm=5.3e-4,
        NA_rms=7.31e-4 / 2.35,
        eps_rms=1.41e-4 / 2.35,
        theta=theta,
        phys_aper=float(2 * np.sqrt(50e-6 * 1.6e-3)) / 0.274,
        date="test",
        beamstop=False,
        output_path=out,
        kernel_meta={
            "qi1_range": 5e-4,
            "qi2_range": 0.75e-2,
            "qi3_range": 0.75e-2,
            "npoints1": 100,
            "npoints2": 80,
            "npoints3": 80,
            "hkl": hkl,
            "keV": keV,
        },
    )

    try:
        # MC image
        fm._analytic_eval = None
        fm._load_default_kernel(str(out), expected_hkl=hkl, expected_keV=keV)
        img_mc = fm.forward(fm.Hg, phi=0.0, chi=0.0)

        # Analytic image (same geometry / Hg already loaded)
        cfg = ReciprocalConfig.from_dict({"beamstop": False, "zeta_h_fwhm": 5.3e-4})
        fm._load_analytic_resolution(cfg)
        img_an = fm.forward(fm.Hg, phi=0.0, chi=0.0)
    finally:
        fm._analytic_eval = None

    # Structural correlation is scale-invariant; compute on peak-normalized.
    a_pk = img_an / img_an.max()
    m_pk = img_mc / img_mc.max()
    corr = np.corrcoef(a_pk.ravel(), m_pk.ravel())[0, 1]
    # Mean-normalize for the quantitative shape gate (RNG-stable; see docstring).
    a = img_an / img_an.mean()
    m = img_mc / img_mc.mean()
    mad = np.mean(np.abs(a - m))
    # 0.9 (not 0.99): see docstring -- a near-flat-field scene caps achievable
    # Pearson against a finite-N MC reference. mad is the tight physics gate.
    assert corr > 0.9, f"analytic/MC correlation {corr:.4f} too low (structural)"
    assert mad < 0.03, f"analytic/MC mean-norm abs diff {mad:.4f} too large"
