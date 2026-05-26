# tests/test_analytic_backend_integration.py
import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import dfxm_geo.direct_space.forward_model as fm  # noqa: E402
from dfxm_geo.pipeline import (  # noqa: E402
    ReciprocalConfig,
    _load_resolution,
    _lookup_and_load_kernel,
)
from dfxm_geo.reciprocal_space.kernel import _validate_reflection  # noqa: E402
from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func  # noqa: E402


def _require_kernel() -> None:
    """Skip unless a bootstrapped (-1,1,-1) 17 keV kernel npz is on disk.

    Tests that load the MC kernel (for geometry or the MC dispatch path) cannot
    run on a bare checkout (CI), where no kernel has been bootstrapped.
    """
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip(f"no kernel npz found in {kernel_dir}")


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
    _require_kernel()
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
    _require_kernel()
    cfg = ReciprocalConfig.from_dict({"beamstop": True})  # auto + beamstop on
    fm._analytic_eval = None
    _load_resolution(cfg)
    assert fm._analytic_eval is None  # MC path: kernel loaded, no analytic
    assert fm._loaded_kernel_path is not None


def test_dispatch_explicit_analytic_with_beamstop_errors():
    cfg = ReciprocalConfig.from_dict({"backend": "analytic", "beamstop": True})
    with pytest.raises(ValueError, match="beamstop"):
        _load_resolution(cfg)


def test_run_simulation_analytic_writes_hdf5_without_kernel(tmp_path):
    """Regression: a full run_simulation with the analytic backend and NO MC
    kernel loaded must complete and write a valid HDF5 master.

    The HDF5 provenance writer used to hard-require a loaded kernel
    (`_fm._loaded_kernel_path`), so the entire analytic CLI path crashed at
    write time. The earlier unit tests masked it by pre-loading a kernel for
    geometry; here we force the genuine analytic-only state.
    """
    import h5py

    from dfxm_geo.pipeline import SimulationConfig, run_simulation

    # Force the clean / analytic-only state: no MC kernel anywhere.
    fm._loaded_kernel_path = None
    fm._analytic_eval = None

    toml = tmp_path / "analytic.toml"
    toml.write_text('[reciprocal]\nbackend = "analytic"\nbeamstop = false\n')
    cfg = SimulationConfig.from_toml(toml)
    cfg.io.include_perfect_crystal = False  # speed: single scan only

    try:
        out = run_simulation(cfg, tmp_path)
        assert isinstance(out, dict)
        master_h5 = tmp_path / "dfxm_geo.h5"
        assert master_h5.exists()
        with h5py.File(master_h5, "r") as f:
            assert "/dfxm_geo" in f
            # Analytic runs have no MC kernel, so no kernel provenance sub-group.
            assert "/dfxm_geo/kernel" not in f
            # The backend choice is captured in the embedded config TOML.
            assert "analytic" in f["/dfxm_geo/config_toml"][()].decode()
    finally:
        # Don't leak the analytic evaluator into later (MC) tests -- forward()
        # prefers it over the LUT whenever it is set.
        fm._analytic_eval = None


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
    line -- most pixels sit near the bright-field resolution plateau, so Pearson
    is driven by the small varying fraction and floored by residual MC plateau
    noise. The physics that matters (the dislocation structure + plateau level)
    agrees: central row/column profiles of the two images overlay to within the
    MC wiggle. The smoothness *difference* that survives here is exactly what
    Task 8 turns into the COM headline.

    Determinism note: the MC kernel build is seeded (rng=default_rng(0)), so the
    LUT -- and therefore both metrics -- are reproducible run-to-run. With the
    seeded, well-populated kernel the observed values are corr ~0.985 and
    mean-norm mad ~0.007, so the gates below are tight rather than defensive.

    Normalization note: the mad gate compares *mean*-normalized images, not
    peak-normalized. Peak-normalization divides by `img.max()`, a single pixel
    whose value tracks the MC LUT's noisiest (most-populated) bin -- a brittle
    spike -- while the analytic max/mean is a stable 1.019. Mean-norm is the
    robust shape comparison: mad ~0.007. Correlation is scale-invariant so the
    choice doesn't affect it.
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
        rng=np.random.default_rng(0),
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
    # Tight gates: the seeded, well-populated MC kernel makes this deterministic
    # (observed corr ~0.985, mad ~0.007). Not 1.0 because a near-flat-field
    # scene caps achievable Pearson against a finite-N MC reference; mad is the
    # tight physics gate.
    assert corr > 0.97, f"analytic/MC correlation {corr:.4f} too low (structural)"
    assert mad < 0.02, f"analytic/MC mean-norm abs diff {mad:.4f} too large"


def _com_value_hist_modes(com: np.ndarray, *, bins: int = 400, prom_frac: float = 0.01):
    """Number of prominent modes in the histogram of COM values.

    Banding (the MC LUT quantizing COM onto discrete grid sub-levels) shows up
    as a "grassy" multi-peaked COM-value histogram. A smooth (analytic) COM map
    has only the few physical lobes from the strain field. Returns (n_modes,
    hist, bin_centers) so the caller can both assert and plot.
    """
    from scipy.signal import find_peaks

    finite = com[np.isfinite(com)]
    lo, hi = np.percentile(finite, [1, 99])
    counts, edges = np.histogram(finite, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    peaks, _ = find_peaks(counts.astype(float), prominence=prom_frac * counts.max())
    return len(peaks), counts, centers


@pytest.mark.slow
def test_analytic_com_has_no_grid_banding(tmp_path):
    """A phi rocking-scan COM map from the analytic backend should be smooth,
    not quantized onto LUT grid sub-levels like the MC backend.

    Metric note (replaces the original distinct-float-levels heuristic, which
    was not discriminating): COM is a ratio of intensity stacks, and that
    division *dequantizes* the per-pixel float values -- so both backends yield
    ~3e4 distinct rounded COM levels and `uniq_an > 5*uniq_mc` is meaningless
    here (verified: 34477 vs 33062). The physically real signature of MC
    banding is instead a *multi-peaked ("grassy") histogram of COM values*: the
    LUT quantization piles COM onto discrete sub-levels, adding many spurious
    secondary modes. The analytic COM histogram has only the few genuine lobes
    from the dislocation strain field. We count prominent histogram modes with
    scipy.find_peaks; this is threshold-stable (MC 4-17 modes vs analytic 2
    across bins in {200,400,800} x prominence in {0.01,0.02,0.05}) and matches
    the saved figure. The comparison is deterministic: the MC kernel is the
    fixed on-disk default and the analytic backend has no RNG.
    """
    cfg = ReciprocalConfig.from_dict({"beamstop": False, "zeta_h_fwhm": 5.3e-4})
    # Geometry + Hg from the on-disk default kernel lookup, then analytic eval.
    _lookup_and_load_kernel(cfg.hkl, cfg.keV)
    Hg = fm.Hg

    phis = np.linspace(-3e-4, 3e-4, 21)  # small rocking scan (radians)

    def com_map(eval_setup):
        eval_setup()
        stack = np.stack([fm.forward(Hg, phi=p, chi=0.0) for p in phis], axis=0)
        weight = stack.sum(axis=0) + 1e-30
        return (stack * phis[:, None, None]).sum(axis=0) / weight

    com_an = com_map(lambda: fm._load_analytic_resolution(cfg))
    fm._analytic_eval = None
    # MC reference (uses the on-disk default kernel; quantized).
    com_mc = com_map(lambda: _lookup_and_load_kernel(cfg.hkl, cfg.keV))

    modes_an, hist_an, centers_an = _com_value_hist_modes(com_an)
    modes_mc, hist_mc, centers_mc = _com_value_hist_modes(com_mc)
    # The analytic COM histogram is clean (just the physical lobes); the MC one
    # is banded (many spurious modes). Both gates together encode "smoother".
    assert modes_an <= 3, (
        f"analytic COM histogram unexpectedly multi-modal: {modes_an} modes "
        "(expected ~2 physical lobes)"
    )
    assert modes_mc >= 2 * modes_an, (
        f"MC COM not more banded than analytic: {modes_mc} vs {modes_an} modes"
    )

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].imshow(com_mc)
    ax[0, 0].set_title(f"MC COM map ({modes_mc} hist modes)")
    ax[0, 1].imshow(com_an)
    ax[0, 1].set_title(f"analytic COM map ({modes_an} hist modes)")
    ax[1, 0].plot(centers_mc, hist_mc)
    ax[1, 0].set_title("MC COM value histogram (grassy = banded)")
    ax[1, 1].plot(centers_an, hist_an)
    ax[1, 1].set_title("analytic COM value histogram (clean)")
    fig.tight_layout()
    # CWD-independent: resolve the tracked figure path from this file's location.
    fig_path = Path(__file__).resolve().parent.parent / "docs" / "img" / "analytic_vs_mc_com.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)
