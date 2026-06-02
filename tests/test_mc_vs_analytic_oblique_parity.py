"""MC LUT vs analytic closed form parity at three (θ, η) points.

Tolerance: per-axis marginal RMS ≤ 5e-3 (on peak-normalized marginals).
Global Pearson correlation ≥ 0.98.

Design choices and tradeoffs
-------------------------------
- **Comparison metric: marginal projections, not full 3D RMS.**
  The plan's original `RMS ≤ 5e-4` on the full 3D grid is unachievable at any
  feasible Nrays.  The core issue: the MC LUT is a histogram; ~85 % of the
  4M grid bins (100×200×200) receive zero counts at 20 M rays (~5 counts/bin
  average), while the analytic Gaussian is smooth and positive everywhere.  The
  resulting RMS is dominated by those empty-vs-small-positive bins and comes out
  ~0.018 regardless of eta — not a physics signal, just shot noise on the
  sparse histogram.  Projecting onto each axis first (sum over the other two)
  averages over ~20 000 bins before normalizing, reducing shot noise by 1/√20k
  ≈ 0.7 % — giving marginal RMS < 2e-3 with 20 M rays.

- **Nrays = int(2e7)** (20 million). The plan specified 2e8 (200 M), which
  would take >20 min per case.  20 M runs ~30 s per case and gives adequate
  statistics for the marginal comparison (observed marginal RMS ≤ 1.5e-3,
  correlation ≥ 0.995).

- **Tolerances**: marginal RMS ≤ 5e-3 (4× the observed max), Pearson ≥ 0.98
  (observed ≥ 0.995).  These are deliberately loose: the test's purpose is to
  catch a *wrong* eta rotation (which would shift or shear the marginals by
  >> 5e-3), not to quantify shot noise.

- **@pytest.mark.slow**: REQUIRED.  3 cases × ~30–60 s = ~2 min total.  The
  default pyproject.toml `addopts` deselects `slow`, so this never runs in
  `pytest -q` or CI (which uses the default suite).

Kwarg adaptations from the plan template
------------------------------------------
- Plan used sigma_zv/sigma_NA/sigma_eps: real AnalyticResolution API uses
  zeta_v_fwhm/NA_rms/eps_rms (fwhm/rms form).  Conversion:
    zeta_v_fwhm = sigma_zv × 2.355 = (5.3e-4/2.355) × 2.355 = 5.3e-4
    NA_rms = sigma_NA = 7.31e-4/2.35 (pass directly)
    eps_rms = sigma_eps = 1.41e-4/2.35 (pass directly)
- phys_aper: plan's 4.0e-4 is too narrow (clips > 50 % of NA draws and raises
  ValueError).  Replaced with the canonical CDD_inc value:
    phys_aper = 2 × sqrt(50e-6 × 1.6e-3) / 0.274 ≈ 6.6e-3
- reciprocal_res_func returns None by default; use save_resqi=True +
  output_path=tmp_path/... then load the .npz to obtain normResq_i.
- AnalyticResolution.__call__ expects qi shape (3, N); the full 3D grid is
  built with np.meshgrid(..., indexing='ij') and evaluated in one vectorized
  call.

Wrong-eta discrimination check (non-vacuousness guard)
--------------------------------------------------------
The spec reviewer empirically demonstrated that at zeta_h=0 (isotropic NA),
qroll and q2th have nearly identical widths (~5.85e-4 vs ~6.91e-4), so a 20
degree eta rotation causes only ~2% width change per axis -- well within the
5e-3 marginal-RMS tolerance.  A buggy MC that silently used eta=0 instead of
eta=20.233 deg would score Pearson r=0.9941 and marginal RMS<=0.0018, both
passing the tolerance tests.  To prevent this class of silent no-op bug, each
non-zero-eta case also runs a **wrong-eta cross-test** using the 2D
cross-covariance <qroll * q2th> in the joint qroll-q2th marginal.

The key physics: eta applies R_x(eta) which rotates q in the (qroll, q2th)
plane.  At eta=0 qroll and q2th are driven by independent instrument
variables (xi vs eps/zeta_v), so their joint distribution has zero
cross-covariance.  At eta != 0 the rotation mixes them, producing a
nonzero <qroll * q2th> ~ -sin(2*eta)/2 * sigma^2.  This cross-covariance
is:
  - analytically zero at eta=0  (< 1e-20 numerically)
  - clearly nonzero at eta=20.233 deg  (~-6.3e-9, both MC and analytic)
  - ratio |cov_correct / cov_zero| > 1e6 -- unambiguous even at eta=10 deg

We assert that the MC LUT cross-covariance has the SAME sign and is within
2x of the analytic cross-covariance.  A buggy MC@eta=0 would have near-zero
cross-covariance and fail this test.  Empirically observed values:
  eta=20.233 deg: MC/analytic ratio ~1.003 (0.3% agreement)
  eta=10 deg:     ratio expected similarly close.
The discrimination check is skipped for eta=0 (cross-covariance is zero
for both correct and wrong cases).
"""

import numpy as np
import pytest

from dfxm_geo.reciprocal_space.analytic_resolution import AnalyticResolution
from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func

# Physical aperture from the published CDD_inc / test_reciprocal_resolution canon.
_PHYS_APER = float(2 * np.sqrt(50e-6 * 1.6e-3) / 0.274)

# Instrument parameters (matching the plan's intent, in the real API's form).
_ZETA_V_FWHM = 5.3e-4
_ZETA_H_FWHM = 0.0
_NA_RMS = 7.31e-4 / 2.35
_EPS_RMS = 1.41e-4 / 2.35
_ZETA_V_CLIP = 1.4e-4

# Reduced from plan's 2e8 — see module docstring for rationale.
_NRAYS = int(2e7)

# Grid shape — 100×200×200; same as plan.
_NP1, _NP2, _NP3 = 100, 200, 200
_QI1_RANGE, _QI2_RANGE, _QI3_RANGE = 5e-4, 7.5e-3, 7.5e-3

# Parity tolerances (see module docstring for rationale).
_MARGINAL_RMS_TOL = 5e-3  # per-axis marginal RMS (peak-normalized)
_CORRELATION_TOL = 0.98  # Pearson r on flattened peak-normalized arrays

PARAMS = [
    # (theta_rad, eta_rad)
    (0.2691, 0.0),  # paper Figure 3B θ at η=0 (legacy regression)
    (0.2691, float(np.deg2rad(20.233))),  # paper Figure 3B full geometry (η=20.233°)
    (0.35, float(np.deg2rad(10.0))),  # independent (θ, η) point
]


def _bin_centers(n: int, full_range: float) -> np.ndarray:
    """Centre of each histogram bin: (-r/2 + (i+0.5)*r/n) for i in 0..n-1."""
    return np.linspace(
        -full_range / 2 + full_range / (2 * n),
        full_range / 2 - full_range / (2 * n),
        n,
    )


@pytest.mark.slow
@pytest.mark.parametrize("theta,eta", PARAMS)
def test_mc_vs_analytic_marginal_rms(theta: float, eta: float, tmp_path) -> None:
    """MC LUT and analytic closed form agree at arbitrary η.

    Comparison method: project each backend's 3D distribution onto each of the
    three coordinate axes (sum over the other two), normalize the resulting 1D
    profile to peak=1, and assert RMS ≤ _MARGINAL_RMS_TOL.  Also assert
    Pearson correlation on the full 3D arrays ≥ _CORRELATION_TOL.

    See the module docstring for the full rationale.
    """
    # ---- Build MC LUT -------------------------------------------------------
    out_path = tmp_path / f"resqi_th{theta:.4f}_eta{eta:.4f}.npz"
    reciprocal_res_func(
        Nrays=_NRAYS,
        npoints1=_NP1,
        npoints2=_NP2,
        npoints3=_NP3,
        qi1_range=_QI1_RANGE,
        qi2_range=_QI2_RANGE,
        qi3_range=_QI3_RANGE,
        plot_figs=False,
        save_resqi=True,
        zeta_v_fwhm=_ZETA_V_FWHM,
        zeta_h_fwhm=_ZETA_H_FWHM,
        NA_rms=_NA_RMS,
        eps_rms=_EPS_RMS,
        theta=theta,
        eta=eta,
        phys_aper=_PHYS_APER,
        date="parity_test",
        beamstop=False,
        bs_height=0.025,
        aperture=False,
        knife_edge=False,
        dphi_range=0.0,
        rng=np.random.default_rng(seed=11),
        output_path=out_path,
    )
    data = np.load(out_path)
    lut: np.ndarray = data["Resq_i"]  # (NP1, NP2, NP3), peak-normalized to 1

    # ---- Build analytic distribution on the same grid -----------------------
    ar = AnalyticResolution(
        theta=theta,
        eta=eta,
        zeta_v_fwhm=_ZETA_V_FWHM,
        zeta_h_fwhm=_ZETA_H_FWHM,
        NA_rms=_NA_RMS,
        eps_rms=_EPS_RMS,
        zeta_v_clip=_ZETA_V_CLIP,
    )

    c1 = _bin_centers(_NP1, _QI1_RANGE)  # qrock_prime axis
    c2 = _bin_centers(_NP2, _QI2_RANGE)  # qroll axis
    c3 = _bin_centers(_NP3, _QI3_RANGE)  # q2th axis

    # Evaluate on the full 3D grid in a single vectorized call.
    g1, g2, g3 = np.meshgrid(c1, c2, c3, indexing="ij")  # each (NP1, NP2, NP3)
    qi_flat = np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=0)  # (3, N)
    analytic: np.ndarray = ar(qi_flat).reshape(_NP1, _NP2, _NP3)  # peak-normalized

    # ---- Pearson correlation (full 3D) --------------------------------------
    corr = float(np.corrcoef(lut.ravel(), analytic.ravel())[0, 1])
    assert corr >= _CORRELATION_TOL, (
        f"MC vs analytic Pearson r = {corr:.4f} < {_CORRELATION_TOL} "
        f"at theta={theta:.4f} rad, eta={eta:.4f} rad"
    )

    # ---- Marginal RMS (per axis) --------------------------------------------
    for ax_idx, ax_name in enumerate(["qrock_prime", "qroll", "q2th"]):
        sum_axes = tuple(i for i in range(3) if i != ax_idx)
        m_lut = lut.sum(axis=sum_axes)
        m_an = analytic.sum(axis=sum_axes)
        m_lut_n = m_lut / m_lut.max()
        m_an_n = m_an / m_an.max()
        rms = float(np.sqrt(np.mean((m_lut_n - m_an_n) ** 2)))
        assert rms <= _MARGINAL_RMS_TOL, (
            f"MC vs analytic marginal-{ax_name} RMS = {rms:.6f} "
            f"exceeds {_MARGINAL_RMS_TOL} "
            f"at theta={theta:.4f} rad, eta={eta:.4f} rad"
        )

    # ---- Wrong-eta discrimination check ------------------------------------
    # Guard against a silent eta no-op bug.  At zeta_h=0 the 1D marginals are
    # nearly insensitive to eta rotations in the (qroll, q2th) plane (Pearson r
    # gap < 0.001, marginal-RMS ratio ~ 1.0 -- verified empirically).  Instead
    # we use the cross-covariance <qroll * q2th> of the 2D joint marginal.
    # See the module docstring "Wrong-eta discrimination check" section for the
    # full physics rationale.  Skipped for eta=0.
    if eta != 0.0:
        ar_wrong = AnalyticResolution(
            theta=theta,
            eta=0.0,  # deliberately wrong eta
            zeta_v_fwhm=_ZETA_V_FWHM,
            zeta_h_fwhm=_ZETA_H_FWHM,
            NA_rms=_NA_RMS,
            eps_rms=_EPS_RMS,
            zeta_v_clip=_ZETA_V_CLIP,
        )
        analytic_wrong: np.ndarray = ar_wrong(qi_flat).reshape(_NP1, _NP2, _NP3)

        # 2D joint marginal in the qroll-q2th plane (sum over qrock_prime=axis 0).
        joint_lut = lut.sum(axis=0)  # (NP2, NP3)
        joint_correct = analytic.sum(axis=0)
        joint_wrong = analytic_wrong.sum(axis=0)

        # Normalize to probability distributions for moment computation.
        joint_lut_p = joint_lut / joint_lut.sum()
        joint_correct_p = joint_correct / joint_correct.sum()
        joint_wrong_p = joint_wrong / joint_wrong.sum()

        # <qroll * q2th> = weighted sum of c2[i] * c3[j] over the joint distribution.
        c2_grid, c3_grid = np.meshgrid(c2, c3, indexing="ij")
        cross_cov_mc = float((joint_lut_p * c2_grid * c3_grid).sum())
        cross_cov_correct = float((joint_correct_p * c2_grid * c3_grid).sum())
        cross_cov_wrong = float((joint_wrong_p * c2_grid * c3_grid).sum())

        print(
            f"\nWrong-eta cross-covariance discriminator at "
            f"theta={theta:.4f} rad, eta={eta:.4f} rad:\n"
            f"  MC <qroll*q2th>           = {cross_cov_mc:.4e}\n"
            f"  Analytic@correct_eta      = {cross_cov_correct:.4e}\n"
            f"  Analytic@eta=0 (wrong)    = {cross_cov_wrong:.4e}\n"
            f"  |MC / analytic_correct|   = {abs(cross_cov_mc / cross_cov_correct):.4f}"
        )

        # At eta != 0, the analytic cross-covariance is clearly nonzero
        # (observed ~-6.3e-9 at eta=20 deg, ~-1.9e-9 at eta=10 deg).
        # Assert the MC has the same sign — wrong-eta MC would be near-zero.
        assert (cross_cov_mc > 0) == (cross_cov_correct > 0), (
            f"Wrong-eta discrimination FAILED: MC <qroll*q2th> = {cross_cov_mc:.4e} "
            f"has opposite sign to analytic@correct_eta = {cross_cov_correct:.4e}.  "
            f"This indicates the eta rotation is not applied correctly in the MC."
        )

        # Assert the MC cross-covariance magnitude is within 2x of the analytic.
        # At eta=0 the ratio would be effectively undefined (analytic ~0) — this
        # is the exact case where a buggy no-op MC would fail the sign check above.
        mc_to_analytic_ratio = abs(cross_cov_mc / cross_cov_correct)
        assert 0.5 < mc_to_analytic_ratio < 2.0, (
            f"MC <qroll*q2th> = {cross_cov_mc:.4e} differs from analytic "
            f"{cross_cov_correct:.4e} by more than 2x (ratio={mc_to_analytic_ratio:.3f}).  "
            f"Expected ratio in (0.5, 2.0) — the test uses 20 M rays which "
            f"gives ~0.3% agreement empirically."
        )
