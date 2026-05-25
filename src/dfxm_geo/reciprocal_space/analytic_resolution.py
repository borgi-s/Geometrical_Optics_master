# src/dfxm_geo/reciprocal_space/analytic_resolution.py
"""Grid-free closed-form reciprocal-space resolution function.

Replaces the Monte-Carlo Resq_i histogram for no-beamstop configs. The 5
instrument variables [eps, zeta_v, zeta_h, delta_2theta, xi] map linearly to
imaging-space q = M @ x. zeta_v is hard-truncated at +-zeta_v_clip (the
condenser physical aperture, Carlsen 2022); the other 4 are Gaussian. The
resolution density p_Q(q) is the marginal over the truncated zeta_v of a
conditional 3D Gaussian, which is closed-form (an erf difference).

Math verified 2026-05-24 (quadrature 2.3e-13, MC <0.1% cov, norm 1 to 7e-15).
See docs/superpowers/specs/2026-05-24-analytic-resolution-design.md sec. 4.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from scipy.special import erf


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF Phi(x) via erf (avoids a scipy.stats import)."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _build_M(theta: float) -> np.ndarray:
    """3x5 linear map from instrument variables to imaging-space q.

    Columns: [eps, zeta_v, zeta_h, delta_2theta, xi].
    Rows:    [qrock_prime, qroll, q2th].
    Built by applying the resolution.py:250-256 transform (dphi=0, no
    truncation) to the identity, since the map is linear: M[:, j] = transform(e_j).
    """
    s, c = np.sin(theta), np.cos(theta)
    cot = c / s

    def transform(x: np.ndarray) -> np.ndarray:  # x: (5, N) -> (3, N)
        eps, zeta_v, zeta_h, d2t, xi = x
        qrock = -zeta_v / 2 - d2t / 2
        qroll = -zeta_h / (2 * s) - xi / (2 * s)
        qpar = eps + cot * (-zeta_v / 2 + d2t / 2)
        qrock_prime = c * qrock + s * qpar
        q2th = -s * qrock + c * qpar
        return np.array([qrock_prime, qroll, q2th])

    return transform(np.eye(5))


class AnalyticResolution:
    """Closed-form p_Q(q), peak-normalized to 1 (matches MC Resq_i/Resq_i.max()).

    Parameters mirror the MC bootstrap (reciprocal_space.kernel.generate_kernel):
    sigma_zv = zeta_v_fwhm/2.355, sigma_zh = zeta_h_fwhm/2.35, sigma_NA = NA_rms,
    sigma_eps = eps_rms. The objective NA square aperture (phys_aper) is dropped
    here (validated <1% vs MC); use the MC backend if it ever matters.

    `cond_max` guards the intrinsic ill-conditioning of C_rest: the qrock'
    direction gets Gaussian spread only from eps, so as eps_rms -> 0 the true
    density collapses onto a 2D plane and no finite p_Q exists. We raise rather
    than ridge-regularize (which would fabricate variance and bias the result).
    """

    def __init__(
        self,
        *,
        theta: float,
        zeta_v_fwhm: float,
        zeta_h_fwhm: float,
        NA_rms: float,
        eps_rms: float,
        zeta_v_clip: float = 1.4e-4,
        cond_max: float = 1e8,
    ) -> None:
        self.theta = float(theta)
        self.c = float(zeta_v_clip)
        sig_eps = float(eps_rms)
        sig_zv = zeta_v_fwhm / 2.355
        sig_zh = zeta_h_fwhm / 2.35
        sig_na = float(NA_rms)
        self.sig_zv = sig_zv

        M = _build_M(theta)
        self.M = M
        self.m_u = M[:, 1]  # zeta_v column
        M_g = M[:, [0, 2, 3, 4]]  # eps, zeta_h, d2t, xi
        Sigma_g = np.diag([sig_eps**2, sig_zh**2, sig_na**2, sig_na**2])
        C_rest = M_g @ Sigma_g @ M_g.T

        eigvals = np.linalg.eigvalsh(C_rest)
        cond = eigvals[-1] / eigvals[0] if eigvals[0] > 0 else np.inf
        if not np.isfinite(cond) or cond > cond_max:
            raise ValueError(
                f"Analytic resolution: C_rest is degenerate (cond={cond:.3e} > "
                f"{cond_max:.0e}). The rocking-direction Gaussian spread vanishes "
                "as eps_rms -> 0. Use a finite eps_rms, or backend='mc'."
            )

        self._cho = scipy.linalg.cho_factor(C_rest, lower=True)
        self._logdet_C = 2.0 * float(np.sum(np.log(np.diag(self._cho[0]))))
        self._gaussian_only = sig_zv == 0.0  # no vertical divergence

        if not self._gaussian_only:
            cinv_mu = scipy.linalg.cho_solve(self._cho, self.m_u)
            self.A = float(self.m_u @ cinv_mu) + 1.0 / sig_zv**2
            z_trunc = (
                sig_zv
                * np.sqrt(2 * np.pi)
                * (_norm_cdf(self.c / sig_zv) - _norm_cdf(-self.c / sig_zv))
            )
            self.N0 = (
                (1.0 / z_trunc)
                * (2 * np.pi) ** (-1.5)
                * np.exp(-0.5 * self._logdet_C)
                * np.sqrt(2 * np.pi / self.A)
            )
        else:
            self.A = np.inf
            self.N0 = (2 * np.pi) ** (-1.5) * np.exp(-0.5 * self._logdet_C)

        self._peak = float(self._raw_pq(np.zeros((3, 1)))[0])

    def _raw_pq(self, qi: np.ndarray) -> np.ndarray:
        """Unnormalized closed-form density. qi: (3, N) -> (N,)."""
        cinv_q = scipy.linalg.cho_solve(self._cho, qi)  # (3, N)
        q0 = np.einsum("in,in->n", qi, cinv_q)  # q^T C^-1 q
        if self._gaussian_only:
            return self.N0 * np.exp(-0.5 * q0)
        b = self.m_u @ cinv_q  # (N,)
        mu = b / self.A
        scale = np.sqrt(self.A / 2.0)
        erf_term = 0.5 * (erf((self.c - mu) * scale) - erf((-self.c - mu) * scale))
        return self.N0 * np.exp(-0.5 * (q0 - b**2 / self.A)) * erf_term

    def __call__(self, qi: np.ndarray) -> np.ndarray:
        """Peak-normalized p_Q. qi: (3, N) imaging-space q -> prob (N,)."""
        qi = np.asarray(qi, dtype=float)
        return self._raw_pq(qi) / self._peak
