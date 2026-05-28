"""Gate B (v2.3.0 oblique ship-gate): dfxm_geo's oblique reciprocal resolution
matches the paper's own code (darkmod PentaGauss) in ellipsoid shape.

The paper (arXiv:2503.22022) was produced with darkmod (Henningsson,
github.com/AxelHenningsson/darkmod), not dfxm_geo. darkmod's PentaGauss
resolution map ``_get_M(theta, eta)`` is the "vectorized Poulsen 2017" model. We
vendor that ~15-line map here as an oracle (darkmod itself needs Astra/GPU to
import, so it is NOT a test dependency) and assert that dfxm_geo's
``_build_M(theta, eta)`` produces the same resolution-covariance ellipsoid SHAPE
for the paper's oblique (-1,-1,3) geometry.

Known, bounded approximation: dfxm applies eta as a post-hoc rigid rotation
``R_x(eta)`` (eigenvalues eta-invariant), whereas darkmod weaves eta into the
diffraction geometry, deforming the thinnest (rocking) eigenvalue by ~3.9%. Shape
parity is therefore asserted on the NORMALIZED eigenvalues; absolute scale differs
by a units convention (dfxm normalized-q vs darkmod absolute k*Q), so it is not
compared.

See docs/superpowers/specs/2026-05-29-v230-oblique-ship-gate-rescope.md.
"""

from __future__ import annotations

import numpy as np

from dfxm_geo.reciprocal_space.analytic_resolution import _build_M

# Paper sec. 6.1/6.2 geometry: Al (-1,-1,3) @ 19.1 keV.
THETA = np.deg2rad(15.416)
ETA = np.deg2rad(20.233)
LAMBDA = 12.398419 / 19.1  # Angstrom (hc/E)
K = 2 * np.pi / LAMBDA

# Matched instrument sigmas (paper sec. 6.2).
_FWHM_TO_SIG = 2.0 * np.sqrt(2.0 * np.log(2.0))
SIG_EPS = 6e-5  # dk/k std
SIG_ZV = 0.027e-3 / _FWHM_TO_SIG  # vertical beam divergence
SIG_ZH = 0.0  # horizontal beam divergence
SIG_NA = 0.556e-3 / _FWHM_TO_SIG  # CRL acceptance


def _darkmod_get_M(theta: float, eta: float) -> np.ndarray:
    """Vendored verbatim from darkmod ``PentaGauss._get_M`` (darkmod/resolution.py).

    Columns: [eps, beam_h, beam_v, CRL_h, CRL_v]. Lab frame, scaled by k.
    """
    th0, e0 = theta, eta
    return K * np.array(
        [
            [np.cos(2 * th0) - 1, 0, 0, 0, -np.sin(2 * th0)],
            [-np.sin(e0) * np.sin(2 * th0), -1, 0, np.cos(e0), -np.sin(e0) * np.cos(2 * th0)],
            [np.cos(e0) * np.sin(2 * th0), 0, -1, np.sin(e0), np.cos(e0) * np.cos(2 * th0)],
        ]
    )


# dfxm column order:    [eps, zeta_v, zeta_h, d2t, xi]
_SIG_DFXM = np.array([SIG_EPS**2, SIG_ZV**2, SIG_ZH**2, SIG_NA**2, SIG_NA**2])
# darkmod column order: [eps, beam_h, beam_v, CRL_h, CRL_v]
_SIG_DK = np.array([SIG_EPS**2, SIG_ZH**2, SIG_ZV**2, SIG_NA**2, SIG_NA**2])


def _cov_eigvals(M: np.ndarray, sig2_diag: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of the resolution covariance M @ diag(sig2) @ M.T."""
    cov = M @ np.diag(sig2_diag) @ M.T
    return np.sort(np.linalg.eigvalsh(cov))[::-1]


def test_oblique_resolution_ellipsoid_shape_matches_darkmod() -> None:
    """Normalized resolution-covariance eigenvalues (ellipsoid shape) agree within 1%."""
    e_dfxm = _cov_eigvals(_build_M(THETA, ETA), _SIG_DFXM)
    e_dk = _cov_eigvals(_darkmod_get_M(THETA, ETA), _SIG_DK)
    n_dfxm = e_dfxm / e_dfxm[0]
    n_dk = e_dk / e_dk[0]
    assert np.allclose(n_dfxm, n_dk, atol=0.01), (
        f"oblique resolution shape mismatch: dfxm {n_dfxm} vs darkmod {n_dk}"
    )


def test_dfxm_eta_is_rigid_rotation() -> None:
    """dfxm treats eta as a rigid R_x rotation -> eigenvalues are eta-invariant."""
    e0 = _cov_eigvals(_build_M(THETA, 0.0), _SIG_DFXM)
    e_eta = _cov_eigvals(_build_M(THETA, ETA), _SIG_DFXM)
    assert np.allclose(e0, e_eta, rtol=1e-9)


def test_darkmod_eta_deformation_is_bounded() -> None:
    """darkmod weaves eta in (eigenvalues shift slightly). The shift bounds the error
    dfxm incurs by treating eta as a rigid rotation; it must stay < 5%."""
    e0 = _cov_eigvals(_darkmod_get_M(THETA, 0.0), _SIG_DK)
    e_eta = _cov_eigvals(_darkmod_get_M(THETA, ETA), _SIG_DK)
    rel = float(np.max(np.abs(e_eta - e0) / e0))
    assert rel < 0.05, f"darkmod eta deformation {rel:.4f} exceeds 5%"
