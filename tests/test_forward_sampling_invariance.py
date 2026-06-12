"""Nsub-invariance of the normalized forward image (spec §3.2(a))."""

import numpy as np
import pytest

from dfxm_geo.direct_space.forward_model import (
    ForwardContext,
    GeometryContext,
    InstrumentContext,
    ResolutionContext,
    forward_from_static,
)
from dfxm_geo.reciprocal_space.analytic_resolution import AnalyticResolution


def _make_ctx(npixels: int, nsub: int, theta: float) -> ForwardContext:
    """Hand-built mini context replicating the module grid math
    (forward_model.py lines ~140-195 and build_geometry_context ~645-654);
    hand-built because build_instrument_context() snapshots module globals
    and build_geometry_context reads module-level yl_range/zl_range."""
    psize = 40e-9
    zl_rms = 0.15e-6 / 2.35
    nn1 = npixels // 3 * nsub
    nn2 = npixels * nsub
    nn3 = npixels // 30 * nsub
    yl_start = -psize * npixels / 2 + psize / (2 * nsub)
    yl_range = -yl_start
    zl_range = 0.5 * zl_rms * 6
    YI = (np.arange(nn1) // nsub).repeat(nn3 * nn2)
    ZI = np.tile((np.arange(nn2) // nsub).repeat(nn3), nn1)
    flat_indices = ZI.astype(np.int64) * (nn1 // nsub) + YI.astype(np.int64)
    instr = InstrumentContext(
        psize=psize,
        zl_rms=zl_rms,
        Npixels=npixels,
        Nsub=nsub,
        NN1=nn1,
        NN2=nn2,
        NN3=nn3,
        Ud=np.identity(3),
        Us=np.identity(3),
        flat_indices=flat_indices,
        yl_start=yl_start,
        xl_steps=nn1,
        yl_steps=nn2,
        zl_steps=nn3,
    )
    Theta_ = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    xl_start = yl_start / np.tan(2 * theta) / 3
    xl_range = -xl_start
    rl = np.vstack(
        np.mgrid[
            -xl_range : xl_range : complex(nn1),
            -yl_range : yl_range : complex(nn2),
            -zl_range : zl_range : complex(nn3),
        ]
    ).reshape(3, -1)
    geom = GeometryContext(
        theta_0=theta,
        Theta=Theta_,
        xl_start=xl_start,
        xl_range=xl_range,
        rl=rl,
        prob_z=np.exp(-0.5 * (rl[2] / zl_rms) ** 2),
    )
    # Analytic backend: grid-free, no kernel file needed. Constructor args
    # mirror _load_analytic_resolution (forward_model.py ~570-599) — copy the
    # exact kwargs that call uses (zeta_v_fwhm/zeta_h_fwhm/NA_rms/eps_rms).
    analytic = AnalyticResolution(
        theta=theta,
        zeta_v_fwhm=0.53e-3,
        zeta_h_fwhm=0.0,
        NA_rms=7.31e-4 / 2.35,
        eps_rms=0.014 / 2.35,
    )
    res = ResolutionContext(
        Resq_i=None,
        qi1_start=0.0,
        qi1_step=0.0,
        qi2_start=0.0,
        qi2_step=0.0,
        qi3_start=0.0,
        qi3_step=0.0,
        npoints1=None,
        npoints2=None,
        npoints3=None,
        analytic_eval=analytic,
        loaded_kernel_path=None,
    )
    q = np.array([-1.0, 1.0, -1.0]) / np.sqrt(3.0)
    return ForwardContext(instrument=instr, geometry=geom, resolution=res, q_hkl=q)


@pytest.mark.slow
def test_normalized_image_invariant_under_nsub():
    """base_qc = 0 puts every ray at the acceptance peak (the analytic
    backend is peak-normalized in deviation space), so the image reduces to
    the pure accumulation Σ p_Q(Theta@ang)·prob_z per pixel — exactly the
    quantity the normalization must make sampling-invariant. No Hg physics
    needed; invariance is a property of the accumulation, not the scene."""
    theta = 0.31  # ~17.8 deg, Al(111)-like
    images = {}
    for nsub in (1, 2):
        ctx = _make_ctx(90, nsub, theta)
        n = ctx.geometry.rl.shape[1]
        base_qc = np.zeros((3, n))
        images[nsub] = np.asarray(forward_from_static(base_qc, ctx))
    im1, im2 = images[1], images[2]
    assert im1.shape == im2.shape  # (npixels, npixels//3) regardless of Nsub
    assert im1.sum() > 0  # rays sit at the acceptance peak by construction
    # Without normalization the level ratio would be Nsub^2*(NN3 ratio) = 8.
    ratio = im2.sum() / im1.sum()
    assert ratio == pytest.approx(1.0, abs=0.05)
