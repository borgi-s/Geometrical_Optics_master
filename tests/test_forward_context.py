"""Parity tests for the ForwardContext dataclasses and their constructors.

Slice 2 of the ForwardContext refactor (#16): additive constructors that
snapshot the current module globals into frozen dataclasses.
"""

from pathlib import Path

import numpy as np
import pytest

from dfxm_geo.direct_space import forward_model as fm


def test_build_instrument_context_matches_globals():
    instr = fm.build_instrument_context()
    assert instr.psize == fm.psize
    assert instr.zl_rms == fm.zl_rms
    assert instr.Npixels == fm.Npixels
    assert instr.Nsub == fm.Nsub
    assert (instr.NN1, instr.NN2, instr.NN3) == (fm.NN1, fm.NN2, fm.NN3)
    assert np.array_equal(instr.Ud, fm.Ud)
    assert np.array_equal(instr.Us, fm.Us)
    assert np.array_equal(instr.flat_indices, fm._flat_indices)
    assert instr.yl_start == fm.yl_start
    assert (instr.xl_steps, instr.yl_steps, instr.zl_steps) == (
        fm.xl_steps,
        fm.yl_steps,
        fm.zl_steps,
    )


def test_build_geometry_context_is_internally_consistent():
    # build_geometry_context is the sole source of the theta-dependent geometry
    # (#16 Slice 5 deleted the module globals Theta/rl/prob_z/xl_* it used to
    # mirror), so assert it reproduces the geometry from theta directly.
    instr = fm.build_instrument_context()
    theta = 0.15661142  # true Bragg for (-1,1,-1) @ 17 keV (rad)
    geom = fm.build_geometry_context(theta, instr)
    assert geom.theta_0 == theta
    assert np.allclose(
        geom.Theta,
        np.array(
            [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
        ),
    )
    assert geom.xl_start == instr.yl_start / np.tan(2 * theta) / 3
    assert geom.xl_range == -geom.xl_start
    assert geom.rl.shape == (3, instr.xl_steps * instr.yl_steps * instr.zl_steps)
    assert geom.prob_z.shape == (geom.rl.shape[1],)


_KERNELS = sorted(Path(fm.pkl_fpath).glob("Resq_i_h-1_k1_l-1_17keV_*.npz"))


@pytest.mark.skipif(not _KERNELS, reason="no bootstrapped kernel on disk")
def test_forward_ctx_path_does_not_recompile_kernel():
    # Threading ctx must not trigger a numba recompile of _mc_lut_forward
    # (the dtype/shape signature is stable across calls) and is deterministic.
    from dfxm_geo.pipeline import ReciprocalConfig, SimulationConfig, run_theta

    res = fm._load_default_kernel(_KERNELS[0])
    cfg = SimulationConfig(reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0))
    ctx = fm.build_forward_context(run_theta(cfg), res, (-1, 1, -1))
    Hg, _ = fm.Find_Hg(4.0, 151, fm.psize, fm.zl_rms, ctx=ctx)
    # Warm up the JIT; the first in-process call compiles the kernel.
    _ = fm.forward(Hg, ctx, phi=1e-4, chi=2e-4)
    n_sig_before = len(fm._mc_lut_forward.signatures)
    img1 = fm.forward(Hg, ctx, phi=1e-4, chi=2e-4)
    img2 = fm.forward(Hg, ctx, phi=1e-4, chi=2e-4)
    assert np.array_equal(img1, img2)  # bit-exact, deterministic
    assert len(fm._mc_lut_forward.signatures) == n_sig_before  # no new compile


@pytest.mark.skipif(not _KERNELS, reason="no bootstrapped kernel on disk")
def test_load_default_kernel_returns_populated_resolution_context():
    res = fm._load_default_kernel(_KERNELS[0])
    assert res is not None
    assert res.Resq_i is not None
    assert res.qi1_step != 0.0 and res.qi2_step != 0.0 and res.qi3_step != 0.0
    assert res.npoints1 is not None and res.npoints2 is not None and res.npoints3 is not None
    assert res.analytic_eval is None
    assert res.loaded_kernel_path == Path(_KERNELS[0])
