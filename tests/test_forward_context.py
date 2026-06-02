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


def test_build_geometry_context_matches_default_globals():
    instr = fm.build_instrument_context()
    geom = fm.build_geometry_context(fm.theta_0, instr)
    assert geom.theta_0 == fm.theta_0
    assert np.array_equal(geom.Theta, fm.Theta)
    assert geom.xl_start == fm.xl_start
    assert geom.xl_range == fm.xl_range
    assert np.array_equal(geom.rl, fm.rl)
    assert np.array_equal(geom.prob_z, fm.prob_z)


_KERNELS = sorted(Path(fm.pkl_fpath).glob("Resq_i_h-1_k1_l-1_17keV_*.npz"))


@pytest.mark.skipif(not _KERNELS, reason="no bootstrapped kernel on disk")
def test_load_default_kernel_returns_matching_resolution_context():
    res = fm._load_default_kernel(_KERNELS[0], compute_Hg=False)
    assert res is not None
    assert np.array_equal(res.Resq_i, fm.Resq_i)
    assert res.qi1_start == fm.qi1_start and res.qi1_step == fm.qi1_step
    assert res.qi2_start == fm.qi2_start and res.qi2_step == fm.qi2_step
    assert res.qi3_start == fm.qi3_start and res.qi3_step == fm.qi3_step
    assert (res.npoints1, res.npoints2, res.npoints3) == (
        fm.npoints1,
        fm.npoints2,
        fm.npoints3,
    )
    assert res.analytic_eval is None
    assert res.loaded_kernel_path == fm._loaded_kernel_path
