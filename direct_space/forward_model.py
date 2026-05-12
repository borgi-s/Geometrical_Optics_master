"""DEPRECATED — re-exports from `dfxm_geo.direct_space.forward_model`.

This module is kept as a compatibility shim during Phase 4 of the
cleanup. Importing it triggers the same import-time setup as the new
module (lazy-loading the default kernel pickle iff present).

New code should import directly:

    from dfxm_geo.direct_space.forward_model import forward, Find_Hg
"""

from dfxm_geo.direct_space.forward_model import *  # noqa: F401, F403
from dfxm_geo.direct_space.forward_model import (  # noqa: F401
    NN1,
    NN2,
    NN3,
    YI,
    ZI,
    Find_Hg,
    Hg,
    Npixels,
    Nsub,
    Resq_i,
    Theta,
    Ud,
    Us,
    _load_default_kernel,
    dis,
    forward,
    indices,
    ndis,
    pkl_fn,
    pkl_fpath,
    prob_z,
    psize,
    q_hkl,
    qi1_range,
    qi1_start,
    qi1_step,
    qi2_range,
    qi2_start,
    qi2_step,
    qi3_range,
    qi3_start,
    qi3_step,
    qi_starts,
    qi_steps,
    rl,
    theta,
    theta_0,
    vars_fn,
    xl_range,
    xl_start,
    xl_steps,
    yl_range,
    yl_start,
    yl_steps,
    zl_range,
    zl_rms,
    zl_start,
    zl_steps,
)
