"""S1 regression: _load_resolution must return a ResolutionContext.

Slice 1 of the ForwardContext refactor (#16): the resolution loaders now
return their ResolutionContext instead of discarding it, and the
idempotent-load fast-path is re-homed to a module-level cache in pipeline.
"""

from pathlib import Path

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import ReciprocalConfig, _load_resolution

_KERNELS = sorted(Path(fm.pkl_fpath).glob("Resq_i_h-1_k1_l-1_17keV_*.npz"))


@pytest.mark.skipif(not _KERNELS, reason="no bootstrapped kernel on disk")
def test_load_resolution_returns_populated_ctx():
    res = _load_resolution(ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0))
    assert res is not None
    assert res.Resq_i is not None
    assert res.loaded_kernel_path in _KERNELS
    assert res.analytic_eval is None


@pytest.mark.skipif(not _KERNELS, reason="no bootstrapped kernel on disk")
def test_load_resolution_returns_ctx_on_repeat_call_idempotent():
    """Cache hit: second call returns a ResolutionContext without disk reload."""
    res1 = _load_resolution(ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0))
    res2 = _load_resolution(ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0))
    assert res2 is not None
    assert np.array_equal(res2.Resq_i, res1.Resq_i)
    assert res2.loaded_kernel_path == res1.loaded_kernel_path


def test_load_resolution_analytic_returns_ctx():
    """Analytic branch: returned ctx has analytic_eval set, Resq_i None."""
    config = ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0, backend="analytic", beamstop=False)
    res = _load_resolution(config)
    assert res is not None
    assert res.analytic_eval is not None
    assert res.Resq_i is None
