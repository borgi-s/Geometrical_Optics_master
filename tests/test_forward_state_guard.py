import inspect

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo import pipeline


def test_multi_identify_generator_does_not_assign_fm_q_hkl():
    """_iter_identification_multi must not assign the fm.q_hkl global.

    Regression for #10: a generator writing a process global is a data race
    on the planned persistent-worker pool. The original write was a no-op
    (assigned back the value just read), so this guards the source directly.
    """
    src = inspect.getsource(pipeline._iter_identification_multi)
    # No assignment to fm.q_hkl in any spacing (reads `... = np.asarray(fm.q_hkl...)`
    # are fine; only `fm.q_hkl = ...` is the forbidden write).
    compact = src.replace(" ", "")
    assert "fm.q_hkl=" not in compact, "generator still assigns fm.q_hkl"


def test_forward_state_guard_restores_all_mutable_globals():
    # Drive off the guard's own list so this test auto-covers any name added to
    # (or removed from) _GUARDED_GLOBALS — no stale hardcoded copy to drift.
    names = list(fm._GUARDED_GLOBALS)
    before = {n: getattr(fm, n, None) for n in names}
    with fm._forward_state_guard():
        fm.theta_0 = 1.2345
        fm.Hg = np.array([[9.0]])
        fm._analytic_eval = object()
    after = {n: getattr(fm, n, None) for n in names}
    for n in names:
        b, a = before[n], after[n]
        if isinstance(b, np.ndarray) or isinstance(a, np.ndarray):
            assert np.array_equal(np.asarray(b), np.asarray(a)), n
        else:
            assert b is a or b == a, n


@pytest.fixture
def _kernel_on_disk() -> None:
    from pathlib import Path

    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip("no (-1,1,-1) 17keV kernel npz on disk; run dfxm-bootstrap")


def test_run_postprocess_self_loads_kernel_after_guarded_run_simulation(
    tmp_path, _kernel_on_disk, monkeypatch
):
    from dfxm_geo.pipeline import (
        AxisScanConfig,
        CrystalConfig,
        IOConfig,
        ReciprocalConfig,
        ScanConfig,
        SimulationConfig,
        WallCrystalConfig,
        run_postprocess,
        run_simulation,
    )

    # Reset kernel globals to None before the test so the guard snapshots None
    # and restores None, regardless of what a prior test may have left loaded.
    # Without this, the guard correctly restores the pre-call value (whatever
    # the previous test loaded), making "is None" order-dependent.
    monkeypatch.setattr(fm, "Resq_i", None)
    monkeypatch.setattr(fm, "_loaded_kernel_path", None)

    cfg = SimulationConfig(
        crystal=CrystalConfig(
            mode="wall", wall=WallCrystalConfig(dis=4.0, ndis=10, sample_remount="S1")
        ),
        scan=ScanConfig(
            phi=AxisScanConfig(range=0.0006 * 180 / np.pi, steps=3),
            chi=AxisScanConfig(range=0.002 * 180 / np.pi, steps=3),
        ),
        io=IOConfig(include_perfect_crystal=True, max_workers=1),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    out = tmp_path / "run"
    run_simulation(cfg, out)
    assert fm.Resq_i is None, "guard should have restored Resq_i to None on run_simulation exit"
    run_postprocess(out, cfg)  # must self-load the kernel, not crash
    import h5py

    with h5py.File(out / "dfxm_geo.h5", "r") as f:
        assert "/1.1/dfxm_geo/analysis/qi_field" in f
