"""End-to-end: run_simulation writes a single .h5 instead of .npy dirs."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    AxisScanConfig,
    CrystalConfig,
    IOConfig,
    ReciprocalConfig,
    ScanConfig,
    SimulationConfig,
    WallCrystalConfig,
    run_simulation,
)


def test_run_simulation_writes_hdf5(tmp_path: Path) -> None:
    from pathlib import Path as _P

    # Skip if no real bootstrapped kernel is available (the multi-reflection tests
    # may have left a toy zero kernel in fm.Hg, but that's not the real kernel).
    kernel_dir = _P(fm.pkl_fpath)
    matches = sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz"))
    if not matches:
        pytest.skip(f"No bootstrapped kernel npz found in {kernel_dir}; skipping.")
    if fm.Hg is None:
        pytest.skip("kernel not loaded")

    cfg = SimulationConfig(
        crystal=CrystalConfig(
            mode="wall",
            wall=WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S1"),
        ),
        scan=ScanConfig(
            phi=AxisScanConfig(range=0.0006 * 180 / np.pi, steps=3),
            # chi fixed at 0 (no range/steps) — single chi slice
        ),
        io=IOConfig(include_perfect_crystal=True),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    out = tmp_path / "run"
    run_simulation(cfg, out)
    h5 = out / "dfxm_geo.h5"
    assert h5.exists()
    # And the old .npy directories are NOT created.
    assert not (out / "images10").exists()
    assert not (out / "images10_perf_crystal").exists()
    with h5py.File(h5, "r") as f:
        assert "/1.1/instrument/dfxm_sim_detector/data" in f
        assert "/2.1/instrument/dfxm_sim_detector/data" in f
