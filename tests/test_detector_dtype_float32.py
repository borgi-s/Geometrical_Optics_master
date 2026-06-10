"""Detector image stack is stored as float32 (Phase 2a: lossless ~2x slim)."""

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


def _kernel_or_skip() -> None:
    kernel_dir = Path(fm.pkl_fpath)
    # #16 Slice 5: fm.Hg is no longer set by the loader; gate only on file presence.
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip("No bootstrapped kernel npz found; skipping.")


def test_detector_image_is_float32(tmp_path: Path) -> None:
    _kernel_or_skip()
    cfg = SimulationConfig(
        crystal=CrystalConfig(
            mode="wall",
            wall=WallCrystalConfig(dis=4.0, ndis=151, sample_remount="S1"),
        ),
        scan=ScanConfig(phi=AxisScanConfig(range=0.0006 * 180 / np.pi, steps=3)),
        io=IOConfig(include_perfect_crystal=False),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    out = tmp_path / "run"
    run_simulation(cfg, out)
    det = out / "scan0001" / "dfxm_sim_detector_0000.h5"
    assert det.is_file()
    with h5py.File(det, "r") as f:
        img = f["/entry_0000/dfxm_sim_detector/image"]
        assert img.dtype == np.float32, f"expected float32, got {img.dtype}"
        data = img[()]
    assert np.isfinite(data).all()
    assert float(data.sum()) > 0.0
