"""Task 2 (#10): run_postprocess recovers Hg from the HDF5, not a stale global."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm


@pytest.fixture
def _kernel_on_disk() -> None:
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip("no (-1,1,-1) 17keV kernel npz on disk; run dfxm-bootstrap")


def test_run_postprocess_reads_Hg_from_file_when_global_absent(
    tmp_path: Path, _kernel_on_disk: None, monkeypatch
) -> None:
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

    cfg = SimulationConfig(
        crystal=CrystalConfig(
            mode="wall",
            wall=WallCrystalConfig(dis=4.0, ndis=10, sample_remount="S1"),
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
    # Simulate a fresh process / --postprocess-only: clear the strain global.
    monkeypatch.setattr(fm, "Hg", None, raising=False)
    # Must NOT raise; must recover Hg from /1.1/dfxm_geo/Hg and write qi_field.
    run_postprocess(out, cfg)
    with h5py.File(out / "dfxm_geo.h5", "r") as f:
        assert "/1.1/dfxm_geo/analysis/qi_field" in f
