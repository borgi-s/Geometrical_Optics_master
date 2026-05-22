"""Smoke test for the write_identification_h5 orchestrator with a fake scan_iter."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.io.hdf5 import (
    MASTER_IDENTIFY,
    ScanSpec,
    write_identification_h5,
)
from dfxm_geo.pipeline import _lookup_and_load_kernel


def _require_kernel() -> None:
    """Skip unless a bootstrapped (-1,1,-1) 17 keV kernel npz is on disk."""
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip(f"no kernel npz found in {kernel_dir}")


def _fake_scan_iter(Hg: np.ndarray) -> Iterator[ScanSpec]:
    # Two scan entries, each with one detector and one frame.
    for k in range(2):
        yield ScanSpec(
            title=f"single scan {k}",
            sample={
                "name": "simulated, dislocation identification (single)",
                "slip_plane_normal": np.asarray([1, 1, 1], dtype=np.int32),
                "burgers": np.asarray([1, 0, 1], dtype=np.int32),
                "rotation_deg": float(k * 10),
            },
            positioners={"phi": 1.5e-4, "chi": 0.0},
            dfxm_geo={"Hg": Hg, "q_hkl": np.array([0.0, 0.0, 1.0])},
            detectors={"dfxm_sim_detector": [(0, Hg, 1.5e-4, 0.0, 0.0)]},
            attrs={
                "scan_mode": "single",
                "scanned_axes": [],
                "identify_mode": "single",
            },
        )


def test_write_identification_h5_basic(tmp_path: Path) -> None:
    _require_kernel()
    _lookup_and_load_kernel((-1, 1, -1), 17.0)
    if fm.Hg is None:
        pytest.skip("forward_model.Hg not populated; run dfxm-bootstrap.")
    try:
        Hg = fm.Hg
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        n_scans = write_identification_h5(
            out_dir,
            scan_iter=_fake_scan_iter(Hg),
            cli="pytest test_write_identification_h5",
            config_toml='mode = "single"\n',
        )
        assert n_scans == 2
        master = out_dir / MASTER_IDENTIFY
        assert master.is_file()
        assert (out_dir / "scan0001" / "dfxm_sim_detector_0000.h5").is_file()
        assert (out_dir / "scan0002" / "dfxm_sim_detector_0000.h5").is_file()
        with h5py.File(master, "r") as f:
            assert "/1.1" in f and "/2.1" in f
            assert f["/1.1"].attrs["identify_mode"] == "single"
            # External link resolves; forward() returns (NN2//Nsub, NN1//Nsub)
            det = f["/1.1/instrument/dfxm_sim_detector/data"]
            assert det.shape[1:] == (fm.NN2 // fm.Nsub, fm.NN1 // fm.Nsub)
    finally:
        fm.Hg = None
        fm._loaded_kernel_path = None
