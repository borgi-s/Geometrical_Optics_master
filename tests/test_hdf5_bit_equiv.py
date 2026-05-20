"""Layer 2: bit-equivalence — new HDF5 writer == old .npy writer (golden)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.crystal.remount import SAMPLE_REMOUNT_OPTIONS
from dfxm_geo.io.hdf5 import _save_scan_parallel_to_h5

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "data" / "golden" / "forward_legacy_writer_4frames_8x8.npy"


def test_hdf5_writer_bit_equivalent_to_legacy_npy_golden(tmp_path: Path) -> None:
    """Same Hg + (phi, chi) grid -> identical images via HDF5 writer."""
    if fm.Hg is None:
        pytest.skip("kernel not loaded")

    Hg, q_hkl = fm.Find_Hg(
        4.0,
        151,
        fm.psize,
        fm.zl_rms,
        S=SAMPLE_REMOUNT_OPTIONS["S1"],
        remount_name="S1",
    )
    fm.Hg = Hg
    fm.q_hkl = q_hkl

    # MUST use the same tiny half-range and crop window as the golden
    # generator (tests/_gen_forward_legacy_golden.py).
    TINY_HALF_RANGE_RAD = 5e-5
    out = tmp_path / "test.h5"
    _save_scan_parallel_to_h5(
        out,
        scan_id="1.1",
        Hg=Hg,
        phi_range=TINY_HALF_RANGE_RAD * 180 / np.pi,
        phi_steps=2,
        chi_range=TINY_HALF_RANGE_RAD * 180 / np.pi,
        chi_steps=2,
        max_workers=1,
    )

    expected = np.load(GOLDEN)  # shape (4, 8, 8), float64
    with h5py.File(out, "r") as f:
        actual_full = f["/1.1/instrument/dfxm_sim_detector/data"][...]
    actual_crop = actual_full[:, 322:330, 51:59]
    np.testing.assert_array_equal(
        actual_crop,
        expected,
        err_msg="HDF5 writer output deviates from legacy .npy writer at 8x8 corner",
    )
