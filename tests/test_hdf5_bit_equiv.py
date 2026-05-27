"""Layer 2: bit-equivalence — new HDF5 writer == old .npy writer (golden)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.crystal.remount import SAMPLE_REMOUNT_OPTIONS
from dfxm_geo.io.hdf5 import _compute_and_write_detector_file_parallel

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "data" / "golden" / "forward_legacy_writer_4frames_8x8.npy"


def test_hdf5_writer_bit_equivalent_to_legacy_npy_golden(tmp_path: Path) -> None:
    """HDF5 writer is byte-for-byte equivalent to the legacy .npy writer.

    The golden (tests/data/golden/forward_legacy_writer_4frames_8x8.npy) is
    captured at the Nsub=1 default via the legacy `.npy` writer by
    tests/_gen_forward_legacy_golden.py. Here the same Hg + (phi, chi) grid is
    rendered via the HDF5 writer (`_compute_and_write_detector_file_parallel`)
    and the 8x8 crop is asserted bit-identical. Both writers route through
    `forward_from_static`, so any divergence is a real writer bug.

    Loads the canonical MC kernel itself; skips only on a bare checkout where
    no kernel `.npz` is present (e.g. CI).
    """
    from dfxm_geo.pipeline import _lookup_and_load_kernel

    if fm.Resq_i is None:
        try:
            _lookup_and_load_kernel((-1, 1, -1), 17.0)
        except Exception:  # noqa: BLE001 - any load failure -> skip
            pytest.skip("no MC kernel available on this checkout")
    if fm.Resq_i is None:
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
    # v1.2.0 layout: detector file lives under scan0001/.
    out = tmp_path / "scan0001" / "dfxm_sim_detector_0000.h5"
    # Build the same (phi, chi) grid the old _save_scan_parallel_to_h5
    # exercised: phi inner, chi outer, frame_idx = chi_idx * phi_steps + phi_idx.
    Phi = np.linspace(-TINY_HALF_RANGE_RAD, TINY_HALF_RANGE_RAD, 2)
    Chi = np.linspace(-TINY_HALF_RANGE_RAD, TINY_HALF_RANGE_RAD, 2)
    base_qc = fm.precompute_forward_static(Hg)
    args = []
    for chi_idx in range(2):
        for phi_idx in range(2):
            k = chi_idx * 2 + phi_idx
            args.append((k, base_qc, float(Phi[phi_idx]), float(Chi[chi_idx]), 0.0))
    _compute_and_write_detector_file_parallel(out, args, max_workers=1)

    expected = np.load(GOLDEN)  # shape (4, 8, 8), float64
    with h5py.File(out, "r") as f:
        actual_full = f["/entry_0000/dfxm_sim_detector/image"][...]
    actual_crop = actual_full[:, 322:330, 51:59]
    np.testing.assert_array_equal(
        actual_crop,
        expected,
        err_msg="HDF5 writer output deviates from legacy .npy writer at 8x8 corner",
    )
