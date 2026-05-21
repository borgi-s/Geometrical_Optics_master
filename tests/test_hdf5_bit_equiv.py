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


@pytest.mark.xfail(
    reason=(
        "Bit-equivalence golden was captured against a specific Fg cache + Nsub=2. "
        "Find_Hg uses non-seeded np.random.default_rng(), so fresh checkouts "
        "regenerate Fg with different dislocation positions; and the Nsub default "
        "is now 1 (was 2 when the golden was captured). The snapshot therefore "
        "only matches on machines that retained the original Fg cache AND run "
        "with Nsub=2 manually. Same root cause as the sibling xfail on "
        "test_forward_output_matches_pickle_era_snapshot. Follow-up: seed Find_Hg "
        "for reproducible test fixtures, then regenerate the golden."
    ),
    strict=False,
)
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
    # v1.2.0 layout: detector file lives under scan0001/.
    out = tmp_path / "scan0001" / "dfxm_sim_detector_0000.h5"
    # Build the same (phi, chi) grid the old _save_scan_parallel_to_h5
    # exercised: phi inner, chi outer, frame_idx = chi_idx * phi_steps + phi_idx.
    Phi = np.linspace(-TINY_HALF_RANGE_RAD, TINY_HALF_RANGE_RAD, 2)
    Chi = np.linspace(-TINY_HALF_RANGE_RAD, TINY_HALF_RANGE_RAD, 2)
    args = []
    for chi_idx in range(2):
        for phi_idx in range(2):
            k = chi_idx * 2 + phi_idx
            args.append((k, Hg, float(Phi[phi_idx]), float(Chi[chi_idx])))
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
