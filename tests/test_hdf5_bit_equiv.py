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
        "Two real blockers, neither about RNG seeding (the wall path Find_Hg -> "
        "Fd_find is fully deterministic -- no np.random anywhere): (1) the test "
        "needs a loaded MC kernel to render via forward(), so it SKIPS on a bare "
        "checkout with no kernel; (2) the golden was captured at Nsub=2 while the "
        "default is now Nsub=1, so even with a kernel the rendered image differs. "
        "To un-xfail: load a kernel, then regenerate the golden at the current "
        "Nsub default. (Kernel MC generation is now seedable via "
        "dfxm-bootstrap --seed / generate_kernel(seed=), but this golden predates "
        "that and was not captured with a pinned seed.) Same situation as the "
        "sibling xfail on test_forward_output_matches_pickle_era_snapshot."
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
