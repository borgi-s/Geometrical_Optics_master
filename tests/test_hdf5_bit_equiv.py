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

    # Always (re)load the canonical kernel rather than trusting fm.Resq_i: in a
    # full-suite run an earlier test may have left a DIFFERENT kernel / oblique
    # theta in the fm module globals, which would make this comparison render
    # the wrong geometry (fm-globals pollution; see the ForwardContext refactor
    # follow-up). On a bare checkout with no kernel npz the load raises -> skip.
    try:
        _lookup_and_load_kernel((-1, 1, -1), 17.0)
    except Exception:  # noqa: BLE001 - any load failure -> skip
        pytest.skip("no MC kernel available on this checkout")
    if fm.Resq_i is None:
        pytest.skip("kernel not loaded")
    # Force the MC-LUT path: a prior test may have left fm._analytic_eval set
    # (analytic backend), which forward() would otherwise prefer over the LUT.
    fm._analytic_eval = None

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
    ctx = fm._context_from_globals()
    base_qc = fm.precompute_forward_static(Hg, ctx)
    args = []
    for chi_idx in range(2):
        for phi_idx in range(2):
            k = chi_idx * 2 + phi_idx
            args.append((k, base_qc, float(Phi[phi_idx]), float(Chi[chi_idx]), 0.0, ctx))
    _compute_and_write_detector_file_parallel(out, args, max_workers=1)

    expected = np.load(GOLDEN)  # shape (4, 8, 8), float64
    with h5py.File(out, "r") as f:
        actual_full = f["/entry_0000/dfxm_sim_detector/image"][...]
    actual_crop = actual_full[:, 322:330, 51:59]
    # float32 detector storage (Phase 2a): compare at float32 tolerance.
    # The legacy golden is float64; the stored stack is now float32, so the
    # only difference is float32 rounding (~1e-7 relative).
    np.testing.assert_allclose(
        actual_crop.astype(np.float64),
        expected,
        rtol=1e-6,
        atol=1e-6,
        err_msg="HDF5 writer output deviates from legacy .npy writer at 8x8 corner",
    )
