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
    reason="#16 Slice 5: the true-Bragg theta-fix (and the pre-existing unseeded "
    "Find_Hg / Nsub divergence) intentionally diverge from the pickle-era legacy "
    "golden. The golden is NOT regenerated (Sina-authorized); kept as a documented "
    "xfail so a future re-baseline at true-Bragg surfaces as xpass.",
    strict=False,
)
def test_hdf5_writer_bit_equivalent_to_legacy_npy_golden(tmp_path: Path) -> None:
    """HDF5 writer vs the legacy .npy writer golden (xfail; see marker).

    The golden (tests/data/golden/forward_legacy_writer_4frames_8x8.npy) was
    captured via the legacy `.npy` writer at the pickle-era geometry. Here the
    same Hg + (phi, chi) grid is rendered via the HDF5 writer
    (`_compute_and_write_detector_file_parallel`) and the 8x8 crop is compared.
    The geometry now uses the true-Bragg theta (#16 Slice 5), so the comparison
    is expected to diverge — hence xfail.

    Loads the canonical MC kernel itself; skips only on a bare checkout where
    no kernel `.npz` is present (e.g. CI).
    """
    from dfxm_geo.pipeline import (
        ReciprocalConfig,
        SimulationConfig,
        _lookup_and_load_kernel,
        run_theta,
    )

    try:
        res = _lookup_and_load_kernel((-1, 1, -1), 17.0)
    except Exception:  # noqa: BLE001 - any load failure -> skip
        pytest.skip("no MC kernel available on this checkout")

    cfg = SimulationConfig(reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0))
    ctx = fm.build_forward_context(run_theta(cfg), res, (-1, 1, -1))

    Hg, q_hkl = fm.Find_Hg(
        4.0,
        151,
        fm.psize,
        fm.zl_rms,
        S=SAMPLE_REMOUNT_OPTIONS["S1"],
        remount_name="S1",
        ctx=ctx,
    )

    # MUST use the same tiny half-range and crop window as the golden
    # generator (tests/_gen_forward_legacy_golden.py).
    TINY_HALF_RANGE_RAD = 5e-5
    # v1.2.0 layout: detector file lives under scan0001/.
    out = tmp_path / "scan0001" / "dfxm_sim_detector_0000.h5"
    # Build the same (phi, chi) grid the old _save_scan_parallel_to_h5
    # exercised: phi inner, chi outer, frame_idx = chi_idx * phi_steps + phi_idx.
    Phi = np.linspace(-TINY_HALF_RANGE_RAD, TINY_HALF_RANGE_RAD, 2)
    Chi = np.linspace(-TINY_HALF_RANGE_RAD, TINY_HALF_RANGE_RAD, 2)
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
