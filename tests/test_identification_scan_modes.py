"""Verify identify sub-modes consume [scan.phi] / [scan.chi] correctly.

End-to-end tests that exercise the real forward kernel (no monkeypatching
of `fm.forward`), mirroring `test_pipeline_identification_hdf5.py`'s
multi-mode test. Requires the default Resq_i .npz to be present locally;
skipped on CI runners without one (kernel auto-load raises a clear
KeyError in that case).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    AxisScanConfig,
    IdentificationConfig,
    IdentificationCrystalConfig,
    IdentificationMonteCarloConfig,
    IdentificationNoiseConfig,
    IOConfig,
    ReciprocalConfig,
    ScanConfig,
    run_identification,
)


def _require_kernel() -> None:
    """Skip unless a bootstrapped (-1,1,-1) 17 keV kernel npz is on disk."""
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip(f"no kernel npz found in {kernel_dir}")


def test_single_with_phi_scanned_produces_phi_steps_frames(tmp_path: Path) -> None:
    _require_kernel()
    cfg = IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=(1, 1, 1),
            angle_start_deg=0.0,
            angle_stop_deg=0.0,
            angle_step_deg=10.0,
            b_vector_indices=[0],
            sweep_all_slip_planes=False,
            exclude_invisibility=False,
        ),
        scan=ScanConfig(phi=AxisScanConfig(range=0.01, steps=4)),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)
    with h5py.File(tmp_path / "dfxm_identify.h5", "r") as f:
        assert f["/1.1/instrument/dfxm_sim_detector/data"].shape[0] == 4
        assert f["/1.1"].attrs["scan_mode"] == "rocking"
        assert list(f["/1.1"].attrs["scanned_axes"]) == ["phi"]


def test_single_with_z_scanned_emits_one_scanspec_per_z(tmp_path: Path) -> None:
    """[scan.z] in identification single mode produces (n_z * n_planes * n_b * n_alpha) ScanSpecs."""
    _require_kernel()
    cfg = IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=(1, 1, 1),
            angle_start_deg=0.0,
            angle_stop_deg=0.0,
            angle_step_deg=10.0,
            b_vector_indices=[0],
            sweep_all_slip_planes=False,
            exclude_invisibility=False,
        ),
        scan=ScanConfig(
            phi=AxisScanConfig(value=1.5e-4),
            z=AxisScanConfig(range=2.0, steps=3),  # 3 z values
        ),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)
    with h5py.File(tmp_path / "dfxm_identify.h5", "r") as f:
        scan_keys = sorted(k for k in f if k != "dfxm_geo")
        # 3 z * 1 plane * 1 b * 1 alpha = 3 scans
        assert scan_keys == ["1.1", "2.1", "3.1"]
        for sid in scan_keys:
            # Each scan's positioners include a z entry (scalar — z is fixed within a scan)
            assert "z" in f[f"/{sid}/instrument/positioners"]


def test_multi_with_phi_and_chi_scanned_produces_phi_x_chi_frames(
    tmp_path: Path,
) -> None:
    _require_kernel()
    cfg = IdentificationConfig(
        mode="multi",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=ScanConfig(
            phi=AxisScanConfig(range=0.01, steps=3),
            chi=AxisScanConfig(range=0.01, steps=2),
        ),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        multi=IdentificationMonteCarloConfig(n_samples=1, pos_std_um=5.0),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)
    with h5py.File(tmp_path / "dfxm_identify.h5", "r") as f:
        assert f["/1.1/instrument/dfxm_sim_detector/data"].shape[0] == 6
        assert f["/1.1"].attrs["scan_mode"] == "mosa"
        assert sorted(f["/1.1"].attrs["scanned_axes"]) == ["chi", "phi"]
        pos = f["/1.1/instrument/positioners"]
        assert pos["phi"].shape == (6,)
        assert pos["chi"].shape == (6,)
