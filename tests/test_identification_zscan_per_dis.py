"""render_per_dislocation for z-scan identification.

When a secondary dislocation is present, render_per_dislocation=True writes
two extra noiseless detectors per scan -- the primary and the secondary
rendered in isolation -- mirroring multi mode's `_dis0`/`_dis1` instance
labels. The flag is meaningless without a secondary, so it is rejected at
config time when include_secondary=False.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    AxisScanConfig,
    DetectorConfig,
    IdentificationConfig,
    IdentificationCrystalConfig,
    IdentificationZScanConfig,
    ReciprocalConfig,
    ScanConfig,
    run_identification,
)


def _require_kernel() -> None:
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip(f"no kernel npz found in {kernel_dir}")


def _zscan_cfg(*, render_per_dislocation: bool, include_secondary: bool) -> IdentificationConfig:
    return IdentificationConfig(
        mode="z-scan",
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
            phi=AxisScanConfig(range=6e-4, steps=2),
            chi=AxisScanConfig(range=2e-3, steps=2),
        ),
        detector=DetectorConfig(model="ideal", rng_seed=0),
        zscan=IdentificationZScanConfig(
            z_offsets_um=[0.0],
            include_secondary=include_secondary,
            render_per_dislocation=render_per_dislocation,
        ),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )


def test_render_per_dislocation_requires_secondary() -> None:
    """render_per_dislocation=True without a secondary is a config error."""
    with pytest.raises(ValueError, match="render_per_dislocation"):
        _zscan_cfg(render_per_dislocation=True, include_secondary=False)


def test_zscan_render_per_dislocation_writes_three_detectors(tmp_path: Path) -> None:
    _require_kernel()
    cfg = _zscan_cfg(render_per_dislocation=True, include_secondary=True)
    run_identification(cfg, tmp_path)
    scan_dir = tmp_path / "scan0001"
    assert (scan_dir / "dfxm_sim_detector_0000.h5").is_file()
    assert (scan_dir / "dfxm_sim_detector_primary_0000.h5").is_file()
    assert (scan_dir / "dfxm_sim_detector_secondary_0000.h5").is_file()

    with h5py.File(tmp_path / "dfxm_identify.h5", "r") as f:
        instr = f["/1.1/instrument"]
        for name in (
            "dfxm_sim_detector",
            "dfxm_sim_detector_primary",
            "dfxm_sim_detector_secondary",
        ):
            assert name in instr
            assert instr[name].attrs["NX_class"] == "NXdetector"


def test_zscan_no_per_dislocation_by_default(tmp_path: Path) -> None:
    """Default (render_per_dislocation=False): only the combined detector."""
    _require_kernel()
    cfg = _zscan_cfg(render_per_dislocation=False, include_secondary=True)
    run_identification(cfg, tmp_path)
    scan_dir = tmp_path / "scan0001"
    assert (scan_dir / "dfxm_sim_detector_0000.h5").is_file()
    assert not (scan_dir / "dfxm_sim_detector_primary_0000.h5").exists()
    assert not (scan_dir / "dfxm_sim_detector_secondary_0000.h5").exists()


def test_zscan_per_dislocation_files_are_noiseless(tmp_path: Path) -> None:
    """Per-dislocation files bypass the Poisson pass -> deterministic.

    With Poisson noise ON, the combined detector is stochastic but the
    primary/secondary files must be reproducible across two runs.
    """
    _require_kernel()

    def run(out: Path) -> dict[str, np.ndarray]:
        cfg = IdentificationConfig(
            mode="z-scan",
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
                phi=AxisScanConfig(range=6e-4, steps=2),
                chi=AxisScanConfig(range=2e-3, steps=2),
            ),
            detector=DetectorConfig(rng_seed=7),
            zscan=IdentificationZScanConfig(
                z_offsets_um=[0.0],
                include_secondary=True,
                render_per_dislocation=True,
            ),
            reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
        )
        run_identification(cfg, out)
        result = {}
        for name in ("dfxm_sim_detector_primary", "dfxm_sim_detector_secondary"):
            with h5py.File(out / "scan0001" / f"{name}_0000.h5", "r") as f:
                result[name] = f["/entry_0000/dfxm_sim_detector/image"][...]
        return result

    a = run(tmp_path / "a")
    b = run(tmp_path / "b")
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
