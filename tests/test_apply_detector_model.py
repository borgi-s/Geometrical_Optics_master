"""Detector-model seam: dataset replacement + orchestrator pass."""

from __future__ import annotations

import tomllib
from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.detector import PCO_EDGE_4P2_ID03
from dfxm_geo.io.hdf5 import (
    DETECTOR_INTERNAL_PATH,
    _write_detector_file,
    replace_detector_image,
)
from dfxm_geo.pipeline import (
    AxisScanConfig,
    DetectorConfig,
    IdentificationConfig,
    IdentificationCrystalConfig,
    IdentificationZScanConfig,
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


def test_replace_detector_image_swaps_dtype_and_keeps_attrs(tmp_path: Path) -> None:
    p = tmp_path / "det_0000.h5"
    _write_detector_file(p, np.ones((3, 8, 8), dtype=np.float32))
    new = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
    with h5py.File(p, "a") as f:
        old_attrs = dict(f[DETECTOR_INTERNAL_PATH].attrs)
        replace_detector_image(f, new, extra_attrs={"detector_model": "pco_edge_4.2_id03"})
        ds = f[DETECTOR_INTERNAL_PATH]
        assert ds.dtype == np.uint16
        assert np.array_equal(ds[...], new)
        assert ds.compression == "gzip"
        assert ds.chunks == (1, 8, 8)
        for k, v in old_attrs.items():
            assert ds.attrs[k] == v  # e.g. the NeXus 'interpretation' attr
        assert ds.attrs["detector_model"] == "pco_edge_4.2_id03"
    # softlinks (NXdata/measurement) must still resolve through the new dataset
    with h5py.File(p, "r") as f:
        assert f[DETECTOR_INTERNAL_PATH].shape == (3, 8, 8)
        # core design property: softlinks resolve through the replaced uint16 dataset
        assert np.array_equal(f["entry_0000/plot/image"][...], new)
        assert np.array_equal(f["entry_0000/measurement"][...], new)


# === post-write detector-model pass (identification single mode) e2e ===


def _run_single(tmp_path: Path, detector_toml: str) -> Path:
    """Run the smallest identification single-mode e2e (config copied from
    test_pipeline_identification_hdf5.test_single_mode_writes_master_plus_scan_dirs,
    swapping in `detector_toml` for its [detector] section), return the
    per-scan combined detector file path."""
    _require_kernel()
    out = tmp_path / "out"
    detector_kw = tomllib.loads(detector_toml)["detector"]
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
        scan=ScanConfig(phi=AxisScanConfig(value=1e-4)),
        detector=DetectorConfig(**detector_kw),
        io=IOConfig(),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, out)
    # 1 plane x 1 b x 1 alpha = 1 scan -> scan0001
    return out / "scan0001" / "dfxm_sim_detector_0000.h5"


@pytest.mark.slow
def test_identify_single_writes_uint16_with_measured_floor(tmp_path: Path) -> None:
    # counts_scale=1.0 (not the 1e4 default): a single dislocation's contrast
    # fills nearly the whole frame, so at the default anchor even the 5th
    # percentile carries ~960 ADU of signal and swamps the floor. A tiny
    # counts_scale gives a genuinely dark-dominated scene that isolates the
    # measured offset floor — which is exactly what this test checks.
    det = _run_single(
        tmp_path, "[detector]\nexposure_time = 1.0\ncounts_scale = 1.0\nrng_seed = 0\n"
    )
    with h5py.File(det, "r") as f:
        ds = f[DETECTOR_INTERNAL_PATH]
        assert ds.dtype == np.uint16
        vals = ds[...].astype(np.float64)
        m = PCO_EDGE_4P2_ID03
        # dark pixels sit at the measured floor: offset(1s) ~ 110 ADU
        dark = np.percentile(vals, 5)
        assert m.offset(1.0) - 10 < dark < m.offset(1.0) + 15
        assert ds.attrs["detector_model"] == "pco_edge_4.2_id03"
        assert ds.attrs["exposure_time"] == 1.0


@pytest.mark.slow
def test_identify_single_ideal_is_float32_passthrough(tmp_path: Path) -> None:
    det = _run_single(tmp_path, '[detector]\nmodel = "ideal"\n')
    with h5py.File(det, "r") as f:
        ds = f[DETECTOR_INTERNAL_PATH]
        assert ds.dtype == np.float32
        assert "detector_model" not in ds.attrs


@pytest.mark.slow
def test_same_seed_reproduces_frames_and_seeds_differ(tmp_path: Path) -> None:
    a = _run_single(tmp_path / "a", "[detector]\nrng_seed = 1\n")
    b = _run_single(tmp_path / "b", "[detector]\nrng_seed = 1\n")
    c = _run_single(tmp_path / "c", "[detector]\nrng_seed = 2\n")
    with h5py.File(a) as fa, h5py.File(b) as fb, h5py.File(c) as fc:
        ia, ib, ic = (f[DETECTOR_INTERNAL_PATH][...] for f in (fa, fb, fc))
    assert np.array_equal(ia, ib)
    assert not np.array_equal(ia, ic)


@pytest.mark.slow
def test_per_dislocation_labels_stay_noiseless_while_combined_is_uint16(
    tmp_path: Path,
) -> None:
    """Combined file gets the real model (uint16); per-dislocation instance
    label files (`_primary_`, `_secondary_`) stay noiseless float32."""
    _require_kernel()
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
        detector=DetectorConfig(rng_seed=0),
        zscan=IdentificationZScanConfig(
            z_offsets_um=[0.0],
            include_secondary=True,
            render_per_dislocation=True,
        ),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)
    scan_dir = tmp_path / "scan0001"

    with h5py.File(scan_dir / "dfxm_sim_detector_0000.h5", "r") as f:
        assert f[DETECTOR_INTERNAL_PATH].dtype == np.uint16
        assert f[DETECTOR_INTERNAL_PATH].attrs["detector_model"] == "pco_edge_4.2_id03"
    for name in ("dfxm_sim_detector_primary", "dfxm_sim_detector_secondary"):
        with h5py.File(scan_dir / f"{name}_0000.h5", "r") as f:
            ds = f[DETECTOR_INTERNAL_PATH]
            assert ds.dtype == np.float32
            assert "detector_model" not in ds.attrs
