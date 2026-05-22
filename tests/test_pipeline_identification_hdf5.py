"""End-to-end identify → HDF5 tests (single, multi, z-scan)."""

from __future__ import annotations

from pathlib import Path

import h5py

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


def _minimal_single_cfg() -> IdentificationConfig:
    return IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=(1, 1, 1),
            angle_start_deg=0.0,
            angle_stop_deg=10.0,
            angle_step_deg=10.0,
            b_vector_indices=[0],
            sweep_all_slip_planes=False,
            exclude_invisibility=False,
        ),
        scan=ScanConfig(phi=AxisScanConfig(value=1e-4)),
        noise=IdentificationNoiseConfig(poisson_noise=False),
        io=IOConfig(),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )


def test_single_mode_writes_master_plus_scan_dirs(tmp_path: Path) -> None:
    cfg = _minimal_single_cfg()
    run_identification(cfg, tmp_path)

    master = tmp_path / "dfxm_identify.h5"
    assert master.is_file()
    # 1 plane × 1 b × 2 angles = 2 scans
    assert (tmp_path / "scan0001" / "dfxm_sim_detector_0000.h5").is_file()
    assert (tmp_path / "scan0002" / "dfxm_sim_detector_0000.h5").is_file()

    with h5py.File(master, "r") as f:
        assert sorted(k for k in f if k != "dfxm_geo") == ["1.1", "2.1"]
        for sid in ("1.1", "2.1"):
            scan = f[sid]
            assert scan.attrs["identify_mode"] == "single"
            assert scan.attrs["scan_mode"] == "single"
            assert list(scan.attrs["scanned_axes"]) == []
            samp = scan["sample"]
            assert (
                samp["name"][()]
                .decode()
                .startswith("simulated, dislocation identification (single)")
            )
            assert "slip_plane_normal" in samp
            assert "burgers" in samp
            assert "rotation_deg" in samp
        # Each /N.1 has 1 frame per detector
        assert f["/1.1/instrument/dfxm_sim_detector/data"].shape[0] == 1


def test_single_mode_drops_legacy_sidecars(tmp_path: Path) -> None:
    cfg = _minimal_single_cfg()
    run_identification(cfg, tmp_path)
    assert not (tmp_path / "manifest.csv").exists()
    # No per-plane subdirs left over from the old .npy layout
    for plane_dirname in ("n_1_1_1", "n_1_m1_1", "n_1_1_m1", "n_m1_1_1"):
        assert not (tmp_path / plane_dirname).exists()


def test_multi_mode_writes_master_plus_scan_dirs(tmp_path: Path) -> None:
    cfg = IdentificationConfig(
        mode="multi",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=ScanConfig(phi=AxisScanConfig(value=1e-4)),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        multi=IdentificationMonteCarloConfig(n_samples=3, pos_std_um=5.0),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)
    assert (tmp_path / "dfxm_identify.h5").is_file()
    for k in range(1, 4):
        assert (tmp_path / f"scan{k:04d}" / "dfxm_sim_detector_0000.h5").is_file()
    with h5py.File(tmp_path / "dfxm_identify.h5", "r") as f:
        scan_keys = [k for k in f if k != "dfxm_geo"]
        assert sorted(scan_keys) == ["1.1", "2.1", "3.1"]
        for sid in scan_keys:
            scan = f[sid]
            assert scan.attrs["identify_mode"] == "multi"
            samp = scan["sample"]
            assert "dislocations" in samp
            disl = samp["dislocations"]
            assert disl.attrs["NX_class"] == "NXcollection"
            assert sorted(disl) == ["0", "1"]
            for idx in ("0", "1"):
                d = disl[idx]
                assert d.attrs["NX_class"] == "NXsample"
                assert "slip_plane_normal" in d
                assert "burgers" in d
                assert "rotation_deg" in d
                assert "position_um" in d


def test_zscan_mode_writes_master_plus_scan_dirs(tmp_path: Path) -> None:
    from dfxm_geo.pipeline import IdentificationZScanConfig

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
            phi=AxisScanConfig(range=0.034377, steps=3),
            chi=AxisScanConfig(range=0.114, steps=3),
        ),
        noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
        io=IOConfig(),
        zscan=IdentificationZScanConfig(
            z_offsets_um=[-1.0, 0.0, 1.0],
            include_secondary=False,
        ),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    run_identification(cfg, tmp_path)

    # 3 z-layers × 1 plane × 1 b × 1 alpha = 3 configs
    master = tmp_path / "dfxm_identify.h5"
    assert master.is_file()
    for k in (1, 2, 3):
        assert (tmp_path / f"scan{k:04d}" / "dfxm_sim_detector_0000.h5").is_file()

    with h5py.File(master, "r") as f:
        scan_keys = sorted(k for k in f if k != "dfxm_geo")
        assert scan_keys == ["1.1", "2.1", "3.1"]
        for sid in scan_keys:
            scan = f[sid]
            assert scan.attrs["identify_mode"] == "z-scan"
            assert scan.attrs["scan_mode"] == "mosa"
            samp = scan["sample"]
            assert "z_offset_um" in samp
            assert "primary" in samp
            assert "secondary" not in samp  # include_secondary=False
        # Each /N.1 has phi_steps * chi_steps = 9 frames
        det = f["/1.1/instrument/dfxm_sim_detector/data"]
        assert det.shape[0] == 9
