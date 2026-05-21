"""Unit tests for the MasterWriter context manager."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dfxm_geo.io.hdf5 import (
    DETECTOR_INTERNAL_PATH,
    MasterWriter,
    _write_detector_file,
)


def _kernel_for_tests() -> Path:
    """Pick the bundled kernel for provenance + lookup tests."""
    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo.pipeline import _lookup_and_load_kernel

    _lookup_and_load_kernel((-1, 1, -1), 17.0)
    assert fm._loaded_kernel_path is not None
    return Path(fm._loaded_kernel_path)


def test_master_writer_open_close_writes_provenance(tmp_path: Path) -> None:
    kernel = _kernel_for_tests()
    master_path = tmp_path / "dfxm_geo.h5"
    with MasterWriter(
        master_path,
        cli="pytest test_master_writer",
        config_toml="[scan]\nphi_range = 0.01\n",
        kernel_npz=kernel,
    ):
        pass  # no add_scan calls

    assert master_path.is_file()
    with h5py.File(master_path, "r") as f:
        assert "/dfxm_geo" in f
        g = f["/dfxm_geo"]
        assert g["cli"][()].decode() == "pytest test_master_writer"
        assert g["config_toml"][()].decode().startswith("[scan]")
        assert "kernel" in g
        assert g["kernel"]["pkl_fn"][()].decode() == kernel.name


def test_master_writer_provenance_written_exactly_once(tmp_path: Path) -> None:
    """Re-entering the context manager on the same file must not double-write."""
    kernel = _kernel_for_tests()
    master_path = tmp_path / "dfxm_geo.h5"
    with MasterWriter(master_path, cli="first", config_toml="", kernel_npz=kernel):
        pass
    # Second open should overwrite cleanly (no duplicate-key errors).
    with MasterWriter(master_path, cli="second", config_toml="", kernel_npz=kernel):
        pass
    with h5py.File(master_path, "r") as f:
        assert f["/dfxm_geo/cli"][()].decode() == "second"


def test_add_scan_writes_external_link_and_metadata(tmp_path: Path) -> None:
    kernel = _kernel_for_tests()
    master_path = tmp_path / "dfxm_geo.h5"
    # Pre-create a detector file the master will link to.
    scan_dir = tmp_path / "scan0001"
    det_path = scan_dir / "dfxm_sim_detector_0000.h5"
    stack = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    _write_detector_file(det_path, stack)

    with MasterWriter(master_path, cli="t", config_toml="", kernel_npz=kernel) as m:
        m.add_scan(
            scan_id="1.1",
            title="fscan2d phi -0.01 0.01 2 chi 0 0 1 1.0",
            start_time="2026-05-21T10:00:00",
            end_time="2026-05-21T10:00:05",
            sample={
                "name": "simulated, dislocation identification (single)",
                "slip_plane_normal": np.asarray([1, 1, 1], dtype=np.int32),
                "burgers": np.asarray([1, 0, 1], dtype=np.int32),
                "rotation_deg": 45.0,
            },
            positioners={"phi": np.array([-0.01, 0.01]), "chi": 0.0},
            detector_links={
                "dfxm_sim_detector": (
                    det_path.relative_to(tmp_path),
                    DETECTOR_INTERNAL_PATH,
                )
            },
            dfxm_geo={
                "Hg": np.eye(3).reshape(1, 3, 3),
                "q_hkl": np.array([0.0, 0.0, 1.0]),
                "theta": 11.5,
                "psize": 7.5e-7,
                "zl_rms": 1.0,
            },
            attrs={
                "scan_mode": "rocking",
                "scanned_axes": ["phi"],
                "identify_mode": "single",
            },
        )

    with h5py.File(master_path, "r") as f:
        assert "1.1" in f
        scan = f["1.1"]
        assert scan.attrs["NX_class"] == "NXentry"
        assert scan.attrs["scan_mode"] == "rocking"
        assert list(scan.attrs["scanned_axes"]) == ["phi"]
        assert scan.attrs["identify_mode"] == "single"
        # Sample metadata
        samp = scan["sample"]
        assert samp.attrs["NX_class"] == "NXsample"
        assert samp["name"][()].decode().startswith("simulated, dislocation")
        np.testing.assert_array_equal(samp["slip_plane_normal"][...], [1, 1, 1])
        np.testing.assert_array_equal(samp["burgers"][...], [1, 0, 1])
        assert samp["rotation_deg"][()] == 45.0
        # Positioners
        pos = scan["instrument"]["positioners"]
        assert pos.attrs["NX_class"] == "NXcollection"
        np.testing.assert_allclose(pos["phi"][...], np.degrees([-0.01, 0.01]))
        assert pos["chi"][()] == 0.0  # scalar fixed axis
        # ExternalLink to detector file resolves transparently
        det_ds = scan["instrument"]["dfxm_sim_detector"]["data"]
        np.testing.assert_array_equal(det_ds[...], stack)
        # And the link target itself is recorded as a relative path
        link = scan["instrument"]["dfxm_sim_detector"].get("data", getlink=True)
        assert isinstance(link, h5py.ExternalLink)
        # Stored as POSIX path for cross-platform HDF5 portability (ESRF tools
        # expect forward-slash separators regardless of host OS).
        assert link.filename == "scan0001/dfxm_sim_detector_0000.h5"


def test_add_scan_multi_detector_links(tmp_path: Path) -> None:
    """render_per_dislocation case: three NXdetector groups per /N.1."""
    kernel = _kernel_for_tests()
    master_path = tmp_path / "dfxm_identify.h5"
    scan_dir = tmp_path / "scan0001"
    detectors = {}
    for name in ("dfxm_sim_detector", "dfxm_sim_detector_dis0", "dfxm_sim_detector_dis1"):
        p = scan_dir / f"{name}_0000.h5"
        _write_detector_file(p, np.zeros((1, 2, 2)))
        detectors[name] = (p.relative_to(tmp_path), DETECTOR_INTERNAL_PATH)

    with MasterWriter(master_path, cli="t", config_toml="", kernel_npz=kernel) as m:
        m.add_scan(
            scan_id="1.1",
            title="t",
            start_time="t",
            end_time="t",
            sample={"name": "x"},
            positioners={"phi": 0.0, "chi": 0.0},
            detector_links=detectors,
            dfxm_geo={},
            attrs={},
        )

    with h5py.File(master_path, "r") as f:
        instr = f["/1.1/instrument"]
        for name in detectors:
            assert name in instr
            assert instr[name].attrs["NX_class"] == "NXdetector"
            # measurement soft-links exist per detector
            assert name in f["/1.1/measurement"]
