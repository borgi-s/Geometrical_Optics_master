"""DarlingReader: resolve external-linked detector data for the darling library.

Builds a minimal master + per-scan HDF5 (no kernel needed) and checks that
(a) resolve_detector_data follows the ExternalLink, and (b) a real
darling.DataSet can load the scan through DarlingReader — which is the whole
point, since darling's own readers use visititems and cannot cross the link.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm_geo.io.darling_reader import DarlingReader, resolve_detector_data
from dfxm_geo.io.hdf5 import (
    DETECTOR_INTERNAL_PATH,
    MasterWriter,
    _write_detector_file,
    replace_detector_image,
)

# Frame layout: phi inner (2 steps), chi outer (3 steps) -> 6 frames.
PHI_STEPS, CHI_STEPS = 2, 3
H, W = 5, 5
N_FRAMES = PHI_STEPS * CHI_STEPS


def _build_master(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Write a tiny master+per-scan HDF5 with varying phi/chi motors."""
    scan_dir = tmp_path / "scan0001"
    det_path = scan_dir / "dfxm_sim_detector_0000.h5"
    stack = np.arange(N_FRAMES * H * W, dtype=np.float64).reshape(N_FRAMES, H, W)
    _write_detector_file(det_path, stack)

    # phi inner / chi outer: frame k = chi_idx * PHI_STEPS + phi_idx
    phi = np.array([0.0, 1e-3] * CHI_STEPS)  # 2 distinct values
    chi = np.repeat([0.0, 1e-3, 2e-3], PHI_STEPS)  # 3 distinct values

    master_path = tmp_path / "dfxm_geo.h5"
    cfg = (
        f"[scan.phi]\nrange = 0.01\nsteps = {PHI_STEPS}\n"
        f"[scan.chi]\nrange = 0.01\nsteps = {CHI_STEPS}\n"
    )
    with MasterWriter(master_path, cli="t", config_toml=cfg, kernel_npz=None) as m:
        m.add_scan(
            scan_id="1.1",
            title="t",
            start_time="t",
            end_time="t",
            sample={"name": "x"},
            positioners={"phi": phi, "chi": chi},
            detector_links={
                "dfxm_sim_detector": (
                    Path("scan0001") / "dfxm_sim_detector_0000.h5",
                    DETECTOR_INTERNAL_PATH,
                )
            },
            dfxm_geo={},
            attrs={},
        )
    return master_path, stack


def test_resolve_detector_data_follows_external_link(tmp_path: Path) -> None:
    master_path, stack = _build_master(tmp_path)
    data = resolve_detector_data(master_path, scan_id="1.1")
    assert data.shape == (N_FRAMES, H, W)
    np.testing.assert_array_equal(data, stack)


def test_resolve_detector_data_roi(tmp_path: Path) -> None:
    master_path, stack = _build_master(tmp_path)
    data = resolve_detector_data(master_path, scan_id="1.1", roi=(1, 4, 0, 2))
    assert data.shape == (N_FRAMES, 3, 2)
    np.testing.assert_array_equal(data, stack[:, 1:4, 0:2])


def test_reader_returns_darling_shaped_arrays(tmp_path: Path) -> None:
    master_path, _ = _build_master(tmp_path)
    reader = DarlingReader(master_path)
    data, motors = reader("1.1")
    # data: (a, b, m, n) = (H, W, scan dims), scan dims multiply to N_FRAMES.
    assert data.shape[:2] == (H, W)
    assert data.shape[2] * data.shape[3] == N_FRAMES
    assert data.dtype == np.float32
    # motors: (k, m, n), k=2 (phi, chi both vary).
    assert motors.shape == (2, data.shape[2], data.shape[3])
    assert motors.dtype == np.float32


def test_reader_preserves_frame_motor_correspondence(tmp_path: Path) -> None:
    """Every detector frame must stay paired with its own motor values."""
    master_path, stack = _build_master(tmp_path)
    reader = DarlingReader(master_path)
    data, motors = reader("1.1")
    a, b, m, n = data.shape
    # Each (i,j) cell's frame must equal some original frame whose stored
    # phi/chi (in degrees, float32 per darling's contract) match the motor
    # grids at that cell.
    phi_deg = np.degrees([0.0, 1e-3] * CHI_STEPS).astype(np.float32)
    chi_deg = np.degrees(np.repeat([0.0, 1e-3, 2e-3], PHI_STEPS)).astype(np.float32)
    for i in range(m):
        for j in range(n):
            frame = data[:, :, i, j]
            # find which original frame this is
            matches = [k for k in range(N_FRAMES) if np.array_equal(frame, stack[k])]
            assert len(matches) == 1, "frame not uniquely recovered"
            k = matches[0]
            cell_motors = motors[:, i, j]
            assert np.any(np.isclose(cell_motors, phi_deg[k]))
            assert np.any(np.isclose(cell_motors, chi_deg[k]))


def test_reader_casts_uint16_source_to_float32(tmp_path: Path) -> None:
    """A uint16-source detector (post detector-model) still loads as float32.

    The detector model writes uint16 ADU frames; DarlingReader's contract is
    a float32 stack, so it must cast on load (else darling sees uint16).
    """
    master_path, _ = _build_master(tmp_path)
    det_path = tmp_path / "scan0001" / "dfxm_sim_detector_0000.h5"
    uint16_stack = np.arange(N_FRAMES * H * W, dtype=np.uint16).reshape(N_FRAMES, H, W)
    with h5py.File(det_path, "a") as f:
        replace_detector_image(f, uint16_stack)
        assert f[DETECTOR_INTERNAL_PATH].dtype == np.uint16  # source is genuinely uint16

    reader = DarlingReader(master_path)
    data, motors = reader("1.1")
    assert data.dtype == np.float32
    assert data.shape[:2] == (H, W)
    assert data.shape[2] * data.shape[3] == N_FRAMES


def test_no_varying_motors_raises(tmp_path: Path) -> None:
    """A fixed (single-image) scan has no scan motors -> clear error."""
    scan_dir = tmp_path / "scan0001"
    det_path = scan_dir / "dfxm_sim_detector_0000.h5"
    _write_detector_file(det_path, np.zeros((1, H, W)))
    master_path = tmp_path / "dfxm_geo.h5"
    with MasterWriter(master_path, cli="t", config_toml="", kernel_npz=None) as mw:
        mw.add_scan(
            scan_id="1.1",
            title="t",
            start_time="t",
            end_time="t",
            sample={"name": "x"},
            positioners={"phi": 0.0, "chi": 0.0},  # scalars -> fixed
            detector_links={
                "dfxm_sim_detector": (
                    Path("scan0001") / "dfxm_sim_detector_0000.h5",
                    DETECTOR_INTERNAL_PATH,
                )
            },
            dfxm_geo={},
            attrs={},
        )
    with pytest.raises(ValueError, match="no varying motors"):
        DarlingReader(master_path)("1.1")


def test_real_darling_dataset_loads_through_reader(tmp_path: Path) -> None:
    """End-to-end: a real darling.DataSet loads the scan via DarlingReader.

    This is the regression that matters: darling's built-in readers can't
    discover our detector data (visititems doesn't cross ExternalLinks), but
    DarlingReader hands it pre-resolved arrays.
    """
    darling = pytest.importorskip("darling")
    master_path, _ = _build_master(tmp_path)
    dset = darling.DataSet(DarlingReader(master_path))
    dset.load_scan("1.1")
    # darling stores the loaded stack on .data with detector dims first.
    assert dset.data.shape[:2] == (H, W)
    assert dset.data.shape[2] * dset.data.shape[3] == N_FRAMES
