"""Layer 1 writer tests for dfxm_geo.io.hdf5."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dfxm_geo.io.hdf5 import write_h5_scan


def test_write_h5_scan_creates_image_dataset(tmp_path: Path) -> None:
    images = np.arange(4 * 8 * 8, dtype=np.float64).reshape(4, 8, 8)
    out = tmp_path / "test.h5"
    write_h5_scan(out, scan_id="1.1", images=images)

    with h5py.File(out, "r") as f:
        data = f["/1.1/instrument/dfxm_sim_detector/data"][...]
        assert data.shape == (4, 8, 8)
        assert data.dtype == np.float64
        np.testing.assert_array_equal(data, images)


def test_write_h5_scan_positioners_in_degrees(tmp_path: Path) -> None:
    images = np.zeros((4, 8, 8), dtype=np.float64)
    # phi/chi in RADIANS coming in (matches internal sim convention)
    phi = np.array([-0.001, 0.001, -0.001, 0.001])
    chi = np.array([-0.002, -0.002, 0.002, 0.002])
    out = tmp_path / "test.h5"
    write_h5_scan(out, scan_id="1.1", images=images, phi=phi, chi=chi)

    with h5py.File(out, "r") as f:
        phi_h5 = f["/1.1/instrument/positioners/phi"]
        chi_h5 = f["/1.1/instrument/positioners/chi"]
        # Stored in DEGREES regardless of input units.
        np.testing.assert_allclose(phi_h5[...], np.degrees(phi))
        np.testing.assert_allclose(chi_h5[...], np.degrees(chi))
        assert phi_h5.attrs["units"] == "degree"
        assert chi_h5.attrs["units"] == "degree"


def test_write_h5_scan_measurement_soft_links(tmp_path: Path) -> None:
    images = np.zeros((4, 8, 8), dtype=np.float64)
    phi = np.array([-0.001, 0.001, -0.001, 0.001])
    chi = np.array([-0.002, -0.002, 0.002, 0.002])
    out = tmp_path / "test.h5"
    write_h5_scan(out, scan_id="1.1", images=images, phi=phi, chi=chi)

    with h5py.File(out, "r") as f:
        # Measurement entries should resolve to the same data as instrument/.
        np.testing.assert_array_equal(
            f["/1.1/measurement/dfxm_sim_detector"][...],
            f["/1.1/instrument/dfxm_sim_detector/data"][...],
        )
        np.testing.assert_array_equal(
            f["/1.1/measurement/phi"][...],
            f["/1.1/instrument/positioners/phi"][...],
        )
        np.testing.assert_array_equal(
            f["/1.1/measurement/chi"][...],
            f["/1.1/instrument/positioners/chi"][...],
        )
