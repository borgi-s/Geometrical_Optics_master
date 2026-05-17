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
