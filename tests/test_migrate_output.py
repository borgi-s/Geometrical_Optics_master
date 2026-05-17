"""Tests for dfxm-migrate-output CLI."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dfxm_geo.io.migrate import migrate_npy_dir_to_h5


def test_migrate_round_trips_images(tmp_path: Path) -> None:
    # Fake a legacy output dir: a few .npy files in images10/.
    images_dir = tmp_path / "images10"
    images_dir.mkdir()
    perf_dir = tmp_path / "images10_perf_crystal"
    perf_dir.mkdir()
    for chi_i in range(2):
        for phi_j in range(2):
            arr = np.full((4, 4), chi_i * 10 + phi_j, dtype=np.float64)
            np.save(images_dir / f"mosa_test_0000_{chi_i:04d}_{phi_j:04d}.npy", arr)
            np.save(perf_dir / f"mosa_test_0000_{chi_i:04d}_{phi_j:04d}.npy", arr * -1)

    h5_path = tmp_path / "dfxm_geo.h5"
    migrate_npy_dir_to_h5(
        npy_dir=tmp_path,
        h5_path=h5_path,
        phi_steps=2,
        chi_steps=2,
        phi_range_deg=0.0006 * 180 / np.pi,
        chi_range_deg=0.002 * 180 / np.pi,
        dis=4.0,
        ndis=151,
        sample_remount="S1",
    )

    with h5py.File(h5_path, "r") as f:
        d1 = f["/1.1/instrument/dfxm_sim_detector/data"][...]
        d2 = f["/2.1/instrument/dfxm_sim_detector/data"][...]
        # Frame order: chi-outer, phi-inner.
        for chi_i in range(2):
            for phi_j in range(2):
                k = chi_i * 2 + phi_j
                np.testing.assert_array_equal(d1[k], np.full((4, 4), chi_i * 10 + phi_j))
                np.testing.assert_array_equal(d2[k], np.full((4, 4), -(chi_i * 10 + phi_j)))
