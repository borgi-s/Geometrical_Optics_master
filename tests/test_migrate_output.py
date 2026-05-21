"""Tests for dfxm-migrate-output CLI."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as _fm
from dfxm_geo.io.migrate import migrate_npy_dir_to_h5


def test_migrate_round_trips_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Plant a tiny fixture kernel npz and point forward_model at it. The
    # migrate provenance writer SHAs the kernel and mirrors its scalar
    # params; without this fixture the test fails on fresh checkouts
    # (e.g. a cluster that hasn't run dfxm-bootstrap yet) because the
    # canonical Resq_i_*.npz isn't on disk.
    kernel_dir = tmp_path / "_fixture_kernel"
    kernel_dir.mkdir()
    fake_kernel = kernel_dir / "fake_kernel.npz"
    np.savez(
        fake_kernel,
        Resq_i=np.zeros((2, 2, 2), dtype=np.float64),
        qi1_range=np.float64(1.0),
        qi2_range=np.float64(1.0),
        qi3_range=np.float64(1.0),
        npoints1=np.int64(2),
        npoints2=np.int64(2),
        npoints3=np.int64(2),
    )
    monkeypatch.setattr(_fm, "_loaded_kernel_path", fake_kernel)

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

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    h5_path = out_dir / "dfxm_geo.h5"
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

    # v1.2.0 layout: master + per-scan detector files.
    out_master = out_dir / "dfxm_geo.h5"
    assert out_master.is_file()
    assert (out_dir / "scan0001" / "dfxm_sim_detector_0000.h5").is_file()
    assert (out_dir / "scan0002" / "dfxm_sim_detector_0000.h5").is_file()

    with h5py.File(out_master, "r") as f:
        # ExternalLink follow-through: reads under /1.1/.../data resolve to
        # the per-scan detector file's image stack.
        d1 = f["/1.1/instrument/dfxm_sim_detector/data"][...]
        d2 = f["/2.1/instrument/dfxm_sim_detector/data"][...]
        # Frame order: chi-outer, phi-inner.
        for chi_i in range(2):
            for phi_j in range(2):
                k = chi_i * 2 + phi_j
                np.testing.assert_array_equal(d1[k], np.full((4, 4), chi_i * 10 + phi_j))
                np.testing.assert_array_equal(d2[k], np.full((4, 4), -(chi_i * 10 + phi_j)))
        # Provenance: the fixture kernel was hashed + mirrored.
        assert f["/dfxm_geo/kernel/pkl_fn"][()].decode() == "fake_kernel.npz"
        assert len(f["/dfxm_geo/kernel/sha256"][()].decode()) == 64  # hex SHA256
