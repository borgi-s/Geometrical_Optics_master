"""Layer 1 provenance tests for /dfxm_geo/ root group."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dfxm_geo.io.hdf5 import _write_provenance


def test_write_provenance_basic_fields(tmp_path: Path) -> None:
    out = tmp_path / "test.h5"
    with h5py.File(out, "w") as f:
        _write_provenance(f, cli="dfxm-forward --config x.toml --output out/")

    with h5py.File(out, "r") as f:
        g = f["/dfxm_geo"]
        assert g["version"][()].decode().startswith("1.")
        # git_sha is "unknown" outside a git repo, a 40-char SHA inside one
        sha = g["git_sha"][()].decode()
        assert sha == "unknown" or len(sha) == 40
        assert isinstance(g["git_dirty"][()], (bool, np.bool_))
        assert g["hostname"][()].decode()  # non-empty
        assert g["python_version"][()].decode().startswith("3.")
        assert g["numpy_version"][()].decode()
        assert g["generated_at"][()].decode()  # ISO-ish
        assert g["cli"][()].decode().startswith("dfxm-forward")


def test_write_provenance_kernel_subgroup(tmp_path: Path) -> None:
    npz_path = tmp_path / "Resq_i_test.npz"
    # Fake kernel npz with the expected param keys
    np.savez(
        npz_path,
        Resq_i=np.zeros((4, 4, 4), dtype=np.float64),
        qi1_range=1.0,
        qi2_range=2.0,
        qi3_range=3.0,
        npoints1=4,
        npoints2=4,
        npoints3=4,
        Nrays=int(1e6),
    )
    out = tmp_path / "test.h5"
    with h5py.File(out, "w") as f:
        _write_provenance(f, cli="x", kernel_npz=npz_path)
    with h5py.File(out, "r") as f:
        k = f["/dfxm_geo/kernel"]
        assert k["pkl_fn"][()].decode() == "Resq_i_test.npz"
        sha = k["sha256"][()].decode()
        assert len(sha) == 64  # hex digest
        assert float(k["qi1_range"][()]) == 1.0
        assert int(k["Nrays"][()]) == int(1e6)
