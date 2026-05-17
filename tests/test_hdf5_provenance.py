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
