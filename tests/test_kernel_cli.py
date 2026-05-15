"""Tests for `dfxm_geo.reciprocal_space.kernel.generate_kernel` and `cli_main`.

Uses Nrays=1000 + a 20**3 grid to keep each test under ~1 s.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class TestGenerateKernelOutputPath:
    def test_writes_to_explicit_path(self, tmp_path: Path) -> None:
        from dfxm_geo.reciprocal_space.kernel import generate_kernel

        out = tmp_path / "subdir" / "Resq_i_test.pkl"
        result_path = generate_kernel(
            Nrays=1000,
            npoints1=20,
            npoints2=20,
            npoints3=20,
            output_path=out,
        )
        assert Path(result_path) == out
        assert out.is_file()
        with out.open("rb") as f:
            arr = pickle.load(f)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (20, 20, 20)

    def test_writes_vars_sidecar_next_to_pickle(self, tmp_path: Path) -> None:
        """The `<stem>_vars.txt` sidecar lands next to the pickle, not in CWD."""
        from dfxm_geo.reciprocal_space.kernel import generate_kernel

        out = tmp_path / "Resq_i_explicit.pkl"
        generate_kernel(Nrays=1000, npoints1=20, npoints2=20, npoints3=20, output_path=out)
        sidecar = tmp_path / "Resq_i_explicit_vars.txt"
        assert sidecar.is_file()
        # Sidecar contains the kwargs (sanity check on serialisation).
        text = sidecar.read_text()
        assert "Nrays" in text
        assert "qi1_range" in text
