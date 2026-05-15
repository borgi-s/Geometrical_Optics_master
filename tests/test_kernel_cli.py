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
        # Defensive: the sidecar must land next to the pickle ONLY, not also in CWD.
        # Mirrors the Round 11 fix that hoisted check_folder("", "pkl_files") out of
        # module-import time.
        assert not (Path.cwd() / "pkl_files" / "Resq_i_explicit_vars.txt").exists()


class TestDefaultConfigReciprocalBlock:
    def test_default_toml_has_reciprocal_block(self) -> None:
        """configs/default.toml must include a `[reciprocal]` block that
        dfxm-bootstrap can drive without any extra args.
        """
        import tomllib

        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.toml"
        with cfg_path.open("rb") as f:
            data = tomllib.load(f)
        assert "reciprocal" in data, "configs/default.toml missing [reciprocal] block"
        recip = data["reciprocal"]
        # CDD_inc canonical recipe (spec §1 + kernel.py defaults).
        assert recip["Nrays"] == int(1e8)
        assert recip["beamstop"] is True
        assert recip["aperture"] is True
        assert recip["knife_edge"] is False
        assert recip["bs_height"] == 25e-3
        # qi ranges (units consistent with generate_kernel kwargs).
        assert recip["qi1_range"] == 5e-4
        assert recip["qi2_range"] == 0.75e-2
        assert recip["qi3_range"] == 0.75e-2
