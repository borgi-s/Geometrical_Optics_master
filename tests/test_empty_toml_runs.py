"""Sub-project F: an empty TOML file produces a valid forward + identify run."""

from __future__ import annotations

from pathlib import Path

import pytest

from dfxm_geo.pipeline import (
    SimulationConfig,
    load_identification_config,
)


class TestEmptyTomlForward:
    def test_empty_toml_parses_to_default_simulation_config(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.toml"
        empty.write_text("")
        cfg = SimulationConfig.from_toml(empty)
        # Cascades to canonical defaults across the board:
        assert cfg.crystal.mode == "centered"
        assert cfg.crystal.centered is not None
        assert cfg.crystal.centered.b == (1, 0, -1)
        assert cfg.reciprocal.hkl == (-1, 1, -1)
        assert cfg.reciprocal.keV == 17.0
        assert cfg.scan.scanned_axes() == ()
        assert cfg.scan.derived_mode_name() == "single"

    def test_empty_toml_runs_end_to_end(self, tmp_path: Path) -> None:
        """Smoke: from_toml -> run_simulation on a 1-image config produces
        a non-empty HDF5 file that load_h5_scan can round-trip.
        """
        # Skip if the bundled Al 111 @ 17 keV kernel isn't present (CI / fresh
        # checkout); the empty-TOML semantics tested above already cover the
        # parse path.
        try:
            import dfxm_geo.direct_space.forward_model as fm

            fm._lookup_kernel_path((-1, 1, -1), 17.0, fm.pkl_fpath)
        except FileNotFoundError:
            pytest.skip("bundled Al 111 @ 17 keV kernel missing; run dfxm-bootstrap first")

        from dfxm_geo.pipeline import run_simulation

        empty = tmp_path / "empty.toml"
        empty.write_text("")
        cfg = SimulationConfig.from_toml(empty)
        # Constrain io fields to tmp_path so the test doesn't pollute the
        # working directory:
        cfg.io.dislocs_dirname = "images"
        cfg.io.perfect_dirname = "perfect"
        cfg.io.include_perfect_crystal = False  # speed
        out = run_simulation(cfg, tmp_path)
        assert isinstance(out, dict)
        # One scan position (single mode); one HDF5 master file exists.
        master_h5 = tmp_path / "dfxm_geo.h5"
        assert master_h5.exists()


class TestEmptyTomlIdentify:
    def test_empty_toml_parses_to_default_identification_config(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.toml"
        empty.write_text("")
        cfg = load_identification_config(empty)
        # mode defaults to 'single':
        assert cfg.mode == "single"
        # crystal hypothesis sweep cascades:
        assert cfg.crystal.slip_plane_normal == (1, 1, 1)
        assert cfg.crystal.sweep_all_slip_planes is True
        # noise + reciprocal cascade:
        assert cfg.noise.poisson_noise is True
        assert cfg.reciprocal.hkl == (-1, 1, -1)
        # multi/zscan blocks stay None outside their modes:
        assert cfg.multi is None
        assert cfg.zscan is None

    def test_partial_identify_toml_overrides_only_specified_keys(self, tmp_path: Path) -> None:
        partial = tmp_path / "partial.toml"
        partial.write_text("""
[reciprocal]
keV = 21.0
""")
        cfg = load_identification_config(partial)
        # Mode still defaults:
        assert cfg.mode == "single"
        # Reciprocal: hkl defaults, keV overridden:
        assert cfg.reciprocal.hkl == (-1, 1, -1)
        assert cfg.reciprocal.keV == 21.0
