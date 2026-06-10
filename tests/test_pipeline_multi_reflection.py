"""Sub-project D integration tests: lookup-driven forward + identify."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from dfxm_geo.pipeline import SimulationConfig, run_simulation


def _make_kernel_npz(
    path: Path,
    hkl: tuple[int, int, int] = (-1, 1, -1),
    keV: float = 17.0,
    include_metadata: bool = True,
) -> Path:
    """Inline copy of the helper from test_kernel_lookup.py to keep this file self-contained."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {
        "Resq_i": np.zeros((4, 4, 4), dtype=np.float64),
        "Nrays": np.int64(1000),
        "npoints1": np.int64(4),
        "npoints2": np.int64(4),
        "npoints3": np.int64(4),
        "qi1_range": np.float64(1e-3),
        "qi2_range": np.float64(1e-3),
        "qi3_range": np.float64(1e-3),
        "zeta_v_fwhm": np.float64(5.3e-4),
        "zeta_h_fwhm": np.float64(0.0),
        "NA_rms": np.float64(3.1e-4),
        "eps_rms": np.float64(6e-5),
        "theta": np.float64(0.165),
        "D": np.float64(5.6e-4),
        "d1": np.float64(0.274),
        "phys_aper": np.float64(2e-3),
        "beamstop": np.bool_(True),
        "bs_height": np.float64(25e-3),
        "aperture": np.bool_(True),
        "knife_edge": np.bool_(False),
        "dphi_range": np.float64(0.0),
    }
    if include_metadata:
        data["hkl"] = np.array(hkl, dtype=np.int64)
        data["keV"] = np.float64(keV)
    np.savez(path, **data)
    return path


class TestForwardMultiReflection:
    @pytest.fixture(autouse=True)
    def _reset_kernel_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reset fm._loaded_kernel_path between tests to avoid cross-test bleed.

        The module-level global persists across tests; monkeypatch restores
        fm.pkl_fpath but not _loaded_kernel_path, so the idempotency guard
        in _lookup_and_load_kernel could (in pathological cases) skip a
        reload it should perform.
        """
        import dfxm_geo.pipeline as p

        # #16 Slice 5: the per-process kernel-path global is gone; idempotency is
        # the module-level _KERNEL_CTX_CACHE, so clear it to avoid cross-test bleed.
        monkeypatch.setattr(p, "_KERNEL_CTX_CACHE", {})

    def test_happy_path_with_explicit_hkl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_simulation looks up the kernel from [reciprocal].hkl/keV, loads it,
        runs the (monkey-patched) forward, records actual loaded path in module state."""
        import dfxm_geo.direct_space.forward_model as fm
        import dfxm_geo.pipeline as p
        from dfxm_geo.pipeline import SimulationConfig

        # Stage a kernel matching (2, 0, 0) @ 17 keV
        kernel_path = _make_kernel_npz(
            tmp_path / "pkl_files" / "Resq_i_h2_k0_l0_17keV_20260520_2014.npz",
            hkl=(2, 0, 0),
            keV=17.0,
        )
        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path / "pkl_files") + os.sep)

        # Monkey-patch the heavy pieces. (save_images_parallel is no longer
        # called directly from pipeline.py post-Phase-9; the only remaining
        # heavy call site is write_simulation_h5.)
        monkeypatch.setattr(fm, "Find_Hg", lambda *a, **k: (np.zeros((4, 4, 4)), np.zeros(3)))
        _captured: dict[str, object] = {}
        monkeypatch.setattr(
            p,
            "write_simulation_h5",
            # #16 Slice 5: the loaded kernel path now rides on the run's ctx
            # (ctx.resolution.loaded_kernel_path), not a module global.
            lambda *a, **k: _captured.__setitem__("path", k["ctx"].resolution.loaded_kernel_path),
        )

        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[crystal]\nmode = "wall"\n[crystal.wall]\n'
            'dis = 4\nndis = 151\nsample_remount = "S1"\n'
            "[scan.phi]\nrange = 0.034\nsteps = 2\n[scan.chi]\nrange = 0.115\nsteps = 2\n"
            '[io]\nfn_prefix = "/x"\nftype = ".npy"\n'
            'dislocs_dirname = "d"\nperfect_dirname = "p"\ninclude_perfect_crystal = false\n'
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            'figures_dirname = "f"\ndata_dirname = "a"\n'
            "[reciprocal]\nhkl = [2, 0, 0]\nkeV = 17.0\n"
        )
        config = SimulationConfig.from_toml(cfg)
        p.run_simulation(config, tmp_path / "out")

        # Verify the actually-loaded path (captured from the run's ctx that
        # write_simulation_h5 received).
        assert _captured["path"] == kernel_path

    def test_lookup_miss_errors_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dfxm_geo.direct_space.forward_model as fm
        import dfxm_geo.pipeline as p
        from dfxm_geo.pipeline import SimulationConfig

        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path / "pkl_files_empty") + os.sep)
        (tmp_path / "pkl_files_empty").mkdir()

        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[crystal]\nmode = "wall"\n[crystal.wall]\n'
            'dis = 4\nndis = 151\nsample_remount = "S1"\n'
            "[scan.phi]\nrange = 0.034\nsteps = 2\n[scan.chi]\nrange = 0.115\nsteps = 2\n"
            '[io]\nfn_prefix = "/x"\nftype = ".npy"\n'
            'dislocs_dirname = "d"\nperfect_dirname = "p"\ninclude_perfect_crystal = false\n'
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            'figures_dirname = "f"\ndata_dirname = "a"\n'
            "[reciprocal]\nhkl = [2, 0, 0]\nkeV = 17.0\n"
        )
        config = SimulationConfig.from_toml(cfg)
        with pytest.raises(KeyError, match=r"no kernel found for hkl=\(2, 0, 0\)"):
            p.run_simulation(config, tmp_path / "out")

    def test_metadata_mismatch_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """File named for (2,0,0) but contains bundled hkl=(-1,1,-1) → ValueError."""
        import dfxm_geo.direct_space.forward_model as fm
        import dfxm_geo.pipeline as p
        from dfxm_geo.pipeline import SimulationConfig

        # Filename says 200 but metadata says -1,1,-1.
        _make_kernel_npz(
            tmp_path / "pkl_files" / "Resq_i_h2_k0_l0_17keV_20260520_2014.npz",
            hkl=(-1, 1, -1),
            keV=17.0,
        )
        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path / "pkl_files") + os.sep)

        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[crystal]\nmode = "wall"\n[crystal.wall]\n'
            'dis = 4\nndis = 151\nsample_remount = "S1"\n'
            "[scan.phi]\nrange = 0.034\nsteps = 2\n[scan.chi]\nrange = 0.115\nsteps = 2\n"
            '[io]\nfn_prefix = "/x"\nftype = ".npy"\n'
            'dislocs_dirname = "d"\nperfect_dirname = "p"\ninclude_perfect_crystal = false\n'
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            'figures_dirname = "f"\ndata_dirname = "a"\n'
            "[reciprocal]\nhkl = [2, 0, 0]\nkeV = 17.0\n"
        )
        config = SimulationConfig.from_toml(cfg)
        with pytest.raises(
            ValueError, match=r"has hkl=\(-1, 1, -1\) but lookup requested hkl=\(2, 0, 0\)"
        ):
            p.run_simulation(config, tmp_path / "out")


class TestRunSimulationCrystalModes:
    """Smoke tests covering all 3 crystal modes via run_simulation."""

    @pytest.fixture(autouse=True)
    def _reset_kernel_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reset fm module-level state between tests."""
        import dfxm_geo.pipeline as p

        # #16 Slice 5: the per-process kernel-path global is gone; idempotency is
        # the module-level _KERNEL_CTX_CACHE, so clear it to avoid cross-test bleed.
        monkeypatch.setattr(p, "_KERNEL_CTX_CACHE", {})

    def _make_kernel(self, tmp_path: Path) -> Path:
        return _make_kernel_npz(
            tmp_path / "pkl_files" / "Resq_i_h-1_k1_l-1_17keV_20260520_0000.npz",
            hkl=(-1, 1, -1),
            keV=17.0,
        )

    def _base_toml(self, mode_block: str) -> str:
        return (
            "[reciprocal]\n"
            "hkl = [-1, 1, -1]\n"
            "keV = 17.0\n"
            "\n"
            "[scan.phi]\n"
            "range = 6e-4\n"
            "steps = 3\n"
            "[scan.chi]\n"
            "range = 2e-3\n"
            "steps = 3\n"
            "\n"
            f"{mode_block}\n"
            "\n"
            "[io]\n"
            "include_perfect_crystal = false\n"
            "\n"
            "[postprocess]\n"
            "enabled = false\n"
        )

    def test_centered_mode_writes_h5(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import dfxm_geo.direct_space.forward_model as fm

        self._make_kernel(tmp_path)
        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path / "pkl_files") + os.sep)

        toml_text = self._base_toml(
            "[crystal]\n"
            'mode = "centered"\n'
            "[crystal.centered]\n"
            "b = [1, -1, 0]\n"
            "n = [1, 1, 1]\n"
            "t = [1, 1, -2]\n"
        )
        cfg_path = tmp_path / "centered.toml"
        cfg_path.write_text(toml_text)
        cfg = SimulationConfig.from_toml(cfg_path)
        out_dir = tmp_path / "out"
        result = run_simulation(cfg, out_dir)
        assert result["h5_path"].exists()
        # v1.2.0 layout: master + per-scan detector file.
        assert (out_dir / "dfxm_geo.h5").is_file()
        assert (out_dir / "scan0001" / "dfxm_sim_detector_0000.h5").is_file()
        # This config has include_perfect_crystal = false → no scan0002.
        assert not (out_dir / "scan0002").exists()
        assert not (out_dir / "dfxm_geo_random_dislocations.json").exists()

    def test_wall_mode_preserves_legacy_behavior(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dfxm_geo.direct_space.forward_model as fm

        self._make_kernel(tmp_path)
        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path / "pkl_files") + os.sep)

        toml_text = self._base_toml(
            "[crystal]\n"
            'mode = "wall"\n'
            "[crystal.wall]\n"
            "dis = 4.0\n"
            "ndis = 151\n"
            'sample_remount = "S1"\n'
        )
        cfg_path = tmp_path / "wall.toml"
        cfg_path.write_text(toml_text)
        cfg = SimulationConfig.from_toml(cfg_path)
        out_dir = tmp_path / "out"
        result = run_simulation(cfg, out_dir)
        assert result["h5_path"].exists()
        # v1.2.0 layout: master + per-scan detector file.
        assert (out_dir / "dfxm_geo.h5").is_file()
        assert (out_dir / "scan0001" / "dfxm_sim_detector_0000.h5").is_file()
        assert not (out_dir / "scan0002").exists()
        assert not (out_dir / "dfxm_geo_random_dislocations.json").exists()

    def test_random_dislocations_mode_writes_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dfxm_geo.direct_space.forward_model as fm

        self._make_kernel(tmp_path)
        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path / "pkl_files") + os.sep)

        toml_text = self._base_toml(
            "[crystal]\n"
            'mode = "random_dislocations"\n'
            "[crystal.random_dislocations]\n"
            "ndis = 2\n"
            "sigma = 3.0\n"
            "seed = 42\n"
        )
        cfg_path = tmp_path / "rd.toml"
        cfg_path.write_text(toml_text)
        cfg = SimulationConfig.from_toml(cfg_path)
        out_dir = tmp_path / "out"
        run_simulation(cfg, out_dir)
        # v1.2.0 layout: master + per-scan detector file.
        assert (out_dir / "dfxm_geo.h5").is_file()
        assert (out_dir / "scan0001" / "dfxm_sim_detector_0000.h5").is_file()
        assert not (out_dir / "scan0002").exists()
        sidecar = out_dir / "dfxm_geo_random_dislocations.json"
        assert sidecar.exists()
        import json

        sidecar_data = json.loads(sidecar.read_text())
        assert sidecar_data["ndis"] == 2
        assert sidecar_data["seed"] == 42
