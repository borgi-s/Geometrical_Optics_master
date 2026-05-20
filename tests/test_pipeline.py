"""Unit tests for dfxm_geo.pipeline.

Only covers the kernel-independent surface: config parsing/dataclass
defaults and the preflight that surfaces a missing reciprocal-space
kernel before the thread pool fires. End-to-end pipeline tests that
exercise `run_simulation` need a fixture kernel — deferred per the
plan's Phase 7.x note.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dfxm_geo.pipeline import (
    CrystalConfig,
    IOConfig,
    PostprocessConfig,
    ScanConfig,
    SimulationConfig,
    cli_main,
    run_postprocess,
)


class TestSimulationConfigDefaults:
    def test_default_factory_constructs(self) -> None:
        """SimulationConfig() with no args uses the documented defaults."""
        cfg = SimulationConfig()
        assert cfg.crystal.dis == 4.0
        assert cfg.crystal.ndis == 151
        assert cfg.scan.phi_steps == 61
        assert cfg.scan.chi_steps == 61
        # phi_range and chi_range are the same numeric values used in
        # init_forward.py (0.0006 rad and 0.002 rad converted to degrees).
        assert cfg.scan.phi_range == pytest.approx(0.0343774677, rel=1e-6)
        assert cfg.scan.chi_range == pytest.approx(0.1145915590, rel=1e-6)
        assert cfg.io.include_perfect_crystal is True
        assert cfg.io.ftype == ".npy"


class TestSimulationConfigFromToml:
    def test_round_trip_default(self, tmp_path: Path) -> None:
        """Parse the shipped configs/default.toml and check key fields."""
        repo_root = Path(__file__).resolve().parents[1]
        cfg = SimulationConfig.from_toml(repo_root / "configs" / "default.toml")
        assert cfg.crystal.dis == 4
        assert cfg.crystal.ndis == 151
        assert cfg.scan.phi_steps == 61
        assert cfg.scan.chi_steps == 61
        assert cfg.postprocess.enabled is True
        assert cfg.postprocess.chi_oversample == 20

    def test_omitted_optional_sections_use_defaults(self, tmp_path: Path) -> None:
        """[crystal] and [io] are optional; only [scan] is required."""
        p = tmp_path / "minimal.toml"
        p.write_text(
            "[scan]\nphi_range = 0.1\nphi_steps = 10\nchi_range = 0.2\nchi_steps = 20\n"
            "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n",
            encoding="utf-8",
        )
        cfg = SimulationConfig.from_toml(p)
        assert cfg.crystal == CrystalConfig()
        assert cfg.scan == ScanConfig(0.1, 10, 0.2, 20)
        assert cfg.io == IOConfig()

    def test_missing_scan_section_raises(self, tmp_path: Path) -> None:
        """The [scan] section is mandatory."""
        p = tmp_path / "no_scan.toml"
        p.write_text("[crystal]\ndis = 1\nndis = 1\n", encoding="utf-8")
        with pytest.raises(KeyError):
            SimulationConfig.from_toml(p)

    def test_all_shipped_variants_parse(self) -> None:
        """Every dfxm-forward config under configs/ + configs/variants/ parses cleanly.

        Excludes ``identification_*.toml`` — those use a different schema
        consumed by ``load_identification_config``, not ``SimulationConfig``.
        """
        repo_root = Path(__file__).resolve().parents[1]
        configs = [
            p
            for p in list((repo_root / "configs").glob("*.toml"))
            + list((repo_root / "configs" / "variants").glob("*.toml"))
            if not p.name.startswith("identification_")
        ]
        assert len(configs) >= 5, "expected default.toml + at least 4 variants"
        for path in configs:
            cfg = SimulationConfig.from_toml(path)
            assert cfg.crystal.ndis > 0, f"{path.name} has bad ndis"
            assert cfg.scan.phi_steps > 0, f"{path.name} has bad phi_steps"


class TestCrystalConfigSampleRemount:
    """Tests for the sample_remount field on CrystalConfig."""

    def test_default_is_S1(self) -> None:
        from dfxm_geo.pipeline import CrystalConfig

        cfg = CrystalConfig()
        assert cfg.sample_remount == "S1"

    def test_accepts_S2_S3_S4(self) -> None:
        from dfxm_geo.pipeline import CrystalConfig

        for name in ("S2", "S3", "S4"):
            cfg = CrystalConfig(sample_remount=name)
            assert cfg.sample_remount == name

    def test_rejects_unknown_remount(self) -> None:
        from dfxm_geo.pipeline import CrystalConfig

        with pytest.raises(ValueError, match="sample_remount must be one of"):
            CrystalConfig(sample_remount="S99")

    def test_toml_round_trip_with_sample_remount(self, tmp_path: Path) -> None:
        """TOML containing sample_remount round-trips through SimulationConfig."""
        from dfxm_geo.pipeline import SimulationConfig

        toml_text = """
[crystal]
dis = 4.0
ndis = 151
sample_remount = "S3"

[scan]
phi_range = 0.05
phi_steps = 5
chi_range = 0.05
chi_steps = 5

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0
"""
        path = tmp_path / "cfg.toml"
        path.write_text(toml_text, encoding="utf-8")
        cfg = SimulationConfig.from_toml(path)
        assert cfg.crystal.sample_remount == "S3"


class TestPostprocessConfigDefaults:
    def test_default_values(self) -> None:
        pc = PostprocessConfig()
        assert pc.enabled is True
        assert pc.chi_oversample == 20
        assert pc.phi_oversample == 20
        assert pc.chi_oversample_for_shift == 100
        assert pc.figures_dirname == "figures"
        assert pc.data_dirname == "analysis"

    def test_simulation_config_includes_postprocess_field(self) -> None:
        cfg = SimulationConfig()
        assert cfg.postprocess == PostprocessConfig()


class TestPostprocessConfigFromToml:
    def test_section_present(self, tmp_path: Path) -> None:
        p = tmp_path / "with_pp.toml"
        p.write_text(
            "[scan]\nphi_range = 0.1\nphi_steps = 10\nchi_range = 0.2\nchi_steps = 20\n"
            "\n[postprocess]\nenabled = false\nchi_oversample = 5\n"
            "\n[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n",
            encoding="utf-8",
        )
        cfg = SimulationConfig.from_toml(p)
        assert cfg.postprocess.enabled is False
        assert cfg.postprocess.chi_oversample == 5
        # Unspecified keys retain their default
        assert cfg.postprocess.phi_oversample == 20

    def test_section_absent_uses_defaults(self, tmp_path: Path) -> None:
        p = tmp_path / "no_pp.toml"
        p.write_text(
            "[scan]\nphi_range = 0.1\nphi_steps = 10\nchi_range = 0.2\nchi_steps = 20\n"
            "\n[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n",
            encoding="utf-8",
        )
        cfg = SimulationConfig.from_toml(p)
        assert cfg.postprocess == PostprocessConfig()


@pytest.fixture
def tiny_h5_simulation_output(tmp_path: Path) -> tuple[Path, SimulationConfig]:
    """Write a tiny synthetic dfxm_geo.h5 with /1.1 and /2.1 scans.

    Frames are (H, W) = (4, 4). Both scans have the same shape so
    run_postprocess can read and process them with mocked forward().
    """
    import h5py as _h5py

    chi_steps, phi_steps = 5, 5
    H, W = 4, 4
    config = SimulationConfig(
        crystal=CrystalConfig(dis=1.0, ndis=2),
        scan=ScanConfig(phi_range=0.05, phi_steps=phi_steps, chi_range=0.05, chi_steps=chi_steps),
        io=IOConfig(),
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    h5_path = output_dir / "dfxm_geo.h5"

    rng = np.random.default_rng(42)
    n_frames = phi_steps * chi_steps
    dis_stack = rng.normal(1.0, 0.01, size=(n_frames, H, W))
    perf_stack = rng.normal(1.0, 0.01, size=(n_frames, H, W))

    with _h5py.File(h5_path, "w") as f:
        for scan_id, stack in [("1.1", dis_stack), ("2.1", perf_stack)]:
            det = f.require_group(f"/{scan_id}/instrument/dfxm_sim_detector")
            det.create_dataset("data", data=stack)
        # Embed a minimal config_toml so load_h5_scan can parse step counts.
        g = f.require_group("/dfxm_geo")
        g.create_dataset(
            "config_toml",
            data=(
                f"[scan]\nphi_range = 0.05\nphi_steps = {phi_steps}\n"
                f"chi_range = 0.05\nchi_steps = {chi_steps}\n"
            ),
        )
    return output_dir, config


class TestRunPostprocess:
    def test_golden_path(
        self,
        tiny_h5_simulation_output: tuple[Path, SimulationConfig],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import h5py as _h5py

        output_dir, config = tiny_h5_simulation_output

        # Mock forward() to avoid needing the kernel npz.
        fake_qi = np.zeros((3, 4, 4, 4))
        fake_im = np.zeros((4, 4))
        monkeypatch.setattr(
            "dfxm_geo.pipeline.fm.forward",
            lambda Hg, phi=0, chi=0, qi_return=False: (
                (fake_im, fake_qi) if qi_return else (fake_im, None)
            ),
        )
        # Sidestep the geometry globals — provide defaults that match the fake qi shape.
        monkeypatch.setattr("dfxm_geo.pipeline.fm.xl_start", 1e-5)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.yl_start", 1e-5)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.xl_steps", 4)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.yl_steps", 4)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.zl_steps", 4)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.Hg", np.zeros((3, 3, 4, 4, 4)))

        result = run_postprocess(output_dir, config)

        # Analysis stored in /1.1/dfxm_geo/analysis/ inside the .h5
        h5_path = output_dir / "dfxm_geo.h5"
        with _h5py.File(h5_path, "r") as f:
            assert "/1.1/dfxm_geo/analysis/phi_list" in f
            assert "/1.1/dfxm_geo/analysis/chi_list" in f
            assert "/1.1/dfxm_geo/analysis/qi_field" in f
            assert "/1.1/dfxm_geo/analysis/chi_shift_deg" in f
        # Figures still on disk (F1 decision)
        fig_dir = output_dir / "figures"
        assert (fig_dir / "mosaicity_maps.svg").exists()
        assert (fig_dir / "qi_cross_section.svg").exists()
        # Return dict carries the expected keys
        assert "phi_list" in result
        assert "chi_list" in result
        assert "chi_shift" in result
        assert "qi_field" in result
        assert "h5_path" in result
        assert "figures_dir" in result

    def test_missing_h5_raises(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = SimulationConfig()
        with pytest.raises(FileNotFoundError, match="dfxm_geo.h5"):
            run_postprocess(tmp_path, cfg)

    def test_missing_perfect_scan_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """run_postprocess errors clearly if /2.1 (perfect crystal) is absent."""
        import h5py as _h5py

        cfg = SimulationConfig()
        # Create an .h5 with only /1.1 — no /2.1 perfect crystal scan.
        h5_path = tmp_path / "dfxm_geo.h5"
        with _h5py.File(h5_path, "w") as f:
            f.require_group("/1.1")
        with pytest.raises(FileNotFoundError, match="perfect crystal"):
            run_postprocess(tmp_path, cfg)

    def test_missing_hg_raises(
        self,
        tiny_h5_simulation_output: tuple[Path, SimulationConfig],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If fm.Hg is None (no run_simulation, no kernel load),
        run_postprocess should fail fast with a clear error rather than
        crashing inside fm.forward()."""
        output_dir, config = tiny_h5_simulation_output
        monkeypatch.setattr("dfxm_geo.pipeline.fm.Hg", None)
        with pytest.raises(RuntimeError, match="fm.Hg is not set"):
            run_postprocess(output_dir, config)


class TestCliMainFlags:
    def test_default_runs_sim_then_postprocess(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_path = tmp_path / "cfg.toml"
        config_path.write_text(
            "[scan]\nphi_range=0.05\nphi_steps=5\nchi_range=0.05\nchi_steps=5\n"
            "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n",
            encoding="utf-8",
        )
        calls: list[str] = []
        monkeypatch.setattr(
            "dfxm_geo.pipeline.run_simulation",
            lambda cfg, out: calls.append("sim") or {},
        )
        monkeypatch.setattr(
            "dfxm_geo.pipeline.run_postprocess",
            lambda out, cfg: calls.append("pp") or {},
        )

        cli_main(["--config", str(config_path), "--output", str(tmp_path / "out")])
        assert calls == ["sim", "pp"]

    def test_no_postprocess_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_path = tmp_path / "cfg.toml"
        config_path.write_text(
            "[scan]\nphi_range=0.05\nphi_steps=5\nchi_range=0.05\nchi_steps=5\n"
            "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n",
            encoding="utf-8",
        )
        calls: list[str] = []
        monkeypatch.setattr(
            "dfxm_geo.pipeline.run_simulation",
            lambda cfg, out: calls.append("sim") or {},
        )
        monkeypatch.setattr(
            "dfxm_geo.pipeline.run_postprocess",
            lambda out, cfg: calls.append("pp") or {},
        )

        cli_main(
            [
                "--config",
                str(config_path),
                "--output",
                str(tmp_path / "out"),
                "--no-postprocess",
            ]
        )
        assert calls == ["sim"]

    def test_postprocess_only_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_path = tmp_path / "cfg.toml"
        config_path.write_text(
            "[scan]\nphi_range=0.05\nphi_steps=5\nchi_range=0.05\nchi_steps=5\n"
            "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n",
            encoding="utf-8",
        )
        calls: list[str] = []
        monkeypatch.setattr(
            "dfxm_geo.pipeline.run_simulation",
            lambda cfg, out: calls.append("sim") or {},
        )
        monkeypatch.setattr(
            "dfxm_geo.pipeline.run_postprocess",
            lambda out, cfg: calls.append("pp") or {},
        )

        cli_main(
            [
                "--config",
                str(config_path),
                "--output",
                str(tmp_path / "out"),
                "--postprocess-only",
            ]
        )
        assert calls == ["pp"]

    def test_postprocess_disabled_in_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """[postprocess].enabled = false skips the stage even without --no-postprocess."""
        config_path = tmp_path / "cfg.toml"
        config_path.write_text(
            "[scan]\nphi_range=0.05\nphi_steps=5\nchi_range=0.05\nchi_steps=5\n"
            "\n[postprocess]\nenabled = false\n"
            "\n[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n",
            encoding="utf-8",
        )
        calls: list[str] = []
        monkeypatch.setattr(
            "dfxm_geo.pipeline.run_simulation",
            lambda cfg, out: calls.append("sim") or {},
        )
        monkeypatch.setattr(
            "dfxm_geo.pipeline.run_postprocess",
            lambda out, cfg: calls.append("pp") or {},
        )

        cli_main(["--config", str(config_path), "--output", str(tmp_path / "out")])
        assert calls == ["sim"]


class TestDfxmForwardSampleRemountCLI:
    """End-to-end CLI smoke: dfxm-forward with sample_remount=S2."""

    @pytest.mark.slow
    def test_dfxm_forward_with_sample_remount_S2_runs(self, tmp_path: Path) -> None:
        import shutil
        import subprocess
        import sys
        from pathlib import Path as _P

        # Skip if no kernel npz is on disk — glob for the default hkl=(-1,1,-1)
        # at 17 keV, same pattern as _lookup_kernel_path uses.
        import dfxm_geo.direct_space.forward_model as fm

        kernel_dir = _P(fm.pkl_fpath)
        matches = sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz"))
        if not matches:
            pytest.skip(f"No kernel npz found in {kernel_dir}; skipping CLI smoke.")

        # Locate dfxm-forward in the active venv's Scripts/bin directory first,
        # then fall back to shutil.which (works when venv is activated).
        scripts_dir = _P(sys.executable).parent
        venv_cli = scripts_dir / "dfxm-forward"
        if not venv_cli.exists():
            venv_cli = scripts_dir / "dfxm-forward.exe"
        if not venv_cli.exists():
            found = shutil.which("dfxm-forward")
            if found is None:
                pytest.skip("dfxm-forward CLI not found on PATH; skipping CLI smoke.")
            venv_cli = _P(found)

        # Run dfxm-forward with the S2 variant config, output to tmp_path
        repo_root = _P(__file__).resolve().parents[1]
        variant_config = repo_root / "configs" / "variants" / "sample_remount_S2.toml"
        result = subprocess.run(
            [
                str(venv_cli),
                "--config",
                str(variant_config),
                "--output",
                str(tmp_path / "out"),
                "--no-postprocess",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        # CLI should succeed
        assert result.returncode == 0, (
            f"dfxm-forward failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        # Output dir was populated with the HDF5 file
        h5_out = tmp_path / "out" / "dfxm_geo.h5"
        assert h5_out.is_file(), "dfxm_geo.h5 missing from output"


class TestReciprocalConfigParsing:
    """Sub-project D: SimulationConfig + IdentificationConfig parse [reciprocal]."""

    def _write_minimal_sim_toml(self, tmp_path: Path, body: str) -> Path:
        cfg = tmp_path / "config.toml"
        cfg.write_text(body)
        return cfg

    def test_simulation_config_parses_reciprocal_block(self, tmp_path: Path) -> None:
        from dfxm_geo.pipeline import SimulationConfig

        cfg = self._write_minimal_sim_toml(
            tmp_path,
            '[crystal]\ndis = 4\nndis = 151\nsample_remount = "S1"\n'
            "[scan]\nphi_range = 0.034\nphi_steps = 2\nchi_range = 0.115\nchi_steps = 2\n"
            '[io]\nfn_prefix = "/x"\nftype = ".npy"\n'
            'dislocs_dirname = "d"\nperfect_dirname = "p"\ninclude_perfect_crystal = true\n'
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            'figures_dirname = "f"\ndata_dirname = "a"\n'
            "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n",
        )
        config = SimulationConfig.from_toml(cfg)
        assert config.reciprocal is not None
        assert config.reciprocal.hkl == (-1, 1, -1)
        assert config.reciprocal.keV == 17.0

    def test_simulation_config_missing_reciprocal_raises(self, tmp_path: Path) -> None:
        from dfxm_geo.pipeline import SimulationConfig

        cfg = self._write_minimal_sim_toml(
            tmp_path,
            '[crystal]\ndis = 4\nndis = 151\nsample_remount = "S1"\n'
            "[scan]\nphi_range = 0.034\nphi_steps = 2\nchi_range = 0.115\nchi_steps = 2\n"
            '[io]\nfn_prefix = "/x"\nftype = ".npy"\n'
            'dislocs_dirname = "d"\nperfect_dirname = "p"\ninclude_perfect_crystal = true\n'
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            'figures_dirname = "f"\ndata_dirname = "a"\n',
        )
        with pytest.raises(ValueError, match=r"missing \[reciprocal\] block"):
            SimulationConfig.from_toml(cfg)

    def test_simulation_config_missing_hkl_raises(self, tmp_path: Path) -> None:
        from dfxm_geo.pipeline import SimulationConfig

        cfg = self._write_minimal_sim_toml(
            tmp_path,
            '[crystal]\ndis = 4\nndis = 151\nsample_remount = "S1"\n'
            "[scan]\nphi_range = 0.034\nphi_steps = 2\nchi_range = 0.115\nchi_steps = 2\n"
            '[io]\nfn_prefix = "/x"\nftype = ".npy"\n'
            'dislocs_dirname = "d"\nperfect_dirname = "p"\ninclude_perfect_crystal = true\n'
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            'figures_dirname = "f"\ndata_dirname = "a"\n'
            "[reciprocal]\nkeV = 17.0\n",
        )
        with pytest.raises(ValueError, match=r"missing `hkl` in \[reciprocal\]"):
            SimulationConfig.from_toml(cfg)

    def test_simulation_config_missing_keV_raises(self, tmp_path: Path) -> None:
        from dfxm_geo.pipeline import SimulationConfig

        cfg = self._write_minimal_sim_toml(
            tmp_path,
            '[crystal]\ndis = 4\nndis = 151\nsample_remount = "S1"\n'
            "[scan]\nphi_range = 0.034\nphi_steps = 2\nchi_range = 0.115\nchi_steps = 2\n"
            '[io]\nfn_prefix = "/x"\nftype = ".npy"\n'
            'dislocs_dirname = "d"\nperfect_dirname = "p"\ninclude_perfect_crystal = true\n'
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            'figures_dirname = "f"\ndata_dirname = "a"\n'
            "[reciprocal]\nhkl = [-1, 1, -1]\n",
        )
        with pytest.raises(ValueError, match=r"missing `keV` in \[reciprocal\]"):
            SimulationConfig.from_toml(cfg)

    def test_simulation_config_invalid_hkl_propagates_validate_error(self, tmp_path: Path) -> None:
        from dfxm_geo.pipeline import SimulationConfig

        cfg = self._write_minimal_sim_toml(
            tmp_path,
            '[crystal]\ndis = 4\nndis = 151\nsample_remount = "S1"\n'
            "[scan]\nphi_range = 0.034\nphi_steps = 2\nchi_range = 0.115\nchi_steps = 2\n"
            '[io]\nfn_prefix = "/x"\nftype = ".npy"\n'
            'dislocs_dirname = "d"\nperfect_dirname = "p"\ninclude_perfect_crystal = true\n'
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            'figures_dirname = "f"\ndata_dirname = "a"\n'
            "[reciprocal]\nhkl = [0, 0, 0]\nkeV = 17.0\n",
        )
        with pytest.raises(ValueError, match=r"hkl=\(0,0,0\) is not a valid reflection"):
            SimulationConfig.from_toml(cfg)

    def test_identification_config_parses_reciprocal_block(self) -> None:
        from dfxm_geo.pipeline import load_identification_config

        config = load_identification_config(Path("configs/identification_single.toml"))
        assert config.reciprocal is not None
        assert config.reciprocal.hkl == (-1, 1, -1)
        assert config.reciprocal.keV == 17.0

    def test_identification_config_missing_reciprocal_raises(self, tmp_path: Path) -> None:
        from dfxm_geo.pipeline import load_identification_config

        cfg = tmp_path / "identify.toml"
        cfg.write_text(
            'mode = "single"\n'
            "[crystal]\nslip_plane_normal = [1, 1, 1]\n"
            "angle_start_deg = 0.0\nangle_stop_deg = 10.0\nangle_step_deg = 1.0\n"
            "sweep_all_slip_planes = false\nexclude_invisibility = false\n"
            "invisibility_threshold_deg = 10.0\n"
            "[scan]\nphi_rad = 1.5e-4\npoisson_noise = false\n"
            "rng_seed = 0\nintensity_scale = 7.0\n"
            '[io]\nfn_prefix = "/x"\nftype = ".npy"\n'
            'dislocs_dirname = "d"\nperfect_dirname = "p"\ninclude_perfect_crystal = false\n'
        )
        with pytest.raises(ValueError, match=r"missing \[reciprocal\] block"):
            load_identification_config(cfg)
