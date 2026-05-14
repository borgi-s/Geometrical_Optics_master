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

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    CrystalConfig,
    IOConfig,
    PostprocessConfig,
    ScanConfig,
    SimulationConfig,
    _ensure_kernel_loaded,
    cli_main,
    run_postprocess,
    run_simulation,
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
        p.write_text("[scan]\nphi_range = 0.1\nphi_steps = 10\nchi_range = 0.2\nchi_steps = 20\n")
        cfg = SimulationConfig.from_toml(p)
        assert cfg.crystal == CrystalConfig()
        assert cfg.scan == ScanConfig(0.1, 10, 0.2, 20)
        assert cfg.io == IOConfig()

    def test_missing_scan_section_raises(self, tmp_path: Path) -> None:
        """The [scan] section is mandatory."""
        p = tmp_path / "no_scan.toml"
        p.write_text("[crystal]\ndis = 1\nndis = 1\n")
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
"""
        path = tmp_path / "cfg.toml"
        path.write_text(toml_text)
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
        )
        cfg = SimulationConfig.from_toml(p)
        assert cfg.postprocess.enabled is False
        assert cfg.postprocess.chi_oversample == 5
        # Unspecified keys retain their default
        assert cfg.postprocess.phi_oversample == 20

    def test_section_absent_uses_defaults(self, tmp_path: Path) -> None:
        p = tmp_path / "no_pp.toml"
        p.write_text("[scan]\nphi_range = 0.1\nphi_steps = 10\nchi_range = 0.2\nchi_steps = 20\n")
        cfg = SimulationConfig.from_toml(p)
        assert cfg.postprocess == PostprocessConfig()


class TestPreflight:
    def test_raises_when_kernel_not_loaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the resolution kernel, _ensure_kernel_loaded() errors clearly."""
        monkeypatch.setattr(fm, "Resq_i", None)
        with pytest.raises(RuntimeError, match="Reciprocal-space resolution kernel"):
            _ensure_kernel_loaded()

    def test_run_simulation_short_circuits_without_kernel(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """run_simulation() bails via the preflight before the thread pool starts."""
        monkeypatch.setattr(fm, "Resq_i", None)
        with pytest.raises(RuntimeError, match="Reciprocal-space resolution kernel"):
            run_simulation(SimulationConfig(), tmp_path)
        # The output directory was created before the preflight check; that's a
        # minor wart but not worth a Phase 6 follow-up. Just confirm no
        # image-stack subdirectories were created.
        assert not (tmp_path / "images10").exists()
        assert not (tmp_path / "images10_perf_crystal").exists()


class TestRunSimulation:
    def test_golden_path_writes_both_stacks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """run_simulation calls save_images_parallel for both dislocs and
        perfect-crystal directories, and synchronises fm.Hg / fm.q_hkl."""
        config = SimulationConfig(
            crystal=CrystalConfig(dis=2.0, ndis=4),
            scan=ScanConfig(phi_range=0.05, phi_steps=5, chi_range=0.05, chi_steps=5),
            io=IOConfig(),
        )
        output_dir = tmp_path / "out"

        fake_Hg = np.ones((3, 3, 4, 4, 4))
        fake_q = np.array([0.0, 0.0, 1.0])
        find_hg_called = {"count": 0}

        def fake_find_hg(dis, ndis, psize, zl_rms, **kwargs):
            find_hg_called["count"] += 1
            return fake_Hg, fake_q

        save_calls: list[dict] = []

        def fake_save_images_parallel(Hg, *args, **kwargs):
            save_calls.append({"is_zero": bool(np.all(Hg == 0)), "args": args})
            return True

        monkeypatch.setattr("dfxm_geo.pipeline._ensure_kernel_loaded", lambda: None)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.Find_Hg", fake_find_hg)
        monkeypatch.setattr(
            "dfxm_geo.pipeline.save_images_parallel",
            fake_save_images_parallel,
        )
        # forward_model module-level globals expected by the implementation
        monkeypatch.setattr("dfxm_geo.pipeline.fm.psize", 0.1)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.zl_rms", 1.0)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.Hg", None)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.q_hkl", None)

        result = run_simulation(config, output_dir)

        # Find_Hg was called once with the config values
        assert find_hg_called["count"] == 1
        # Two calls to save_images_parallel: dislocs then perfect-crystal
        assert len(save_calls) == 2
        assert save_calls[0]["is_zero"] is False  # dislocs uses real Hg
        assert save_calls[1]["is_zero"] is True  # perfect-crystal uses zeros
        # Module globals are synced to the new Hg
        import dfxm_geo.direct_space.forward_model as _fm

        assert _fm.Hg is fake_Hg
        assert _fm.q_hkl is fake_q
        # Output dir was created and result dict carries the expected keys
        assert output_dir.is_dir()
        assert result["dislocs_path"] == output_dir / config.io.dislocs_dirname
        assert result["perfect_path"] == output_dir / config.io.perfect_dirname
        assert result["Hg"] is fake_Hg

    def test_skip_perfect_crystal(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When io.include_perfect_crystal=False, only one stack is written
        and result['perfect_path'] is None."""
        config = SimulationConfig(
            io=IOConfig(include_perfect_crystal=False),
        )
        output_dir = tmp_path / "out"

        save_count = {"n": 0}

        def fake_save(*a, **kw):
            save_count["n"] += 1
            return True

        monkeypatch.setattr("dfxm_geo.pipeline._ensure_kernel_loaded", lambda: None)
        monkeypatch.setattr(
            "dfxm_geo.pipeline.fm.Find_Hg",
            lambda *a, **kw: (np.ones((3, 3, 4, 4, 4)), np.array([0.0, 0.0, 1.0])),
        )
        monkeypatch.setattr("dfxm_geo.pipeline.save_images_parallel", fake_save)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.psize", 0.1)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.zl_rms", 1.0)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.Hg", None)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.q_hkl", None)

        result = run_simulation(config, output_dir)

        assert save_count["n"] == 1
        assert result["perfect_path"] is None

    def test_run_simulation_passes_resolved_S_to_find_hg(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """sample_remount='S2' must arrive at fm.Find_Hg as the S2 matrix."""
        from dfxm_geo.crystal.remount import S2

        config = SimulationConfig(
            crystal=CrystalConfig(dis=2.0, ndis=4, sample_remount="S2"),
            scan=ScanConfig(phi_range=0.05, phi_steps=5, chi_range=0.05, chi_steps=5),
            io=IOConfig(),
        )

        captured: dict = {}
        fake_Hg = np.ones((3, 3, 4, 4, 4))
        fake_q = np.array([0.0, 0.0, 1.0])

        def fake_find_hg(dis, ndis, psize, zl_rms, **kwargs):
            captured["kwargs"] = kwargs
            return fake_Hg, fake_q

        monkeypatch.setattr("dfxm_geo.pipeline._ensure_kernel_loaded", lambda: None)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.Find_Hg", fake_find_hg)
        monkeypatch.setattr("dfxm_geo.pipeline.save_images_parallel", lambda *a, **k: True)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.psize", 0.1)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.zl_rms", 1.0)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.Hg", None)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.q_hkl", None)

        run_simulation(config, tmp_path / "out")

        assert "S" in captured["kwargs"]
        np.testing.assert_array_equal(captured["kwargs"]["S"], S2)
        assert captured["kwargs"]["remount_name"] == "S2"


@pytest.fixture
def tiny_simulation_output(tmp_path: Path) -> tuple[Path, SimulationConfig]:
    """Write tiny synthetic stacks that mimic save_images_parallel output.

    Names match the (j, i) convention in save_image: ``<prefix><i:04d>_<j:04d>.npy``.
    """
    chi_steps, phi_steps = 5, 5
    H, W = 4, 4
    config = SimulationConfig(
        crystal=CrystalConfig(dis=1.0, ndis=2),
        scan=ScanConfig(phi_range=0.05, phi_steps=phi_steps, chi_range=0.05, chi_steps=chi_steps),
        io=IOConfig(),
    )
    output_dir = tmp_path / "out"
    dislocs_dir = output_dir / config.io.dislocs_dirname
    perfect_dir = output_dir / config.io.perfect_dirname
    dislocs_dir.mkdir(parents=True)
    perfect_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    for i in range(chi_steps):
        for j in range(phi_steps):
            suffix = f"{i:04d}_{j:04d}.npy"
            np.save(
                dislocs_dir / f"{config.io.fn_prefix.lstrip('/')}{suffix}",
                rng.normal(1.0, 0.01, size=(H, W)),
            )
            np.save(
                perfect_dir / f"{config.io.fn_prefix.lstrip('/')}{suffix}",
                rng.normal(1.0, 0.01, size=(H, W)),
            )
    return output_dir, config


class TestRunPostprocess:
    def test_golden_path(
        self,
        tiny_simulation_output: tuple[Path, SimulationConfig],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        output_dir, config = tiny_simulation_output

        # Mock forward() to avoid needing the kernel pickle.
        fake_qi = np.zeros((3, 4, 4, 4))
        fake_im = np.zeros((4, 4))
        monkeypatch.setattr(
            "dfxm_geo.pipeline.fm.forward",
            lambda Hg, phi=0, chi=0, qi_return=False: (
                (fake_im, fake_qi) if qi_return else (fake_im, None)
            ),
        )
        # Bypass the kernel preflight.
        monkeypatch.setattr("dfxm_geo.pipeline._ensure_kernel_loaded", lambda: None)
        # Sidestep the geometry globals — provide defaults that match the fake qi shape.
        monkeypatch.setattr("dfxm_geo.pipeline.fm.xl_start", 1e-5)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.yl_start", 1e-5)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.xl_steps", 4)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.yl_steps", 4)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.zl_steps", 4)
        monkeypatch.setattr("dfxm_geo.pipeline.fm.Hg", np.zeros((3, 3, 4, 4, 4)))

        result = run_postprocess(output_dir, config)

        # Data products
        data_dir = output_dir / "analysis"
        assert (data_dir / "phi_list.npy").exists()
        assert (data_dir / "chi_list.npy").exists()
        assert (data_dir / "qi_field.npy").exists()
        assert (data_dir / "chi_shift_deg.txt").exists()
        # File contents must be a bare float for downstream parsing
        chi_shift_value = float((data_dir / "chi_shift_deg.txt").read_text())
        assert chi_shift_value == result["chi_shift"]
        # Figures
        fig_dir = output_dir / "figures"
        assert (fig_dir / "mosaicity_maps.svg").exists()
        assert (fig_dir / "qi_cross_section.svg").exists()
        # Return dict carries the arrays
        assert "phi_list" in result
        assert "chi_list" in result
        assert "chi_shift" in result
        assert "qi_field" in result
        assert "data_dir" in result
        assert "figures_dir" in result

    def test_missing_dislocs_dir_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("dfxm_geo.pipeline._ensure_kernel_loaded", lambda: None)
        cfg = SimulationConfig()
        with pytest.raises(FileNotFoundError, match="dislocs"):
            run_postprocess(tmp_path, cfg)

    def test_missing_perfect_dir_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """run_postprocess errors clearly if only the dislocs dir exists."""
        monkeypatch.setattr("dfxm_geo.pipeline._ensure_kernel_loaded", lambda: None)
        cfg = SimulationConfig()
        # Create only the dislocs dir; perfect dir is absent.
        (tmp_path / cfg.io.dislocs_dirname).mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="perfect-crystal"):
            run_postprocess(tmp_path, cfg)

    def test_missing_hg_raises(
        self,
        tiny_simulation_output: tuple[Path, SimulationConfig],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If fm.Hg is None (no run_simulation, no kernel auto-load),
        run_postprocess should fail fast with a clear error rather than
        crashing inside fm.forward()."""
        output_dir, config = tiny_simulation_output
        monkeypatch.setattr("dfxm_geo.pipeline._ensure_kernel_loaded", lambda: None)
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
        config_path.write_text("[scan]\nphi_range=0.05\nphi_steps=5\nchi_range=0.05\nchi_steps=5\n")
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
        config_path.write_text("[scan]\nphi_range=0.05\nphi_steps=5\nchi_range=0.05\nchi_steps=5\n")
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
        config_path.write_text("[scan]\nphi_range=0.05\nphi_steps=5\nchi_range=0.05\nchi_steps=5\n")
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

    def test_dfxm_forward_with_sample_remount_S2_runs(self, tmp_path: Path) -> None:
        import subprocess
        import sys as _sys
        from pathlib import Path as _P

        # Skip if the Resq_i kernel pickle is not on disk — same gating pattern
        # as Round 16's CLI smoke. The path mirrors what forward_model loads.
        repo_root = _P(__file__).resolve().parents[1]
        kernel_path = repo_root / "reciprocal_space" / "pkl_files" / "Resq_i_20230913_1308.pkl"
        if not kernel_path.exists():
            pytest.skip(f"Kernel pickle {kernel_path} not present; skipping CLI smoke.")

        # Run dfxm-forward with the S2 variant config, output to tmp_path
        variant_config = repo_root / "configs" / "variants" / "sample_remount_S2.toml"
        out_dir = tmp_path / "out"
        result = subprocess.run(
            [
                _sys.executable,
                "-c",
                "from dfxm_geo.pipeline import cli_main; "
                f"raise SystemExit(cli_main(['--config', r'{variant_config}', '--output', r'{out_dir}', '--no-postprocess']))",
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        # CLI should succeed
        assert result.returncode == 0, (
            f"dfxm-forward failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        # Output dir was populated
        dislocs_dir = tmp_path / "out" / "images10"
        assert dislocs_dir.is_dir(), "dislocs images dir missing"
        assert any(dislocs_dir.iterdir()), "dislocs images dir empty"
