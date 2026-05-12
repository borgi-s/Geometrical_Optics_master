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
        """Every config under configs/ + configs/variants/ parses cleanly."""
        repo_root = Path(__file__).resolve().parents[1]
        configs = list((repo_root / "configs").glob("*.toml")) + list(
            (repo_root / "configs" / "variants").glob("*.toml")
        )
        assert len(configs) >= 5, "expected default.toml + at least 4 variants"
        for path in configs:
            cfg = SimulationConfig.from_toml(path)
            assert cfg.crystal.ndis > 0, f"{path.name} has bad ndis"
            assert cfg.scan.phi_steps > 0, f"{path.name} has bad phi_steps"


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
