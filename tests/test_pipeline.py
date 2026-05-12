"""Unit tests for dfxm_geo.pipeline.

Only covers the kernel-independent surface: config parsing/dataclass
defaults and the preflight that surfaces a missing reciprocal-space
kernel before the thread pool fires. End-to-end pipeline tests that
exercise `run_simulation` need a fixture kernel — deferred per the
plan's Phase 7.x note.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    CrystalConfig,
    IOConfig,
    PostprocessConfig,
    ScanConfig,
    SimulationConfig,
    _ensure_kernel_loaded,
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
