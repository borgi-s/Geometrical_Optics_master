# tests/test_detector_geometry_config.py
"""[detector_geometry] config block: pixel_size/magnification -> object pitch."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from dfxm_geo.config import DetectorGeometryConfig, SimulationConfig


def test_omitted_block_resolves_to_module_defaults():
    g = DetectorGeometryConfig.from_dict(None)
    assert g.object_psize == 40e-9
    assert g.Npixels == 510
    assert g.Nsub == 1


def test_pixel_size_over_magnification_resolves_object_psize():
    g = DetectorGeometryConfig.from_dict(
        {"pixel_size": 0.65e-6, "magnification": 17.31, "Npixels": 510, "Nsub": 1}
    )
    assert math.isclose(g.object_psize, 0.65e-6 / 17.31, rel_tol=0, abs_tol=0)
    assert g.pixel_size == 0.65e-6 and g.magnification == 17.31


def test_pixel_size_without_magnification_raises():
    with pytest.raises(ValueError, match="together"):
        DetectorGeometryConfig.from_dict({"pixel_size": 0.65e-6})


def test_nonpositive_values_raise():
    with pytest.raises(ValueError):
        DetectorGeometryConfig.from_dict({"pixel_size": 0.65e-6, "magnification": -1.0})
    with pytest.raises(ValueError):
        DetectorGeometryConfig.from_dict({"Npixels": 0})


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown"):
        DetectorGeometryConfig.from_dict({"psize": 40e-9})


def test_from_toml_parses_block(tmp_path: Path):
    cfg = tmp_path / "g.toml"
    cfg.write_text(
        "[detector_geometry]\npixel_size = 0.65e-6\nmagnification = 17.31\nNpixels = 120\n",
        encoding="utf-8",
    )
    sc = SimulationConfig.from_toml(cfg)
    assert sc.detector_geometry.Npixels == 120
    assert math.isclose(sc.detector_geometry.object_psize, 0.65e-6 / 17.31)


def test_from_toml_default_when_block_absent(tmp_path: Path):
    cfg = tmp_path / "g.toml"
    cfg.write_text('mode = "single"\n', encoding="utf-8")
    sc = SimulationConfig.from_toml(cfg)
    assert sc.detector_geometry.object_psize == 40e-9
    assert sc.detector_geometry.Npixels == 510
