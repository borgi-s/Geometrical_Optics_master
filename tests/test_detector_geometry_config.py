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


def test_from_config_default_matches_module_globals():
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm

    a = fm.build_instrument_context()  # module globals
    b = fm.build_instrument_context_from_config(psize=40e-9, zl_rms=fm.zl_rms, Npixels=510, Nsub=1)
    for name in (
        "psize",
        "zl_rms",
        "Npixels",
        "Nsub",
        "NN1",
        "NN2",
        "NN3",
        "yl_start",
        "xl_steps",
        "yl_steps",
        "zl_steps",
    ):
        assert getattr(a, name) == getattr(b, name), name
    assert np.array_equal(a.flat_indices, b.flat_indices)
    assert np.array_equal(a.Ud, b.Ud) and np.array_equal(a.Us, b.Us)


def test_from_config_changed_pitch_changes_grid():
    import dfxm_geo.direct_space.forward_model as fm

    b = fm.build_instrument_context_from_config(
        psize=37.6e-9, zl_rms=fm.zl_rms, Npixels=120, Nsub=1
    )
    assert b.Npixels == 120 and b.NN1 == 40 and b.NN2 == 120  # 120//3, 120
    assert b.psize == 37.6e-9
    # yl_start scales with psize*Npixels: -37.6e-9*120/2 + 37.6e-9/2
    assert abs(b.yl_start - (-37.6e-9 * 120 / 2 + 37.6e-9 / 2)) < 1e-20


def test_geometry_context_y_extent_follows_instrument():
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm

    instr = fm.build_instrument_context_from_config(
        psize=37.6e-9, zl_rms=fm.zl_rms, Npixels=120, Nsub=1
    )
    geo = fm.build_geometry_context(0.2, instr)
    # The y-extent of the ray grid must reflect the overridden instrument,
    # not the module default (510*40nm). Max |yl| ~ -instr.yl_start.
    assert abs(float(np.abs(geo.rl[1]).max()) - (-instr.yl_start)) < 1e-12


@pytest.mark.slow
def test_forward_run_honors_npixels_override(tmp_path: Path):
    """A [detector_geometry] Npixels override changes the output image dims."""
    import h5py

    from dfxm_geo.io.hdf5 import DETECTOR_INTERNAL_PATH
    from dfxm_geo.pipeline import SimulationConfig, run_simulation

    toml = (
        "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n"
        'backend = "analytic"\nbeamstop = false\naperture = false\n'
        "zeta_v_fwhm = 5.3e-4\nzeta_h_fwhm = 0.0\n"
        "NA_rms = 3.1106382978723403e-4\neps_rms = 6.0e-5\n\n"
        '[geometry]\nmode = "simplified"\n\n'
        '[crystal]\nmode = "centered"\nlattice = "cubic"\na = 4.05e-10\n\n'
        "[crystal.centered]\nb = [1, 0, -1]\nn = [1, 1, 1]\nt = [1, -2, 1]\n\n"
        "[scan.phi]\nvalue = 1.75e-4\n\n"
        '[detector]\nmodel = "ideal"\n\n'
        "[detector_geometry]\npixel_size = 0.65e-6\nmagnification = 17.31\nNpixels = 120\n\n"
        "[io]\ninclude_perfect_crystal = false\nwrite_strain_provenance = false\n"
    )
    cfg_path = tmp_path / "ov.toml"
    cfg_path.write_text(toml, encoding="utf-8")
    cfg = SimulationConfig.from_toml(cfg_path)
    out = tmp_path / "out"
    run_simulation(cfg, out)
    det = out / "scan0001" / "dfxm_sim_detector_0000.h5"
    with h5py.File(det, "r") as f:
        img = f[DETECTOR_INTERNAL_PATH][0]
    # Npixels=120 -> (NN2, NN1) = (120, 40); default would be (510, 170).
    assert img.shape == (120, 40), img.shape
