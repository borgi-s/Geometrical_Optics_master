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


@pytest.mark.slow
def test_wall_fg_cache_key_uses_run_npixels(tmp_path: Path):
    """Wall-mode Fg cache filename encodes the run's Npixels, not the module global.

    A wall forward with Npixels=120 override must write an Fg cache file whose
    name contains ``px120`` (not ``px510``).  This is the regression test for
    Fixes 1+2: before the fix, the cache key was hardcoded to the module-global
    Npixels, so a [detector_geometry] override would silently thrash or re-use
    the default-geometry cache.

    The test uses dis=7777.0 (unique sentinel) to guarantee the glob is
    attributable to this run and won't be confused with any other pre-existing
    px120 cache file.
    """
    import glob
    import os

    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo.pipeline import SimulationConfig, run_simulation

    # Use a unique dis so the Fg filename is unambiguous.
    DIS_SENTINEL = 7777.0
    DIS_STR = str(DIS_SENTINEL).replace(".", "")  # "77770"

    Fg_dir = fm._REPO_ROOT / "direct_space" / "deformation_gradient_tensors"

    # Clean any pre-existing px120 caches from this sentinel dis to make the
    # assertion meaningful (do NOT touch px510 caches — byte-identity gates
    # depend on those).
    for p in glob.glob(str(Fg_dir / f"Fg_{DIS_STR}_*px120*.npy")):
        os.remove(p)

    # Simplified geometry FCC wall, analytic backend, single rocking-peak
    # frame (scan.phi value=0), Npixels=120 override.
    # dis=7777.0 is chosen as a unique sentinel to avoid cache collisions.
    cfg_toml = (
        "[reciprocal]\n"
        "hkl = [-1, 1, -1]\n"
        "keV = 17.0\n"
        'backend = "analytic"\n'
        "beamstop = false\n"
        "\n"
        "[crystal]\n"
        'mode = "wall"\n'
        "\n"
        "[crystal.wall]\n"
        f"dis = {DIS_SENTINEL}\n"
        "ndis = 3\n"
        'sample_remount = "S1"\n'
        "\n"
        "[scan.phi]\n"
        "value = 0.0\n"
        "\n"
        '[detector]\nmodel = "ideal"\n\n'
        "[detector_geometry]\n"
        "pixel_size = 0.65e-6\n"
        "magnification = 17.31\n"
        "Npixels = 120\n"
        "\n"
        "[io]\n"
        "include_perfect_crystal = false\n"
        "write_strain_provenance = true\n"
        "\n"
        "[postprocess]\n"
        "enabled = false\n"
    )
    cfg_path = tmp_path / "wall_px120.toml"
    cfg_path.write_text(cfg_toml, encoding="utf-8")
    cfg = SimulationConfig.from_toml(cfg_path)
    out = tmp_path / "out"
    run_simulation(cfg, out)

    # (a) Detector image shape must reflect Npixels=120 -> (NN2, NN1) = (120, 40).
    import h5py

    from dfxm_geo.io.hdf5 import DETECTOR_INTERNAL_PATH

    det = out / "scan0001" / "dfxm_sim_detector_0000.h5"
    assert det.is_file(), f"detector h5 not written under {out / 'scan0001'}"
    with h5py.File(det, "r") as f:
        img = f[DETECTOR_INTERNAL_PATH][0]
    assert img.shape == (120, 40), f"Expected (120, 40) for Npixels=120 wall run, got {img.shape}"

    # (b) An Fg cache file whose name contains ``px120`` must have been written,
    # proving the Fg cache key now uses the run's instrument geometry.
    px120_caches = glob.glob(str(Fg_dir / f"Fg_{DIS_STR}_*px120*.npy"))
    assert px120_caches, (
        f"No Fg_*px120*.npy cache written for dis={DIS_SENTINEL}. "
        "Fix 1 regression: cache key still uses the module-global Npixels (510).\n"
        f"Fg_dir={Fg_dir}\n"
        f"All Fg files: {glob.glob(str(Fg_dir / 'Fg_*.npy'))}"
    )
