"""Integration tests for write_simulation_h5 + run_simulation."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.crystal.remount import SAMPLE_REMOUNT_OPTIONS
from dfxm_geo.io.hdf5 import write_simulation_h5
from dfxm_geo.pipeline import ScanFrames


@pytest.fixture
def _kernel_loaded() -> None:
    if fm.Hg is None:
        pytest.skip("forward_model kernel not auto-loaded; run dfxm-bootstrap.")


def test_write_simulation_h5_creates_both_scans(tmp_path: Path, _kernel_loaded: None) -> None:
    Hg, q_hkl = fm.Find_Hg(
        4.0,
        151,
        fm.psize,
        fm.zl_rms,
        S=SAMPLE_REMOUNT_OPTIONS["S1"],
        remount_name="S1",
    )
    # 3 phi steps × 1 chi step = 3 frames; phi/chi in radians.
    _phi = np.linspace(-0.0006, 0.0006, 3)
    _chi = np.zeros(1)
    _phi_pf = np.tile(_phi, len(_chi))
    _chi_pf = np.repeat(_chi, len(_phi))
    _n = len(_phi_pf)
    _frames = ScanFrames(
        phi_pf=_phi_pf,
        chi_pf=_chi_pf,
        two_dtheta_pf=np.zeros(_n),
        z_pf=np.zeros(_n),
        n_frames=_n,
    )
    out = tmp_path / "dfxm_geo.h5"
    write_simulation_h5(
        out,
        Hg=Hg,
        q_hkl=q_hkl,
        frames=_frames,
        include_perfect_crystal=True,
        sample_dis=4.0,
        sample_ndis=151,
        sample_remount="S1",
        config_toml="[crystal]\ndis = 4.0\n",
        cli="dfxm-forward --config x.toml --output y/",
    )
    with h5py.File(out, "r") as f:
        # /1.1 = dislocations, /2.1 = perfect crystal
        assert f["/1.1/sample/name"][()].decode() == "simulated, dislocations"
        assert f["/2.1/sample/name"][()].decode() == "simulated, perfect crystal"
        # Hg in /1.1 is the real strain field; in /2.1 it's zeros.
        np.testing.assert_array_equal(f["/1.1/dfxm_geo/Hg"][...], Hg)
        np.testing.assert_array_equal(f["/2.1/dfxm_geo/Hg"][...], np.zeros_like(Hg))
        # Global provenance present.
        assert f["/dfxm_geo/version"][()]
        assert f["/dfxm_geo/config_toml"][()].decode().startswith("[crystal]")


def test_write_simulation_h5_skips_perfect_when_disabled(
    tmp_path: Path, _kernel_loaded: None
) -> None:
    Hg, q_hkl = fm.Find_Hg(
        4.0,
        151,
        fm.psize,
        fm.zl_rms,
        S=SAMPLE_REMOUNT_OPTIONS["S1"],
        remount_name="S1",
    )
    # 3 phi steps × 1 chi step = 3 frames; phi/chi in radians.
    _phi = np.linspace(-0.0006, 0.0006, 3)
    _chi = np.zeros(1)
    _phi_pf = np.tile(_phi, len(_chi))
    _chi_pf = np.repeat(_chi, len(_phi))
    _n = len(_phi_pf)
    _frames = ScanFrames(
        phi_pf=_phi_pf,
        chi_pf=_chi_pf,
        two_dtheta_pf=np.zeros(_n),
        z_pf=np.zeros(_n),
        n_frames=_n,
    )
    out = tmp_path / "dfxm_geo.h5"
    write_simulation_h5(
        out,
        Hg=Hg,
        q_hkl=q_hkl,
        frames=_frames,
        include_perfect_crystal=False,
        sample_dis=4.0,
        sample_ndis=151,
        sample_remount="S1",
        config_toml="x",
        cli="x",
    )
    with h5py.File(out, "r") as f:
        assert "/1.1" in f
        assert "/2.1" not in f


def test_run_postprocess_reads_h5(tmp_path: Path, _kernel_loaded: None) -> None:
    """run_postprocess against an .h5 produces the analysis outputs in the .h5.

    Uses a tiny 3×3 grid (9 frames) to keep memory pressure low; ndis=10 for
    quick Find_Hg, single-threaded (max_workers=1) to avoid OOM in CI.
    """
    from dfxm_geo.pipeline import (
        AxisScanConfig,
        CrystalConfig,
        IOConfig,
        ReciprocalConfig,
        ScanConfig,
        SimulationConfig,
        WallCrystalConfig,
        run_postprocess,
        run_simulation,
    )

    cfg = SimulationConfig(
        crystal=CrystalConfig(
            mode="wall",
            wall=WallCrystalConfig(dis=4.0, ndis=10, sample_remount="S1"),
        ),
        scan=ScanConfig(
            phi=AxisScanConfig(range=0.0006 * 180 / np.pi, steps=3),
            chi=AxisScanConfig(range=0.002 * 180 / np.pi, steps=3),
        ),
        io=IOConfig(include_perfect_crystal=True, max_workers=1),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
    )
    out = tmp_path / "run"
    run_simulation(cfg, out)
    # v1.2.0 layout: master + per-scan detector files written by run_simulation.
    assert (out / "dfxm_geo.h5").is_file()
    assert (out / "scan0001" / "dfxm_sim_detector_0000.h5").is_file()
    if cfg.io.include_perfect_crystal:
        assert (out / "scan0002" / "dfxm_sim_detector_0000.h5").is_file()
    run_postprocess(out, cfg)
    # Outputs land inside the existing .h5
    with h5py.File(out / "dfxm_geo.h5", "r") as f:
        assert "/1.1/dfxm_geo/analysis/phi_list" in f
        assert "/1.1/dfxm_geo/analysis/chi_list" in f
        assert "/1.1/dfxm_geo/analysis/qi_field" in f
        assert "/1.1/dfxm_geo/analysis/chi_shift_deg" in f
    # And SVG figures still go on disk per F1 decision.
    assert (out / "figures" / "mosaicity_maps.svg").exists()
    assert (out / "figures" / "qi_cross_section.svg").exists()
