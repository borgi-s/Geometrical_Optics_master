"""G1.6 (roadmap M2): sample remount x oblique geometry interaction.

The S1-S4 remount matrices (crystal/remount.py) act in the CRYSTAL frame
inside Find_Hg; the oblique machinery acts on the lab-frame Bragg geometry
(theta/eta) via the ForwardContext. They are mechanically decoupled at the
input seam (different frames, different code paths), so the combination is
allowed — but until M2 it was completely untested.

The outputs DO interact: the oblique geometry projects the S-rotated strain
field differently onto the detector, which is exactly why the S1-vs-S2
image-difference assertion below works.

This test pins BOTH:
- that a wall-mode forward with a non-trivial remount under the paper oblique
  geometry runs end-to-end on the analytic backend and records the oblique
  attrs AND the remount in provenance; AND
- that the sample_remount S-matrix is demonstrably applied (S2 ≠ S1 images),
  not just recorded.

Marked `slow` (two analytic frames: one S2 run + one S1 control, same
per-frame budget as Gate C).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm_geo.data import configs_root
from dfxm_geo.pipeline import (
    CrystalConfig,
    ScanConfig,
    SimulationConfig,
    WallCrystalConfig,
    run_simulation,
)


def _find_detector(grp: h5py.Group) -> h5py.Dataset | None:
    """Recursively search for the detector dataset (ndim >= 2, 'detector' in name)."""
    for key in grp:
        item = grp[key]
        if isinstance(item, h5py.Dataset) and item.ndim >= 2 and "detector" in item.name:
            return item
        if isinstance(item, h5py.Group):
            found = _find_detector(item)
            if found is not None:
                return found
    return None


def _run_wall_oblique(tmp_path: Path, remount: str) -> tuple[Path, np.ndarray]:
    """Run a single-frame wall+oblique forward with the given remount label.

    Returns (master_h5_path, detector_frame_float64).
    """
    cfg = SimulationConfig.from_toml(configs_root() / "al_oblique_figure3.toml")
    cfg = replace(
        cfg,
        crystal=CrystalConfig(
            mode="wall",
            wall=WallCrystalConfig(dis=4.0, ndis=11, sample_remount=remount),
        ),
        scan=ScanConfig(),  # rocking peak, single frame (Gate C pattern)
        io=replace(cfg.io, write_strain_provenance=False, include_perfect_crystal=False),
    )
    run_simulation(cfg, tmp_path)
    master = tmp_path / "dfxm_geo.h5"
    with h5py.File(master, "r") as f:
        img = _find_detector(f["/1.1"])
        assert img is not None, "could not find detector dataset under /1.1"
        frame = img[0].astype(np.float64)
    return master, frame


@pytest.mark.slow
def test_wall_remount_oblique_forward_e2e(tmp_path: Path) -> None:
    """Wall-mode forward with S2 remount under oblique geometry runs e2e.

    Checks:
    - oblique provenance attrs (geometry_mode, eta, theta) intact under
      wall+remount combination
    - remount provenance recorded at /1.1/sample/sample_remount (the dict key
      used in hdf5.py:886-889 maps directly to the HDF5 dataset name via
      _write_sample_dict)
    - the wall actually diffracts (detector frame is finite, non-negative,
      and has at least one non-zero pixel)
    - the S-matrix is actually APPLIED, not just recorded: S2 and S1 renders
      must differ (a silently-dropped S / identity behaviour would make them
      identical)
    """
    dir_s2 = tmp_path / "s2"
    dir_s1 = tmp_path / "s1"
    dir_s2.mkdir()
    dir_s1.mkdir()

    master_s2, img_s2 = _run_wall_oblique(dir_s2, "S2")
    _, img_s1 = _run_wall_oblique(dir_s1, "S1")

    # --- existing assertions on the S2 run ---
    with h5py.File(master_s2, "r") as f:
        attrs = f["/1.1"].attrs
        # oblique provenance intact under wall+remount
        assert attrs["geometry_mode"] == "oblique"
        assert np.isclose(float(attrs["eta"]), 0.353140, atol=1e-4)
        assert np.isclose(float(attrs["theta"]), np.deg2rad(15.416), atol=1e-3)

        # remount provenance recorded alongside oblique provenance.
        # _write_sample_dict (hdf5.py:484-501) uses the dict key directly as
        # the HDF5 dataset name, so the key "sample_remount" from the sample
        # dict at hdf5.py:889 becomes dataset /1.1/sample/sample_remount.
        samp = f["/1.1/sample"]
        assert "sample_remount" in samp, (
            f"expected /1.1/sample/sample_remount; got keys: {list(samp.keys())}"
        )
        assert samp["sample_remount"][()].decode() == "S2"

    # the wall actually diffracts (not an all-zero frame)
    assert np.isfinite(img_s2).all()
    assert img_s2.min() >= 0.0
    assert img_s2.max() > 0.0

    # --- S1 control: the S-matrix must change the image ---
    # A silently-dropped S (identity behaviour) would make S2 and S1
    # renders identical.
    assert img_s2.shape == img_s1.shape
    assert not np.allclose(img_s2, img_s1), (
        "S2 and S1 remounts produced identical images - the sample_remount "
        "S-matrix is not being applied in the wall+oblique forward path"
    )
