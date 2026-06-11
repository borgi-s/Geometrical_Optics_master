"""G1.6 (roadmap M2): sample remount x oblique geometry interaction.

The S1-S4 remount matrices (crystal/remount.py) act in the CRYSTAL frame
inside Find_Hg; the oblique machinery acts on the lab-frame Bragg geometry
(theta/eta) via the ForwardContext. They are mechanically orthogonal, so the
combination is allowed — but until M2 it was completely untested. This test
pins: a wall-mode forward with a non-trivial remount under the paper oblique
geometry runs end-to-end on the analytic backend and records BOTH the oblique
attrs and the remount in provenance. Marked `slow` (one real analytic frame
at the module-default grid, same budget as Gate C).
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
    """
    cfg = SimulationConfig.from_toml(configs_root() / "al_oblique_figure3.toml")
    cfg = replace(
        cfg,
        crystal=CrystalConfig(
            mode="wall",
            wall=WallCrystalConfig(dis=4.0, ndis=11, sample_remount="S2"),
        ),
        scan=ScanConfig(),  # rocking peak, single frame (Gate C pattern)
        io=replace(cfg.io, write_strain_provenance=False, include_perfect_crystal=False),
    )

    run_simulation(cfg, tmp_path)

    with h5py.File(tmp_path / "dfxm_geo.h5", "r") as f:
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
        img = _find_detector(f["/1.1"])
        assert img is not None, "could not find detector dataset under /1.1"
        img_arr = img[0].astype(np.float64)

    assert np.isfinite(img_arr).all()
    assert img_arr.min() >= 0.0
    assert img_arr.max() > 0.0
