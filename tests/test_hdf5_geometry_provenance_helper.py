"""Unit tests for io.hdf5.geometry_provenance_attrs (M2 provenance parity).

The helper is the single source of the per-scan geometry attrs block written
by BOTH write_simulation_h5 (extracted from it, bit-identical) and
write_identification_h5 (new in M2).
"""

from __future__ import annotations

import numpy as np

from dfxm_geo.crystal.oblique import CrystalMount
from dfxm_geo.io.hdf5 import geometry_provenance_attrs


def test_default_mount_is_al_identity() -> None:
    attrs = geometry_provenance_attrs(
        geometry_mode="simplified", eta=0.0, theta_0=0.1567, mount=None
    )
    assert attrs["geometry_mode"] == "simplified"
    assert attrs["eta"] == 0.0
    assert attrs["theta"] == 0.1567
    assert attrs["lattice"] == "cubic"
    assert np.isclose(attrs["a"], 4.0493e-10, rtol=1e-3)  # Al default
    np.testing.assert_array_equal(attrs["mount_x"], [1, 0, 0])
    np.testing.assert_array_equal(attrs["mount_y"], [0, 1, 0])
    np.testing.assert_array_equal(attrs["mount_z"], [0, 0, 1])
    assert attrs["mount_x"].dtype == np.int64


def test_oblique_mount_passthrough() -> None:
    mount = CrystalMount(
        lattice="cubic",
        a=4.0493e-10,
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )
    attrs = geometry_provenance_attrs(
        geometry_mode="oblique", eta=0.353140, theta_0=np.deg2rad(15.417), mount=mount
    )
    assert attrs["geometry_mode"] == "oblique"
    assert np.isclose(attrs["eta"], 0.353140)
    assert np.isclose(attrs["theta"], np.deg2rad(15.417))
    assert np.isclose(attrs["a"], 4.0493e-10)
