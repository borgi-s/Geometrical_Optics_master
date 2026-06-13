"""M4 Stage 4.3b: 4-index Miller–Bravais config acceptance + HCP lattice guard."""

from __future__ import annotations

import pytest

from dfxm_geo.config import GeometryConfig, IdentificationConfig, IdentificationCrystalConfig
from dfxm_geo.crystal.oblique import CrystalMount


def test_slip_plane_normal_accepts_4index():
    # (0001) basal in 4-index -> (0,0,1) 3-index, accepted for an HCP mount.
    mount = CrystalMount(
        lattice="hexagonal",
        a=2.951e-10,
        c=4.684e-10,
        structure_type="hcp",
        mount_x=(2, -1, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )
    cfg = IdentificationConfig(
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=(0, 0, 0, 1), sweep_all_slip_planes=False
        ),
        geometry=GeometryConfig(mode="oblique", eta=0.1, mount=mount),
    )
    assert cfg.crystal.slip_plane_normal == (0, 0, 1)


def test_structure_type_hcp_requires_hexagonal_lattice():
    with pytest.raises(ValueError, match="hcp"):
        CrystalMount(lattice="cubic", a=3.0e-10, structure_type="hcp")
