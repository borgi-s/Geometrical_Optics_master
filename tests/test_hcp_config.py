"""M4 Stage 4.3b: 4-index Miller–Bravais config acceptance + HCP lattice guard."""

from __future__ import annotations

import pytest

from dfxm_geo.config import (
    CenteredCrystalConfig,
    GeometryConfig,
    IdentificationConfig,
    IdentificationCrystalConfig,
)
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


# ---------------------------------------------------------------------------
# CenteredCrystalConfig: 4-index Miller–Bravais acceptance (Fix 1, 4.3b review)
# ---------------------------------------------------------------------------


def test_centered_accepts_4index_bnt():
    """4-index b/n/t are converted to 3-index; stored values + validation pass.

    System: basal ⟨a₁⟩ glide in HCP (4-index notation):
      b = [2,-1,-1,0] → uvtw_to_uvw → [1,0,0]
      n = (0, 0, 0,1) → hkil_to_hkl  → (0,0,1)
      t = [-1,2,-1,0] → uvtw_to_uvw  → [0,1,0]

    Geometric check:
      b·n = (1,0,0)·(0,0,1) = 0           ✓
      n×b = (0,0,1)×(1,0,0) = (0,1,0) = t ✓
    """
    cfg = CenteredCrystalConfig(b=(2, -1, -1, 0), n=(0, 0, 0, 1), t=(-1, 2, -1, 0))
    assert cfg.b == (1, 0, 0), f"b not converted: {cfg.b}"
    assert cfg.n == (0, 0, 1), f"n not converted: {cfg.n}"
    assert cfg.t == (0, 1, 0), f"t not converted: {cfg.t}"


def test_centered_accepts_mixed_4index_and_3index():
    """Mixing 4-index n with 3-index b/t is accepted; only 4-index fields convert."""
    # n in 4-index (0001) → (0,0,1); b and t already 3-index
    # valid system: b=(1,0,0), n=(0,0,0,1)→(0,0,1), t=(0,1,0)
    # b·n = 0 ✓; nxb = (0,0,1)×(1,0,0) = (0,1,0) = t ✓
    cfg = CenteredCrystalConfig(b=(1, 0, 0), n=(0, 0, 0, 1), t=(0, 1, 0))
    assert cfg.n == (0, 0, 1)
    assert cfg.b == (1, 0, 0)
    assert cfg.t == (0, 1, 0)


def test_centered_rejects_bad_length_b():
    """b with length != 3 or 4 raises ValueError naming the field."""
    with pytest.raises(ValueError, match="b"):
        CenteredCrystalConfig(b=(1, 0), n=(1, 1, 1), t=(1, -2, 1))


def test_centered_rejects_bad_length_n():
    """n with length != 3 or 4 raises ValueError naming the field."""
    with pytest.raises(ValueError, match="n"):
        CenteredCrystalConfig(b=(1, 0, -1), n=(1, 1, 1, 1, 1), t=(1, -2, 1))


def test_centered_rejects_bad_length_t():
    """t with length != 3 or 4 raises ValueError naming the field."""
    with pytest.raises(ValueError, match="t"):
        CenteredCrystalConfig(b=(1, 0, -1), n=(1, 1, 1), t=(1, 2))


def test_centered_3index_unchanged():
    """3-index inputs are accepted unchanged (FCC/BCC configs unaffected)."""
    cfg = CenteredCrystalConfig(b=(1, 0, -1), n=(1, 1, 1), t=(1, -2, 1))
    assert cfg.b == (1, 0, -1)
    assert cfg.n == (1, 1, 1)
    assert cfg.t == (1, -2, 1)
