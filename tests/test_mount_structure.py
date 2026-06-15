"""Stage 4.3a Task 6: structure_type / material / poisson_ratio / slip_families on CrystalMount."""

import pytest

from dfxm_geo.crystal.oblique import CrystalMount


def test_default_mount_resolves_fcc_and_al_nu():
    m = CrystalMount(lattice="cubic", a=4.0495e-10)
    assert m.resolved_structure_type == "fcc"
    assert m.resolved_poisson_ratio == pytest.approx(0.334)


def test_explicit_bcc_and_material():
    m = CrystalMount(lattice="cubic", a=2.8665e-10, structure_type="bcc", material="Fe")
    assert m.resolved_structure_type == "bcc"
    assert m.resolved_poisson_ratio == pytest.approx(0.29, abs=0.005)


def test_poisson_override_wins():
    m = CrystalMount(lattice="cubic", a=2.8665e-10, structure_type="bcc", poisson_ratio=0.31)
    assert m.resolved_poisson_ratio == pytest.approx(0.31)


def test_slip_families_stored_as_tuple():
    m = CrystalMount(
        lattice="cubic", a=2.8665e-10, structure_type="bcc", slip_families=["{110}<111>"]
    )
    assert m.slip_families == ("{110}<111>",)


def test_structure_contradicts_space_group():
    pytest.importorskip("gemmi")
    with pytest.raises(ValueError, match="contradicts"):
        CrystalMount(lattice="cubic", a=3.6e-10, structure_type="bcc", space_group="Fm-3m")


# --- repo-audit #2: ν gate — BCC/HCP without material/poisson_ratio must raise ---


def test_bcc_without_material_or_override_raises():
    """BCC mount with no material and no poisson_ratio override → resolved_poisson_ratio raises."""
    m = CrystalMount(lattice="cubic", a=2.8665e-10, structure_type="bcc")
    with pytest.raises(ValueError, match=r"material"):
        _ = m.resolved_poisson_ratio


def test_hcp_without_material_or_override_raises():
    """HCP mount with no material and no poisson_ratio override → resolved_poisson_ratio raises."""
    # HCP (Ti-like) needs a proper non-cubic mount_x/y/z (2,-1,0)/(0,1,0)/(0,0,1).
    m = CrystalMount(
        lattice="hexagonal",
        a=2.95e-10,
        c=4.68e-10,
        structure_type="hcp",
        mount_x=(2, -1, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )
    with pytest.raises(ValueError, match=r"material"):
        _ = m.resolved_poisson_ratio


def test_bcc_with_material_resolves_table_value():
    """BCC mount with material='Fe' → resolved_poisson_ratio returns Fe's ν (≈0.29)."""
    m = CrystalMount(lattice="cubic", a=2.8665e-10, structure_type="bcc", material="Fe")
    assert m.resolved_poisson_ratio == pytest.approx(0.29, abs=0.005)


def test_bcc_with_override_resolves_without_material():
    """BCC mount with poisson_ratio override (no material) → gate is satisfied."""
    m = CrystalMount(lattice="cubic", a=2.8665e-10, structure_type="bcc", poisson_ratio=0.28)
    assert m.resolved_poisson_ratio == pytest.approx(0.28)


def test_fcc_without_material_resolves_al_default():
    """FCC mount with no material/override → resolved_poisson_ratio returns Al default (0.334).

    The gate must NOT fire for FCC — byte-identity guard.
    """
    m = CrystalMount(lattice="cubic", a=4.0495e-10)  # default → resolved_structure_type='fcc'
    assert m.resolved_poisson_ratio == pytest.approx(0.334)
