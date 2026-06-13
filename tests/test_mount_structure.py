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
