"""Stage 4.2: space_group on CrystalMount + opt-in extinction filtering."""

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.crystal.oblique import CrystalMount, find_reflections

AL = dict(lattice="cubic", a=4.0495e-10, mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1))


def test_space_group_default_none() -> None:
    m = CrystalMount(**AL)
    assert m.space_group is None


def test_space_group_canonicalized() -> None:
    m = CrystalMount(**AL, space_group="Fm-3m")
    assert m.space_group == "F m -3 m"


def test_space_group_lattice_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        CrystalMount(**AL, space_group="P63/mmc")


def test_bad_symbol_rejected() -> None:
    with pytest.raises(ValueError, match="space-group"):
        CrystalMount(**AL, space_group="Fm-3x")


def test_find_reflections_filters_absences() -> None:
    bare = CrystalMount(**AL)
    fcc = CrystalMount(**AL, space_group="Fm-3m")
    keV = 17.0
    all_refl = find_reflections(bare, keV, hkl_max=2)
    filtered = find_reflections(fcc, keV, hkl_max=2)
    assert len(filtered) < len(all_refl)
    for g in filtered:
        h, k, l = g.hkl
        parities = {h % 2, k % 2, l % 2}
        assert len(parities) == 1, f"mixed-parity {g.hkl} survived Fm-3m filtering"
    # The classic FCC reflection family is still there.
    assert any(set(map(abs, g.hkl)) == {1} for g in filtered)


def test_find_reflections_unchanged_without_space_group() -> None:
    bare = CrystalMount(**AL)
    refl = find_reflections(bare, 17.0, hkl_max=1)
    assert len(refl) > 0
