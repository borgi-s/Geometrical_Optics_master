"""Slip-system registry + enumerator tests."""

import numpy as np
import pytest

from dfxm_geo.crystal.cell import UnitCell
from dfxm_geo.crystal.slip_systems import (
    burgers_in_plane,
    burgers_in_plane_int,
    burgers_magnitude,
    plane_normals,
    slip_systems,
)


def _canon(v):
    """Canonical-sign integer direction (first nonzero positive) for set compare."""
    v = tuple(int(round(x)) for x in v)
    for c in v:
        if c != 0:
            return v if c > 0 else tuple(-x for x in v)
    return v


# The hand-authored v2.x FCC slip table (forward_model._SLIP_SYSTEM_111), kept
# here verbatim as the bit-identity oracle. slip_systems("fcc") must be
# ORDER-identical to this (b, n, t) sequence so random_dislocations RNG draws
# + the wall first-system stay byte-identical to v2.x. The table itself was
# deleted from forward_model.py in M4 Stage 4.3a Task 8.
_LEGACY_FCC_SLIP_111 = (
    ((1, -1, 0), (1, 1, 1), (1, 1, -2)),
    ((-1, 0, 1), (1, 1, 1), (-1, 2, -1)),
    ((0, 1, -1), (1, 1, 1), (-2, 1, 1)),
    ((1, 1, 0), (1, -1, 1), (1, -1, -2)),
    ((-1, 0, 1), (1, -1, 1), (-1, -2, -1)),
    ((0, -1, -1), (1, -1, 1), (-2, -1, 1)),
    ((0, 1, -1), (-1, 1, 1), (-2, -1, -1)),
    ((1, 0, 1), (-1, 1, 1), (1, 2, -1)),
    ((1, 1, 0), (-1, 1, 1), (-1, 1, -2)),
    ((0, 1, 1), (1, 1, -1), (2, -1, 1)),
    ((1, -1, 0), (1, 1, -1), (-1, -1, -2)),
    ((1, 0, 1), (1, 1, -1), (1, -2, -1)),
)


def test_fcc_counts_and_geometry():
    sys = slip_systems("fcc")
    assert len(sys) == 12
    assert len({_canon(s.n) for s in sys}) == 4  # 4 distinct {111} planes
    for s in sys:
        assert np.dot(s.b, s.n) == 0  # glide: Burgers in plane
        # t collinear with n x b (UP TO SIGN). The FCC table is the legacy
        # _SLIP_SYSTEM_111 verbatim, whose `t` is the physical line direction —
        # sometimes -(n x b). _ud_matrix_from_bnt is sign-invariant in t, so the
        # Ud (the only thing forward output depends on) is identical either way.
        assert np.allclose(np.cross(np.cross(s.n, s.b), s.t), 0.0)  # t ∥ n x b


def test_bcc_default_is_110_plus_112():
    sys = slip_systems("bcc")
    assert len(sys) == 24  # {110}<111> 12 + {112}<111> 12
    assert len({_canon(s.n) for s in sys}) == 18  # 6 {110} + 12 {112}
    for s in sys:
        assert np.dot(s.b, s.n) == 0
        assert np.allclose(np.cross(s.n, s.b), s.t)


def test_bcc_single_family_selection():
    assert len(slip_systems("bcc", families=["{110}<111>"])) == 12
    assert len(slip_systems("bcc", families=["{112}<111>"])) == 12


def test_fcc_registry_equals_legacy_slip_table():
    """The registry must reproduce the hand-written legacy FCC slip set
    (up to sign) — the {111}<110> family is complete (nothing dropped)."""
    legacy = {(_canon(b), _canon(n)) for b, n, _t in _LEGACY_FCC_SLIP_111}
    reg = {(_canon(s.b), _canon(s.n)) for s in slip_systems("fcc")}
    assert reg == legacy


def test_fcc_slip_systems_ordered_equals_legacy():
    """slip_systems('fcc') must be ORDER-identical to the legacy _SLIP_SYSTEM_111
    (b,n,t) so random_dislocations RNG draws + wall first-system stay bit-identical."""
    got = slip_systems("fcc")
    assert tuple((s.b, s.n, s.t) for s in got) == _LEGACY_FCC_SLIP_111


def test_fcc_ordered_table_is_complete_111_110_family():
    """The explicit ordered FCC table must SET-equal the symmetry enumerator
    output of SlipFamily('{111}<110>', (1,1,1), (1,1,0)) — proving nothing was
    dropped relative to the full {111}<110> glide family."""
    from dfxm_geo.crystal.slip_systems import SlipFamily, _enumerate_family

    enumerated = _enumerate_family(SlipFamily("{111}<110>", (1, 1, 1), (1, 1, 0)))
    enum_set = {(_canon(s.b), _canon(s.n)) for s in enumerated}
    table_set = {(_canon(b), _canon(n)) for b, n, _t in _LEGACY_FCC_SLIP_111}
    assert table_set == enum_set
    assert len(_LEGACY_FCC_SLIP_111) == len(enumerated) == 12


def test_plane_normals_distinct():
    assert len(plane_normals("fcc")) == 4
    assert len(plane_normals("bcc")) == 18


def test_unknown_structure_raises():
    with pytest.raises(ValueError, match="unknown structure"):
        slip_systems("diamond")


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="not defined"):
        slip_systems("fcc", families=["{110}<111>"])


# ---------------------------------------------------------------------------
# Task 2: burgers_in_plane + burgers_magnitude
# ---------------------------------------------------------------------------


def test_burgers_in_plane_fcc_matches_legacy_basis():
    """For each {111} plane, the 6 unit Burgers match the pre-branch
    _BASIS_TABLE / sqrt(2) (FCC bit-identity)."""
    legacy = {
        (1, 1, 1): np.array([[-1, 1, 0], [1, 0, -1], [0, 1, -1]], float),
        (1, -1, 1): np.array([[1, 1, 0], [1, 0, -1], [0, 1, 1]], float),
        (1, 1, -1): np.array([[1, -1, 0], [1, 0, 1], [0, -1, -1]], float),
        (-1, 1, 1): np.array([[-1, -1, 0], [-1, 0, -1], [0, 1, -1]], float),
    }
    for plane, basis in legacy.items():
        got = burgers_in_plane("fcc", plane)
        want = np.vstack([basis, -basis]) / np.sqrt(2)
        gs = sorted(tuple(np.round(v, 9)) for v in got)
        ws = sorted(tuple(np.round(v, 9)) for v in want)
        assert np.allclose(gs, ws)


def test_burgers_in_plane_bcc_unit_and_count():
    got110 = burgers_in_plane("bcc", (1, 1, 0))
    for v in got110:
        assert np.isclose(np.linalg.norm(v), 1.0)
    # {110} plane hosts 2 distinct <111> Burgers -> 4 with negatives
    # (the bcc default registry includes {112}<111> too, but those have
    # different planes; for the (1,1,0) plane only <111> in-plane count).
    assert got110.shape[0] == 4


def test_burgers_in_plane_int_aligns_with_unit_and_is_integer_fcc():
    """FCC {111}: burgers_in_plane_int row i is the INTEGER ⟨110⟩ direction whose
    unit-normalization equals burgers_in_plane row i (index alignment), i.e.
    int_i == round(unit_i * √2). Integer entries are exact (no rounding error)."""
    for plane in ((1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, 1, 1)):
        unit = burgers_in_plane("fcc", plane)
        ints = burgers_in_plane_int("fcc", plane)
        assert ints.dtype.kind in ("i", "u")
        assert ints.shape == unit.shape
        # FCC ⟨110⟩: |b_int| == √2, so int == unit * √2 row-for-row.
        assert np.allclose(ints, unit * np.sqrt(2))
        # Components are exactly integral (±1/0 for ⟨110⟩).
        assert np.array_equal(ints, np.rint(ints).astype(int))


def test_burgers_in_plane_int_bcc_is_111_family():
    """BCC {110}: integer Burgers are ⟨111⟩ (NOT ⟨110⟩×√2). Rows align with
    burgers_in_plane (each unit row × √3 == the integer ⟨111⟩ row)."""
    plane = (1, 1, 0)
    unit = burgers_in_plane("bcc", plane)
    ints = burgers_in_plane_int("bcc", plane)
    assert ints.dtype.kind in ("i", "u")
    assert ints.shape == unit.shape
    # Every integer row is a ⟨111⟩ variant: all |components| == 1.
    for v in ints:
        assert sorted(abs(int(c)) for c in v) == [1, 1, 1]
    # |⟨111⟩| == √3, so int == unit * √3 row-for-row (alignment with unit table).
    assert np.allclose(ints, unit * np.sqrt(3))


def test_burgers_magnitude_fcc_al_exact():
    al = UnitCell.cubic(4.0495e-10)
    assert burgers_magnitude("fcc", "{111}<110>", al) == pytest.approx(2.862e-4, rel=1e-3)


def test_burgers_magnitude_bcc_fe():
    fe = UnitCell.cubic(2.8665e-10)
    expected_um = 2.8665e-10 * np.sqrt(3) / 2 * 1e6
    assert burgers_magnitude("bcc", "{110}<111>", fe) == pytest.approx(expected_um, rel=1e-9)


def test_burgers_magnitude_unknown_family_raises():
    al = UnitCell.cubic(4.0495e-10)
    with pytest.raises(ValueError, match="not defined"):
        burgers_magnitude("fcc", "{110}<111>", al)


# ---------------------------------------------------------------------------
# Task 3: derive_structure_type
# ---------------------------------------------------------------------------

from dfxm_geo.crystal.slip_systems import derive_structure_type  # noqa: E402


def test_derive_explicit_wins():
    assert derive_structure_type(structure_type="bcc", space_group=None, lattice="cubic") == "bcc"


def test_derive_default_is_fcc():
    assert derive_structure_type(structure_type=None, space_group=None, lattice="cubic") == "fcc"


@pytest.mark.parametrize(
    "sg, expected",
    [("Fm-3m", "fcc"), ("Im-3m", "bcc"), ("P6_3/mmc", "hcp")],
)
def test_derive_from_space_group(sg, expected):
    pytest.importorskip("gemmi")
    assert derive_structure_type(structure_type=None, space_group=sg, lattice=None) == expected


def test_derive_contradiction_raises():
    pytest.importorskip("gemmi")
    with pytest.raises(ValueError, match="contradicts"):
        derive_structure_type(structure_type="bcc", space_group="Fm-3m", lattice="cubic")


def test_derive_explicit_agrees_with_space_group():
    pytest.importorskip("gemmi")
    assert (
        derive_structure_type(structure_type="fcc", space_group="Fm-3m", lattice="cubic") == "fcc"
    )
