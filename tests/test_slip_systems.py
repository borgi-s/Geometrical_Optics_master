"""Slip-system registry + enumerator tests."""

import numpy as np
import pytest

from dfxm_geo.crystal.cell import UnitCell
from dfxm_geo.crystal.slip_systems import (
    burgers_in_plane,
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


def test_fcc_counts_and_geometry():
    sys = slip_systems("fcc")
    assert len(sys) == 12
    assert len({_canon(s.n) for s in sys}) == 4  # 4 distinct {111} planes
    for s in sys:
        assert np.dot(s.b, s.n) == 0  # glide: Burgers in plane
        assert np.allclose(np.cross(s.n, s.b), s.t)  # t = n x b


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
    """The enumerator must reproduce the hand-written _SLIP_SYSTEM_111 set
    (up to sign) BEFORE that table is deleted (later task)."""
    from dfxm_geo.direct_space.forward_model import _SLIP_SYSTEM_111

    legacy = {(_canon(b), _canon(n)) for b, n, _t in _SLIP_SYSTEM_111}
    reg = {(_canon(s.b), _canon(s.n)) for s in slip_systems("fcc")}
    assert reg == legacy


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
    from dfxm_geo.crystal.burgers import _BASIS_TABLE  # still present pre-Task 5

    def _slug_to_plane(slug):
        # "1-11" -> (1,-1,1); "-111" -> (-1,1,1)
        out, i = [], 0
        while i < len(slug):
            if slug[i] == "-":
                out.append(-int(slug[i + 1]))
                i += 2
            else:
                out.append(int(slug[i]))
                i += 1
        return tuple(out)

    for slug, basis in _BASIS_TABLE.items():
        plane = _slug_to_plane(slug)
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
