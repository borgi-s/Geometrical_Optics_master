"""Slip-system registry + enumerator tests."""

import numpy as np
import pytest

from dfxm_geo.crystal.slip_systems import (
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
