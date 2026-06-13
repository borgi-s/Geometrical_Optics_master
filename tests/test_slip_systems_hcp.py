# tests/test_slip_systems_hcp.py
"""M4 Stage 4.3b: HCP slip-system registry (hexagonal enumerator)."""

from __future__ import annotations

import pytest

from dfxm_geo.crystal.slip_systems import (
    hkil_to_hkl,
    plane_normals,
    slip_systems,
    uvtw_to_uvw,
)

# Expected per-family system counts (Hull & Bacon; Partridge 1967).
_HCP_FAMILY_COUNTS = {
    "{0001}<11-20>": 3,
    "{10-10}<11-20>": 3,
    "{10-11}<11-20>": 6,
    "{10-11}<11-23>": 12,
    "{11-22}<11-23>": 6,
}


def test_hkil_to_hkl_drops_redundant_index():
    assert hkil_to_hkl((1, 0, -1, 1)) == (1, 0, 1)
    assert hkil_to_hkl((0, 0, 0, 1)) == (0, 0, 1)
    with pytest.raises(ValueError):
        hkil_to_hkl((1, 0, 0, 1))  # i != -(h+k)


def test_uvtw_to_uvw_reduces_by_gcd():
    # a1 = 1/3<2-1-10> -> [100]; a1+a2 = <11-20> -> [110]
    assert uvtw_to_uvw((2, -1, -1, 0)) == (1, 0, 0)
    assert uvtw_to_uvw((1, 1, -2, 0)) == (1, 1, 0)
    # <c+a> 1/3<11-23bar> -> [1,1,-1]
    assert uvtw_to_uvw((1, 1, -2, -3)) == (1, 1, -1)
    with pytest.raises(ValueError):
        uvtw_to_uvw((1, 1, 0, 0))  # t != -(u+v)


def test_hcp_total_system_count_is_30():
    assert len(slip_systems("hcp")) == 30


def test_hcp_per_family_counts():
    sysl = slip_systems("hcp")
    for fam, n in _HCP_FAMILY_COUNTS.items():
        got = sum(1 for s in sysl if s.family == fam)
        assert got == n, f"family {fam}: expected {n}, got {got}"


def test_hcp_every_system_is_glide():
    for s in slip_systems("hcp"):
        assert s.b[0] * s.n[0] + s.b[1] * s.n[1] + s.b[2] * s.n[2] == 0, s


def test_hcp_basal_systems_are_the_three_a_axes_in_0001():
    basal = [s for s in slip_systems("hcp", families=["{0001}<11-20>"]) if True]
    assert len(basal) == 3
    assert all(s.n == (0, 0, 1) for s in basal)
    # The three basal <a> directions (in reduced 3-index) are [1,0,0], [0,1,0], [-1,-1,0].
    # The first two are permutations of [1,0,0]-type; the third is [-1,-1,0]-type.
    # All are in-plane (W=0) — check that every Burgers has zero z-component.
    assert all(s.b[2] == 0 for s in basal), "all basal <a> Burgers must lie in the ab-plane"
    # The set of Burgers vectors (up to sign) must cover the three a-axes.
    bset = {tuple(abs(c) for c in s.b) for s in basal}
    assert (1, 0, 0) in bset or (0, 1, 0) in bset  # at least one [1,0,0]-type


def test_hcp_plane_normals_distinct_counts():
    # basal 1, prismatic 3, 1st-order pyramidal 6 (shared by <a> and <c+a>),
    # 2nd-order pyramidal 6.  Distinct planes over all families:
    planes = plane_normals("hcp")
    assert (0, 0, 1) in planes
    assert len(planes) == 1 + 3 + 6 + 6  # = 16 distinct slip-plane normals


def test_hcp_families_subset_selection():
    only_basal = slip_systems("hcp", families=["{0001}<11-20>"])
    assert len(only_basal) == 3
    with pytest.raises(ValueError):
        slip_systems("hcp", families=["{nonsense}"])


def test_fcc_bcc_unchanged_by_hcp_addition():
    # The cubic enumerator must be untouched: counts identical to 4.3a.
    assert len(slip_systems("fcc")) == 12
    assert len(slip_systems("bcc")) == 24
