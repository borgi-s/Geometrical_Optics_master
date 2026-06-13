# tests/test_slip_systems_hcp.py
"""M4 Stage 4.3b: HCP slip-system registry (hexagonal enumerator)."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.crystal.slip_systems import (
    _REGISTRY,
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
    basal = slip_systems("hcp", families=["{0001}<11-20>"])
    assert len(basal) == 3
    assert all(s.n == (0, 0, 1) for s in basal)
    # The three basal <a> directions (in reduced 3-index) are [1,0,0], [0,1,0], [-1,-1,0].
    # The first two are permutations of [1,0,0]-type; the third is [-1,-1,0]-type.
    # All are in-plane (W=0) — check that every Burgers has zero z-component.
    assert all(s.b[2] == 0 for s in basal), "all basal <a> Burgers must lie in the ab-plane"
    # The set of canonical Burgers vectors must exactly pin the three a-axes.
    from dfxm_geo.crystal.slip_systems import _canon

    canon_set = {_canon(s.b) for s in basal}
    assert canon_set == {(1, 0, 0), (0, 1, 0), (1, 1, 0)}, f"got {sorted(canon_set)}"


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


def test_burgers_magnitude_of_int_hcp_a_and_ca():
    from dfxm_geo.crystal.cell import UnitCell
    from dfxm_geo.crystal.slip_systems import burgers_magnitude_of

    # alpha-Ti: a = 2.951 A, c = 4.684 A.
    cell = UnitCell.from_lattice("hexagonal", a=2.951e-10, c=4.684e-10)
    a_um = 2.951e-10 * 1e6
    ca_um = float(np.sqrt(2.951e-10**2 + 4.684e-10**2) * 1e6)
    assert np.isclose(burgers_magnitude_of((1, 0, 0), cell, fraction=1.0), a_um, rtol=1e-12)
    assert np.isclose(burgers_magnitude_of((1, 0, 1), cell, fraction=1.0), ca_um, rtol=1e-12)


def test_hcp_family_magnitudes_via_registry():
    from dfxm_geo.crystal.cell import UnitCell
    from dfxm_geo.crystal.slip_systems import burgers_magnitude

    cell = UnitCell.from_lattice("hexagonal", a=2.951e-10, c=4.684e-10)
    a_um = 2.951e-10 * 1e6
    ca_um = float(np.sqrt(2.951e-10**2 + 4.684e-10**2) * 1e6)
    assert np.isclose(burgers_magnitude("hcp", "{0001}<11-20>", cell), a_um, rtol=1e-12)
    assert np.isclose(burgers_magnitude("hcp", "{10-11}<11-23>", cell), ca_um, rtol=1e-12)


# ---------------------------------------------------------------------------
# FU2: mutation-safe memoization of _enumerate_family
# ---------------------------------------------------------------------------


def test_enumerate_family_memoized():
    """_enumerate_family must be memoized: same object returned on repeat calls."""
    from dfxm_geo.crystal.slip_systems import _enumerate_family

    # Clear cache to get a clean count.
    _enumerate_family.cache_clear()
    fam = _REGISTRY["hcp"][0]
    a = _enumerate_family(fam)
    b = _enumerate_family(fam)

    # The same tuple object should be returned from the cache.
    assert a is b, (
        "_enumerate_family must return the SAME tuple object on repeat calls (cache miss)"
    )
    # The returned type must be a tuple (immutable / hashable-safe).
    assert isinstance(a, tuple), f"_enumerate_family must return tuple, got {type(a)}"
    # Results are equal and non-empty.
    assert len(a) > 0
    assert a == b

    # Confirm the two calls hit: 1 miss + 1 hit.
    info = _enumerate_family.cache_info()
    assert info.hits >= 1, f"Expected at least 1 cache hit, got {info}"
    assert info.misses >= 1, f"Expected at least 1 cache miss, got {info}"


def test_slip_systems_returns_fresh_list():
    """slip_systems must return a fresh list each call (callers may mutate it)
    but the contents must be equal — the cache provides the shared tuples."""
    result_a = slip_systems("fcc")
    result_b = slip_systems("fcc")

    # A fresh list every call — not the same object.
    assert result_a is not result_b, "slip_systems must return a NEW list each call"
    # But the contents are equal.
    assert result_a == result_b, "slip_systems must return EQUAL results each call"

    # Also check HCP.
    hcp_a = slip_systems("hcp")
    hcp_b = slip_systems("hcp")
    assert hcp_a is not hcp_b
    assert hcp_a == hcp_b
