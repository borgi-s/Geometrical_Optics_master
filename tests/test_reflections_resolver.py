"""[[reflections]] / [reflections_auto] resolution — pure-logic layer."""

from __future__ import annotations

import pytest

from dfxm_geo.crystal.oblique import CrystalMount, compute_omega_eta
from dfxm_geo.crystal.reflections import (
    resolve_reflections,
    resolve_reflections_auto,
)

PAPER_MOUNT = CrystalMount(
    lattice="cubic", a=4.0493e-10, mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1)
)
KEV = 19.1
# Paper Table A.2 group 1: four {113}-family reflections share (θ=15.417°, η=20.233°)
GROUP1 = [(1, 1, 3), (-1, -1, 3), (1, -1, 3), (-1, 1, 3)]


def _entries(hkls, **common):
    return [dict(hkl=list(h), **common) for h in hkls]


def test_single_entry_defaults_to_solution_1():
    runs = resolve_reflections(_entries([(1, 1, 3)]), PAPER_MOUNT, KEV)
    assert len(runs) == 1
    r = runs[0]
    geom = compute_omega_eta(PAPER_MOUNT, (1, 1, 3), KEV)
    assert r.hkl == (1, 1, 3)
    assert r.eta == pytest.approx(geom.eta_1)
    assert r.omega == pytest.approx(geom.omega_1)
    assert r.theta == pytest.approx(geom.theta_1)
    assert r.group == 0


def test_omega_solution_2_selects_second_branch():
    runs = resolve_reflections(_entries([(1, 1, 3)], omega_solution=2), PAPER_MOUNT, KEV)
    geom = compute_omega_eta(PAPER_MOUNT, (1, 1, 3), KEV)
    assert runs[0].omega == pytest.approx(geom.omega_2)
    assert runs[0].eta == pytest.approx(geom.eta_2)


def test_explicit_eta_must_match_a_solution():
    with pytest.raises(ValueError, match="does not match"):
        resolve_reflections(_entries([(1, 1, 3)], eta=0.123456), PAPER_MOUNT, KEV)


def test_explicit_eta_picks_matching_solution():
    geom = compute_omega_eta(PAPER_MOUNT, (1, 1, 3), KEV)
    runs = resolve_reflections(_entries([(1, 1, 3)], eta=geom.eta_2), PAPER_MOUNT, KEV)
    assert runs[0].omega == pytest.approx(geom.omega_2)


def test_group1_reflections_share_one_group():
    """The paper's η=20.233° quadruple must dedup to a single kernel group."""
    entries = []
    for h in GROUP1:
        geom = compute_omega_eta(PAPER_MOUNT, h, KEV)
        # pick whichever solution carries the shared η=0.3531 rad
        target = 0.3531
        sol = 1 if abs(geom.eta_1 - target) < abs(geom.eta_2 - target) else 2
        entries.append(dict(hkl=list(h), omega_solution=sol))
    runs = resolve_reflections(entries, PAPER_MOUNT, KEV)
    assert {r.group for r in runs} == {0}


def test_mixed_groups_get_distinct_ids():
    runs = resolve_reflections(_entries([(1, 1, 1), (2, 0, 0)]), PAPER_MOUNT, KEV)
    assert runs[0].group != runs[1].group
    # group ids are dense, ordered by first appearance
    assert sorted({r.group for r in runs}) == [0, 1]


def test_unreachable_reflection_raises_with_cli_hint():
    # (7,7,7) at 19.1 keV: sin θ > 1 → no Laue solution
    with pytest.raises(ValueError, match="dfxm-find-reflections"):
        resolve_reflections(_entries([(7, 7, 7)]), PAPER_MOUNT, KEV)


def test_empty_list_raises():
    with pytest.raises(ValueError, match=r"\[\[reflections\]\]"):
        resolve_reflections([], PAPER_MOUNT, KEV)


def test_default_eta_applies_to_entries_without_eta():
    geom = compute_omega_eta(PAPER_MOUNT, (1, 1, 3), KEV)
    runs = resolve_reflections(_entries([(1, 1, 3)]), PAPER_MOUNT, KEV, default_eta=geom.eta_2)
    assert runs[0].omega == pytest.approx(geom.omega_2)


def test_reflection_run_is_frozen():
    runs = resolve_reflections(_entries([(1, 1, 3)]), PAPER_MOUNT, KEV)
    with pytest.raises(AttributeError):
        runs[0].eta = 0.0  # type: ignore[misc]


def test_auto_expands_paper_group1():
    """[reflections_auto] with the paper's η must recover the 4-reflection group."""
    runs = resolve_reflections_auto({"eta_target": 0.3531, "hkl_max": 3}, PAPER_MOUNT, KEV)
    assert {r.hkl for r in runs} >= set(GROUP1)
    assert {r.group for r in runs} == {0}


def test_auto_zero_matches_raises():
    with pytest.raises(ValueError, match="matched no reflections"):
        resolve_reflections_auto({"eta_target": 1.5, "hkl_max": 2}, PAPER_MOUNT, KEV)


def test_auto_requires_eta_target():
    with pytest.raises(ValueError, match="eta_target"):
        resolve_reflections_auto({}, PAPER_MOUNT, KEV)
