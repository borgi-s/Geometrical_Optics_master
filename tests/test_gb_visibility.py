"""g·b visibility helpers — extraction of pipeline._passes_invisibility math."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.crystal.burgers import gb_cos, gb_visible


def test_gb_cos_perpendicular_is_zero():
    assert gb_cos(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_gb_cos_parallel_is_one():
    assert gb_cos(np.array([1.0, 1.0, 0.0]), np.array([2.0, 2.0, 0.0])) == pytest.approx(1.0)


def test_gb_cos_normalization_invariant():
    rng = np.random.default_rng(42)
    for _ in range(20):
        q, b = rng.normal(size=3), rng.normal(size=3)
        assert gb_cos(q, b) == pytest.approx(gb_cos(3.7 * q, 0.2 * b))


def test_gb_visible_threshold_semantics():
    # angle(G,b)=90° → invisible at any positive threshold
    assert not gb_visible(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 10.0)
    # angle(G,b)=0° → visible
    assert gb_visible(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 10.0)


def test_gb_visible_exact_threshold_boundary_is_visible():
    """gb_cos exactly equal to the cutoff → visible (>= semantics, not >).

    Uses the float-exact case gb_cos = 1.0 with threshold 90° (cutoff
    cos(0°) = 1.0): no rounding anywhere, so a >= → > regression flips it.
    """
    g = np.array([1.0, 0.0, 0.0])
    assert gb_cos(g, g) == 1.0
    assert gb_visible(g, g, 90.0)


def test_gb_visible_matches_pipeline_passes_invisibility():
    """The pipeline guard must delegate: identical verdicts on random input."""
    from dfxm_geo.pipeline import _passes_invisibility

    rng = np.random.default_rng(7)
    for _ in range(50):
        q, b = rng.normal(size=3), rng.normal(size=3)
        thr = float(rng.uniform(0.0, 45.0))
        assert _passes_invisibility(q, b, thr) == gb_visible(q, b, thr)
