"""Parse [geometry] block: mode + eta, with back-compat rules."""

import pytest

from dfxm_geo.reciprocal_space.kernel import _parse_geometry_block


def test_geometry_block_absent_returns_simplified_eta0() -> None:
    """No [geometry] → simplified, eta=0 (v2.2.0 default behaviour)."""
    mode, eta = _parse_geometry_block(None)
    assert mode == "simplified"
    assert eta == 0.0


def test_geometry_simplified_explicit() -> None:
    mode, eta = _parse_geometry_block({"mode": "simplified"})
    assert mode == "simplified"
    assert eta == 0.0


def test_geometry_simplified_with_nonzero_eta_warns_but_forces_zero(capsys) -> None:
    mode, eta = _parse_geometry_block({"mode": "simplified", "eta": 0.3})
    assert mode == "simplified"
    assert eta == 0.0
    captured = capsys.readouterr()
    assert "ignoring [geometry] eta" in captured.err


def test_geometry_oblique_requires_eta() -> None:
    with pytest.raises(ValueError, match="requires \\[geometry\\] eta"):
        _parse_geometry_block({"mode": "oblique"})


def test_geometry_oblique_with_eta_returns_both() -> None:
    mode, eta = _parse_geometry_block({"mode": "oblique", "eta": 0.3531})
    assert mode == "oblique"
    assert eta == 0.3531


def test_geometry_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        _parse_geometry_block({"mode": "bogus", "eta": 0.0})
