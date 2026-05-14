"""Tests for the sample-remount rotation matrices (Purdue 2024 paper)."""

import numpy as np

from dfxm_geo.crystal.remount import (
    S1,
    S2,
    S3,
    S4,
    SAMPLE_REMOUNT_OPTIONS,
)


def test_constants_are_proper_rotations() -> None:
    """Each S_i must be orthogonal with det=+1 (proper rotation)."""
    for name, S in [("S1", S1), ("S2", S2), ("S3", S3), ("S4", S4)]:
        np.testing.assert_allclose(
            S.T @ S, np.identity(3), atol=1e-10, err_msg=f"{name} not orthogonal"
        )
        assert np.isclose(np.linalg.det(S), 1.0, atol=1e-10), f"{name} det != +1"


def test_S1_is_identity() -> None:
    np.testing.assert_array_equal(S1, np.identity(3))


def test_S2_S3_S4_match_purdue_source_verbatim() -> None:
    """Pin the literal values from origin/Purdue_Paper direct_space/forward_model.py.

    Regression test: catches accidental edits to the constants.
    """
    np.testing.assert_array_equal(
        S2,
        np.array(
            [
                [1 / 3, -2 / 3, -2 / 3],
                [2 / 3, -1 / 3, 2 / 3],
                [-2 / 3, -2 / 3, 1 / 3],
            ]
        ),
    )
    np.testing.assert_array_equal(
        S3,
        np.array(
            [
                [1 / 3, -2 / 3, 2 / 3],
                [2 / 3, 2 / 3, 1 / 3],
                [-2 / 3, 1 / 3, 2 / 3],
            ]
        ),
    )
    np.testing.assert_array_equal(
        S4,
        np.array(
            [
                [1 / 3, 2 / 3, 2 / 3],
                [2 / 3, 1 / 3, -2 / 3],
                [-2 / 3, 2 / 3, -1 / 3],
            ]
        ),
    )


def test_sample_remount_options_map_is_complete() -> None:
    """Map keys exactly match the public names; values are the same array objects."""
    assert set(SAMPLE_REMOUNT_OPTIONS.keys()) == {"S1", "S2", "S3", "S4"}
    assert SAMPLE_REMOUNT_OPTIONS["S1"] is S1
    assert SAMPLE_REMOUNT_OPTIONS["S2"] is S2
    assert SAMPLE_REMOUNT_OPTIONS["S3"] is S3
    assert SAMPLE_REMOUNT_OPTIONS["S4"] is S4
