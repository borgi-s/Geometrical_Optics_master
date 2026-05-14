"""Unit tests for dfxm_geo.crystal.burgers — slip-plane geometry helpers."""

import numpy as np
import pytest

from dfxm_geo.crystal.burgers import burgers_vectors

# The four {111}-family normals from the branch source's lookup table.
SLIP_PLANE_NORMALS = [
    (1, 1, 1),
    (1, -1, 1),
    (1, 1, -1),
    (-1, 1, 1),
]


@pytest.mark.parametrize("normal", SLIP_PLANE_NORMALS)
def test_burgers_vectors_shape(normal):
    """Returns (6, 3) — 3 basis Burgers vectors + 3 negatives."""
    b = burgers_vectors(normal)
    assert b.shape == (6, 3)


@pytest.mark.parametrize("normal", SLIP_PLANE_NORMALS)
def test_burgers_vectors_perpendicular_to_normal(normal):
    """All 6 Burgers vectors satisfy b · n = 0 (b lies in the slip plane)."""
    n = np.asarray(normal, dtype=float)
    b = burgers_vectors(normal)
    dots = b @ n
    np.testing.assert_allclose(dots, 0.0, atol=1e-12)


@pytest.mark.parametrize("normal", SLIP_PLANE_NORMALS)
def test_burgers_vectors_paired_negatives(normal):
    """The 6 vectors come in 3 ± pairs: b[i+3] == -b[i]."""
    b = burgers_vectors(normal)
    np.testing.assert_array_equal(b[3:], -b[:3])


def test_burgers_vectors_unit_magnitude_in_aluminum_units():
    """Vectors are normalized to magnitude 1 (matches branch code:
    `np.vstack([basis, -basis]) / np.sqrt(2)` — each basis vector like
    [-1, 1, 0] has magnitude sqrt(2), so dividing by sqrt(2) gives unit).
    """
    b = burgers_vectors((1, 1, 1))
    mags = np.linalg.norm(b, axis=1)
    np.testing.assert_allclose(mags, 1.0, rtol=1e-12)


def test_burgers_vectors_invalid_normal_raises():
    """Non-{111} normal raises ValueError."""
    with pytest.raises(ValueError, match="not one of the four"):
        burgers_vectors((2, 0, 0))
