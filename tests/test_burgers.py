"""Unit tests for dfxm_geo.crystal.burgers — slip-plane geometry helpers."""

import numpy as np
import pytest

from dfxm_geo.crystal.burgers import burgers_vectors, rotated_t_vectors, ud_matrices

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


def test_rotated_t_vectors_shape():
    """Shape is (n_angles, n_burgers, 3)."""
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b = burgers_vectors((1, 1, 1))
    angles = np.array([0.0, 90.0, 180.0])
    result = rotated_t_vectors(n, b, angles)
    assert result.shape == (3, 6, 3)


def test_rotated_t_vectors_zero_angle_is_b_cross_n():
    """At angle=0, the rotated vector equals t_0 = b × n (initial in-plane)."""
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b = burgers_vectors((1, 1, 1))
    result = rotated_t_vectors(n, b, np.array([0.0]))

    t_expected = np.cross(b, n)
    np.testing.assert_allclose(result[0], t_expected, atol=1e-12)


def test_rotated_t_vectors_180_negates():
    """At angle=180°, the rotated vector is -t_0."""
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b = burgers_vectors((1, 1, 1))
    result = rotated_t_vectors(n, b, np.array([180.0]))
    t_expected = -np.cross(b, n)
    np.testing.assert_allclose(result[0], t_expected, atol=1e-12)


def test_ud_matrices_shape():
    """Shape is (n_angles, n_burgers, 3, 3)."""
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b = burgers_vectors((1, 1, 1))
    rotated = rotated_t_vectors(n, b, np.array([0.0, 45.0]))
    Ud = ud_matrices(n, rotated)
    assert Ud.shape == (2, 6, 3, 3)


def test_ud_matrices_columns_are_basis():
    """Each Ud has columns (n × t, n, t) per branch source convention."""
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    b = burgers_vectors((1, 1, 1))
    rotated = rotated_t_vectors(n, b, np.array([30.0]))
    Ud = ud_matrices(n, rotated)
    for j in range(6):
        np.testing.assert_allclose(Ud[0, j, :, 2], rotated[0, j], atol=1e-12)
        np.testing.assert_allclose(Ud[0, j, :, 1], n, atol=1e-12)


def test_burgers_vectors_fcc_bit_identical_after_registry():
    """burgers_vectors stays bit-identical for the four {111} planes."""
    for plane, basis in [
        ((1, 1, 1), [[-1, 1, 0], [1, 0, -1], [0, 1, -1]]),
        ((1, -1, 1), [[1, 1, 0], [1, 0, -1], [0, 1, 1]]),
        ((1, 1, -1), [[1, -1, 0], [1, 0, 1], [0, -1, -1]]),
        ((-1, 1, 1), [[-1, -1, 0], [-1, 0, -1], [0, 1, -1]]),
    ]:
        b = np.array(basis, float)
        want = np.vstack([b, -b]) / np.sqrt(2)
        got = burgers_vectors(plane)
        gs = sorted(tuple(np.round(v, 9)) for v in got)
        ws = sorted(tuple(np.round(v, 9)) for v in want)
        assert np.allclose(gs, ws)
