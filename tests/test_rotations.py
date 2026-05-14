"""Unit tests for dfxm_geo.crystal.rotations."""

import numpy as np
import pytest

from dfxm_geo.crystal.rotations import (
    fast_inverse2,
    is_valid_rotation_matrix,
    rotate_matrix_z_axis,
    rotatedU,
)


def _is_orthogonal(M: np.ndarray, atol: float = 1e-10) -> bool:
    """A 3x3 rotation matrix satisfies M @ M.T == I and det(M) == 1."""
    if M.shape != (3, 3):
        return False
    return np.allclose(M @ M.T, np.eye(3), atol=atol) and np.isclose(
        np.linalg.det(M), 1.0, atol=atol
    )


@pytest.fixture
def axis_z() -> np.ndarray:
    """Unit z-axis."""
    return np.array([0.0, 0.0, 1.0])


@pytest.fixture
def axis_xy() -> np.ndarray:
    """45° axis in the xy-plane, normalised."""
    a = np.array([1.0, 1.0, 0.0])
    return a / np.linalg.norm(a)


class TestRotatedU:
    def test_zero_rotation_is_identity(self, axis_z):
        """rotatedU(axis, 0, I, sample) returns I."""
        result = rotatedU(axis_z, 0.0, np.eye(3), "sample")
        np.testing.assert_allclose(result, np.eye(3), atol=1e-12)

    def test_2pi_rotation_returns_to_start(self, axis_z):
        """rotatedU(axis, 2π, I, sample) returns I (within float tolerance)."""
        result = rotatedU(axis_z, 2 * np.pi, np.eye(3), "sample")
        np.testing.assert_allclose(result, np.eye(3), atol=1e-10)

    def test_preserves_orthogonality(self, axis_xy):
        """Composing a random rotation onto an identity gives an orthogonal matrix."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            alpha = rng.uniform(-np.pi, np.pi)
            result = rotatedU(axis_xy, alpha, np.eye(3), "sample")
            assert _is_orthogonal(result), f"Not orthogonal for alpha={alpha}"

    def test_cryst_vs_sample_differ_under_nontrivial_U(self):
        """coordtype='cryst' first rotates the axis into the sample frame via U."""
        # Use a non-identity U so the two paths actually differ.
        U = np.array(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        axis = np.array([1.0, 0.0, 0.0])
        rot_sample = rotatedU(axis, 0.7, U, "sample")
        rot_cryst = rotatedU(axis, 0.7, U, "cryst")
        # They should differ for a non-identity U.
        assert not np.allclose(rot_sample, rot_cryst)


class TestFastInverse2:
    def test_matches_numpy_inv_random(self):
        """For random invertible matrices, fast_inverse2 ≈ np.linalg.inv."""
        rng = np.random.default_rng(42)
        N = 32
        A = rng.normal(size=(N, 3, 3))
        # Reject near-singular draws.
        dets = np.linalg.det(A)
        A = A[np.abs(dets) > 1e-3]
        expected = np.linalg.inv(A)
        got = fast_inverse2(A.astype(np.float64))
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)

    def test_inverse_of_identity_is_identity(self):
        """For a stack of identities, the inverse is the same stack."""
        A = np.broadcast_to(np.eye(3), (5, 3, 3)).copy()
        got = fast_inverse2(A)
        np.testing.assert_allclose(got, A, atol=1e-12)

    def test_inverse_times_original_is_identity(self):
        """A @ A⁻¹ ≈ I."""
        rng = np.random.default_rng(7)
        A = rng.normal(size=(8, 3, 3))
        # Keep determinants away from 0.
        A = A[np.abs(np.linalg.det(A)) > 0.5]
        inv = fast_inverse2(A)
        product = np.einsum("nij,njk->nik", A, inv)
        expected = np.broadcast_to(np.eye(3), product.shape)
        np.testing.assert_allclose(product, expected, atol=1e-9)


def test_rotate_matrix_z_axis_zero_is_identity():
    """A 0° rotation around z leaves any matrix unchanged."""
    M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    np.testing.assert_allclose(rotate_matrix_z_axis(M, 0.0), M, atol=1e-15)


def test_rotate_matrix_z_axis_90_permutes_first_two_rows():
    """Rotating identity by 90° around z swaps and signs the first two rows
    of the result (left-multiplication by R_z(90°))."""
    I = np.identity(3)
    R90 = rotate_matrix_z_axis(I, 90.0)
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(R90, expected, atol=1e-12)


def test_rotate_matrix_z_axis_360_is_identity():
    """A 360° rotation returns to identity (within FP tolerance)."""
    M = np.array([[0.5, 0.1, 0.0], [0.2, 0.7, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(rotate_matrix_z_axis(M, 360.0), M, atol=1e-12)


def test_is_valid_rotation_matrix_accepts_identity():
    assert is_valid_rotation_matrix(np.identity(3)) is True


def test_is_valid_rotation_matrix_accepts_scipy_random_rotation():
    """A scipy-generated random rotation is always valid."""
    from scipy.spatial.transform import Rotation as R

    M = R.random(random_state=0).as_matrix()
    assert is_valid_rotation_matrix(M) is True


def test_is_valid_rotation_matrix_rejects_scaled_identity():
    """det != 1 → invalid."""
    assert is_valid_rotation_matrix(2.0 * np.identity(3)) is False


def test_is_valid_rotation_matrix_rejects_non_orthogonal():
    """R @ R.T != I → invalid."""
    M = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert is_valid_rotation_matrix(M) is False


def test_is_valid_rotation_matrix_atol_kwarg():
    """A slightly-non-orthogonal matrix is accepted if atol is loosened.

    Uses a *symmetric* perturbation so M @ M.T - I has first-order entries
    ~2e-4 (an antisymmetric perturbation cancels at first order and would
    pass even at atol=1e-6).
    """
    M = np.identity(3) + 1e-4 * np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert is_valid_rotation_matrix(M, atol=1e-3) is True
    assert is_valid_rotation_matrix(M, atol=1e-6) is False
