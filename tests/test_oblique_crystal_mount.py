"""Tests for CrystalMount dataclass in crystal/oblique.py."""

import numpy as np
import pytest

from dfxm_geo.crystal.oblique import CrystalMount


def test_paper_al_mount_is_identity_U():
    """Paper §6.1 mount: (100)//x̂, (010)//ŷ, (001)//ẑ → U_mount = I."""
    mount = CrystalMount(
        lattice="cubic",
        a=4.0493e-10,
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )
    np.testing.assert_array_almost_equal(mount.U_mount, np.eye(3))


def test_C_s_is_diagonal_a_for_cubic():
    """Cubic cell matrix C_s = a · I."""
    mount = CrystalMount(
        lattice="cubic",
        a=4.0e-10,
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )
    np.testing.assert_array_almost_equal(mount.C_s, 4.0e-10 * np.eye(3))


def test_rotated_mount_gives_rotation_U():
    """Mount with 90° rotation about z (x→y, y→-x): U_mount columns reflect that."""
    mount = CrystalMount(
        lattice="cubic",
        a=4.0e-10,
        mount_x=(0, 1, 0),  # crystal y aligned with lab x
        mount_y=(-1, 0, 0),  # crystal -x aligned with lab y
        mount_z=(0, 0, 1),
    )
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    np.testing.assert_array_almost_equal(mount.U_mount, expected)


def test_non_orthogonal_mount_raises():
    with pytest.raises(ValueError, match="mutually orthogonal"):
        CrystalMount(
            lattice="cubic",
            a=4.0e-10,
            mount_x=(1, 0, 0),
            mount_y=(1, 1, 0),  # not orthogonal to mount_x
            mount_z=(0, 0, 1),
        )


def test_non_integer_mount_raises():
    with pytest.raises(ValueError, match="integers"):
        CrystalMount(
            lattice="cubic",
            a=4.0e-10,
            mount_x=(1.5, 0, 0),  # not integer
            mount_y=(0, 1, 0),
            mount_z=(0, 0, 1),
        )


def test_non_cubic_lattice_raises():
    with pytest.raises(ValueError, match="cubic"):
        CrystalMount(
            lattice="hexagonal",
            a=4.0e-10,
            mount_x=(1, 0, 0),
            mount_y=(0, 1, 0),
            mount_z=(0, 0, 1),
        )
