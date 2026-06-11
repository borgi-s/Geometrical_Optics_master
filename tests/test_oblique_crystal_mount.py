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


def test_unknown_lattice_raises():
    # v3.0.0: non-cubic lattices are accepted; unknown lattice strings are rejected.
    with pytest.raises(ValueError, match="lattice must be one of"):
        CrystalMount(
            lattice="fcc",  # not a valid lattice system
            a=4.0e-10,
            mount_x=(1, 0, 0),
            mount_y=(0, 1, 0),
            mount_z=(0, 0, 1),
        )


class TestNonCubicMount:
    """M4 Stage 4.1 — CrystalMount beyond cubic.

    Hexagonal standard setting: a* = (2pi/a)(1, 1/sqrt(3), 0),
    b* = (2pi/a)(0, 2/sqrt(3), 0), c* = (2pi/c)(0, 0, 1). So the plane-normal
    triple (2,-1,0)/(0,1,0)/(0,0,1) maps to Cartesian x/y/z — a valid
    orthogonal mount — while (1,0,0)/(0,1,0)/(0,0,1) does NOT (a* not
    orthogonal to b*). For cubic these two triples are equivalent, so this
    is a direct test that the metric is in play.
    """

    MG_A = 3.2094e-10
    MG_C = 5.2108e-10

    def _hex_mount(self):
        return CrystalMount(
            lattice="hexagonal",
            a=self.MG_A,
            c=self.MG_C,
            mount_x=(2, -1, 0),
            mount_y=(0, 1, 0),
            mount_z=(0, 0, 1),
        )

    def test_hexagonal_orthogonal_triple_accepted(self):
        m = self._hex_mount()
        U = m.U_mount
        assert pytest.approx(np.eye(3), abs=1e-12) == U @ U.T
        # This particular triple maps exactly onto the lab axes.
        assert pytest.approx(np.eye(3), abs=1e-12) == U

    def test_hexagonal_C_s_is_cell_A(self):
        m = self._hex_mount()
        assert np.array_equal(m.C_s, m.cell.A)

    def test_hexagonal_non_orthogonal_triple_rejected(self):
        with pytest.raises(ValueError, match="orthogonal"):
            CrystalMount(
                lattice="hexagonal",
                a=self.MG_A,
                c=self.MG_C,
                mount_x=(1, 0, 0),
                mount_y=(0, 1, 0),
                mount_z=(0, 0, 1),
            )

    def test_cubic_C_s_bit_identical(self):
        m = CrystalMount(
            lattice="cubic",
            a=4.0495e-10,
            mount_x=(1, 0, 0),
            mount_y=(0, 1, 0),
            mount_z=(0, 0, 1),
        )
        assert np.array_equal(m.C_s, 4.0495e-10 * np.eye(3))

    def test_cubic_U_mount_bit_identical_to_legacy_normalization(self):
        m = CrystalMount(
            lattice="cubic",
            a=4.0495e-10,
            mount_x=(1, 1, 0),
            mount_y=(-1, 1, 0),
            mount_z=(0, 0, 1),
        )
        cols = []
        for mil in ((1, 1, 0), (-1, 1, 0), (0, 0, 1)):
            v = np.array(mil, dtype=float)
            cols.append(v / np.linalg.norm(v))
        assert np.array_equal(m.U_mount, np.column_stack(cols))

    def test_missing_c_for_hexagonal_raises(self):
        with pytest.raises(ValueError, match="requires c"):
            CrystalMount(
                lattice="hexagonal",
                a=self.MG_A,
                mount_x=(2, -1, 0),
                mount_y=(0, 1, 0),
                mount_z=(0, 0, 1),
            )
