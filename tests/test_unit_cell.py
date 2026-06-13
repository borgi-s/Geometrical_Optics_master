"""UnitCell — general triclinic cell geometry (M4 Stage 4.1).

Cubic fast-paths must be BIT-IDENTICAL to the legacy formulas
(a·I cell matrix, d = a/sqrt(h^2+k^2+l^2)) — exact ==, not approx.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dfxm_geo.crystal.cell import UnitCell

A_AL = 4.0495e-10  # Al lattice parameter (m), legacy default
MG_A = 3.2094e-10  # Mg hexagonal a (m)
MG_C = 5.2108e-10  # Mg hexagonal c (m)


class TestCubicBitIdentity:
    def test_A_is_exactly_a_times_eye(self):
        cell = UnitCell.cubic(A_AL)
        assert np.array_equal(cell.A, A_AL * np.eye(3))

    def test_B_is_exactly_2pi_over_a_times_eye(self):
        cell = UnitCell.cubic(A_AL)
        assert np.array_equal(cell.B, (2.0 * np.pi / A_AL) * np.eye(3))

    def test_d_spacing_matches_legacy_formula_exactly(self):
        cell = UnitCell.cubic(A_AL)
        for h, k, l in [(-1, 1, -1), (2, 0, 0), (3, -1, 1), (1, 1, 3)]:
            legacy = A_AL / np.sqrt(h * h + k * k + l * l)
            assert cell.d_spacing((h, k, l)) == legacy

    def test_is_cubic(self):
        assert UnitCell.cubic(A_AL).is_cubic
        assert not UnitCell(
            a=MG_A, b=MG_A, c=MG_C, alpha_deg=90.0, beta_deg=90.0, gamma_deg=120.0
        ).is_cubic

    def test_cubic_q_hkl_shortcut_equals_B_dot_G(self):
        """Spec §3.5: for a cubic cell the forward path's q/|q| (B_0=I form)
        equals the metric-tensor B·G/|B·G|, so the cubic q_hkl shortcut is valid
        for BCC (a 2.8665 Å Fe cell here). Holds because B ∝ I for cubic."""
        cell = UnitCell.cubic(2.8665e-10)  # Fe BCC
        for hkl in [(1, 1, 0), (2, 0, 0), (2, 1, 1), (-1, 1, -1)]:
            g = np.asarray(hkl, dtype=float)
            q_shortcut = g / np.linalg.norm(g)  # forward path's B_0=I form
            Bg = cell.B @ g
            q_metric = Bg / np.linalg.norm(Bg)
            assert np.allclose(q_shortcut, q_metric, atol=1e-15)


class TestHexagonalDSpacing:
    def test_matches_textbook_formula(self):
        # 1/d^2 = (4/3)(h^2 + hk + k^2)/a^2 + l^2/c^2
        cell = UnitCell(a=MG_A, b=MG_A, c=MG_C, alpha_deg=90.0, beta_deg=90.0, gamma_deg=120.0)
        for h, k, l in [(1, 0, 0), (0, 0, 2), (1, 0, 1), (2, -1, 0), (1, 1, 2)]:
            inv_d2 = (4.0 / 3.0) * (h * h + h * k + k * k) / MG_A**2 + l * l / MG_C**2
            assert cell.d_spacing((h, k, l)) == pytest.approx(1.0 / math.sqrt(inv_d2), rel=1e-12)

    def test_2m10_d_is_exactly_a_over_2(self):
        cell = UnitCell(a=MG_A, b=MG_A, c=MG_C, alpha_deg=90.0, beta_deg=90.0, gamma_deg=120.0)
        assert cell.d_spacing((2, -1, 0)) == pytest.approx(MG_A / 2.0, rel=1e-12)


class TestTriclinicConsistency:
    def test_B_round_trips_A(self):
        cell = UnitCell(
            a=5.1e-10,
            b=6.2e-10,
            c=7.3e-10,
            alpha_deg=81.0,
            beta_deg=98.5,
            gamma_deg=105.2,
        )
        # B = 2*pi*inv(A).T  =>  A.T @ B = 2*pi*I
        assert cell.A.T @ cell.B / (2.0 * np.pi) == pytest.approx(np.eye(3), abs=1e-10)

    def test_a_vector_along_x_b_in_xy_plane(self):
        cell = UnitCell(
            a=5.1e-10,
            b=6.2e-10,
            c=7.3e-10,
            alpha_deg=81.0,
            beta_deg=98.5,
            gamma_deg=105.2,
        )
        a_vec, b_vec = cell.A[:, 0], cell.A[:, 1]
        assert a_vec[1] == 0.0 and a_vec[2] == 0.0
        assert b_vec[2] == 0.0


class TestValidation:
    def test_nonpositive_length_rejected(self):
        with pytest.raises(ValueError, match="must be finite and > 0"):
            UnitCell(a=-1e-10, b=1e-10, c=1e-10, alpha_deg=90.0, beta_deg=90.0, gamma_deg=90.0)

    def test_angle_out_of_range_rejected(self):
        with pytest.raises(ValueError, match=r"\(0, 180\)"):
            UnitCell(a=1e-10, b=1e-10, c=1e-10, alpha_deg=90.0, beta_deg=181.0, gamma_deg=90.0)

    def test_geometrically_impossible_angles_rejected(self):
        # alpha=10, beta=10, gamma=170 has no real cell volume
        with pytest.raises(ValueError, match="do not form a valid cell"):
            UnitCell(a=1e-10, b=1e-10, c=1e-10, alpha_deg=10.0, beta_deg=10.0, gamma_deg=170.0)


class TestFromLattice:
    def test_cubic_fills_everything(self):
        cell = UnitCell.from_lattice("cubic", a=A_AL)
        assert cell == UnitCell.cubic(A_AL)

    def test_hexagonal_fills_b_and_angles(self):
        cell = UnitCell.from_lattice("hexagonal", a=MG_A, c=MG_C)
        assert cell.b == MG_A
        assert (cell.alpha_deg, cell.beta_deg, cell.gamma_deg) == (90.0, 90.0, 120.0)

    def test_hexagonal_missing_c_rejected(self):
        with pytest.raises(ValueError, match="requires c"):
            UnitCell.from_lattice("hexagonal", a=MG_A)

    def test_cubic_conflicting_gamma_rejected(self):
        with pytest.raises(ValueError, match="constrains gamma_deg=90"):
            UnitCell.from_lattice("cubic", a=A_AL, gamma_deg=120.0)

    def test_tetragonal(self):
        cell = UnitCell.from_lattice("tetragonal", a=3e-10, c=5e-10)
        assert (cell.b, cell.c) == (3e-10, 5e-10)
        assert (cell.alpha_deg, cell.beta_deg, cell.gamma_deg) == (90.0, 90.0, 90.0)

    def test_orthorhombic_requires_b_and_c(self):
        with pytest.raises(ValueError, match="requires b"):
            UnitCell.from_lattice("orthorhombic", a=3e-10, c=5e-10)

    def test_trigonal_rhombohedral_setting(self):
        cell = UnitCell.from_lattice("trigonal", a=4e-10, alpha_deg=85.0)
        assert (cell.b, cell.c) == (4e-10, 4e-10)
        assert (cell.alpha_deg, cell.beta_deg, cell.gamma_deg) == (85.0, 85.0, 85.0)

    def test_trigonal_missing_alpha_rejected(self):
        with pytest.raises(ValueError, match="requires alpha_deg"):
            UnitCell.from_lattice("trigonal", a=4e-10)

    def test_monoclinic(self):
        cell = UnitCell.from_lattice("monoclinic", a=5e-10, b=6e-10, c=7e-10, beta_deg=101.0)
        assert (cell.alpha_deg, cell.beta_deg, cell.gamma_deg) == (90.0, 101.0, 90.0)

    def test_triclinic_requires_all_six(self):
        with pytest.raises(ValueError, match="requires alpha_deg"):
            UnitCell.from_lattice(
                "triclinic", a=5e-10, b=6e-10, c=7e-10, beta_deg=98.0, gamma_deg=105.0
            )

    def test_unknown_lattice_rejected(self):
        with pytest.raises(ValueError, match="unknown lattice"):
            UnitCell.from_lattice("bcc", a=A_AL)
