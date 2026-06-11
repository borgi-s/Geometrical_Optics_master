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
