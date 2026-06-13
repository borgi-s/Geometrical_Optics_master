# tests/test_hcp_frame.py
"""M4 Stage 4.3b: Cartesian-frame Ud + q_hkl; cubic byte-identity of both."""

from __future__ import annotations

import numpy as np

from dfxm_geo.crystal.cell import UnitCell
from dfxm_geo.crystal.slip_systems import slip_systems
from dfxm_geo.direct_space.forward_model import _ud_matrix_from_bnt, _ud_matrix_from_bnt_cell


def test_ud_cell_cubic_is_byte_identical_to_legacy():
    cell = UnitCell.cubic(4.0495e-10)
    for s in slip_systems("fcc") + slip_systems("bcc"):
        legacy = _ud_matrix_from_bnt(s.b, s.n, s.t)
        viacell = _ud_matrix_from_bnt_cell(s.b, s.n, cell, t_int=s.t)
        assert np.array_equal(legacy, viacell), s  # EXACT


def test_ud_cell_hcp_is_orthonormal_rotation():
    cell = UnitCell.from_lattice("hexagonal", a=2.951e-10, c=4.684e-10)
    for s in slip_systems("hcp"):
        Ud = _ud_matrix_from_bnt_cell(s.b, s.n, cell, t_int=s.t)
        # proper rotation
        assert np.isclose(np.linalg.det(Ud), 1.0, atol=1e-10)
        assert np.allclose(Ud @ Ud.T, np.eye(3), atol=1e-10)
        # column 0 (b̂) is the Cartesian slip direction; column 1 (n̂) the plane normal
        b_hat = cell.A @ np.array(s.b, float)
        b_hat /= np.linalg.norm(b_hat)
        n_hat = cell.B @ np.array(s.n, float)
        n_hat /= np.linalg.norm(n_hat)
        assert np.allclose(Ud[:, 0], b_hat, atol=1e-10)
        assert np.allclose(Ud[:, 1], n_hat, atol=1e-10)
        assert abs(b_hat @ n_hat) < 1e-10  # glide => orthogonal in Cartesian
