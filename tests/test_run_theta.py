"""Tests for the `run_theta` resolver (Slice 5, S0 of ForwardContext refactor #16).

`run_theta(config)` returns the run's Bragg angle (radians):
  - simplified mode: uses `_validate_reflection(hkl, keV, lattice_a)` — the
    true reflection-specific angle, NOT the legacy import-time constant.
  - oblique mode: uses `config.geometry.theta_validated` (solver result baked
    into the kernel filename at bootstrap time).

S0 is PURELY ADDITIVE — no existing pipeline behaviour changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.crystal.cell import UnitCell
from dfxm_geo.pipeline import GeometryConfig, ReciprocalConfig, SimulationConfig, run_theta
from dfxm_geo.reciprocal_space.kernel import _validate_reflection


def test_run_theta_simplified_uses_reflection_bragg_angle() -> None:
    """For a non-default reflection, run_theta returns the correct Bragg angle."""
    cfg = SimulationConfig(reciprocal=ReciprocalConfig(hkl=(2, 2, 0), keV=17.0))
    expected = _validate_reflection((2, 2, 0), 17.0, UnitCell.cubic(cfg.reciprocal.lattice_a))
    assert run_theta(cfg) == expected


def test_run_theta_default_reflection_matches_validate_reflection() -> None:
    """Default (-1,1,-1)@17keV: run_theta returns the TRUE Bragg angle.

    This must be ~0.156611 rad / 8.97317 deg, NOT the legacy import constant
    17.953/2 deg (~0.15663 rad) — the two differ by more than 1e-5 rad.
    """
    cfg = SimulationConfig(reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0))
    expected = _validate_reflection((-1, 1, -1), 17.0, UnitCell.cubic(cfg.reciprocal.lattice_a))
    assert run_theta(cfg) == expected
    # Explicitly confirm it differs from the legacy constant.
    assert abs(run_theta(cfg) - 17.953 / 2 * np.pi / 180) > 1e-5


def test_run_theta_oblique_uses_theta_validated() -> None:
    """Oblique mode: run_theta returns geometry.theta_validated, not the Bragg formula.

    The oblique theta comes from the solver (compute_omega_eta) and is stored at
    bootstrap time in the kernel filename.  To verify that run_theta returns the
    *stored* value (not a re-derived formula value), we inject a theta_validated
    that is measurably different from what _validate_reflection would return, and
    confirm that run_theta honours it.
    """
    # Inject a theta_validated that is deliberately offset from the formula result.
    sentinel_theta = 0.30  # ~17.2 deg — distinct from the (-1,-1,3)@19.1keV formula
    cfg = SimulationConfig(
        reciprocal=ReciprocalConfig(hkl=(-1, -1, 3), keV=19.1),
        geometry=GeometryConfig(
            mode="oblique",
            eta=0.353140,
            theta_validated=sentinel_theta,
        ),
    )
    assert cfg.geometry.theta_validated is not None
    # run_theta must return the stored value, not re-derive it.
    assert run_theta(cfg) == float(sentinel_theta)
    # Confirm the stored value differs from the formula result.
    formula_theta = _validate_reflection(
        (-1, -1, 3), 19.1, UnitCell.cubic(cfg.reciprocal.lattice_a)
    )
    assert abs(run_theta(cfg) - formula_theta) > 1e-3


def test_validate_reflection_cubic_bit_identical_to_legacy_formula():
    """M4 Stage 4.1: the UnitCell-based d-spacing must be EXACTLY the legacy value."""
    from dfxm_geo.crystal.cell import UnitCell
    from dfxm_geo.reciprocal_space.kernel import _validate_reflection

    a = 4.0495e-10
    theta = _validate_reflection((-1, 1, -1), 17.0, UnitCell.cubic(a))
    d = a / np.sqrt(3)
    lam = 1.239841984e-9 / 17.0
    assert theta == float(np.arcsin(lam / (2 * d)))


def test_validate_reflection_hexagonal():
    from dfxm_geo.crystal.cell import UnitCell
    from dfxm_geo.reciprocal_space.kernel import _validate_reflection

    cell = UnitCell.from_lattice("hexagonal", a=3.2094e-10, c=5.2108e-10)
    theta = _validate_reflection((2, -1, 0), 17.0, cell)
    lam = 1.239841984e-9 / 17.0
    assert theta == pytest.approx(float(np.arcsin(lam / 3.2094e-10)), rel=1e-12)
