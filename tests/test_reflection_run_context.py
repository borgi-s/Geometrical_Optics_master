"""Per-ReflectionRun resolution + context helpers."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.crystal.oblique import CrystalMount
from dfxm_geo.crystal.reflections import resolve_reflections
from dfxm_geo.pipeline import (
    GeometryConfig,
    ReciprocalConfig,
    _context_for_run,
    _resolution_for_run,
)

MOUNT = CrystalMount(
    lattice="cubic", a=4.0493e-10, mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1)
)


@pytest.fixture
def run113():
    return resolve_reflections([{"hkl": [1, 1, 3]}], MOUNT, 19.1)[0]


def _analytic_recip() -> ReciprocalConfig:
    return ReciprocalConfig(
        hkl=(1, 1, 3),
        keV=19.1,
        backend="analytic",
        beamstop=False,
        lattice_a=4.0493e-10,
    )


def test_analytic_resolution_for_run(run113):
    geom = GeometryConfig(mode="oblique", eta=0.0, mount=MOUNT)
    res = _resolution_for_run(_analytic_recip(), geom, run113)
    assert res.analytic_eval is not None
    assert res.loaded_kernel_path is None
    # eta must be the RUN's eta, not the config-level placeholder
    assert res.analytic_eval.eta == pytest.approx(run113.eta)


def test_context_for_run_carries_run_geometry(run113):
    geom = GeometryConfig(mode="oblique", eta=0.0, mount=MOUNT)
    res = _resolution_for_run(_analytic_recip(), geom, run113)
    ctx = _context_for_run(res, run113)
    assert ctx.geometry.theta_0 == pytest.approx(run113.theta)
    assert ctx.geometry.omega == pytest.approx(run113.omega)
    expected_q = np.asarray(run113.hkl, dtype=float)
    expected_q /= np.linalg.norm(expected_q)
    np.testing.assert_allclose(ctx.q_hkl, expected_q)


def test_resolution_for_run_uses_run_hkl_in_recip(run113):
    """recip_run must have run.hkl so that the analytic backend derives the
    correct Bragg angle for this reflection (not the config-level hkl)."""
    geom = GeometryConfig(mode="oblique", eta=0.0, mount=MOUNT)
    # Use a different hkl on the top-level config to confirm the run overrides it.
    recip = ReciprocalConfig(
        hkl=(1, 1, 1),  # different from run113's (1,1,3)
        keV=19.1,
        backend="analytic",
        beamstop=False,
        lattice_a=4.0493e-10,
    )
    res = _resolution_for_run(recip, geom, run113)
    # Must resolve to run113.theta, not (1,1,1)'s theta
    from dfxm_geo.reciprocal_space.kernel import _validate_reflection

    theta_113 = float(_validate_reflection((1, 1, 3), 19.1, 4.0493e-10))
    assert res.analytic_eval is not None
    assert res.analytic_eval.theta == pytest.approx(theta_113)


def test_context_for_run_q_hkl_direction(run113):
    """q_hkl must be normalized and point in the run's hkl direction."""
    geom = GeometryConfig(mode="oblique", eta=0.0, mount=MOUNT)
    res = _resolution_for_run(_analytic_recip(), geom, run113)
    ctx = _context_for_run(res, run113)
    q = ctx.q_hkl
    # Unit vector
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-12)
    # Parallel to hkl
    h = np.asarray(run113.hkl, dtype=float)
    assert np.abs(np.dot(q, h / np.linalg.norm(h))) == pytest.approx(1.0, abs=1e-12)


def test_analytic_with_beamstop_rejected_through_helper(run113):
    """_resolution_for_run must propagate the analytic+beamstop ValueError from
    _load_resolution — the guard must not be silently bypassed for multi-reflection
    runs."""
    geom = GeometryConfig(mode="oblique", eta=0.0, mount=MOUNT)
    recip = ReciprocalConfig(
        hkl=(1, 1, 3),
        keV=19.1,
        backend="analytic",
        beamstop=True,
        lattice_a=4.0493e-10,
    )
    with pytest.raises(ValueError, match="beamstop"):
        _resolution_for_run(recip, geom, run113)
