"""forward_model.reflection_theta_if_oblique: set the run's Bragg angle for
oblique mode (recomputing the theta-dependent geometry globals), restore on exit.

theta_0 is hardcoded to the Al (1,1,1) @ 17 keV default (8.98 deg). Oblique
reflections (e.g. the paper -1-13 @ 19.1 keV, theta=15.416 deg) need the forward
geometry (Theta rotation, strain offset, ray grid) built at their own theta. The
context manager restores the module defaults on exit so simplified-mode runs and
other tests are never polluted.
"""

from __future__ import annotations

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm


def _R_y(theta: float) -> np.ndarray:
    return np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )


def test_oblique_context_sets_then_restores_geometry() -> None:
    th0_before = fm.theta_0
    theta_before = fm.theta
    Theta_before = fm.Theta.copy()
    rl_before = fm.rl.copy()
    new_theta = np.deg2rad(15.416)

    with fm.reflection_theta_if_oblique("oblique", new_theta):
        assert np.isclose(fm.theta_0, new_theta)
        assert np.isclose(fm.theta, new_theta)
        assert np.allclose(fm.Theta, _R_y(new_theta))
        # ray grid x-extent rebuilt for the new theta
        assert not np.allclose(fm.rl, rl_before)
        assert fm.rl.shape == rl_before.shape

    # restored exactly on exit
    assert fm.theta_0 == th0_before
    assert fm.theta == theta_before
    assert np.allclose(fm.Theta, Theta_before)
    assert np.allclose(fm.rl, rl_before)


def test_simplified_mode_is_noop() -> None:
    th0_before = fm.theta_0
    Theta_before = fm.Theta.copy()
    with fm.reflection_theta_if_oblique("simplified", None):
        assert fm.theta_0 == th0_before
        assert np.allclose(fm.Theta, Theta_before)
    assert fm.theta_0 == th0_before
    assert np.allclose(fm.Theta, Theta_before)


def test_restores_even_on_exception() -> None:
    th0_before = fm.theta_0
    rl_before = fm.rl.copy()
    with (
        pytest.raises(RuntimeError),
        fm.reflection_theta_if_oblique("oblique", np.deg2rad(15.416)),
    ):
        raise RuntimeError("boom")
    assert fm.theta_0 == th0_before
    assert np.allclose(fm.rl, rl_before)
