"""Parity: fused Find_Hg numba kernel vs the NumPy reference oracle.

Correctness bar (brainstorming decision): allclose at rtol=1e-12. Allows tiny
FMA/reassociation drift from the fused kernel; tighter than the legacy
Fd_find_smoke golden (rtol=1e-10) deliberately, since this path is the new hot
loop and we want drift flagged early.
"""

from __future__ import annotations

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.crystal.dislocations import Fd_find_mixed, find_hg_population
from dfxm_geo.crystal.rotations import fast_inverse2
from dfxm_geo.pipeline import (
    CenteredCrystalConfig,
    CrystalConfig,
    RandomDislocationsConfig,
)


def _centered_pop():
    crystal = CrystalConfig(mode="centered", centered=CenteredCrystalConfig())
    return fm.build_dislocation_population(crystal, fov_lateral_um=abs(fm.yl_start) * 2e6, rng=None)


def _random_pop(ndis: int = 4, seed: int = 42):
    crystal = CrystalConfig(
        mode="random_dislocations",
        random_dislocations=RandomDislocationsConfig(
            ndis=ndis, sigma=5.0, min_distance=4.0, seed=seed
        ),
    )
    rng = np.random.default_rng(seed)
    return fm.build_dislocation_population(crystal, fov_lateral_um=abs(fm.yl_start) * 2e6, rng=rng)


@pytest.mark.parametrize("pop_factory", [_centered_pop, _random_pop])
def test_fused_matches_numpy_oracle(pop_factory):
    pop = pop_factory()
    Hg_fast, q_fast = fm.Find_Hg_from_population(pop, h=-1, k=1, l=-1)
    Hg_ref, q_ref = fm._find_hg_from_population_numpy(pop, -1, 1, -1, S=fm._S_IDENTITY, rl=None)
    np.testing.assert_allclose(q_fast, q_ref, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(Hg_fast, Hg_ref, rtol=1e-12, atol=1e-14)


def test_mixed_character_matches_fd_find_mixed_oracle():
    """Cover the screw/mixed path (rotation_deg != 0) that the production
    caller never exercises: a single mixed-character dislocation must match
    Fd_find_mixed + fast_inverse2 directly, at rtol=1e-12."""
    rng = np.random.default_rng(7)
    X = 200
    rl_um = rng.standard_normal((3, X)) * 10.0  # micrometres
    rotation_deg = 37.0
    # Random orthonormal Ud (dislocation-to-grain) and Us; identity Theta/S.
    Ud = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    Us = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    Theta = np.eye(3)

    # Oracle: Fd_find_mixed expects rl already in micrometres (no extra *1e6).
    Fg_ref = Fd_find_mixed(rl_um, Us, Ud, rotation_deg, Theta)
    Hg_ref = np.transpose(fast_inverse2(Fg_ref), [0, 2, 1]) - np.identity(3)

    # Fused kernel: M = Ud.T @ Us.T @ S.T @ Theta (S=I), offset=0, cos/sin of rotation_deg.
    M = (Ud.T @ Us.T @ Theta)[None, :, :]
    offset = np.zeros((1, 3))
    Ud_stack = Ud[None, :, :]
    cos_rot = np.array([np.cos(np.deg2rad(rotation_deg))])
    sin_rot = np.array([np.sin(np.deg2rad(rotation_deg))])
    Hg_fast = find_hg_population(rl_um, M, offset, Ud_stack, cos_rot, sin_rot)

    np.testing.assert_allclose(Hg_fast, Hg_ref, rtol=1e-12, atol=1e-12)
