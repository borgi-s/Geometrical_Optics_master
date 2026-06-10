"""find_hg_scene — the single Hg seam for the identify orchestrators (v2.6.0).

NumPy-engine tests prove bit-identity with the legacy compositions; numba
engine tests (added in Tasks 5-6) prove parity at tight tolerance.
No resolution kernel needed: pure crystal/dislocations level.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.crystal.dislocations import (
    Fd_find_mixed,
    Fd_find_multi_dislocs_mixed,
    MixedDislocSpec,
    find_hg_scene,
)
from dfxm_geo.crystal.rotations import fast_inverse2


def _rand_rotation(rng: np.random.Generator) -> np.ndarray:
    """Random proper rotation via QR (det +1)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


@pytest.fixture()
def scene():
    rng = np.random.default_rng(42)
    rl_um = rng.uniform(-50.0, 50.0, size=(3, 200))
    Us = _rand_rotation(rng)
    Theta = _rand_rotation(rng)
    specs = [
        MixedDislocSpec(
            Ud_mix=_rand_rotation(rng),
            rotation_deg=float(rng.uniform(0, 180)),
            position_lab_um=(1.5, -2.0, 0.5),
        ),
        MixedDislocSpec(
            Ud_mix=_rand_rotation(rng),
            rotation_deg=float(rng.uniform(0, 180)),
            position_lab_um=(-3.0, 1.0, 0.0),
        ),
    ]
    return rl_um, Us, Theta, specs


def _legacy_hg_single(rl_um, Us, spec, Theta):
    """Exactly the single-dislocation composition in pipeline._iter_identification_single."""
    Fg = Fd_find_mixed(
        rl_um,
        Us,
        Ud_mix=spec.Ud_mix,
        rotation_deg=spec.rotation_deg,
        Theta=Theta,
        position_lab_um=spec.position_lab_um,
    )
    return np.transpose(fast_inverse2(Fg), [0, 2, 1]) - np.identity(3)


def test_single_spec_bit_identical_to_legacy(scene):
    rl_um, Us, Theta, specs = scene
    hg, solos = find_hg_scene(rl_um, Us, [specs[0]], Theta, engine="numpy")
    expected = _legacy_hg_single(rl_um, Us, specs[0], Theta)
    assert solos is None
    np.testing.assert_array_equal(hg, expected)  # BIT-identical


def test_combined_bit_identical_to_multi_dislocs(scene):
    rl_um, Us, Theta, specs = scene
    hg, solos = find_hg_scene(rl_um, Us, specs, Theta, engine="numpy")
    Fg = Fd_find_multi_dislocs_mixed(rl_um, Us, specs, Theta)
    expected = np.transpose(fast_inverse2(Fg), [0, 2, 1]) - np.identity(3)
    assert solos is None
    np.testing.assert_array_equal(hg, expected)


def test_per_dislocation_solos_bit_identical(scene):
    rl_um, Us, Theta, specs = scene
    hg, solos = find_hg_scene(rl_um, Us, specs, Theta, per_dislocation=True, engine="numpy")
    assert solos is not None and len(solos) == 2
    for spec, solo in zip(specs, solos, strict=True):
        np.testing.assert_array_equal(solo, _legacy_hg_single(rl_um, Us, spec, Theta))
    # combined unchanged by requesting components
    hg2, _ = find_hg_scene(rl_um, Us, specs, Theta, engine="numpy")
    np.testing.assert_array_equal(hg, hg2)


def test_empty_specs_raises(scene):
    rl_um, Us, Theta, _ = scene
    with pytest.raises(ValueError):
        find_hg_scene(rl_um, Us, [], Theta, engine="numpy")


def test_unknown_engine_raises(scene):
    rl_um, Us, Theta, specs = scene
    with pytest.raises(ValueError):
        find_hg_scene(rl_um, Us, [specs[0]], Theta, engine="fortran")


NUMBA_RTOL = 1e-12
NUMBA_ATOL = 1e-14
# Engines differ in FP op order (fast_inverse2 is fastmath=True; the fused
# per-dislocation kernel's inline inverse is fastmath=False) — parity, not
# bit-identity.  The binding tolerance at this scene's Hg magnitudes
# (O(1e-10)–O(1e-5)) is atol=1e-14 (rtol·|Hg| ≤ 1e-17 is negligible); rtol
# is kept at 1e-12 for consistency with the Phase-1 population-kernel
# parity precedent.


def test_numba_combined_matches_numpy_two_specs(scene):
    rl_um, Us, Theta, specs = scene
    hg_np, _ = find_hg_scene(rl_um, Us, specs, Theta, engine="numpy")
    hg_nb, solos = find_hg_scene(rl_um, Us, specs, Theta, engine="numba")
    assert solos is None
    np.testing.assert_allclose(hg_nb, hg_np, rtol=NUMBA_RTOL, atol=NUMBA_ATOL)


def test_numba_combined_matches_numpy_single_spec(scene):
    rl_um, Us, Theta, specs = scene
    hg_np, _ = find_hg_scene(rl_um, Us, [specs[0]], Theta, engine="numpy")
    hg_nb, _ = find_hg_scene(rl_um, Us, [specs[0]], Theta, engine="numba")
    np.testing.assert_allclose(hg_nb, hg_np, rtol=NUMBA_RTOL, atol=NUMBA_ATOL)


def test_numba_per_dislocation_matches_numpy(scene):
    rl_um, Us, Theta, specs = scene
    hg_np, solos_np = find_hg_scene(rl_um, Us, specs, Theta, per_dislocation=True, engine="numpy")
    hg_nb, solos_nb = find_hg_scene(rl_um, Us, specs, Theta, per_dislocation=True, engine="numba")
    np.testing.assert_allclose(hg_nb, hg_np, rtol=NUMBA_RTOL, atol=NUMBA_ATOL)
    assert solos_nb is not None and len(solos_nb) == len(solos_np)
    for nb, np_ in zip(solos_nb, solos_np, strict=True):
        np.testing.assert_allclose(nb, np_, rtol=NUMBA_RTOL, atol=NUMBA_ATOL)


def test_numba_perdis_combined_equals_combined_only(scene):
    # Requesting components must not change the combined result (numba path).
    rl_um, Us, Theta, specs = scene
    hg_only, _ = find_hg_scene(rl_um, Us, specs, Theta, engine="numba")
    hg_with, _ = find_hg_scene(rl_um, Us, specs, Theta, per_dislocation=True, engine="numba")
    np.testing.assert_array_equal(hg_only, hg_with)


def test_numba_combined_matches_numpy_with_remount_S(scene):
    rl_um, Us, Theta, specs = scene
    rng = np.random.default_rng(7)
    S = _rand_rotation(rng)
    hg_np, _ = find_hg_scene(rl_um, Us, specs, Theta, S=S, engine="numpy")
    hg_nb, _ = find_hg_scene(rl_um, Us, specs, Theta, S=S, engine="numba")
    np.testing.assert_allclose(hg_nb, hg_np, rtol=NUMBA_RTOL, atol=NUMBA_ATOL)
