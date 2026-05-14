"""Unit tests for dfxm_geo.crystal.dislocations mixed-character functions.

References:
    Borgi, S., Winther, G., Poulsen, H. F. (2025). J. Appl. Cryst. 58, 813-821.
    DOI: 10.1107/S1600576725002614. Eq. 1 defines the mixed-character F_d.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from dfxm_geo.constants import BURGERS_VECTOR, POISSON_RATIO
from dfxm_geo.crystal.dislocations import Fd_find, Fd_find_mixed, MixedDislocSpec


@pytest.fixture
def identity_rotations():
    """Identity Us / Theta and a simple Ud_mix for isolating dislocation-frame math."""
    return np.identity(3), np.identity(3), np.identity(3)


@pytest.fixture
def simple_rl_grid():
    """A small lab-frame grid avoiding the dislocation core singularity."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(3, 50)) * 1e-6  # 50 random points in metres


def test_mixed_disloc_spec_defaults():
    """MixedDislocSpec stores Ud_mix, rotation_deg, and a default zero position."""
    Ud = np.identity(3)
    spec = MixedDislocSpec(Ud_mix=Ud, rotation_deg=45.0)
    assert spec.rotation_deg == 45.0
    assert spec.position_lab_um == (0.0, 0.0, 0.0)
    np.testing.assert_array_equal(spec.Ud_mix, Ud)


def test_mixed_disloc_spec_with_position():
    """MixedDislocSpec accepts an explicit position offset (lab-frame µm)."""
    spec = MixedDislocSpec(Ud_mix=np.identity(3), rotation_deg=0.0, position_lab_um=(1.0, 2.0, 3.0))
    assert spec.position_lab_um == (1.0, 2.0, 3.0)


def test_mixed_disloc_spec_is_frozen():
    """MixedDislocSpec is immutable — mutation raises FrozenInstanceError."""
    spec = MixedDislocSpec(Ud_mix=np.identity(3), rotation_deg=0.0)
    with pytest.raises(FrozenInstanceError):
        spec.rotation_deg = 10.0  # type: ignore[misc]


def test_Fd_find_mixed_pure_edge_matches_Fd_find_ndis1(identity_rotations, simple_rl_grid):
    """At rotation_deg=0 (paper's α=90°, pure edge), Fd_find_mixed must equal
    Fd_find(ndis=1) with matching Ud. This is the regression guard tying the
    new mixed math to the cleanup's existing edge formula (Eq. 1 of Borgi
    2025 collapses to the edge term for α=90°).
    """
    Us, Ud, Theta = identity_rotations
    rl = simple_rl_grid

    Fg_edge = Fd_find(rl, Ud, Us, Theta, dis=1.0, ndis=1)
    Fg_mixed = Fd_find_mixed(rl, Us, Ud_mix=Ud, rotation_deg=0.0, Theta=Theta)

    np.testing.assert_allclose(Fg_mixed, Fg_edge, atol=1e-15, rtol=1e-12)


def test_Fd_find_mixed_pure_screw_has_only_screw_terms(identity_rotations):
    """At rotation_deg=90° (paper's α=0°, pure screw), the cos(rotation) factor
    zeros the edge contributions; the only nonzero off-diagonal entries are
    the screw out-of-plane terms ∂u_dx/∂y and ∂u_dx/∂z (Eq. 1 of Borgi 2025).
    """
    Us, Ud, Theta = identity_rotations
    # Use a deterministic rd that maps directly through identity rotations.
    rd = np.array([[1e-7, 2e-7, 3e-7], [4e-7, 5e-7, 6e-7], [7e-7, 8e-7, 9e-7]])

    Fg = Fd_find_mixed(rd, Us, Ud_mix=Ud, rotation_deg=90.0, Theta=Theta)

    # Identity-frame check: Fg = I + screw_terms (no edge).
    np.testing.assert_allclose(np.diagonal(Fg, axis1=1, axis2=2), 1.0, atol=1e-15)
    np.testing.assert_allclose(Fg[:, 1, 0], 0.0, atol=1e-15)
    np.testing.assert_allclose(Fg[:, 2, 0], 0.0, atol=1e-15)
    np.testing.assert_allclose(Fg[:, 2, 1], 0.0, atol=1e-15)
    np.testing.assert_allclose(Fg[:, 1, 2], 0.0, atol=1e-15)
    sqz = rd[2] ** 2
    sqy = rd[1] ** 2
    denom1 = sqz + sqy + 1e-20
    bfactor1 = BURGERS_VECTOR / (2 * np.pi)
    expected_01 = -rd[2] / denom1 * bfactor1
    expected_02 = rd[1] / denom1 * bfactor1
    np.testing.assert_allclose(Fg[:, 0, 1], expected_01, rtol=1e-12)
    np.testing.assert_allclose(Fg[:, 0, 2], expected_02, rtol=1e-12)


def test_Fd_find_mixed_position_offset_shifts_singularity(identity_rotations):
    """Translating the dislocation core by `position_lab_um` is equivalent to
    evaluating Fd_find_mixed at rl shifted in the opposite direction.
    """
    Us, Ud, Theta = identity_rotations
    rl = np.array([[5e-6, 0.0], [0.0, 0.0], [0.0, 0.0]])

    Fg_at_origin = Fd_find_mixed(rl, Us, Ud_mix=Ud, rotation_deg=0.0, Theta=Theta)

    rl_at_offset = np.array([[5e-6], [0.0], [0.0]])
    Fg_at_offset_pos = Fd_find_mixed(
        rl_at_offset,
        Us,
        Ud_mix=Ud,
        rotation_deg=0.0,
        Theta=Theta,
        position_lab_um=(5.0, 0.0, 0.0),
    )

    diff = np.abs(Fg_at_origin[0] - Fg_at_offset_pos[0]).max()
    assert diff > 1e-6, f"position_lab_um had no effect (diff={diff})"


def test_Fd_find_mixed_uses_module_constants_as_defaults(identity_rotations, simple_rl_grid):
    """Calling without b/ny kwargs uses BURGERS_VECTOR/POISSON_RATIO from constants."""
    Us, Ud, Theta = identity_rotations
    rl = simple_rl_grid

    Fg_default = Fd_find_mixed(rl, Us, Ud_mix=Ud, rotation_deg=30.0, Theta=Theta)
    Fg_explicit = Fd_find_mixed(
        rl,
        Us,
        Ud_mix=Ud,
        rotation_deg=30.0,
        Theta=Theta,
        b=BURGERS_VECTOR,
        ny=POISSON_RATIO,
    )
    np.testing.assert_array_equal(Fg_default, Fg_explicit)
