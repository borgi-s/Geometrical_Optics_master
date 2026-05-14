"""Unit tests for dfxm_geo.crystal.dislocations mixed-character functions.

References:
    Borgi, S., Winther, G., Poulsen, H. F. (2025). J. Appl. Cryst. 58, 813-821.
    DOI: 10.1107/S1600576725002614. Eq. 1 defines the mixed-character F_d.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from dfxm_geo.constants import BURGERS_VECTOR, POISSON_RATIO
from dfxm_geo.crystal.dislocations import (
    Fd_find,
    Fd_find_mixed,
    Fd_find_multi_dislocs_mixed,
    MixedDislocSpec,
)


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

    With position_lab_um=(2µm, 0, 0), `rl_shifted - (2e-6, 0, 0)` internally
    equals `rl_unshifted` — so the two calls must return bit-identical fields.
    """
    Us, Ud, Theta = identity_rotations
    rl_unshifted = np.array([[3e-6], [1e-6], [0.0]])
    rl_shifted = np.array([[5e-6], [1e-6], [0.0]])

    Fg_no_offset = Fd_find_mixed(rl_unshifted, Us, Ud_mix=Ud, rotation_deg=0.0, Theta=Theta)
    Fg_with_offset = Fd_find_mixed(
        rl_shifted,
        Us,
        Ud_mix=Ud,
        rotation_deg=0.0,
        Theta=Theta,
        position_lab_um=(2.0, 0.0, 0.0),
    )

    np.testing.assert_allclose(Fg_no_offset, Fg_with_offset, atol=1e-15, rtol=1e-12)


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


def test_Fd_find_multi_N1_matches_single(identity_rotations, simple_rl_grid):
    """For one crystal, Fd_find_multi_dislocs_mixed must equal Fd_find_mixed."""
    Us, Ud, Theta = identity_rotations
    rl = simple_rl_grid

    spec = MixedDislocSpec(Ud_mix=Ud, rotation_deg=37.0)
    Fg_multi = Fd_find_multi_dislocs_mixed(rl, Us, [spec], Theta)
    Fg_single = Fd_find_mixed(rl, Us, Ud_mix=Ud, rotation_deg=37.0, Theta=Theta)

    np.testing.assert_allclose(Fg_multi, Fg_single, atol=1e-15, rtol=1e-12)


def test_Fd_find_multi_N2_is_superposition(identity_rotations, simple_rl_grid):
    """Two crystals: result = (Fdd1 - I) + (Fdd2 - I) + I."""
    Us, Ud, Theta = identity_rotations
    rl = simple_rl_grid

    spec1 = MixedDislocSpec(Ud_mix=Ud, rotation_deg=10.0, position_lab_um=(1.0, 0.0, 0.0))
    spec2 = MixedDislocSpec(Ud_mix=Ud, rotation_deg=80.0, position_lab_um=(-1.0, 0.0, 0.0))

    Fg_multi = Fd_find_multi_dislocs_mixed(rl, Us, [spec1, spec2], Theta)

    Fg1 = Fd_find_mixed(
        rl, Us, Ud_mix=Ud, rotation_deg=10.0, Theta=Theta, position_lab_um=(1.0, 0.0, 0.0)
    )
    Fg2 = Fd_find_mixed(
        rl, Us, Ud_mix=Ud, rotation_deg=80.0, Theta=Theta, position_lab_um=(-1.0, 0.0, 0.0)
    )
    I = np.identity(3)
    expected = (Fg1 - I) + (Fg2 - I) + I

    np.testing.assert_allclose(Fg_multi, expected, atol=1e-15, rtol=1e-12)


def test_Fd_find_multi_empty_raises(identity_rotations, simple_rl_grid):
    """Empty crystals list is a programmer error — fail loudly."""
    Us, _, Theta = identity_rotations
    with pytest.raises(ValueError, match="at least one"):
        Fd_find_multi_dislocs_mixed(simple_rl_grid, Us, [], Theta)


# --- Sample-remount (S) for mixed-character ---


class TestFdFindMixedSampleRemount:
    """Tests for the S kwarg on Fd_find_mixed (Purdue_Paper port)."""

    def _inputs(self):
        rl = np.linspace(-1.0, 1.0, 12).reshape(1, -1)
        rl = np.vstack([rl, rl, rl])  # (3, 12)
        Us = np.eye(3)
        Theta = np.eye(3)
        # Ud_mix: arbitrary proper rotation. Pick one whose columns are unit
        # length (Eq.3 of Borgi 2025 takes (b, n, t) columns).
        Ud_mix = np.array(
            [
                [1 / np.sqrt(2), 1 / np.sqrt(3), 1 / np.sqrt(6)],
                [-1 / np.sqrt(2), 1 / np.sqrt(3), 1 / np.sqrt(6)],
                [0, -1 / np.sqrt(3), 2 / np.sqrt(6)],
            ]
        )
        return rl, Us, Ud_mix, Theta

    def test_S_kwarg_default_matches_omitted(self) -> None:
        from dfxm_geo.crystal.dislocations import Fd_find_mixed

        rl, Us, Ud_mix, Theta = self._inputs()
        without = Fd_find_mixed(rl, Us, Ud_mix, rotation_deg=30.0, Theta=Theta)
        with_I = Fd_find_mixed(rl, Us, Ud_mix, rotation_deg=30.0, Theta=Theta, S=np.identity(3))
        np.testing.assert_array_equal(without, with_I)

    def test_S2_yields_distinct_output(self) -> None:
        from dfxm_geo.crystal.dislocations import Fd_find_mixed
        from dfxm_geo.crystal.remount import S2

        rl, Us, Ud_mix, Theta = self._inputs()
        with_I = Fd_find_mixed(rl, Us, Ud_mix, rotation_deg=30.0, Theta=Theta, S=np.identity(3))
        with_S2 = Fd_find_mixed(rl, Us, Ud_mix, rotation_deg=30.0, Theta=Theta, S=S2)
        assert not np.allclose(with_I, with_S2)
