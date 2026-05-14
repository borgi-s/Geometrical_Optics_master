"""Unit tests for dfxm_geo.crystal.dislocations mixed-character functions.

References:
    Borgi, S., Winther, G., Poulsen, H. F. (2025). J. Appl. Cryst. 58, 813-821.
    DOI: 10.1107/S1600576725002614. Eq. 1 defines the mixed-character F_d.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from dfxm_geo.crystal.dislocations import MixedDislocSpec


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
