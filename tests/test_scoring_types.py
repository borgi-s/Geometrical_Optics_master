import numpy as np
import pytest

from dfxm_geo.scoring.types import CandidateLabel, CandidateLibrary, GridSpec


def _label(plane=(1, 1, 1), b=(1, 0, 1), alpha=0.0):
    return CandidateLabel(
        slip_plane_normal=plane,
        burgers=b,
        rotation_deg=alpha,
        gb_cos=0.8,
        gb_visible=True,
        q_hkl=(0.0, 2.0, 0.0),
        scan_index=1,
        source_file="m.h5",
    )


def test_class_key_modes():
    lbl = _label(plane=(1, 1, 1), b=(1, 0, 1), alpha=30.0)
    assert lbl.class_key("plane_burgers") == ((1, 1, 1), (1, 0, 1))
    assert lbl.class_key("burgers") == ((1, 0, 1),)
    assert lbl.class_key("plane_burgers_alpha") == ((1, 1, 1), (1, 0, 1), 30.0)
    assert hash(lbl.class_key("plane_burgers"))  # hashable -> usable as dict key


def test_class_key_unknown_mode():
    with pytest.raises(ValueError):
        _label().class_key("nonsense")


def test_library_len():
    frames = np.zeros((2, 4, 4), dtype=np.float32)
    lib = CandidateLibrary(
        frames=frames, labels=[_label(), _label()], grid=GridSpec(pitch_um=(0.1, 0.1), shape=(4, 4))
    )
    assert len(lib) == 2
