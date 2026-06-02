"""Characterization: the static/dynamic split reproduces forward() byte-for-byte.

forward(Hg, phi, chi, 2theta) must equal
forward_from_static(precompute_forward_static(Hg), phi, chi, 2theta) for the
same inputs, with NO numerical drift. These tests assert exact equality, not
allclose -- any difference is a real bug (an accidental reassociation).
"""

from __future__ import annotations

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm


def _ensure_loaded() -> None:
    """Load the default MC kernel if present; skip the test otherwise."""
    from dfxm_geo.pipeline import _lookup_and_load_kernel

    if fm.Resq_i is None and fm._analytic_eval is None:
        try:
            _lookup_and_load_kernel((-1, 1, -1), 17.0)
        except Exception:  # noqa: BLE001 - any load failure -> skip
            pytest.skip("no MC kernel available on this checkout")
    if fm.Resq_i is None and fm._analytic_eval is None:
        pytest.skip("kernel state not initialized")


@pytest.fixture(scope="module")
def loaded_Hg() -> np.ndarray:
    _ensure_loaded()
    Hg, q_hkl = fm.Find_Hg(4.0, 1, fm.psize, fm.zl_rms)
    fm.q_hkl = q_hkl
    return Hg


@pytest.mark.parametrize(
    "phi, chi, two_dtheta",
    [
        (0.0, 0.0, 0.0),
        (1e-4, -2e-4, 0.0),
        (0.0, 0.0, 3e-4),  # exercises the TwoDeltaTheta != 0 Theta branch
        (5e-5, 1e-4, -1e-4),  # all three axes non-zero
    ],
)
def test_split_matches_forward_bitwise(
    loaded_Hg: np.ndarray, phi: float, chi: float, two_dtheta: float
) -> None:
    expected = fm.forward(loaded_Hg, phi=phi, chi=chi, TwoDeltaTheta=two_dtheta)
    base_qc = fm.precompute_forward_static(loaded_Hg)
    actual = fm.forward_from_static(base_qc, phi=phi, chi=chi, TwoDeltaTheta=two_dtheta)
    np.testing.assert_array_equal(actual, expected)


def test_precompute_is_reusable_across_frames(loaded_Hg: np.ndarray) -> None:
    """One base_qc drives many frames; reuse must not mutate it."""
    base_qc = fm.precompute_forward_static(loaded_Hg)
    snapshot = base_qc.copy()
    fm.forward_from_static(base_qc, phi=1e-4, chi=0.0)
    fm.forward_from_static(base_qc, phi=-1e-4, chi=2e-4)
    np.testing.assert_array_equal(base_qc, snapshot)


def test_forward_from_static_qi_return_shape(loaded_Hg: np.ndarray) -> None:
    base_qc = fm.precompute_forward_static(loaded_Hg)
    im, qi_field = fm.forward_from_static(base_qc, phi=0.0, chi=0.0, qi_return=True)
    assert im.shape == (fm.NN2 // fm.Nsub, fm.NN1 // fm.Nsub)
    assert qi_field.shape == (3, fm.NN1, fm.NN2, fm.NN3)


def test_scan_frames_args_carry_base_qc(loaded_Hg: np.ndarray) -> None:
    """_scan_frames_args precomputes base_qc once and ships it per frame."""
    import numpy as np

    from dfxm_geo.pipeline import (
        ScanConfig,
        _build_scan_frames,
        _scan_frames_args,
    )

    scan = ScanConfig.from_dict(
        {"phi": {"range": 6e-4, "steps": 2}, "chi": {"range": 2e-3, "steps": 2}}
    )
    frames = _build_scan_frames(scan)
    args_list, _positioners = _scan_frames_args(loaded_Hg, frames, scan)

    expected_base_qc = fm.precompute_forward_static(loaded_Hg)
    # Every frame tuple carries the SAME base_qc object (shared, read-only);
    # index 1 is base_qc in the (frame_idx, base_qc, phi, chi, 2dt, ctx) tuple.
    first_base_qc = args_list[0][1]
    np.testing.assert_array_equal(first_base_qc, expected_base_qc)
    for args in args_list[1:]:
        assert args[1] is first_base_qc


def test_compute_frame_matches_forward(loaded_Hg: np.ndarray) -> None:
    """_compute_frame(base_qc, ...) image == forward(Hg, ...) image."""
    import numpy as np

    from dfxm_geo.io.hdf5 import _compute_frame

    ctx = fm._context_from_globals()
    base_qc = fm.precompute_forward_static(loaded_Hg, ctx)
    phi, chi, two_dtheta = 1e-4, -2e-4, 3e-4
    idx, im = _compute_frame((7, base_qc, phi, chi, two_dtheta, ctx))
    assert idx == 7
    expected = fm.forward(loaded_Hg, phi=phi, chi=chi, TwoDeltaTheta=two_dtheta)
    np.testing.assert_array_equal(im, expected)
