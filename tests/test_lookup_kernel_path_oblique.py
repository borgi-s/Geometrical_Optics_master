"""_lookup_kernel_path resolves both simplified (legacy) and oblique LUT patterns."""

import numpy as np
import pytest

from dfxm_geo.direct_space.forward_model import _lookup_kernel_path


def test_finds_legacy_simplified_lut(tmp_path) -> None:
    """A v2.2.0-era LUT (legacy filename, hkl in meta) is found in simplified mode."""
    path = tmp_path / "Resq_i_h-1_k1_l-1_17keV_20260528_1500.npz"
    np.savez(
        path,
        kernel=np.zeros((4, 4, 4)),
        hkl=np.array([-1, 1, -1], dtype=np.int64),
        keV=np.float64(17.0),
        geometry_mode=np.str_("simplified"),
        eta=np.float64(0.0),
    )
    found = _lookup_kernel_path(
        directory=tmp_path,
        mode="simplified",
        hkl=(-1, 1, -1),
        keV=17.0,
    )
    assert found == path


def test_finds_oblique_lut_by_theta_eta_keV(tmp_path) -> None:
    """An oblique-mode LUT is found via (θ, η, keV) tuple."""
    path = tmp_path / "Resq_i_theta0.2691rad_eta0.3531rad_19.1keV_20260528_1500.npz"
    np.savez(
        path,
        kernel=np.zeros((4, 4, 4)),
        theta=np.float64(0.2691),
        eta=np.float64(0.3531),
        keV=np.float64(19.1),
        geometry_mode=np.str_("oblique"),
        hkl=np.array([-1, -1, 3], dtype=np.int64),
    )
    found = _lookup_kernel_path(
        directory=tmp_path,
        mode="oblique",
        theta=0.2691,
        eta=0.3531,
        keV=19.1,
    )
    assert found == path


def test_missing_oblique_lut_raises_with_bootstrap_hint(tmp_path) -> None:
    with pytest.raises(KeyError, match="dfxm-bootstrap"):
        _lookup_kernel_path(
            directory=tmp_path,
            mode="oblique",
            theta=0.2691,
            eta=0.3531,
            keV=19.1,
        )


def test_v220_era_lut_loadable_in_simplified_mode(tmp_path) -> None:
    """A LUT written BEFORE the eta metadata was added still loads in simplified mode."""
    path = tmp_path / "Resq_i_h-1_k1_l-1_17keV_20260520_0000.npz"
    np.savez(
        path,
        kernel=np.zeros((4, 4, 4)),
        hkl=np.array([-1, 1, -1], dtype=np.int64),
        keV=np.float64(17.0),
        # NO eta or geometry_mode keys
    )
    found = _lookup_kernel_path(
        directory=tmp_path,
        mode="simplified",
        hkl=(-1, 1, -1),
        keV=17.0,
    )
    assert found == path
