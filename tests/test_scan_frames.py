"""Unit tests for ScanFrames + _build_scan_frames (v1.3.0-A)."""

from __future__ import annotations

import numpy as np

from dfxm_geo.pipeline import (
    AxisScanConfig,
    ScanConfig,
    ScanFrames,
    _build_scan_frames,
)


def test_zero_scanned_axes_yields_one_frame():
    """Single mode: all axes fixed -> n_frames = 1, all per-frame arrays length 1."""
    cfg = ScanConfig()
    frames = _build_scan_frames(cfg)
    assert isinstance(frames, ScanFrames)
    assert frames.n_frames == 1
    assert frames.phi_pf.shape == (1,)
    assert frames.chi_pf.shape == (1,)
    assert frames.two_dtheta_pf.shape == (1,)
    assert frames.z_pf.shape == (1,)
    # All values come from the axis `.value` (defaults: 0).
    assert frames.phi_pf[0] == 0.0
    assert frames.z_pf[0] == 0.0


def test_phi_only_scanned_matches_legacy_layout():
    """Phi-only rocking: phi values walk linspace, others repeat."""
    cfg = ScanConfig(phi=AxisScanConfig(range=1e-3, steps=5))
    frames = _build_scan_frames(cfg)
    assert frames.n_frames == 5
    np.testing.assert_allclose(frames.phi_pf, np.linspace(-1e-3, 1e-3, 5))
    np.testing.assert_array_equal(frames.chi_pf, np.zeros(5))
    np.testing.assert_array_equal(frames.two_dtheta_pf, np.zeros(5))
    np.testing.assert_array_equal(frames.z_pf, np.zeros(5))


def test_phi_chi_ordering_phi_innermost():
    """Mosa: 3 phi x 2 chi = 6 frames; phi cycles inside chi."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=3),
        chi=AxisScanConfig(range=2e-3, steps=2),
    )
    frames = _build_scan_frames(cfg)
    assert frames.n_frames == 6
    phi_grid = np.linspace(-1e-3, 1e-3, 3)
    chi_grid = np.linspace(-2e-3, 2e-3, 2)
    # phi-innermost: frames 0,1,2 share chi[0]; frames 3,4,5 share chi[1].
    np.testing.assert_allclose(frames.phi_pf, np.tile(phi_grid, 2))
    np.testing.assert_allclose(frames.chi_pf, np.repeat(chi_grid, 3))


def test_four_axes_cartesian_product():
    """All 4 axes scanned: n_frames = product; phi-innermost, z-outermost."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1.0, steps=2),
        chi=AxisScanConfig(range=2.0, steps=2),
        two_dtheta=AxisScanConfig(range=3.0, steps=2),
        z=AxisScanConfig(range=4.0, steps=2),
    )
    frames = _build_scan_frames(cfg)
    assert frames.n_frames == 16  # 2^4
    # phi-innermost means phi values toggle every frame.
    # z-outermost means the first 8 frames share z[0]=-4, the last 8 share z[1]=+4.
    np.testing.assert_allclose(frames.z_pf[:8], -4.0)
    np.testing.assert_allclose(frames.z_pf[8:], 4.0)
    # Two_dtheta cycles every 4 frames (between phi/chi inner and z outer).
    np.testing.assert_allclose(frames.two_dtheta_pf[:4], -3.0)
    np.testing.assert_allclose(frames.two_dtheta_pf[4:8], 3.0)


def test_n_frames_matches_array_length():
    """frames.n_frames is consistent with the per-frame array length."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=4),
        two_dtheta=AxisScanConfig(range=1e-3, steps=3),
    )
    frames = _build_scan_frames(cfg)
    assert frames.n_frames == 12
    assert frames.phi_pf.size == 12
    assert frames.two_dtheta_pf.size == 12
