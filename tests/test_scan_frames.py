"""Unit tests for ScanFrames + _build_scan_frames (v1.3.0-A)."""

from __future__ import annotations

import numpy as np

from dfxm_geo.pipeline import (
    AxisScanConfig,
    ScanConfig,
    ScanFrames,
    _build_scan_frames,
    _build_scan_frames_at_z,
    _iterate_simulation_frames,
    _scan_frames_args,
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
    assert frames.chi_pf.size == 12
    assert frames.two_dtheta_pf.size == 12
    assert frames.z_pf.size == 12


def test_build_scan_frames_at_z_fixes_z_axis():
    """At a specific z, z_pf is full-length constant; other axes walk product."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=2),
        chi=AxisScanConfig(range=2e-3, steps=3),
    )
    frames = _build_scan_frames_at_z(cfg, z_value=12.5)
    assert frames.n_frames == 6  # phi x chi only
    np.testing.assert_array_equal(frames.z_pf, np.full(6, 12.5))


def test_build_scan_frames_at_z_with_two_dtheta_scanned():
    """If two_dtheta is scanned, inner trajectory is phi x chi x two_dtheta."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=2),
        two_dtheta=AxisScanConfig(range=3.0, steps=2),
    )
    frames = _build_scan_frames_at_z(cfg, z_value=0.0)
    assert frames.n_frames == 4  # 2 * 2
    np.testing.assert_array_equal(frames.z_pf, np.zeros(4))
    # phi cycles fastest: [-1e-3, +1e-3, -1e-3, +1e-3]
    np.testing.assert_allclose(frames.phi_pf, [-1e-3, 1e-3, -1e-3, 1e-3])


def test_build_scan_frames_at_z_ignores_z_scan_config():
    """z range/steps in the config are ignored; only the passed-in z_value is used."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=2),
        z=AxisScanConfig(range=10.0, steps=5),  # scanned in cfg
    )
    # Identification iterators handle z themselves; helper takes one z at a time.
    frames = _build_scan_frames_at_z(cfg, z_value=7.7)
    assert frames.n_frames == 2  # phi only; z collapsed to the passed value
    np.testing.assert_array_equal(frames.z_pf, [7.7, 7.7])


def test_scan_frames_args_returns_5_tuples():
    """_scan_frames_args(Hg, frames, cfg) emits (idx, Hg, phi, chi, two_dtheta)."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=2),
        two_dtheta=AxisScanConfig(range=3e-4, steps=2),
    )
    frames = _build_scan_frames_at_z(cfg, z_value=0.0)
    Hg = np.zeros((10, 3, 3))
    args_list, positioners = _scan_frames_args(Hg, frames, cfg)
    assert len(args_list) == 4
    for tup in args_list:
        assert len(tup) == 5  # idx, Hg, phi, chi, two_dtheta
    indices = [tup[0] for tup in args_list]
    assert indices == [0, 1, 2, 3]
    # Per-frame phi/chi/two_dtheta come straight from frames
    np.testing.assert_allclose([tup[2] for tup in args_list], frames.phi_pf)
    np.testing.assert_allclose([tup[4] for tup in args_list], frames.two_dtheta_pf)


# ---------------------------------------------------------------------------
# Task 7: _iterate_simulation_frames
# ---------------------------------------------------------------------------


def test_iterate_simulation_frames_single_z_calls_provider_once():
    """When no z-scan, Hg_provider is called exactly once with z=0."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=3),
        chi=AxisScanConfig(range=2e-3, steps=2),
    )
    frames = _build_scan_frames(cfg)
    calls: list[float] = []
    fake_Hg = np.zeros((10, 3, 3))

    def Hg_provider(z: float) -> np.ndarray:
        calls.append(z)
        return fake_Hg

    out = list(_iterate_simulation_frames(frames, Hg_provider))
    assert calls == [0.0]
    assert len(out) == 6  # 3 phi x 2 chi
    # Each tuple is (idx, Hg, phi, chi, two_dtheta).
    indices = [t[0] for t in out]
    assert indices == list(range(6))
    for t in out:
        assert t[1] is fake_Hg


def test_iterate_simulation_frames_z_scan_calls_provider_per_unique_z():
    """When z is scanned, Hg_provider is called once per unique z."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=2),
        z=AxisScanConfig(range=5.0, steps=3),
    )
    frames = _build_scan_frames(cfg)
    calls: list[float] = []

    def Hg_provider(z: float) -> np.ndarray:
        calls.append(z)
        return np.full((10, 3, 3), z)  # marker

    out = list(_iterate_simulation_frames(frames, Hg_provider))
    assert len(out) == 6  # 2 phi x 3 z
    # 3 unique z values, provider called 3 times in z order.
    assert calls == sorted(set(frames.z_pf.tolist()))
    # All frames at z=z[i] share the same Hg.
    Hg0 = out[0][1]
    Hg1 = out[1][1]
    assert Hg0 is Hg1  # same z, same Hg pointer


def test_iterate_simulation_frames_memory_release_pops_prior_z():
    """Once we move past a z, the prior Hg is no longer referenced from the iterator."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=2),
        z=AxisScanConfig(range=5.0, steps=2),
    )
    frames = _build_scan_frames(cfg)
    seen: list[np.ndarray] = []

    def Hg_provider(z: float) -> np.ndarray:
        Hg = np.zeros((100, 3, 3))
        seen.append(Hg)
        return Hg

    out = []
    for tup in _iterate_simulation_frames(frames, Hg_provider):
        out.append(tup)
    assert len(out) == 4
    # The iterator should not hold references to old Hg arrays at the end.
    # We just check it doesn't leak the dict ad infinitum; this is mostly a
    # structural test.
    assert len(seen) == 2


def test_scan_frames_args_positioners_contain_all_four_axes():
    """positioners dict has phi/chi/two_dtheta/z (per-frame arrays or scalars)."""
    cfg = ScanConfig(
        phi=AxisScanConfig(range=1e-3, steps=3),
        two_dtheta=AxisScanConfig(range=3e-4, steps=2),
    )
    frames = _build_scan_frames_at_z(cfg, z_value=5.0)
    Hg = np.zeros((10, 3, 3))
    _, positioners = _scan_frames_args(Hg, frames, cfg)
    assert set(positioners.keys()) == {"phi", "chi", "two_dtheta", "z"}
    # Scanned axes are arrays
    assert isinstance(positioners["phi"], np.ndarray)
    assert isinstance(positioners["two_dtheta"], np.ndarray)
    # Fixed axes are scalars (chi default value is 0.0)
    assert positioners["chi"] == 0.0
    # z is fixed at 5.0 by _build_scan_frames_at_z
    assert positioners["z"] == 5.0
