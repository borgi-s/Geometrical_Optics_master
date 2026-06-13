"""Structural smoke tests for direct_space.forward_model.

These guard the lazy-load refactor: the module must import on a
clean clone (no reciprocal-space kernel npz present), and `forward()`
must raise a clear error when called before kernel state is loaded.

A future test (Phase 7) will add numerical pinning by generating a tiny
fixture kernel and parameterizing the module geometry to small grids.
"""

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm


def _empty_resolution() -> "fm.ResolutionContext":
    """A ResolutionContext with neither MC LUT nor analytic backend."""
    return fm.ResolutionContext(
        Resq_i=None,
        qi1_start=0.0,
        qi1_step=0.0,
        qi2_start=0.0,
        qi2_step=0.0,
        qi3_start=0.0,
        qi3_step=0.0,
        npoints1=None,
        npoints2=None,
        npoints3=None,
        analytic_eval=None,
        loaded_kernel_path=None,
    )


def _stub_ctx() -> "fm.ForwardContext":
    """A ForwardContext with real geometry/instrument but no resolution backend.

    Sufficient for Find_Hg tests that stub ``load_or_generate_Hg`` and only need
    ctx.geometry (rl/Theta/xl_range) + ctx.resolution.loaded_kernel_path; no
    kernel npz is required on disk (#16 Slice 5 — ctx replaces module globals).
    """
    instr = fm.build_instrument_context()
    geom = fm.build_geometry_context(0.30, instr)
    return fm.ForwardContext(
        instrument=instr,
        geometry=geom,
        resolution=_empty_resolution(),
        q_hkl=np.array([-1.0, 1.0, -1.0]) / np.sqrt(3),
    )


def test_module_imports_with_required_symbols():
    """Module-level public surface remains stable across refactors."""
    assert hasattr(fm, "forward")
    assert hasattr(fm, "Find_Hg")
    assert hasattr(fm, "_load_default_kernel")
    # #16 Slice 5: the per-reflection geometry/resolution globals (theta_0,
    # Theta, rl, prob_z, …) are gone — the context builders are the public
    # surface now. Assert the ctx API + the kept instrument constants.
    assert hasattr(fm, "build_forward_context")
    assert hasattr(fm, "build_geometry_context")
    assert hasattr(fm, "build_instrument_context")
    assert hasattr(fm, "ForwardContext")
    # Numerical constants that init_forward.py and others depend on
    assert hasattr(fm, "psize")
    assert hasattr(fm, "zl_rms")
    assert hasattr(fm, "Npixels")
    assert hasattr(fm, "Ud")
    assert hasattr(fm, "Us")
    assert hasattr(fm, "indices")


def _uninitialized_ctx() -> "fm.ForwardContext":
    """A ForwardContext whose resolution has neither an MC LUT nor an analytic
    backend — i.e. the kernel-not-loaded state the forward() guard must catch.

    #16 Slice 5: there is no longer a module-global to null out; the
    uninitialized state lives on ctx.resolution, so build one explicitly.
    """
    return _stub_ctx()


def test_forward_raises_when_kernel_not_loaded():
    """forward() must signal clearly when state hasn't been bootstrapped.

    Regression guard for the lazy-pickle-load refactor (commit 610c445):
    if a future change re-introduces eager pickle loading at import time,
    this test won't break — but if someone removes the kernel-state guard
    from forward(), this test catches it.
    """
    with pytest.raises(RuntimeError, match="not initialized"):
        # ctx is now required; pass one whose resolution is uninitialized so
        # the guard (ctx.resolution.Resq_i is None and analytic_eval is None) fires.
        fm.forward(None, _uninitialized_ctx())


def _default_geometry() -> "fm.GeometryContext":
    """Geometry built from the module-default Bragg angle (#16 Slice 5: the
    per-reflection rl/xl_range live on the GeometryContext, not on globals)."""
    from dfxm_geo.pipeline import (
        ReciprocalConfig,
        SimulationConfig,
        run_theta,
    )

    instr = fm.build_instrument_context()
    cfg = SimulationConfig(reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0))
    return fm.build_geometry_context(run_theta(cfg), instr)


def test_Z_shift_zero_offset_matches_module_rl():
    """Z_shift(0.0, xl_range=...) reproduces the geometry rl grid bit-for-bit."""
    geom = _default_geometry()
    rl_shifted = fm.Z_shift(0.0, xl_range=geom.xl_range)
    np.testing.assert_array_equal(rl_shifted, geom.rl)


def test_Z_shift_shifts_z_column_only():
    """Z_shift(5.0, xl_range=...) shifts z by 5 µm; x and y unchanged."""
    geom = _default_geometry()
    offset_um = 5.0
    rl_shifted = fm.Z_shift(offset_um, xl_range=geom.xl_range)
    np.testing.assert_array_equal(rl_shifted[0], geom.rl[0])
    np.testing.assert_array_equal(rl_shifted[1], geom.rl[1])
    # The z column is shifted by -offset_um * 1e-6 m (Z_shift moves the
    # dislocation core *up* in lab z, equivalent to translating rl *down*).
    np.testing.assert_allclose(rl_shifted[2], geom.rl[2] - offset_um * 1e-6, atol=1e-18, rtol=1e-15)


class TestFindHgSampleRemount:
    """Find_Hg passes the resolved S and remount_name through correctly."""

    def test_find_hg_passes_S_and_filename_to_load_or_generate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Fg cache filename includes _remount{name} and S kwarg arrives."""
        import dfxm_geo.direct_space.forward_model as fm
        from dfxm_geo.constants import BURGERS_VECTOR, POISSON_RATIO
        from dfxm_geo.crystal.remount import S2

        captured: dict = {}

        def fake_load(rl, Ud, Us, Theta, dis, ndis, file_path=None, *, b=None, ny=None, S=None):
            captured["file_path"] = file_path
            captured["S"] = S
            captured["b"] = b
            captured["ny"] = ny
            # Return a plausibly-shaped Fg-derived Hg
            return np.zeros((rl.shape[1], 3, 3))

        monkeypatch.setattr("dfxm_geo.direct_space.forward_model.load_or_generate_Hg", fake_load)

        Hg, q_hkl = fm.Find_Hg(
            dis=4,
            ndis=2,
            psize=fm.psize,
            zl_rms=fm.zl_rms,
            S=S2,
            remount_name="S2",
            ctx=_stub_ctx(),
        )

        assert captured["file_path"] is not None
        assert "_remountS2.npy" in captured["file_path"]
        np.testing.assert_array_equal(captured["S"], S2)
        # Default |b| is the FCC calibrated constant (byte-identical to v2.x).
        assert captured["b"] == BURGERS_VECTOR
        # Default ν is the Al POISSON_RATIO (byte-identical to v2.x).
        assert captured["ny"] == POISSON_RATIO

    def test_find_hg_default_uses_S1_filename(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Omitting S / remount_name defaults to identity / 'S1'."""
        import dfxm_geo.direct_space.forward_model as fm

        captured: dict = {}

        def fake_load(rl, Ud, Us, Theta, dis, ndis, file_path=None, *, b=None, ny=None, S=None):
            captured["file_path"] = file_path
            captured["S"] = S
            captured["b"] = b
            captured["ny"] = ny
            return np.zeros((rl.shape[1], 3, 3))

        monkeypatch.setattr("dfxm_geo.direct_space.forward_model.load_or_generate_Hg", fake_load)

        fm.Find_Hg(dis=4, ndis=2, psize=fm.psize, zl_rms=fm.zl_rms, ctx=_stub_ctx())

        assert "_remountS1.npy" in captured["file_path"]
        np.testing.assert_array_equal(captured["S"], np.identity(3))

    def test_find_hg_rejects_unknown_remount_name(self) -> None:
        """Find_Hg raises ValueError on a remount_name not in SAMPLE_REMOUNT_OPTIONS.

        Hardens the surface for ad-hoc / notebook callers that bypass
        CrystalConfig validation. Without this guard, a typo silently writes
        a junk cache filename like Fg_..._remountbogus.npy.
        """
        import dfxm_geo.direct_space.forward_model as fm

        with pytest.raises(ValueError, match="Unknown remount_name 'bogus'"):
            fm.Find_Hg(
                dis=4,
                ndis=2,
                psize=fm.psize,
                zl_rms=fm.zl_rms,
                remount_name="bogus",
                ctx=_stub_ctx(),
            )
