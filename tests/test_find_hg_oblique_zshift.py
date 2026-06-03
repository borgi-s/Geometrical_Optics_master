"""S3 (#16): Find_Hg threads ctx.geometry.xl_range into Z_shift for z-scans.

Regression for the missed 5th Z_shift call site: a wall-mode + oblique +
[scan.z] run computes its shifted ray grid via Find_Hg -> Z_shift, which must
use the run's (oblique) xl_range, not the import-time simplified global. The
oblique gate had no z-scan coverage, so this pins the threading directly.
"""

from __future__ import annotations

import numpy as np

import dfxm_geo.direct_space.forward_model as fm


def _empty_resolution() -> fm.ResolutionContext:
    """A ResolutionContext with no backend; sufficient when Find_Hg's
    load_or_generate_Hg is stubbed (only ctx.geometry is exercised)."""
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


def _small_instr() -> fm.InstrumentContext:
    """A tiny-grid instrument. These tests stub load_or_generate_Hg and only
    read the scalar geometry.xl_range (theta-dependent, grid-size-independent),
    so a 10x10x3 grid avoids building a ~35 MB px510 ray grid for nothing."""
    return fm.InstrumentContext(
        psize=fm.psize,
        zl_rms=fm.zl_rms,
        Npixels=30,
        Nsub=1,
        NN1=10,
        NN2=10,
        NN3=3,
        Ud=fm.Ud,
        Us=fm.Us,
        flat_indices=np.zeros(300, dtype=np.int64),
        yl_start=fm.yl_start,
        xl_steps=10,
        yl_steps=10,
        zl_steps=3,
    )


def test_find_hg_zscan_uses_ctx_xl_range(monkeypatch):
    captured: dict[str, float | None] = {}

    def _spy_zshift(offset_um: float, *, xl_range: float) -> np.ndarray:
        captured["xl_range"] = xl_range
        # shape doesn't matter for this test; load_or_generate_Hg is stubbed.
        return np.zeros((3, 8))

    monkeypatch.setattr(fm, "Z_shift", _spy_zshift)
    # Avoid the heavy Fg compute and the on-disk sidecar write.
    monkeypatch.setattr(fm, "load_or_generate_Hg", lambda *a, **k: np.zeros((8, 3, 3)))
    monkeypatch.setattr(fm.os.path, "exists", lambda p: True)

    # Build a ctx whose geometry.xl_range differs from the default geometry by
    # using a non-default (oblique-like) Bragg angle.
    instr = _small_instr()
    geom = fm.build_geometry_context(0.30, instr)  # ~17.2 deg -> distinct xl_range
    default_geom = fm.build_geometry_context(0.34, instr)  # ~19.5 deg, distinct
    ctx = fm.ForwardContext(
        instrument=instr,
        geometry=geom,
        resolution=_empty_resolution(),
        q_hkl=np.array([-1.0, 1.0, -1.0]) / np.sqrt(3),
    )
    assert geom.xl_range != default_geom.xl_range  # sanity: xl_range is theta-dependent

    fm.Find_Hg(4.0, 10, fm.psize, fm.zl_rms, ctx=ctx, z_offset_um=5.0, remount_name="S1")

    assert captured["xl_range"] == geom.xl_range, (
        "Find_Hg must forward ctx.geometry.xl_range to Z_shift for z-scans, "
        f"got {captured['xl_range']!r}, expected {geom.xl_range!r}"
    )


def test_find_hg_zscan_uses_module_xl_range_for_default_ctx(monkeypatch):
    """A ctx built from the default Bragg angle forwards the matching
    xl_range to Z_shift (the #16 Slice 5 successor to the deleted ctx=None
    global-fallback path — geometry now always flows through ctx).
    """
    captured: dict[str, float | None] = {}

    def _spy_zshift(offset_um: float, *, xl_range: float) -> np.ndarray:
        captured["xl_range"] = xl_range
        return np.zeros((3, 8))

    monkeypatch.setattr(fm, "Z_shift", _spy_zshift)
    monkeypatch.setattr(fm, "load_or_generate_Hg", lambda *a, **k: np.zeros((8, 3, 3)))
    monkeypatch.setattr(fm.os.path, "exists", lambda p: True)

    instr = _small_instr()
    geom = fm.build_geometry_context(0.30, instr)
    ctx = fm.ForwardContext(
        instrument=instr,
        geometry=geom,
        resolution=_empty_resolution(),
        q_hkl=np.array([-1.0, 1.0, -1.0]) / np.sqrt(3),
    )

    fm.Find_Hg(4.0, 10, fm.psize, fm.zl_rms, ctx=ctx, z_offset_um=5.0, remount_name="S1")

    assert captured["xl_range"] == geom.xl_range
