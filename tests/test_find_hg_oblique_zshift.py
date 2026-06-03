"""S3 (#16): Find_Hg threads ctx.geometry.xl_range into Z_shift for z-scans.

Regression for the missed 5th Z_shift call site: a wall-mode + oblique +
[scan.z] run computes its shifted ray grid via Find_Hg -> Z_shift, which must
use the run's (oblique) xl_range, not the import-time simplified global. The
oblique gate had no z-scan coverage, so this pins the threading directly.
"""

from __future__ import annotations

import numpy as np

import dfxm_geo.direct_space.forward_model as fm


def test_find_hg_zscan_uses_ctx_xl_range(monkeypatch):
    captured: dict[str, float | None] = {}

    def _spy_zshift(offset_um: float, *, xl_range_override: float | None = None) -> np.ndarray:
        captured["xl_range_override"] = xl_range_override
        # shape doesn't matter for this test; load_or_generate_Hg is stubbed.
        return np.zeros((3, 8))

    monkeypatch.setattr(fm, "Z_shift", _spy_zshift)
    # Avoid the heavy Fg compute and the on-disk sidecar write.
    monkeypatch.setattr(fm, "load_or_generate_Hg", lambda *a, **k: np.zeros((8, 3, 3)))
    monkeypatch.setattr(fm.os.path, "exists", lambda p: True)

    # Build a ctx whose geometry.xl_range differs from the module global by using
    # a non-default (oblique-like) Bragg angle.
    instr = fm.build_instrument_context()
    geom = fm.build_geometry_context(0.30, instr)  # ~17.2 deg -> distinct xl_range
    res = fm._context_from_globals().resolution
    ctx = fm.ForwardContext(
        instrument=instr,
        geometry=geom,
        resolution=res,
        q_hkl=np.array([-1.0, 1.0, -1.0]) / np.sqrt(3),
    )
    assert geom.xl_range != fm.xl_range  # sanity: ctx xl_range really differs

    fm.Find_Hg(4.0, 10, fm.psize, fm.zl_rms, ctx=ctx, z_offset_um=5.0, remount_name="S1")

    assert captured["xl_range_override"] == geom.xl_range, (
        "Find_Hg must forward ctx.geometry.xl_range to Z_shift for z-scans, "
        f"got {captured['xl_range_override']!r}, expected {geom.xl_range!r}"
    )


def test_find_hg_zscan_without_ctx_uses_global(monkeypatch):
    """Legacy ctx=None path: Z_shift gets xl_range_override=None (reads the global)."""
    captured: dict[str, float | None] = {}

    def _spy_zshift(offset_um: float, *, xl_range_override: float | None = None) -> np.ndarray:
        captured["xl_range_override"] = xl_range_override
        return np.zeros((3, 8))

    monkeypatch.setattr(fm, "Z_shift", _spy_zshift)
    monkeypatch.setattr(fm, "load_or_generate_Hg", lambda *a, **k: np.zeros((8, 3, 3)))
    monkeypatch.setattr(fm.os.path, "exists", lambda p: True)

    fm.Find_Hg(4.0, 10, fm.psize, fm.zl_rms, z_offset_um=5.0, remount_name="S1")

    assert captured["xl_range_override"] is None
