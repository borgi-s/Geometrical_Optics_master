"""Tests for repo-audit #1: Fg cache filename includes reflection geometry.

TDD tests — written BEFORE the fix. They fail on the old code (no geom suffix,
no sidecar guard) and pass once the fix lands.

Design:
  * geom_suffix = "" for the default reflection (-1,1,-1) → filename unchanged.
  * geom_suffix = f"_hkl{h}_{k}_{l}_th{round(theta_0*1e6)}" for others → new filename.
  * A .geom.json sidecar is written on save; on load, a signature mismatch forces
    regeneration (new sidecar overrides the stale one).
  * sidecar-absent → fall back to the shape guard (back-compat with old caches).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import dfxm_geo.direct_space.forward_model as fm
import dfxm_geo.io.strain_cache as sc

# ---------------------------------------------------------------------------
# Helpers — build a minimal ctx without a real kernel
# ---------------------------------------------------------------------------


def _empty_resolution() -> fm.ResolutionContext:
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
    """A tiny 10x10x3 grid — only geometry scalars matter here."""
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


def _ctx(theta_0: float = 0.3) -> fm.ForwardContext:
    instr = _small_instr()
    geom = fm.build_geometry_context(theta_0, instr)
    return fm.ForwardContext(
        instrument=instr,
        geometry=geom,
        resolution=_empty_resolution(),
        q_hkl=np.array([-1.0, 1.0, -1.0]) / np.sqrt(3),
    )


# ---------------------------------------------------------------------------
# 1. Filename suffix tests
# ---------------------------------------------------------------------------


class TestGeomSuffix:
    """The geom suffix is empty for the default (-1,1,-1) reflection."""

    def test_default_reflection_keeps_legacy_filename(self, tmp_path, monkeypatch):
        """h=-1,k=1,l=-1 must produce NO _hkl... suffix (byte-identity guard)."""
        monkeypatch.setattr(fm, "_REPO_ROOT", tmp_path)
        monkeypatch.setattr(fm, "load_or_generate_Hg", lambda *a, **k: np.zeros((300, 3, 3)))
        monkeypatch.setattr(fm.os.path, "exists", lambda p: True)

        ctx = _ctx(0.3)
        fm.Find_Hg(4.0, 0, fm.psize, fm.zl_rms, h=-1, k=1, l=-1, ctx=ctx)

        # The test checks the PATH PASSED to load_or_generate_Hg via a capturing stub.
        # We re-run with a capturing stub:
        captured: dict[str, str] = {}

        def _capture_path(rl, Ud, Us, Theta, dis, ndis, file_path=None, **kw):
            if file_path is not None:
                captured["path"] = file_path
            return np.zeros((300, 3, 3))

        monkeypatch.setattr(fm, "load_or_generate_Hg", _capture_path)
        fm.Find_Hg(4.0, 0, fm.psize, fm.zl_rms, h=-1, k=1, l=-1, ctx=ctx)
        assert "path" in captured
        name = Path(captured["path"]).name
        assert "_hkl" not in name, (
            f"Default (-1,1,-1) reflection must NOT add a _hkl suffix; got {name!r}"
        )

    def test_non_default_reflection_adds_hkl_suffix(self, tmp_path, monkeypatch):
        """h=1,k=1,k=0 must produce a _hkl1_1_0_th… suffix in the cache filename."""
        monkeypatch.setattr(fm, "_REPO_ROOT", tmp_path)
        monkeypatch.setattr(fm.os.path, "exists", lambda p: True)

        ctx = _ctx(0.3)
        captured: dict[str, str] = {}

        def _capture_path(rl, Ud, Us, Theta, dis, ndis, file_path=None, **kw):
            if file_path is not None:
                captured["path"] = file_path
            return np.zeros((300, 3, 3))

        monkeypatch.setattr(fm, "load_or_generate_Hg", _capture_path)
        fm.Find_Hg(4.0, 0, fm.psize, fm.zl_rms, h=1, k=1, l=0, ctx=ctx)
        assert "path" in captured
        name = Path(captured["path"]).name
        assert "_hkl1_1_0_" in name, (
            f"Non-default (1,1,0) reflection must add a _hkl1_1_0_… suffix; got {name!r}"
        )
        assert "_th" in name, (
            f"Non-default reflection must include a _th… theta token; got {name!r}"
        )

    def test_different_reflections_produce_different_filenames(self, tmp_path, monkeypatch):
        """The same dis/psize/etc but different hkl must yield different Fg paths."""
        monkeypatch.setattr(fm, "_REPO_ROOT", tmp_path)
        monkeypatch.setattr(fm.os.path, "exists", lambda p: True)

        ctx = _ctx(0.3)
        paths: list[str] = []

        def _capture_path(rl, Ud, Us, Theta, dis, ndis, file_path=None, **kw):
            if file_path is not None:
                paths.append(file_path)
            return np.zeros((300, 3, 3))

        monkeypatch.setattr(fm, "load_or_generate_Hg", _capture_path)

        fm.Find_Hg(4.0, 0, fm.psize, fm.zl_rms, h=-1, k=1, l=-1, ctx=ctx)
        fm.Find_Hg(4.0, 0, fm.psize, fm.zl_rms, h=1, k=1, l=0, ctx=ctx)
        assert len(paths) == 2
        assert paths[0] != paths[1], (
            f"Different reflections must produce different Fg paths; both got {paths[0]!r}"
        )

    def test_theta_in_suffix_is_round_to_microradian(self, tmp_path, monkeypatch):
        """The _th… value must be round(theta_0 * 1e6) — microradians integer."""
        monkeypatch.setattr(fm, "_REPO_ROOT", tmp_path)
        monkeypatch.setattr(fm.os.path, "exists", lambda p: True)

        theta_0 = 0.30
        ctx = _ctx(theta_0)
        captured: dict[str, str] = {}

        def _capture_path(rl, Ud, Us, Theta, dis, ndis, file_path=None, **kw):
            if file_path is not None:
                captured["path"] = file_path
            return np.zeros((300, 3, 3))

        monkeypatch.setattr(fm, "load_or_generate_Hg", _capture_path)
        fm.Find_Hg(4.0, 0, fm.psize, fm.zl_rms, h=0, k=0, l=2, ctx=ctx)

        expected_th = round(ctx.geometry.theta_0 * 1e6)
        name = Path(captured["path"]).name
        assert f"_th{expected_th}" in name, f"Expected _th{expected_th} in filename; got {name!r}"


# ---------------------------------------------------------------------------
# 2. Geom sidecar guard tests (strain_cache level)
# ---------------------------------------------------------------------------


class TestGeomSidecarGuard:
    """load_or_generate_Hg writes a .geom.json sidecar and regenerates on mismatch."""

    def _fake_rl(self, n: int = 8) -> np.ndarray:
        return np.zeros((3, n))

    def _fake_Fg(self, n: int = 8) -> np.ndarray:
        Fg = np.zeros((n, 3, 3))
        Fg[:] = np.eye(3)
        return Fg

    def test_sidecar_written_on_save(self, tmp_path):
        """When Fg is generated, a .geom.json sidecar is also saved."""
        file_path = str(tmp_path / "Fg_test.npy")
        sig = (-1, 1, -1, 0.300000001)

        rl = self._fake_rl()
        Ud = np.eye(3)
        Us = np.eye(3)
        Theta = np.eye(3)

        # Stub Fd_find to avoid physics
        original_fd = sc.Fd_find
        try:
            sc.Fd_find = lambda *a, **k: self._fake_Fg()  # type: ignore[attr-defined]
            sc.load_or_generate_Hg(rl, Ud, Us, Theta, 4.0, 1, file_path, geom_signature=sig)
        finally:
            sc.Fd_find = original_fd  # type: ignore[attr-defined]

        sidecar = Path(file_path.replace(".npy", ".geom.json"))
        assert sidecar.exists(), "Expected a .geom.json sidecar to be written"
        data = json.loads(sidecar.read_text())
        assert "geom_signature" in data

    def test_matching_sidecar_loads_cache(self, tmp_path):
        """A cache file with a matching sidecar signature → Fg loaded, not regenerated."""
        file_path = str(tmp_path / "Fg_test.npy")
        sig = (-1, 1, -1, 0.300000001)
        n = 8
        Fg = self._fake_Fg(n)
        np.save(file_path, Fg)

        # Write matching sidecar manually
        sidecar = Path(file_path.replace(".npy", ".geom.json"))
        sidecar.write_text(json.dumps({"geom_signature": list(sig)}))

        rl = self._fake_rl(n)
        Ud = np.eye(3)
        Us = np.eye(3)
        Theta = np.eye(3)

        call_count = {"n": 0}
        original_fd = sc.Fd_find
        try:

            def _counting(*a, **k):
                call_count["n"] += 1
                return self._fake_Fg(n)

            sc.Fd_find = _counting  # type: ignore[attr-defined]
            sc.load_or_generate_Hg(rl, Ud, Us, Theta, 4.0, 1, file_path, geom_signature=sig)
        finally:
            sc.Fd_find = original_fd  # type: ignore[attr-defined]

        assert call_count["n"] == 0, "Fd_find must NOT be called when sidecar matches"

    def test_mismatching_sidecar_forces_regeneration(self, tmp_path):
        """A .geom.json with a different signature → Fg regenerated, not loaded."""
        file_path = str(tmp_path / "Fg_test.npy")
        n = 8
        Fg = self._fake_Fg(n)
        np.save(file_path, Fg)

        # Write WRONG signature in sidecar (different hkl)
        sidecar = Path(file_path.replace(".npy", ".geom.json"))
        sidecar.write_text(json.dumps({"geom_signature": [1, 1, 0, 300000]}))

        correct_sig = (-1, 1, -1, 300000)
        rl = self._fake_rl(n)
        Ud = np.eye(3)
        Us = np.eye(3)
        Theta = np.eye(3)

        call_count = {"n": 0}
        original_fd = sc.Fd_find
        try:

            def _counting(*a, **k):
                call_count["n"] += 1
                return self._fake_Fg(n)

            sc.Fd_find = _counting  # type: ignore[attr-defined]
            sc.load_or_generate_Hg(rl, Ud, Us, Theta, 4.0, 1, file_path, geom_signature=correct_sig)
        finally:
            sc.Fd_find = original_fd  # type: ignore[attr-defined]

        assert call_count["n"] == 1, (
            "Fd_find must be called ONCE to regenerate when sidecar mismatches"
        )

    def test_absent_sidecar_falls_back_to_shape_guard(self, tmp_path):
        """No .geom.json present → back-compat: shape guard only, loads Fg if shape matches."""
        file_path = str(tmp_path / "Fg_test.npy")
        n = 8
        Fg = self._fake_Fg(n)
        np.save(file_path, Fg)
        # Deliberately NO sidecar

        sig = (-1, 1, -1, 300000)
        rl = self._fake_rl(n)
        Ud = np.eye(3)
        Us = np.eye(3)
        Theta = np.eye(3)

        call_count = {"n": 0}
        original_fd = sc.Fd_find
        try:

            def _counting(*a, **k):
                call_count["n"] += 1
                return self._fake_Fg(n)

            sc.Fd_find = _counting  # type: ignore[attr-defined]
            sc.load_or_generate_Hg(rl, Ud, Us, Theta, 4.0, 1, file_path, geom_signature=sig)
        finally:
            sc.Fd_find = original_fd  # type: ignore[attr-defined]

        # With no sidecar, falls back to shape guard — shape matches → loads (no regen).
        assert call_count["n"] == 0, (
            "Absent sidecar → fall back to shape guard; matching shape → load, no regen"
        )

    def test_no_geom_signature_kwarg_works_as_before(self, tmp_path):
        """load_or_generate_Hg called without geom_signature → old behaviour unchanged."""
        file_path = str(tmp_path / "Fg_test.npy")
        n = 8
        Fg = self._fake_Fg(n)
        np.save(file_path, Fg)

        rl = self._fake_rl(n)
        Ud = np.eye(3)
        Us = np.eye(3)
        Theta = np.eye(3)

        call_count = {"n": 0}
        original_fd = sc.Fd_find
        try:

            def _counting(*a, **k):
                call_count["n"] += 1
                return self._fake_Fg(n)

            sc.Fd_find = _counting  # type: ignore[attr-defined]
            # No geom_signature kwarg → old path
            sc.load_or_generate_Hg(rl, Ud, Us, Theta, 4.0, 1, file_path)
        finally:
            sc.Fd_find = original_fd  # type: ignore[attr-defined]

        assert call_count["n"] == 0, "No geom_signature → shape guard only, loads existing cache"

    def test_geom_signature_passed_from_find_hg(self, tmp_path, monkeypatch):
        """Find_Hg must pass geom_signature=(h,k,l,round(theta_0*1e6)) to load_or_generate_Hg."""
        monkeypatch.setattr(fm, "_REPO_ROOT", tmp_path)
        monkeypatch.setattr(fm.os.path, "exists", lambda p: True)

        theta_0 = 0.30
        ctx = _ctx(theta_0)
        captured: dict = {}

        def _capture(**kwargs):
            pass

        def _capturing_lohg(
            rl, Ud, Us, Theta, dis, ndis, file_path=None, *, geom_signature=None, **kw
        ):
            captured["geom_signature"] = geom_signature
            return np.zeros((300, 3, 3))

        monkeypatch.setattr(fm, "load_or_generate_Hg", _capturing_lohg)

        h, k, l = 1, 1, 0
        fm.Find_Hg(4.0, 0, fm.psize, fm.zl_rms, h=h, k=k, l=l, ctx=ctx)

        assert "geom_signature" in captured
        sig = captured["geom_signature"]
        assert sig is not None, "geom_signature must not be None for non-default reflection"
        assert sig[0] == h and sig[1] == k and sig[2] == l, (
            f"geom_signature hkl must be ({h},{k},{l}); got {sig}"
        )
        expected_th = round(ctx.geometry.theta_0 * 1e6)
        assert sig[3] == expected_th, (
            f"geom_signature theta token must be round(theta_0*1e6)={expected_th}; got {sig[3]}"
        )

    def test_default_reflection_geom_signature_is_none_or_default_hkl(self, tmp_path, monkeypatch):
        """For (-1,1,-1) the geom_signature passed to load_or_generate_Hg is either None
        or carries the default hkl — the important thing is NO _hkl suffix in the filename."""
        monkeypatch.setattr(fm, "_REPO_ROOT", tmp_path)
        monkeypatch.setattr(fm.os.path, "exists", lambda p: True)

        theta_0 = 0.30
        ctx = _ctx(theta_0)
        captured: dict = {}

        def _capturing_lohg(
            rl, Ud, Us, Theta, dis, ndis, file_path=None, *, geom_signature=None, **kw
        ):
            captured["path"] = file_path
            captured["geom_signature"] = geom_signature
            return np.zeros((300, 3, 3))

        monkeypatch.setattr(fm, "load_or_generate_Hg", _capturing_lohg)

        fm.Find_Hg(4.0, 0, fm.psize, fm.zl_rms, h=-1, k=1, l=-1, ctx=ctx)

        name = Path(captured["path"]).name
        assert "_hkl" not in name, (
            f"Default reflection must NOT have _hkl suffix in filename; got {name!r}"
        )
