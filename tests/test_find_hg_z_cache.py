"""Unit tests for the Find_Hg z-aware disk cache (v1.3.0-A task 5)."""

from __future__ import annotations

from pathlib import Path

import pytest

import dfxm_geo.direct_space.forward_model as fm


def _require_kernel() -> None:
    """Skip unless a bootstrapped (-1,1,-1) 17 keV kernel npz is on disk."""
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip(f"no kernel npz found in {kernel_dir}")


def test_z_offset_zero_keeps_legacy_filename(tmp_path, monkeypatch):
    """z_offset_um=0.0 must produce the same Fg cache filename as v1.2.0."""
    _require_kernel()
    monkeypatch.setattr(fm, "_REPO_ROOT", tmp_path)
    # Force fresh cache dir so we observe the file that actually lands.
    Fg_dir = tmp_path / "direct_space" / "deformation_gradient_tensors"

    from dfxm_geo.pipeline import _lookup_and_load_kernel

    fm.Hg = None
    fm._loaded_kernel_path = None
    _lookup_and_load_kernel((-1, 1, -1), 17.0)

    fm.Find_Hg(dis=4.0, ndis=1, psize=fm.psize, zl_rms=fm.zl_rms, h=-1, k=1, l=-1)
    files = list(Fg_dir.glob("Fg_*.npy"))
    assert files, f"no Fg cache file landed in {Fg_dir}"
    name = files[0].name
    assert "_z" not in name, f"z=0 should not add z suffix; got {name!r}"


def test_z_offset_nonzero_adds_z_suffix(tmp_path, monkeypatch):
    """z_offset_um=12.5 produces a filename containing _z12500nm (round(z*1000))."""
    _require_kernel()
    monkeypatch.setattr(fm, "_REPO_ROOT", tmp_path)
    Fg_dir = tmp_path / "direct_space" / "deformation_gradient_tensors"
    from dfxm_geo.pipeline import _lookup_and_load_kernel

    fm.Hg = None
    fm._loaded_kernel_path = None
    _lookup_and_load_kernel((-1, 1, -1), 17.0)

    fm.Find_Hg(
        dis=4.0,
        ndis=1,
        psize=fm.psize,
        zl_rms=fm.zl_rms,
        h=-1,
        k=1,
        l=-1,
        z_offset_um=12.5,
    )
    files = list(Fg_dir.glob("Fg_*_z*nm.npy"))
    assert files, f"expected file with _z…nm suffix in {Fg_dir}"
    assert "_z12500nm" in files[0].name


def test_z_offset_nonzero_uses_shifted_rl(tmp_path, monkeypatch):
    """Find_Hg with z_offset_um!=0 must build rl via Z_shift, not use module rl."""
    _require_kernel()
    monkeypatch.setattr(fm, "_REPO_ROOT", tmp_path)
    from dfxm_geo.pipeline import _lookup_and_load_kernel

    fm.Hg = None
    fm._loaded_kernel_path = None
    _lookup_and_load_kernel((-1, 1, -1), 17.0)

    # Spy on Z_shift to confirm it's invoked.
    calls: list[float] = []
    real_z_shift = fm.Z_shift

    def spy(offset_um: float, **kwargs: object):
        # Accept Find_Hg's xl_range_override kwarg (#16 S3) and forward it.
        calls.append(float(offset_um))
        return real_z_shift(offset_um, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(fm, "Z_shift", spy)
    fm.Find_Hg(
        dis=4.0,
        ndis=1,
        psize=fm.psize,
        zl_rms=fm.zl_rms,
        h=-1,
        k=1,
        l=-1,
        z_offset_um=5.0,
    )
    assert calls == [5.0], f"Z_shift should be called once with 5.0; got {calls}"


def test_z_offset_zero_does_not_call_z_shift(tmp_path, monkeypatch):
    """z_offset_um=0.0 must NOT call Z_shift (keep the module-level rl)."""
    _require_kernel()
    monkeypatch.setattr(fm, "_REPO_ROOT", tmp_path)
    from dfxm_geo.pipeline import _lookup_and_load_kernel

    fm.Hg = None
    fm._loaded_kernel_path = None
    _lookup_and_load_kernel((-1, 1, -1), 17.0)

    calls: list[float] = []
    monkeypatch.setattr(fm, "Z_shift", lambda off: calls.append(off) or fm.rl)

    fm.Find_Hg(dis=4.0, ndis=1, psize=fm.psize, zl_rms=fm.zl_rms, h=-1, k=1, l=-1)
    assert calls == [], f"Z_shift should not be called for z=0; got {calls}"


def test_find_hg_from_population_accepts_rl_kwarg():
    """Find_Hg_from_population(population, rl=Z_shift(z)) uses the passed rl."""
    import numpy as np

    _require_kernel()
    from dfxm_geo.pipeline import _lookup_and_load_kernel

    fm.Hg = None
    fm._loaded_kernel_path = None
    _lookup_and_load_kernel((-1, 1, -1), 17.0)

    # Single centered dislocation
    from dfxm_geo.pipeline import CenteredCrystalConfig, CrystalConfig

    cfg = CrystalConfig(
        mode="centered",
        centered=CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 1, -2)),
    )
    pop = fm.build_dislocation_population(cfg, fov_lateral_um=20.0, rng=None)

    rl_shifted = fm.Z_shift(3.0)
    Hg_shifted, _ = fm.Find_Hg_from_population(pop, h=-1, k=1, l=-1, rl=rl_shifted)
    Hg_zero, _ = fm.Find_Hg_from_population(pop, h=-1, k=1, l=-1)
    # Different rl -> different Hg
    assert not np.allclose(Hg_shifted, Hg_zero), "Hg from z-shifted rl should differ from Hg at z=0"
