"""Lane B-pipeline-io backlog regressions.

Covers:
  #4  `_dataclass_to_toml_str` must emit the [geometry] block (mode, eta) and
      the crystal mount fields (lattice/a/mount_x/y/z) so an oblique
      SimulationConfig round-trips through SimulationConfig.from_toml (the
      serialized string is embedded as HDF5 provenance — a lossy round-trip
      misattributes oblique runs as 'simplified').
  #13a write_strain_provenance=False must drop the per-ray Hg in the
      identification writer too (it was a silent no-op in identify).
  #13b non-wall crystal modes must NOT write a meaningless sample/dis value:
      the dataset is omitted entirely when sample_dis is None.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
import dfxm_geo.io.hdf5 as hdf5_mod
from dfxm_geo.io.hdf5 import (
    ScanSpec,
    write_identification_h5,
    write_simulation_h5,
)
from dfxm_geo.pipeline import (
    GeometryConfig,
    ReciprocalConfig,
    ScanFrames,
    SimulationConfig,
    _dataclass_to_toml_str,
)

# Known-good oblique geometry from configs/al_oblique_figure3.toml.
_OBLIQUE_ETA = 0.353140
_OBLIQUE_HKL = (-1, -1, 3)
_OBLIQUE_KEV = 19.1
_OBLIQUE_A = 4.0493e-10

_TINY_HG = np.zeros((1, 3, 3))
_TINY_Q = np.array([-1.0, -1.0, 3.0])


# ---------------------------------------------------------------------------
# #4 — oblique [geometry] + crystal mount survive the TOML round-trip
# ---------------------------------------------------------------------------


def _oblique_sim_config() -> SimulationConfig:
    """Build an oblique SimulationConfig via the real from_toml machinery.

    Reuses the paper Fig-3 reflection so the eta cross-check in
    _build_geometry_config actually resolves (theta, omega).
    """
    import tempfile

    toml_text = (
        "[reciprocal]\n"
        f"hkl = [{_OBLIQUE_HKL[0]}, {_OBLIQUE_HKL[1]}, {_OBLIQUE_HKL[2]}]\n"
        f"keV = {_OBLIQUE_KEV}\n"
        "beamstop = false\n"
        "\n"
        "[crystal]\n"
        'lattice = "cubic"\n'
        f"a = {_OBLIQUE_A}\n"
        "mount_x = [1, 0, 0]\n"
        "mount_y = [0, 1, 0]\n"
        "mount_z = [0, 0, 1]\n"
        'mode = "centered"\n'
        "[crystal.centered]\n"
        "b = [1, -1, 0]\n"
        "n = [1, 1, -1]\n"
        "t = [1, 1, 2]\n"
        "\n"
        "[geometry]\n"
        'mode = "oblique"\n'
        f"eta = {_OBLIQUE_ETA}\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write(toml_text)
        p = Path(fh.name)
    try:
        return SimulationConfig.from_toml(p)
    finally:
        p.unlink(missing_ok=True)


def test_oblique_geometry_round_trips_through_toml_str(tmp_path: Path) -> None:
    cfg = _oblique_sim_config()
    assert cfg.geometry.mode == "oblique"  # sanity on the fixture

    rendered = _dataclass_to_toml_str(cfg)
    out = tmp_path / "round.toml"
    out.write_text(rendered, encoding="utf-8")
    cfg2 = SimulationConfig.from_toml(out)

    assert cfg2.geometry.mode == "oblique"
    assert np.isclose(cfg2.geometry.eta, _OBLIQUE_ETA, atol=1e-9)
    # The mount must survive too (it is what the eta cross-check resolves
    # theta/omega from; a dropped mount would re-derive a different geometry).
    assert cfg2.geometry.mount is not None
    np.testing.assert_array_equal(cfg2.geometry.mount.mount_x, (1, 0, 0))
    np.testing.assert_array_equal(cfg2.geometry.mount.mount_y, (0, 1, 0))
    np.testing.assert_array_equal(cfg2.geometry.mount.mount_z, (0, 0, 1))
    assert np.isclose(cfg2.geometry.mount.a, _OBLIQUE_A, rtol=1e-9)
    # Full geometry equality (theta_validated + omega re-derived identically).
    assert cfg2.geometry == cfg.geometry


def test_simplified_geometry_omits_geometry_block(tmp_path: Path) -> None:
    """A simplified config must not gain a spurious [geometry] block."""
    cfg = SimulationConfig(reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0))
    assert cfg.geometry == GeometryConfig()  # default simplified
    rendered = _dataclass_to_toml_str(cfg)
    assert "[geometry]" not in rendered
    # And it still round-trips back to simplified.
    out = tmp_path / "simple.toml"
    out.write_text(rendered, encoding="utf-8")
    cfg2 = SimulationConfig.from_toml(out)
    assert cfg2.geometry.mode == "simplified"


# ---------------------------------------------------------------------------
# Shared HDF5 stub fixture (no bootstrapped kernel needed)
# ---------------------------------------------------------------------------


def _make_minimal_detector_file(path: Path, n_frames: int = 1, h: int = 4, w: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["NX_class"] = "NXroot"
        entry = f.create_group("entry_0000")
        entry.attrs["NX_class"] = "NXentry"
        det = entry.create_group("dfxm_sim_detector")
        det.attrs["NX_class"] = "NXdetector"
        det.create_dataset("image", data=np.zeros((n_frames, h, w), dtype=np.float32))


def _fake_parallel_writer(
    path: Path,
    args_list: list,
    *,
    max_workers: int | None = None,
    detector_shape: tuple[int, int] | None = None,
) -> None:
    _make_minimal_detector_file(path, n_frames=max(1, len(args_list)))


@pytest.fixture()
def _fm_stub(monkeypatch: pytest.MonkeyPatch) -> fm.ForwardContext:
    """Stub the render + return a ForwardContext for the writers (#16 Slice 5).

    The render is stubbed (_fake_parallel_writer + a no-op precompute), so the
    ctx only needs a backend that satisfies the writer's "backend loaded" guard;
    an analytic_eval sentinel keeps kernel_npz None (no real kernel SHA).
    """
    monkeypatch.setattr(fm, "psize", 4e-8)
    monkeypatch.setattr(fm, "zl_rms", 1e-7)
    monkeypatch.setattr(fm, "precompute_forward_static", lambda Hg_in, ctx=None: np.zeros((3, 1)))
    monkeypatch.setattr(
        hdf5_mod, "_compute_and_write_detector_file_parallel", _fake_parallel_writer
    )
    res = fm.ResolutionContext(
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
        analytic_eval=object(),
        loaded_kernel_path=None,
    )
    return fm.build_forward_context(0.165, res, (-1, 1, -1))


def _single_frame() -> ScanFrames:
    return ScanFrames(
        phi_pf=np.zeros(1),
        chi_pf=np.zeros(1),
        two_dtheta_pf=np.zeros(1),
        z_pf=np.zeros(1),
        n_frames=1,
    )


def _identify_scanspec(write_hg: bool) -> ScanSpec:
    dfxm_geo: dict = {
        "q_hkl": _TINY_Q,
        "theta": 0.165,
        "psize": 4e-8,
        "zl_rms": 1e-7,
    }
    if write_hg:
        dfxm_geo["Hg"] = _TINY_HG
    return ScanSpec(
        title="fscan2d phi 0 0 1 chi 0 0 1 1.0",
        sample={"name": "simulated, dislocation identification (single)"},
        positioners={"phi": 0.0, "chi": 0.0, "two_dtheta": 0.0},
        dfxm_geo=dfxm_geo,
        detectors={"dfxm_sim_detector": [(0, np.zeros((3, 1)), 0.0, 0.0, 0.0)]},
        attrs={"identify_mode": "single"},
    )


# ---------------------------------------------------------------------------
# #13a — write_strain_provenance gates Hg in the identification writer
# ---------------------------------------------------------------------------


def test_identify_writes_hg_by_default(tmp_path: Path, _fm_stub: fm.ForwardContext) -> None:
    write_identification_h5(
        tmp_path,
        scan_iter=[_identify_scanspec(write_hg=True)],
        cli="test",
        config_toml="",
        ctx=_fm_stub,
    )
    with h5py.File(tmp_path / hdf5_mod.MASTER_IDENTIFY, "r") as f:
        assert "/1.1/dfxm_geo/Hg" in f
        assert "/1.1/dfxm_geo/q_hkl" in f


def test_identify_omits_hg_when_strain_provenance_disabled(
    tmp_path: Path, _fm_stub: fm.ForwardContext
) -> None:
    write_identification_h5(
        tmp_path,
        scan_iter=[_identify_scanspec(write_hg=True)],
        cli="test",
        config_toml="",
        write_strain_provenance=False,
        ctx=_fm_stub,
    )
    with h5py.File(tmp_path / hdf5_mod.MASTER_IDENTIFY, "r") as f:
        assert "/1.1/dfxm_geo/Hg" not in f  # big array dropped
        assert "/1.1/dfxm_geo/q_hkl" in f  # tiny scalars retained


# ---------------------------------------------------------------------------
# #13b — non-wall modes omit the sample/dis dataset (no -1.0 sentinel)
# ---------------------------------------------------------------------------


def test_non_wall_mode_omits_sample_dis(tmp_path: Path, _fm_stub: fm.ForwardContext) -> None:
    out = tmp_path / "dfxm_geo.h5"
    write_simulation_h5(
        out,
        Hg=_TINY_HG,
        q_hkl=_TINY_Q,
        frames=_single_frame(),
        include_perfect_crystal=False,
        sample_dis=None,  # centered / random_dislocations
        sample_ndis=1,
        sample_remount="N/A",
        config_toml="",
        cli="test",
        ctx=_fm_stub,
    )
    with h5py.File(out, "r") as f:
        assert "/1.1/sample/dis" not in f  # omitted, not a -1.0 sentinel
        assert "/1.1/sample/ndis" in f


def test_wall_mode_writes_sample_dis(tmp_path: Path, _fm_stub: fm.ForwardContext) -> None:
    out = tmp_path / "dfxm_geo.h5"
    write_simulation_h5(
        out,
        Hg=_TINY_HG,
        q_hkl=_TINY_Q,
        frames=_single_frame(),
        include_perfect_crystal=False,
        sample_dis=4.0,  # wall mode
        sample_ndis=151,
        sample_remount="S1",
        config_toml="",
        cli="test",
        ctx=_fm_stub,
    )
    with h5py.File(out, "r") as f:
        assert "/1.1/sample/dis" in f
        assert np.isclose(float(f["/1.1/sample/dis"][()]), 4.0)
