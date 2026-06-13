"""When write_simulation_h5 is called with oblique params, master HDF5 carries
eta + mount provenance attrs on /1.1 (and /2.1 if perfect crystal included).

Two cases tested:
  1. default (simplified, eta=0, no mount supplied) — v2.2.0 back-compat: the
     new provenance attrs are written with safe defaults.
  2. oblique (eta=0.3531, explicit paper Al mount) — the attrs carry the supplied
     values.

The heavy forward computation (numba LUT) is stubbed out via monkeypatch so
these tests run without a bootstrapped kernel on disk.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
import dfxm_geo.io.hdf5 as hdf5_mod
from dfxm_geo.crystal.oblique import CrystalMount
from dfxm_geo.io.hdf5 import write_simulation_h5
from dfxm_geo.pipeline import ScanFrames

# Correct shape for Hg: (N, 3, 3); use N=1 for the minimal test.
_TINY_HG = np.zeros((1, 3, 3))
_TINY_Q = np.array([-1.0, 1.0, -1.0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_detector_file(path: Path, n_frames: int = 1, h: int = 4, w: int = 4) -> None:
    """Write a minimal LIMA-style HDF5 detector file (skeleton only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["NX_class"] = "NXroot"
        entry = f.create_group("entry_0000")
        entry.attrs["NX_class"] = "NXentry"
        det = entry.create_group("dfxm_sim_detector")
        det.attrs["NX_class"] = "NXdetector"
        det.create_dataset(
            "image",
            data=np.zeros((n_frames, h, w), dtype=np.float32),
        )


def _fake_parallel_writer(
    path: Path,
    args_list: list,
    *,
    max_workers: int | None = None,
    detector_shape: tuple[int, int] | None = None,
) -> None:
    """Stub: write a minimal skeleton instead of computing forward frames."""
    _make_minimal_detector_file(path, n_frames=len(args_list))


def _single_frame() -> ScanFrames:
    """Minimal 1-frame scan at phi=chi=0."""
    return ScanFrames(
        phi_pf=np.zeros(1),
        chi_pf=np.zeros(1),
        two_dtheta_pf=np.zeros(1),
        z_pf=np.zeros(1),
        n_frames=1,
    )


# ---------------------------------------------------------------------------
# Shared fixture: set up a tiny synthetic fm state so write_simulation_h5 can
# build attrs without a real bootstrapped kernel on disk.
# ---------------------------------------------------------------------------


@pytest.fixture()
def _fm_stub(monkeypatch: pytest.MonkeyPatch) -> fm.ForwardContext:
    """Stub the heavy compute and return a ForwardContext for write_simulation_h5.

    #16 Slice 5: the /1.1 ``theta`` attr is read from ctx.geometry.theta_0
    (= 0.165 here, ~9.5 deg). An analytic_eval sentinel satisfies the
    "resolution backend loaded" guard and keeps kernel_npz None (no real SHA).
    Both precompute_forward_static and the parallel detector writer are stubbed
    so no numba kernel is needed.
    """
    monkeypatch.setattr(
        fm,
        "precompute_forward_static",
        lambda Hg_in, ctx=None: np.zeros((3, 1)),  # dummy base_qc shape (3, N)
    )
    monkeypatch.setattr(
        hdf5_mod,
        "_compute_and_write_detector_file_parallel",
        _fake_parallel_writer,
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_simplified_provenance_attrs(tmp_path: Path, _fm_stub: fm.ForwardContext) -> None:
    """Default call (no geometry_mode / eta / mount) writes safe defaults on /1.1."""
    out = tmp_path / "dfxm_geo.h5"
    frames = _single_frame()
    Hg = _TINY_HG
    q_hkl = _TINY_Q

    write_simulation_h5(
        out,
        Hg=Hg,
        q_hkl=q_hkl,
        frames=frames,
        include_perfect_crystal=False,
        sample_dis=None,
        sample_ndis=1,
        sample_remount="N/A",
        config_toml="",
        cli="test",
        ctx=_fm_stub,
    )

    with h5py.File(out, "r") as f:
        attrs = f["/1.1"].attrs

        # New provenance attrs present with safe defaults.
        assert attrs["geometry_mode"] == "simplified"
        assert np.isclose(float(attrs["eta"]), 0.0)
        assert np.isclose(float(attrs["theta"]), 0.165)
        assert attrs["lattice"] == "cubic"
        # Default Al mount: (1,0,0) / (0,1,0) / (0,0,1)
        np.testing.assert_array_equal(attrs["mount_x"], [1, 0, 0])
        np.testing.assert_array_equal(attrs["mount_y"], [0, 1, 0])
        np.testing.assert_array_equal(attrs["mount_z"], [0, 0, 1])


def test_oblique_provenance_attrs(tmp_path: Path, _fm_stub: fm.ForwardContext) -> None:
    """Oblique call writes eta + paper Al mount attrs on /1.1."""
    out = tmp_path / "dfxm_geo.h5"
    frames = _single_frame()
    Hg = _TINY_HG
    q_hkl = _TINY_Q

    paper_mount = CrystalMount(
        lattice="cubic",
        a=4.0493e-10,
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
    )

    write_simulation_h5(
        out,
        Hg=Hg,
        q_hkl=q_hkl,
        frames=frames,
        include_perfect_crystal=False,
        sample_dis=None,
        sample_ndis=1,
        sample_remount="N/A",
        config_toml="",
        cli="test",
        geometry_mode="oblique",
        eta=0.3531,
        mount=paper_mount,
        ctx=_fm_stub,
    )

    with h5py.File(out, "r") as f:
        attrs = f["/1.1"].attrs

        assert attrs["geometry_mode"] == "oblique"
        assert np.isclose(float(attrs["eta"]), 0.3531, atol=1e-6)
        assert np.isclose(float(attrs["theta"]), 0.165)
        assert attrs["lattice"] == "cubic"
        assert np.isclose(float(attrs["a"]), 4.0493e-10, rtol=1e-5)
        np.testing.assert_array_equal(attrs["mount_x"], [1, 0, 0])
        np.testing.assert_array_equal(attrs["mount_y"], [0, 1, 0])
        np.testing.assert_array_equal(attrs["mount_z"], [0, 0, 1])

        # Oblique FCC runs (mount is not None) MUST emit structure-family provenance.
        assert attrs["structure_type"] == "fcc", "oblique FCC run must emit structure_type='fcc'"
        assert "poisson_ratio" in attrs, "oblique FCC run must emit poisson_ratio"
        assert "burgers_magnitude_um" in attrs, "oblique FCC run must emit burgers_magnitude_um"


def test_fcc_simplified_no_structure_attrs(tmp_path: Path, _fm_stub: fm.ForwardContext) -> None:
    """FCC simplified path (mount=None) writes NO structure_type / poisson attrs.

    Byte-identity gate: existing FCC outputs are unchanged.
    """
    out = tmp_path / "dfxm_geo.h5"
    frames = _single_frame()

    write_simulation_h5(
        out,
        Hg=_TINY_HG,
        q_hkl=_TINY_Q,
        frames=frames,
        include_perfect_crystal=False,
        sample_dis=None,
        sample_ndis=1,
        sample_remount="N/A",
        config_toml="",
        cli="test",
        ctx=_fm_stub,
        # mount=None (default) → simplified FCC path
    )

    with h5py.File(out, "r") as f:
        attrs = dict(f["/1.1"].attrs)
        # Structure attrs MUST NOT appear in FCC simplified output.
        for key in ("structure_type", "poisson_ratio", "poisson_source", "burgers_magnitude_um"):
            assert key not in attrs, f"unexpected attr {key!r} on FCC simplified /1.1"


def test_bcc_structure_attrs_written(tmp_path: Path, _fm_stub: fm.ForwardContext) -> None:
    """BCC structure-aware run writes structure_type, poisson_ratio/source, burgers_magnitude_um."""
    out = tmp_path / "dfxm_geo.h5"
    frames = _single_frame()

    # Fe BCC, {110}<111> family, a=2.866 Å
    bcc_mount = CrystalMount(
        lattice="cubic",
        a=2.866e-10,
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
        structure_type="bcc",
        material="Fe",
        slip_families=("{110}<111>",),
    )

    write_simulation_h5(
        out,
        Hg=_TINY_HG,
        q_hkl=_TINY_Q,
        frames=frames,
        include_perfect_crystal=False,
        sample_dis=None,
        sample_ndis=1,
        sample_remount="N/A",
        config_toml="",
        cli="test",
        geometry_mode="oblique",
        eta=0.0,
        mount=bcc_mount,
        ctx=_fm_stub,
    )

    with h5py.File(out, "r") as f:
        attrs = f["/1.1"].attrs

        # Structure provenance attrs must be present.
        assert attrs["structure_type"] == "bcc"
        assert np.isclose(float(attrs["poisson_ratio"]), 0.29, atol=1e-6)
        assert attrs["poisson_source"] == "KL"
        # slip_families round-trips as a list of strings.
        sf = list(attrs["slip_families"])
        assert "{110}<111>" in sf
        # burgers_magnitude_um: BCC Fe {110}<111> |b| = a√3/2 ≈ 0.2476 µm
        b_um = float(attrs["burgers_magnitude_um"])
        expected_b = 2.866e-4 * np.sqrt(3) / 2  # a in µm * sqrt(3)/2
        assert np.isclose(b_um, expected_b, rtol=1e-4), f"got {b_um}, expected {expected_b}"
        # material is present.
        assert attrs["material"] == "Fe"


def test_bcc_mount_fcc_structure_type_no_structure_attrs(
    tmp_path: Path, _fm_stub: fm.ForwardContext
) -> None:
    """A mount with structure_type=None and material=None still emits structure attrs
    when mount is not None (oblique run), but structure_type defaults to 'fcc'."""
    out = tmp_path / "dfxm_geo.h5"
    frames = _single_frame()

    # Oblique run with a mount but no explicit structure metadata.
    plain_mount = CrystalMount(
        lattice="cubic",
        a=4.0495e-10,
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
        # structure_type=None, material=None → defaults to fcc / Al poisson
    )

    write_simulation_h5(
        out,
        Hg=_TINY_HG,
        q_hkl=_TINY_Q,
        frames=frames,
        include_perfect_crystal=False,
        sample_dis=None,
        sample_ndis=1,
        sample_remount="N/A",
        config_toml="",
        cli="test",
        geometry_mode="oblique",
        eta=0.1,
        mount=plain_mount,
        ctx=_fm_stub,
    )

    with h5py.File(out, "r") as f:
        attrs = f["/1.1"].attrs
        # mount is not None → structure attrs are written (even for an FCC-defaulted mount).
        assert "structure_type" in attrs
        assert attrs["structure_type"] == "fcc"
        assert "poisson_ratio" in attrs
        assert "poisson_source" in attrs
