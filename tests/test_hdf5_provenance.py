"""Layer 1 provenance tests for /dfxm_geo/ root group."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm_geo.io.hdf5 import _write_provenance


def test_write_provenance_basic_fields(tmp_path: Path) -> None:
    out = tmp_path / "test.h5"
    with h5py.File(out, "w") as f:
        _write_provenance(f, cli="dfxm-forward --config x.toml --output out/")

    with h5py.File(out, "r") as f:
        g = f["/dfxm_geo"]
        # Provenance must record the actual installed package version.
        from importlib.metadata import version as _pkg_version

        assert g["version"][()].decode() == _pkg_version("dfxm-geo")
        # git_sha is "unknown" outside a git repo, a 40-char SHA inside one
        sha = g["git_sha"][()].decode()
        assert sha == "unknown" or len(sha) == 40
        assert isinstance(g["git_dirty"][()], (bool, np.bool_))
        assert g["hostname"][()].decode()  # non-empty
        assert g["python_version"][()].decode().startswith("3.")
        assert g["numpy_version"][()].decode()
        assert g["generated_at"][()].decode()  # ISO-ish
        assert g["cli"][()].decode().startswith("dfxm-forward")


def test_write_provenance_kernel_subgroup(tmp_path: Path) -> None:
    npz_path = tmp_path / "Resq_i_test.npz"
    # Fake kernel npz with the expected param keys
    np.savez(
        npz_path,
        Resq_i=np.zeros((4, 4, 4), dtype=np.float64),
        qi1_range=1.0,
        qi2_range=2.0,
        qi3_range=3.0,
        npoints1=4,
        npoints2=4,
        npoints3=4,
        Nrays=int(1e6),
    )
    out = tmp_path / "test.h5"
    with h5py.File(out, "w") as f:
        _write_provenance(f, cli="x", kernel_npz=npz_path)
    with h5py.File(out, "r") as f:
        k = f["/dfxm_geo/kernel"]
        assert k["pkl_fn"][()].decode() == "Resq_i_test.npz"
        sha = k["sha256"][()].decode()
        assert len(sha) == 64  # hex digest
        assert float(k["qi1_range"][()]) == 1.0
        assert int(k["Nrays"][()]) == int(1e6)
        # Kernel disk shape (npoints1, npoints2, npoints3) must match the
        # Resq_i array — provenance is the only on-disk record of the kernel
        # grid since Resq_i itself is excluded from provenance.
        assert int(k["npoints1"][()]) == 4
        assert int(k["npoints2"][()]) == 4
        assert int(k["npoints3"][()]) == 4
        with np.load(npz_path) as arch:
            assert arch["Resq_i"].shape == (
                int(k["npoints1"][()]),
                int(k["npoints2"][()]),
                int(k["npoints3"][()]),
            )


def test_write_provenance_config_toml(tmp_path: Path) -> None:
    config_str = "[crystal]\ndis = 4.0\nndis = 151\n"
    out = tmp_path / "test.h5"
    with h5py.File(out, "w") as f:
        _write_provenance(f, cli="x", config_toml=config_str)
    with h5py.File(out, "r") as f:
        got = f["/dfxm_geo/config_toml"][()].decode()
        assert got == config_str
        # And it must round-trip through tomllib.
        import tomllib

        parsed = tomllib.loads(got)
        assert parsed["crystal"]["dis"] == 4.0


def _make_kernel_npz_for_provenance(
    path: Path,
    hkl: tuple[int, int, int] = (-1, 1, -1),
    keV: float = 17.0,
) -> Path:
    """Minimal toy kernel for provenance tests that need run_simulation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {
        "Resq_i": np.zeros((4, 4, 4), dtype=np.float64),
        "Nrays": np.int64(1000),
        "npoints1": np.int64(4),
        "npoints2": np.int64(4),
        "npoints3": np.int64(4),
        "qi1_range": np.float64(1e-3),
        "qi2_range": np.float64(1e-3),
        "qi3_range": np.float64(1e-3),
        "zeta_v_fwhm": np.float64(5.3e-4),
        "zeta_h_fwhm": np.float64(0.0),
        "NA_rms": np.float64(3.1e-4),
        "eps_rms": np.float64(6e-5),
        "theta": np.float64(0.165),
        "D": np.float64(5.6e-4),
        "d1": np.float64(0.274),
        "phys_aper": np.float64(2e-3),
        "beamstop": np.bool_(True),
        "bs_height": np.float64(25e-3),
        "aperture": np.bool_(True),
        "knife_edge": np.bool_(False),
        "dphi_range": np.float64(0.0),
        "hkl": np.array(hkl, dtype=np.int64),
        "keV": np.float64(keV),
    }
    np.savez(path, **data)
    return path


class TestHdf5NewAttrs:
    @pytest.fixture(autouse=True)
    def _reset_kernel_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reset fm module-level state between tests."""
        import dfxm_geo.direct_space.forward_model as fm

        monkeypatch.setattr(fm, "_loaded_kernel_path", None)

    def test_scan_mode_attrs_written(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import dfxm_geo.direct_space.forward_model as fm
        from dfxm_geo.pipeline import (
            AxisScanConfig,
            CenteredCrystalConfig,
            CrystalConfig,
            IOConfig,
            PostprocessConfig,
            ReciprocalConfig,
            ScanConfig,
            SimulationConfig,
            run_simulation,
        )

        # Stage a toy kernel so the lookup succeeds.
        _make_kernel_npz_for_provenance(
            tmp_path / "pkl_files" / "Resq_i_h-1_k1_l-1_17keV_20260520_0000.npz",
            hkl=(-1, 1, -1),
            keV=17.0,
        )
        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path / "pkl_files") + os.sep)

        cfg = SimulationConfig(
            crystal=CrystalConfig(
                mode="centered",
                centered=CenteredCrystalConfig(b=(1, -1, 0), n=(1, 1, 1), t=(1, 1, -2)),
            ),
            scan=ScanConfig(
                phi=AxisScanConfig(range=6e-4, steps=3),
                chi=AxisScanConfig(range=2e-3, steps=3),
            ),
            io=IOConfig(include_perfect_crystal=False),
            postprocess=PostprocessConfig(enabled=False),
            reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
        )
        out_dir = tmp_path / "out"
        run_simulation(cfg, out_dir)
        h5_path = out_dir / "dfxm_geo.h5"
        with h5py.File(h5_path, "r") as f:
            grp = f["/1.1"]
            # scan_mode = "mosa" (phi + chi both scanned)
            assert grp.attrs["scan_mode"] == "mosa"
            # scanned_axes: ["phi", "chi"] in canonical order
            assert list(grp.attrs["scanned_axes"]) == ["phi", "chi"]
            # crystal_mode = "centered"
            assert grp.attrs["crystal_mode"] == "centered"
