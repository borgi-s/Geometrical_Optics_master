"""Sub-project D: kernel lookup + load-verification tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


def _make_kernel_npz(
    path: Path,
    hkl: tuple[int, int, int] = (-1, 1, -1),
    keV: float = 17.0,
    include_metadata: bool = True,
) -> Path:
    """Write a tiny synthetic kernel npz at `path` matching the schema."""
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
    }
    if include_metadata:
        data["hkl"] = np.array(hkl, dtype=np.int64)
        data["keV"] = np.float64(keV)
    np.savez(path, **data)
    return path


class TestLookupKernelPath:
    def test_single_match_returns_path(self, tmp_path: Path) -> None:
        from dfxm_geo.direct_space.forward_model import _lookup_kernel_path

        p = _make_kernel_npz(tmp_path / "Resq_i_h-1_k1_l-1_17keV_20260520_2014.npz")
        result = _lookup_kernel_path((-1, 1, -1), 17.0, tmp_path)
        assert result == p

    def test_multi_match_returns_newest_and_warns(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from dfxm_geo.direct_space.forward_model import _lookup_kernel_path

        older = _make_kernel_npz(tmp_path / "Resq_i_h-1_k1_l-1_17keV_20260518_1142.npz")
        middle = _make_kernel_npz(tmp_path / "Resq_i_h-1_k1_l-1_17keV_20260520_2014.npz")
        newest = _make_kernel_npz(tmp_path / "Resq_i_h-1_k1_l-1_17keV_20260521_0930.npz")
        # Force mtimes for deterministic ordering on Windows.
        os.utime(older, (1000, 1000))
        os.utime(middle, (2000, 2000))
        os.utime(newest, (3000, 3000))

        result = _lookup_kernel_path((-1, 1, -1), 17.0, tmp_path)
        assert result == newest

        err = capsys.readouterr().err
        assert "found 3 kernels" in err
        assert "(newest, will use)" in err
        assert "20260521_0930" in err

    def test_zero_match_raises_file_not_found(self, tmp_path: Path) -> None:
        from dfxm_geo.direct_space.forward_model import _lookup_kernel_path

        with pytest.raises(FileNotFoundError, match=r"no kernel found for hkl=\(2, 0, 0\)"):
            _lookup_kernel_path((2, 0, 0), 17.0, tmp_path)

    def test_glob_isolates_by_hkl(self, tmp_path: Path) -> None:
        from dfxm_geo.direct_space.forward_model import _lookup_kernel_path

        _make_kernel_npz(tmp_path / "Resq_i_h-1_k1_l-1_17keV_20260520_2014.npz")
        # Looking up a different reflection should not return the (-1,1,-1) file.
        with pytest.raises(FileNotFoundError):
            _lookup_kernel_path((2, 0, 0), 17.0, tmp_path)

    def test_keV_int_and_float_both_match_same_file(self, tmp_path: Path) -> None:
        from dfxm_geo.direct_space.forward_model import _lookup_kernel_path

        p = _make_kernel_npz(tmp_path / "Resq_i_h-1_k1_l-1_17keV_20260520_2014.npz")
        assert _lookup_kernel_path((-1, 1, -1), 17, tmp_path) == p
        assert _lookup_kernel_path((-1, 1, -1), 17.0, tmp_path) == p


class TestLoadDefaultKernelVerification:
    def test_metadata_match_loads_cleanly(self, tmp_path: Path) -> None:
        from dfxm_geo.direct_space import forward_model as fm

        p = _make_kernel_npz(tmp_path / "Resq_i_h-1_k1_l-1_17keV_test.npz")
        fm._load_default_kernel(
            str(p),
            expected_hkl=(-1, 1, -1),
            expected_keV=17.0,
            compute_Hg=False,
        )
        assert fm._loaded_kernel_path == p

    def test_hkl_mismatch_raises(self, tmp_path: Path) -> None:
        from dfxm_geo.direct_space import forward_model as fm

        p = _make_kernel_npz(
            tmp_path / "Resq_i_h-1_k1_l-1_17keV_test.npz",
            hkl=(-1, 1, -1),
            keV=17.0,
        )
        with pytest.raises(
            ValueError, match=r"has hkl=\(-1, 1, -1\) but lookup requested hkl=\(2, 0, 0\)"
        ):
            fm._load_default_kernel(
                str(p),
                expected_hkl=(2, 0, 0),
                expected_keV=17.0,
                compute_Hg=False,
            )

    def test_keV_mismatch_raises(self, tmp_path: Path) -> None:
        from dfxm_geo.direct_space import forward_model as fm

        p = _make_kernel_npz(
            tmp_path / "Resq_i_h-1_k1_l-1_17keV_test.npz",
            hkl=(-1, 1, -1),
            keV=17.0,
        )
        with pytest.raises(ValueError, match=r"has keV=17.0 but lookup requested keV=15.0"):
            fm._load_default_kernel(
                str(p),
                expected_hkl=(-1, 1, -1),
                expected_keV=15.0,
                compute_Hg=False,
            )

    def test_legacy_npz_missing_metadata_raises_keyerror(self, tmp_path: Path) -> None:
        from dfxm_geo.direct_space import forward_model as fm

        p = _make_kernel_npz(
            tmp_path / "Resq_i_legacy.npz",
            include_metadata=False,
        )
        with pytest.raises(KeyError, match=r"lacks `hkl` metadata"):
            fm._load_default_kernel(
                str(p),
                expected_hkl=(-1, 1, -1),
                expected_keV=17.0,
                compute_Hg=False,
            )
