"""Sub-project D integration tests: lookup-driven forward + identify."""

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
    """Inline copy of the helper from test_kernel_lookup.py to keep this file self-contained."""
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


class TestForwardMultiReflection:
    def test_happy_path_with_explicit_hkl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_simulation looks up the kernel from [reciprocal].hkl/keV, loads it,
        runs the (monkey-patched) forward, records actual loaded path in module state."""
        import dfxm_geo.direct_space.forward_model as fm
        import dfxm_geo.pipeline as p
        from dfxm_geo.pipeline import SimulationConfig

        # Stage a kernel matching (2, 0, 0) @ 17 keV
        kernel_path = _make_kernel_npz(
            tmp_path / "pkl_files" / "Resq_i_h2_k0_l0_17keV_20260520_2014.npz",
            hkl=(2, 0, 0),
            keV=17.0,
        )
        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path / "pkl_files") + os.sep)

        # Monkey-patch the heavy pieces.
        monkeypatch.setattr(fm, "Find_Hg", lambda *a, **k: (np.zeros((4, 4, 4)), np.zeros(3)))
        monkeypatch.setattr(p, "save_images_parallel", lambda *a, **k: [])
        monkeypatch.setattr(
            p,
            "write_simulation_h5",
            lambda *a, **k: None,
        )

        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[crystal]\ndis = 4\nndis = 151\nsample_remount = "S1"\n'
            "[scan]\nphi_range = 0.034\nphi_steps = 2\nchi_range = 0.115\nchi_steps = 2\n"
            '[io]\nfn_prefix = "/x"\nftype = ".npy"\n'
            'dislocs_dirname = "d"\nperfect_dirname = "p"\ninclude_perfect_crystal = false\n'
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            'figures_dirname = "f"\ndata_dirname = "a"\n'
            "[reciprocal]\nhkl = [2, 0, 0]\nkeV = 17.0\n"
        )
        config = SimulationConfig.from_toml(cfg)
        p.run_simulation(config, tmp_path / "out")

        # Verify the actually-loaded path is the staged kernel.
        assert fm._loaded_kernel_path == kernel_path

    def test_lookup_miss_errors_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dfxm_geo.direct_space.forward_model as fm
        import dfxm_geo.pipeline as p
        from dfxm_geo.pipeline import SimulationConfig

        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path / "pkl_files_empty") + os.sep)
        (tmp_path / "pkl_files_empty").mkdir()

        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[crystal]\ndis = 4\nndis = 151\nsample_remount = "S1"\n'
            "[scan]\nphi_range = 0.034\nphi_steps = 2\nchi_range = 0.115\nchi_steps = 2\n"
            '[io]\nfn_prefix = "/x"\nftype = ".npy"\n'
            'dislocs_dirname = "d"\nperfect_dirname = "p"\ninclude_perfect_crystal = false\n'
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            'figures_dirname = "f"\ndata_dirname = "a"\n'
            "[reciprocal]\nhkl = [2, 0, 0]\nkeV = 17.0\n"
        )
        config = SimulationConfig.from_toml(cfg)
        with pytest.raises(FileNotFoundError, match=r"no kernel found for hkl=\(2, 0, 0\)"):
            p.run_simulation(config, tmp_path / "out")

    def test_metadata_mismatch_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """File named for (2,0,0) but contains bundled hkl=(-1,1,-1) → ValueError."""
        import dfxm_geo.direct_space.forward_model as fm
        import dfxm_geo.pipeline as p
        from dfxm_geo.pipeline import SimulationConfig

        # Filename says 200 but metadata says -1,1,-1.
        _make_kernel_npz(
            tmp_path / "pkl_files" / "Resq_i_h2_k0_l0_17keV_20260520_2014.npz",
            hkl=(-1, 1, -1),
            keV=17.0,
        )
        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path / "pkl_files") + os.sep)

        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[crystal]\ndis = 4\nndis = 151\nsample_remount = "S1"\n'
            "[scan]\nphi_range = 0.034\nphi_steps = 2\nchi_range = 0.115\nchi_steps = 2\n"
            '[io]\nfn_prefix = "/x"\nftype = ".npy"\n'
            'dislocs_dirname = "d"\nperfect_dirname = "p"\ninclude_perfect_crystal = false\n'
            "[postprocess]\nenabled = false\n"
            "chi_oversample = 1\nphi_oversample = 1\nchi_oversample_for_shift = 1\n"
            'figures_dirname = "f"\ndata_dirname = "a"\n'
            "[reciprocal]\nhkl = [2, 0, 0]\nkeV = 17.0\n"
        )
        config = SimulationConfig.from_toml(cfg)
        with pytest.raises(
            ValueError, match=r"has hkl=\(-1, 1, -1\) but lookup requested hkl=\(2, 0, 0\)"
        ):
            p.run_simulation(config, tmp_path / "out")
