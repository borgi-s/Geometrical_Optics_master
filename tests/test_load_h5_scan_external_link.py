"""Verify load_h5_scan reads through ExternalLink correctly (v1.2.0 layout)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.io.hdf5 import (
    DETECTOR_INTERNAL_PATH,
    MasterWriter,
    _write_detector_file,
    load_h5_scan,
)
from dfxm_geo.pipeline import _lookup_and_load_kernel


def _require_kernel() -> None:
    """Skip unless a bootstrapped (-1,1,-1) 17 keV kernel npz is on disk."""
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip(f"no kernel npz found in {kernel_dir}")


@pytest.fixture(autouse=True)
def _reset_kernel_state():
    """Restore module-level forward_model state after each test.

    `_lookup_and_load_kernel` sets `fm.Hg` and `fm._loaded_kernel_path` as
    side effects. Downstream tests rely on `_loaded_kernel_path is None`,
    so we reset both after every test in this file to avoid cross-test
    bleed.
    """
    yield
    fm.Hg = None
    fm._loaded_kernel_path = None


def test_load_h5_scan_follows_external_link(tmp_path: Path) -> None:
    """Verify load_h5_scan reads through ExternalLink correctly (v1.2.0 layout)."""
    _require_kernel()
    _lookup_and_load_kernel((-1, 1, -1), 17.0)
    try:
        scan_dir = tmp_path / "scan0001"
        det_path = scan_dir / "dfxm_sim_detector_0000.h5"
        # 2 phi x 3 chi = 6 frames; need phi inner, chi outer ordering preserved
        h, w = 5, 5
        stack = np.arange(2 * 3 * h * w, dtype=np.float64).reshape(2 * 3, h, w)
        _write_detector_file(det_path, stack)

        master_path = tmp_path / "dfxm_geo.h5"
        cfg = "[scan.phi]\nrange = 0.01\nsteps = 2\n[scan.chi]\nrange = 0.01\nsteps = 3\n"
        with MasterWriter(master_path, cli="t", config_toml=cfg, kernel_npz=None) as m:
            m.add_scan(
                scan_id="1.1",
                title="t",
                start_time="t",
                end_time="t",
                sample={"name": "x"},
                positioners={"phi": np.zeros(6), "chi": np.zeros(6)},
                detector_links={
                    "dfxm_sim_detector": (
                        Path("scan0001") / "dfxm_sim_detector_0000.h5",
                        DETECTOR_INTERNAL_PATH,
                    )
                },
                dfxm_geo={},
                attrs={},
            )

        flat, reshape, dim_h, dim_w = load_h5_scan(
            master_path, scan_id="1.1", phi_steps=2, chi_steps=3
        )
        np.testing.assert_array_equal(flat, stack)
        assert dim_h == h and dim_w == w
    finally:
        fm.Hg = None
        fm._loaded_kernel_path = None
