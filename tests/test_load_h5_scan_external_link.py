"""Verify load_h5_scan reads through ExternalLink correctly (v1.2.0 layout)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dfxm_geo.io.hdf5 import (
    DETECTOR_INTERNAL_PATH,
    MasterWriter,
    _write_detector_file,
    load_h5_scan,
)


def test_load_h5_scan_follows_external_link(tmp_path: Path) -> None:
    """Verify load_h5_scan reads through ExternalLink correctly (v1.2.0 layout).

    #16 Slice 5: load_h5_scan is a pure HDF5 read — no kernel or forward-model
    globals are involved, so the prior load + reset dance is gone.
    """
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

    flat, reshape, dim_h, dim_w = load_h5_scan(master_path, scan_id="1.1", phi_steps=2, chi_steps=3)
    np.testing.assert_array_equal(flat, stack)
    assert dim_h == h and dim_w == w
    # Reshape is (chi_steps, phi_steps, H, W) — the order the mosaicity
    # COM functions (compute_com_maps / compute_chi_shift) consume, so the
    # φ axis is axis 1 and χ is axis 0. Frames are phi-inner/chi-outer, so
    # reshape[c, p] == flat[c * phi_steps + p].
    assert reshape.shape == (3, 2, h, w)
    for c in range(3):
        for p in range(2):
            np.testing.assert_array_equal(reshape[c, p], stack[c * 2 + p])
