"""Layer 1 tests for load_h5_scan."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dfxm_geo.io.hdf5 import load_h5_scan, write_h5_scan


def test_load_h5_scan_returns_postprocess_tuple(tmp_path: Path) -> None:
    images = np.arange(4 * 8 * 8, dtype=np.float64).reshape(4, 8, 8)
    phi = np.array([-0.001, 0.001, -0.001, 0.001])
    chi = np.array([-0.002, -0.002, 0.002, 0.002])
    out = tmp_path / "test.h5"
    write_h5_scan(out, scan_id="1.1", images=images, phi=phi, chi=chi)

    stack, stack_reshape, dim_1, dim_2 = load_h5_scan(
        out,
        scan_id="1.1",
        phi_steps=2,
        chi_steps=2,
    )
    assert stack.shape == (4, 8, 8)
    assert stack_reshape.shape == (2, 2, 8, 8)
    assert dim_1 == 8
    assert dim_2 == 8
    np.testing.assert_array_equal(stack, images)
