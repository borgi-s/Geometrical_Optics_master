"""Unit tests for the LIMA-style per-scan detector file writer."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.io.hdf5 import (
    DETECTOR_INTERNAL_PATH,
    _compute_and_write_detector_file_parallel,
    _compute_frame,
    _write_detector_file,
)
from dfxm_geo.pipeline import _lookup_and_load_kernel


def _require_kernel() -> None:
    """Skip unless a bootstrapped (-1,1,-1) 17 keV kernel npz is on disk."""
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip(f"no kernel npz found in {kernel_dir}")


def test_write_detector_file_structure(tmp_path: Path) -> None:
    stack = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    out = tmp_path / "dfxm_sim_detector_0000.h5"

    _write_detector_file(out, stack)

    assert out.is_file()
    with h5py.File(out, "r") as f:
        assert f.attrs["NX_class"] == "NXroot"
        assert f.attrs["creator"] == "dfxm-geo"
        assert f.attrs["default"] == "entry_0000"
        e = f["entry_0000"]
        assert e.attrs["NX_class"] == "NXentry"
        det = e["dfxm_sim_detector"]
        assert det.attrs["NX_class"] == "NXdetector"
        img = det["image"]
        assert img.shape == (2, 3, 4)
        assert img.attrs["interpretation"] == "image"
        assert img.chunks == (1, 3, 4)
        assert img.compression == "gzip"
        assert img.compression_opts == 4
        np.testing.assert_array_equal(img[...], stack)
        # NXdata + measurement soft links
        plot = e["plot"]
        assert plot.attrs["NX_class"] == "NXdata"
        assert plot.attrs["signal"] == "image"
        np.testing.assert_array_equal(plot["image"][...], stack)
        np.testing.assert_array_equal(
            e["measurement"][...], stack
        )  # h5py auto-follows SoftLink to a dataset


def test_write_detector_file_internal_path_matches_constant(tmp_path: Path) -> None:
    out = tmp_path / "det.h5"
    _write_detector_file(out, np.zeros((1, 2, 2)))
    with h5py.File(out, "r") as f:
        # The dataset at DETECTOR_INTERNAL_PATH is what ExternalLink targets will use.
        assert DETECTOR_INTERNAL_PATH in f
        assert f[DETECTOR_INTERNAL_PATH].shape == (1, 2, 2)


def test_compute_and_write_detector_file_parallel_roundtrip(tmp_path: Path) -> None:
    """Workers run forward() and stream into one detector file; pixels match a
    serial reference (probed-frame-0 plus the workers' results)."""
    _require_kernel()
    # Ensure a kernel is loaded (test fixtures use the bundled kernel).
    # `_lookup_and_load_kernel` also calls `Find_Hg`, which populates
    # `fm.Hg` to the correct (NN1*NN2*NN3, 3, 3) shape that forward()
    # requires. A synthetic (1, 3, 3) Hg would IndexError inside forward().
    _lookup_and_load_kernel((-1, 1, -1), 17.0)
    if fm.Hg is None:
        pytest.skip("forward_model.Hg not populated; run dfxm-bootstrap.")
    try:
        Hg = fm.Hg
        base_qc = fm.precompute_forward_static(Hg)
        args = [
            (0, base_qc, 0.0, 0.0, 0.0),
            (1, base_qc, 1e-5, 0.0, 0.0),
            (2, base_qc, 0.0, 1e-5, 0.0),
            (3, base_qc, 1e-5, 1e-5, 0.0),
        ]
        out = tmp_path / "scan0001" / "dfxm_sim_detector_0000.h5"
        _compute_and_write_detector_file_parallel(out, args, max_workers=2)

        # Reference: run forward() serially using _compute_frame
        ref = np.empty((4,) + _compute_frame(args[0])[1].shape, dtype=np.float64)
        for a in args:
            idx, im = _compute_frame(a)
            ref[idx] = im
        with h5py.File(out, "r") as f:
            np.testing.assert_array_equal(f[DETECTOR_INTERNAL_PATH][...], ref)
    finally:
        # Restore the pre-test "nothing loaded" sentinels so downstream tests
        # whose skip guards check `fm.Hg is None` keep skipping as they did
        # in the baseline. We also clear `fm._loaded_kernel_path` so the next
        # `_lookup_and_load_kernel` call actually re-loads the kernel rather
        # than short-circuiting on a stale path that survived a monkeypatch.
        # revert in an intermediate test (see test_hdf5_provenance's
        # `TestHdf5NewAttrs` which monkeypatches `_loaded_kernel_path = None`
        # then reverts on test exit; without our clear, the revert restores
        # the real-kernel path here while `Resq_i` is left as toy zeros,
        # breaking the next pipeline test that relies on a fresh reload).
        fm.Hg = None
        fm._loaded_kernel_path = None
