"""Detector-model seam: dataset replacement + orchestrator pass."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dfxm_geo.io.hdf5 import (
    DETECTOR_INTERNAL_PATH,
    _write_detector_file,
    replace_detector_image,
)


def test_replace_detector_image_swaps_dtype_and_keeps_attrs(tmp_path: Path) -> None:
    p = tmp_path / "det_0000.h5"
    _write_detector_file(p, np.ones((3, 8, 8), dtype=np.float32))
    new = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
    with h5py.File(p, "a") as f:
        old_attrs = dict(f[DETECTOR_INTERNAL_PATH].attrs)
        replace_detector_image(f, new, extra_attrs={"detector_model": "pco_edge_4.2_id03"})
        ds = f[DETECTOR_INTERNAL_PATH]
        assert ds.dtype == np.uint16
        assert np.array_equal(ds[...], new)
        assert ds.compression == "gzip"
        assert ds.chunks == (1, 8, 8)
        for k, v in old_attrs.items():
            assert ds.attrs[k] == v  # e.g. the NeXus 'interpretation' attr
        assert ds.attrs["detector_model"] == "pco_edge_4.2_id03"
    # softlinks (NXdata/measurement) must still resolve through the new dataset
    with h5py.File(p, "r") as f:
        assert f[DETECTOR_INTERNAL_PATH].shape == (3, 8, 8)
        # core design property: softlinks resolve through the replaced uint16 dataset
        assert np.array_equal(f["entry_0000/plot/image"][...], new)
        assert np.array_equal(f["entry_0000/measurement"][...], new)
