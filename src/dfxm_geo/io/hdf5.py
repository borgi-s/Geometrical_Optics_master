"""HDF5 output format for dfxm-forward simulations.

Writes BLISS-style ESRF HDF5 (compatible with darfix / darling) with
sim-specific provenance metadata. One file per `run_simulation` call,
containing 1-2 BLISS scans (`/1.1` dislocations, `/2.1` optional
perfect crystal).

See docs/output-format.md for the full schema.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def write_h5_scan(
    path: Path,
    scan_id: str,
    images: np.ndarray,
) -> None:
    """Write a single BLISS scan to an HDF5 file (creates or appends).

    Args:
        path: Output HDF5 file path. Created if missing; appended if exists.
        scan_id: BLISS scan identifier, e.g. "1.1" for the first scan.
        images: Image stack, shape (N_frames, H, W), dtype float64.
    """
    mode = "a" if path.exists() else "w"
    with h5py.File(path, mode) as f:
        scan = f.require_group(scan_id)
        det = scan.require_group("instrument/dfxm_sim_detector")
        det.create_dataset("data", data=images)
