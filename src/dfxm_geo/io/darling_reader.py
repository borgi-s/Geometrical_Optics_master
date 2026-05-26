"""darling-compatible reader for dfxm-geo's master + per-scan HDF5 layout.

The simulation output is a BLISS-style *master* file whose per-scan detector
data lives in a separate LIMA-style file, linked in via an ``h5py.ExternalLink``
at ``/<scan_id>/instrument/<detector>/data``. h5py resolves that link
transparently when you index the dataset *by its explicit path* — but
``darling`` 2.0.0 discovers the detector dataset with ``h5py.visititems``,
which does **not** traverse external links. As a result darling's built-in
readers cannot find our detector data (and pointing darling at the per-scan
LIMA file fails too, because that file lacks the BLISS scan structure darling
expects).

:class:`DarlingReader` is a drop-in ``darling`` reader (it honours darling's
``Reader`` protocol: ``__call__(scan_id, roi) -> (data, motors)``) that opens
the master, reads the detector stack through the explicit linked path, and
reshapes it into darling's ``(a, b, m, n)`` convention with motor meshgrids of
shape ``(k, m, n)``. It has **no import-time dependency on darling**, so it
lives happily in ``dfxm_geo`` whether or not darling is installed::

    import darling
    from dfxm_geo.io.darling_reader import DarlingReader

    dset = darling.DataSet(DarlingReader("output/dfxm_geo.h5"))
    dset.load_scan("1.1")

For a plain in-memory array (no darling involved) use
:func:`resolve_detector_data`.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

DEFAULT_DETECTOR = "dfxm_sim_detector"

# darling's DataSet does an ``isinstance(data_source, darling.io.reader.Reader)``
# check, so DarlingReader must genuinely subclass that base when darling is
# installed. We import it lazily as the base class and fall back to ``object``
# when darling is absent — keeping this module importable in a darling-free
# environment (darling is an optional, interop-only dependency).
try:  # pragma: no cover - trivial import shim
    from darling.io.reader import Reader as _ReaderBase  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - darling not installed
    _ReaderBase = object


def resolve_detector_data(
    path: str | Path,
    scan_id: str = "1.1",
    detector_name: str = DEFAULT_DETECTOR,
    roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Read a per-scan detector stack into memory, resolving the external link.

    This is the one-liner that sidesteps darling's ``visititems`` blindness:
    indexing the linked dataset by its explicit path makes h5py follow the
    ``ExternalLink`` and return the real ``(n_frames, H, W)`` array.

    Args:
        path: Path to the *master* HDF5 file (e.g. ``output/dfxm_geo.h5``).
        scan_id: BLISS scan id, default ``"1.1"``.
        detector_name: NXdetector group name, default ``"dfxm_sim_detector"``.
        roi: optional ``(row_min, row_max, col_min, col_max)`` detector crop.

    Returns:
        ``(n_frames, H, W)`` ndarray.
    """
    internal = f"/{scan_id}/instrument/{detector_name}/data"
    with h5py.File(str(path), "r") as f:
        dset = f[internal]
        if roi is not None:
            r1, r2, c1, c2 = roi
            return np.asarray(dset[:, r1:r2, c1:c2])
        return np.asarray(dset[...])


def _scan_motors(f: h5py.File, scan_id: str, n_frames: int) -> dict[str, np.ndarray]:
    """Return the *varying* positioner arrays for a scan (the scan motors).

    A positioner is a scan motor when it is stored as a per-frame array
    (length ``n_frames``) with more than one distinct value. Fixed axes are
    written as scalars by the master writer and are skipped.
    """
    pos = f[f"/{scan_id}/instrument/positioners"]
    motors: dict[str, np.ndarray] = {}
    for name, dset in pos.items():
        arr = np.asarray(dset[...])
        if arr.ndim > 0 and arr.size == n_frames and np.unique(arr).size > 1:
            motors[name] = arr.astype(np.float64)
    return motors


class DarlingReader(_ReaderBase):
    """darling ``Reader`` for the dfxm-geo master+per-scan HDF5 layout.

    Args:
        abs_path_to_h5_file: path to the master HDF5 file.
        detector_name: NXdetector group name, default ``"dfxm_sim_detector"``.

    The instance is callable as ``reader(scan_id, roi=None)`` and returns
    ``(data, motors)`` matching darling's contract:

    * ``data`` — ``(a, b, m, n)`` float32, where ``a, b`` are the detector
      dimensions and ``m, n`` the two scan dimensions.
    * ``motors`` — ``(k, m, n)`` float32 meshgrids, one per scan motor.

    Only rectilinear scans over **one or two** varying motors are supported
    (the case darling itself targets). For >2 scanned axes, use
    :func:`dfxm_geo.io.hdf5.load_h5_scan` or read with silx directly.
    """

    def __init__(
        self,
        abs_path_to_h5_file: str | Path,
        detector_name: str = DEFAULT_DETECTOR,
    ) -> None:
        self.abs_path_to_h5_file = str(abs_path_to_h5_file)
        self.detector_name = detector_name

    def __call__(
        self, scan_id: str, roi: tuple[int, int, int, int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        data_raw = resolve_detector_data(self.abs_path_to_h5_file, scan_id, self.detector_name, roi)
        n_frames, h, w = data_raw.shape

        with h5py.File(self.abs_path_to_h5_file, "r") as f:
            motors = _scan_motors(f, scan_id, n_frames)

        names = sorted(motors)
        if len(names) == 1:
            scan_shape = (n_frames, 1)
        elif len(names) == 2:
            m = int(np.unique(motors[names[0]]).size)
            n = int(np.unique(motors[names[1]]).size)
            if m * n != n_frames:
                raise ValueError(
                    f"scan {scan_id!r} is not a rectilinear grid: motors "
                    f"{names} have {m}x{n} distinct values but {n_frames} "
                    "frames. DarlingReader only handles rectilinear scans."
                )
            scan_shape = (m, n)
        elif len(names) == 0:
            raise ValueError(
                f"scan {scan_id!r} has no varying motors (single image / fixed "
                "scan); darling needs a 2D scan. Use resolve_detector_data() "
                "for the raw stack instead."
            )
        else:
            raise NotImplementedError(
                f"scan {scan_id!r} varies {len(names)} motors ({names}); "
                "DarlingReader supports at most 2. Use load_h5_scan() or silx."
            )

        # (m, n, H, W) -> (H, W, m, n), matching darling's MosaScan convention.
        data = data_raw.reshape(*scan_shape, h, w)
        data = data.swapaxes(0, -2).swapaxes(1, -1)

        motor_grids = np.array(
            [motors[name].reshape(scan_shape) for name in names],
            dtype=np.float32,
        )
        return np.ascontiguousarray(data, dtype=np.float32), np.ascontiguousarray(motor_grids)
