"""HDF5 output format for dfxm-forward simulations.

Writes BLISS-style ESRF HDF5 (compatible with darfix / darling) with
sim-specific provenance metadata. One file per `run_simulation` call,
containing 1-2 BLISS scans (`/1.1` dislocations, `/2.1` optional
perfect crystal).

See docs/output-format.md for the full schema.
"""

from __future__ import annotations

import datetime as _dt
import socket as _socket
import subprocess as _subprocess
import sys as _sys
from importlib.metadata import version as _pkg_version
from pathlib import Path

import h5py
import numpy as np


def _set_nx_class(grp: h5py.Group, cls: str) -> None:
    grp.attrs["NX_class"] = cls


def _get_git_sha_and_dirty() -> tuple[str, bool]:
    """Return (sha, dirty) for the current repo, or ("unknown", False)."""
    try:
        sha = _subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=_subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            _subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=_subprocess.DEVNULL, text=True
            ).strip()
        )
        return sha, dirty
    except (_subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False


def _write_provenance(f: h5py.File, *, cli: str = "") -> None:
    """Write the global /dfxm_geo/ provenance group at file root.

    Idempotent: safe to call on a file that already has /dfxm_geo/.
    """
    g = f.require_group("/dfxm_geo")
    sha, dirty = _get_git_sha_and_dirty()
    try:
        ver = _pkg_version("dfxm-geo")
    except Exception:
        ver = "unknown"

    fields = {
        "version": ver,
        "git_sha": sha,
        "git_dirty": dirty,
        "hostname": _socket.gethostname(),
        "python_version": _sys.version.split()[0],
        "numpy_version": np.__version__,
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "cli": cli,
    }
    for name, val in fields.items():
        if name in g:
            del g[name]
        g.create_dataset(name, data=val)


def write_h5_scan(
    path: Path,
    scan_id: str,
    images: np.ndarray,
    phi: np.ndarray | None = None,
    chi: np.ndarray | None = None,
    title: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    sample_name: str | None = None,
    sample_dis: float | None = None,
    sample_ndis: int | None = None,
    sample_remount: str | None = None,
    Hg: np.ndarray | None = None,
    q_hkl: np.ndarray | None = None,
    theta: float | None = None,
    psize: float | None = None,
    zl_rms: float | None = None,
) -> None:
    """Write a single BLISS scan to an HDF5 file (creates or appends).

    Args:
        path: Output HDF5 file path. Created if missing; appended if exists.
        scan_id: BLISS scan identifier, e.g. "1.1" for the first scan.
        images: Image stack, shape (N_frames, H, W), dtype float64.
        phi: Phi motor positions per frame, shape (N_frames,), in radians.
            Stored on disk in degrees with units attr.
        chi: Chi motor positions per frame, shape (N_frames,), in radians.
            Stored on disk in degrees with units attr.
        title: Human-readable scan title string (e.g. BLISS fscan2d command).
        start_time: ISO-8601 start timestamp, e.g. "2026-05-17T10:00:00".
        end_time: ISO-8601 end timestamp, e.g. "2026-05-17T10:00:30".
        sample_name: Human-readable sample description string.
        sample_dis: Dislocation density (1/µm²).
        sample_ndis: Number of dislocations in the simulation.
        sample_remount: Sample remount label, e.g. "S1".
        Hg: Grain orientation matrices, shape (N_grains, 3, 3).
        q_hkl: Scattering vector in crystal coordinates, shape (3,).
        theta: Bragg angle in degrees.
        psize: Pixel size in mm.
        zl_rms: RMS depth-of-field parameter in µm.
    """
    mode = "a" if path.exists() else "w"
    with h5py.File(path, mode) as f:
        scan = f.require_group(scan_id)
        _set_nx_class(scan, "NXentry")
        if title is not None:
            scan.create_dataset("title", data=title)
        if start_time is not None:
            scan.create_dataset("start_time", data=start_time)
        if end_time is not None:
            scan.create_dataset("end_time", data=end_time)
        det = scan.require_group("instrument/dfxm_sim_detector")
        _set_nx_class(scan["instrument"], "NXinstrument")
        _set_nx_class(det, "NXdetector")
        n_frames, h, w = images.shape
        det.create_dataset(
            "data",
            data=images,
            chunks=(1, h, w),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        if phi is not None and chi is not None:
            pos = scan.require_group("instrument/positioners")
            _set_nx_class(pos, "NXcollection")
            phi_ds = pos.create_dataset("phi", data=np.degrees(phi))
            phi_ds.attrs["units"] = "degree"
            chi_ds = pos.create_dataset("chi", data=np.degrees(chi))
            chi_ds.attrs["units"] = "degree"
        meas = scan.require_group("measurement")
        meas["dfxm_sim_detector"] = h5py.SoftLink(f"/{scan_id}/instrument/dfxm_sim_detector/data")
        if phi is not None and chi is not None:
            meas["phi"] = h5py.SoftLink(f"/{scan_id}/instrument/positioners/phi")
            meas["chi"] = h5py.SoftLink(f"/{scan_id}/instrument/positioners/chi")
        if any(x is not None for x in (sample_name, sample_dis, sample_ndis, sample_remount)):
            samp = scan.require_group("sample")
            _set_nx_class(samp, "NXsample")
            if sample_name is not None:
                samp.create_dataset("name", data=sample_name)
            if sample_dis is not None:
                samp.create_dataset("dis", data=float(sample_dis))
            if sample_ndis is not None:
                samp.create_dataset("ndis", data=int(sample_ndis))
            if sample_remount is not None:
                samp.create_dataset("sample_remount", data=sample_remount)
        if any(x is not None for x in (Hg, q_hkl, theta, psize, zl_rms)):
            d = scan.require_group("dfxm_geo")
            if Hg is not None:
                d.create_dataset("Hg", data=Hg)
            if q_hkl is not None:
                d.create_dataset("q_hkl", data=q_hkl)
            if theta is not None:
                d.create_dataset("theta", data=float(theta))
            if psize is not None:
                d.create_dataset("psize", data=float(psize))
            if zl_rms is not None:
                d.create_dataset("zl_rms", data=float(zl_rms))
