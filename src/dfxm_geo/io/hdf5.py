"""HDF5 output format for dfxm-forward simulations.

Writes BLISS-style ESRF HDF5 (compatible with darfix / darling) with
sim-specific provenance metadata. One file per `run_simulation` call,
containing 1-2 BLISS scans (`/1.1` dislocations, `/2.1` optional
perfect crystal).

See docs/output-format.md for the full schema.
"""

from __future__ import annotations

import datetime as _dt
import hashlib as _hashlib
import os
import socket as _socket
import subprocess as _subprocess
import sys as _sys
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import cast

import h5py
import numpy as np
from tqdm import tqdm

from dfxm_geo.direct_space import forward_model as _fm


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


def _sha256_of(path: Path) -> str:
    h = _hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _auto_max_workers() -> int:
    """Cap thread-pool workers by both CPU count and free memory.

    Each ``forward()`` call needs ~1 GiB of intermediates (``qs.squeeze().T``
    and ``qi`` are each ~270 MiB float64 at the default 510-pixel detector,
    plus the ``prob``/``pro``/``index`` arrays). We aim for ~1 GiB headroom
    per worker.

    Resolution order (highest precedence wins):
      1. ``DFXM_MAX_WORKERS`` env var, if set to a positive int.
      2. ``min(cpu_count, max(1, available_memory_gb - 2))`` if psutil is
         installed (subtracting ~2 GiB reserved for persistent forward_model
         state + OS).
      3. ``min(cpu_count, 4)`` fixed conservative cap when psutil is missing.
    """
    env = os.environ.get("DFXM_MAX_WORKERS")
    if env is not None:
        try:
            n = int(env)
            if n >= 1:
                return n
        except ValueError:
            pass  # fall through to auto-detect

    cpu = os.cpu_count() or 1
    try:
        import psutil
    except ImportError:
        return min(cpu, 4)

    avail_gb = psutil.virtual_memory().available / (1024**3)
    usable_gb = max(1.0, avail_gb - 2.0)  # reserve 2 GiB for persistent state + OS
    mem_cap = max(1, int(usable_gb // 1))
    return min(cpu, mem_cap)


def _compute_frame(args: tuple) -> tuple[int, np.ndarray]:
    """Worker function: run forward() and return (frame_idx, image).

    args = (frame_idx, Hg, phi, chi)
    """
    frame_idx, Hg, phi, chi = args
    im = cast(np.ndarray, _fm.forward(Hg, phi=phi, chi=chi))
    return frame_idx, im


def _save_scan_parallel_to_h5(
    path: Path,
    scan_id: str,
    Hg: np.ndarray,
    phi_range: float,
    phi_steps: int,
    chi_range: float,
    chi_steps: int,
    max_workers: int | None = None,
    detector_shape: tuple[int, int] | None = None,
) -> None:
    """W2 producer-consumer: workers compute forward() and return; main thread writes to HDF5.

    Pre-allocates `(N_frames, H, W)` dataset, dispatches workers, and writes
    each result into its frame_idx slot as it returns. Frame ordering follows
    fscan2d convention: phi inner, chi outer. frame_idx = chi_idx * phi_steps + phi_idx.

    Args:
        path: Output HDF5 file (created or appended).
        scan_id: BLISS scan id like "1.1".
        Hg: Strain field passed to forward().
        phi_range, phi_steps, chi_range, chi_steps: rocking grid params (degrees + counts).
        max_workers: Override for `_auto_max_workers()`.
        detector_shape: (H, W) of the forward() output. If None, probes one frame
            up-front to discover the shape.
    """
    Phi = np.linspace(-np.deg2rad(phi_range), np.deg2rad(phi_range), phi_steps)
    Chi = np.linspace(-np.deg2rad(chi_range), np.deg2rad(chi_range), chi_steps)

    if detector_shape is None:
        # Run one frame to learn the detector shape, so we can pre-allocate.
        probe = cast(np.ndarray, _fm.forward(Hg, phi=float(Phi[0]), chi=float(Chi[0])))
        H, W = probe.shape
    else:
        H, W = detector_shape
        probe = None

    N = phi_steps * chi_steps
    mode = "a" if path.exists() else "w"
    workers = max_workers if max_workers is not None else _auto_max_workers()

    with h5py.File(path, mode) as f:
        scan = f.require_group(scan_id)
        _set_nx_class(scan, "NXentry")
        det = scan.require_group("instrument/dfxm_sim_detector")
        _set_nx_class(scan["instrument"], "NXinstrument")
        _set_nx_class(det, "NXdetector")
        if "data" in det:
            del det["data"]
        ds = det.create_dataset(
            "data",
            shape=(N, H, W),
            dtype=np.float64,
            chunks=(1, H, W),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        args_list = []
        for chi_idx in range(chi_steps):
            for phi_idx in range(phi_steps):
                k = chi_idx * phi_steps + phi_idx
                if probe is not None and k == 0:
                    ds[0] = probe  # we already computed frame 0
                    continue
                args_list.append((k, Hg, float(Phi[phi_idx]), float(Chi[chi_idx])))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, im in tqdm(ex.map(_compute_frame, args_list), total=len(args_list)):
                ds[k] = im


def _write_provenance(
    f: h5py.File, *, cli: str = "", kernel_npz: Path | None = None, config_toml: str | None = None
) -> None:
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

    if kernel_npz is not None:
        k = g.require_group("kernel")
        if "pkl_fn" in k:
            del k["pkl_fn"]
        k.create_dataset("pkl_fn", data=Path(kernel_npz).name)
        if "sha256" in k:
            del k["sha256"]
        k.create_dataset("sha256", data=_sha256_of(Path(kernel_npz)))
        # Mirror the kernel's bundled params for self-description.
        with np.load(kernel_npz) as arch:
            for key in arch.files:
                if key == "Resq_i":
                    continue  # the array itself, not metadata
                if key in k:
                    del k[key]
                k.create_dataset(key, data=arch[key])

    if config_toml is not None:
        if "config_toml" in g:
            del g["config_toml"]
        g.create_dataset("config_toml", data=config_toml)


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
        if images.size > 0:
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
        if images.size > 0:
            meas["dfxm_sim_detector"] = h5py.SoftLink(
                f"/{scan_id}/instrument/dfxm_sim_detector/data"
            )
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


def load_h5_scan(
    path: Path,
    scan_id: str = "1.1",
    phi_steps: int | None = None,
    chi_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load a BLISS scan stack from a dfxm-geo HDF5.

    Returns the same tuple shape `load_images` did so postprocess code
    stays unchanged.

    Args:
        path: HDF5 file path.
        scan_id: BLISS scan id, default "1.1" (dislocations).
        phi_steps, chi_steps: needed for the (phi, chi, H, W) reshape. If
            omitted, both are inferred from the embedded /dfxm_geo/config_toml.

    Returns:
        (stack, stack_reshape, dim_1, dim_2) — same as the legacy
        `load_images` signature.
    """
    with h5py.File(path, "r") as f:
        data = f[f"/{scan_id}/instrument/dfxm_sim_detector/data"][...]
        if phi_steps is None or chi_steps is None:
            import tomllib

            cfg = tomllib.loads(f["/dfxm_geo/config_toml"][()].decode())
            phi_steps = phi_steps or cfg["scan"]["phi_steps"]
            chi_steps = chi_steps or cfg["scan"]["chi_steps"]

    n, h, w = data.shape
    if n != phi_steps * chi_steps:
        raise ValueError(
            f"Scan {scan_id} has {n} frames, expected "
            f"{phi_steps}*{chi_steps}={phi_steps * chi_steps}"
        )
    # Reshape: fscan2d order is phi-inner, chi-outer.
    # data[k] for k = chi_idx * phi_steps + phi_idx
    # We want stack_reshape[phi_idx, chi_idx] == data[chi_idx*phi_steps + phi_idx]
    # which means reshape to (chi_steps, phi_steps, H, W) then transpose.
    stack_reshape = data.reshape(chi_steps, phi_steps, h, w).transpose(1, 0, 2, 3)
    return data, stack_reshape, h, w


def _scan_title(phi_range: float, phi_steps: int, chi_range: float, chi_steps: int) -> str:
    """fscan2d title string that darfix auto-detects on."""
    return (
        f"fscan2d phi {-phi_range:.6f} {phi_range:.6f} {phi_steps} "
        f"chi {-chi_range:.6f} {chi_range:.6f} {chi_steps} 1.0"
    )


def write_simulation_h5(
    path: Path,
    *,
    Hg: np.ndarray,
    q_hkl: np.ndarray,
    phi_range: float,
    phi_steps: int,
    chi_range: float,
    chi_steps: int,
    include_perfect_crystal: bool = True,
    sample_dis: float,
    sample_ndis: int,
    sample_remount: str,
    config_toml: str,
    cli: str,
    kernel_npz: Path | None = None,
    max_workers: int | None = None,
    crystal_mode: str | None = None,
    scan_mode: str | None = None,
    scanned_axes: list[str] | None = None,
) -> None:
    """One-call entry point: writes /dfxm_geo/ provenance + /1.1 + optional /2.1.

    Called by pipeline.run_simulation. Holds all info needed for a fully
    self-contained, reproducible output file.
    """
    import datetime as _dt2

    # Resolve kernel path for provenance hashing if not given.
    if kernel_npz is None:
        kernel_npz = _fm._loaded_kernel_path
        if kernel_npz is None:
            raise RuntimeError(
                "no kernel loaded — call _lookup_and_load_kernel(hkl, keV) "
                "before writing HDF5 provenance."
            )

    Phi = np.linspace(-np.deg2rad(phi_range), np.deg2rad(phi_range), phi_steps)
    Chi = np.linspace(-np.deg2rad(chi_range), np.deg2rad(chi_range), chi_steps)
    n = phi_steps * chi_steps
    phi_per_frame = np.empty(n, dtype=np.float64)
    chi_per_frame = np.empty(n, dtype=np.float64)
    for chi_idx in range(chi_steps):
        for phi_idx in range(phi_steps):
            k = chi_idx * phi_steps + phi_idx
            phi_per_frame[k] = Phi[phi_idx]
            chi_per_frame[k] = Chi[chi_idx]

    title = _scan_title(phi_range, phi_steps, chi_range, chi_steps)

    def _now() -> str:
        return _dt2.datetime.now(_dt2.UTC).isoformat(timespec="seconds")

    # Phase 1: /1.1 dislocations — image data first via the parallel writer,
    # then metadata-only append via write_h5_scan.
    start_dislocs = _now()
    _save_scan_parallel_to_h5(
        path,
        "1.1",
        Hg,
        phi_range,
        phi_steps,
        chi_range,
        chi_steps,
        max_workers=max_workers,
    )
    end_dislocs = _now()
    write_h5_scan(
        path,
        scan_id="1.1",
        images=np.empty(0),  # metadata-only append
        phi=phi_per_frame,
        chi=chi_per_frame,
        title=title,
        start_time=start_dislocs,
        end_time=end_dislocs,
        sample_name="simulated, dislocations",
        sample_dis=sample_dis,
        sample_ndis=sample_ndis,
        sample_remount=sample_remount,
        Hg=Hg,
        q_hkl=q_hkl,
        theta=float(_fm.theta),
        psize=float(_fm.psize),
        zl_rms=float(_fm.zl_rms),
    )

    # Phase 2: global provenance + B+C Task 13 scan/crystal-mode attrs on /N.1.
    with h5py.File(path, "a") as f:
        _write_provenance(f, cli=cli, kernel_npz=kernel_npz, config_toml=config_toml)
        # Write scan/crystal mode attrs on /1.1 for darfix/darling/silx inspection.
        grp_1_1 = f["1.1"]
        if scan_mode is not None:
            grp_1_1.attrs["scan_mode"] = scan_mode
        if scanned_axes is not None:
            grp_1_1.attrs["scanned_axes"] = list(scanned_axes)
        if crystal_mode is not None:
            grp_1_1.attrs["crystal_mode"] = crystal_mode

    # Phase 3: optional perfect-crystal scan /2.1
    if include_perfect_crystal:
        start_perf = _now()
        _save_scan_parallel_to_h5(
            path,
            "2.1",
            np.zeros_like(Hg),
            phi_range,
            phi_steps,
            chi_range,
            chi_steps,
            max_workers=max_workers,
        )
        end_perf = _now()
        write_h5_scan(
            path,
            scan_id="2.1",
            images=np.empty(0),
            phi=phi_per_frame,
            chi=chi_per_frame,
            title=title,
            start_time=start_perf,
            end_time=end_perf,
            sample_name="simulated, perfect crystal",
            sample_dis=sample_dis,
            sample_ndis=sample_ndis,
            sample_remount=sample_remount,
            Hg=np.zeros_like(Hg),
            q_hkl=q_hkl,
            theta=float(_fm.theta),
            psize=float(_fm.psize),
            zl_rms=float(_fm.zl_rms),
        )
