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

# Output layout constants (v1.2.0).
MASTER_FORWARD = "dfxm_geo.h5"
MASTER_IDENTIFY = "dfxm_identify.h5"
SCAN_DIR_FMT = "scan{:04d}"
DETECTOR_FILE_FMT = "{name}_0000.h5"
DETECTOR_INTERNAL_PATH = "/entry_0000/dfxm_sim_detector/image"


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


# (frame_idx, Hg, phi, chi) — one frame's worth of args for `_compute_frame`.
_FrameArgs = tuple[int, np.ndarray, float, float]


def _compute_frame(args: _FrameArgs) -> tuple[int, np.ndarray]:
    """Worker function: run forward() and return (frame_idx, image).

    args = (frame_idx, Hg, phi, chi)
    """
    frame_idx, Hg, phi, chi = args
    im = cast(np.ndarray, _fm.forward(Hg, phi=phi, chi=chi))
    return frame_idx, im


def _create_detector_skeleton(
    f: h5py.File,
    *,
    n_frames: int,
    height: int,
    width: int,
    data: np.ndarray | None = None,
) -> h5py.Dataset:
    """Create the LIMA-style NX skeleton in ``f`` and return the image dataset.

    Builds /entry_0000 (NXentry) → dfxm_sim_detector (NXdetector) → image,
    plus the /entry_0000/plot (NXdata) and /entry_0000/measurement SoftLinks
    that point at `DETECTOR_INTERNAL_PATH`. When ``data`` is given, the
    dataset is created with the array inline (full-stack write). When
    ``data is None`` the dataset is pre-allocated at (n_frames, height,
    width) for streaming writes from worker threads.
    """
    f.attrs["NX_class"] = "NXroot"
    f.attrs["creator"] = "dfxm-geo"
    f.attrs["default"] = "entry_0000"
    entry = f.create_group("entry_0000")
    _set_nx_class(entry, "NXentry")
    det = entry.create_group("dfxm_sim_detector")
    _set_nx_class(det, "NXdetector")
    if data is not None:
        img = det.create_dataset(
            "image",
            data=data,
            chunks=(1, height, width),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
    else:
        img = det.create_dataset(
            "image",
            shape=(n_frames, height, width),
            dtype=np.float64,
            chunks=(1, height, width),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
    img.attrs["interpretation"] = "image"
    plot = entry.create_group("plot")
    _set_nx_class(plot, "NXdata")
    plot.attrs["signal"] = "image"
    plot["image"] = h5py.SoftLink(DETECTOR_INTERNAL_PATH)
    entry["measurement"] = h5py.SoftLink(DETECTOR_INTERNAL_PATH)
    return img


def _write_detector_file(path: Path, image_stack: np.ndarray) -> None:
    """Write a pre-computed (N, H, W) image stack as a LIMA-style detector file.

    Produces /entry_0000/dfxm_sim_detector/image with chunks=(1, H, W),
    gzip-4 + shuffle, @interpretation="image", plus NXdata/measurement
    soft-links to that dataset. Used by identification single/multi paths
    when frames are computed serially in RAM.
    """
    _, h, w = image_stack.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        _create_detector_skeleton(
            f, n_frames=image_stack.shape[0], height=h, width=w, data=image_stack
        )


def _compute_and_write_detector_file_parallel(
    path: Path,
    args_list: list[_FrameArgs],
    *,
    max_workers: int | None = None,
    detector_shape: tuple[int, int] | None = None,
) -> None:
    """Producer-consumer parallel writer for a LIMA-style detector file.

    Workers run `forward()` via `_compute_frame` and stream results into a
    pre-allocated (N_frames, H, W) dataset. The dataset is at the canonical
    `DETECTOR_INTERNAL_PATH` so masters can ExternalLink to it.

    Args:
        path: Output detector file. Parent dirs are created if missing.
        args_list: Sequence of (frame_idx, Hg, phi, chi) tuples, one per frame.
            Frame indices must be a contiguous 0..N-1 set.
        max_workers: Override for `_auto_max_workers()`.
        detector_shape: (H, W). If None, probes args_list[0] to discover shape.
    """
    if not args_list:
        raise ValueError("args_list must contain at least one frame")
    n_frames = len(args_list)

    if detector_shape is None:
        # probe doubles as frame-0 result when shape unknown
        probe_idx, probe_im = _compute_frame(args_list[0])
        h, w = probe_im.shape
    else:
        h, w = detector_shape
        probe_idx, probe_im = None, None

    path.parent.mkdir(parents=True, exist_ok=True)
    workers = max_workers if max_workers is not None else _auto_max_workers()

    with h5py.File(path, "w") as f:
        img = _create_detector_skeleton(f, n_frames=n_frames, height=h, width=w)

        if probe_im is not None:
            img[probe_idx] = probe_im
            workers_args = args_list[1:]
        else:
            workers_args = args_list

        if workers_args:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for k, im in tqdm(
                    ex.map(_compute_frame, workers_args),
                    total=len(workers_args),
                ):
                    img[k] = im


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


class MasterWriter:
    """Context manager that owns the master HDF5 handle for one pipeline run.

    Use:
        with MasterWriter(path, cli=..., config_toml=..., kernel_npz=...) as m:
            m.add_scan(scan_id="1.1", ...)
            m.add_scan(scan_id="2.1", ...)
        # provenance written on close

    The master is opened in mode 'w' (fresh file every run) so the writer
    is idempotent across re-runs to the same path.
    """

    def __init__(
        self,
        path: Path,
        *,
        cli: str,
        config_toml: str,
        kernel_npz: Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.cli = cli
        self.config_toml = config_toml
        self.kernel_npz = kernel_npz
        self._fh: h5py.File | None = None

    def __enter__(self) -> MasterWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = h5py.File(self.path, "w")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._fh is not None and exc_type is None:
                _write_provenance(
                    self._fh,
                    cli=self.cli,
                    kernel_npz=self.kernel_npz,
                    config_toml=self.config_toml,
                )
        finally:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def add_scan(
        self,
        *,
        scan_id: str,
        title: str,
        start_time: str,
        end_time: str,
        sample: dict,
        positioners: dict[str, np.ndarray | float],
        detector_links: dict[str, tuple[Path, str]],
        dfxm_geo: dict,
        attrs: dict[str, str | list[str]],
    ) -> None:
        """Append one BLISS scan entry `/<scan_id>` to the master.

        Args:
            scan_id: BLISS scan identifier, e.g. "1.1", "2.1", "3.1".
            title: Scan title string (e.g. fscan2d command).
            start_time, end_time: ISO-8601 timestamps recorded on `/N.1`.
            sample: Dict of NXsample contents. Numpy arrays become datasets;
                scalars become 0-D datasets. Special key "name" expected.
                Nested dicts (e.g. "dislocations") become NXcollection groups.
            positioners: Dict of motor axis name -> 1-D array (scanned) or
                scalar (fixed). Arrays are stored in degrees with units attr.
                ASSUMES input is in radians (matches existing convention).
            detector_links: Dict of detector name -> (rel_file_path, internal_h5_path).
                Each becomes an NXdetector group with a `data` ExternalLink.
                A `/N.1/measurement/<name>` SoftLink is also created.
            dfxm_geo: Sim-specific per-scan metadata: Hg, q_hkl, theta, psize, zl_rms.
                Any subset may be supplied; missing keys are skipped.
            attrs: Per-`/N.1` attributes (scan_mode, scanned_axes,
                crystal_mode | identify_mode).
        """
        if self._fh is None:
            raise RuntimeError("MasterWriter is not open; use as a context manager")
        f = self._fh

        scan = f.require_group(scan_id)
        _set_nx_class(scan, "NXentry")
        if "title" in scan:
            del scan["title"]
        scan.create_dataset("title", data=title)
        if "start_time" in scan:
            del scan["start_time"]
        scan.create_dataset("start_time", data=start_time)
        if "end_time" in scan:
            del scan["end_time"]
        scan.create_dataset("end_time", data=end_time)

        # Attrs (scan_mode / scanned_axes / crystal_mode | identify_mode)
        for k, v in attrs.items():
            if isinstance(v, list):
                scan.attrs[k] = list(v)
            else:
                scan.attrs[k] = v

        # /N.1/sample/
        samp = scan.require_group("sample")
        _set_nx_class(samp, "NXsample")
        _write_sample_dict(samp, sample)

        # /N.1/instrument/<detector_name>/data -> ExternalLink
        instr = scan.require_group("instrument")
        _set_nx_class(instr, "NXinstrument")
        meas = scan.require_group("measurement")
        _set_nx_class(meas, "NXcollection")
        for det_name, (rel_path, internal_path) in detector_links.items():
            det = instr.require_group(det_name)
            _set_nx_class(det, "NXdetector")
            if "data" in det:
                del det["data"]
            det["data"] = h5py.ExternalLink(str(rel_path).replace("\\", "/"), internal_path)
            if det_name in meas:
                del meas[det_name]
            meas[det_name] = h5py.SoftLink(f"/{scan_id}/instrument/{det_name}/data")

        # /N.1/instrument/positioners/
        pos = instr.require_group("positioners")
        _set_nx_class(pos, "NXcollection")
        for axis_name, val in positioners.items():
            if axis_name in pos:
                del pos[axis_name]
            if isinstance(val, np.ndarray):
                ds = pos.create_dataset(axis_name, data=np.degrees(val))
            else:
                ds = pos.create_dataset(axis_name, data=float(np.degrees(val)))
            ds.attrs["units"] = "degree"
            if axis_name in meas:
                del meas[axis_name]
            meas[axis_name] = h5py.SoftLink(f"/{scan_id}/instrument/positioners/{axis_name}")

        # /N.1/dfxm_geo/
        if dfxm_geo:
            d = scan.require_group("dfxm_geo")
            for key, val in dfxm_geo.items():
                if val is None:
                    continue
                if key in d:
                    del d[key]
                if isinstance(val, (int, float)):
                    d.create_dataset(key, data=float(val))
                else:
                    d.create_dataset(key, data=val)


def _write_sample_dict(group: h5py.Group, sample: dict) -> None:
    """Write a sample dict into an NXsample group.

    Scalars and arrays become datasets; nested dicts become sub-NXcollection
    or NXsample groups. Dict keys "dislocations", "primary", "secondary" get
    NXcollection / NXsample NX_class respectively; everything else is a scalar.
    """
    for key, val in sample.items():
        if key in group:
            del group[key]
        if isinstance(val, dict):
            sub = group.create_group(key)
            if key == "dislocations":
                _set_nx_class(sub, "NXcollection")
                for idx_key, sub_sample in val.items():
                    item = sub.create_group(str(idx_key))
                    _set_nx_class(item, "NXsample")
                    _write_sample_dict(item, sub_sample)
            else:
                _set_nx_class(sub, "NXsample")
                _write_sample_dict(sub, val)
        elif isinstance(val, (int, float)):
            group.create_dataset(key, data=float(val) if isinstance(val, float) else int(val))
        elif isinstance(val, str):
            group.create_dataset(key, data=val)
        else:
            group.create_dataset(key, data=np.asarray(val))


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
