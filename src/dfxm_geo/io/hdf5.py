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
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from dfxm_geo.pipeline import ScanFrames

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

# (frame_idx, Hg, phi, chi, two_dtheta) — one frame's worth of args for `_compute_frame`.
_FrameArgs = tuple[int, np.ndarray, float, float, float]
"""(frame_idx, Hg, phi_rad, chi_rad, two_dtheta_rad)"""


@dataclass(frozen=True)
class ScanSpec:
    """One BLISS scan worth of work, yielded by an identification runner.

    Consumed by `write_identification_h5`: the orchestrator turns each
    `ScanSpec` into one master `/N.1` entry plus one or more per-scan
    detector files (one per key in `detectors`).

    Attributes:
        title: Scan title string written to `/N.1/title`.
        sample: NXsample contents for this scan (see per-mode layouts in
            the design spec).
        positioners: motor-axis name → 1-D array (scanned) or scalar (fixed).
            ASSUMES input is in radians.
        dfxm_geo: sim-specific per-scan metadata (Hg, q_hkl, theta, psize,
            zl_rms). Any subset may be supplied.
        detectors: detector_name → list of frame args tuples
            `(frame_idx, Hg, phi, chi, two_dtheta)` for the parallel writer. Each
            detector becomes its own LIMA-style file inside the scan dir.
        attrs: per-`/N.1` attrs — at minimum `scan_mode`, `scanned_axes`,
            `identify_mode`.
    """

    title: str
    sample: dict
    positioners: dict[str, np.ndarray | float]
    dfxm_geo: dict
    detectors: dict[str, list[_FrameArgs]]
    attrs: dict[str, str | list[str]]


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


def _compute_frame(args: _FrameArgs) -> tuple[int, np.ndarray]:
    """Worker function: run forward() and return (frame_idx, image).

    args = (frame_idx, Hg, phi, chi, two_dtheta)
    """
    frame_idx, Hg, phi, chi, two_dtheta = args
    im = cast(np.ndarray, _fm.forward(Hg, phi=phi, chi=chi, TwoDeltaTheta=two_dtheta))
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
        sample: dict[str, object],
        positioners: dict[str, np.ndarray | float],
        detector_links: dict[str, tuple[Path, str]],
        dfxm_geo: dict[str, object],
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
            # z is a translation in micrometers; the other axes are angular
            # (radians upstream → degrees on disk).
            if axis_name == "z":
                if isinstance(val, np.ndarray):
                    ds = pos.create_dataset(axis_name, data=val)
                else:
                    ds = pos.create_dataset(axis_name, data=float(val))
                ds.attrs["units"] = "micrometer"
            else:
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
            for key, g_val in dfxm_geo.items():
                if g_val is None:
                    continue
                if key in d:
                    del d[key]
                if isinstance(g_val, (int, float)):
                    d.create_dataset(key, data=float(g_val))
                else:
                    d.create_dataset(key, data=g_val)


def _write_sample_dict(group: h5py.Group, sample: dict[str, object]) -> None:
    """Write a sample dict into an NXsample group.

    Scalars and arrays become datasets; nested dicts become sub-groups.
    Only the key ``"dislocations"`` is special-cased: it becomes an
    NXcollection whose children are NXsample sub-groups (one per
    dislocation index). Every other nested dict becomes a plain NXsample
    sub-group, with its contents recursed into.
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
        elif isinstance(val, (int, float, str)):
            group.create_dataset(key, data=val)
        else:
            group.create_dataset(key, data=np.asarray(val))


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
        phi_steps, chi_steps: needed for the (chi, phi, H, W) reshape (chi
            outer, phi inner — the layout the mosaicity COM functions consume).
            If omitted, both are inferred from the embedded /dfxm_geo/config_toml.

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
    # Reshape: fscan2d order is phi-inner, chi-outer, i.e.
    # data[k] for k = chi_idx * phi_steps + phi_idx. Reshaping to
    # (chi_steps, phi_steps, H, W) therefore lands chi on axis 0 and phi on
    # axis 1 directly (stack_reshape[chi_idx, phi_idx] == data[k]). This is
    # the (chi, phi) layout the mosaicity COM functions consume
    # (`compute_com_maps`, `compute_chi_shift`); do NOT transpose to (phi, chi)
    # — that silently swaps the φ and χ COM maps (every grid had φ↔χ swapped
    # until v2.0.2; only undetected because production used phi_steps==chi_steps).
    stack_reshape = data.reshape(chi_steps, phi_steps, h, w)
    return data, stack_reshape, h, w


def _scan_title(phi_range: float, phi_steps: int, chi_range: float, chi_steps: int) -> str:
    """fscan2d title string that darfix auto-detects on."""
    return (
        f"fscan2d phi {-phi_range:.6f} {phi_range:.6f} {phi_steps} "
        f"chi {-chi_range:.6f} {chi_range:.6f} {chi_steps} 1.0"
    )


def _scan_title_from_frames(frames: ScanFrames, phi_steps: int, chi_steps: int) -> str:
    """fscan2d title string derived from a ScanFrames (radians, explicit step counts).

    The darfix-parsed title encodes per-axis step counts separately; for non-
    square grids, they cannot be inferred from n_frames alone.
    """
    phi_min = float(frames.phi_pf.min())
    phi_max = float(frames.phi_pf.max())
    chi_min = float(frames.chi_pf.min())
    chi_max = float(frames.chi_pf.max())
    return (
        f"fscan2d phi {phi_min:.6f} {phi_max:.6f} {phi_steps} "
        f"chi {chi_min:.6f} {chi_max:.6f} {chi_steps} 1.0"
    )


def write_identification_h5(
    output_dir: Path,
    *,
    scan_iter: Iterable[ScanSpec],
    cli: str,
    config_toml: str,
    kernel_npz: Path | None = None,
    max_workers: int | None = None,
) -> int:
    """Drive an identification run: consume ScanSpecs, write master + per-scan dirs.

    For each ScanSpec yielded:
      1. Create `output_dir/scanNNNN/`.
      2. For each `(detector_name, args_list)` in `spec.detectors`, write
         `output_dir/scanNNNN/<name>_0000.h5` via the parallel writer.
      3. Call `master.add_scan(scan_id=f"{N}.1", detector_links=..., ...)`.

    Returns the count of scans written.
    """

    if kernel_npz is None:
        kernel_npz = _fm._loaded_kernel_path
        if kernel_npz is None:
            raise RuntimeError(
                "no kernel loaded — call _lookup_and_load_kernel(hkl, keV) "
                "before writing identification HDF5 provenance."
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / MASTER_IDENTIFY
    n_scans = 0

    def _now() -> str:
        return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")

    with MasterWriter(
        master_path, cli=cli, config_toml=config_toml, kernel_npz=kernel_npz
    ) as master:
        for idx, spec in enumerate(scan_iter):
            scan_id = f"{idx + 1}.1"
            scan_dir_rel = Path(SCAN_DIR_FMT.format(idx + 1))
            scan_dir = output_dir / scan_dir_rel
            scan_dir.mkdir(parents=True, exist_ok=True)
            detector_links: dict[str, tuple[Path, str]] = {}
            start_time = _now()
            for det_name, args_list in spec.detectors.items():
                det_file = scan_dir / DETECTOR_FILE_FMT.format(name=det_name)
                _compute_and_write_detector_file_parallel(
                    det_file, args_list, max_workers=max_workers
                )
                detector_links[det_name] = (
                    scan_dir_rel / DETECTOR_FILE_FMT.format(name=det_name),
                    DETECTOR_INTERNAL_PATH,
                )
            end_time = _now()
            master.add_scan(
                scan_id=scan_id,
                title=spec.title,
                start_time=start_time,
                end_time=end_time,
                sample=spec.sample,
                positioners=spec.positioners,
                detector_links=detector_links,
                dfxm_geo=spec.dfxm_geo,
                attrs=spec.attrs,
            )
            n_scans += 1
    return n_scans


def write_simulation_h5(
    path: Path,
    *,
    Hg: np.ndarray,
    q_hkl: np.ndarray,
    frames: ScanFrames,
    include_perfect_crystal: bool = True,
    sample_dis: float | None,
    sample_ndis: int,
    sample_remount: str,
    config_toml: str,
    cli: str,
    kernel_npz: Path | None = None,
    max_workers: int | None = None,
    crystal_mode: str | None = None,
    scan_mode: str | None = None,
    scanned_axes: list[str] | None = None,
    positioners: dict[str, np.ndarray | float] | None = None,
    Hg_provider: Callable[[float], tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    """One-call entry point for forward mode, v1.2.0 layout.

    `path` is the master file (`<out_dir>/dfxm_geo.h5`); detector pixels live
    in sibling `scan0001/` / `scan0002/` directories alongside it, linked
    via ExternalLink.

    `frames` replaces the legacy `phi_range/phi_steps/chi_range/chi_steps` kwargs
    (removed in v1.3.0-A).  All per-frame arrays in `frames` are in radians /
    micrometers — no degree conversion is applied here.

    `positioners`, if provided, is passed verbatim to `master.add_scan`.
    If None, it is derived from `frames` + `scanned_axes`: scanned axes
    receive the full per-frame array, fixed axes default to 0.0.
    """

    if kernel_npz is None:
        kernel_npz = _fm._loaded_kernel_path
        if kernel_npz is None:
            raise RuntimeError(
                "no kernel loaded — call _lookup_and_load_kernel(hkl, keV) "
                "before writing HDF5 provenance."
            )

    out_dir = path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    n = frames.n_frames
    phi_per_frame = frames.phi_pf
    chi_per_frame = frames.chi_pf
    two_dtheta_per_frame = frames.two_dtheta_pf
    z_per_frame = frames.z_pf

    phi_steps = int(np.unique(frames.phi_pf).size)
    chi_steps = int(np.unique(frames.chi_pf).size)
    title = _scan_title_from_frames(frames, phi_steps=phi_steps, chi_steps=chi_steps)

    def _now() -> str:
        return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")

    def _build_args(Hg_in: np.ndarray) -> list[_FrameArgs]:
        return [
            (
                i,
                Hg_in,
                float(phi_per_frame[i]),
                float(chi_per_frame[i]),
                float(two_dtheta_per_frame[i]),
            )
            for i in range(n)
        ]

    attrs_1_1: dict[str, str | list[str]] = {}
    if scan_mode is not None:
        attrs_1_1["scan_mode"] = scan_mode
    if scanned_axes is not None:
        attrs_1_1["scanned_axes"] = list(scanned_axes)
    if crystal_mode is not None:
        attrs_1_1["crystal_mode"] = crystal_mode

    # Build positioners dict once; reused for both /1.1 and /2.1 add_scan calls.
    if positioners is None:
        pos_dict: dict[str, np.ndarray | float] = {
            "phi": phi_per_frame,
            "chi": chi_per_frame,
        }
        if scan_mode and "two_dtheta" in (scanned_axes or []):
            pos_dict["two_dtheta"] = two_dtheta_per_frame
        else:
            pos_dict["two_dtheta"] = 0.0
        if scan_mode and "z" in (scanned_axes or []):
            pos_dict["z"] = z_per_frame
        else:
            pos_dict["z"] = 0.0
    else:
        pos_dict = positioners

    # Build the per-scan plan: /1.1 dislocations always, /2.1 perfect crystal
    # only if requested. Both scans share positioners, attrs, sample metadata
    # (apart from name), and physical params; only Hg + scan index differ.
    scans: list[tuple[int, str, np.ndarray]] = [(1, "simulated, dislocations", Hg)]
    if include_perfect_crystal:
        scans.append((2, "simulated, perfect crystal", np.zeros_like(Hg)))

    with MasterWriter(path, cli=cli, config_toml=config_toml, kernel_npz=kernel_npz) as master:
        for scan_idx, sample_name, Hg_for_scan in scans:
            scan_dir = out_dir / SCAN_DIR_FMT.format(scan_idx)
            det_path = scan_dir / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector")
            start = _now()
            if Hg_provider is not None and scan_idx == 1:
                # Dislocation scan: use Hg_provider for z-aware per-frame Hg.
                # z-outermost frame order (from _build_scan_frames) means all frames
                # at a given z are contiguous; we drop each Hg after its last frame.
                z_to_Hg: dict[float, np.ndarray] = {}
                provider_args_list: list[_FrameArgs] = []
                for k in range(n):
                    z = float(z_per_frame[k])
                    if z not in z_to_Hg:
                        z_to_Hg[z] = Hg_provider(z)[0]  # discard q_hkl on per-z calls
                    provider_args_list.append(
                        (
                            k,
                            z_to_Hg[z],
                            float(phi_per_frame[k]),
                            float(chi_per_frame[k]),
                            float(two_dtheta_per_frame[k]),
                        )
                    )
                    if k == n - 1 or float(z_per_frame[k + 1]) != z:
                        del z_to_Hg[z]
                args_list_for_scan = provider_args_list
            else:
                # Perfect crystal /2.1, or back-compat single-Hg case.
                args_list_for_scan = _build_args(Hg_for_scan)
            _compute_and_write_detector_file_parallel(
                det_path, args_list_for_scan, max_workers=max_workers
            )
            end = _now()
            sample = {
                "name": sample_name,
                "ndis": int(sample_ndis),
                "sample_remount": sample_remount,
            }
            # `dis` (wall-mode width) is only meaningful for `wall` crystal
            # mode; centered/random_dislocations pass None and omit the key.
            if sample_dis is not None:
                sample["dis"] = float(sample_dis)
            master.add_scan(
                scan_id=f"{scan_idx}.1",
                title=title,
                start_time=start,
                end_time=end,
                sample=sample,
                positioners=pos_dict,
                detector_links={
                    "dfxm_sim_detector": (
                        Path(SCAN_DIR_FMT.format(scan_idx))
                        / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector"),
                        DETECTOR_INTERNAL_PATH,
                    )
                },
                dfxm_geo={
                    "Hg": Hg_for_scan,
                    "q_hkl": q_hkl,
                    "theta": float(_fm.theta),
                    "psize": float(_fm.psize),
                    "zl_rms": float(_fm.zl_rms),
                },
                attrs=attrs_1_1,
            )
