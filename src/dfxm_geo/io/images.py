"""Image writer for identification z-scan.

The `.npy` write path (`save_images_parallel` / `save_image`) is kept here
until v1.2.0, when identification will migrate to HDF5 output.  All read
helpers (`load_image`, `load_images`, `load_images_parallel`) and the legacy
EDF writer (`save_edfs`) were removed in v1.1.0; see `io/migrate.py` for the
migration helper and `io/hdf5.py` for the HDF5 equivalents.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from dfxm_geo.direct_space import forward_model as _fm

# Measured per-worker transient peak of `forward_from_static`, in bytes per
# ray (NN1*NN2*NN3). The live intermediates per call are `qi` (3*8 B/ray),
# the three int16 index arrays (3*2 B/ray), the bool mask, and `prob`
# (~8 B/ray plus a float32 copy) — measured at ~48 B/ray on the px510/Nsub=1
# detector. We budget 2x that as a safety envelope (allocator overhead +
# transient double-buffering like `prob.astype(np.float32)`). This SCALES
# with ray count, so Nsub=2 publication runs (8x rays) reserve ~8x per worker
# automatically. `base_qc` is shared read-only across workers (24 B/ray,
# counted once below), not per-worker. See tests/_perf_static_hoist.py and
# the static-hoist handoff for the measurement.
_PER_WORKER_BYTES_PER_RAY = 96
_BASE_QC_BYTES_PER_RAY = 24


def _auto_max_workers() -> int:
    """Cap thread-pool workers by both CPU count and free memory.

    Per-worker memory is estimated from the *actual* detector ray count
    (``NN1*NN2*NN3``) at ~96 B/ray (see ``_PER_WORKER_BYTES_PER_RAY``), not a
    fixed slab — the static hoist made ``base_qc`` a shared read-only array, so
    the per-worker footprint is just the dynamic intermediates and is small at
    Nsub=1 (~67 MiB/worker) while scaling up for Nsub=2 publication runs.

    Resolution order (highest precedence wins):
      1. ``DFXM_MAX_WORKERS`` env var, if set to a positive int.
      2. ``min(cpu_count, usable_gb / per_worker_gb)`` if psutil is installed,
         where ``usable_gb`` reserves ~2 GiB for persistent forward_model state
         (the Resq_i LUT, Hg) + OS, minus the once-shared ``base_qc``.
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

    ray_count = (_fm.NN1 * _fm.NN2 * _fm.NN3) or 1
    base_qc_gb = ray_count * _BASE_QC_BYTES_PER_RAY / (1024**3)  # shared once
    per_worker_gb = ray_count * _PER_WORKER_BYTES_PER_RAY / (1024**3)

    avail_gb = psutil.virtual_memory().available / (1024**3)
    # Reserve ~2 GiB for persistent state + OS and the once-shared base_qc.
    usable_gb = max(0.5, avail_gb - 2.0 - base_qc_gb)
    mem_cap = max(1, int(usable_gb / per_worker_gb))
    return min(cpu, mem_cap)


def save_image(args: tuple) -> None:
    """Render one frame via forward_from_static and save it as .npy.

    args = (base_qc, phi, chi, j, i, fpath, fn_prefix, ftype), where
    base_qc = forward_model.precompute_forward_static(Hg) is shared across
    all frames of the scan.
    """
    base_qc, phi, chi, j, i, fpath, fn_prefix, ftype = args
    im = _fm.forward_from_static(base_qc, phi=phi, chi=chi)
    fn_suffix = f"{i}".zfill(4) + "_" + f"{j}".zfill(4) + ftype
    np.save(os.path.join(fpath + fn_prefix + fn_suffix), im)


def save_images_parallel(
    Hg: np.ndarray,
    phi_range: float,
    phi_steps: int,
    chi_range: float,
    chi_steps: int,
    fpath: str,
    fn_prefix: str,
    ftype: str,
    max_workers: int | None = None,
) -> bool:
    """
    Generate a grid of parameter combinations and save images in parallel.
    ----------------------------------------------------------------------------------------------
    Parameters:
        Hg (float): A parameter.
        phi_range (float): Half-range of phi values in radians.
        phi_steps (int): Number of steps in the phi range.
        chi_range (float): Half-range of chi values in radians.
        chi_steps (int): Number of steps in the chi range.
        fpath (str): Path to a folder where the images will be saved.
        fn_prefix (str): Prefix for the image filenames.
        ftype (str): Filetype extension for the images (e.g., '.png').
        max_workers (int | None): Maximum parallel workers for the thread
            pool. None (default) uses the DFXM_MAX_WORKERS env var if set,
            otherwise auto-detects based on CPU count and available memory
            (see _auto_max_workers).
    ----------------------------------------------------------------------------------------------
    Returns:
        True (bool): True if the function completes successfully.
    """
    # Scan ranges are radians (project-wide convention); used directly.
    Phi = np.linspace(-phi_range, phi_range, phi_steps)
    Chi = np.linspace(-chi_range, chi_range, chi_steps)

    # Create a folder in the specified path if it doesn't exist
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    base_qc = _fm.precompute_forward_static(Hg)
    args_list = [
        (base_qc, Phi[j], Chi[i], j, i, fpath, fn_prefix, ftype)
        for i in range(chi_steps)
        for j in range(phi_steps)
    ]

    workers = max_workers if max_workers is not None else _auto_max_workers()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Consume the lazy executor.map iterator via list(...) so tqdm
        # actually advances; the per-task return values are unused.
        list(tqdm(executor.map(save_image, args_list), total=len(args_list)))

    return True
