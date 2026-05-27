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


def _auto_max_workers() -> int:
    """Cap thread-pool workers by both CPU count and free memory.

    The ``base_qc`` array (``qs.squeeze().T``, ~270 MiB float64 at the default
    510-pixel detector) is now precomputed once per scan and shared read-only
    across workers, so it no longer counts toward per-worker footprint. The
    dominant per-worker intermediate is ``qi`` (~270 MiB float64), plus the
    ``prob``/``pro``/``index`` arrays — roughly half the previous per-worker
    estimate. We still aim for ~1 GiB headroom per worker as a conservative
    margin.

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
