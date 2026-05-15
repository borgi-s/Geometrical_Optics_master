"""Image I/O: save/load simulated DFXM image stacks as .npy or .edf files."""

import os
from concurrent.futures import ThreadPoolExecutor

import fabio
import numpy as np
from tqdm import tqdm

from dfxm_geo.direct_space import forward_model as _fm
from dfxm_geo.io import check_folder


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


def save_image(args: tuple) -> None:
    """
    Save an image with specified parameters.
    ---------------------------------------------------------------------
    Parameters:
        args (tuple): A tuple containing the following elements:
            Hg (float): A parameter.
            phi (float): Angle in radians.
            chi (float): Angle in radians.
            j (int): Step in phi
            i (int): Step in chi
            fpath (str): Path to a folder where the image will be saved.
            fn_prefix (str): Prefix for the image filename.
            ftype (str): Filetype extension for the image (e.g., '.png').
    ---------------------------------------------------------------------
    Returns:
        None: The function saves an image but does not return a value.
    """
    Hg, phi, chi, j, i, fpath, fn_prefix, ftype = args
    im = _fm.forward(Hg, phi=phi, chi=chi)
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
        phi_range (float): Range of phi values in degrees.
        phi_steps (int): Number of steps in the phi range.
        chi_range (float): Range of chi values in degrees.
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
    Phi = np.linspace(-np.deg2rad(phi_range), np.deg2rad(phi_range), phi_steps)
    Chi = np.linspace(-np.deg2rad(chi_range), np.deg2rad(chi_range), chi_steps)

    # Create a folder in the specified path if it doesn't exist
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    args_list = [
        (Hg, Phi[j], Chi[i], j, i, fpath, fn_prefix, ftype)
        for i in range(chi_steps)
        for j in range(phi_steps)
    ]

    workers = max_workers if max_workers is not None else _auto_max_workers()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Consume the lazy executor.map iterator via list(...) so tqdm
        # actually advances; the per-task return values are unused.
        list(tqdm(executor.map(save_image, args_list), total=len(args_list)))

    return True


def save_edfs(
    imstack: np.ndarray,
    v: np.ndarray,
    u: np.ndarray,
    fpath: str,
    fn_prefix: str,
) -> bool:
    """
    Save EDF images from a given image stack and a set of v and u angles.
    ---------------------------------------------------------------------------
    Parameters:
        - image_stack: a 2D array of images
        - v_angles: a 1D array of v angles in degrees
        - u_angles: a 1D array of u angles in degrees
        - f_path: a string representing the path of the output file
        - fn_prefix: a string representing the prefix of the output file name
    """
    for i in range(len(v)):
        for j in range(len(u)):
            img1 = fabio.edfimage.EdfImage(data=imstack[j, i])
            chi1 = str(v[i] * 180 / np.pi)
            phi1 = str(u[j] * 180 / np.pi)
            img1.header = {
                "HeaderID": "EH:000001:000000:000000",
                "Image": "1",
                "ByteOrder": "LowByteFirst",
                "DataType": "UnsignedShort",
                "Dim_1": "170",
                "Dim_2": "510",
                "Size": "11059200",
                "scan": "mesh  diffry 8.535 8.555 10  chi 0.116607 0.236607 6  1",
                "date": "Mon Dec 12 22:05:32 2022",
                "epoch": "1670604623.6910",
                "offset": "0",
                "count_time": "1",
                "point_no": "5",
                "scan_no": "1573",
                "preset": "0",
                "col_end": "2559",
                "col_beg": "0",
                "row_end": "2159",
                "row_beg": "0",
                "counter_pos": "-1 39.4025 1.24457e-07 1.27043e-09 1.17154e-13 2.40476e-11 -1 -1 0 98.7543 10.3245 0 19886 0.396031 0 0 0 0 0 -1 -1 0 1 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
                "counter_mne": "diffstd srcur pico1 pico2 pico3 pico4 nfavg nfstd europv ffavg ffstd oxtemp p201_1 mc5 mc6 delV delI samV samI basavg basstd horstd sec horavg diffavg foc1 foc2 foc3 foc4 xrfSum roiCa roiSr devolt decurr depow lstemp zap_avg Tempwsp Tempsp Temp Tempout",
                "motor_pos": f"480 6.684 7.55132 0.600022 -0.124985 0.600022 -0.103324 4 -0.0485 6 -3.1645 7.28663 12.2036 0 -58.88 0 1618.46 0 0.0015 1.8425 0 -0.822 -0.0015 0.138 342 1.94081e-06 -0.700001 4.08922e-05 0 25 50 256.75 0.1072 83.3046 17.9763 0.144375 -0.009995 8.545 6.38635e-07 0 0.02 2.75495 {chi1} {phi1} 3.78 1908.95 91.867 3.03665 -1.12401 10.3341 19.5 0.5 2.11181 0.0499971 0.00647404 -11.4698 -6.12734 2.09967 -3.67327 1.98883 912.776 866 1300 550 150 -30 0 60 50 0 -2.85714e-07 -5000 -5.09849e-05 0.0999796 2.68525e-07 0.0999896 0.00532441 1.052 0.32 7.974 25 150 0 -27 0 0 0 0 0 -3.15866e-07 -0.5 0 0 0 1.438 2.437 9.42503 -2.75239 0.0255 25 0 0",
                "motor_mne": "unused mono x2z s5hg s5ho s5vg s5vo s3vg s3vo s3hg s3ho fffoc1 fffoc2 ffsel ffrotc ffy ffz ffpitch hxx hxy hxz hxyaw hxroll hxpitch cdx cdy cdz cdpitch cdyaw euro_sp icx obx oby obz obpitch obyaw difftz diffry diffrz diffty corz corx chi phi detz dety detx pcofoc hfoc hrotc htth s4hg s4ho s4vg s4vo nffoc s6hg s6ho s6vg s6vo auxx auxy auxz dcx dcy dcz nfrz nfx nfy nfz nfrotc mainx bstop s7hg s7ho s7vg s7vo smx smz smy samfoc furny mrot decoh DEVolt DECurr DEPow lenssel sptmemb wirey wirez delVsp py pz phpz phpy dty dtz focpad Tsp rpi_1 obz3",
                "suffix": ".edf",
                "prefix": "mosalocal_2x_00_",
                "dir": "/data/id06-hxm/inhouse/2022/run5/ihma320/id06-hxm/Al_sample_5/Al_sample_5_tick_750/mosalocal_2x_00",
                "run": "5",
                "title": "CCD Image",
            }
            fn_suffix = f"{i}".zfill(4) + "_" + f"{j}".zfill(4) + ".edf"
            check_folder("", fpath)
            img1.write(fpath + fn_prefix + fn_suffix)
    return True


def load_images(
    fpath: str,
    u_steps: int,
    v_steps: int,
    file_ext: str = ".npy",
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Load and reshape a stack of image files from a directory.
    --------------------------------------------------------------------------
    Parameters:
        fpath (str): Path to directory containing image files.
        u_steps (int): Number of steps to use for u values in reshaped stack.
        v_steps (int): Number of steps to use for v values in reshaped stack.
    --------------------------------------------------------------------------
    Returns:
        stack (ndarray): 3D NumPy array of loaded image files.
        stack_reshape (ndarray): 4D NumPy array of reshaped image stack.
        dim_1 (int): Dimension 1 size of loaded image files.
        dim_2 (int): Dimension 2 size of loaded image files.
    """
    if not os.path.isdir(fpath):
        raise ValueError(f"Directory does not exist: {fpath}")
    file_list = [f for f in os.listdir(fpath) if f.endswith(file_ext)]

    if not file_list:
        raise ValueError(f"Directory is empty or does not contain any {file_ext} files: {fpath}")

    file_list.sort()
    stack = np.empty(
        (len(file_list), *np.load(os.path.join(fpath, file_list[0]), allow_pickle=True).shape)
    )
    for i, file in enumerate(file_list):
        file_path = os.path.join(fpath, file)
        stack[i, :, :] = np.load(file_path)
    dim_1, dim_2 = stack.shape[1], stack.shape[2]
    stack_reshape = stack.reshape((u_steps, v_steps, dim_1, dim_2))

    return stack, stack_reshape, dim_1, dim_2


def load_image(file_path: str) -> np.ndarray:
    """Load a single .npy file. Thin wrapper around np.load."""
    return np.load(file_path)


def load_images_parallel(
    fpath: str,
    u_steps: int,
    v_steps: int,
    file_ext: str = ".npy",
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if not os.path.isdir(fpath):
        raise ValueError(f"Directory does not exist: {fpath}")

    file_list = [f for f in os.listdir(fpath) if f.endswith(file_ext)]

    if not file_list:
        raise ValueError(f"Directory is empty or does not contain any {file_ext} files: {fpath}")

    file_list.sort()
    num_files = len(file_list)

    with ThreadPoolExecutor(max_workers=4) as executor:
        loaded_images = list(
            executor.map(lambda file: np.load(os.path.join(fpath, file)), file_list)
        )

    stack = np.empty((num_files, *loaded_images[0].shape))

    for i, img in enumerate(loaded_images):
        stack[i, :, :] = img

    dim_1, dim_2 = stack.shape[1], stack.shape[2]
    stack_reshape = stack.reshape((v_steps, u_steps, dim_1, dim_2))

    return stack, stack_reshape, dim_1, dim_2
