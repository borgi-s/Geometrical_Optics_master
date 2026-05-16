"""Direct-space DFXM forward model.

Builds the module-level geometry (lab/sample/crystal frames, detector grid,
beam profile) at import time, then exposes `forward()` which projects a
displacement-gradient field `Hg` through the imaging system into a 2D image.

The reciprocal-space resolution kernel `Resq_i` is loaded lazily by
`_load_default_kernel()` — at module import iff the default pickle is on
disk, otherwise on explicit call. This lets the module be imported on a
clean clone or in CI without the precomputed pickle present.

Default geometry constants match ID06 at the ESRF; see `dfxm_geo.constants`.
"""

import os
import pickle
from pathlib import Path
from pprint import pprint

import numpy as np

from dfxm_geo.crystal.remount import SAMPLE_REMOUNT_OPTIONS
from dfxm_geo.crystal.rotations import fast_inverse2
from dfxm_geo.io.strain_cache import load_or_generate_Hg

# Module-level default for the sample-remount rotation matrix.
# Defined here to avoid cross-module imports and satisfy ruff-B008.
_S_IDENTITY: np.ndarray = np.identity(3)

# Repo root: the directory containing pyproject.toml. Derived from this
# file's location (src/dfxm_geo/direct_space/forward_model.py → 4 levels up).
# Previously this was inferred from sys.path[0], which silently broke when
# the module was imported via an installed entry point or via `python -c`.
_REPO_ROOT = Path(__file__).resolve().parents[3]

fast_inverse2(
    np.random.default_rng().random(size=(100, 3, 3))
)  # DO NOT OUTCOMMENT, this line jit compiles "fast_inverse2" function so performance on larger arrays are obtained

# INPUT instrumental settings, related to direct space resolution function
psize = 40e-9  # pixel size in units of m, in the object plane
zl_rms = 0.15e-6 / 2.35  # rms value of Gaussian beam profile, in m, centered at 0
theta_0 = 17.953 / 2 * np.pi / 180  # in rad

# INPUT FOV
Npixels = 510  # nr of pixels on detector (same in both y and z) - sets the FOV.
Nsub = 2  # NN1^3 = (Nsub*Npixels)^3 is the total number of "rays" probed
NN1 = int(Npixels // 3 * Nsub)  # 3 is used as 1/sin(2*~18 deg) = 3.24
NN2 = int(Npixels * Nsub)
NN3 = int(Npixels // 30 * Nsub)

# Default reciprocal-space resolution kernel paths.
# These are loaded lazily by `_load_default_kernel()` only if the file exists,
# so the module can be imported on a clean checkout that lacks the precomputed
# pickle (e.g. CI, tests, fresh clones).
pkl_fpath = str(_REPO_ROOT / "reciprocal_space" / "pkl_files") + os.sep
pkl_fn = "Resq_i_20230913_1308.pkl"  # Change accordingly
vars_fn = os.path.splitext(pkl_fn)[0] + "_vars.txt"

theta = theta_0
yl_start = -psize * Npixels / 2 + psize / (
    2 * Nsub
)  # start in yl direction, in unit of m, centered at 0
xl_start = yl_start / np.tan(2 * theta) / 3  # start in xl direction, in m, for zl=0
zl_start = -0.5 * zl_rms * 6  # start in zl direction, in m, for zl=0

# CREATE MATRICES according to Eqs 2,3,7:
Ud = np.array(
    [
        [1 / np.sqrt(2), 1 / np.sqrt(3), 1 / np.sqrt(6)],
        [-1 / np.sqrt(2), 1 / np.sqrt(3), 1 / np.sqrt(6)],
        [0, -1 / np.sqrt(3), 2 / np.sqrt(6)],
    ]
)
Us = np.array(
    [
        [1 / np.sqrt(2), -1 / np.sqrt(6), -1 / np.sqrt(3)],
        [0, -2 / np.sqrt(6), 1 / np.sqrt(3)],
        [-1 / np.sqrt(2), -1 / np.sqrt(6), -1 / np.sqrt(3)],
    ]
).T
Theta = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

YI = (np.arange(NN1) // Nsub).repeat(NN3 * NN2)
ZI = np.tile((np.arange(NN2) // Nsub).repeat(NN3), NN1)
indices = np.vstack((ZI, YI)).T

# Precomputed C-order flat index into the (NN2 // Nsub, NN1 // Nsub) output
# image, used by `forward()`'s scatter accumulator. Built once at import; the
# (ZI, YI) row/col mapping is fixed by the detector geometry.
_flat_indices = indices[:, 0].astype(np.int64) * (NN1 // Nsub) + indices[:, 1].astype(np.int64)

xl_range, xl_steps = -xl_start, NN1
yl_range, yl_steps = -yl_start, NN2
zl_range, zl_steps = -zl_start, NN3
rl = np.vstack(  # type: ignore[call-overload]
    np.mgrid[
        -xl_range : xl_range : complex(xl_steps),
        -yl_range : yl_range : complex(yl_steps),
        -zl_range : zl_range : complex(zl_steps),
    ]
).reshape(3, -1)

prob_z = np.exp(-0.5 * (rl[2] / zl_rms) ** 2)

# To avoid edge effects:
# for dis 0.25, ndis >= 7501
# for dis 0.5, ndis >= 1151
# for dis 1, ndis >= 501
# for dis 2, ndis >= 251
# for dis 4, ndis >= 151

ndis = 151  # number of dislocations
dis = 4  # units of micrometer

# Pickle-dependent globals — populated by `_load_default_kernel()` if the
# default pickle exists; otherwise these stay `None` and `forward()` will
# raise a clear error at call time.
Resq_i = None
qi1_range = qi2_range = qi3_range = None
npoints1 = npoints2 = npoints3 = None
qi1_start = qi1_step = None
qi2_start = qi2_step = None
qi3_start = qi3_step = None
qi_starts = qi_steps = None
Hg = q_hkl = None


def Find_Hg(
    dis: float,
    ndis: int,
    psize: float,
    zl_rms: float,
    h: int = -1,
    k: int = 1,
    l: int = -1,
    *,
    S: np.ndarray = _S_IDENTITY,
    remount_name: str = "S1",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the displacement gradient field Hg and reciprocal vector q_hkl.

    Wraps `dfxm_geo.io.strain_cache.load_or_generate_Hg`, caching Fg to disk
    at `direct_space/deformation_gradient_tensors/`. Also writes a sidecar
    `_vars.txt` describing the parameters used.

    Args:
        dis: Distance between dislocations (µm).
        ndis: Number of dislocations.
        psize: Detector pixel size (m).
        zl_rms: RMS of the beam profile in zl (m).
        h, k, l: Miller indices of the active reflection.
        S: 3x3 sample-remount rotation (default identity).
        remount_name: Name used in the Fg cache filename (default "S1").

    Returns:
        (Hg, q_hkl) where Hg has shape (X, 3, 3) and q_hkl has shape (3,).

    Raises:
        ValueError: If `remount_name` is not one of the named remount options
            in `SAMPLE_REMOUNT_OPTIONS`. Guards against silently writing a
            cache filename like `..._remountbogus.npy` when a non-pipeline
            caller passes a typo or free-form string.
    """
    if remount_name not in SAMPLE_REMOUNT_OPTIONS:
        raise ValueError(
            f"Unknown remount_name {remount_name!r}; expected one of "
            f"{sorted(SAMPLE_REMOUNT_OPTIONS)}"
        )

    Q_norm = np.sqrt(h * h + k * k + l * l)  # We have assumed B_0 = I
    q_hkl = np.asarray([h, k, l]) / Q_norm

    # The cache key must include every parameter that affects the rl ray grid
    # shape: dis/psize/zl_rms determine the physics, Npixels/Nsub determine the
    # detector ray-grid dimensions. Without Npixels/Nsub, changing those
    # module-level constants silently loaded a wrong-shape cache (see
    # `load_or_generate_Hg`'s shape guard).
    Fg_dir = _REPO_ROOT / "direct_space" / "deformation_gradient_tensors"
    Fg_dir.mkdir(parents=True, exist_ok=True)
    Fg_path = str(
        Fg_dir
        / "Fg_{}_{}nm_{}nm_px{}_sub{}_remount{}.npy".format(
            str(dis).replace(".", ""),
            int(psize * 1e9),
            int(zl_rms * 2.35e9),
            Npixels,
            Nsub,
            remount_name,
        )
    )
    Hg = load_or_generate_Hg(rl, Ud, Us, Theta, dis, ndis, Fg_path, S=S)

    if not os.path.exists(Fg_path.replace(".npy", "_vars.txt")):
        vars = {
            "Resq_i": pkl_fn,
            "psize [nm]": psize,
            "zl_rms": zl_rms,
            "theta_0 [rad]": theta_0,
            "Npixels": Npixels,
            "Nsub": Nsub,
            "Ud": Ud.tolist(),
            "Us": Us.tolist(),
            "Theta": Theta.tolist(),
            "ndis": ndis,
            "dis [micrometer]": dis,
            "q_hkl": q_hkl.tolist(),
            "S_remount_name": remount_name,
            "S_remount_matrix": S.tolist(),
        }

        with open(Fg_path.replace(".npy", "_vars.txt"), "w") as data:
            for key, value in vars.items():
                # Use pprint for matrices to format them nicely
                if isinstance(value, list) and isinstance(value[0], list):
                    data.write(f"{key}:\n")
                    pprint(value, data)
                    data.write("\n")
                else:
                    data.write(f"{key}: {value}\n\n")
    return Hg, q_hkl


def _load_default_kernel(
    pkl_path: str | None = None,
    vars_path: str | None = None,
    compute_Hg: bool = True,
) -> None:
    """Load the reciprocal-space resolution kernel from disk into module state.

    Called at import time iff the default pickle exists. Can be called
    manually with an explicit path to bootstrap the module after a clean
    import (e.g. from tests with a fixture kernel).

    Sets module-level globals: `Resq_i`, `qi1_range`/`qi2_range`/`qi3_range`,
    `npoints1`/`npoints2`/`npoints3`, `qi1_start`/`qi1_step`/etc.,
    `qi_starts`, `qi_steps`. If `compute_Hg=True`, also computes the default
    `Hg`/`q_hkl` by calling `Find_Hg(dis, ndis, psize, zl_rms)`.
    """
    global Resq_i, qi1_range, qi2_range, qi3_range, npoints1, npoints2, npoints3
    global qi1_start, qi2_start, qi3_start, qi1_step, qi2_step, qi3_step
    global qi_starts, qi_steps, Hg, q_hkl

    if pkl_path is None:
        pkl_path = os.path.join(pkl_fpath, pkl_fn)
    if vars_path is None:
        vars_path = os.path.join(pkl_fpath, vars_fn)

    print("Loading Resq_i.")
    with open(pkl_path, "rb") as f:
        Resq_i = pickle.load(f)
    with open(vars_path) as f:
        var_d = eval(f.read())
    qi1_range, npoints1 = var_d["qi1_range"], var_d["npoints1"]
    qi2_range, npoints2 = var_d["qi2_range"], var_d["npoints2"]
    qi3_range, npoints3 = var_d["qi3_range"], var_d["npoints3"]
    print("Resq_i loaded.")

    qi1_start, qi1_step = -qi1_range / 2, qi1_range / (npoints1 - 1)
    qi2_start, qi2_step = -qi2_range / 2, qi2_range / (npoints2 - 1)
    qi3_start, qi3_step = -qi3_range / 2, qi3_range / (npoints3 - 1)
    qi_starts = np.asarray([qi1_start, qi2_start, qi3_start])
    qi_steps = 1 / np.asarray([qi1_step, qi2_step, qi3_step])

    if compute_Hg:
        Hg, q_hkl = Find_Hg(dis, ndis, psize, zl_rms)


def forward(
    Hg: np.ndarray,
    phi: float = 0,
    chi: float = 0,
    TwoDeltaTheta: float = 0,
    qi_return: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the DFXM forward-model image for the given goniometer angles.

    Reads module-level state populated by `_load_default_kernel()` (Resq_i,
    qi*_start/step, etc.). Raises `RuntimeError` if state is not initialized.

    Args:
        Hg: Displacement gradient field, shape (X, 3, 3) where X = NN1*NN2*NN3.
        phi: Radians off the Bragg condition (rotation around y_l axis).
        chi: Radians off the Bragg condition (rotation around x_l axis).
        TwoDeltaTheta: Radians off the Bragg angle (2θ shift).
        qi_return: If True, also return the scattering vector field qi.

    Returns:
        Forward-model image of shape (NN2//Nsub, NN1//Nsub). If qi_return is
        True, returns (image, qi_field) where qi_field has shape (3, NN1, NN2, NN3).
    """
    if Resq_i is None:
        raise RuntimeError(
            "forward_model state is not initialized. Call "
            "_load_default_kernel(pkl_path, vars_path) before calling forward()."
        )

    if TwoDeltaTheta != 0:
        theta = theta_0 + TwoDeltaTheta
        Theta = np.array(
            [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
        )
    else:
        Theta = np.array(
            [
                [np.cos(theta_0), 0, np.sin(theta_0)],
                [0, 1, 0],
                [-np.sin(theta_0), 0, np.cos(theta_0)],
            ]
        )

    # Initialize forward model image with zeros
    im_1 = np.zeros([(NN2 // Nsub), NN1 // Nsub])

    # Define angles
    ang_arr = np.asarray(
        [[phi - TwoDeltaTheta / 2], [chi], [(TwoDeltaTheta / 2) / np.tan(theta_0)]]
    )

    # Calculate scattering vector in sample space
    qs = Us @ Hg @ q_hkl

    # Calculate scattering vector in crystal/grain space
    qc = qs.squeeze().T + ang_arr

    # Calculate scattering vector in imaging space and reshape it
    qi = Theta @ qc
    qi_field = qi.reshape(3, NN1, NN2, NN3)

    # Calculate indices for Resq_i that pass through the mask
    index1 = np.floor((qi[0] - qi1_start) / qi1_step).astype(np.int16)
    index2 = np.floor((qi[1] - qi2_start) / qi2_step).astype(np.int16)
    index3 = np.floor((qi[2] - qi3_start) / qi3_step).astype(np.int16)

    # Calculate index of values within bounds and extract corresponding values from Resq_i
    idx = (
        (index3 >= 0)
        * (index2 >= 0)
        * (index1 >= 0)
        * (index1 < npoints1)
        * (index2 < npoints2)
        * (index3 < npoints3)
    )
    prob = Resq_i[index1[idx], index2[idx], index3[idx]] * prob_z[idx]

    # Scatter-accumulate the probability into the (chi, phi)-rocking image.
    # np.bincount on the precomputed flat indices is ~10x faster than the
    # previous np.add.at(im_1, tuple(indices.T), pro) at production sizes
    # (the np.add.at scatter-iterator is famously slow). Restricting to the
    # idx-mask subset is exact: invalid positions contribute zero, which
    # adds nothing under either algorithm.
    # The pre-cleanup code went through a float32 `pro` array; we preserve
    # that float32 quantization on the way into bincount so output bytes
    # match the np.add.at era bit-for-bit. Using float64 prob directly here
    # would be more accurate (~1e-8 rel) but would not be a bit-equivalent
    # change.
    contribution = np.bincount(
        _flat_indices[idx],
        weights=prob.astype(np.float32),
        minlength=im_1.size,
    )
    im_1 += contribution.reshape(im_1.shape)
    if qi_return:
        qi_field = qi.reshape(3, NN1, NN2, NN3)
        # Return scattering vector in imaging space and forward model image
        return im_1, qi_field
    # Return the forward model image
    return im_1


def Z_shift(offset_um: float) -> np.ndarray:
    """Return an `rl` grid shifted along the z axis by `offset_um` µm.

    Uses the module's existing detector ray-grid parameters (xl_range,
    xl_steps, yl_range, yl_steps, zl_range, zl_steps) to build the same
    mgrid as `rl` does at import time, but with the z range translated by
    ``-offset_um * 1e-6`` m. The module-level ``rl`` is not modified.

    Used by the z-scan pipeline mode to scan dislocations through the
    sample depth without rebuilding the detector ray grid for each layer.

    Args:
        offset_um: z offset in micrometres. Positive values move the
            dislocation core "up" in the lab z direction (equivalent to
            shifting `rl` "down" by the same amount).

    Returns:
        (3, X) coordinates in metres, same shape as `rl`.
    """
    offset_m = offset_um * 1e-6
    return np.vstack(  # type: ignore[call-overload]
        np.mgrid[
            -xl_range : xl_range : complex(xl_steps),
            -yl_range : yl_range : complex(yl_steps),
            -zl_range - offset_m : zl_range - offset_m : complex(zl_steps),
        ]
    ).reshape(3, -1)


# Auto-load the default kernel iff it exists on disk. Preserves the
# pre-cleanup behavior for callers (e.g. init_forward.py) that expect
# `Resq_i`, `Hg`, `q_hkl`, etc. to be ready at import time.
if os.path.exists(os.path.join(pkl_fpath, pkl_fn)):
    _load_default_kernel()
else:
    print(
        f"NOTE: default kernel pickle not found at {os.path.join(pkl_fpath, pkl_fn)!r}; "
        f"call _load_default_kernel(pkl_path, vars_path) before forward()."
    )
