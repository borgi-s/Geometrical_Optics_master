"""Direct-space DFXM forward model.

Builds the module-level geometry (lab/sample/crystal frames, detector grid,
beam profile) at import time, then exposes `forward()` which projects a
displacement-gradient field `Hg` through the imaging system into a 2D image.

The reciprocal-space resolution kernel `Resq_i` is loaded on demand by
`_load_default_kernel(pkl_path)` — called via `_lookup_and_load_kernel(hkl, keV)`
in `dfxm_geo.pipeline`. This lets the module be imported on a clean clone or
in CI without any precomputed kernel present.

Default geometry constants match ID06 at the ESRF; see `dfxm_geo.constants`.
"""

import os
from dataclasses import dataclass as _dataclass
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING

import numpy as np

from dfxm_geo.crystal.remount import SAMPLE_REMOUNT_OPTIONS
from dfxm_geo.crystal.rotations import fast_inverse2
from dfxm_geo.io.strain_cache import load_or_generate_Hg

if TYPE_CHECKING:
    from dfxm_geo.pipeline import CrystalConfig, ScanConfig

# Module-level default for the sample-remount rotation matrix.
# Defined here to avoid cross-module imports and satisfy ruff-B008.
_S_IDENTITY: np.ndarray = np.identity(3)

# Repo root: the directory containing pyproject.toml. Derived from this
# file's location (src/dfxm_geo/direct_space/forward_model.py → 4 levels up).
# Previously this was inferred from sys.path[0], which silently broke when
# the module was imported via an installed entry point or via `python -c`.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Sub-project C: random_dislocations placement constants.
_MAX_REJECTION_TRIES = 10_000

# A subset of FCC slip systems on the {111}/<-110> family used to draw
# random orientations for random_dislocations mode. Each entry is (b, n, t)
# with b.n=0 and t parallel to n x b. Six variants spanning two of the
# four {111} planes -- sufficient for v1.2.0 sampling.
_SLIP_SYSTEM_111: tuple[
    tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]], ...
] = (
    ((1, -1, 0), (1, 1, 1), (1, 1, -2)),
    ((-1, 0, 1), (1, 1, 1), (-1, 2, -1)),
    ((0, 1, -1), (1, 1, 1), (-2, 1, 1)),
    ((1, 1, 0), (1, -1, 1), (1, -1, -2)),
    ((-1, 0, 1), (1, -1, 1), (-1, -2, -1)),
    ((0, -1, -1), (1, -1, 1), (-2, -1, 1)),
)

fast_inverse2(
    np.random.default_rng().random(size=(100, 3, 3))
)  # DO NOT OUTCOMMENT, this line jit compiles "fast_inverse2" function so performance on larger arrays are obtained

# INPUT instrumental settings, related to direct space resolution function
psize = 40e-9  # pixel size in units of m, in the object plane
zl_rms = 0.15e-6 / 2.35  # rms value of Gaussian beam profile, in m, centered at 0
theta_0 = 17.953 / 2 * np.pi / 180  # in rad

# INPUT FOV
Npixels = 510  # nr of pixels on detector (same in both y and z) - sets the FOV.
Nsub = 1  # NN1^3 = (Nsub*Npixels)^3 is the total number of "rays" probed.
# Nsub = 1 is the typical real-run choice (~8x faster forward calls).
# Borgi 2024 (IUCrJ; doi:10.1107/S1600576724001183) used Nsub = 2 for
# publication-quality figures — flip to 2 to reproduce paper-grade output.
NN1 = int(Npixels // 3 * Nsub)  # 3 is used as 1/sin(2*~18 deg) = 3.24
NN2 = int(Npixels * Nsub)
NN3 = int(Npixels // 30 * Nsub)

# Default directory for reciprocal-space resolution kernel files.
# Kernel files are discovered via _lookup_kernel_path(hkl, keV, pkl_fpath)
# which globs Resq_i_h{h}_k{k}_l{l}_{keV}keV_*.npz and picks the newest.
# There is no longer a module-level `pkl_fn` constant — sub-project D removed it.
pkl_fpath = str(_REPO_ROOT / "reciprocal_space" / "pkl_files") + os.sep

# Sub-project D: set by `_load_default_kernel` on successful load; read by
# `io/hdf5.py` and `io/migrate.py` for provenance recording. `None` until a
# kernel is loaded.
_loaded_kernel_path: Path | None = None

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

# Kernel-dependent globals — populated by `_load_default_kernel()` if the
# default kernel exists; otherwise these stay `None` and `forward()` will
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
            "Resq_i": _loaded_kernel_path.name if _loaded_kernel_path else "<not loaded>",
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

        with open(Fg_path.replace(".npy", "_vars.txt"), "w", encoding="utf-8") as data:
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
    pkl_path: str | Path | None = None,
    *,
    expected_hkl: tuple[int, int, int] | None = None,
    expected_keV: float | None = None,
    compute_Hg: bool = True,
) -> None:
    """Load the reciprocal-space resolution kernel from disk into module state.

    Must be called with an explicit `pkl_path`. Use
    `_lookup_kernel_path(hkl, keV, pkl_fpath)` to find the matching file,
    or call via `pipeline._lookup_and_load_kernel(hkl, keV)` which composes
    the lookup and load together.

    Sets module-level globals: `Resq_i`, `qi1_range`/`qi2_range`/`qi3_range`,
    `npoints1`/`npoints2`/`npoints3`, `qi1_start`/`qi1_step`/etc.,
    `qi_starts`, `qi_steps`. If `compute_Hg=True`, also computes the default
    `Hg`/`q_hkl` by calling `Find_Hg(dis, ndis, psize, zl_rms)`.

    Args:
        pkl_path: Path to the .npz kernel file. Required — pass the result
            of `_lookup_kernel_path(hkl, keV, pkl_fpath)`.
        expected_hkl: If given, verify that the kernel's bundled ``hkl``
            metadata matches. Raises ``KeyError`` if the metadata is absent
            (pre-sub-project-D bootstrap), ``ValueError`` on mismatch.
        expected_keV: If given, verify that the kernel's bundled ``keV``
            metadata matches. Raises ``KeyError`` if the metadata is absent,
            ``ValueError`` on mismatch.
        compute_Hg: If True (default), also compute the default ``Hg``/
            ``q_hkl`` by calling ``Find_Hg(dis, ndis, psize, zl_rms)``.

    On success, sets ``_loaded_kernel_path`` to the resolved ``Path``.
    """
    global Resq_i, qi1_range, qi2_range, qi3_range, npoints1, npoints2, npoints3
    global qi1_start, qi2_start, qi3_start, qi1_step, qi2_step, qi3_step
    global qi_starts, qi_steps, Hg, q_hkl
    global _loaded_kernel_path

    if pkl_path is None:
        raise ValueError(
            "_load_default_kernel requires an explicit `pkl_path` after sub-project D. "
            "Use `_lookup_kernel_path(hkl, keV, pkl_fpath)` to find the matching kernel, "
            "or call via `pipeline._lookup_and_load_kernel(hkl, keV)`."
        )

    if str(pkl_path).endswith(".pkl"):
        raise RuntimeError(
            f"Detected legacy pickle at {pkl_path!r}; pickle support was "
            "removed in v1.0.3. Run `dfxm-bootstrap --config configs/default.toml` "
            "to regenerate the kernel as .npz."
        )

    print(f"Loading kernel from {pkl_path}.")
    data = np.load(pkl_path)
    try:
        # Sub-project D: verify bundled metadata against the lookup request.
        if expected_hkl is not None:
            if "hkl" not in data.files:
                raise KeyError(
                    f"kernel at {pkl_path} lacks `hkl` metadata — "
                    f"pre-sub-project-D bootstrap.\n"
                    f"Re-run: dfxm-bootstrap --config <yourconfig.toml>"
                )
            meta_hkl = tuple(int(x) for x in data["hkl"])
            if meta_hkl != tuple(expected_hkl):
                raise ValueError(
                    f"kernel at {pkl_path} has hkl={meta_hkl} but lookup requested "
                    f"hkl={tuple(expected_hkl)} — file may have been manually "
                    f"renamed or copied wrong."
                )
        if expected_keV is not None:
            if "keV" not in data.files:
                raise KeyError(
                    f"kernel at {pkl_path} lacks `keV` metadata — "
                    f"pre-sub-project-D bootstrap.\n"
                    f"Re-run: dfxm-bootstrap --config <yourconfig.toml>"
                )
            meta_keV = float(data["keV"])
            if meta_keV != expected_keV:
                raise ValueError(
                    f"kernel at {pkl_path} has keV={meta_keV} but lookup requested "
                    f"keV={expected_keV} — file may have been manually renamed or "
                    f"copied wrong."
                )

        Resq_i = np.array(data["Resq_i"])
        qi1_range = float(data["qi1_range"])
        qi2_range = float(data["qi2_range"])
        qi3_range = float(data["qi3_range"])
        npoints1 = int(data["npoints1"])
        npoints2 = int(data["npoints2"])
        npoints3 = int(data["npoints3"])
        print("Kernel loaded.")
    finally:
        data.close()

    qi1_start, qi1_step = -qi1_range / 2, qi1_range / (npoints1 - 1)
    qi2_start, qi2_step = -qi2_range / 2, qi2_range / (npoints2 - 1)
    qi3_start, qi3_step = -qi3_range / 2, qi3_range / (npoints3 - 1)
    qi_starts = np.asarray([qi1_start, qi2_start, qi3_start])
    qi_steps = 1 / np.asarray([qi1_step, qi2_step, qi3_step])

    if compute_Hg:
        Hg, q_hkl = Find_Hg(dis, ndis, psize, zl_rms)

    _loaded_kernel_path = Path(pkl_path)


def _lookup_kernel_path(
    hkl: tuple[int, int, int],
    keV: float,
    pkl_fpath: str | Path,
) -> Path:
    """Find the newest kernel npz on disk matching the requested (hkl, keV).

    Globs ``<pkl_fpath>/Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_*.npz``, sorts by
    mtime descending, returns the newest. Emits a stderr WARN listing all
    matches when more than one exists. Raises FileNotFoundError with a
    ``dfxm-bootstrap`` instruction on zero matches.

    Sub-project D: replaces the previous ``pkl_fn``-constant lookup.
    """
    import sys

    h, k, l = hkl
    pkl_fpath = Path(pkl_fpath)
    pattern = f"Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_*.npz"
    matches = sorted(
        pkl_fpath.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"no kernel found for hkl={hkl} at {keV} keV in {pkl_fpath}/.\n"
            f"Run: dfxm-bootstrap --config <yourconfig.toml>\n"
            f"(produces Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_<date>.npz, "
            f"~50 s wall-clock at default Nrays=1e8)"
        )
    if len(matches) > 1:
        lines = [
            f"warning: found {len(matches)} kernels matching hkl={hkl} keV={keV:g} in {pkl_fpath}:"
        ]
        for i, m in enumerate(matches):
            tag = "  (newest, will use)" if i == 0 else ""
            lines.append(f"  {m.name}{tag}")
        print("\n".join(lines), file=sys.stderr)
    return matches[0]


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
            "_load_default_kernel(pkl_path) before calling forward()."
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

    # Calculate scattering vector in crystal/grain space. After `qc` is built
    # `qs` and `ang_arr` are not used again; free them before allocating `qi`
    # so the (3, NN1*NN2*NN3) float64 intermediates don't all live at once.
    qc = qs.squeeze().T + ang_arr
    del qs, ang_arr

    # Calculate scattering vector in imaging space; `qc` is no longer needed.
    qi = Theta @ qc
    del qc

    # Calculate indices for Resq_i that pass through the mask
    index1 = np.floor((qi[0] - qi1_start) / qi1_step).astype(np.int16)
    index2 = np.floor((qi[1] - qi2_start) / qi2_step).astype(np.int16)
    index3 = np.floor((qi[2] - qi3_start) / qi3_step).astype(np.int16)
    # In the qi_return path we need `qi` again to build qi_field; in the
    # common (non-return) path it is unused after the index1/2/3 floors.
    if not qi_return:
        del qi

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
    del index1, index2, index3  # only `idx` and `prob` are needed below

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
    del idx, prob
    im_1 += contribution.reshape(im_1.shape)
    del contribution
    if qi_return:
        # qi was kept alive above; build qi_field only when actually returned.
        qi_field = qi.reshape(3, NN1, NN2, NN3)
        return im_1, qi_field
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


# Sub-project D: no module-import auto-load. Kernel is loaded on demand
# via pipeline._lookup_and_load_kernel(hkl, keV) or an explicit
# _load_default_kernel(pkl_path) call. This lets the module be imported on
# any host regardless of whether the kernel npz is on disk.


# ---------------------------------------------------------------------------
# Sub-project B: scan-grid helpers
# ---------------------------------------------------------------------------


@_dataclass
class ScanGrid:
    """Realized trajectory for a ScanConfig.

    `axes` is the canonical 4-tuple ("phi", "chi", "two_dtheta", "z").
    `samples` is parallel: per-axis 1-D arrays of position values
    (units: radians for angular axes, micrometers for z). Fixed axes
    have shape (1,); scanned axes have shape (steps,).

    The forward kernel iterates the Cartesian product over `samples`.
    """

    axes: tuple[str, str, str, str]
    samples: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def build_scan_grid(scan: "ScanConfig") -> ScanGrid:
    """Build a ScanGrid from a ScanConfig.

    For each canonical axis, returns either:
      - linspace(value-range, value+range, steps) if scanned
      - np.array([value]) if fixed (singleton)
    """
    from dfxm_geo.pipeline import _CANONICAL_AXES  # local import: avoid cycle

    samples = []
    for axis_name in _CANONICAL_AXES:
        axis = getattr(scan, axis_name)
        if axis.is_scanned:
            arr = np.linspace(
                axis.value - axis.range,
                axis.value + axis.range,
                axis.steps,
                dtype=np.float64,
            )
        else:
            arr = np.array([axis.value], dtype=np.float64)
        samples.append(arr)
    return ScanGrid(
        axes=("phi", "chi", "two_dtheta", "z"),
        samples=(samples[0], samples[1], samples[2], samples[3]),
    )


# ---------------------------------------------------------------------------
# Sub-project C: dislocation-population helpers
# ---------------------------------------------------------------------------


@_dataclass
class DislocationPopulation:
    """A realized set of dislocations.

    positions_um: shape (N, 3) — (x, y, z) sample-frame coordinates.
    Ud: shape (N, 3, 3) — column-stacked (b_hat, n_hat, t_hat) rotation matrices.
    sidecar: dict to be written as JSON, or None if no sidecar needed.
    """

    positions_um: np.ndarray
    Ud: np.ndarray
    sidecar: dict | None


def _ud_matrix_from_bnt(
    b: tuple[int, int, int],
    n: tuple[int, int, int],
    t: tuple[int, int, int],
) -> np.ndarray:
    """Build a 3x3 column-stacked rotation matrix [b_hat | n_hat | t_hat].

    Input vectors are crystallographic integer indices; output columns
    are unit-normalized.
    """
    arr = np.asarray([b, n, t], dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / norms).T  # columns = b_hat, n_hat, t_hat


def build_dislocation_population(
    crystal: "CrystalConfig",
    fov_lateral_um: float,
    rng: np.random.Generator | None,
) -> DislocationPopulation:
    """Dispatch on crystal.mode and realize the dislocation population.

    Centered: 1 dislocation at origin with explicit (b, n, t).
    Wall: existing dis-spaced grid; positions evenly spaced along y at z=0.
    Random_dislocations: implemented in Task 8.
    """
    if crystal.mode == "centered":
        c = crystal.centered
        assert c is not None  # __post_init__ guarantees
        positions = np.zeros((1, 3), dtype=np.float64)
        Ud = _ud_matrix_from_bnt(c.b, c.n, c.t)[np.newaxis, :, :]  # (1, 3, 3)
        return DislocationPopulation(positions_um=positions, Ud=Ud, sidecar=None)

    if crystal.mode == "wall":
        w = crystal.wall
        assert w is not None
        # Positions: ndis dislocations along a wall at z=0, x=0, y evenly
        # spaced by `dis` micrometers centered at 0.
        ys = (np.arange(w.ndis) - (w.ndis - 1) / 2.0) * w.dis
        positions = np.zeros((w.ndis, 3), dtype=np.float64)
        positions[:, 1] = ys
        # All dislocations in a wall share the same (b, n, t) — the canonical
        # {111}/<-110>/<11-2> slip system for the Borgi/Purdue layout.
        Ud_single = _ud_matrix_from_bnt((1, -1, 0), (1, 1, 1), (1, 1, -2))
        Ud = np.broadcast_to(Ud_single, (w.ndis, 3, 3)).copy()
        return DislocationPopulation(positions_um=positions, Ud=Ud, sidecar=None)

    if crystal.mode == "random_dislocations":
        rd = crystal.random_dislocations
        assert rd is not None

        # Resolve sigma: FOV-derived default if user didn't supply.
        if rd.sigma is None:
            sigma_um = (fov_lateral_um / 2.0) / 2.0
            sigma_source = "default-fov"
        else:
            sigma_um = rd.sigma
            sigma_source = "user"

        # Resolve seed + rng. If caller passed an rng, use it; otherwise fall
        # back to rd.seed or fresh entropy.
        if rng is None:
            if rd.seed is None:
                _entropy = np.random.SeedSequence().entropy
                resolved_seed = (
                    int(_entropy)
                    if isinstance(_entropy, int)
                    else int(np.random.SeedSequence().generate_state(1)[0])
                )
                seed_source = "entropy"
            else:
                resolved_seed = rd.seed
                seed_source = "user"
            rng = np.random.default_rng(resolved_seed)
        else:
            resolved_seed = rd.seed if rd.seed is not None else -1
            seed_source = "user" if rd.seed is not None else "entropy"

        # Draw positions with optional min_distance rejection sampling.
        positions = np.zeros((rd.ndis, 3), dtype=np.float64)
        for i in range(rd.ndis):
            for _ in range(_MAX_REJECTION_TRIES):
                cand = rng.normal(loc=0.0, scale=sigma_um, size=2)
                if rd.min_distance is None or i == 0:
                    positions[i, 0] = cand[0]
                    positions[i, 1] = cand[1]
                    break
                diffs = positions[:i, :2] - cand
                dists = np.linalg.norm(diffs, axis=1)
                if np.all(dists >= rd.min_distance):
                    positions[i, 0] = cand[0]
                    positions[i, 1] = cand[1]
                    break
            else:
                raise RuntimeError(
                    f"random_dislocations placement exceeded retry budget "
                    f"({_MAX_REJECTION_TRIES}) at dislocation {i}/{rd.ndis}; "
                    f"check min_distance={rd.min_distance}, sigma={sigma_um}, "
                    f"ndis={rd.ndis} - configuration may be impossible."
                )

        # Draw slip-system per dislocation (uniform over {111} family).
        slip_indices = rng.integers(0, len(_SLIP_SYSTEM_111), size=rd.ndis)
        Ud = np.zeros((rd.ndis, 3, 3), dtype=np.float64)
        sidecar_dislocations: list[dict] = []
        for i in range(rd.ndis):
            b, n, t = _SLIP_SYSTEM_111[slip_indices[i]]
            Ud[i] = _ud_matrix_from_bnt(b, n, t)
            sidecar_dislocations.append(
                {
                    "index": i,
                    "x_um": float(positions[i, 0]),
                    "y_um": float(positions[i, 1]),
                    "z_um": float(positions[i, 2]),
                    "b": list(b),
                    "n": list(n),
                    "t": list(t),
                }
            )

        sidecar = {
            "ndis": rd.ndis,
            "sigma_um": float(sigma_um),
            "sigma_source": sigma_source,
            "min_distance_um": rd.min_distance,
            "seed": resolved_seed,
            "seed_source": seed_source,
            "dislocations": sidecar_dislocations,
        }
        return DislocationPopulation(positions_um=positions, Ud=Ud, sidecar=sidecar)

    raise AssertionError(f"unreachable crystal.mode={crystal.mode!r}")  # pragma: no cover
