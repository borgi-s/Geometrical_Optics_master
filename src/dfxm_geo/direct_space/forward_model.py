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

import contextlib
import os
from collections.abc import Iterator
from dataclasses import dataclass as _dataclass
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from dfxm_geo.crystal.remount import SAMPLE_REMOUNT_OPTIONS
from dfxm_geo.crystal.rotations import fast_inverse2
from dfxm_geo.io.strain_cache import load_or_generate_Hg

if TYPE_CHECKING:
    from dfxm_geo.pipeline import CrystalConfig, ReciprocalConfig, ScanConfig
    from dfxm_geo.reciprocal_space.analytic_resolution import AnalyticResolution

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

# The full 12 FCC slip systems on the {111}/<110> family used to draw random
# orientations for random_dislocations mode. Each entry is (b, n, t) with
# b.n=0 (Burgers in the glide plane) and t parallel to n x b (line direction).
# All four distinct {111} plane normals are covered, three <110> Burgers
# vectors per plane (4 planes x 3 = 12 systems). The first six entries are the
# original v1.2.0 sampling subset (planes (1,1,1) and (1,-1,1)), retained
# verbatim; the last six complete planes (-1,1,1) and (1,1,-1).
_SLIP_SYSTEM_111: tuple[
    tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]], ...
] = (
    # Plane (1, 1, 1)
    ((1, -1, 0), (1, 1, 1), (1, 1, -2)),
    ((-1, 0, 1), (1, 1, 1), (-1, 2, -1)),
    ((0, 1, -1), (1, 1, 1), (-2, 1, 1)),
    # Plane (1, -1, 1)
    ((1, 1, 0), (1, -1, 1), (1, -1, -2)),
    ((-1, 0, 1), (1, -1, 1), (-1, -2, -1)),
    ((0, -1, -1), (1, -1, 1), (-2, -1, 1)),
    # Plane (-1, 1, 1)
    ((0, 1, -1), (-1, 1, 1), (-2, -1, -1)),
    ((1, 0, 1), (-1, 1, 1), (1, 2, -1)),
    ((1, 1, 0), (-1, 1, 1), (-1, 1, -2)),
    # Plane (1, 1, -1)
    ((0, 1, 1), (1, 1, -1), (2, -1, 1)),
    ((1, -1, 0), (1, 1, -1), (-1, -1, -2)),
    ((1, 0, 1), (1, 1, -1), (1, -2, -1)),
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
# Kernel files are discovered via _lookup_kernel_path(directory=..., mode=..., hkl=..., keV=...)
# which globs the appropriate pattern and picks the newest match.
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


@contextlib.contextmanager
def reflection_theta_if_oblique(mode: str, theta_new: float | None) -> Iterator[None]:
    """Temporarily set the forward Bragg angle for an oblique run, then restore.

    ``theta_0`` (and the geometry it drives: ``Theta``, ``xl_start``, the ray
    grid ``rl``, ``prob_z``) defaults to Al (1,1,1) @ 17 keV (8.98 deg). Oblique
    reflections diffract at their own theta, so for ``mode == "oblique"`` this
    rebuilds the theta-dependent globals at ``theta_new`` (radians) for the
    duration of the ``with`` block and restores the defaults on exit (so
    simplified-mode runs and other callers/tests are never polluted). For any
    other mode, or ``theta_new is None``, it is a no-op.

    Must wrap the population build + forward: the ray grid ``rl`` (which the
    strain field is sampled on) and the imaging rotation ``Theta`` both depend
    on theta.
    """
    global theta_0, theta, Theta, xl_start, xl_range, rl, prob_z
    if mode != "oblique" or theta_new is None:
        yield
        return
    saved = (theta_0, theta, Theta, xl_start, xl_range, rl, prob_z)
    try:
        theta_0 = float(theta_new)
        theta = theta_0
        Theta = np.array(
            [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
        )
        xl_start = yl_start / np.tan(2 * theta) / 3
        xl_range = -xl_start
        rl = np.vstack(  # type: ignore[call-overload]
            np.mgrid[
                -xl_range : xl_range : complex(xl_steps),
                -yl_range : yl_range : complex(yl_steps),
                -zl_range : zl_range : complex(zl_steps),
            ]
        ).reshape(3, -1)
        prob_z = np.exp(-0.5 * (rl[2] / zl_rms) ** 2)
        yield
    finally:
        theta_0, theta, Theta, xl_start, xl_range, rl, prob_z = saved


_GUARDED_GLOBALS = (
    "theta_0",
    "theta",
    "Theta",
    "xl_start",
    "xl_range",
    "rl",
    "prob_z",
    "Resq_i",
    "qi1_start",
    "qi1_step",
    "qi2_start",
    "qi2_step",
    "qi3_start",
    "qi3_step",
    "npoints1",
    "npoints2",
    "npoints3",
    "qi_starts",
    "qi_steps",
    "_analytic_eval",
    "Hg",
    "q_hkl",
    "_loaded_kernel_path",
)


@contextlib.contextmanager
def _forward_state_guard() -> Iterator[None]:
    """Snapshot every mutable forward-model global on entry; restore on exit.

    Makes cross-config leakage on a persistent worker impossible: whatever a
    run mutates (kernel load, oblique theta rebuild, strain field) is rolled
    back when the run completes or raises. Subsumes
    ``reflection_theta_if_oblique``'s restore responsibility.

    Usable as a context manager OR a decorator: calling ``_forward_state_guard()``
    returns a ``_GeneratorContextManager`` (which is a ``ContextDecorator``), so
    ``@_forward_state_guard()`` re-snapshots on each decorated call (via
    ``_recreate_cm``). ``globals()`` here is forward_model's module dict
    regardless of where the decorator is applied.
    """
    g = globals()
    saved = {n: g.get(n) for n in _GUARDED_GLOBALS}
    try:
        yield
    finally:
        g.update(saved)


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

# Analytic resolution backend (Task 5). When set, forward() evaluates this
# closed-form p_Q(qi) instead of the Resq_i lookup. None => MC LUT path.
_analytic_eval: "AnalyticResolution | None" = None


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
    z_offset_um: float = 0.0,
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
        z_offset_um: z offset in micrometres for z-scan mode (default 0.0).
            When non-zero, a shifted reciprocal-lattice grid is computed via
            ``Z_shift(z_offset_um)`` and the cache filename gains a
            ``_z{round(z_offset_um*1000)}nm`` suffix so each z layer has its
            own Fg cache. When zero, behaviour is identical to v1.2.0.

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

    # z-aware cache filename. When z_offset_um == 0.0, identical to v1.2.0
    # filename — non-z scans hit the same cache file as before.
    z_suffix = "" if z_offset_um == 0.0 else f"_z{round(z_offset_um * 1000)}nm"
    Fg_path = str(
        Fg_dir
        / "Fg_{}_{}nm_{}nm_px{}_sub{}_remount{}{}.npy".format(
            str(dis).replace(".", ""),
            int(psize * 1e9),
            int(zl_rms * 2.35e9),
            Npixels,
            Nsub,
            remount_name,
            z_suffix,
        )
    )

    # Pick the rl grid: shifted if z_offset_um != 0, else the module-level rl.
    rl_eff = Z_shift(z_offset_um) if z_offset_um != 0.0 else rl
    Hg = load_or_generate_Hg(rl_eff, Ud, Us, Theta, dis, ndis, Fg_path, S=S)

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
    `_lookup_kernel_path(directory=..., mode=..., hkl=..., keV=...)` to find
    the matching file, or call via `pipeline._lookup_and_load_kernel(hkl, keV)`
    which composes the lookup and load together.

    Sets module-level globals: `Resq_i`, `qi1_range`/`qi2_range`/`qi3_range`,
    `npoints1`/`npoints2`/`npoints3`, `qi1_start`/`qi1_step`/etc.,
    `qi_starts`, `qi_steps`. If `compute_Hg=True`, also computes the default
    `Hg`/`q_hkl` by calling `Find_Hg(dis, ndis, psize, zl_rms)`.

    Args:
        pkl_path: Path to the .npz kernel file. Required — pass the result
            of `_lookup_kernel_path(directory=..., mode=..., hkl=..., keV=...)`.
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
            "Use `_lookup_kernel_path(directory=..., mode=..., hkl=..., keV=...)` to find "
            "the matching kernel, or call via `pipeline._lookup_and_load_kernel(hkl, keV)`."
        )

    if str(pkl_path).endswith(".pkl"):
        raise RuntimeError(
            f"Detected legacy pickle at {pkl_path!r}; pickle support was "
            "removed in v1.0.3. If you don't have a configs/ directory yet, run "
            "`dfxm-init` first, then `dfxm-bootstrap --config configs/default.toml` "
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

    # Bin width must match the FILL convention in reciprocal_space/resolution.py
    # (index = floor((q + range/2) / range * npoints) lays down `npoints` equal
    # bins of width range/npoints). Reading with range/(npoints-1) — the linspace
    # "fencepost" spacing — drifts the gather index low toward the grid edge
    # (~0.25-0.5% at npoints 400/200). Use range/npoints so READ inverts FILL.
    qi1_start, qi1_step = -qi1_range / 2, qi1_range / npoints1
    qi2_start, qi2_step = -qi2_range / 2, qi2_range / npoints2
    qi3_start, qi3_step = -qi3_range / 2, qi3_range / npoints3
    qi_starts = np.asarray([qi1_start, qi2_start, qi3_start])
    qi_steps = 1 / np.asarray([qi1_step, qi2_step, qi3_step])

    if compute_Hg:
        Hg, q_hkl = Find_Hg(dis, ndis, psize, zl_rms)

    _loaded_kernel_path = Path(pkl_path)


def _lookup_legacy_simplified(
    directory: Path,
    hkl: tuple[int, int, int],
    keV: float,
) -> Path:
    """Find the newest simplified-mode kernel npz on disk matching (hkl, keV).

    Globs ``<directory>/Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_*.npz``, sorts by
    mtime descending, returns the newest. Emits a stderr WARN listing all
    matches when more than one exists. Raises KeyError with a
    ``dfxm-bootstrap`` instruction on zero matches.

    Sub-project D: replaces the previous ``pkl_fn``-constant lookup.
    Internal helper — call via ``_lookup_kernel_path``.
    """
    import sys

    h, k, l = hkl
    pattern = f"Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_*.npz"
    matches = sorted(
        directory.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise KeyError(
            f"no kernel found for hkl={hkl} at {keV} keV in {directory}/.\n"
            f"Run: dfxm-bootstrap --config <yourconfig.toml>\n"
            f"(produces Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_<date>.npz, "
            f"~50 s wall-clock at default Nrays=1e8)"
        )
    if len(matches) > 1:
        lines = [
            f"warning: found {len(matches)} kernels matching hkl={hkl} keV={keV:g} in {directory}:"
        ]
        for i, m in enumerate(matches):
            tag = "  (newest, will use)" if i == 0 else ""
            lines.append(f"  {m.name}{tag}")
        print("\n".join(lines), file=sys.stderr)
    return matches[0]


def _lookup_kernel_path(
    *,
    directory: Path | str,
    mode: str = "simplified",
    hkl: tuple[int, int, int] | None = None,
    keV: float,
    theta: float | None = None,
    eta: float | None = None,
    tol: float = 1e-6,
) -> Path:
    """Resolve a LUT npz on disk.

    simplified: glob the legacy pattern, verify (hkl, keV) in metadata.
    oblique: glob the new pattern, verify (theta, eta, keV) in metadata
        within ``tol``.
    Raises KeyError with a dfxm-bootstrap hint when no match exists.

    Args:
        directory: Directory to search for kernel npz files.
        mode: ``"simplified"`` (default) or ``"oblique"``.
        hkl: Required for simplified mode.
        keV: X-ray energy in keV. Required for both modes.
        theta: Bragg angle in radians. Required for oblique mode.
        eta: Azimuthal tilt angle in radians. Required for oblique mode.
        tol: Absolute tolerance for float comparisons (theta, eta).
    """
    directory = Path(directory)

    if mode == "simplified":
        if hkl is None:
            raise ValueError("simplified mode lookup requires hkl.")
        return _lookup_legacy_simplified(directory, hkl, keV)

    if mode == "oblique":
        if theta is None or eta is None:
            raise ValueError("oblique mode lookup requires theta and eta.")
        glob_pattern = f"Resq_i_theta*_eta*_{keV:g}keV_*.npz"
        candidates = sorted(
            directory.glob(glob_pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            data = np.load(path)
            try:
                if (
                    abs(float(data["theta"]) - theta) <= tol
                    and abs(float(data["eta"]) - eta) <= tol
                    and abs(float(data["keV"]) - keV) <= 1e-9
                ):
                    return path
            except KeyError:
                continue  # incomplete metadata; skip
        raise KeyError(
            f"No bootstrapped kernel matching "
            f"(mode=oblique, theta={theta:.4f}, eta={eta:.4f}, keV={keV:g}) "
            f"in {directory}. Run: dfxm-bootstrap --config <your-config>"
        )

    raise ValueError(f"unknown geometry mode: {mode!r}")


def _load_analytic_resolution(config: "ReciprocalConfig") -> None:
    """Build the closed-form resolution evaluator and register it for forward().

    Derives theta from (hkl, keV) the same way the MC bootstrap does, then
    builds an AnalyticResolution from the config's instrument params. Also
    computes the default Hg/q_hkl if not already present (parity with
    _load_default_kernel(compute_Hg=True)).
    """
    global _analytic_eval, Hg, q_hkl
    from dfxm_geo.reciprocal_space.analytic_resolution import AnalyticResolution
    from dfxm_geo.reciprocal_space.kernel import _validate_reflection

    # Derive theta from the config's cubic lattice parameter `a`. Falls back to
    # the legacy Al value (4.0495e-10 m) for v2.2.0-era configs that don't carry
    # one — same getattr-default pattern as `eta` below. Mirrors how
    # reciprocal_space/kernel.py threads the mount's lattice into the MC path.
    lattice_a = float(getattr(config, "lattice_a", 4.0495e-10))
    theta = _validate_reflection(config.hkl, config.keV, lattice_a)
    eta_val = float(getattr(config, "eta", 0.0))  # safe default for v2.2.0-era configs
    _analytic_eval = AnalyticResolution(
        theta=theta,
        eta=eta_val,
        zeta_v_fwhm=config.zeta_v_fwhm,
        zeta_h_fwhm=config.zeta_h_fwhm,
        NA_rms=config.NA_rms,
        eps_rms=config.eps_rms,
        zeta_v_clip=config.zeta_v_clip,
    )
    if Hg is None:
        Hg, q_hkl = Find_Hg(dis, ndis, psize, zl_rms)


def precompute_forward_static(Hg: np.ndarray) -> np.ndarray:
    """Compute the phi/chi/2theta-independent part of the forward model.

    Returns ``base_qc = (Us @ Hg @ q_hkl).squeeze().T`` (shape ``(3, N)``),
    the single most expensive step in the forward model (~73% of a call) and
    the *only* part that depends solely on ``Hg``. Compute this once per scan
    and pass it to ``forward_from_static`` for every frame. The result is
    read-only and safe to share across worker threads (the dynamic half never
    mutates it).
    """
    qs = Us @ Hg @ q_hkl
    return qs.squeeze().T


@njit(cache=True, nogil=True, fastmath=False)
def _mc_lut_forward(
    base_qc: np.ndarray,
    ang0: float,
    ang1: float,
    ang2: float,
    Theta: np.ndarray,
    prob_z: np.ndarray,
    flat_indices: np.ndarray,
    Resq_flat: np.ndarray,
    im_flat: np.ndarray,
    qi1_start: float,
    qi1_step: float,
    qi2_start: float,
    qi2_step: float,
    qi3_start: float,
    qi3_step: float,
    np1: int,
    np2: int,
    np3: int,
    stride1: int,
    stride2: int,
) -> None:
    """Whole per-frame MC LUT forward model, fused into one nogil numba pass.

    Replaces the numpy chain ``qc = base_qc + ang`` → ``qi = Theta @ qc`` →
    ``np.floor(...).astype(int16)`` (x3) → in-bounds mask →
    ``Resq_i[i1,i2,i3] * prob_z`` gather → ``np.bincount(weights=float32)``.
    Computing ``qc``/``qi`` per ray inside the loop means none of the per-ray
    (3, N) / (N,) arrays are ever materialized — that memory traffic was the
    post-static-hoist bottleneck — for ~6.6x over the numpy path at
    px510/Nsub=1. ``ang0/1/2`` are the three components of the goniometer
    offset vector; ``Theta`` is the 3x3 imaging-frame rotation.

    Bit-identical to the numpy chain by construction:
      * the 3-term ``Theta @ qc`` dot products are evaluated left-to-right with
        no FMA contraction (``fastmath=False``), matching numpy's matmul;
      * ``np.int16`` casts mirror ``astype(np.int16)`` exactly, including the
        wrap/INT16_MIN behaviour on NaN/inf core rays (which then fail the
        ``0 <= i < npoints`` bounds test and are dropped, same as the mask);
      * contributions are float32-quantized then accumulated into a float64
        image in ascending ray order, exactly as ``np.bincount`` does.
    Because every op is plain IEEE-754 (no BLAS), the result is also
    deterministic across platforms — unlike the numpy matmul it replaces.

    Kept single-threaded (``range``, not ``prange``) so the accumulation order
    is deterministic and bit-exact; ``nogil=True`` lets the per-frame
    ThreadPoolExecutor in the writers parallelize across frames instead.
    ``Resq_flat`` is the C-order ravel of ``Resq_i``; stride1 = npoints2*npoints3,
    stride2 = npoints3.
    """
    t00, t01, t02 = Theta[0, 0], Theta[0, 1], Theta[0, 2]
    t10, t11, t12 = Theta[1, 0], Theta[1, 1], Theta[1, 2]
    t20, t21, t22 = Theta[2, 0], Theta[2, 1], Theta[2, 2]
    n_rays = base_qc.shape[1]
    for r in range(n_rays):
        c0 = base_qc[0, r] + ang0
        c1 = base_qc[1, r] + ang1
        c2 = base_qc[2, r] + ang2
        qi0 = t00 * c0 + t01 * c1 + t02 * c2
        qi1 = t10 * c0 + t11 * c1 + t12 * c2
        qi2 = t20 * c0 + t21 * c1 + t22 * c2
        i1 = np.int16(np.floor((qi0 - qi1_start) / qi1_step))
        i2 = np.int16(np.floor((qi1 - qi2_start) / qi2_step))
        i3 = np.int16(np.floor((qi2 - qi3_start) / qi3_step))
        if 0 <= i1 < np1 and 0 <= i2 < np2 and 0 <= i3 < np3:
            gathered = Resq_flat[np.int64(i1) * stride1 + np.int64(i2) * stride2 + np.int64(i3)]
            im_flat[flat_indices[r]] += np.float32(gathered * prob_z[r])


def forward_from_static(
    base_qc: np.ndarray,
    phi: float = 0,
    chi: float = 0,
    TwoDeltaTheta: float = 0,
    qi_return: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Per-frame half of the forward model.

    Takes ``base_qc`` from ``precompute_forward_static`` and the goniometer
    angles, and produces the detector image. Reproduces the exact float
    operations of the historical monolithic ``forward()`` tail (no
    reassociation), so output is bit-identical.
    """
    if Resq_i is None and _analytic_eval is None:
        raise RuntimeError(
            "forward_model state is not initialized. Load a kernel "
            "(_lookup_and_load_kernel) or register the analytic backend "
            "(_load_analytic_resolution) before computing the forward model."
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

    # Goniometer offset vector components (added to base_qc to form qc).
    ang0 = phi - TwoDeltaTheta / 2
    ang1 = float(chi)
    ang2 = (TwoDeltaTheta / 2) / np.tan(theta_0)

    # The analytic backend and the qi_return diagnostic both need the full
    # materialized qi array. The common MC path does NOT -- the fused kernel
    # below recomputes qc/qi per ray, so we skip building them here (that
    # materialization was the per-frame memory bottleneck).
    qi = None
    if _analytic_eval is not None or qi_return:
        qc = base_qc + np.asarray([[ang0], [ang1], [ang2]])
        qi = Theta @ qc

    if _analytic_eval is not None:
        # Grid-free: evaluate the closed-form p_Q at every ray's qi. No
        # grid-bounds mask -- the closed form returns ~0 in the tails, and
        # _flat_indices already maps every ray to its detector pixel.
        assert qi is not None
        prob = (_analytic_eval(qi) * prob_z).astype(np.float32)
        contribution = np.bincount(_flat_indices, weights=prob, minlength=im_1.size)
        im_1 += contribution.reshape(im_1.shape)
        if qi_return:
            return im_1, qi.reshape(3, NN1, NN2, NN3)
        return im_1

    # MC LUT path: the analytic branch above returned early, so reaching here
    # guarantees Resq_i is loaded. Assert re-narrows it for the type checker
    # (the guard now only requires that *one* of Resq_i / _analytic_eval is set).
    assert Resq_i is not None
    # A loaded kernel sets all the grid metadata together; narrow for mypy.
    assert npoints1 is not None and npoints2 is not None and npoints3 is not None

    # Whole per-frame forward (qc add, qi=Theta@qc, floor, in-bounds mask,
    # Resq_i gather, float32 scatter) fused into one nogil numba pass that
    # scatters directly into `im_1`'s flat view -- never materializing the
    # per-ray (3, N)/(N,) arrays that dominated the post-static-hoist memory
    # traffic. ~6.6x over the numpy path at px510/Nsub=1; bit-identical and
    # platform-deterministic (see _mc_lut_forward).
    _mc_lut_forward(
        base_qc,
        ang0,
        ang1,
        ang2,
        Theta,
        prob_z,
        _flat_indices,
        Resq_i.reshape(-1),
        im_1.reshape(-1),
        qi1_start,
        qi1_step,
        qi2_start,
        qi2_step,
        qi3_start,
        qi3_step,
        npoints1,
        npoints2,
        npoints3,
        npoints2 * npoints3,
        npoints3,
    )
    if qi_return:
        assert qi is not None  # built above because qi_return is True
        return im_1, qi.reshape(3, NN1, NN2, NN3)
    return im_1


def forward(
    Hg: np.ndarray,
    phi: float = 0,
    chi: float = 0,
    TwoDeltaTheta: float = 0,
    qi_return: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the DFXM forward-model image for the given goniometer angles.

    Contrast formation. Each detector pixel is a depth-integrated projection
    along the diffracted 2theta direction: the ``(NN1, NN2, NN3)`` ray grid is
    summed onto the ``(NN2//Nsub, NN1//Nsub)`` image (the ``flat_indices``
    scatter in ``_mc_lut_forward``). A ray contributes intensity only when its
    local scattering vector ``qi = Theta @ (Us @ Hg @ q_hkl + ang)`` lands
    inside the reciprocal-space resolution function ``Res_q`` (the bootstrapped
    LUT, or its analytic closed form); rays whose ``qi`` falls outside the
    LUT bounds are dropped (dark). So a pixel is bright where the locally
    strained lattice still satisfies the rocking condition and dark where the
    strain has rotated/dilated it out of the acceptance window — this is the
    weak-beam contrast that images dislocation strain fields. Features that look
    concentrated near a dislocation core are the physically-correct deep
    weak-beam regime, not an artifact: far from the core the strain is small and
    the lattice sits squarely in ``Res_q`` (uniform bright background), while the
    steep near-core strain sweeps ``qi`` through and out of the window, carving
    the characteristic contrast. See Poulsen et al. 2021 (J. Appl. Cryst.) for
    the DFXM resolution-function / projection geometry and Borgi et al. 2024
    (IUCrJ; doi:10.1107/S1600576724001183) for this kinematic implementation.

    Thin wrapper over ``precompute_forward_static`` + ``forward_from_static``,
    preserved for one-shot callers (single images, tests). For scans, call
    ``precompute_forward_static(Hg)`` once and ``forward_from_static`` per frame
    to avoid recomputing the Hg-only ``base_qc`` every frame.

    Args:
        Hg: Displacement gradient field, shape (X, 3, 3) where X = NN1*NN2*NN3.
        phi: Radians off the Bragg condition (rotation around y_l axis).
        chi: Radians off the Bragg condition (rotation around x_l axis).
        TwoDeltaTheta: Radians off the Bragg angle (2theta shift).
        qi_return: If True, also return the scattering vector field qi.

    Returns:
        Forward-model image of shape (NN2//Nsub, NN1//Nsub). If qi_return is
        True, returns (image, qi_field) where qi_field has shape (3, NN1, NN2, NN3).
    """
    # Guard BEFORE precompute so an uninitialized call (e.g. forward(Hg=None)
    # with no kernel loaded) raises the clear RuntimeError rather than a
    # TypeError from `Us @ None`.
    if Resq_i is None and _analytic_eval is None:
        raise RuntimeError(
            "forward_model state is not initialized. Load a kernel "
            "(_lookup_and_load_kernel) or register the analytic backend "
            "(_load_analytic_resolution) before computing the forward model."
        )
    base_qc = precompute_forward_static(Hg)
    return forward_from_static(
        base_qc, phi=phi, chi=chi, TwoDeltaTheta=TwoDeltaTheta, qi_return=qi_return
    )


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
    rotation_deg: shape (N,) — per-dislocation mixed-character rotation angle
        (degrees) of the line direction `t` around the slip-plane normal `n`,
        starting from `t_0 = b x n`. ``None`` (the default for every current
        builder) means pure edge for all dislocations (rotation_deg = 0), the
        legacy behaviour. See ``crystal.dislocations.Fd_find_mixed`` for the
        screw/edge convention (rotation_deg=0 edge, 90 screw).
    """

    positions_um: np.ndarray
    Ud: np.ndarray
    sidecar: dict | None
    rotation_deg: np.ndarray | None = None


def _ud_matrix_from_bnt(
    b: tuple[int, int, int],
    n: tuple[int, int, int],
    t: tuple[int, int, int],
) -> np.ndarray:
    """Build a 3x3 column-stacked rotation matrix [b_hat | n_hat | t_hat].

    Input vectors are crystallographic integer indices; output columns
    are unit-normalized. If the user supplied `t` antiparallel to the
    right-handed (n x b) direction, the raw column stack would have
    det=-1 (a reflection rather than a rotation). Flip the t column
    in that case so the result is always a proper rotation (det=+1) —
    matches the legacy IUCrJ 2024 hardcoded Ud convention.
    """
    arr = np.asarray([b, n, t], dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    Ud = (arr / norms).T  # columns = b_hat, n_hat, t_hat
    if np.linalg.det(Ud) < 0:
        Ud[:, 2] = -Ud[:, 2]
    return Ud


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


def _population_rotation_deg(population: DislocationPopulation, n: int) -> np.ndarray:
    """Per-dislocation mixed-character rotation angles (degrees), length n.

    ``population.rotation_deg is None`` (every current builder) means pure edge
    for all dislocations -> all zeros, the legacy behaviour. Otherwise the
    supplied array is broadcast/validated to shape (n,).
    """
    if population.rotation_deg is None:
        return np.zeros(n, dtype=np.float64)
    rot = np.asarray(population.rotation_deg, dtype=np.float64).reshape(-1)
    if rot.shape[0] != n:
        raise ValueError(
            f"population.rotation_deg has length {rot.shape[0]}, expected {n} "
            f"(one per dislocation)."
        )
    return rot


def _find_hg_from_population_numpy(
    population: DislocationPopulation,
    h: int,
    k: int,
    l: int,
    *,
    S: np.ndarray,
    rl: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference NumPy implementation of the population Hg field.

    Kept verbatim as the parity oracle for the fused numba kernel
    (see docs/superpowers/plans/2026-05-27-find-hg-numba-fusion.md). Composes
    Fd_find_multi_dislocs_mixed (Fg) with fast_inverse2 (Fg -> Hg). Do not
    "optimize" this; its whole purpose is to be the slow, obviously-correct
    truth the kernel is checked against at rtol=1e-12.
    """
    from dfxm_geo.crystal.dislocations import Fd_find_multi_dislocs_mixed, MixedDislocSpec

    rl_eff = rl if rl is not None else globals()["rl"]

    Q_norm = np.sqrt(h * h + k * k + l * l)
    q_hkl = np.asarray([h, k, l]) / Q_norm

    # Build MixedDislocSpec per dislocation. Fd_find_multi_dislocs_mixed
    # supports per-crystal Ud + lab-frame offset. Honour each dislocation's
    # mixed-character rotation_deg (None -> pure edge for all, the default).
    rot_deg = _population_rotation_deg(population, len(population.positions_um))
    crystals = [
        MixedDislocSpec(
            Ud_mix=population.Ud[i],
            rotation_deg=float(rot_deg[i]),
            position_lab_um=(
                float(population.positions_um[i, 0]),
                float(population.positions_um[i, 1]),
                float(population.positions_um[i, 2]),
            ),
        )
        for i in range(len(population.positions_um))
    ]

    # Compute Fg via the multi-dislocation kernel.
    # rl is in metres here; Fd_find_mixed's field formula expects micrometres
    # (b = BURGERS_VECTOR is in µm), so convert — exactly as the wall path does
    # via Fd_find(rl * 1e6, ...) and the reference disloc_identify.py does via
    # Fd_find_mixed(rl * 1e6, ...). Omitting this made |rd| 1e6x too small and
    # the 1/r field 1e6x too large (sub-project C regression).
    # Returns shape (X, 3, 3) with identity already added (Fg, not Fdd).
    Fg = Fd_find_multi_dislocs_mixed(rl_eff * 1e6, Us, crystals, Theta, S=S)

    # Convert Fg → Hg using the same convention as load_or_generate_Hg:
    #   Hg = transpose(Fg^-1) - I
    # This is the displacement gradient form that forward() expects.
    Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1])
    Hg -= np.identity(3)

    return Hg, q_hkl


def Find_Hg_from_population(
    population: DislocationPopulation,
    h: int = -1,
    k: int = 1,
    l: int = -1,
    *,
    S: np.ndarray = _S_IDENTITY,
    rl: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Hg + q_hkl from an arbitrary DislocationPopulation.

    For mode='centered' (1 dislocation) and 'random_dislocations' (N drawn
    dislocations). Mirrors `Find_Hg` but takes explicit positions + Ud matrices
    instead of the wall-layout (dis, ndis) parameters.

    Hg convention: same as `load_or_generate_Hg` — Hg = transpose(Fg^-1) - I
    (displacement gradient, not deformation gradient), consistent with
    `forward()` which uses Hg to compute scattering vectors.

    Args:
        population: DislocationPopulation from `build_dislocation_population`.
        h, k, l: Miller indices of the active reflection.
        S: 3x3 sample-remount rotation (default identity).
        rl: Detector ray grid to evaluate the strain on. Defaults to the
            module-level `fm.rl`. Pass `Z_shift(z_um)` to evaluate at a
            non-zero sample-depth offset (z-scan support).

    Returns:
        (Hg, q_hkl) where Hg has shape (X, 3, 3) and q_hkl has shape (3,).
    """
    from dfxm_geo.crystal.dislocations import find_hg_population

    rl_eff = rl if rl is not None else globals()["rl"]
    Q_norm = np.sqrt(h * h + k * k + l * l)
    q_hkl = np.asarray([h, k, l]) / Q_norm

    n = len(population.positions_um)
    # Collapse the per-dislocation transform to M_d = Ud_d.T @ Us.T @ S.T @ Theta.
    base = Us.T @ S.T @ Theta  # (3, 3), shared across dislocations
    M = np.empty((n, 3, 3))
    Ud = np.empty((n, 3, 3))
    offset = np.empty((n, 3))
    for i in range(n):
        Ud[i] = population.Ud[i]
        M[i] = population.Ud[i].T @ base
        offset[i] = population.positions_um[i]
    # Honour each dislocation's mixed-character rotation_deg (edge<->screw).
    # None means pure edge for the whole population (rotation_deg = 0), the
    # legacy default: cos=1, sin=0.
    rot_deg = _population_rotation_deg(population, n)
    cos_rot = np.cos(np.deg2rad(rot_deg))
    sin_rot = np.sin(np.deg2rad(rot_deg))

    # rl is in metres; the field formula expects micrometres (b in µm) — *1e6,
    # exactly as the NumPy path and the reference disloc_identify.py do.
    Hg = find_hg_population(rl_eff * 1e6, M, offset, Ud, cos_rot, sin_rot)
    return Hg, q_hkl
