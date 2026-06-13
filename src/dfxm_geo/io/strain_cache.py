"""Strain-field (Hg) loading and caching utilities."""

import json

import numpy as np

from dfxm_geo.constants import BURGERS_VECTOR, POISSON_RATIO
from dfxm_geo.crystal.dislocations import Fd_find
from dfxm_geo.crystal.rotations import fast_inverse2

# Module-level identity used as the default for the S kwarg in load_or_generate_Hg.
# Defined here (not inline) to satisfy ruff B008 (no function calls in defaults).
_S_IDENTITY: np.ndarray = np.identity(3)


def _geom_sidecar_path(file_path: str) -> str:
    """Path of the sidecar recording the geometry signature a cached Fg was built for."""
    if file_path.endswith(".npy"):
        return file_path[:-4] + ".geom.json"
    return file_path + ".geom.json"


def _geom_signature_matches(file_path: str, geom_signature: "tuple | None") -> bool:
    """True if the cached Fg's recorded geometry matches ``geom_signature``.

    Returns True (guard disabled) when no signature is supplied, OR when the
    sidecar is missing/garbled — a legacy cache written before this guard falls
    back to the shape-only check rather than being force-regenerated.
    """
    if geom_signature is None:
        return True
    try:
        with open(_geom_sidecar_path(file_path), encoding="utf-8") as fh:
            stored = json.load(fh)
    except (FileNotFoundError, OSError, ValueError):
        return True
    return stored == list(geom_signature)  # JSON serialises the tuple as a list


def _write_geom_signature(file_path: str, geom_signature: "tuple | None") -> None:
    """Record the geometry signature alongside a freshly-saved Fg cache (best-effort)."""
    if geom_signature is None:
        return
    try:
        with open(_geom_sidecar_path(file_path), "w", encoding="utf-8") as fh:
            json.dump(list(geom_signature), fh)
    except OSError:
        pass  # a missing sidecar merely disables the guard on the next load


def load_or_generate_Hg(
    rl: np.ndarray,
    Ud: np.ndarray,
    Us: np.ndarray,
    Theta: np.ndarray,
    dis: float,
    ndis: int,
    file_path: str | None = None,
    *,
    b: float = BURGERS_VECTOR,
    ny: float = POISSON_RATIO,
    S: np.ndarray = _S_IDENTITY,
    geom_signature: "tuple | None" = None,
) -> np.ndarray:
    """Return the displacement gradient field Hg, loading from disk if cached.

    If `file_path` is given and the file exists *and* its leading dimension
    matches `rl.shape[1]`, load Fg from it. A shape mismatch (e.g. cache was
    written under a different detector ray grid) regenerates Fg and overwrites
    the cache rather than silently corrupting the result. Hg is derived from
    Fg by bulk inversion + transpose.

    The optional ``S`` rotation matrix is the sample-remount transformation
    (Purdue 2024 paper). When `S = identity` (default), the call matches the
    pre-port behaviour bit-for-bit; with `S != identity`, the strain field is
    computed in a remounted-sample frame.

    ``b`` is the Burgers magnitude (µm); default ``BURGERS_VECTOR`` (FCC, the
    v2.x value — byte-identical). Non-FCC walls pass the cell-derived |b|
    (M4 Stage 4.3a). ``b`` is forwarded to ``Fd_find`` where it linearly
    scales the displacement gradient (physics). ``b`` does NOT enter the cache
    filename HERE, so a non-default ``b`` must be paired with a distinct
    ``file_path`` (the wall path keys ``b`` into the cache filename in
    ``Find_Hg`` via its ``_b...`` suffix) to avoid loading a stale-|b| cache.

    ``ny`` is the isotropic Poisson ratio; default ``POISSON_RATIO`` (0.334, Al
    — byte-identical to v2.x). Like ``b`` it is forwarded to ``Fd_find`` (physics)
    and does NOT enter the cache filename here; ``Find_Hg`` keys a non-default ``ny``
    into the filename via its ``_ny...`` suffix.

    ``geom_signature`` (M4.3a follow-up #1) is the geometry the Fg was computed
    for — ``(h, k, l, theta_0)`` from ``Find_Hg``. When given, it is written to a
    ``.geom.json`` sidecar on save and verified on load: a cache computed for a
    DIFFERENT reflection / Bragg angle is regenerated rather than silently reused
    (the leading-dimension shape guard can't catch this — the ray count is
    identical across reflections). ``None`` disables the guard (legacy callers).
    """
    expected_n = rl.shape[1]
    Fg: np.ndarray | None = None

    if file_path is not None:
        fname = file_path.rsplit("/", 1)[-1]
        try:
            candidate = np.load(file_path)
        except FileNotFoundError:
            print(f"File '{fname}' not found. \nGenerating a new Fg array.")
        else:
            if candidate.shape[0] != expected_n:
                print(
                    f"Cached Fg in {fname} has shape[0]={candidate.shape[0]} "
                    f"but rl needs {expected_n} rays; regenerating."
                )
            elif not _geom_signature_matches(file_path, geom_signature):
                print(
                    f"Cached Fg in {fname} was computed for a different "
                    f"reflection/Bragg angle; regenerating."
                )
            else:
                print(f"Loaded Fg from {fname}")
                Fg = candidate

    if Fg is None:
        if ndis == 0:
            Fg = np.zeros([expected_n, 3, 3])
            Fg += np.identity(Fg.shape[1])
        else:
            Fg = Fd_find(rl * 1e6, Ud, Us, Theta, dis, ndis, b=b, ny=ny, S=S)
        if file_path is not None:
            np.save(file_path, Fg)
            _write_geom_signature(file_path, geom_signature)
            print(f"Saved Fg to {file_path.rsplit('/', 1)[-1]}")

    Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1])
    Hg -= np.identity(3)

    return Hg
