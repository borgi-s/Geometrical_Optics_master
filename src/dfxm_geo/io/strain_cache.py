"""Strain-field (Hg) loading and caching utilities."""

import json
from pathlib import Path

import numpy as np

from dfxm_geo.constants import BURGERS_VECTOR, POISSON_RATIO
from dfxm_geo.crystal.dislocations import Fd_find
from dfxm_geo.crystal.rotations import fast_inverse2

# Module-level identity used as the default for the S kwarg in load_or_generate_Hg.
# Defined here (not inline) to satisfy ruff B008 (no function calls in defaults).
_S_IDENTITY: np.ndarray = np.identity(3)


def _geom_sidecar_path(file_path: str) -> Path:
    """Return the .geom.json sidecar path alongside `file_path`."""
    return Path(file_path).with_suffix(".geom.json")


def _geom_signature_matches(file_path: str, geom_signature: "tuple") -> "bool | None":
    """Check the .geom.json sidecar against `geom_signature`.

    Returns:
        ``True``  — sidecar exists and matches (safe to load).
        ``False`` — sidecar exists but does NOT match (must regenerate).
        ``None``  — sidecar is absent (fall back to legacy shape-only guard,
                    for back-compat with caches predating this guard).
    Corrupted JSON is treated as a mismatch (``False``) → safe regeneration.
    """
    sidecar = _geom_sidecar_path(file_path)
    if not sidecar.exists():
        return None  # absent → back-compat fall-through
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        stored = tuple(data["geom_signature"])
        return stored == tuple(geom_signature)
    except Exception:
        return False  # corrupted → regenerate


def _write_geom_signature(file_path: str, geom_signature: "tuple") -> None:
    """Write `geom_signature` to the .geom.json sidecar alongside `file_path`."""
    sidecar = _geom_sidecar_path(file_path)
    sidecar.write_text(json.dumps({"geom_signature": list(geom_signature)}), encoding="utf-8")


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

    ``geom_signature`` is an optional tuple ``(h, k, l, theta_int)`` that
    identifies the reflection geometry.  When provided:

    * On **save**: a ``<file_path>.geom.json`` sidecar is written alongside the
      ``.npy`` cache so future loads can verify the geometry.
    * On **load**: the sidecar is read and compared; a mismatch forces
      regeneration (the stale cache is overwritten).  A *missing* sidecar falls
      back to the legacy shape-only guard — this preserves back-compat with
      caches written before this guard existed.

    When ``geom_signature`` is ``None`` (default) the old behaviour is
    unchanged: only the ray-count shape guard applies.
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
            elif geom_signature is not None and (
                _geom_signature_matches(file_path, geom_signature) is False
            ):
                # Sidecar is PRESENT but the signature does NOT match (e.g. a
                # stale pre-fix cache that was later given a sidecar for a
                # different reflection, or the same filename re-used across runs
                # with different keV). Regenerate and overwrite.
                # Note: ``is False`` (not just falsy) — ``None`` means absent sidecar,
                # which falls through to the shape-guard path below (back-compat).
                print(f"Cached Fg in {fname} has a mismatching geometry signature; regenerating.")
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
            print(f"Saved Fg to {file_path.rsplit('/', 1)[-1]}")
            if geom_signature is not None:
                _write_geom_signature(file_path, geom_signature)

    Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1])
    Hg -= np.identity(3)

    return Hg
