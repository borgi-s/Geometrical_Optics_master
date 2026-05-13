"""First- and second-moment analysis of DFXM rocking-curve image stacks.

Two entry points:

- :func:`fastgrainplot`: per-pixel COM (1st moment) and FWHM (from 2nd moment)
  over an image stack acquired on a (u, v) angular grid. Despite the legacy
  name, this function does *not* plot — it returns four ``(H, W)`` arrays.
- :func:`calc_moments`: scalar moment dictionary for a single image, intended
  for single-pixel orientation spreads.

A direct port of the original MATLAB analysis used in ``init_forward.py``.
"""

import numpy as np


def fastgrainplot(
    imagestack: np.ndarray,
    vlist: np.ndarray,
    ulist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel center-of-mass and FWHM maps over a (u, v) rocking grid.

    Iterates a flat ``imagestack`` of length ``len(ulist) * len(vlist)``,
    treating ``vlist`` as the fast (inner) motor and ``ulist`` as the slow
    (outer) motor — index ``j`` maps to ``v = vlist[j % len(vlist)]`` and
    ``u = ulist[j // len(vlist)]``. The MATLAB original computed skewness
    and kurtosis as well; those terms were dead and were removed during the
    Phase 3.2 cleanup.

    Note:
        The legacy name is a misnomer — this returns arrays, not plots.
        Renamed only at the import-shim level for now to avoid breaking
        downstream callers.

    Args:
        imagestack: Shape ``(len(ulist) * len(vlist), H, W)`` flat stack of
            detector images, ordered ``v``-major (fast motor inner).
        vlist: Fast-motor angles (1D), length ``v_steps``.
        ulist: Slow-motor angles (1D), length ``u_steps``.

    Returns:
        ``(unorm, vnorm, ufwhm, vfwhm)``, each of shape ``(H, W)``:

        - ``unorm``, ``vnorm``: per-pixel center of mass in ``u`` / ``v``.
        - ``ufwhm``, ``vfwhm``: per-pixel FWHM (``2.355 * sqrt(variance)``);
          pixels with non-positive variance are returned as ``NaN``.
    """
    imglist = imagestack

    oridist = np.zeros(len(vlist) * len(ulist))
    inttot = np.zeros_like(imglist[0])
    v1sum = np.zeros_like(inttot)
    v2sum = np.zeros_like(inttot)
    v3sum = np.zeros_like(inttot)
    v4sum = np.zeros_like(inttot)
    u1sum = np.zeros_like(inttot)
    u2sum = np.zeros_like(inttot)
    u3sum = np.zeros_like(inttot)
    u4sum = np.zeros_like(inttot)

    for j in range(len(imglist)):
        img = imglist[j]
        img[img < 0] = 0

        # For grain shape
        inttot += img

        # For calculating moments.
        # vlist is the fast motor (inner loop), ulist is the slow motor (outer loop).
        vv = vlist[j % len(vlist)]  # fast motor: cycles through all v steps
        uu = ulist[j // len(vlist)]  # slow motor: advances once per full v sweep

        v1sum += vv * img
        v2sum += vv**2 * img
        v3sum += vv**3 * img
        v4sum += vv**4 * img

        u1sum += uu * img
        u2sum += uu**2 * img
        u3sum += uu**3 * img
        u4sum += uu**4 * img

        # For orientation distribution
        oridist[j] = img.sum()

    # Expectation value
    vnorm = v1sum / (inttot * 2)
    unorm = u1sum / (inttot * 2)

    # Variance
    vvar = v2sum / inttot * 2 - vnorm**2
    uvar = u2sum / inttot * 2 - unorm**2

    # Replace negative variances with NaN
    vvar[vvar <= 0] = np.nan
    uvar[uvar <= 0] = np.nan

    # FWHM
    vfwhm = 2.355 * np.sqrt(vvar)
    ufwhm = 2.355 * np.sqrt(uvar)

    # Skewness and kurtosis were computed here but never returned or used.
    # Removed during cleanup; reinstate via git history if you need them
    # and route through the return tuple.
    return unorm, vnorm, ufwhm, vfwhm


def calc_moments(
    image: np.ndarray,
    u_range: float,
    v_range: float,
    u_steps: int,
    v_steps: int,
) -> dict[str, float]:
    """Raw, central, and standardized moments of a single 2D image.

    Intended for single-pixel orientation spreads — pass a single pixel's
    rocking-curve image of shape ``(u_steps, v_steps)`` and the corresponding
    grid extents. The ``(u, v)`` grid is reconstructed internally via
    ``np.mgrid``.

    Args:
        image: Shape ``(u_steps, v_steps)``. Typically a single pixel's
            rocking-curve sampled on a phi/chi grid.
        u_range: Half-range of the ``u`` axis (radians by convention; same
            unit as ``init_forward.py``'s ``phi_range``).
        v_range: Half-range of the ``v`` axis (radians).
        u_steps: Number of samples along ``u``.
        v_steps: Number of samples along ``v``.

    Returns:
        A dict with keys ``mean_u``, ``mean_v`` (1st moments / COM);
        ``m00`` … ``m20`` (raw spatial moments); ``mu02`` … ``mu30``
        (central moments); ``nu03`` and ``nu30`` (standardized 3rd-order
        moments / skewness proxies).
    """
    u, v = np.mgrid[-u_range : u_range : complex(u_steps), -v_range : v_range : complex(v_steps)]

    moments = {}
    moments["mean_v"] = np.sum(v * image) / np.sum(image)
    moments["mean_u"] = np.sum(u * image) / np.sum(image)

    # raw or spatial moments
    moments["m00"] = np.sum(image) * 2
    moments["m01"] = np.sum(u * image)
    moments["m10"] = np.sum(v * image)
    moments["m02"] = np.sum(u**2 * image)
    moments["m20"] = np.sum(v**2 * image)

    # central moments
    moments["mu02"] = np.sum((u - moments["mean_u"]) ** 2 * image)  # variance
    moments["mu20"] = np.sum((v - moments["mean_v"]) ** 2 * image)  # variance
    moments["mu03"] = np.sum((u - moments["mean_u"]) ** 3 * image)
    moments["mu30"] = np.sum((v - moments["mean_v"]) ** 3 * image)

    # central standardized or normalized or scale invariant moments
    moments["nu03"] = moments["mu03"] / np.sum(image) ** (3 / 2 + 1)  # skewness
    moments["nu30"] = moments["mu30"] / np.sum(image) ** (3 / 2 + 1)  # skewness
    return moments
