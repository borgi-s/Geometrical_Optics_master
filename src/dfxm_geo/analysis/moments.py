"""First- and second-moment analysis of a DFXM detector image.

:func:`calc_moments` returns a scalar moment dictionary for a single image,
intended for single-pixel orientation spreads. A direct port of the original
MATLAB analysis used in ``init_forward.py``.

For per-pixel COM/FWHM maps over a rocking stack, see
:mod:`dfxm_geo.analysis.mosaicity` (`compute_com_maps`).
"""

import numpy as np


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
