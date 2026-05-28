"""Colormap helpers for DFXM orientation visualizations.

Provides :func:`inv_polefigure_colors`, a 2D mosaicity colour key keyed to
seven anchor points (white/cyan/green/orange/red/blue/magenta) interpolated by
:func:`scipy.interpolate.griddata`.

.. note:: Not a crystallographic inverse pole figure

    Despite the historical name, this is **not** a crystallographic
    inverse-pole-figure mapping: there is no stereographic projection and no
    cubic standard (001-011-111) triangle. It is a flat 2D colour gradient
    over a ``(chi, diffry)`` plane — a legend/colour key for the two mosaicity
    components. The name is retained for backward compatibility.
"""

import numpy as np
from scipy.interpolate import griddata


def inv_polefigure_colors(
    o_grid: tuple[np.ndarray, np.ndarray],
    test_grid: tuple[np.ndarray, np.ndarray],
    float_bit: type = np.float16,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a 2D mosaicity RGB+alpha colour key onto a grid of points.

    The colour gradient is anchored at seven points in ``(chi, diffry)`` space
    derived from the ``test_grid`` extents; RGBA values at the ``o_grid``
    sample points are interpolated from those anchors and clipped to ``[0, 1]``.
    Sample points outside the convex hull of the seven anchors (where linear
    interpolation is undefined) fall back to the nearest anchor's colour, so
    the returned array never contains NaN.

    .. note:: This is a flat 2D colour key, **not** a crystallographic inverse
        pole figure — see the module docstring.

    Args:
        o_grid: ``(chi, diffry)`` tuple of 1D arrays defining the points to
            be coloured.
        test_grid: ``(chi, diffry)`` tuple of 1D arrays defining the extents
            used to place the seven colour anchors (white centre, cyan/red on
            the ``diffry`` extrema, etc.).
        float_bit: NumPy float dtype for the RGB/coordinate arrays. Default
            ``np.float16`` (small memory, ample precision for 8-bit display).

    Returns:
        ``(xy_colors_griddata, xydata)``:

        - ``xy_colors_griddata``: shape ``(N, 4)``, the RGBA value at each
          ``o_grid`` sample point, clipped to ``[0, 1]``.
        - ``xydata``: shape ``(N, 2)``, the matching coordinate pairs in
          ``(chi, diffry)`` space.
    """
    # Define the RGB values for the color gradient
    key_xy_RGBs: np.ndarray = np.array(
        [
            [1, 1, 1, 1],  # White
            [0, 1, 1, 1],  # Cyan
            [0, 1, 0, 1],  # Green
            [1, 0.65, 0, 1],  # Orange
            [1, 0, 0, 1],  # Red
            [0, 0, 1, 1],  # Blue
            [1, 0, 0.5, 1],  # Magenta
        ],
        dtype=float_bit,
    )

    # Extract the angular data for the original and test grids
    o_chi, o_diffry = o_grid[0], o_grid[1]
    test_chi, test_diffry = test_grid[0], test_grid[1]

    # Define the coordinates for the RGB values
    key_xy_points: np.ndarray = np.array(
        [
            [0, 0],  # White center
            [0, np.min(test_diffry)],  # Cyan min y-axis center x-axis
            [np.max(test_chi), np.min(test_diffry)],  # Green min y-axis max x-axis
            [np.max(test_chi), np.max(test_diffry)],  # Orange max y-axis max x-axis
            [0, np.max(test_diffry)],  # Red max y-axis center x-axis
            [np.min(test_chi), np.min(test_diffry)],  # Blue min y-axis min x-axis
            [np.min(test_chi), np.max(test_diffry)],  # Magenta max y-axis min x-axis
        ],
        dtype=float_bit,
    )

    # Create a 2D array of all coordinate combinations
    xydata: np.ndarray = np.array([(x, y) for x in o_chi for y in o_diffry], dtype=float_bit)

    # Map the RGB values onto the 2D array of coordinates using griddata.
    # Linear griddata returns NaN outside the convex hull of the seven anchor
    # points (and the [0, 1] clip below does not catch NaN, since `NaN < 0` and
    # `NaN > 1` are both False). Fall back to nearest-neighbour interpolation
    # for any out-of-hull point so no NaN RGBA leaks to the caller.
    def _interp(values: np.ndarray) -> np.ndarray:
        linear = griddata(key_xy_points, values, xydata, method="linear")
        outside = np.isnan(linear)
        if outside.any():
            linear[outside] = griddata(
                key_xy_points, values, xydata[outside], method="nearest"
            )
        return linear

    reds = _interp(key_xy_RGBs.T[0])
    greens = _interp(key_xy_RGBs.T[1])
    blues = _interp(key_xy_RGBs.T[2])
    alphas = _interp(key_xy_RGBs.T[3])
    xy_colors_griddata = np.vstack((reds, greens, blues, alphas)).T

    # Force the RGB values to be within range [0, 1]
    xy_colors_griddata[xy_colors_griddata < 0] = 0.0
    xy_colors_griddata[xy_colors_griddata > 1] = 1.0

    return xy_colors_griddata, xydata
