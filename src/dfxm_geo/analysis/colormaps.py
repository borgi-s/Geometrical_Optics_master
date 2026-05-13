"""Colormap helpers for DFXM orientation visualizations.

Provides :func:`inv_polefigure_colors`, an inverse-pole-figure colour gradient
keyed to seven anchor points (white/cyan/green/orange/red/blue/magenta)
interpolated by :func:`scipy.interpolate.griddata`.
"""

import numpy as np
from scipy.interpolate import griddata


def inv_polefigure_colors(
    o_grid: tuple[np.ndarray, np.ndarray],
    test_grid: tuple[np.ndarray, np.ndarray],
    float_bit: type = np.float16,
) -> tuple[np.ndarray, np.ndarray]:
    """Map an inverse-pole-figure RGB+alpha gradient onto a grid of points.

    The colour gradient is anchored at seven points in ``(chi, diffry)`` space
    derived from the ``test_grid`` extents; RGBA values at the ``o_grid``
    sample points are interpolated from those anchors and clipped to ``[0, 1]``.

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

    # Map the RGB values onto the 2D array of coordinates using griddata
    reds = griddata(key_xy_points, key_xy_RGBs.T[0], xydata)
    greens = griddata(key_xy_points, key_xy_RGBs.T[1], xydata)
    blues = griddata(key_xy_points, key_xy_RGBs.T[2], xydata)
    alphas = griddata(key_xy_points, key_xy_RGBs.T[3], xydata)
    xy_colors_griddata = np.vstack((reds, greens, blues, alphas)).T

    # Force the RGB values to be within range [0, 1]
    xy_colors_griddata[xy_colors_griddata < 0] = 0.0
    xy_colors_griddata[xy_colors_griddata > 1] = 1.0

    return xy_colors_griddata, xydata
