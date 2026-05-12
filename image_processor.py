"""DEPRECATED — re-exports from `dfxm_geo.io.images`, `dfxm_geo.analysis.*`.

This module is kept as a compatibility shim during Phase 4 of the cleanup.
New code should import directly:

    from dfxm_geo.io.images import save_images_parallel, load_images
    from dfxm_geo.analysis.moments import calc_moments, fastgrainplot
    from dfxm_geo.analysis.colormaps import inv_polefigure_colors
"""

from dfxm_geo.analysis.colormaps import inv_polefigure_colors  # noqa: F401
from dfxm_geo.analysis.moments import calc_moments, fastgrainplot  # noqa: F401
from dfxm_geo.io.images import (  # noqa: F401
    load_image,
    load_images,
    load_images_parallel,
    save_edfs,
    save_image,
    save_images,
    save_images_parallel,
)
