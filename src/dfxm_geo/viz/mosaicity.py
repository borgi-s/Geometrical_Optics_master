"""Figure-making for DFXM mosaicity and qi-field outputs.

Wraps matplotlib in functions that save SVG files (no ``plt.show()``). Use
together with :mod:`dfxm_geo.analysis.mosaicity` to produce the SVGs that the
original ``init_forward.py`` script saved as ``extrem_phi+chi2.svg`` and
``qi1+qi2_fields1.svg``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Force the non-interactive Agg backend before pyplot is loaded. Safe for
# this module (SVG export only). NB: this is process-global — if another
# module (e.g. reciprocal_space/resolution.py) wants an interactive backend
# in the same process, it must be imported before this one.
matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402


def plot_mosaicity_maps(
    phi_list: np.ndarray,
    chi_list: np.ndarray,
    xl_start: float,
    yl_start: float,
    out_path: Path | str,
    *,
    vmin: float = -1e-4,
    vmax: float = 1e-4,
) -> None:
    """Save the two-panel "Extreme Phi" / "Extreme Chi" mosaicity SVG.

    Port of ``init_forward.py:167-214``. Both panels are negated and
    transposed relative to the array layout (matching the original script's
    convention).

    Args:
        phi_list, chi_list: Mosaicity maps in radians, shape ``(H, W)``.
        xl_start, yl_start: Expected to be negative (the module-level globals
            from :mod:`dfxm_geo.direct_space.forward_model` are negative; pass
            them directly). The image extent is built as
            ``[xl_start, -xl_start, yl_start, -yl_start]``.
        out_path: Where to save the SVG. Either a :class:`pathlib.Path` or a
            string path is accepted.
        vmin, vmax: Color limits in radians (default ±1e-4).
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    panels = [
        (phi_list, "Extreme Phi", axs[0]),
        (chi_list, "Extreme Chi", axs[1]),
    ]
    for data, title, ax in panels:
        im = ax.imshow(
            (data * -1).T,
            interpolation="none",
            extent=[xl_start, -xl_start, yl_start, -yl_start],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            origin="lower",
        )
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(r"$y_{\ell}$ ($\mu$m)", fontsize=12)
        ax.set_ylabel(r"$x_{\ell}$ ($\mu$m)", fontsize=12)
        ax.grid(False)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        cbar = fig.colorbar(im, ax=ax, format=formatter)
        cbar.update_ticks()

    plt.tight_layout()
    try:
        fig.savefig(out_path)
    finally:
        plt.close(fig)
