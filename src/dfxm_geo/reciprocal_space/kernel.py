"""Driver script: generates a reciprocal-space resolution kernel pickle.

Configures the Monte Carlo integration parameters and calls
:func:`dfxm_geo.reciprocal_space.resolution.reciprocal_res_func`. Side
effects: writes ``pkl_files/Resq_i_<timestamp>.pkl`` and
``pkl_files/Resq_i_<timestamp>_vars.txt`` to the current working directory.

Run as a script::

    python -m dfxm_geo.reciprocal_space.kernel
"""

from datetime import datetime

import numpy as np

from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func


def generate_kernel(date: str | None = None) -> str:
    """Run the kernel-generation Monte Carlo and write the pickle to ``pkl_files/``.

    Args:
        date: Timestamp tag for the output filenames. Defaults to
            ``YYYYmmdd_HHMM`` from the current local time.

    Returns:
        The timestamp tag that was used (i.e. the value of ``date``).
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d_%H%M")

    Nrays = int(1e8)
    npoints1, npoints2, npoints3 = 400, 200, 200
    qi1_range, qi2_range, qi3_range = 1e-2, 5e-3, 5e-2
    plot_figs = False
    save_resqi = True

    zeta_v_fwhm = 5.3e-04
    zeta_h_fwhm = 0
    NA_rms = 7.31e-4 / 2.35
    eps_rms = 1.41e-4 / 2.35
    theta = (17.953 / 2) * (np.pi / 180)
    D = 2 * np.sqrt(50e-6 * 1.6e-3)
    d1 = 0.274
    phys_aper = D / d1

    reciprocal_res_func(
        Nrays,
        npoints1,
        npoints2,
        npoints3,
        qi1_range,
        qi2_range,
        qi3_range,
        plot_figs,
        save_resqi,
        zeta_v_fwhm,
        zeta_h_fwhm,
        NA_rms,
        eps_rms,
        theta,
        phys_aper,
        date,
    )

    vars_used = {
        "Nrays": Nrays,
        "npoints1": npoints1,
        "npoints2": npoints2,
        "npoints3": npoints3,
        "qi1_range": qi1_range,
        "qi2_range": qi2_range,
        "qi3_range": qi3_range,
        "zeta_v_fwhm": zeta_v_fwhm,
        "zeta_h_fwhm": zeta_h_fwhm,
        "NA_rms": NA_rms,
        "eps_rms": eps_rms,
        "theta": theta,
        "D": D,
        "d1": d1,
        "phys_aper": phys_aper,
    }

    with open(f"pkl_files/Resq_i_{date}_vars.txt", "w") as data:
        data.write(str(vars_used))

    return date


if __name__ == "__main__":
    generate_kernel()
