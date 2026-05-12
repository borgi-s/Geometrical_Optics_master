"""Driver script: generates a reciprocal-space resolution kernel pickle.

Configures the Monte Carlo integration parameters and calls
`reciprocal_res_func` from `dfxm_geo.reciprocal_space.resolution`. Side
effects: writes `pkl_files/Resq_i_<timestamp>.pkl` and
`pkl_files/Resq_i_<timestamp>_vars.txt` to the current working directory.

Run as a script:
    python -m dfxm_geo.reciprocal_space.kernel
"""

from datetime import datetime

import numpy as np

from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func

now = datetime.now()
date = now.strftime("%Y%m%d_%H%M")

# High accuracy requires Nrays = 100 million rays
Nrays = int(1e8)  # nr of rays
npoints1, npoints2, npoints3 = 400, 200, 200
qi1_range, qi2_range, qi3_range = 1e-2, 5e-3, 5e-2
plot_figs = False
save_resqi = True

# Instrumental parameters
zeta_v_fwhm = 5.3e-04  # incoming divergence in vertical direction, in rad
zeta_h_fwhm = 0  # 1e-05 # incoming divergence in horizontal direction, in rad
NA_rms = 7.31e-4 / 2.35  # NA of objective, in rad
eps_rms = 1.41e-4 / 2.35  # rms widht of x-ray energy bandwidth
theta = (17.953 / 2) * (np.pi / 180)  # scattering angle, in rad
D = 2 * np.sqrt(50e-6 * 1.6e-3)  # physical aperture of objective, in mx_cond
d1 = 0.274  # sample-objective distance, in m
phys_aper = D / d1

# Execute the resolution function
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

# Output will be Resq_i as a (npoints1, npoints2*npoints3) array .csv file
# If plot_figs = True and Nrays < 1000001 will also include figures
# save configs
vars = {
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
    data.write(str(vars))
