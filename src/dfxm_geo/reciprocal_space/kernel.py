"""Driver: generates a reciprocal-space resolution kernel pickle.

Configures the Monte Carlo integration parameters and calls
:func:`dfxm_geo.reciprocal_space.resolution.reciprocal_res_func`. Side
effects: writes ``pkl_files/Resq_i_<timestamp>.pkl`` and
``pkl_files/Resq_i_<timestamp>_vars.txt`` to the current working directory.

Defaults reproduce the CDD_inc canonical recipe (Al 111 reflection at
17 keV, beamstop ON via square aperture of side 25 mm at the BFP).

Run as a script::

    python -m dfxm_geo.reciprocal_space.kernel
"""

from datetime import datetime

import numpy as np

from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func


def _default_theta_al_111(keV: float = 17) -> float:
    """Bragg angle for Al 111 at the given beam energy (default 17 keV)."""
    a = 4.0495e-10  # Al lattice parameter, m
    d_111 = a / np.sqrt(3)
    wavelength = 1.239841984e-9 / keV
    return float(np.arcsin(wavelength / (2 * d_111)))


def generate_kernel(
    date: str | None = None,
    *,
    Nrays: int = int(1e8),
    npoints1: int = 400,
    npoints2: int = 200,
    npoints3: int = 200,
    qi1_range: float = 5e-4,
    qi2_range: float = 0.75e-2,
    qi3_range: float = 0.75e-2,
    zeta_v_fwhm: float = 5.3e-04,
    zeta_h_fwhm: float = 0,
    NA_rms: float = 7.31e-4 / 2.35,
    eps_rms: float = 1.41e-4 / 2.35,
    theta: float = _default_theta_al_111(17),
    D: float = float(2 * np.sqrt(50e-6 * 1.6e-3)),
    d1: float = 0.274,
    beamstop: bool = True,
    bs_height: float = 25e-3,
    aperture: bool = True,
    knife_edge: bool = False,
    dphi_range: float = 0.0,
) -> str:
    """Run the kernel-generation Monte Carlo and write the pickle to ``pkl_files/``.

    Defaults reproduce the CDD_inc canonical recipe (Al 111 reflection at
    17 keV; square-aperture beamstop with 25 mm side at the BFP).

    Args:
        date: Timestamp tag for the output filenames. Defaults to
            ``YYYYmmdd_HHMM`` from the current local time.
        Nrays: number of Monte Carlo rays.
        npoints1/2/3: voxel counts for the qi grid.
        qi1_range/qi2_range/qi3_range: half-widths of the qi grid.
        zeta_v_fwhm/zeta_h_fwhm: incoming-beam divergence FWHM (rad).
        NA_rms/eps_rms: objective NA / energy-bandwidth rms.
        theta: Bragg angle (rad).
        D: physical objective aperture (m).
        d1: sample-objective distance (m).
        beamstop/bs_height/aperture/knife_edge: beamstop config; see
            :func:`dfxm_geo.reciprocal_space.resolution.reciprocal_res_func`.
        dphi_range: rocking-curve sweep half-width (rad).

    Returns:
        The timestamp tag that was used.
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d_%H%M")

    phys_aper = D / d1

    reciprocal_res_func(
        Nrays,
        npoints1,
        npoints2,
        npoints3,
        qi1_range,
        qi2_range,
        qi3_range,
        plot_figs=False,
        save_resqi=True,
        zeta_v_fwhm=zeta_v_fwhm,
        zeta_h_fwhm=zeta_h_fwhm,
        NA_rms=NA_rms,
        eps_rms=eps_rms,
        theta=theta,
        phys_aper=phys_aper,
        date=date,
        beamstop=beamstop,
        bs_height=bs_height,
        aperture=aperture,
        knife_edge=knife_edge,
        dphi_range=dphi_range,
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
        "beamstop": beamstop,
        "bs_height": bs_height,
        "aperture": aperture,
        "knife_edge": knife_edge,
        "dphi_range": dphi_range,
    }

    with open(f"pkl_files/Resq_i_{date}_vars.txt", "w") as data:
        data.write(str(vars_used))

    return date


if __name__ == "__main__":
    generate_kernel()
