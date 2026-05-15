"""Auxiliary single-objective exposure model for the DFXM paper.

Models the objective as an isotropic Gaussian (with NA) plus a square physical
aperture of side length D, and reports the fraction of incoming rays that are
transmitted. Originally used to enable absolute comparison of intensities.

Run as a script::

    python -m dfxm_geo.reciprocal_space.exposure

H.F. Poulsen, Sept 16, 2023, version 1.0
"""

import numpy as np


def run_exposure_simulation(
    Nrays: int = 1_000_000,
    *,
    zeta_v_fwhm: float = 0.53e-3,
    eps_rms: float = 0.014 / 2.35,
    NA_rms: float = 7.31e-4 / 2.35,
    energy0: float = 17.00,
) -> tuple[int, float]:
    """Monte Carlo a single-objective DFXM exposure and return transmission stats.

    Args:
        Nrays: Number of rays to trace. Default 1e6.
        zeta_v_fwhm: Incoming divergence in the vertical direction (rad).
        eps_rms: RMS width of the x-ray energy bandwidth.
        NA_rms: NA of the objective (rad).
        energy0: Beam energy (keV).

    Returns:
        ``(total_transmitted, fraction_transmitted)``.
    """
    rng = np.random.default_rng()

    lambda0 = 12.398 / energy0  # in Å
    d0 = 4.0495 / np.sqrt(3)  # in Å
    theta0 = np.arcsin(lambda0 / (2 * d0))  # in rad
    G = 2 * np.pi / d0 * np.asarray([-np.sin(theta0), np.cos(theta0)])
    D = 2 * np.sqrt(50e-6 * 1e-3)  # physical aperture of objective, in m
    d1 = 0.274  # sample-objective distance, in m
    phys_aper = D / d1

    transmit = np.zeros(Nrays)
    for ii in range(Nrays):
        zeta_v = rng.standard_normal() * zeta_v_fwhm / 2.35
        energy = energy0 * (1 + rng.standard_normal() * eps_rms)

        wavelength = 12.398 / energy
        k0 = 2 * np.pi / wavelength

        k_in = k0 * np.array([1, zeta_v])
        k_out = G + k_in  # Laue condition

        twotheta = np.arctan2(k_out[1], k_out[0])
        xi = twotheta - 2 * theta0

        if abs(xi) > (phys_aper / 2):
            transmit[ii] = 0
        else:
            test = rng.standard_normal() * NA_rms
            transmit[ii] = 0 if abs(xi) > test else 1

    total_transmitted = int(np.sum(transmit))
    fraction_transmitted = total_transmitted / Nrays
    return total_transmitted, fraction_transmitted


if __name__ == "__main__":
    total, fraction = run_exposure_simulation()
    print("Total Transmitted:", total)
    print("Fraction Transmitted:", fraction)
