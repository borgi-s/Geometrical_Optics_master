#function absolute_rec_space
# Auxiliary model for use with rec. space resolution function for DFXM paper.
# THis allows for absolute comparison of intensities
# The objective is modelled as an isotropic Gaussian with an NA and in addition
# a square phyical aperture of d side length D.
 
# H.F. Poulsen, Sept 16, 2023, version 1.0

import numpy as np

# Input parameters
Nrays = 1000000
zeta_v_fwhm = 0.53E-3  # incoming divergence in vertical direction, in rad
eps_rms = 0.014 / 2.35  # rms width of x-ray energy bandwidth
NA_rms = 7.31E-4 / 2.35  # NA of objective, in rad
energy0 = 17.00  # in keV

lambda0 = 12.398 / energy0  # in Å
d0 = 4.0495 / np.sqrt(3)  # in Å
theta0 = np.arcsin(lambda0 / (2 * d0))  # in rad
G = 2 * np.pi / d0 * np.asarray([-np.sin(theta0), np.cos(theta0)])
D = 2 * np.sqrt(50E-6 * 1E-3)  # physical aperture of objective, in m
d1 = 0.274  # sample-objective distance, in m
phys_aper = D / d1

# Initialize variables to store transmitted rays
transmit = np.zeros(Nrays)

# Ray tracing in lab system
for ii in range(Nrays):
    # Make ray tracing for one ray
    zeta_v = np.random.randn() * zeta_v_fwhm / 2.35  # Gaussian
    energy = energy0 * (1 + np.random.randn() * eps_rms)

    wavelength = 12.398 / energy  # in Å
    k0 = 2 * np.pi / wavelength

    k_in = k0 * np.array([1, zeta_v])
    k_out = G + k_in  # The Laue condition

    twotheta = np.arctan2(k_out[1], k_out[0])
    xi = twotheta - 2 * theta0

    # Figure out if it is transmitted through the objective or not
    if abs(xi) > (phys_aper / 2):
        transmit[ii] = 0
    else:
        test = np.random.randn() * NA_rms
        if abs(xi) > test:
            transmit[ii] = 0
        else:
            transmit[ii] = 1

# Sum the number of transmitted rays
total_transmitted = np.sum(transmit)
fraction_transmitted = total_transmitted / Nrays

print("Total Transmitted:", total_transmitted)
print("Fraction Transmitted:", fraction_transmitted)
