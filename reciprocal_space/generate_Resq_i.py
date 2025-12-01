import numpy as np

# d1_list = np.linspace(0.18, 0.3)
# for i in range(len(d1_list)):

from datetime import datetime
import sys
# sys.path.insert(1, )
from recspace_res import reciprocal_res_func
now = datetime.now()
date = now.strftime("%Y%m%d_%H%M")
# import matplotlib.pyplot as plt


#High accuracy requires Nrays = 100 million rays
Nrays = int(1e8) #nr of rays
npoints1, npoints2, npoints3 = 400, 200, 200
qi1_range, qi2_range, qi3_range = 5E-4, 0.75E-2, 0.75E-2
plot_figs = False
save_resqi = True
return_qs = False
BS_on = True # Use beamstop

# Instrumental parameters
zeta_v_fwhm = 5.3E-04 # incoming divergence in vertical direction, in rad
zeta_h_fwhm = 0 # 1e-05 # incoming divergence in horizontal direction, in rad
NA_rms = 7.31E-4/2.35 # NA of objective, in rad
eps_rms = 1.41e-4/2.35 # rms widht of x-ray energy bandwidth
keV = 17
wavelength = 1.239841984e-9 / keV  # kev to wavelength in m
a = 4.0495e-10  # lattice parameter in m
d_111 = a / np.sqrt(3) # D-Spacing 111 reflection
theta = np.arcsin(wavelength / (2 * d_111))  # scattering angle, in radians

# theta = (17.953/2)*(np.pi/180) # scattering angle, in rad
D = 2*np.sqrt(50E-6*1.6E-3) # physical aperture of objective, in m
d1 = 0.274 # sample-objective distance, in m
phys_aper = D/d1
bs_height = 25e-3 # in mm, height of beamstop (diameter/length)

reciprocal_res_func(Nrays, npoints1, npoints2, npoints3, 
                    qi1_range, qi2_range, qi3_range, 
                    plot_figs, save_resqi, zeta_v_fwhm, 
                    zeta_h_fwhm, NA_rms, eps_rms, theta, 
                    phys_aper, date, beamstop = BS_on,
                    bs_height=bs_height, return_qs = return_qs, 
                    aperture=True, knife_edge=False, dphi_range = 0)

# # Execute recspace_res.py script
# qrock, qroll, qpar, qrock_prime, q2th, Res_stack, delta_2theta = reciprocal_res_func(Nrays, npoints1, npoints2, npoints3, 
#                     qi1_range, qi2_range, qi3_range, 
#                     plot_figs, save_resqi, zeta_v_fwhm, 
#                     zeta_h_fwhm, NA_rms, eps_rms, theta, 
#                     phys_aper, date, beamstop = BS_on,
#                     bs_height=bs_height, return_qs = return_qs, 
#                     aperture=True, knife_edge=False, dphi_range = 0)

# qrock_prime_scan = (qrock_prime + dq[:, np.newaxis]).sum(axis=0)
# qrock_scan = (qrock + dq[:, np.newaxis]).sum(axis=0)

# qrock_scan = (qrock+dq[:,np.newaxis])


# Output will be Resq_i as a (npoints1, npoints2*npoints3) array .csv file
# If plot_figs = True and Nrays < 1000001 will also include figures
# exec(open('reciprocal space/recspace_res.py').read())
# save configs
vars = {'Nrays' : Nrays, 'npoints1': npoints1, 'npoints2' : npoints2, 'npoints3' : npoints3, 'qi1_range' : qi1_range, 'qi2_range' : qi2_range, 'qi3_range' : qi3_range, 'zeta_v_fwhm' : zeta_v_fwhm, 'zeta_h_fwhm' : zeta_h_fwhm, 'NA_rms' : NA_rms, 'eps_rms' : eps_rms, 'theta' : theta, 'D' : D, 'd1' : d1, 'phys_aper' : phys_aper}

with open('pkl_files/Resq_i_{0}_vars.txt'.format(date),'w') as data:
      data.write(str(vars))


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.ticker as mticker
# formatter = mticker.ScalarFormatter(useMathText=True)
# plot_half_range = 0.005



# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.ticker as mticker

# formatter = mticker.ScalarFormatter(useMathText=True)
# plot_half_range = 0.005
# if eps_rms > 1e-3:
#       plot_half_range = 0.03
# plt.rc('font', size=24)

# x1, y1, z1 = qrock, qroll, qpar


# fig1 = plt.figure(figsize=(10, 10))
# ax1 = fig1.add_subplot(111, projection='3d')

# # Projections — return artists so we can rasterize them too

# p_xy = ax1.plot(x1, y1, 'o', color = 'darkviolet', markersize=0.1, alpha=0.8, zdir='z', zs=-plot_half_range)[0]
# p_xz = ax1.plot(x1, z1, 'o', color = 'gold', markersize=0.1, alpha=0.8, zdir='y', zs= plot_half_range)[0]
# p_yz = ax1.plot(y1, z1, 'o', color = 'red', markersize=0.1, alpha=0.8, zdir='x', zs= plot_half_range)[0]

# # Heavy point cloud — rasterize
# pts = ax1.scatter(x1, y1, z1, 'o', color = 'slateblue', s=0.1, alpha=0.8, rasterized=True)

# # Also rasterize the projection dots (they’re many points too)
# p_xy.set_rasterized(True)
# p_xz.set_rasterized(True)
# p_yz.set_rasterized(True)

# # Labels/formatting stay vector
# ax1.set_xlabel(r'$\hat{q}_{rock}$', labelpad=15)
# ax1.set_ylabel(r'$\hat{q}_{roll}$', labelpad=15)
# ax1.set_zlabel(r'$\hat{q}_{par}$', labelpad=15)
# ax1.xaxis.set_major_formatter(formatter)
# ax1.yaxis.set_major_formatter(formatter)
# ax1.zaxis.set_major_formatter(formatter)
# ax1.ticklabel_format(style='sci', scilimits=(-3, -3))

# ax1.set_xlim([-plot_half_range, plot_half_range])
# ax1.set_ylim([ plot_half_range, -plot_half_range])
# ax1.set_zlim([ plot_half_range, -plot_half_range])
# ax1.view_init(200, 310)
# plt.tight_layout()
# # SVG with rasterized layers embedded; dpi controls bitmap resolution of those layers
# # plt.savefig('BFP_aperture_25mu.svg', dpi=300)  # try 200–600 depending on print needs
# plt.show()