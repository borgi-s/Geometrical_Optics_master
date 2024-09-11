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
qi1_range, qi2_range, qi3_range = 1E-2, 5E-3, 5E-2
plot_figs = False
save_resqi = True

# Instrumental parameters
zeta_v_fwhm = 5.3E-04 # incoming divergence in vertical direction, in rad
zeta_h_fwhm = 0 # 1e-05 # incoming divergence in horizontal direction, in rad
NA_rms = 7.31E-4/2.35 # NA of objective, in rad
eps_rms = 1.41e-4/2.35 # rms widht of x-ray energy bandwidth
theta = (17.953/2)*(np.pi/180) # scattering angle, in rad
D = 2*np.sqrt(50E-6*1.6E-3) # physical aperture of objective, in mx_cond
d1 = 0.274 # sample-objective distance, in m
phys_aper = D/d1

# Execute recspace_res.py script
reciprocal_res_func(Nrays, npoints1, npoints2, npoints3, 
                    qi1_range, qi2_range, qi3_range, 
                    plot_figs, save_resqi, zeta_v_fwhm, 
                    zeta_h_fwhm, NA_rms, eps_rms, theta, 
                    phys_aper, date)



# Output will be Resq_i as a (npoints1, npoints2*npoints3) array .csv file
# If plot_figs = True and Nrays < 1000001 will also include figures
# exec(open('reciprocal space/recspace_res.py').read())
# save configs
vars = {'Nrays' : Nrays, 'npoints1': npoints1, 'npoints2' : npoints2, 'npoints3' : npoints3, 'qi1_range' : qi1_range, 'qi2_range' : qi2_range, 'qi3_range' : qi3_range, 'zeta_v_fwhm' : zeta_v_fwhm, 'zeta_h_fwhm' : zeta_h_fwhm, 'NA_rms' : NA_rms, 'eps_rms' : eps_rms, 'theta' : theta, 'D' : D, 'd1' : d1, 'phys_aper' : phys_aper}

with open('pkl_files/Resq_i_{0}_vars.txt'.format(date),'w') as data:
      data.write(str(vars))
