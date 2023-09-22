import numpy as np
import sys, os, gc, pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from functions import (fast_inverse2, load_or_generate_Hg, rotatedU)


fast_inverse2(np.random.random(size = (100,3,3))); # DO NOT OUTCOMMENT, this line jit compiles "fast_inverse2" function so performance on larger arrays are obtained
# INPUT instrumental settings, related to direct space resolution function
psize = 40E-9 # pixel size in units of m, in the object plane
zl_rms = 0.15E-6/2.35  # rms value of Gaussian beam profile, in m, centered at 0
theta_0 = 17.953/2*np.pi/180 # in rad
# input reciprocal space resolution function (in the imaging system)
# by loading an already generated version Resq_i and insertin q_i-ranges and steps here

# INPUT FOV
Npixels = 510 # nr of pixels on detector (same in both y and z) - sets the FOV.
Nsub = 2     # NN1^3 = (Nsub*Npixels)^3 is the total number of "rays" probed
NN1 = int(Npixels//3*Nsub)# 3 is used as 1/sin(2*~18 deg) = 3.24
NN2 = int(Npixels*Nsub)
NN3 = int(Npixels//30*Nsub)

# Choose sys.path[0] or sys.path[1] depending on parent folder
# Define the file paths for reciprocal array
pkl_fpath = sys.path[0]+'/reciprocal_space/pkl_files/'
pkl_fn = 'Resq_i_20230913_1308.pkl' # Change accordingly
vars_fn = os.path.splitext(pkl_fn)[0] + '_vars.txt'
print('Loading Resq_i.')
# Load the pickle file
with open(os.path.join(pkl_fpath, pkl_fn), 'rb') as pkl_file:
    Resq_i = pickle.load(pkl_file)

# Load the instrumental variables file
with open(os.path.join(pkl_fpath, vars_fn), 'r') as vars_file:
    var_d = eval(vars_file.read())

# Get the intstrumental variables from the dictionary
qi1_range, npoints1 = var_d['qi1_range'], var_d['npoints1']
qi2_range, npoints2 = var_d['qi2_range'], var_d['npoints2']
qi3_range, npoints3 = var_d['qi3_range'], var_d['npoints3']
print('Resq_i loaded.')


theta = theta_0
yl_start = (-psize * Npixels / 2 + psize / (2 * Nsub))  # start in yl direction, in unit of m, centered at 0
xl_start = yl_start / np.tan(2 * theta) / 3 # start in xl direction, in m, for zl=0
zl_start = -0.5 * zl_rms * 6  # start in zl direction, in m, for zl=0

qi1_start, qi1_step = -qi1_range / 2, qi1_range / (npoints1 - 1)
qi2_start, qi2_step = -qi2_range / 2, qi2_range / (npoints2 - 1)
qi3_start, qi3_step = -qi3_range / 2, qi3_range / (npoints3 - 1)
qi_starts = np.asarray([qi1_start, qi2_start, qi3_start])
qi_steps = 1 / np.asarray([qi1_step, qi2_step, qi3_step])

# CREATE MATRICES according to Eqs 2,3,7:
Ud = np.array([[1 / np.sqrt(2), 1 / np.sqrt(3), 1 / np.sqrt(6)],
               [-1 / np.sqrt(2), 1 / np.sqrt(3), 1 / np.sqrt(6)],
               [0, -1 / np.sqrt(3), 2 / np.sqrt(6)]])
Us = np.array([[1 / np.sqrt(2), -1 / np.sqrt(6), -1 / np.sqrt(3)],
               [0, -2 / np.sqrt(6), 1 / np.sqrt(3)],
               [-1 / np.sqrt(2), -1 / np.sqrt(6), -1 / np.sqrt(3)]]).T
Theta = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])



YI = (np.arange(NN1)//Nsub).repeat(NN3*NN2)
ZI = np.tile((np.arange(NN2)//Nsub).repeat(NN3),NN1)
indices = np.vstack((ZI, YI)).T


xl_range, xl_steps = -xl_start, NN1
yl_range, yl_steps = -yl_start, NN2
zl_range, zl_steps = -zl_start, NN3
rl = np.vstack(np.mgrid[-xl_range:xl_range:complex(xl_steps), 
                         -yl_range:yl_range:complex(yl_steps),
                         -zl_range:zl_range:complex(zl_steps)]).reshape(3,-1)





prob_z = np.exp(-0.5*(rl[2]/zl_rms)**2)

# To avoid edge effects:
# for dis 0.25, ndis >= 7501
# for dis 0.5, ndis >= 1151
# for dis 1, ndis >= 501 
# for dis 2, ndis >= 251
# for dis 4, ndis >= 151


ndis = 151 # number of dislocations
dis = 4 # units of micrometer
def Find_Hg(dis, ndis, psize, zl_rms, I = np.identity(3), h=-1, k=1, l=-1):
    Q_norm = np.sqrt(h * h + k * k + l * l) # We have assumed B_0 = I
    q_hkl = np.asarray([h, k, l]) / Q_norm

    Fg_path = sys.path[0]+'/direct_space/deformation_gradient_tensors/Fg_{0}_{1}nm_{2}nm.npy'.format(str(dis).replace('.', ''), int(psize*1e9), int(zl_rms*2.35e9))
    Hg = load_or_generate_Hg(rl, Ud, Us, Theta, dis, ndis, Fg_path)

    if not os.path.exists(Fg_path.replace('.npy', '_vars.txt')):
        vars = {
            'Resq_i': pkl_fn, 'psize [nm]': psize, 'zl_rms': zl_rms,
            'theta_0 [rad]': theta_0, 'Npixels': Npixels, 'Nsub': Nsub,
            'Ud': Ud.tolist(), 'Us': Us.tolist(), 'Theta': Theta.tolist(),
            'ndis': ndis, 'dis [micrometer]': dis, 'q_hkl': q_hkl.tolist()}

        with open(Fg_path.replace('.npy', '_vars.txt'), 'w') as data:
            for key, value in vars.items():
                # Use pprint for matrices to format them nicely
                if isinstance(value, list) and isinstance(value[0], list):
                    data.write(f"{key}:\n")
                    pprint(value, data)
                    data.write("\n")
                else:
                    data.write(f"{key}: {value}\n\n")
    return Hg, q_hkl

Hg, q_hkl = Find_Hg(dis, ndis, psize, zl_rms)

def forward(Hg, phi = 0, chi = 0, TwoDeltaTheta = 0, qi_return = False):
    '''
    This function calculates the forward model image for a given set of angles on
    the goniometer holding the sample, this is based on equations from a paper. 
    It calculates the scattering vector in different spaces and uses it to generate 
    a probability distribution, which is added to the forward model image.
    -----------------------------------------------------------------------------
    Parameters:
    phi, chi, TwoDeltaTheta: float.
        The radians off the Bragg condition in each rotation stage.
    
    Returns
    -----------------------------------------------------------------------------
    numpy.ndarray, numpy.ndarray
        The scattering vector in imaging space, shape(3,X), X being NN1 * NN2 * NN3
        The forward modelled image, shape (NN3,NN1)

    '''
    if TwoDeltaTheta != 0:
        theta = theta_0 + TwoDeltaTheta
        Theta = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    else:
        Theta = np.array([[np.cos(theta_0), 0, np.sin(theta_0)],
                  [0, 1, 0],
                  [-np.sin(theta_0), 0, np.cos(theta_0)]])

    # Initialize forward model image with zeros
    im_1 = np.zeros([(NN2//Nsub), NN1//Nsub])
    
    # Define angles
    ang_arr = np.asarray([[phi - TwoDeltaTheta/2], [chi], [(TwoDeltaTheta/2)/np.tan(theta_0)]])
    
    # Calculate scattering vector in sample space
    # qs = np.zeros_like(rl.T)
    # qs[negative_indices] = U_sr @ Hg[negative_indices] @ q_hkl # eq. 20 in GOF
    # qs[positive_indices] = U_sl @ Hg[positive_indices] @ q_hkl # eq. 20 in GOF
    qs = Us @ Hg @ q_hkl
    
    # Calculate scattering vector in crystal/grain space
    qc = qs.squeeze().T + ang_arr

    
    
    # Calculate scattering vector in imaging space and reshape it
    qi = Theta @ qc
    qi_field = qi.reshape(3,NN1,NN2,NN3)
    
    # Calculate indices for Resq_i that pass through the mask
    index1 = np.floor( (qi[0] - qi1_start)/qi1_step).astype(np.int16)
    index2 = np.floor( (qi[1] - qi2_start)/qi2_step).astype(np.int16)
    index3 = np.floor( (qi[2] - qi3_start)/qi3_step).astype(np.int16)
    
    # Calculate index of values within bounds and extract corresponding values from Resq_i
    idx = (index3 >=0)*(index2 >=0)*(index1 >=0)*(index1 < npoints1)*(index2 < npoints2)*(index3 < npoints3)
    prob = Resq_i[index1[idx],index2[idx],index3[idx]] * prob_z[idx]
    
    # Initialize array for probability distribution
    pro = np.zeros((NN1*NN2*NN3), dtype = np.float32)
    
    # Assign values to the probability distribution array for valid indices
    pro[idx] = prob
    
    # Add probability distribution to forward model image
    np.add.at(im_1, tuple(indices.T), pro)
    if qi_return == True:
        qi_field = qi.reshape(3,NN1,NN2,NN3)
        # Return scattering vector in imaging space and forward model image
        return im_1, qi_field
    else:
        # Return the forward model image
        return im_1
