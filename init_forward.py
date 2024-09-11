# Import necessary libraries
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import center_of_mass, label
from scipy.interpolate import griddata
from direct_space.forward_model import *
from concurrent.futures import ThreadPoolExecutor
from image_processor import *
from matplotlib.colors import ListedColormap
from matplotlib.ticker import ScalarFormatter

# Define the range and number of steps for phi and chi
phi_range, phi_steps = 0.0004*180/np.pi, 61
chi_range, chi_steps = 0.001*180/np.pi, 61

# Define the filepath and filename prefix for the output images
fpath = r'/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/borgi/Purdue-Paper1/Images/Test/Images5_perf'
fn_prefix = r'/mosa_test_0000_'
ftype = ".npy"

# Define the filepath and filename prefix for the output images
fpath1 = r'/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/borgi/Purdue-Paper1/Images/Test/Images5'
fn_prefix1 = r'/mosa_test_0000_'
ftype1 = ".npy"

# Generate arrays for phi and chi values
Phi = np.linspace(-phi_range, phi_range, phi_steps)
Chi = np.linspace(-chi_range, chi_range, chi_steps)

Hg = np.load('Test_S1_Hg_rg.npy')


Hg.shape
Hg_reshaped = Hg_rot.reshape((136,410,13, 3, 3)).transpose([-1,-2, 0, 1, 2])

Us = np.array([[1/np.sqrt(2),  0,            -1/np.sqrt(2)],
              [-1/np.sqrt(6), -2/np.sqrt(6), -1/np.sqrt(6)],
              [-1/np.sqrt(3),  1/np.sqrt(3), -1/np.sqrt(3)]])
Theta = np.array([[np.cos(theta),  0, np.sin(theta)],
                  [0,              1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])

Hg_rot = Theta @ Us.T @ Hg @ Us @ Theta.T
Hg_reshaped = Hg_rot.reshape((136,410,13, 3, 3)).transpose([-1,-2, 0, 1, 2])

# # H11 = Hg_matlab[:,:,:,0,0]
# # H12 = Hg_matlab[:,:,:,0,1]
# # H13 = Hg_matlab[:,:,:,0,2]
# # H21 = Hg_matlab[:,:,:,1,0]
# # H22 = Hg_matlab[:,:,:,1,1]
# # H23 = Hg_matlab[:,:,:,1,2]
# # H31 = Hg_matlab[:,:,:,2,0]
# # H32 = Hg_matlab[:,:,:,2,1]
# # H33 = Hg_matlab[:,:,:,2,2]

H11 = Hg_reshaped[0,0]
H12 = Hg_reshaped[0,1]
H13 = Hg_reshaped[0,2]
H21 = Hg_reshaped[1,0]
H22 = Hg_reshaped[1,1]
H23 = Hg_reshaped[1,2]
H31 = Hg_reshaped[2,0]
H32 = Hg_reshaped[2,1]
H33 = Hg_reshaped[2,2]



# Create subplots
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
plt.suptitle('Hg (rl)')
# Plot each variable in its respective subplot
axs[0, 0].imshow(H11[:,:,6], vmin = -1E-4, vmax = 1E-4, cmap='viridis', interpolation='nearest',
                origin = 'lower', aspect = 'auto')
axs[0, 0].set_title('H11')

axs[0, 1].imshow(H12[:,:,6], vmin = -1E-4, vmax = 1E-4, cmap='viridis', interpolation='nearest',
                origin = 'lower', aspect = 'auto')
axs[0, 1].set_title('H12')

axs[0, 2].imshow(H13[:,:,6], vmin = -1E-4, vmax = 1E-4, cmap='viridis', interpolation='nearest',
                origin = 'lower', aspect = 'auto')
axs[0, 2].set_title('H13')

axs[1, 0].imshow(H21[:,:,6], vmin = -1E-4, vmax = 1E-4, cmap='viridis', interpolation='nearest',
                origin = 'lower', aspect = 'auto')
axs[1, 0].set_title('H21')

axs[1, 1].imshow(H22[:,:,6], vmin = -1E-4, vmax = 1E-4, cmap='viridis', interpolation='nearest',
                origin = 'lower', aspect = 'auto')
axs[1, 1].set_title('H22')

axs[1, 2].imshow(H23[:,:,6], vmin = -1E-4, vmax = 1E-4, cmap='viridis', interpolation='nearest',
                origin = 'lower', aspect = 'auto')
axs[1, 2].set_title('H23')

axs[2, 0].imshow(H31[:,:,6], vmin = -1E-4, vmax = 1E-4, cmap='viridis', interpolation='nearest',
                origin = 'lower', aspect = 'auto')
axs[2, 0].set_title('H31')

axs[2, 1].imshow(H32[:,:,6], vmin = -1E-4, vmax = 1E-4, cmap='viridis', interpolation='nearest',
                origin = 'lower', aspect = 'auto')
axs[2, 1].set_title('H32')

axs[2, 2].imshow(H33[:,:,6], vmin = -1E-4, vmax = 1E-4, cmap='viridis', interpolation='nearest',
                origin = 'lower', aspect = 'auto')
axs[2, 2].set_title('H33')

# Adjust layout to prevent clipping of titles
plt.tight_layout()
plt.show()




# Create subplots
fig, axs = plt.subplots(3, 3, figsize=(10, 8))
plt.suptitle('Hg (rg) - Input -> Geo. Opt.')

# Plot each variable in its respective subplot
im = axs[0, 0].imshow(H11[:,:,6], vmin=-1E-4, vmax=1E-4, cmap='viridis', interpolation='nearest',
                      origin='lower', aspect='equal', extent=[Y.min(), Y.max(), X.min(), X.max()])
axs[0, 0].set_title('H11')
axs[0, 0].set_xlabel('$y_{g}$ ($\mu$m)')
axs[0, 0].set_ylabel('$x_{g}$ ($\mu$m)')
axs[0, 0].set_xlim((-5, 5))
axs[0, 0].set_ylim((-5, 5))
cbar = fig.colorbar(im, ax=axs[0, 0])
cbar.formatter.set_powerlimits((-3, 3))
cbar.update_ticks()

im = axs[0, 1].imshow(H12[:,:,6], vmin=-1E-4, vmax=1E-4, cmap='viridis', interpolation='nearest',
                      origin='lower', aspect='equal', extent=[Y.min(), Y.max(), X.min(), X.max()])
axs[0, 1].set_title('H12')
axs[0, 1].set_xlabel('$y_{g}$ ($\mu$m)')
axs[0, 1].set_ylabel('$x_{g}$ ($\mu$m)')
axs[0, 1].set_xlim((-5, 5))
axs[0, 1].set_ylim((-5, 5))
cbar = fig.colorbar(im, ax=axs[0, 1])
cbar.formatter.set_powerlimits((-3, 3))
cbar.update_ticks()

im = axs[0, 2].imshow(H13[:,:,6], vmin=-1E-4, vmax=1E-4, cmap='viridis', interpolation='nearest',
                      origin='lower', aspect='equal', extent=[Y.min(), Y.max(), X.min(), X.max()])
axs[0, 2].set_title('H13')
axs[0, 2].set_xlabel('$y_{g}$ ($\mu$m)')
axs[0, 2].set_ylabel('$x_{g}$ ($\mu$m)')
axs[0, 2].set_xlim((-5, 5))
axs[0, 2].set_ylim((-5, 5))
cbar = fig.colorbar(im, ax=axs[0, 2])
cbar.formatter.set_powerlimits((-3, 3))
cbar.update_ticks()

im = axs[1, 0].imshow(H21[:,:,6], vmin=-1E-4, vmax=1E-4, cmap='viridis', interpolation='nearest',
                      origin='lower', aspect='equal', extent=[Y.min(), Y.max(), X.min(), X.max()])
axs[1, 0].set_title('H21')
axs[1, 0].set_xlabel('$y_{g}$ ($\mu$m)')
axs[1, 0].set_ylabel('$x_{g}$ ($\mu$m)')
axs[1, 0].set_xlim((-5, 5))
axs[1, 0].set_ylim((-5, 5))
cbar = fig.colorbar(im, ax=axs[1, 0])
cbar.formatter.set_powerlimits((-3, 3))
cbar.update_ticks()

im = axs[1, 1].imshow(H22[:,:,6], vmin=-1E-4, vmax=1E-4, cmap='viridis', interpolation='nearest',
                      origin='lower', aspect='equal', extent=[Y.min(), Y.max(), X.min(), X.max()])
axs[1, 1].set_title('H22')
axs[1, 1].set_xlabel('$y_{g}$ ($\mu$m)')
axs[1, 1].set_ylabel('$x_{g}$ ($\mu$m)')
axs[1, 1].set_xlim((-5, 5))
axs[1, 1].set_ylim((-5, 5))
cbar = fig.colorbar(im, ax=axs[1, 1])
cbar.formatter.set_powerlimits((-3, 3))
cbar.update_ticks()

im = axs[1, 2].imshow(H23[:,:,6], vmin=-1E-4, vmax=1E-4, cmap='viridis', interpolation='nearest',
                      origin='lower', aspect='equal', extent=[Y.min(), Y.max(), X.min(), X.max()])
axs[1, 2].set_title('H23')
axs[1, 2].set_xlabel('$y_{g}$ ($\mu$m)')
axs[1, 2].set_ylabel('$x_{g}$ ($\mu$m)')
axs[1, 2].set_xlim((-5, 5))
axs[1, 2].set_ylim((-5, 5))
cbar = fig.colorbar(im, ax=axs[1, 2])
cbar.formatter.set_powerlimits((-3, 3))
cbar.update_ticks()

im = axs[2, 0].imshow(H31[:,:,6], vmin=-1E-4, vmax=1E-4, cmap='viridis', interpolation='nearest',
                      origin='lower', aspect='equal', extent=[Y.min(), Y.max(), X.min(), X.max()])
axs[2, 0].set_title('H31')
axs[2, 0].set_xlabel('$y_{g}$ ($\mu$m)')
axs[2, 0].set_ylabel('$x_{g}$ ($\mu$m)')
axs[2, 0].set_xlim((-5, 5))
axs[2, 0].set_ylim((-5, 5))
cbar = fig.colorbar(im, ax=axs[2, 0])
cbar.formatter.set_powerlimits((-3, 3))
cbar.update_ticks()

im = axs[2, 1].imshow(H32[:,:,6], vmin=-1E-4, vmax=1E-4, cmap='viridis', interpolation='nearest',
                      origin='lower', aspect='equal', extent=[Y.min(), Y.max(), X.min(), X.max()])
axs[2, 1].set_title('H32')
axs[2, 1].set_xlabel('$y_{g}$ ($\mu$m)')
axs[2, 1].set_ylabel('$x_{g}$ ($\mu$m)')
axs[2, 1].set_xlim((-5, 5))
axs[2, 1].set_ylim((-5, 5))
cbar = fig.colorbar(im, ax=axs[2, 1])
cbar.formatter.set_powerlimits((-3, 3))
cbar.update_ticks()

im = axs[2, 2].imshow(H33[:,:,6], vmin=-1E-4, vmax=1E-4, cmap='viridis', interpolation='nearest',
                      origin='lower', aspect='equal', extent=[Y.min(), Y.max(), X.min(), X.max()])
axs[2, 2].set_title('H33')
axs[2, 2].set_xlabel('$y_{g}$ ($\mu$m)')
axs[2, 2].set_ylabel('$x_{g}$ ($\mu$m)')
axs[2, 2].set_xlim((-5, 5))
axs[2, 2].set_ylim((-5, 5))
cbar = fig.colorbar(im, ax=axs[2, 2])
cbar.formatter.set_powerlimits((-3, 3))
cbar.update_ticks()

# Adjust layout to prevent clipping of titles
plt.tight_layout()
plt.show()
















# import scipy.io

# # Load .mat file
# mat_data = scipy.io.loadmat('TilSina.mat')

# # Access variables in the loaded data
# variable_name = 'your_variable_name'
# your_variable = mat_data[variable_name]

# Now 'your_variable' contains the data from the specified variable in the .mat file





# # Out comment to save images of new results
# save_images(phi_range, phi_steps, 
#             chi_range, chi_steps, 
#             fpath, fn_prefix, ftype)

# # Out comment to save many images of new results
# save_images_parallel(np.zeros_like(Hg), phi_range, phi_steps, 
#                      chi_range, chi_steps, 
#                      fpath, fn_prefix, ftype)


# # Out comment to save many images of new results
# save_images_parallel(Hg, phi_range, phi_steps, 
#                      chi_range, chi_steps, 
#                      fpath1, fn_prefix1, ftype1)



zl_steps1 = np.round(np.linspace(5e-6,-5e-6, 51),9)
for i, j in enumerate(zl_steps1):
        # Define the filepath and filename prefix for the output images
        fpath1 = r'/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/borgi/Purdue-Paper1/Raw_Images/Testing_S3/Layers/layer' + '{0}'.format(i).zfill(4)
        fn_prefix1 = r'/mosa_scan_' + '{0}_'.format(i).zfill(5)
        ftype1 = ".npy"
        
        fpath_phi_com = r'/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/borgi/Purdue-Paper1/COM_maps/Testing_S3/Phi_COM_map'
        fpath_chi_com = r'/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/borgi/Purdue-Paper1/COM_maps/Testing_S3/Chi_COM_map'
        fname_com = r'/com_map_run3_' '{0}'.format(i).zfill(4)
        ftype_com = ".npy"

        if i == 0:
                fpath = r'/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/borgi/Purdue-Paper1/Raw_Images/Testing_S3/Layers/clean_sample'
                fn_prefix = r'/mosa_scan_' + '{0}_'.format(i).zfill(5)
                ftype = ".npy"
                # Out comment to save many images of new results
                # save_images_parallel(np.zeros_like(Hg), phi_range, phi_steps, chi_range, chi_steps, fpath, fn_prefix, ftype)
        
        if i >= 46:
                Hg, q_hkl, prob_z = Find_Hg(xl_start, yl_start, zl_start, dis, ndis, psize, zl_rms, offset=j)
                # Out comment to save many images of new results
                save_images_parallel(Hg, phi_range, phi_steps, 
                                chi_range, chi_steps, 
                                fpath1, fn_prefix1, ftype1)

                stack_no_dislocs, stack_reshape_no_dislocs, dim_1_no_dislocs, dim_2_no_dislocs = load_images(fpath, phi_steps, chi_steps)
                stack_dislocs, stack_reshape_dislocs, dim_1_dislocs, dim_2_dislocs = load_images(fpath1, phi_steps, chi_steps)
                
                pic = stack_dislocs.reshape((chi_steps, phi_steps, dim_1_dislocs, dim_2_dislocs))[chi_steps//2,phi_steps//2]
                reshaped_stack = stack_dislocs.reshape((chi_steps, phi_steps, dim_1_dislocs, dim_2_dislocs))
                reshaped_stack_clean = stack_no_dislocs.reshape((chi_steps, phi_steps, dim_1_no_dislocs, dim_2_no_dislocs))

                # Going to higher resolution, Chi can shift. Here it is corrected.
                com = center_of_mass(reshaped_stack_clean[:,:,-1,-1])

                Chi_high = np.linspace(-chi_range,chi_range,chi_steps*100)
                shift = com[0]*100 - (chi_steps*100 / 2)
                shift_rads = Chi_high[int(abs(shift))]-Chi_high[0]

                shifted_Chi = np.linspace(-chi_range + shift_rads, chi_range+ shift_rads, chi_steps)


                Chi_high = np.linspace(-chi_range,chi_range,chi_steps*100)


                phi_list = np.zeros((dim_1_dislocs, dim_2_dislocs))
                chi_list = np.zeros((dim_1_dislocs, dim_2_dislocs))
                shifted_Chi = np.deg2rad(np.linspace(-chi_range+ shift_rads,chi_range+shift_rads,chi_steps*100))
                Phi_high = np.deg2rad(np.linspace(-phi_range,phi_range,phi_steps*100))

                for k in tqdm(range(reshaped_stack.shape[2])):
                        for l in range(reshaped_stack.shape[3]):
                                if reshaped_stack[:,:,k,l].sum() == 0.0:
                                        phi_list[k,l] = np.nan
                                        chi_list[k,l] = np.nan
                                else:
                                        x_ind, y_ind = center_of_mass(reshaped_stack[:,:,k,l])
                                        phi_list[k,l] = Phi_high[np.round(y_ind*100).astype(int)]
                                        chi_list[k,l] = shifted_Chi[np.round(x_ind*100).astype(int)]
                check_folder('', fpath_phi_com)
                check_folder('', fpath_chi_com)
                np.save(fpath_phi_com+fname_com+'_phi'+ftype_com, phi_list)
                np.save(fpath_chi_com+fname_com+'_chi'+ftype_com, chi_list)

'''
plt.title('Angular spread of example pixel')
plt.imshow(stack_no_dislocs.reshape((chi_steps, phi_steps, dim_1_no_dislocs, dim_2_no_dislocs))[:,:,dim_1_no_dislocs//2-2,dim_2_no_dislocs//2-2], origin='lower', aspect='auto',
           extent = [-np.deg2rad(phi_range), np.deg2rad(phi_range), 
                     -np.deg2rad(chi_range), np.deg2rad(chi_range)])
plt.colorbar()
plt.xlabel(f'$\phi$ [radians]')
plt.ylabel(f'$\chi$ [radians]')
plt.show()

plt.title('Angular spread of example pixel')
plt.imshow(stack_dislocs.reshape((chi_steps, phi_steps, dim_1_dislocs, dim_2_dislocs))[:,:,dim_1_dislocs//2-2,dim_2_dislocs//2-2], origin='lower', aspect='auto',
           extent = [-np.deg2rad(phi_range), np.deg2rad(phi_range), 
                     -np.deg2rad(chi_range), np.deg2rad(chi_range)])
plt.colorbar()
plt.xlabel(f'$\phi$ [radians]')
plt.ylabel(f'$\chi$ [radians]')
plt.show()




pic = stack_dislocs.reshape((chi_steps, phi_steps, dim_1_dislocs, dim_2_dislocs))[chi_steps//2,phi_steps//2]
reshaped_stack = stack_dislocs.reshape((chi_steps, phi_steps, dim_1_dislocs, dim_2_dislocs))
reshaped_stack_clean = stack_no_dislocs.reshape((chi_steps, phi_steps, dim_1_no_dislocs, dim_2_no_dislocs))

# Going to higher resolution, Chi can shift. Here it is corrected.
com = center_of_mass(reshaped_stack_clean[:,:,-1,-1])
plt.imshow(reshaped_stack_clean[:,:,-1,-1].T,)
plt.scatter(*com, label = com)
plt.legend()
Chi_high = np.linspace(-chi_range,chi_range,chi_steps*100)
shift = com[0]*100 - (chi_steps*100 / 2)
shift_rads = Chi_high[int(abs(shift))]-Chi_high[0]

shifted_Chi = np.linspace(-chi_range+ shift_rads,chi_range+shift_rads,chi_steps)



com = center_of_mass(reshaped_stack_clean[:,:,10,10])
x, y = np.deg2rad(Phi[int(com[1])]), np.deg2rad(Chi[int(com[0])])

fig, axs = plt.subplots(1, 2, figsize=(10, 4))


axs[0].imshow(reshaped_stack[chi_steps//2, phi_steps//2].T, aspect='auto', origin='lower')
axs[0].scatter(373,84, s=3, c='red')

axs[1].imshow(reshaped_stack_clean[:,:,10,10], aspect=1/4, origin='lower',
extent = [-np.deg2rad(phi_range), np.deg2rad(phi_range), 
                     -np.deg2rad(chi_range), np.deg2rad(chi_range)])
axs[1].scatter(x,y, s=3, c='red', label = np.round((x,y),8))
axs[1].set_xlabel(f'$\phi$ [radians]')
axs[1].set_ylabel(f'$\chi$ [radians]')
axs[1].legend()
plt.tight_layout()
plt.show()


Chi_high = np.linspace(-chi_range,chi_range,chi_steps*100)


phi_list = np.zeros((dim_1_dislocs, dim_2_dislocs))
chi_list = np.zeros((dim_1_dislocs, dim_2_dislocs))
shifted_Chi = np.deg2rad(np.linspace(-chi_range+ shift_rads,chi_range+shift_rads,chi_steps*100))
Phi_high = np.deg2rad(np.linspace(-phi_range,phi_range,phi_steps*100))
for i in tqdm(range(reshaped_stack.shape[2])):
    for j in range(reshaped_stack.shape[3]):
        if reshaped_stack[:,:,i,j].sum() == 0.0:
            phi_list[i,j] = np.nan
            chi_list[i,j] = np.nan
        else:
            x_ind, y_ind = center_of_mass(reshaped_stack[:,:,i,j])
            phi_list[i,j] = Phi_high[np.round(y_ind*100).astype(int)]
            chi_list[i,j] = shifted_Chi[np.round(x_ind*100).astype(int)]


# Plot of two components of qi in (x, y, z=0) plane
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Plot qi_1
im1 = axs[0].imshow((phi_list*-1).T, interpolation='none',
                    extent=[xl_start*1e6, -xl_start*1e6,
                            yl_start*1e6, -yl_start*1e6],
                    vmin = -1e-4, vmax = 1e-4, cmap='viridis', origin = 'lower')
axs[0].set_aspect('equal')
axs[0].set_title('Extreme Phi')
axs[0].set_xlabel('$y_{\ell}$ ($\mu$m)', fontsize=12)
axs[0].set_ylabel('$x_{\ell}$ ($\mu$m)', fontsize=12)
# axs[0].invert_yaxis()  # To match 'set(gca,'ydir','normal')' in MATLAB
axs[0].grid(False)

# Customize colorbar with scientific notation
cbar1 = fig.colorbar(im1, ax=axs[0], format=ScalarFormatter(useMathText=True))
cbar1.formatter.set_powerlimits((-2, 2))  # Adjust the power limits as needed
cbar1.update_ticks()

# Plot qi_2
im2 = axs[1].imshow(chi_list.T*-1, 
                    extent=[xl_start*1e6, -xl_start*1e6,
                            yl_start*1e6, -yl_start*1e6],
                    interpolation = 'none',
                    vmin = -1e-4, vmax = 1e-4, cmap='viridis', origin = 'lower')
axs[1].set_aspect('equal')
axs[1].set_title('Extreme Chi')
axs[1].set_xlabel('$y_{\ell}$ ($\mu$m)', fontsize=12)
axs[1].set_ylabel('$x_{\ell}$ ($\mu$m)', fontsize=12)
# axs[1].invert_yaxis()  # To match 'set(gca,'ydir','normal')' in MATLAB
axs[1].grid(False)

# Customize colorbar with scientific notation
cbar2 = fig.colorbar(im2, ax=axs[1], format=ScalarFormatter(useMathText=True))
cbar2.formatter.set_powerlimits((-2, 2))  # Adjust the power limits as needed
cbar2.update_ticks()

plt.tight_layout()
plt.savefig('extrem_phi+chi2.svg')

'''

xl_range, xl_steps = -xl_start, NN1
yl_range, yl_steps = -yl_start, NN2
zl_range, zl_steps = -zl_start, NN3
rl = np.vstack(np.mgrid[-xl_range:xl_range:complex(xl_steps), 
                        -yl_range:yl_range:complex(yl_steps),
                        -zl_range:zl_range:complex(zl_steps)]).reshape(3,-1)

imp, qi_fieldp = forward(Hg, phi = 0, qi_return=True)

X = np.linspace(-xl_start, xl_start, xl_steps) * 1E6  # rulers on the x-axis in µm
Y = np.linspace(-yl_start, yl_start, yl_steps) * 1E6  # rulers on the y-axis in µm
Z = np.linspace(-zl_start, zl_start, zl_steps) * 1E6  # rulers on the z-axis in µm

# # Plot of two components of qi in (x, y, z=0) plane
# fig, axs = plt.subplots(1, 3, figsize=(11, 3))

# # Plot qi_1
# im1 = axs[0].imshow(qi_fieldp[0, :, :, zl_steps // 2].squeeze(), extent=[Y.min(), Y.max(), X.min(), X.max()],
#                     vmin=-1E-4, vmax=1E-4, cmap='viridis', origin = 'lower')
# # axs[0].set_xlim((-6,-4))
# # axs[0].set_ylim((1.5,3.5))
# axs[0].set_aspect('equal')
# axs[0].set_title('qi_1 for (x, y)\nplane, z=0')
# axs[0].set_xlabel('$y_{\ell}$ ($\mu$m)', fontsize=12)
# axs[0].set_ylabel('$x_{\ell}$ ($\mu$m)', fontsize=12)
# # axs[0].invert_yaxis()  # To match 'set(gca,'ydir','normal')' in MATLAB
# axs[0].grid(False)

# # Customize colorbar with scientific notation
# cbar1 = fig.colorbar(im1, ax=axs[0], format=ScalarFormatter(useMathText=True))
# cbar1.formatter.set_powerlimits((-2, 2))  # Adjust the power limits as needed
# cbar1.update_ticks()

# # Plot qi_2
# im2 = axs[1].imshow(qi_fieldp[1, :, :, zl_steps // 2].squeeze(), extent=[Y.min(), Y.max(), X.min(), X.max()],
#                     vmin=-1E-4, vmax=1E-4, cmap='viridis', origin = 'lower')
# axs[1].set_aspect('equal')
# axs[1].set_title('qi_2 for (x, y)\nplane, z=0')
# axs[1].set_xlabel('$y_{\ell}$ ($\mu$m)', fontsize=12)
# axs[1].set_ylabel('$x_{\ell}$ ($\mu$m)', fontsize=12)
# # axs[1].invert_yaxis()  # To match 'set(gca,'ydir','normal')' in MATLAB
# axs[1].grid(False)

# # Customize colorbar with scientific notation
# cbar2 = fig.colorbar(im2, ax=axs[1], format=ScalarFormatter(useMathText=True))
# cbar2.formatter.set_powerlimits((-2, 2))  # Adjust the power limits as needed
# cbar2.update_ticks()

# # Plot qi_3
# im2 = axs[2].imshow(qi_fieldp[2, :, :, zl_steps // 2].squeeze(), extent=[Y.min(), Y.max(), X.min(), X.max()],
#                     vmin=-1E-4, vmax=1E-4, cmap='viridis', origin = 'lower')
# axs[2].set_aspect('equal')
# axs[2].set_title('qi_3 for (x, y)\nplane, z=0')
# axs[2].set_xlabel('$y_{\ell}$ ($\mu$m)', fontsize=12)
# axs[2].set_ylabel('$x_{\ell}$ ($\mu$m)', fontsize=12)
# # axs[2].invert_yaxis()  # To match 'set(gca,'ydir','normal')' in MATLAB
# axs[2].grid(False)

# # Customize colorbar with scientific notation
# cbar2 = fig.colorbar(im2, ax=axs[2], format=ScalarFormatter(useMathText=True))
# cbar2.formatter.set_powerlimits((-2, 2))  # Adjust the power limits as needed
# cbar2.update_ticks()



# plt.tight_layout()
# plt.savefig('qi1+qi2_fields1.svg')












# phi_com = np.load('/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/borgi/Purdue-Paper1/COM_maps/Testing/Phi_COM_map/com_map_run3_25_phi.npy')
# chi_com = np.load('/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/borgi/Purdue-Paper1/COM_maps/Testing/Chi_COM_map/com_map_run3_25_chi.npy')

# Plot of two components of qi in (x, y, z=0) plane
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Plot qi_1
im1 = axs[0].imshow(phi_com.T*-1, interpolation='none',
                    extent=[xl_start*1e6, -xl_start*1e6,
                            yl_start*1e6, -yl_start*1e6],
                    vmin = -1e-4, vmax = 1e-4, cmap='viridis', origin = 'lower')
axs[0].set_aspect('equal')
axs[0].set_title('COM Phi')
axs[0].set_xlabel('$y_{\ell}$ ($\mu$m)', fontsize=12)
axs[0].set_ylabel('$x_{\ell}$ ($\mu$m)', fontsize=12)
# axs[0].invert_yaxis()  # To match 'set(gca,'ydir','normal')' in MATLAB
axs[0].grid(False)

# Customize colorbar with scientific notation
cbar1 = fig.colorbar(im1, ax=axs[0], format=ScalarFormatter(useMathText=True))
cbar1.formatter.set_powerlimits((-2, 2))  # Adjust the power limits as needed
cbar1.update_ticks()

# Plot qi_2
im2 = axs[1].imshow(chi_com.T*-1, 
                    extent=[xl_start*1e6, -xl_start*1e6,
                            yl_start*1e6, -yl_start*1e6],
                    interpolation = 'none',
                    vmin = -0.25e-4, vmax = 0.25e-4, cmap='viridis', origin = 'lower')
axs[1].set_aspect('equal')
axs[1].set_title('COM Chi')
axs[1].set_xlabel('$y_{\ell}$ ($\mu$m)', fontsize=12)
axs[1].set_ylabel('$x_{\ell}$ ($\mu$m)', fontsize=12)
# axs[1].invert_yaxis()  # To match 'set(gca,'ydir','normal')' in MATLAB
axs[1].grid(False)

# Customize colorbar with scientific notation
cbar2 = fig.colorbar(im2, ax=axs[1], format=ScalarFormatter(useMathText=True))
cbar2.formatter.set_powerlimits((-2, 2))  # Adjust the power limits as needed
cbar2.update_ticks()

plt.tight_layout()