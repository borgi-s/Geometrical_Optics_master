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
phi_range, phi_steps = 0.0006*180/np.pi, 61
chi_range, chi_steps = 0.002*180/np.pi, 61

# Define the filepath and filename prefix for the output images
fpath = r'C:\Users\borgi\Documents\Production\images10_perf_crystal'
fn_prefix = r'/mosa_test_0000_'
ftype = ".npy"

# Define the filepath and filename prefix for the output images
fpath1 = r'C:\Users\borgi\Documents\Production\images10'
fn_prefix1 = r'/mosa_test_0000_'
ftype1 = ".npy"

# Generate arrays for phi and chi values
Phi = np.linspace(-phi_range, phi_range, phi_steps)
Chi = np.linspace(-chi_range, chi_range, chi_steps)

# # Out comment to save images of new results
# save_images(phi_range, phi_steps, 
#             chi_range, chi_steps, 
#             fpath, fn_prefix, ftype)

# Out comment to save many images of new results
save_images_parallel(np.zeros_like(Hg), phi_range, phi_steps, 
                     chi_range, chi_steps, 
                     fpath, fn_prefix, ftype)

# Out comment to save many images of new results
save_images_parallel(Hg, phi_range, phi_steps, 
                     chi_range, chi_steps, 
                     fpath1, fn_prefix, ftype)

stack_no_dislocs, stack_reshape_no_dislocs, dim_1_no_dislocs, dim_2_no_dislocs = load_images(fpath, phi_steps, chi_steps)
stack_dislocs, stack_reshape_dislocs, dim_1_dislocs, dim_2_dislocs = load_images(fpath1, phi_steps, chi_steps)


plt.title('Angular spread of example pixel')
plt.imshow(stack_no_dislocs.reshape((chi_steps, phi_steps, dim_1_no_dislocs, dim_2_no_dislocs))[:,:,dim_1_no_dislocs//2,dim_2_no_dislocs//2], origin='lower', aspect='auto',
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




pic = stack_dislocs.reshape((chi_steps, phi_steps, dim_1_dislocs, dim_2_dislocs))[15,15]
reshaped_stack = stack_dislocs.reshape((chi_steps, phi_steps, dim_1_dislocs, dim_2_dislocs))
reshaped_stack_clean = stack_no_dislocs.reshape((chi_steps, phi_steps, dim_1_no_dislocs, dim_2_no_dislocs))
# reshaped_residuals = reshaped_stack-reshaped_stack_clean
# reshaped_residuals[reshaped_residuals<0] = 0
# stack_residuals = stack_dislocs-stack_no_dislocs


com = center_of_mass(reshaped_stack_clean[:,:,-1,-1])
plt.imshow(reshaped_stack_clean[:,:,-1,-1].T,)
plt.scatter(*com, label = com)
plt.legend()
Chi_high = np.linspace(-chi_range,chi_range,chi_steps*100)
shift = com[0]*100 - (chi_steps*100 / 2)  # Assuming data is 1D, so center_of_mass[0] gives the center position
shift_rads = Chi_high[int(abs(shift))]-Chi_high[0]

shifted_Chi = np.linspace(-chi_range+ shift_rads,chi_range+shift_rads,chi_steps)



com = center_of_mass(reshaped_stack[:,:,256,85])
x, y = np.deg2rad(Phi[int(com[1])]), np.deg2rad(Chi[int(com[0])])

fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # Create a figure with two subplots


axs[0].imshow(reshaped_stack[chi_steps//2, phi_steps//2].T, aspect='auto', origin='lower')
axs[0].scatter(256,85, s=3, c='red')

axs[1].imshow(reshaped_stack[:,:,256,85], aspect=1/4, origin='lower',
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
shifted_Chi = np.deg2rad(np.linspace(-chi_range+ shift_rads,chi_range+shift_rads,chi_steps*20))
Phi_high = np.deg2rad(np.linspace(-phi_range,phi_range,phi_steps*20))
for i in tqdm(range(reshaped_stack.shape[2])):
    for j in range(reshaped_stack.shape[3]):
        x_ind, y_ind = center_of_mass(reshaped_stack[:,:,i,j])
        phi_list[i,j] = Phi_high[np.round(y_ind*20).astype(int)]
        chi_list[i,j] = shifted_Chi[np.round(x_ind*20).astype(int)]




# Plot of three components of qi in (x, y, z=0) plane
fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # Create a figure with two subplots

# Plot qi_1
im1 = axs[0].imshow((phi_list*-1).T, interpolation='none',
                    extent=[xl_start, -xl_start,
                            yl_start, -yl_start],
                    vmin = -1e-4, vmax = 1e-4, cmap='viridis', origin = 'lower')
axs[0].set_aspect('equal')
axs[0].set_title('Extreme Phi')
axs[0].set_xlabel('$y_{\ell}$ ($\mu$m)', fontsize=12)
axs[0].set_ylabel('$x_{\ell}$ ($\mu$m)', fontsize=12)
# axs[0].invert_yaxis()  # To match 'set(gca,'ydir','normal')' in MATLAB
axs[0].grid(False)

# # Add contours for qi_1
# contour_levels1 = np.linspace(-3E-4, 3E-4, 10)  # Adjust the contour levels as needed
# contour1 = axs[0].contour(Y, X, qi_fieldp[0, :, :, zl_steps // 2].squeeze(), 
#                           levels = contour_levels1, colors='black')


# Customize colorbar with scientific notation
cbar1 = fig.colorbar(im1, ax=axs[0], format=ScalarFormatter(useMathText=True))
cbar1.formatter.set_powerlimits((-2, 2))  # Adjust the power limits as needed
cbar1.update_ticks()

# Plot qi_2
im2 = axs[1].imshow(chi_list.T*-1, 
                    extent=[xl_start, -xl_start,
                            yl_start, -yl_start],
                    interpolation = 'none',
                    vmin = -1e-4, vmax = 1e-4, cmap='viridis', origin = 'lower')
axs[1].set_aspect('equal')
axs[1].set_title('Extreme Chi')
axs[1].set_xlabel('$y_{\ell}$ ($\mu$m)', fontsize=12)
axs[1].set_ylabel('$x_{\ell}$ ($\mu$m)', fontsize=12)
# axs[1].invert_yaxis()  # To match 'set(gca,'ydir','normal')' in MATLAB
axs[1].grid(False)

# # Add contours for qi_2
# contour_levels2 = np.linspace(-3E-4, 3E-4, 10)  # Adjust the contour levels as needed
# contour2 = axs[1].contour(Y, X, qi_fieldp[1, :, :, zl_steps // 2].squeeze(), 
#                           levels = contour_levels2, colors='black')

# Customize colorbar with scientific notation
cbar2 = fig.colorbar(im2, ax=axs[1], format=ScalarFormatter(useMathText=True))
cbar2.formatter.set_powerlimits((-2, 2))  # Adjust the power limits as needed
cbar2.update_ticks()

plt.tight_layout()
plt.savefig('extrem_phi+chi2.svg')


imp, qi_fieldp = forward(Hg, phi = 0, qi_return=True)

X = np.linspace(-xl_start, xl_start, xl_steps) * 1E6  # rulers on the x-axis in µm
Y = np.linspace(-yl_start, yl_start, yl_steps) * 1E6  # rulers on the y-axis in µm
Z = np.linspace(-zl_start, zl_start, zl_steps) * 1E6  # rulers on the z-axis in µm

# Plot of three components of qi in (x, y, z=0) plane
fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # Create a figure with two subplots

# Plot qi_1
im1 = axs[0].imshow(qi_fieldp[0, :, :, zl_steps // 2].squeeze(), extent=[Y.min(), Y.max(), X.min(), X.max()],
                    vmin=-1E-4, vmax=1E-4, cmap='viridis', origin = 'lower')
# axs[0].set_xlim((-6,-4))
# axs[0].set_ylim((1.5,3.5))
axs[0].set_aspect('equal')
axs[0].set_title('qi_1 for (x, y) plane, z=0')
axs[0].set_xlabel('$y_{\ell}$ ($\mu$m)', fontsize=12)
axs[0].set_ylabel('$x_{\ell}$ ($\mu$m)', fontsize=12)
# axs[0].invert_yaxis()  # To match 'set(gca,'ydir','normal')' in MATLAB
axs[0].grid(False)

# # Add contours for qi_1
# contour_levels1 = np.linspace(-3E-4, 3E-4, 10)  # Adjust the contour levels as needed
# contour1 = axs[0].contour(Y, X, qi_fieldp[0, :, :, zl_steps // 2].squeeze(), 
#                           levels = contour_levels1, colors='black')


# Customize colorbar with scientific notation
cbar1 = fig.colorbar(im1, ax=axs[0], format=ScalarFormatter(useMathText=True))
cbar1.formatter.set_powerlimits((-2, 2))  # Adjust the power limits as needed
cbar1.update_ticks()

# Plot qi_2
im2 = axs[1].imshow(qi_fieldp[1, :, :, zl_steps // 2].squeeze(), extent=[Y.min(), Y.max(), X.min(), X.max()],
                    vmin=-1E-4, vmax=1E-4, cmap='viridis', origin = 'lower')
axs[1].set_aspect('equal')
axs[1].set_title('qi_2 for (x, y) plane, z=0')
axs[1].set_xlabel('$y_{\ell}$ ($\mu$m)', fontsize=12)
axs[1].set_ylabel('$x_{\ell}$ ($\mu$m)', fontsize=12)
# axs[1].invert_yaxis()  # To match 'set(gca,'ydir','normal')' in MATLAB
axs[1].grid(False)

# # Add contours for qi_2
# contour_levels2 = np.linspace(-3E-4, 3E-4, 10)  # Adjust the contour levels as needed
# contour2 = axs[1].contour(Y, X, qi_fieldp[1, :, :, zl_steps // 2].squeeze(), 
#                           levels = contour_levels2, colors='black')

# Customize colorbar with scientific notation
cbar2 = fig.colorbar(im2, ax=axs[1], format=ScalarFormatter(useMathText=True))
cbar2.formatter.set_powerlimits((-2, 2))  # Adjust the power limits as needed
cbar2.update_ticks()

plt.tight_layout()
plt.savefig('qi1+qi2_fields1.svg')

