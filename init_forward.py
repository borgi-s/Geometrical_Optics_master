# Import necessary libraries
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import center_of_mass
from scipy.interpolate import griddata
from direct_space.forward_model import *
from concurrent.futures import ThreadPoolExecutor
from image_processor import *
from matplotlib.colors import ListedColormap
from matplotlib.ticker import ScalarFormatter

# Define the range and number of steps for phi and chi
phi_range, phi_steps = 0.0004*180/np.pi, 41
chi_range, chi_steps = 0.003*180/np.pi, 31

# Define the filepath and filename prefix for the output images
fpath = r'C:\Users\borgi\Documents\Production\images7_perf_crystal'
fn_prefix = r'/mosa_test_0000_'
ftype = ".npy"

# Define the filepath and filename prefix for the output images
fpath1 = r'C:\Users\borgi\Documents\Production\images7'
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
save_images_parallel(Hg, phi_range, phi_steps, 
                     chi_range, chi_steps, 
                     fpath1, fn_prefix, ftype)


stack_no_dislocs, stack_reshape_no_dislocs, dim_1_no_dislocs, dim_2_no_dislocs = load_images(fpath, phi_steps, chi_steps)
stack_dislocs, stack_reshape_dislocs, dim_1_dislocs, dim_2_dislocs = load_images(fpath1, phi_steps, chi_steps)


plt.title('Angular spread of example pixel')
plt.imshow(stack_no_dislocs.reshape((phi_steps, chi_steps, dim_1_no_dislocs, dim_2_no_dislocs))[:,:,510//2,170//2], origin='lower', aspect='auto',
           extent = [-np.deg2rad(phi_range), np.deg2rad(phi_range), 
                     -np.deg2rad(chi_range), np.deg2rad(chi_range)])
plt.colorbar()
plt.xlabel(f'$\phi$ [radians]')
plt.ylabel(f'$\chi$ [radians]')
plt.show()

plt.title('Angular spread of example pixel')
plt.imshow(stack_dislocs.reshape((phi_steps, chi_steps, dim_1_no_dislocs, dim_2_no_dislocs))[:,:,510//2,170//2], origin='lower', aspect='auto',
           extent = [-np.deg2rad(phi_range), np.deg2rad(phi_range), 
                     -np.deg2rad(chi_range), np.deg2rad(chi_range)])
plt.colorbar()
plt.xlabel(f'$\phi$ [radians]')
plt.ylabel(f'$\chi$ [radians]')
plt.show()




pic = stack_dislocs.reshape((phi_steps, chi_steps, dim_1_dislocs, dim_2_dislocs))[15,15]
reshaped_stack = stack_dislocs.reshape((phi_steps, chi_steps, dim_1_dislocs, dim_2_dislocs))
reshaped_stack_clean = stack_no_dislocs.reshape((phi_steps, chi_steps, dim_1_no_dislocs, dim_2_no_dislocs))
reshaped_residuals = reshaped_stack-reshaped_stack_clean
reshaped_residuals[reshaped_residuals<0] = 0
stack_residuals = stack_dislocs-stack_no_dislocs
thresh = 0.1
# Find the indices of residual_mask
xtremap = np.zeros((dim_1_dislocs, dim_2_dislocs))
misori_map = np.zeros((dim_1_dislocs, dim_2_dislocs))
residual_mask = np.where(reshaped_stack >= np.max(reshaped_stack)*thresh, 1, 0)
phi_list = np.zeros((dim_1_dislocs, dim_2_dislocs))
chi_list = np.zeros((dim_1_dislocs, dim_2_dislocs))


for i in tqdm(range(reshaped_stack.shape[2])):
    for j in range(reshaped_stack.shape[3]):
                
        indices = np.argwhere(residual_mask[:,:,i,j] > 0)

        y_ind = indices[np.argmax(abs(indices[:,1] - reshaped_stack.shape[1]//2))][-1]
        y_inds = np.where(indices[:,1] == y_ind)[0]
        # x_inds = np.asarray((np.min(indices[y_inds][:,0]), np.max(indices[y_inds][:,0])))
        x_ind = np.round(center_of_mass(reshaped_stack[:,:,i,j])).astype(int)[1]
        phi_list[i,j] = np.deg2rad(Phi[y_ind])
        chi_list[i,j] = np.deg2rad(Chi[x_ind])
        cmap_coords = x_ind, y_ind
        flattened_index = np.ravel_multi_index((y_ind, x_ind), reshaped_stack.shape[:2])
        xtremap[i,j] = flattened_index











# Plot of three components of qi in (x, y, z=0) plane
fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # Create a figure with two subplots

# Plot qi_1
im1 = axs[0].imshow((phi_list*-1).T, interpolation='none',
                    extent=[xl_start, -xl_start,
                            yl_start, -yl_start],
                    vmin=-3E-4, vmax=3E-4, cmap='viridis', origin = 'lower')
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
im2 = axs[1].imshow(chi_list.T, 
                    extent=[xl_start, -xl_start,
                            yl_start, -yl_start],
                    vmin=-3E-3, vmax=3E-3, interpolation = 'none',
                    cmap='viridis', origin = 'lower')
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
plt.savefig('final_figures/extrem_phi+chi.svg')



'''
imp, qi_fieldp = forward(Hg, phi=0)

data = qi_fieldp[:2,:,:,34//2]

qi_phi = np.linspace(-3e-4,3e-4, 101)
qi_chi = np.linspace(-3e-3,3e-3, 101)

# Flatten the 2D array to a 1D array
data_0 = data[0].flatten()
data_1 = data[1].flatten()

# Compute the absolute differences between data_1d and linspace_array
absolute_differences0 = np.abs(data_0[:, np.newaxis] - qi_phi)
absolute_differences1 = np.abs(data_1[:, np.newaxis] - qi_chi)

# Find the index of the minimum absolute difference for each element in data_1d
closest_indices0 = np.argmin(absolute_differences0, axis=1)
closest_indices1 = np.argmin(absolute_differences1, axis=1)

# Reshape the closest_indices array back to the original shape of the 2D array
result0 = closest_indices0.reshape(data[0].shape)
result1 = closest_indices1.reshape(data[1].shape)

flattened_index = np.ravel_multi_index((result1, result0), (101,101))
'''

arrow0_start_inds = np.unravel_index(int(xtremap[360, 105]), (31,31))
arrow0_end_inds = np.unravel_index(int(xtremap[259, 85]), (31,31))
arrow1_start_inds = np.unravel_index(int(xtremap[145, 67]), (31,31))
arrow1_end_inds = np.unravel_index(int(xtremap[251, 84]), (31,31))




# test_grid = np.array((qi_chi,qi_phi),dtype=object)
o_grid = np.array((np.deg2rad(Chi),np.deg2rad(Phi)),dtype=object)
colors, color_data = inv_polefigure_colors1(o_grid, o_grid)
RGBA_colors = np.zeros(((o_grid[0].shape[0]*o_grid[1].shape[0])+2,4))
RGBA_colors[0] = (1.0, 1.0, 1.0, 1.0)
RGBA_colors[1:-1] = colors
RGBA_colors[-1] = RGBA_colors[0]


extreme, strain = extreme_map(reshaped_stack, dim_1_no_dislocs, 
                              dim_2_no_dislocs, thresh=0.10)
# Flatten the array and compute the two most frequent values
most_frequent_values, counts = np.unique(extreme.ravel(), return_counts=True)
most_frequent_values = (most_frequent_values[np.argsort(-counts)][:2])
xtremap = np.copy(extreme)
print("The two most frequent values:", most_frequent_values)
xtremap[xtremap == most_frequent_values[0]] = 0
xtremap[xtremap == most_frequent_values[1]] = 0


# Phi *= -1
fig, axes = plt.subplots(2,2, figsize = (9, 9), facecolor='white', dpi = 200)

axes[0,0].set_title('qi_field mosa')
axes[0,0].imshow(xtremap.T, origin = 'lower', aspect = 'auto',
                cmap=ListedColormap(colors), interpolation='none')
axes[0,0].arrow(360,105,259-360,86-105, length_includes_head = True,
                head_width = 5, color='white')
axes[0,0].arrow(145,67,251-145,84-67, length_includes_head = True,
                head_width = 5, color='black')
axes[0,0].set_xlabel(r'$y_{l}$ $[pixels]$')
axes[0,0].set_ylabel(r'$x_{l}$ $[pixels]$')
axes[0,0].set_facecolor('black')
axes[0,0].grid(False)
axes[0,1].grid(False)
axes[0,1].set_title('Colormap')
axes[0,1].scatter(color_data.T[0], color_data.T[1], c=colors, s = 65, marker = ',')
axes[0,1].set_xlabel(r'$\chi$ $[radians]$')
axes[0,1].set_ylabel(r'$\phi$ $[radians]$')
axes[0,1].arrow(np.deg2rad(Chi[arrow0_start_inds[0]]), np.deg2rad(Phi[arrow0_start_inds[1]]), 
                np.deg2rad(Chi[arrow0_end_inds[0]])-np.deg2rad(Chi[arrow0_start_inds[0]]), 
                np.deg2rad(Phi[arrow0_end_inds[1]])-np.deg2rad(Phi[arrow0_start_inds[1]]),
                width = 0.000002, head_width = 0.00002,
                head_length = 0.0001, length_includes_head = True,
                color='white')
axes[0,1].arrow(np.deg2rad(Chi[arrow1_start_inds[0]]), np.deg2rad(Phi[arrow1_start_inds[1]]), 
                np.deg2rad(Chi[arrow1_end_inds[0]])-np.deg2rad(Chi[arrow1_start_inds[0]]), 
                np.deg2rad(Phi[arrow1_end_inds[1]])-np.deg2rad(Phi[arrow1_start_inds[1]]),
                width = 0.000002, head_width = 0.00002,
                head_length = 0.0001, length_includes_head = True,
                color='black')
# axes[0,1].invert_yaxis()
axes[1,0].set_title('Weak beam mosa removed domains')
axes[1,0].imshow(xtremap.T, origin = 'lower', aspect = 'auto',
                cmap=ListedColormap(RGBA_colors), interpolation='none')
axes[1,0].set_xlabel(r'$y_{l}$ $[pixels]$')
axes[1,0].set_ylabel(r'$x_{l}$ $[pixels]$')
axes[1,0].set_facecolor('black')
axes[1,1].set_title('Colormap')
axes[1,1].scatter(color_data.T[0], color_data.T[1], c=colors, s = 65, marker = ',')
axes[1,1].set_xlabel(r'$\chi$ $[radians]$')
axes[1,1].set_ylabel(r'$\phi$ $[radians]$')

plt.tight_layout()
plt.savefig('weakmosa_mosa5.svg')
