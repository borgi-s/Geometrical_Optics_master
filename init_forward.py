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


b = np.array([-1, 0, -1])/np.sqrt(2)
# n = np.array([1, 1, -1])/np.sqrt(3)
n = np.array([1, -1, -1])/np.sqrt(3)
t = np.array([-0.16610975, 0.41187559, 0.89597213]) # from fit line
# a = np.loadtxt('M:\Documents\PhD DTU\Papers\Disloc identification\DislocationLine_0.txt', skiprows=1, delimiter=',')
# dis_start = a.T[6:9][:,-1]
# dis_end = a.T[6:9][:,0]
# dis_vec = (dis_start-dis_end)
# dis_norm = dis_vec / np.linalg.norm(dis_vec)

t = np.array([-0.00419929,  0.008208,    0.01      ])

t_l1 = np.array([-0.0028, 0.0043, 0.006])

t_norm = t_l1 / np.linalg.norm(t_l1)


t_rot = Us.T @ Theta.T @ t_norm
t_rot = np.array([0.59985723,  0.7796457 , -0.17978845])
def signed_angle_between_vectors(t, b, normal):
    t_edge = np.cross(b, normal)
    # Calculate the angle between the vectors
    angle = np.arccos(np.dot(t, t_edge) / (np.linalg.norm(t) * np.linalg.norm(t_edge)))
    angle = np.rad2deg(angle)
    # Calculate the cross product between t and b
    cross_product = np.cross(t, b)
    n_dot_crossp = np.dot(normal, cross_product)
    b_dot_t = np.dot(b, t)
    # Check the direction of the cross product with the normal vector
    if n_dot_crossp >= 0 and b_dot_t < 0:
        angle += 180
    if n_dot_crossp < 0 and b_dot_t < 0:
        angle += 180
    return angle


b_vecs = np.asarray([[ 1, 1,  0], [1, 0,  1], [0,  -1,  1],
                     [-1, -1, 0],  [-1, 0, -1], [0, 1, -1]])




vec_angles = np.zeros((6))
for i in range(len(b_vecs)):
    vec_angles[i] = signed_angle_between_vectors(t_rot, b_vecs[i],n)


import plotly.graph_objects as go
import numpy as np

# Define the normal vector of the plane
normal_vector = np.array([1, -1, -1])/np.sqrt(3)

# Create a grid of points in the plane
xx, yy = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
zz = (-normal_vector[0] * xx - normal_vector[1] * yy) / (normal_vector[2])

# Define your vectors
vectors = [ t_rot,
            normal_vector,
            b_vecs[0],
            b_vecs[1],
            b_vecs[2],
            b_vecs[3],
            b_vecs[4],
            b_vecs[5],
            ]
labels = ["t_fit", "n = [1 -1 -1]", "b_1 = [1 1 0]", 
          "b_2 = [1 0 1]", "b_3 = [0 -1 1]",
          "b_4 = [-1 -1 0]", "b_5 = [-1 0 -1]", 
          "b_6 = [0 1 -1]"]
# Define distinct colors for each vector
vector_colors = ['#ffe119', '#4363d8', '#f58231', 
                 '#dcbeff', '#800000', '#000075', 
                 '#a9a9a9', '#f032e6', 'red']

# Create a 3D scatter plot
fig = go.Figure()

# Add the plane as a surface
fig.add_trace(go.Surface(z=zz, x=xx, y=yy, showscale=False, 
                         colorscale=[[0, 'black'], [1, 'black']],
                         opacity=0.75))

for vector, label, color in zip(vectors, labels, vector_colors):
    fig.add_trace(go.Scatter3d(x=[0, vector[0]], y=[0, vector[1],], z=[0, vector[2]],
                               mode='lines+markers', name=label,
                               text=[label, ""], 
                               marker=dict(size=5, color=color),
                               line=dict(color=color, width=3)))

# Set axis labels
fig.update_layout(scene=dict(xaxis_title='001', yaxis_title='010', 
                             zaxis_title='100', aspectmode='cube'))

fig.update_layout(scene_zaxis_range=[-1, 1], scene_xaxis_range=[-1, 1], scene_yaxis_range=[-1, 1])
# Show the plot
fig.show()

Ud1 = np.asarray((np.cross(n, t_rot), n, t_rot)).T

def isRotationMatrix(R):
    # square matrix test
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    should_be_identity = np.allclose(R.dot(R.T), np.identity(R.shape[0], float))
    should_be_one = np.allclose(np.linalg.det(R), 1)
    return should_be_identity and should_be_one


fpath1 = (r'C:\Users\borgi\Documents\Production\Identification_paper' +
            r'\2024_retry\lower_highres\Burgers_vector_101_500nmbeamH')
# Ud_mix = find_Ud_mix(n, t_edge, t_screw, 0)
for i in range(3):
    Fg = Fd_find_mixed(rl * 1e6, Us, Ud1, vec_angles[i], Theta)
    Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1])
    Hg -= np.identity(3)
    if i == 0:
        b1_im = forward(Hg, phi = 140e-6)
    if i == 1:
        b2_im = forward(Hg, phi = 140e-6)
    if i == 2:
        b3_im = forward(Hg, phi = 140e-6)


def load_edfs(fnames):
    return fabio.open(fnames).data


path = r'C:\Users\borgi\Documents\Dislocation_Identification\Al_sample_5_tick_775\Z_local_rocking_2x_05' + "\\"
folder_path = 'Z_local_rocking_2x_'
fnames = list_files(path, '.edf')

fabio.open(fnames[0]).header['motor_mne']


data = np.zeros(((len(fnames), 2160, 2560)))
for i in range(len(fnames)):
    data[i] = load_edfs(fnames[i])

fpath1 = (r'C:\Users\borgi\Documents\Production\Identification_paper' +
            r'\New_angles\Burgers_vector_2_tallbeam')
stack_dislocs, stack_reshape_dislocs, dim_1_dislocs, dim_2_dislocs = load_images(fpath1, phi_steps, chi_steps)


plt.imshow(data[5,1120:1165, 1060:1115], vmin = 102, vmax = 150, aspect=1/np.tan(np.deg2rad(17.953)))
plt.colorbar()
plt.show()

b_vec1_im = np.flip(np.copy(b1_im[200:270, 60:110]))
b_vec2_im = np.flip(np.copy(b2_im[200:270, 60:110]))
b_vec3_im = np.flip(np.copy(b3_im[240:310, 60:110]))


plt.imshow(b2_im[210:285, 50:120], vmin = (np.mean(b2_im)+1.65*np.std(b2_im)), 
           vmax = (np.mean(b2_im)+2*np.std(b2_im)), origin='lower',
           )
plt.colorbar()
plt.show()




def load_edfs(fnames1):
    return fabio.open(fnames1).data


path = r'C:\Users\borgi\Documents\Dislocation_Identification\Al_sample_5_tick_revolution_70\Z_local_rocking_2x_00' + "\\"
folder_path = 'Z_local_rocking_2x_'
fnames = list_files(path, '.edf')


data = np.zeros(((len(fnames), 2160, 2560)))
for i in range(len(fnames)):
    data[i] = load_edfs(fnames[i])

# fpath1 = (r'C:\Users\borgi\Documents\Production\Identification_paper' +
#             r'\New_angles\Burgers_vector_2_tallbeam')
# stack_dislocs, stack_reshape_dislocs, dim_1_dislocs, dim_2_dislocs = load_images(fpath1, phi_steps, chi_steps)

data_im = np.flip(data[5,1185:1210, 970:1040])

experimental_strain_field = np.copy(data_im)

# Create a sample 2D NumPy array (replace this with your actual image array)
image_array = np.copy(experimental_strain_field)-104
image_array[image_array<0] = 0
# Stretch the y-axis by a factor of 3
stretched_array = np.repeat(image_array, 1/np.tan(np.deg2rad(17.953)), axis=0)
stretched_array[stretched_array>30] = 30
stretched_array = (stretched_array - stretched_array.min()) / (stretched_array.max() - stretched_array.min())

import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

b_vec1_im[b_vec1_im<5.1] = 0
b_vec2_im[b_vec2_im<5.1] = 0
b_vec3_im[b_vec3_im<5.1] = 0

b1_im[b1_im<5.1] = 0
b2_im[b2_im<5.1] = 0
b3_im[b3_im<5.1] = 0


# Generate a sample experimental strain field and simulated strain fields
simulated_b_vec_1 = np.flip(b1_im)
simulated_b_vec_2 = np.flip(b2_im)
simulated_b_vec_3 = np.flip(b3_im)

# Perform cross-correlation
corr_1 = correlate2d(stretched_array, simulated_b_vec_1/np.max(simulated_b_vec_1), mode='full')
corr_2 = correlate2d(stretched_array, simulated_b_vec_2/np.max(simulated_b_vec_2), mode='full')
corr_3 = correlate2d(stretched_array, simulated_b_vec_3/np.max(simulated_b_vec_3), mode='full')

# Find the index of the maximum correlation value
max_corr_index = np.argmax([corr_1.max(), corr_2.max(), corr_3.max()])
corr_list = [corr_1, corr_2, corr_3]
# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.title('$b_{vec}$ = [0 -1 1]' + f'\nMax Correlation: {corr_list[0].max():.2f}', fontsize = 15)
plt.imshow(simulated_b_vec_1, cmap='viridis')

plt.subplot(2, 3, 2)
plt.title('$b_{vec}$ = [1 0 1]' + f'\nMax Correlation: {corr_list[1].max():.2f}', fontsize = 15)
plt.imshow(simulated_b_vec_2, cmap='viridis')

plt.subplot(2, 3, 3)
plt.title('$b_{vec}$ = [1 1 0]' + f'\nMax Correlation: {corr_list[2].max():.2f}', fontsize = 15)
plt.imshow(simulated_b_vec_3, cmap='viridis')

plt.subplot(2, 3, 4)
plt.title('Experimental Strain Field', fontsize = 15)
plt.imshow(stretched_array, vmax = 0.75, cmap='viridis')

plt.subplot(2, 3, 5)

plt.title(f'Max Cross-Correlation Result\nMax Correlation: {corr_list[max_corr_index].max():.2f}', fontsize = 15)
plt.imshow(corr_list[max_corr_index], cmap='viridis')

plt.subplot(2, 3, 6)
plt.title(f'Selected Simulation: {max_corr_index+1}', fontsize = 15)
plt.imshow(eval(f'simulated_b_vec_{max_corr_index+1}'), cmap='viridis')

plt.tight_layout()
plt.show()
