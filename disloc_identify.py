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



b = np.array([1, 0, -1], dtype=np.float64)/np.sqrt(2)
# n = np.array([1, 1, -1])/np.sqrt(3)
n = np.array([1, -1, 1], dtype=np.float64)/np.sqrt(3)
t_rot_IP = np.cross(b,n)
# # t = np.array([-0.00419929,  0.008208,    0.01      ])

# t_l1 = np.array([-0.0028, 0.0043, 0.006])

# # cross_slip_t = np.array([-0.0091, -0.00648,     0.004], dtype=np.float64)
# # screw_upper_t = np.array([-0.4193, 0.4314, 0.7988], dtype=np.float64)
# # screw_lower_t = np.array([0.0007, 0.00108, 0.002], dtype=np.float64)
# # t_norm = screw_upper_t / np.linalg.norm(screw_upper_t)
# t_norm = t_l1 / np.linalg.norm(t_l1)
# t_rot = Us.T @ Theta.T @ t_norm

# # Calculate dot product
# dot_product = np.dot(t_rot, n)

# # Calculate squared norm
# norm_squared = np.dot(n, n)

# # Calculate projection
# projection = (dot_product / norm_squared) * n

# # Calculate closest point on the plane
# closest_point = t_rot - projection

# t_rot_IP = closest_point

# t_rot_IP /= np.linalg.norm(t_rot_IP) 


Ud1 = np.array((np.cross(n, t_rot_IP),n, t_rot_IP)).T



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


# b_vecs = np.asarray([[ 0, -1,  1], [1, 0,  1], [1, 1, 0],
#                      [0, 1, -1],  [-1, 0, -1], [-1, -1, 0]])
# b_vecs = np.asarray([[ 0, 1,  1], [-1, 0,  1], [1, 1, 0],
#                      [0, -1, -1],  [1, 0, -1], [-1, -1, 0]])
# b_vecs = np.asarray([[ 0, 1,  1], [1, 0,  1], [-1, 1, 0],
#                      [0, -1, -1],  [-1, 0, -1], [1, -1, 0]])

#For loop
b_vecs = np.asarray([[ 0, -1,  1], [-1, 1, 0], [-1, 0, 1],
                     [0, 1, -1],  [1, -1, 0], [1, 0, -1]])


# vec_angles = np.zeros((6))
# for i in range(len(b_vecs)):
#     vec_angles[i] = signed_angle_between_vectors(t_rot, b_vecs[i],n)


import plotly.graph_objects as go
import numpy as np

# Define the normal vector of the plane
normal_vector = n

# Create a grid of points in the plane
xx, yy = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
zz = (-normal_vector[0] * xx - normal_vector[1] * yy) / (normal_vector[2])

# Define your vectors
vectors = [ 
            t_rot_IP,
            normal_vector,
            b_vecs[0],
            b_vecs[1],
            b_vecs[2],
            b_vecs[3],
            b_vecs[4],
            b_vecs[5],
            ]
labels = [
          "t_fit", 
          "n = [1 1 -1]", 
          "b_1 = " + str(b_vecs[0]), 
          "b_2 = " + str(b_vecs[1]), 
          "b_3 = " + str(b_vecs[2]),
          "b_4 = " + str(b_vecs[3]), 
          "b_5 = " + str(b_vecs[4]), 
          "b_6 = " + str(b_vecs[5])]
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


def isRotationMatrix(R):
    # square matrix test
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    should_be_identity = np.allclose(R.dot(R.T), np.identity(R.shape[0], float))
    should_be_one = np.allclose(np.linalg.det(R), 1)
    return should_be_identity and should_be_one


fpath1 = (r'C:\Users\borgi\Documents\Production\Identification_paper' +
            r'\Complete_Loop\(1-11)[10-1]\600nmbeamH')
# Ud_mix = find_Ud_mix(n, t_edge, t_screw, 0)
vec_angles = np.linspace(0,360, 37)
Fg = Fd_find_mixed(rl * 1e6, Us, Ud1, 0, Theta)
Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1])
Hg -= np.identity(3)
im = forward(Hg, phi = 150e-6)
extent = [-yl_range*1e6, yl_range*1e6, -xl_range*1e6, xl_range*1e6]
i = 90
plt.imshow(im.T*7, aspect = 'auto', origin = 'lower', 
        vmin = im.mean()*7, extent = extent)
plt.colorbar()
plt.ylabel('xl (µm)')
plt.xlabel('yl (µm)')
plt.title('Alpha = {0} degrees'.format(i))

check_path(fpath1)

for i in vec_angles:
    Fg = Fd_find_mixed(rl * 1e6, Us, Ud1, i, Theta)
    Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1])
    Hg -= np.identity(3)
    im = forward(Hg, phi = 150e-6)*7
    plt.imshow(im.T, aspect = 'auto', origin = 'lower', 
        vmin = im.mean(), extent = extent)
    plt.colorbar()
    plt.ylabel('xl (µm)')
    plt.xlabel('yl (µm)')
    plt.title('Alpha = {0} degrees'.format(i))
    plt.savefig(fpath1 + "\Alpha_{0}_degrees.png".format(int(i)))
    plt.close()