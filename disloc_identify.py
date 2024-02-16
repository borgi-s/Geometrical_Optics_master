import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from scipy.ndimage import label
from scipy.spatial.transform import Rotation as R
from direct_space.forward_model import *
from image_processor import *
from matplotlib import cm
import plotly.graph_objects as go

class BurgersVectorsPlotter:
    def __init__(self, n, fpath1, extent):
        self.n = n / np.sqrt(3)
        self.slip_plane = str(n).replace(" ", "").replace(".", "").replace("[", "").replace("]", "")
        self.t_rot_IP = np.cross(self.find_b_vectors(), n)
        self.vec_angles = np.linspace(0, 350, 36)
        self.fpath1 = fpath1
        self.extent = extent

    def find_b_vectors(self):
        b_vectors = {
            "111": np.array([[-1, 1, 0], [1, 0, -1], [0, 1, -1]]),
            "1-11": np.array([[1, 1, 0], [1, 0, -1], [0, 1, 1]]),
            "11-1": np.array([[1, -1, 0], [1, 0, 1], [0, -1, -1]]),
            "-111": np.array([[-1, -1, 0], [-1, 0, -1], [0, 1, -1]])
        }
        if self.slip_plane not in b_vectors:
            print(self.slip_plane)
            raise ValueError("Invalid slip plane")
        basis = b_vectors[self.slip_plane]
        return np.vstack([basis, -basis]) / np.sqrt(2)

    def calculate_rotated_vectors(self):
        b = self.find_b_vectors()
        t_rot_IP = np.cross(b, self.n)
        rotated_vectors = np.zeros((len(self.vec_angles), len(b), 3))
        for i, vec_angle in enumerate(self.vec_angles):
            rotation = R.from_rotvec(vec_angle * self.n, degrees=True)
            rotated_vectors[i] = rotation.apply(t_rot_IP)
        return rotated_vectors
    
    def calculate_ud_matrices(self):
        rotated_vectors = self.calculate_rotated_vectors()
        Ud = np.array([np.array((np.cross(self.n, rotated_vectors[i][j]), self.n, rotated_vectors[i][j])).T
                       for i in range(len(rotated_vectors)) for j in range(len(rotated_vectors[i]))])
        Ud = Ud.reshape(len(rotated_vectors), len(rotated_vectors[0]), 3, 3)
        return Ud
    
    def plot_vectors(self):
        rotated_vectors = self.calculate_rotated_vectors()
        normal_vector = self.n

        # Create a grid of points in the plane
        xx, yy = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
        zz = (-normal_vector[0] * xx - normal_vector[1] * yy) / (normal_vector[2])

        fig = go.Figure()
        # Add the plane as a surface
        fig.add_trace(go.Surface(z=zz, x=xx, y=yy, showscale=False, 
                         colorscale=[[0, 'black'], [1, 'black']],
                         opacity=0.75))
        for i, vector in enumerate(rotated_vectors[:, 0]):
            fig.add_trace(go.Scatter3d(x=[0, vector[0]], y=[0, vector[1]], z=[0, vector[2]],
                                       mode='lines+markers', name=f"Rotation {i}",
                                       text=[f"Rotation {i}", ""],
                                       marker=dict(size=5, color='blue'),
                                       line=dict(color='blue', width=3)))

        b = self.find_b_vectors() * np.sqrt(2)
        for i, b_vector in enumerate(b):
            fig.add_trace(go.Scatter3d(x=[0, b_vector[0]], y=[0, b_vector[1]], z=[0, b_vector[2]],
                                       mode='lines+markers', name=f"b_{i+1}",
                                       text=[f"b_{i+1}", ""],
                                       marker=dict(size=5, color='red'),
                                       line=dict(color='red', width=3)))

        fig.update_layout(scene=dict(xaxis_title='001', yaxis_title='010', zaxis_title='100', aspectmode='cube'))
        fig.update_layout(scene_zaxis_range=[-1, 1], scene_xaxis_range=[-1, 1], scene_yaxis_range=[-1, 1])
        fig.show()

    def plot_images(self):
        self.Ud = self.calculate_ud_matrices()
        b_vecs = self.find_b_vectors()
        check_path(self.fpath1 + '\images')
        check_path(self.fpath1 + '\im_data')
        for i in tqdm(range(self.Ud.shape[1])):
            for j in range(self.Ud.shape[0]):
                Fg = Fd_find_mixed(rl * 1e6, Us, self.Ud[j, i], self.vec_angles[j], Theta)
                Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1])
                Hg -= np.identity(3)
                im = forward(Hg, phi=150e-6) * 7
                im_noisy = np.random.poisson(im)
                plt.imshow(im_noisy.T, aspect='auto', origin='lower',
                           vmin=im_noisy.mean() + np.std(im_noisy), extent=extent)
                plt.colorbar()
                plt.ylabel('xl (µm)')
                plt.xlabel('yl (µm)')
                plt.title('Alpha = {0} degrees\nb = {1}\nn = {2}'.format(self.vec_angles[j], b_vecs[i] * np.sqrt(2),
                                                                          self.n * np.sqrt(3)))
                plt.savefig(self.fpath1 + "\images\\n_{0}_b_{1}_alpha_{2}_degrees.png".format(
                    str(self.n * np.sqrt(3)).replace(" ", "").replace("[", "").replace("]", "").replace(".", ""),
                    str(b_vecs[i] * np.sqrt(2)).replace(" ", "").replace("[", "").replace("]", "").replace(".", ""),
                    int(self.vec_angles[j])))
                plt.close()
                
                np.save(self.fpath1 + "\im_data\\n_{0}_b_{1}_alpha_{2}_degrees.npy".format(
                    str(self.n * np.sqrt(3)).replace(" ", "").replace("[", "").replace("]", "").replace(".", ""),
                    str(b_vecs[i] * np.sqrt(2)).replace(" ", "").replace("[", "").replace("]", "").replace(".", ""),
                    int(self.vec_angles[j])), im_noisy)


extent = [-yl_range*1e6, yl_range*1e6, -xl_range*1e6, xl_range*1e6]
fpath1 = (r'C:\Users\borgi\Documents\Production\Identification_paper' +
            r'\Complete_Loop1\all_slip_planes\600nmbeamH')
n = np.array([1, 1, 1], dtype=np.float64)
burgers_vectors_plotter = BurgersVectorsPlotter(n, fpath1, extent)
burgers_vectors_plotter.plot_vectors()
burgers_vectors_plotter.plot_images()
