import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from scipy.ndimage import label, shift
from scipy.spatial.transform import Rotation as R
from direct_space.forward_model import *
from image_processor import *
from matplotlib import cm
import plotly.graph_objects as go
from functions import rotate_matrix_z_axis, Z_shift
import os, shutil, subprocess

class BurgersVectorsPlotter:
    def __init__(self, n, q_hkl, Us, fpath1, extent, 
                 xl_start = None, yl_start = None, zl_start = None, 
                 z_scan = None, phi_range = None, phi_steps = None, 
                 chi_range = None, chi_steps = None,
                 savepath = None, fn_prefix = None, ftype = None):
        '''
        Class to manage Burgers vector computations, rotations, 
        and plotting for crystallographic analysis. Calling the 
        forward model to generate and save images with different 
        types of scans and dislocation structures.
        ---------
        Inputs:
            n (array-like): Slip plane normal vector.
            q_hkl (array-like): Reciprocal lattice vector.
            Us (array): Rotation matrix of between sample and crystal/grain.
            fpath1 (str): File path for output.
            extent (tuple): Plotting limits for image outputs.
            xl_start, yl_start, zl_start (array-like, optional): Initial coordinates in lab coordinate system.
            z_scan (array-like, optional): Scanning steps in the z-direction.
            phi_range (tuple, optional): Angular range for phi rotations (rocking).
            phi_steps (int, optional): Number of steps in phi rotations (rocking).
            chi_range (tuple, optional): Angular range for chi rotations (rolling).
            chi_steps (int, optional): Number of steps in chi rotations (rolling).
            savepath (str, optional): Save path for simulated images and results.
            fn_prefix (str, optional): Filename prefix for saved files.
            ftype (str, optional): File type for saved images or data.
        '''
        self.n = n / np.sqrt(3)
        self.q_hkl = q_hkl
        self.slip_plane = str(n).replace(" ", "").replace(".", "").replace("[", "").replace("]", "")
        self.t_rot_IP = np.cross(self.find_b_vectors(), n)
        self.vec_angles = np.linspace(0, 350, 36)
        self.fpath1 = fpath1
        self.extent = extent
        self.Us = Us
        self.xl_start = xl_start
        self.yl_start = yl_start
        self.zl_start = zl_start
        self.phi_range = phi_range
        self.phi_steps = phi_steps
        self.chi_range = chi_range
        self.chi_steps = chi_steps
        self.savepath = savepath
        self.fn_prefix = fn_prefix
        self.ftype = ftype
        self.z_scan = z_scan

    def find_b_vectors(self):
        '''
        Calculates the Burgers vectors for the given slip plane.
        -----
        Outputs:
            numpy.ndarray: Array of normalized Burgers vectors for the specified slip plane.
        '''
        b_vectors = {
            "111":    np.array([[-1,  1, 0], [ 1, 0, -1], [0,  1, -1]]),
            "1-11":   np.array([[ 1,  1, 0], [ 1, 0, -1], [0,  1,  1]]),
            "11-1":   np.array([[ 1, -1, 0], [ 1, 0,  1], [0, -1, -1]]),
            "-111":   np.array([[-1, -1, 0], [-1, 0, -1], [0,  1, -1]]),
            "1-1-1":  np.array([[-1, -1, 0], [-1, 0, -1], [0,  1, -1]]),
            "-1-11":  np.array([[ 1, -1, 0], [ 1, 0,  1], [0, -1, -1]]),
            "-11-1":  np.array([[ 1,  1, 0], [ 1, 0, -1], [0,  1,  1]]),
            "-1-1-1": np.array([[-1,  1, 0], [ 1, 0, -1], [0,  1, -1]]),
        }
        if self.slip_plane not in b_vectors:
            print(self.slip_plane)
            raise ValueError("Invalid slip plane")
        basis = b_vectors[self.slip_plane]
        return np.vstack([basis, -basis]) / np.sqrt(2)

    def calculate_rotated_vectors(self):
        '''
        Computes the rotated vectors based on Burgers vectors and rotation angles.
        -----
        Outputs:
            numpy.ndarray: Rotated vectors array of shape (num_angles, num_vectors, 3).
        '''
        b = self.find_b_vectors()
        t_rot_IP = np.cross(b, self.n)

        rotated_vectors = np.zeros((len(self.vec_angles), len(b), 3))
        for i, vec_angle in enumerate(self.vec_angles):
            rotation = R.from_rotvec(vec_angle * self.n, degrees=True)
            rotated_vectors[i] = rotation.apply(t_rot_IP)
        return rotated_vectors
    
    def calculate_ud_matrices(self):
        '''
        Generates rotation matrices (Ud, dislocation coordinates) based on the rotated line 
        direction vectors and the slip plane normal.
        -----
        Outputs:
            numpy.ndarray: 4D array of rotation matrices Ud with dimensions (num_angles, num_vectors, 3, 3).
        '''
        rotated_vectors = self.calculate_rotated_vectors()
        Ud = np.array([np.array((np.cross(self.n, rotated_vectors[i][j]), self.n, rotated_vectors[i][j])).T
                       for i in range(len(rotated_vectors)) for j in range(len(rotated_vectors[i]))])
        Ud = Ud.reshape(len(rotated_vectors), len(rotated_vectors[0]), 3, 3)
        return Ud
        
    def is_valid_rotation_matrix(R):
        '''
        Checks if the rotation matrix is valid and right handed.
        -----
        Inputs:
            R (np.ndarray): 3D Array of rotation matrix.
        '''
        if R.shape != (3, 3):
            return False
        is_orthogonal = np.allclose(R @ R.T, np.identity(3), atol=1e-6)
        has_unit_determinant = np.allclose(np.linalg.det(R), 1.0, atol=1e-6)
        return is_orthogonal and has_unit_determinant
        
    def rotate_sample_zl(self, rl, angle):
        '''
        Applies a rotation to 3D coordinates around the z-axis by a given angle.
        -----
        Inputs:
            rl (np.ndarray): Array of 3D lab coordinates to rotate.
            angle (float): Angle in radians for rotation.
        Outputs:
            numpy.ndarray: Rotated 3D coordinates.
        '''
        # Define the rotation matrix for the z-axis
        Rz = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,            0,           1]
        ])
        
        # Apply the rotation matrix
        rl_rotated = Rz @ rl
        return rl_rotated

    
    def plot_vectors(self):
        '''
        Creates a 3D plot of rotated line direction vectors, the Burgers vectors, 
        and the slip plane.
        -----
        Outputs:
            Interactive 3D plot displayed using Plotly.
        '''
        rotated_vectors = self.calculate_rotated_vectors()
        normal_vector = self.n
        t_rot = np.array([0.59985723, 0.7796457, -0.17978845])  # The t_rot vector

        # Create a grid of points in the plane
        xx, yy = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
        zz = (-normal_vector[0] * xx - normal_vector[1] * yy) / (normal_vector[2])

        fig = go.Figure()
        # Add the t_rot vector
        fig.add_trace(go.Scatter3d(x=[0, t_rot[0]], y=[0, t_rot[1]], z=[0, t_rot[2]],
                                   mode='lines+markers', name="t_rot",
                                   text=["t_rot", ""],
                                   marker=dict(size=5, color='green'),
                                   line=dict(color='green', width=3)))
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
        '''
        Generates and saves simulated images of the sample based on Burgers vectors and transformations.
        -----
        Outputs:
            Saves images and corresponding data files to the specified file path.
        '''
        self.Ud = self.calculate_ud_matrices()
        if not is_valid_rotation_matrix(self.Us):
            raise ValueError("Us contains invalid rotation matrices.")
        b_vecs = self.find_b_vectors()
        check_path(self.fpath1 + '/images')
        check_path(self.fpath1 + '/im_data')
        for i in tqdm(range(self.Ud.shape[1])):
            for j in range(self.Ud.shape[0]):
                if not np.allclose(np.dot(np.cross(self.Ud[j,i].T[2], self.Ud[j,i].T[1]), b_vecs[i]), 0) or not np.allclose(np.dot(self.q_hkl, b_vecs[i]), 0):
                    Fg = Fd_find_mixed(rl * 1e6, self.Us, self.Ud[j, i], self.vec_angles[j], Theta)
                    Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1])
                    Hg -= np.identity(3)
                    im = forward(Hg, phi=140e-6) * 10
                    im_noisy = im # np.random.poisson(im)
                    plt.imshow(im_noisy.T, aspect='auto', origin='lower',
                            vmin=im_noisy.mean() + np.std(im_noisy), extent=extent)
                    plt.colorbar()
                    plt.ylabel('xl (µm)')
                    plt.xlabel('yl (µm)')
                    plt.title('Alpha = {0} degrees\nb = {1}\nn = {2}'.format(self.vec_angles[j], b_vecs[i] * np.sqrt(2),
                                                                            self.n * np.sqrt(3)))
                    plt.savefig(self.fpath1 + "/images/n_{0}_b_{1}_alpha_{2}_degrees.png".format(
                        str(self.n * np.sqrt(3)).replace(" ", "").replace("[", "").replace("]", "").replace(".", ""),
                        str(b_vecs[i] * np.sqrt(2)).replace(" ", "").replace("[", "").replace("]", "").replace(".", ""),
                        int(self.vec_angles[j])))
                    plt.close()
                    
                    np.save(self.fpath1 + "/im_data/n_{0}_b_{1}_alpha_{2:03d}_degrees.npy".format(
                        str(self.n * np.sqrt(3)).replace(" ", "").replace("[", "").replace("]", "").replace(".", ""),
                        str(b_vecs[i] * np.sqrt(2)).replace(" ", "").replace("[", "").replace("]", "").replace(".", ""),
                        int(self.vec_angles[j])), im_noisy)
    
    def save_scan(self, batch_size=50):
        '''
        Processes and saves 3D scan results in batches to reduce memory overhead.
        -----
        Inputs:
            batch_size (int): Number of iterations processed before saving results (default is 50).
        Outputs:
            Saves scan results and parameters in the specified save path.
        '''
        self.Ud = self.calculate_ud_matrices()
        b_vecs = self.find_b_vectors()

        if not is_valid_rotation_matrix(self.Us):
            raise ValueError("Us contains invalid rotation matrices.")

        batch_Hg = []  # Store batched Hg matrices
        batch_savepaths = []  # Store corresponding save paths for each batch
        for k, offset in enumerate(self.z_scan):
            rl = Z_shift(self.xl_start, self.yl_start, self.zl_start, NN1, NN2, NN3, offset)
            prob_z = np.exp(-0.5*(rl[2]+offset/zl_rms)**2)
            savepath_layer = "/layer_{:04}/".format(k)
            for i in tqdm(range(self.Ud.shape[1])):
                for j in range(self.Ud.shape[0]):
                    if (not np.allclose(np.dot(np.cross(self.Ud[j, i].T[2], self.Ud[j, i].T[1]), b_vecs[i]), 0) or 
                        not np.allclose(np.dot(self.q_hkl, b_vecs[i]), 0)):
                            
                        rand_idx_j = np.random.randint(0, self.Ud.shape[0])
                        rand_idx_i = np.random.randint(0, self.Ud.shape[1])
                        random_disloc = self.Ud[rand_idx_j, rand_idx_i] # save this if its relevant to know random dislocations b, n, and t values.
                        random_angle  = self.vec_angles[rand_idx_j] # save this if its relevant to know random dislocations alpha
                            
                        Fg = d_find_multi_dislocs_mixed(rl * 1e6, self.Us, self.Ud[j, i], random_disloc, self.vec_angles[j], random_angle, Theta, dis = 4) # Change dis for distance between the dislocations, unit is micrometer
                        Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1])
                        Hg -= np.identity(3)

                        savepath_run = self.savepath + savepath_layer + "/n_{0}_b_{1}_alpha_{2:03d}_degrees/".format(
                            str(self.n * np.sqrt(3)).replace(" ", "").replace("[", "").replace("]", "").replace(".", ""),
                            str(b_vecs[i] * np.sqrt(2)).replace(" ", "").replace("[", "").replace("]", "").replace(".", ""),
                            int(self.vec_angles[j])
                        )
                        
                        # Append to batch
                        batch_Hg.append(Hg)
                        batch_savepaths.append(savepath_run)
                        
                        # If batch is full, process and save
                        if len(batch_Hg) >= batch_size:
                            self.process_and_save_batch(batch_Hg, batch_savepaths)
                            batch_Hg = []  # Reset batch
                            batch_savepaths = []

            # Process remaining items in the last batch
            if batch_Hg:
                self.process_and_save_batch(batch_Hg, batch_savepaths)

            # Save simulation parameters
            save_simulation_parameters(self.phi_range, self.phi_steps, 
                                    self.chi_range, self.chi_steps,
                                    self.z_scan, self.savepath)

    def process_and_save_batch(self, batch_Hg, batch_savepaths):
        '''
        Processes and saves a batch of matrix computations for a scan.
        -----
        Inputs:
            batch_Hg (list): List of matrices to process.
            batch_savepaths (list): Corresponding file paths for saving.
        Outputs:
            Saves processed images and data to disk.
        '''
        for Hg, savepath in zip(batch_Hg, batch_savepaths):
            save_images_parallel(Hg, self.phi_range, self.phi_steps,
                                self.chi_range, self.chi_steps,
                                savepath, self.fn_prefix, self.ftype)
    
    def g_dot_b(self):
        '''
        Filters image data based on the g·b criterion for detecting valid dislocations.
        -----
        Outputs:
            tuple: Lists of valid and masked image file paths.
        '''
        Ud = self.calculate_ud_matrices()
        b_vecs = self.find_b_vectors()
        file_list = []
        masked_files = []
        for i in tqdm(range(Ud.shape[1])):
            for j in range(Ud.shape[0]):
                if not np.allclose(np.dot(np.cross(Ud[j,i].T[2], Ud[j,i].T[1]), b_vecs[i]), 0) or not np.allclose(np.dot(self.q_hkl, b_vecs[i]), 0):
                    file_list.append(self.fpath1 + "/im_data/n_{0}_b_{1}_alpha_{2}_degrees.npy".format(
                        str(self.n * np.sqrt(3)).replace(" ", "").replace("[", "").replace("]", "").replace(".", ""),
                        str(b_vecs[i] * np.sqrt(2)).replace(" ", "").replace("[", "").replace("]", "").replace(".", ""),
                        int(self.vec_angles[j])))
                elif np.allclose(np.dot(np.cross(Ud[j,i].T[2], Ud[j,i].T[1]), b_vecs[i]), 0) or np.allclose(np.dot(self.q_hkl, b_vecs[i]), 0):
                    masked_files.append(self.fpath1 + "/im_data/n_{0}_b_{1}_alpha_{2}_degrees.npy".format(
                        str(self.n * np.sqrt(3)).replace(" ", "").replace("[", "").replace("]", "").replace(".", ""),
                        str(b_vecs[i] * np.sqrt(2)).replace(" ", "").replace("[", "").replace("]", "").replace(".", ""),
                        int(self.vec_angles[j])))
        return file_list, masked_files
    
    def count_files_and_dirs(self):
        '''
        Counts the total number of files and directories in the specified path.
        -----
        Outputs:
            int: Total count of files and directories.
        '''
        total = 0
        for root, dirs, files in os.walk(self.fpath1):
            total += len(files) + len(dirs)
        return total

    def zip_files(self):
        '''
        Compresses the simulation files and folders into a single archive.
        -----
        Outputs:
            Compressed archive file created in the parent directory of `fpath1`.
        '''
        source_folder = self.fpath1
        
        output_file = os.path.join(os.path.dirname(source_folder), "images_{0}.zip".format(source_folder.split("/")[-1]))

        total_items = self.count_files_and_dirs()
        
        command = ["tar", "-cvf", output_file, source_folder]

        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            with tqdm(total=total_items, desc="Archiving", unit="file") as pbar:
                for line in process.stdout:
                    # Update progress for each file/directory printed by tar
                    if line.strip():  # Ensure the line is not empty
                        pbar.update(1)

            # Wait for the process to finish
            process.wait()

        # Check for errors
        if process.returncode == 0:
            print(f"Archive created successfully: {output_file}")
        else:
            print(f"Error creating archive: {process.stderr.read()}")


extent = [-yl_range*1e6, yl_range*1e6, -xl_range*1e6, xl_range*1e6]



# "11-1": np.array([[1, -1, 0], [1, 0, 1], [0, -1, -1]]),
# "111": np.array([[-1, 1, 0], [1, 0, -1], [0, 1, -1]]),
# "1-11": np.array([[1, 1, 0], [1, 0, -1], [0, 1, 1]]),
# "-111": np.array([[-1, -1, 0], [-1, 0, -1], [0, 1, -1]])
slip_planes = [np.array([1,1,-1]), np.array([1,-1,1]), np.array([-1,1,1]), np.array([1,1,1])]
# n = np.array([-1, 1, 1], dtype=np.float64)

phi_range, phi_steps = 0.0005*180/np.pi, 5
chi_range, chi_steps = 0.001*180/np.pi, 1
z_scan = np.round(np.linspace(-10e-6, 10e-6, 41),9) # units in m, centered around 0
fn_prefix = r'/rocking_scan_0000_'
ftype = ".npy"
fpath1 = (r'/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/borgi/' +
          'ESRF_ML/Geometrical_Optics_exp/test_data/600nmbeamH_-11-1_refl')

for n in slip_planes:
    burgers_vectors_plotter = BurgersVectorsPlotter(n, q_hkl, Us, fpath1, extent,
                                                    xl_start, yl_start, zl_start,
                                                    z_scan, phi_range, phi_steps,
                                                    chi_range, chi_steps,
                                                    fpath1, fn_prefix, ftype)
    # burgers_vectors_plotter.plot_vectors()
    # burgers_vectors_plotter.plot_images()
    burgers_vectors_plotter.save_scan()

