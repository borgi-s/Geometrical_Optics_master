import numpy as np
import gc, os
from tqdm import tqdm
import multiprocessing as mp
from numba import jit
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed, cpu_count


def Fd_find_mixed(rl, Us, Ud_mix, a, Theta, dis = 1, ndis = 1, b = 2.862e-4, ny = 0.334, misorientation = False, t_vec = None):
    '''
    Calculates the displacement field due to multiple edge dislocations in a 
    crystal lattice, given the lab coordinates and accompanying rotation matrices.
    The dislocation wall will start at the center of the system and branch out as 
    more is added. If more than 100 dislocations are needed the function will call
    multi_dislocs_parallel to run with multiprocessing for saving time. This requires
    a lot of memory.
    -----------------------------------------------------------------------------
    Parameters:
    rl: np.ndarray of shape (3, X)
        An array of coordinates in lab space, with X being the number of steps in 
        each dimension. rl contains the x_lab, y_lab, z_lab coordinates.
    Ud: np.ndarray of shape (3, 3)
        A rotational matrix used for going from dislocation space to grain space.
    Us: np.ndarray of shape (3, 3)
        A rotational matrix used for going from grain space to sample space.
    Theta: np.ndarray of shape (3, 3)
        A rotational matrix used for going from sample space to lab space.
    dis : int, optional
        Distance in micrometer between each edge dislocation in the dislocation coordinate system. 
        Default is 1.
    ndis : int, optional
        Number of edge dislocations to include. 
        Default is 1.
    b : float, optional
        Magnitude of the Burger's vector. 
        Default is 2.862e-4 [micrometer] - 2.862 [Aangstrom].
    ny : float, optional
        Poisson ratio. Default is 0.334.
        
    Returns
    -----------------------------------------------------------------------------
    numpy.ndarray
        Displacement gradient field induced by set of edge dislocations, 
        in grain space. Shape is (X, 3, 3).
    '''
    # Rotate lab coordinates to dislocation coordinates
    rs = Theta @ rl # sample
    rc = Us.T @ rs # crystal
    rd = Ud_mix.T @ rc # dislocation

    # Initialize 3D tensor for the dislocation-induced deformation gradient field.
    Fdd = np.zeros([len(rd[-1]), 3, 3])
    alpha = 1e-20

    # Calculate intermediate values for deformation gradient tensor calculation.
    sqx = rd[0] * rd[0]
    sqy = rd[1] * rd[1]
    denom = (sqx + sqy) * (sqx + sqy) + alpha
    bfactor = b / (4 * np.pi * (1 - ny))
    nyfactor = 2 * ny * (sqx + sqy)

    # Calculate components of the deformation gradient tensor.
    Fdd[:, 0, 0] = -rd[1] * (3 * sqx + sqy - nyfactor) / denom
    Fdd[:, 0, 1] =  rd[0] * (3 * sqx + sqy - nyfactor) / denom
    Fdd[:, 1, 0] = -rd[0] * (3 * sqy + sqx - nyfactor) / denom
    Fdd[:, 1, 1] =  rd[1] * (sqx - sqy - nyfactor) / denom

    # Finish the calculation of Fdd
    Fdd *= bfactor
    Fdd *= np.cos(np.deg2rad(a))
    # Calculate intermediate values for deformation gradient tensor calculation.
    sqz = rd[2] * rd[2]
    denom1 = (sqz) + (sqy) + alpha
    bfactor1 = b / (2 * np.pi)

    # Calculate components of the deformation gradient tensor.
    Fdd[:, 0, 1] += ((-rd[2] / denom1) * bfactor1 * np.sin(np.deg2rad(a)))
    Fdd[:, 0, 2] += ((rd[1] / denom1) * bfactor1 * np.sin(np.deg2rad(a)))

    # Finish the calculation of Fdd
    Fdd += np.identity(3)
    return Ud_mix @ Fdd @ Ud_mix.T # Return the rotated Fdd -> Fg


def rotatedU(axis, alpha, U, coordtype):
    # rotates U according to rotation vector defined in either sample or cryst coords as specified by coordtype
    # angle in degrees

    if coordtype == 'cryst':
        axis = np.dot(U, axis)  # convert to sample
    # alpha = angle * np.pi / 180 # rotating angle
    r1, r2, r3 = axis[0], axis[1], axis[2]
    cost = np.cos(alpha)
    onecost = 1 - cost
    sint = np.sin(alpha)
    R = np.array([[r1 * r1 * onecost + cost, r1 * r2 * onecost + r3 * sint, r1 * r3 * onecost - r2 * sint],
                  [r1 * r2 * onecost - r3 * sint, r2 * r2 * onecost + cost, r2 * r3 * onecost + r1 * sint],
                  [r1 * r3 * onecost + r2 * sint, r2 * r3 * onecost - r1 * sint, r3 * r3 * onecost + cost]])
    Urot = np.dot(R, U)

    return Urot
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Path {path} created.")
    else:
        print(f"Path {path} already exists.")


def check_folder(path, folder_name):
    folder_path = os.path.join(path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder {folder_path} created.")
    else:
        print(f"Folder {folder_path} already exists.")

def image_range(im_in, lower, upper):
    return np.interp(im_in, (np.min(im_in), np.max(im_in)), (lower, upper))


def multi_dislocs_parallel(chunk, rd, Fdd_shape, dis, ny = 0.334):
    """
    Calculates the components of the deformation gradient tensor for multiple dislocations in parallel.

    Args:
        chunk (list): A chunk of dislocation indices to be processed by the function.
        rd (ndarray): An array containing the x and y coordinates of the reference dislocation.
        Fdd_shape (tuple): The shape of the Fdd tensor to be calculated.
        b (float, optional): A constant parameter. Default is 2.862e-4.
        ny (float, optional): A constant parameter. Default is 0.334.

    Returns:
        ndarray: The Fdd tensor chunk with the accumulated components of the deformation gradient tensor.
    """
    Fdd_chunk = np.zeros(Fdd_shape)
    for i in tqdm(chunk, desc = 'Running: {0}'.format(chunk)):
        rd_new = np.copy(rd[:2])
        rd_new[1] -= (i * dis)



        # Calculate intermediate values for deformation gradient tensor calculation.
        sqx = rd_new[0] * rd_new[0]
        sqy = rd_new[1] * rd_new[1]
        denom = (sqx + sqy) * (sqx + sqy)
        nyfactor = 2 * ny * (sqx + sqy)

        # Calculate the components of the deformation gradient tensor for this iteration.
        
        Fdd_chunk[:, 0, 0] += -rd_new[1] * (3 * sqx + sqy - nyfactor) / denom
        Fdd_chunk[:, 0, 1] += rd_new[0] * (3 * sqx + sqy - nyfactor) / denom
        Fdd_chunk[:, 1, 0] += -rd_new[0] * (3 * sqy + sqx - nyfactor) / denom
        Fdd_chunk[:, 1, 1] += rd_new[1] * (sqx - sqy - nyfactor) / denom


        rd_new = np.copy(rd[:2])
        rd_new[1] += (i * dis)

        # Calculate intermediate values for deformation gradient tensor calculation.
        sqx = rd_new[0] * rd_new[0]
        sqy = rd_new[1] * rd_new[1]
        denom = (sqx + sqy) * (sqx + sqy)
        nyfactor = 2 * ny * (sqx + sqy)

        # Calculate the components of the deformation gradient tensor for this iteration.
        Fdd_chunk[:, 0, 0] += -rd_new[1] * (3 * sqx + sqy - nyfactor) / denom
        Fdd_chunk[:, 0, 1] += rd_new[0] * (3 * sqx + sqy - nyfactor) / denom
        Fdd_chunk[:, 1, 0] += -rd_new[0] * (3 * sqy + sqx - nyfactor) / denom
        Fdd_chunk[:, 1, 1] += rd_new[1] * (sqx - sqy - nyfactor) / denom

    return Fdd_chunk

def Fd_find(rl, Ud, Us, Theta, dis = 1, ndis = 1, b = 2.862e-4, ny = 0.334, misorientation = False, t_vec = None):
    '''
    Calculates the displacement field due to multiple edge dislocations in a 
    crystal lattice, given the lab coordinates and accompanying rotation matrices.
    The dislocation wall will start at the center of the system and branch out as 
    more is added. If more than 100 dislocations are needed the function will call
    multi_dislocs_parallel to run with multiprocessing for saving time. This requires
    a lot of memory.
    -----------------------------------------------------------------------------
    Parameters:
    rl: np.ndarray of shape (3, X)
        An array of coordinates in lab space, with X being the number of steps in 
        each dimension. rl contains the x_lab, y_lab, z_lab coordinates.
    Ud: np.ndarray of shape (3, 3)
        A rotational matrix used for going from dislocation space to grain space.
    Us: np.ndarray of shape (3, 3)
        A rotational matrix used for going from grain space to sample space.
    Theta: np.ndarray of shape (3, 3)
        A rotational matrix used for going from sample space to lab space.
    dis : int, optional
        Distance in micrometer between each edge dislocation in the dislocation coordinate system. 
        Default is 1.
    ndis : int, optional
        Number of edge dislocations to include. 
        Default is 1.
    b : float, optional
        Magnitude of the Burger's vector. 
        Default is 2.862e-4 [micrometer] - 2.862 [Aangstrom].
    ny : float, optional
        Poisson ratio. Default is 0.334.
        
    Returns
    -----------------------------------------------------------------------------
    numpy.ndarray
        Displacement gradient field induced by set of edge dislocations, 
        in grain space. Shape is (X, 3, 3).
    '''
    
    if misorientation == True:
        m_ori = b/(dis*1e-6)
        U_sr = rotatedU(t_vec,  m_ori/2, Us, 1)
        U_sl = rotatedU(t_vec, -m_ori/2, Us, 1)
        
        rs = Theta @ rl # sample
        left_half1 = (rs[0] < 0) & (rs[1] > 0)
        right_half1 = (rs[0] >= 0) & (rs[1] > 0)
        rc = U_sl.T @ rs[:, left_half1] & U_sr.T @ rs[:, right_half1] # crystal

        rd = Ud.T @ rc # dislocation


    # Rotate lab coordinates to dislocation coordinates
    rs = Theta @ rl # sample
    rc = Us.T @ rs # crystal
    rd = Ud.T @ rc # dislocation

    # Initialize 3D tensor for the dislocation-induced deformation gradient field.
    Fdd = np.zeros([len(rd[-1]), 3, 3])
    alpha = 1e-20

    # Calculate intermediate values for deformation gradient tensor calculation.
    sqx = rd[0] * rd[0]
    sqy = rd[1] * rd[1]
    denom = (sqx + sqy) * (sqx + sqy) + alpha
    bfactor = b / (4 * np.pi * (1 - ny))
    nyfactor = 2 * ny * (sqx + sqy)

    # Calculate components of the deformation gradient tensor.
    Fdd[:, 0, 0] = -rd[1] * (3 * sqx + sqy - nyfactor) / denom
    Fdd[:, 0, 1] =  rd[0] * (3 * sqx + sqy - nyfactor) / denom
    Fdd[:, 1, 0] = -rd[0] * (3 * sqy + sqx - nyfactor) / denom
    Fdd[:, 1, 1] =  rd[1] * (sqx - sqy - nyfactor) / denom

    if ndis > 100:
        # Determine the chunk size based on the total number of iterations and the number of jobs
        njobs = cpu_count() # number of processes
        chunk_size = int(np.ceil((ndis - 1) / njobs))

        # Create chunks of iterations
        chunks = [range(start, min(start + chunk_size, ndis)) for start in range(1, ndis, chunk_size)]
        print(chunks)
        # Parallelize the loop using joblib with workload balancing
        # results = Parallel(n_jobs=njobs)(delayed(multi_dislocs_parallel)(chunk, rd, Fdd.shape, dis) for chunk in chunks)
        
        # Create a ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=njobs) as executor:
            results = list(executor.map(multi_dislocs_parallel, chunks, [rd] * len(chunks), [Fdd.shape] * len(chunks), [dis] * len(chunks)))


        # Accumulate the results into the Fdd array
        for Fdd_i in results:
            Fdd += Fdd_i
    elif ndis <= 100:
        # Calculate deformation gradient tensor for multiple edge dislocations, if needed.
        count1, count2, = 1, 1
        for i in tqdm(range(1, ndis)):
            rd_new = np.copy(rd[:2])

            # For even numbers go in negative yd direction
            if i%2 == 0:
                rd_new[1] -= (count1 * dis)
                count1 += 1
            
            # For uneven numbers go in positive yd direction
            if i%2 == 1:
                rd_new[1] += (count2 * dis)
                count2 += 1
            
            # Calculate intermediate values for deformation gradient tensor calculation.
            sqx = rd_new[0] * rd_new[0]
            sqy = rd_new[1] * rd_new[1]
            denom = (sqx + sqy) * (sqx + sqy)
            nyfactor = 2 * ny * (sqx + sqy)

            # Add the components of the deformation gradient tensor 
            # for each new dislocation.
            Fdd[:, 0, 0] += -rd_new[1] * (3 * sqx + sqy - nyfactor) / denom
            Fdd[:, 0, 1] +=  rd_new[0] * (3 * sqx + sqy - nyfactor) / denom
            Fdd[:, 1, 0] += -rd_new[0] * (3 * sqy + sqx - nyfactor) / denom
            Fdd[:, 1, 1] +=  rd_new[1] * (sqx - sqy - nyfactor) / denom

    # Finish the calculation of Fdd
    Fdd *= bfactor
    Fdd += np.identity(3)
    return Ud @ Fdd @ Ud.T # Return the rotated Fdd -> Fg


def load_or_generate_Hg(rl, Ud, Us, Theta, dis, ndis, file_path=None):
    if file_path is not None:
        fname = file_path.rsplit('/', 1)[-1]
        try:
            # Load the Fg array from the .npy file
            Fg = np.load(file_path)
            print(f'Loaded Fg from {fname}')
        except FileNotFoundError:
            print(f"File '{fname}' not found. \nGenerating a new Fg array.")
            if ndis == 0:
                # If ndis is zero, create Fg with the second method
                Fg = np.zeros([len(rl[-1]), 3, 3])
                Fg += np.identity(Fg.shape[1])
            else:
                # Otherwise, generate Fg using Fd_find
                Fg = Fd_find(rl * 1e6, Ud, Us, Theta, dis, ndis)

            if file_path is not None:
                # Save the generated Fg array to a .npy file
                np.save(file_path, Fg)
                print(f'Saved Fg to {fname}')

    else:
        # If no file_path is provided, generate a new Fg array
        if ndis == 0:
            # If ndis is zero, create Fg with the second method
            Fg = np.zeros([len(rl[-1]), 3, 3])
            Fg += np.identity(Fg.shape[1])
        else:
            # Otherwise, generate Fg using Fd_find
            Fg = Fd_find(rl * 1e6, Ud, Us, Theta, dis, ndis)

    # Calculate Hg based on the provided code snippet
    Hg = np.transpose(fast_inverse2(Fg),[0,2,1])
    Hg -= np.identity(3)
    
    return Hg



def square(list):
    return [i * i for i in list]

def m_norm(list):
    return list / np.sqrt(np.sum(square(list)))

def image_range(im_in, lower, upper):
    return np.interp(im_in, (np.min(im_in), np.max(im_in)), (lower, upper))

# def Fd_find(rl, Ud, Us, Theta, dis = 1, ndis = 1, b = 2.862*1e-4, ny = 0.334, sign = 1):
#     xl, yl, zl = rl
#     Fdd = np.zeros([len(xl), 3, 3])
#     print(Ud, Us, Theta)
#     rd = Ud.T @ Us.T @ Theta @ np.array([xl, yl, zl])*1e6
#     # dx, dy =  * sign,  * sign
#     sqx = rd[0] * rd[0]
#     sqy = rd[1] * rd[1]
#     denom = (sqx + sqy) * (sqx + sqy)
#     bfactor = b / (4 * np.pi * (1 - ny))
#     nyfactor = 2 * ny * (sqx + sqy)
#     Fdd[:, 0, 0] = -rd[1] * (3 * sqx + sqy - nyfactor) / denom
#     Fdd[:, 0, 1] =  rd[0] * (3 * sqx + sqy - nyfactor) / denom
#     Fdd[:, 1, 0] = -rd[0] * (3 * sqy + sqx - nyfactor) / denom
#     Fdd[:, 1, 1] =  rd[1] * (sqx - sqy - nyfactor) / denom

#     if ndis > 1:
#         count1, count2, = 1, 1
#         for i in range(1, ndis):
#             rd_new = np.copy(rd[:2])
#             # rd = np.asarray([xg, yg, zg])
#             if i%2 == 0:
#                 #rd_new[0] -= (count1 * np.sqrt((dis * dis) / 2))
#                 rd_new[1] -= (count1 * dis)
#                 count1 += 1
#             if i%2 == 1:
#                 #rd_new[0] += (count2 * np.sqrt((dis * dis) / 2))
#                 rd_new[1] += (count2 * dis)
#                 count2 += 1
#             # rd = Ud @ rd_new

#             # dx, dy = rd_new[0] * sign, rd_new[1] * sign
#             sqx = rd_new[0] * rd_new[0]
#             sqy = rd_new[1] * rd_new[1]
#             denom = (sqx + sqy) * (sqx + sqy)
#             nyfactor = 2 * ny * (sqx + sqy)
#             Fdd[:, 0, 0] += -rd_new[1] * (3 * sqx + sqy - nyfactor) / denom
#             Fdd[:, 0, 1] +=  rd_new[0] * (3 * sqx + sqy - nyfactor) / denom
#             Fdd[:, 1, 0] += -rd_new[0] * (3 * sqy + sqx - nyfactor) / denom
#             Fdd[:, 1, 1] +=  rd_new[1] * (sqx - sqy - nyfactor) / denom
#         del rd, rd_new, sqx, sqy, denom
#     gc.collect()
#     Fdd *= bfactor
#     Fdd += np.identity(3)
#     return Ud @ Fdd @ Ud.T

def Fd_find_domain(xl, yl, zl, Ud, Us, Theta, dis = 1, ndis = 1, b = 3.507*1e-4, ny = 0.334, sign = 1):
    Fdd = np.zeros([len(xl), 3, 3])
    Fdd += np.identity(3)
    num_b1, num_n1, num_t1 = [-1, -1, 0], [-1, 1, 1], [ 1, -1,  -2]
    Ud1 = np.asarray([m_norm(num_b1), m_norm(num_n1), m_norm(num_t1)])
    for mm in range(4):
        if mm == 0:
            mats = np.asarray([xl, yl, zl])*1e6
            mats[0] += 80
            rd = Ud.T @ Us.T @ Theta @ mats
            dx, dy = rd[0] * sign, rd[1] * sign
            sqx = dx * dx
            sqy = dy * dy
            denom = (sqx + sqy) * (sqx + sqy)
            bfactor = b / (4 * np.pi * (1 - ny))
            nyfactor = 2 * ny * (sqx + sqy)
            Fdd[:, 0, 0] += bfactor * (-dy * (3 * sqx + sqy - nyfactor) / denom)
            Fdd[:, 0, 1] += bfactor * ( dx * (3 * sqx + sqy - nyfactor) / denom)
            Fdd[:, 1, 0] += bfactor * (-dx * (3 * sqy + sqx - nyfactor) / denom)
            Fdd[:, 1, 1] += bfactor * ( dy * (sqx - sqy - nyfactor) / denom)
            if ndis > 1:
                count1, count2, = 1, 1
                for i in range(1, ndis):
                    rd_new = np.copy(rd)
                    if i%2 == 0:
                        rd_new[1] -= (count1 * dis)
                        count1 += 1
                    if i%2 == 1:
                        rd_new[1] += (count2 * dis)
                        count2 += 1

                    dx, dy = rd_new[0] * sign, rd_new[1] * sign
                    sqx = dx * dx
                    sqy = dy * dy
                    denom = (sqx + sqy) * (sqx + sqy)
                    nyfactor = 2 * ny * (sqx + sqy)
                    Fdd[:, 0, 0] += bfactor * (-dy * (3 * sqx + sqy - nyfactor) / denom )
                    Fdd[:, 0, 1] += bfactor * ( dx * (3 * sqx + sqy - nyfactor) / denom )
                    Fdd[:, 1, 0] += bfactor * (-dx * (3 * sqy + sqx - nyfactor) / denom )
                    Fdd[:, 1, 1] += bfactor * ( dy * (sqx - sqy - nyfactor) / denom )
        if mm == 1:
            mats = np.asarray([xl, yl, zl])*1e6
            mats[0] -= 80
            rd = Ud1.T @ Us.T @ Theta @ mats
            dx, dy = rd[0] * sign, rd[1] * sign
            sqx = dx * dx
            sqy = dy * dy
            denom = (sqx + sqy) * (sqx + sqy)
            bfactor = b / (4 * np.pi * (1 - ny))
            nyfactor = 2 * ny * (sqx + sqy)
            if ndis > 1:
                count1, count2, = 1, 1
                for i in range(1, ndis):
                    rd_new = np.copy(rd)
                    if i%2 == 0:
                        rd_new[1] -= (count1 * dis)
                        count1 += 1
                    if i%2 == 1:
                        rd_new[1] += (count2 * dis)
                        count2 += 1

                    dx, dy = rd_new[0] * sign, rd_new[1] * sign
                    sqx = dx * dx
                    sqy = dy * dy
                    denom = (sqx + sqy) * (sqx + sqy)
                    nyfactor = 2 * ny * (sqx + sqy)
                    Fdd[:, 0, 0] += bfactor * (-dy * (3 * sqx + sqy - nyfactor) / denom )
                    Fdd[:, 0, 1] += bfactor * ( dx * (3 * sqx + sqy - nyfactor) / denom )
                    Fdd[:, 1, 0] += bfactor * (-dx * (3 * sqy + sqx - nyfactor) / denom )
                    Fdd[:, 1, 1] += bfactor * ( dy * (sqx - sqy - nyfactor) / denom )
        if mm == 2:
            num_b2, num_n2, num_t2 = [-1, 1, 0], [-1, -1, 1], [ 1, 1,  2]
            Ud2 = np.asarray([m_norm(num_b2), m_norm(num_n2), m_norm(num_t2)])
            mats = np.asarray([xl, yl, zl])*1e6
            mats[0] -= 80
            rd = Ud.T @ Us.T @ Theta @ mats
            dx, dy = rd[0] * sign, rd[1] * sign
            sqx = dx * dx
            sqy = dy * dy
            denom = (sqx + sqy) * (sqx + sqy)
            bfactor = b / (4 * np.pi * (1 - ny))
            nyfactor = 2 * ny * (sqx + sqy)
            Fdd[:, 0, 0] += bfactor * (-dy * (3 * sqx + sqy - nyfactor) / denom)
            Fdd[:, 0, 1] += bfactor * ( dx * (3 * sqx + sqy - nyfactor) / denom)
            Fdd[:, 1, 0] += bfactor * (-dx * (3 * sqy + sqx - nyfactor) / denom)
            Fdd[:, 1, 1] += bfactor * ( dy * (sqx - sqy - nyfactor) / denom)
            if ndis > 1:
                count1, count2, = 1, 1
                for i in range(1, ndis):
                    rd_new = np.copy(rd)
                    if i%2 == 0:
                        rd_new[1] -= (count1 * dis)
                        count1 += 1
                    if i%2 == 1:
                        rd_new[1] += (count2 * dis)
                        count2 += 1

                    dx, dy = rd_new[0] * sign, rd_new[1] * sign
                    sqx = dx * dx
                    sqy = dy * dy
                    denom = (sqx + sqy) * (sqx + sqy)
                    nyfactor = 2 * ny * (sqx + sqy)
                    Fdd[:, 0, 0] += bfactor * (-dy * (3 * sqx + sqy - nyfactor) / denom )
                    Fdd[:, 0, 1] += bfactor * ( dx * (3 * sqx + sqy - nyfactor) / denom )
                    Fdd[:, 1, 0] += bfactor * (-dx * (3 * sqy + sqx - nyfactor) / denom )
                    Fdd[:, 1, 1] += bfactor * ( dy * (sqx - sqy - nyfactor) / denom )    
        if mm == 3:
            num_b3, num_n3, num_t3 = [1, 1, 0], [-1, 1, 1], [ 1, -1,  -2]
            Ud3 = np.asarray([m_norm(num_b3), m_norm(num_n3), m_norm(num_t3)])
            mats = np.asarray([xl, yl, zl])*1e6
            mats[0] += 80
            rd = Ud1.T @ Us.T @ Theta @ mats
            dx, dy = rd[0] * sign, rd[1] * sign
            sqx = dx * dx
            sqy = dy * dy
            denom = (sqx + sqy) * (sqx + sqy)
            bfactor = b / (4 * np.pi * (1 - ny))
            nyfactor = 2 * ny * (sqx + sqy)
            if ndis > 1:
                count1, count2, = 1, 1
                for i in range(1, ndis):
                    rd_new = np.copy(rd)
                    if i%2 == 0:
                        rd_new[1] -= (count1 * dis)
                        count1 += 1
                    if i%2 == 1:
                        rd_new[1] += (count2 * dis)
                        count2 += 1

                    dx, dy = rd_new[0] * sign, rd_new[1] * sign
                    sqx = dx * dx
                    sqy = dy * dy
                    denom = (sqx + sqy) * (sqx + sqy)
                    nyfactor = 2 * ny * (sqx + sqy)
                    Fdd[:, 0, 0] += bfactor * (-dy * (3 * sqx + sqy - nyfactor) / denom )
                    Fdd[:, 0, 1] += bfactor * ( dx * (3 * sqx + sqy - nyfactor) / denom )
                    Fdd[:, 1, 0] += bfactor * (-dx * (3 * sqy + sqx - nyfactor) / denom )
                    Fdd[:, 1, 1] += bfactor * ( dy * (sqx - sqy - nyfactor) / denom )
    Fg = Ud @ Fdd @ Ud.T
    return Fg


def norm(data):
    return np.sqrt(np.sum(square(data)))

@jit('float64[:,:,:](float64[:,:,:])', nopython=True,fastmath = True)
def fast_inverse2(A): # Try to rewrite this
    inv = np.empty_like(A)
    a = A[:,0, 0]
    b = A[:,0, 1]
    c = A[:,0, 2]
    d = A[:,1, 0]
    e = A[:,1, 1]
    f = A[:,1, 2]
    g = A[:,2, 0]
    h = A[:,2, 1]
    i = A[:,2, 2]

    inv[:,0, 0] = (e * i - f * h)
    inv[:,1, 0] = -(d* i - f * g)
    inv[:,2, 0] = (d * h - e * g)
    inv_det = 1 / (a * inv[:,0, 0] + b * inv[:,1, 0] + c * inv[:,2, 0])

    inv[:,0, 0] *= inv_det
    inv[:,0, 1] = - inv_det * (b * i - c * h)
    inv[:,0, 2] = inv_det * (b * f - c * e)
    inv[:,1, 0] *= inv_det
    inv[:,1, 1] = inv_det * (a * i - c * g)
    inv[:,1, 2] = -inv_det * (a * f - c * d)
    inv[:,2, 0] *= inv_det
    inv[:,2, 1] = -inv_det * (a * h - b * g)
    inv[:,2, 2] = inv_det * (a * e - b * d)
    return inv

def repeat(arr, count):
    return np.stack([arr for _ in range(count)], axis=0)
