import os
import numpy as np
import fabio
from scipy.ndimage import center_of_mass
from scipy.interpolate import griddata
from functions import check_folder
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from direct_space.forward_model import forward, Find_Hg

def save_image(args):
    '''
    Save an image with specified parameters.
    ---------------------------------------------------------------------
    Parameters:
        args (tuple): A tuple containing the following elements:
            Hg (float): A parameter.
            phi (float): Angle in radians.
            chi (float): Angle in radians.
            j (int): Step in phi
            i (int): Step in chi
            fpath (str): Path to a folder where the image will be saved.
            fn_prefix (str): Prefix for the image filename.
            ftype (str): Filetype extension for the image (e.g., '.png').
    ---------------------------------------------------------------------
    Returns:
        None: The function saves an image but does not return a value.
    '''
    Hg, phi, chi, j, i, fpath, fn_prefix, ftype = args
    im, qi_field = forward(Hg, phi=phi, chi=chi)
    fn_suffix = ("{0}".format(i).zfill(4) + "_" +
                         "{0}".format(j).zfill(4) + ftype)
    np.save(os.path.join(fpath + fn_prefix + fn_suffix), im)

def save_images_parallel(Hg, phi_range, phi_steps, chi_range, chi_steps, fpath, fn_prefix, ftype):
    '''
    Generate a grid of parameter combinations and save images in parallel.
    ----------------------------------------------------------------------------------------------
    Parameters:
        Hg (float): A parameter.
        phi_range (float): Range of phi values in degrees.
        phi_steps (int): Number of steps in the phi range.
        chi_range (float): Range of chi values in degrees.
        chi_steps (int): Number of steps in the chi range.
        fpath (str): Path to a folder where the images will be saved.
        fn_prefix (str): Prefix for the image filenames.
        ftype (str): Filetype extension for the images (e.g., '.png').
    ----------------------------------------------------------------------------------------------
    Returns:
        True (bool): True if the function completes successfully.
    '''
    Phi = np.linspace(-np.deg2rad(phi_range), np.deg2rad(phi_range), phi_steps)
    Chi = np.linspace(-np.deg2rad(chi_range), np.deg2rad(chi_range), chi_steps)

    # Create a folder in the specified path if it doesn't exist
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    args_list = [(Hg, Phi[j], Chi[i], j, i, fpath, fn_prefix, ftype) for i in range(chi_steps) for j in range(phi_steps)]

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(save_image, args_list), total=len(args_list)))

    return True

def inv_polefigure_colors(o_grid, test_grid, float_bit=np.float16):
    """
    Maps the RGB values for a color gradient onto a grid of points.
    ------------------------------------------------------------------------------------------------------------------
    Parameters:
        o_grid (numpy array, dtype: object): Np.array containing the angular data for the original grid (2D).
        test_grid (numpy array, dtype: object): Np.array containing the angular data for the test grid (2D).
        float_bit (numpy dtype, optional): Floating point precision for RGB values.
    ------------------------------------------------------------------------------------------------------------------
    Returns:
        xy_colors_griddata (numpy.ndarray): Array containing the RGB and alpha values for each point in the original grid.
        xydata (numpy.ndarray): Array containing the coordinate data for each point in the original grid.
    """
    
    # Define the RGB values for the color gradient
    key_xy_RGBs = np.array([
        [1, 1, 1, 1],    # White
        [0, 1, 1, 1],    # Cyan
        [0, 1, 0, 1],    # Green
        [1, 0.65, 0, 1], # Orange
        [1, 0, 0, 1],    # Red
        [0, 0, 1, 1],    # Blue
        [1, 0, 0.5, 1]   # Magenta
    ], dtype=float_bit)

    # Extract the angular data for the original and test grids
    o_chi, o_diffry = o_grid[0], o_grid[1]
    test_chi, test_diffry = test_grid[0], test_grid[1]
    
    # Define the coordinates for the RGB values
    key_xy_points = np.array([
        [0, 0],                          # White center
        [0, np.min(test_diffry)],        # Cyan min y-axis center x-axis
        [np.max(test_chi), np.min(test_diffry)], # Green min y-axis max x-axis
        [np.max(test_chi), np.max(test_diffry)], # Orange max y-axis max x-axis
        [0, np.max(test_diffry)],   # Red max y-axis center x-axis
        [np.min(test_chi), np.min(test_diffry)], # Blue min y-axis min x-axis
        [np.min(test_chi), np.max(test_diffry)]  # Magenta max y-axis min x-axis
    ], dtype=float_bit)
        
    # Create a 2D array of all coordinate combinations
    xydata = np.array([(x, y) for x in o_chi for y in o_diffry], dtype=float_bit)
    
    # Map the RGB values onto the 2D array of coordinates using griddata
    reds = griddata(key_xy_points, key_xy_RGBs.T[0], xydata)
    greens = griddata(key_xy_points, key_xy_RGBs.T[1], xydata)
    blues = griddata(key_xy_points, key_xy_RGBs.T[2], xydata)
    alphas = griddata(key_xy_points, key_xy_RGBs.T[3], xydata)
    xy_colors_griddata = np.vstack((reds, greens, blues, alphas)).T
    
    # Force the RGB values to be within range [0, 1]
    xy_colors_griddata[xy_colors_griddata<0] = 0.0
    xy_colors_griddata[xy_colors_griddata>1] = 1.0
    
    return xy_colors_griddata, xydata



def save_images(*args):
    '''
    Saves images generated by the forward function for different phi and chi values in a specified folder path.
    -------------------------------------------------------------------------------------------------------------
    Parameters:
    *args: A tuple of function parameters in the following order:
        phi_r (float): Range of phi values to be iterated over
        phi_s (int): Number of steps for phi values in the range
        chi_r (float): Range of chi values to be iterated over
        chi_s (int): Number of steps for chi values in the range
        fpath (str): Path of the folder to save the images in
        fn_prefix (str): Prefix to be added to the filename of saved images
        ftype (str): File type extension of the saved images
    -------------------------------------------------------------------------------------------------------------
    Returns:
    boolean: True on successful completion
    '''
    
    phi_r, phi_s, chi_r, chi_s, fpath, fn_prefix, ftype = args
    Phi = np.linspace(-np.deg2rad(phi_r), np.deg2rad(phi_r), phi_s)
    Chi = np.linspace(-np.deg2rad(chi_r), np.deg2rad(chi_r), chi_s)
    
    # Create a folder in the specified path if it doesn't exist
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    print('Starting')
    # Iterate through all phi and chi values and save corresponding images
    for i in tqdm(range(chi_s)):
        for j in range(phi_s):
            im = forward(phi = Phi[j], chi = Chi[i])
            fn_suffix = ("{0}".format(i).zfill(4) + "_" +
                         "{0}".format(j).zfill(4) + ftype)
            np.save(fpath + fn_prefix + fn_suffix, im)
            # print(fn_suffix, ': Done')
    return True

def fastgrainplot(imagestack, vlist, ulist):
    '''
    Direct translation from original MATLAB code to python calculating the moments of a pixel in angular space.
    ------------------------------------------------------------------------------------------------------------
    Parameters:
        imagestack (ndarray): stack of images as a 3D numpy array.
        vlist (ndarray): range of first angular dimension as 2D numpy array.
        ulist (ndarray): range of second angular dimension as 2D numpy array.
    ------------------------------------------------------------------------------------------------------------
    Returns:
        unorm (ndarray): Center of mass map in u made from stack
        vnorm (ndarray): Center of mass map in v made from stack
        ufwhm (ndarray): FWHM map in u made from stack
        vfwhm (ndarray): FWHM map in v made from stack
    '''
    imglist = imagestack

    # Initialize some variables

    oridist = np.zeros(len(vlist) * len(ulist))
    inttot = np.zeros_like(imglist[0])
    v1sum = np.zeros_like(inttot)
    v2sum = np.zeros_like(inttot)
    v3sum = np.zeros_like(inttot)
    v4sum = np.zeros_like(inttot)
    u1sum = np.zeros_like(inttot)
    u2sum = np.zeros_like(inttot)
    u3sum = np.zeros_like(inttot)
    u4sum = np.zeros_like(inttot)

    # Loop over the images
    for j in range(len(imglist)):
        img = imglist[j]
        img[img < 0] = 0

        # For grain shape
        inttot += img

        # For calculating moments
        vv = vlist[j % len(vlist)] # fast motor
        uu = ulist[j // len(ulist)] # slow motor

        v1sum += vv * img
        v2sum += vv ** 2 * img
        v3sum += vv ** 3 * img
        v4sum += vv ** 4 * img

        u1sum += uu * img
        u2sum += uu ** 2 * img
        u3sum += uu ** 3 * img
        u4sum += uu ** 4 * img

        # For orientation distribution
        oridist[j] = img.sum()

    # Expectation value
    vnorm = v1sum / (inttot * 2)
    unorm = u1sum / (inttot * 2)

    # Variance
    vvar = v2sum / inttot * 2 - vnorm ** 2
    uvar = u2sum / inttot * 2 - unorm ** 2

    # Replace negative variances with NaN
    vvar[vvar <= 0] = np.nan
    uvar[uvar <= 0] = np.nan

    # FWHM
    vfwhm = 2.355 * np.sqrt(vvar)
    ufwhm = 2.355 * np.sqrt(uvar)

    # Skewness
    vskew = (v3sum / inttot - 3 * vnorm * vvar - vnorm ** 3) / vvar ** (3 / 2)
    uskew = (u3sum / inttot - 3 * unorm * uvar - unorm ** 3) / uvar ** (3 / 2)

    # Kurtosis
    vkurt = (v4sum/inttot - 4*vnorm*v3sum/inttot + 6*vnorm**2*v2sum/inttot - 3*vnorm**4)/(vvar**2)
    ukurt = (u4sum/inttot - 4*unorm*u3sum/inttot + 6*unorm**2*u2sum/inttot - 3*unorm**4)/(uvar**2)
    # unormnn = unorm[~np.isnan(unorm)]
    # vnormnn = vnorm[~np.isnan(vnorm)]
    # sort1 = np.sort(unormnn.flatten())
    # sort2 = np.sort(vnormnn.flatten())
    # Imin1 = sort1[int(round(len(sort1)*0.02))]
    # Imax1 = sort1[int(round(len(sort1)*0.98))]
    # Imin2 = sort2[int(round(len(sort2)*0.02))]
    # Imax2 = sort2[int(round(len(sort2)*0.98))]
    # C = np.zeros(imglist[0].shape+(3,))
    # C[...,0] = (unorm-Imin1)/(Imax1-Imin1)
    # C[...,1] = (vnorm-Imin2)/(Imax2-Imin2)
    # C[...,2] = np.ones(imglist[0].shape)
    # C[np.isnan(C)] = 0
    # C[C < 0] = 0
    # C[C > 1] = 1
    return unorm, vnorm, ufwhm, vfwhm

def calc_moments(image, u_range, v_range, u_steps, v_steps):
    """
    Calculate raw, central, and standardized moments of a given image.
    The image should be a single pixel's orientation spread, in e.g. chi and phi.
    --------------------------------------------------------------------------------------------------------
    Parameters:
        image (ndarray): Input image as a 2D NumPy array.
        u_range (float): Range of u values to use for moment calculation. Default is phi_range.
        v_range (float): Range of v values to use for moment calculation. Default is chi_range.
        u_steps (int): Number of steps to use for u values. Default is phi_steps.
        v_steps (int): Number of steps to use for v values. Default is chi_steps.
    --------------------------------------------------------------------------------------------------------
    Returns:
        moments (dict): Dictionary of calculated moments.
    """
    u,v = np.mgrid[-u_range:u_range:complex(u_steps), 
                   -v_range:v_range:complex(v_steps)]
    
    moments = {}
    moments['mean_v'] = np.sum(v*image)/np.sum(image)
    moments['mean_u'] = np.sum(u*image)/np.sum(image)
    
    # raw or spatial moments
    moments['m00'] = np.sum(image)*2
    moments['m01'] = np.sum(u*image)
    moments['m10'] = np.sum(v*image)
    # moments['m11'] = np.sum(u*v*image)
    moments['m02'] = np.sum(u**2*image)
    moments['m20'] = np.sum(v**2*image)
    # moments['m12'] = np.sum(v*u**2*image)
    # moments['m21'] = np.sum(v**2*u*image)
    # moments['m03'] = np.sum(v**3*image)
    # moments['m30'] = np.sum(u**3*image)

    # central moments
    # moments['mu01']= np.sum((u-moments['mean_u'])*image) # should be 0
    # moments['mu10']= np.sum((v-moments['mean_v'])*image) # should be 0
    # moments['mu11'] = np.sum((v-moments['mean_v'])*(u-moments['mean_u'])*image)
    moments['mu02'] = np.sum((u-moments['mean_u'])**2*image) # variance
    moments['mu20'] = np.sum((v-moments['mean_v'])**2*image) # variance
    # moments['mu12'] = np.sum((v-moments['mean_v'])*(u-moments['mean_u'])**2*image)
    # moments['mu21'] = np.sum((v-moments['mean_v'])**2*(u-moments['mean_u'])*image) 
    moments['mu03'] = np.sum((u-moments['mean_u'])**3*image) 
    moments['mu30'] = np.sum((v-moments['mean_v'])**3*image) 
    
    # central standardized or normalized or scale invariant moments
    # moments['nu11'] = moments['mu11'] / np.sum(image)**(2/2+1)
    # moments['nu12'] = moments['mu12'] / np.sum(image)**(3/2+1)
    # moments['nu21'] = moments['mu21'] / np.sum(image)**(3/2+1)
    # moments['nu20'] = moments['mu20'] / np.sum(image)**(2/2+1)
    moments['nu03'] = moments['mu03'] / np.sum(image)**(3/2+1) # skewness
    moments['nu30'] = moments['mu30'] / np.sum(image)**(3/2+1) # skewness
    return moments

def save_edfs(imstack, v, u, fpath, fn_prefix):
    """
    Save EDF images from a given image stack and a set of v and u angles.
    ---------------------------------------------------------------------------
    Parameters:
        - image_stack: a 2D array of images
        - v_angles: a 1D array of v angles in degrees
        - u_angles: a 1D array of u angles in degrees
        - f_path: a string representing the path of the output file
        - fn_prefix: a string representing the prefix of the output file name
    """
    for i in range(len(v)):
        for j in range(len(u)):
            img1 = fabio.edfimage.EdfImage(data=imstack[j,i])
            chi1 = str((v[i]*180/np.pi))
            phi1 = str((u[j]*180/np.pi))
            img1.header = {
                'HeaderID': 'EH:000001:000000:000000',
                'Image': '1',
                'ByteOrder': 'LowByteFirst',
                'DataType': 'UnsignedShort',
                'Dim_1': '170',
                'Dim_2': '510',
                'Size': '11059200',
                'scan': 'mesh  diffry 8.535 8.555 10  chi 0.116607 0.236607 6  1',
                'date': 'Mon Dec 12 22:05:32 2022',
                'epoch': '1670604623.6910',
                'offset': '0',
                'count_time': '1',
                'point_no': '5',
                'scan_no': '1573',
                'preset': '0',
                'col_end': '2559',
                'col_beg': '0',
                'row_end': '2159',
                'row_beg': '0',
                'counter_pos': '-1 39.4025 1.24457e-07 1.27043e-09 1.17154e-13 2.40476e-11 -1 -1 0 98.7543 10.3245 0 19886 0.396031 0 0 0 0 0 -1 -1 0 1 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0',
                'counter_mne': 'diffstd srcur pico1 pico2 pico3 pico4 nfavg nfstd europv ffavg ffstd oxtemp p201_1 mc5 mc6 delV delI samV samI basavg basstd horstd sec horavg diffavg foc1 foc2 foc3 foc4 xrfSum roiCa roiSr devolt decurr depow lstemp zap_avg Tempwsp Tempsp Temp Tempout',
                'motor_pos': '480 6.684 7.55132 0.600022 -0.124985 0.600022 -0.103324 4 -0.0485 6 -3.1645 7.28663 12.2036 0 -58.88 0 1618.46 0 0.0015 1.8425 0 -0.822 -0.0015 0.138 342 1.94081e-06 -0.700001 4.08922e-05 0 25 50 256.75 0.1072 83.3046 17.9763 0.144375 -0.009995 8.545 6.38635e-07 0 0.02 2.75495 {0} {1} 3.78 1908.95 91.867 3.03665 -1.12401 10.3341 19.5 0.5 2.11181 0.0499971 0.00647404 -11.4698 -6.12734 2.09967 -3.67327 1.98883 912.776 866 1300 550 150 -30 0 60 50 0 -2.85714e-07 -5000 -5.09849e-05 0.0999796 2.68525e-07 0.0999896 0.00532441 1.052 0.32 7.974 25 150 0 -27 0 0 0 0 0 -3.15866e-07 -0.5 0 0 0 1.438 2.437 9.42503 -2.75239 0.0255 25 0 0'.format(chi1,phi1),
                'motor_mne': 'unused mono x2z s5hg s5ho s5vg s5vo s3vg s3vo s3hg s3ho fffoc1 fffoc2 ffsel ffrotc ffy ffz ffpitch hxx hxy hxz hxyaw hxroll hxpitch cdx cdy cdz cdpitch cdyaw euro_sp icx obx oby obz obpitch obyaw difftz diffry diffrz diffty corz corx chi phi detz dety detx pcofoc hfoc hrotc htth s4hg s4ho s4vg s4vo nffoc s6hg s6ho s6vg s6vo auxx auxy auxz dcx dcy dcz nfrz nfx nfy nfz nfrotc mainx bstop s7hg s7ho s7vg s7vo smx smz smy samfoc furny mrot decoh DEVolt DECurr DEPow lenssel sptmemb wirey wirez delVsp py pz phpz phpy dty dtz focpad Tsp rpi_1 obz3',
                'suffix': '.edf',
                'prefix': 'mosalocal_2x_00_',
                'dir': '/data/id06-hxm/inhouse/2022/run5/ihma320/id06-hxm/Al_sample_5/Al_sample_5_tick_750/mosalocal_2x_00',
                'run': '5',
                'title': 'CCD Image'
                }
            fn_suffix = ("{0}".format(i).zfill(4) + "_" +
                        "{0}".format(j).zfill(4) + ".edf")
            check_folder('', fpath)
            img1.write(fpath+fn_prefix+fn_suffix)
    return True

def load_images(fpath, u_steps, v_steps, file_ext = ".npy"):
    """
    Load and reshape a stack of image files from a directory.
    --------------------------------------------------------------------------
    Parameters:
        fpath (str): Path to directory containing image files.
        u_steps (int): Number of steps to use for u values in reshaped stack.
        v_steps (int): Number of steps to use for v values in reshaped stack.
    --------------------------------------------------------------------------
    Returns:
        stack (ndarray): 3D NumPy array of loaded image files.
        stack_reshape (ndarray): 4D NumPy array of reshaped image stack.
        dim_1 (int): Dimension 1 size of loaded image files.
        dim_2 (int): Dimension 2 size of loaded image files.
    """
    # Check if directory exists and is not empty
    if not os.path.isdir(fpath):
        raise ValueError("Directory does not exist: {}".format(fpath))
    file_list = [f for f in os.listdir(fpath) if f.endswith(file_ext)]
    
    if not file_list:
        raise ValueError("Directory is empty or does not contain any {} files: {}".format(file_ext, fpath))
    
    # Sort file list to ensure consistent order
    file_list.sort()
    # print(file_list)
    # Load images from files into stack array
    stack = np.empty((len(file_list), 
                      *np.load(os.path.join(fpath, 
                                            file_list[0]), 
                                            allow_pickle=True).shape))
    for i, file in enumerate(file_list):
        file_path = os.path.join(fpath, file)
        stack[i, :, :] = np.load(file_path)
    # Reshape stack array
    dim_1, dim_2 = stack.shape[1], stack.shape[2] 
    stack_reshape = stack.reshape((u_steps, v_steps, dim_1, dim_2))
    
    # Return stack, stack_reshape, and dimensions
    return stack, stack_reshape, dim_1, dim_2


def load_image(file_path):
    return np.load(file_path)

def load_images_parallel(fpath, u_steps, v_steps, file_ext=".npy"):
    if not os.path.isdir(fpath):
        raise ValueError("Directory does not exist: {}".format(fpath))
    
    file_list = [f for f in os.listdir(fpath) if f.endswith(file_ext)]
    
    if not file_list:
        raise ValueError("Directory is empty or does not contain any {} files: {}".format(file_ext, fpath))
    
    file_list.sort()
    num_files = len(file_list)
    
    # Initialize a ThreadPoolExecutor with a number of threads
    # You can adjust the number of threads based on your system's capabilities
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Load images from files into a list of numpy arrays using parallel execution
        loaded_images = list(executor.map(lambda file: np.load(os.path.join(fpath, file)), file_list))
    
    # Create an empty stack
    stack = np.empty((num_files, *loaded_images[0].shape))
    
    # Populate the stack with loaded images
    for i, img in enumerate(loaded_images):
        stack[i, :, :] = img
    
    dim_1, dim_2 = stack.shape[1], stack.shape[2] 
    stack_reshape = stack.reshape((v_steps, u_steps, dim_1, dim_2))
    
    return stack, stack_reshape, dim_1, dim_2



