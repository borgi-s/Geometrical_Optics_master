"""Moments and FWHM analysis of DFXM image stacks."""

import numpy as np


def fastgrainplot(
    imagestack: np.ndarray,
    vlist: np.ndarray,
    ulist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
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
    """
    imglist = imagestack

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

    for j in range(len(imglist)):
        img = imglist[j]
        img[img < 0] = 0

        # For grain shape
        inttot += img

        # For calculating moments.
        # vlist is the fast motor (inner loop), ulist is the slow motor (outer loop).
        vv = vlist[j % len(vlist)]  # fast motor: cycles through all v steps
        uu = ulist[j // len(vlist)]  # slow motor: advances once per full v sweep

        v1sum += vv * img
        v2sum += vv**2 * img
        v3sum += vv**3 * img
        v4sum += vv**4 * img

        u1sum += uu * img
        u2sum += uu**2 * img
        u3sum += uu**3 * img
        u4sum += uu**4 * img

        # For orientation distribution
        oridist[j] = img.sum()

    # Expectation value
    vnorm = v1sum / (inttot * 2)
    unorm = u1sum / (inttot * 2)

    # Variance
    vvar = v2sum / inttot * 2 - vnorm**2
    uvar = u2sum / inttot * 2 - unorm**2

    # Replace negative variances with NaN
    vvar[vvar <= 0] = np.nan
    uvar[uvar <= 0] = np.nan

    # FWHM
    vfwhm = 2.355 * np.sqrt(vvar)
    ufwhm = 2.355 * np.sqrt(uvar)

    # Skewness and kurtosis were computed here but never returned or used.
    # Removed during cleanup; reinstate via git history if you need them
    # and route through the return tuple.
    return unorm, vnorm, ufwhm, vfwhm


def calc_moments(
    image: np.ndarray,
    u_range: float,
    v_range: float,
    u_steps: int,
    v_steps: int,
) -> dict[str, float]:
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
    u, v = np.mgrid[-u_range : u_range : complex(u_steps), -v_range : v_range : complex(v_steps)]

    moments = {}
    moments["mean_v"] = np.sum(v * image) / np.sum(image)
    moments["mean_u"] = np.sum(u * image) / np.sum(image)

    # raw or spatial moments
    moments["m00"] = np.sum(image) * 2
    moments["m01"] = np.sum(u * image)
    moments["m10"] = np.sum(v * image)
    moments["m02"] = np.sum(u**2 * image)
    moments["m20"] = np.sum(v**2 * image)

    # central moments
    moments["mu02"] = np.sum((u - moments["mean_u"]) ** 2 * image)  # variance
    moments["mu20"] = np.sum((v - moments["mean_v"]) ** 2 * image)  # variance
    moments["mu03"] = np.sum((u - moments["mean_u"]) ** 3 * image)
    moments["mu30"] = np.sum((v - moments["mean_v"]) ** 3 * image)

    # central standardized or normalized or scale invariant moments
    moments["nu03"] = moments["mu03"] / np.sum(image) ** (3 / 2 + 1)  # skewness
    moments["nu30"] = moments["mu30"] / np.sum(image) ** (3 / 2 + 1)  # skewness
    return moments
