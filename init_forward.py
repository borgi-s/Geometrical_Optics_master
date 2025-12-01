# Import necessary libraries
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import center_of_mass, label
from scipy.interpolate import griddata
from direct_space.forward_model import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from image_processor import *
from matplotlib.colors import ListedColormap
from matplotlib.ticker import ScalarFormatter

def _compute_image(args):
    """
    Worker for parallel computation.
    Returns (j, image) so we can place it correctly.
    NOTE: uses global Theta, base_qc computed beforehand.
    """
    Hg, phi, j = args
    im = forward_from_static(Theta, base_qc, phi=phi)
    return j, im


def compute_rocking_curve_parallel(Hg, phi_range, phi_steps):
    """
    Compute rocking curve in memory, in parallel.

    Returns
    -------
    rocking_curve : np.ndarray
        Shape (phi_steps, Ny, Nx)
    Phi : np.ndarray
        Shape (phi_steps,), radians
    """
    # Phi in radians
    Phi = np.linspace(-np.deg2rad(phi_range), np.deg2rad(phi_range), phi_steps)

    # Probe one image to get shape/dtype
    im0 = forward_from_static(Theta, base_qc, phi=Phi[0])
    im0 = np.asarray(im0)
    img_shape = im0.shape
    img_dtype = im0.dtype

    Ny, Nx = img_shape

    # Preallocate (phi, Ny, Nx)
    rocking_curve = np.zeros((phi_steps, Ny, Nx), dtype=img_dtype)

    # Build args: j (phi index)
    args_list = [(Hg, Phi[j], j) for j in range(phi_steps)]

    # Thread pool – plays nicely with VSCode / Windows
    with ThreadPoolExecutor() as executor:
        for j, im in tqdm(executor.map(_compute_image, args_list),
                          total=len(args_list)):
            rocking_curve[j, ...] = im

    return rocking_curve, Phi

def calc_fwhm_moment_1d(image, u_range, u_steps):
    """
    Moment-based COM and FWHM along a 1D axis u.

    Parameters
    ----------
    image : array-like, shape (u_steps,)
        Intensity vs u (e.g. rocking curve for one pixel).
    u_range : float
        Half-range of u (so u runs from -u_range to +u_range).
        Use radians if you want FWHM in radians.
    u_steps : int
        Number of samples along u.

    Returns
    -------
    mean_u : float
        Center of mass along u.
    fwhm_u : float
        FWHM along u, assuming roughly Gaussian profile.
        np.nan if it cannot be computed.
    """
    image = np.asarray(image, dtype=float).ravel()
    if image.size != u_steps:
        raise ValueError(f"image length {image.size} != u_steps {u_steps}")

    u = np.linspace(-u_range, u_range, u_steps)

    total = image.sum()
    if total <= 0 or not np.isfinite(total):
        return np.nan, np.nan

    mean_u = np.sum(u * image) / total           # first moment
    var_u  = np.sum((u - mean_u)**2 * image) / total  # second central moment

    if var_u <= 0:
        return mean_u, np.nan

    sigma  = np.sqrt(var_u)
    fwhm_u = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    return mean_u, fwhm_u

# Define the range and number of steps for phi and chi
phi_range, phi_steps = 600e-6*180/np.pi, 401

# Phi is redefined inside compute_rocking_curve_parallel, but you can keep this if you want
rocking_curve, Phi = compute_rocking_curve_parallel(Hg, phi_range, phi_steps)

data = rocking_curve          # shape (Nphi, Ny, Nx)

# ----- Central ROI instead of downsampling -----
Nphi, Ny_full, Nx_full = data.shape

roi_h, roi_w = 51*4, 17*4  # Ny, Nx in ROI
cy, cx = Ny_full // 2, Nx_full // 2

y0 = cy - roi_h // 2
y1 = y0 + roi_h
x0 = cx - roi_w // 2
x1 = x0 + roi_w

# safety, in case your image is smaller than requested ROI
y0 = max(y0, 0); x0 = max(x0, 0)
y1 = min(y1, Ny_full); x1 = min(x1, Nx_full)

data_roi    = data[:, y0:y1, x0:x1]     # (Nphi, Ny_roi, Nx_roi)
# com_phi_roi = com_phi[y0:y1, x0:x1]     # (Ny_roi, Nx_roi)

Ny, Nx = data_roi.shape[1:]  # these are now ROI dims
Nphi = Phi.shape[0]



# ----- FWHM map on ROI -----

phi_steps = Nphi
# careful: your phi_range is in *degrees* in this script
phi_range_deg = phi_range
phi_range_rad = np.deg2rad(phi_range_deg)  # half-range in radians

fwhm_phi_roi = np.full((Ny, Nx), np.nan, dtype=float)
com_phi_moment_roi = np.full((Ny, Nx), np.nan, dtype=float)

for iy in range(Ny):
    for ix in range(Nx):
        curve = data_roi[:, iy, ix]
        mean_u, fwhm_u = calc_fwhm_moment_1d(curve, u_range=phi_range_rad,
                                             u_steps=phi_steps)
        com_phi_moment_roi[iy, ix] = mean_u
        fwhm_phi_roi[iy, ix] = fwhm_u


from bokeh.plotting import figure, output_file, save
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.events import Tap


# ---------------------------------------------
# Build Bokeh viewer: com_phi + rocking_curve (ROI)
# ---------------------------------------------

# data_roi     : (Nphi, Ny, Nx)
# com_phi_roi  : (Ny, Nx)
# fwhm_phi_roi : (Ny, Nx)
# Phi          : (Nphi,)


# Flatten rocking_curve: (Npix, Nphi)
rock_flat = data_roi.reshape(Nphi, Ny * Nx).T  # (Npix, Nphi)

# Start at center pixel of ROI
ix0, iy0 = Nx // 2, Ny // 2
pix0 = iy0 * Nx + ix0
curve0 = rock_flat[pix0]  # (Nphi,)

source_img = ColumnDataSource(data=dict(
    image=[com_phi_moment_roi],   # left image is COM map on ROI
    x=[0], y=[0], dw=[Nx], dh=[Ny],
))

source_curve = ColumnDataSource(data=dict(
    x=Phi,
    y=curve0,
))

# Helper lines for FWHM visualisation
source_peak = ColumnDataSource(data=dict(x=[], y=[]))   # vertical line at peak
source_fwhm = ColumnDataSource(data=dict(x=[], y=[]))   # horizontal line at half max

# Convert to JS-friendly lists
rock_flat_list   = rock_flat.tolist()
com_map_list     = com_phi_moment_roi.tolist()
fwhm_map_list    = fwhm_phi_roi.tolist()
Phi_list         = Phi.tolist()

# ---------------------------------------------
# Figures
# ---------------------------------------------
p_img = figure(
    width=500, height=500,
    x_range=(0, Nx), y_range=(0, Ny),
    tools="tap,reset,pan,wheel_zoom",
    title="COM φ map (ROI)",
    match_aspect=True,
)
p_img.image(image="image", x="x", y="y", dw="dw", dh="dh", source=source_img)

p_curve = figure(
    width=500, height=500,
    title="Rocking curve",
    x_axis_label="φ [rad]",
    y_axis_label="Intensity",
)
p_curve.line("x", "y", source=source_curve)
p_curve.line("x", "y", source=source_peak,
             line_color="red", line_dash="dashed", line_width=1)
p_curve.line("x", "y", source=source_fwhm,
             line_color="green", line_dash="dashed", line_width=1)

# ---------------------------------------------
# Callback: click image -> update curve + FWHM lines + title
# ---------------------------------------------
callback = CustomJS(args=dict(
    src_curve=source_curve,
    src_peak=source_peak,
    src_fwhm=source_fwhm,
    rock_flat=rock_flat_list,
    nx=Nx,
    ny=Ny,
    Phi=Phi_list,
    com_map=com_map_list,
    fwhm_map=fwhm_map_list,
    title=p_curve.title,
), code="""
    const x = cb_obj.x;
    const y = cb_obj.y;

    const ix = Math.floor(x);
    const iy = Math.floor(y);

    if (ix < 0 || ix >= nx) { return; }
    if (iy < 0 || iy >= ny) { return; }

    const pix_index = iy * nx + ix;
    const curve = rock_flat[pix_index];
    const PhiArr = Phi;

    // update main curve
    const data_curve = src_curve.data;
    data_curve['y'] = curve;
    src_curve.change.emit();

    const N = curve.length;
    if (N < 2) {
        src_peak.data = {x: [], y: []};
        src_fwhm.data = {x: [], y: []};
        src_peak.change.emit();
        src_fwhm.change.emit();
        return;
    }

    // baseline = min(curve)
    let baseline = curve[0];
    for (let k = 1; k < N; k++) {
        if (curve[k] < baseline) baseline = curve[k];
    }
    const y_shift = new Array(N);
    for (let k = 0; k < N; k++) {
        y_shift[k] = curve[k] - baseline;
    }

    // peak
    let peak_idx = 0;
    let peak_amp = y_shift[0];
    for (let k = 1; k < N; k++) {
        if (y_shift[k] > peak_amp) {
            peak_amp = y_shift[k];
            peak_idx = k;
        }
    }

    let fwhm_val = NaN;
    let x_left = NaN;
    let x_right = NaN;
    const x_peak = PhiArr[peak_idx];
    const y_peak = curve[peak_idx];
    let half_level = baseline;

    if (peak_amp > 0) {
        const half = peak_amp / 2.0;
        half_level = baseline + half;

        // left crossing
        let i = peak_idx;
        while (i > 0 && y_shift[i] >= half) {
            i -= 1;
        }

        // right crossing
        let j = peak_idx;
        while (j < N - 1 && y_shift[j] >= half) {
            j += 1;
        }

        if (i < peak_idx && j > peak_idx) {
            // interpolate left
            const x1L = PhiArr[i];
            const x2L = PhiArr[i+1];
            const y1L = y_shift[i];
            const y2L = y_shift[i+1];
            if (y2L === y1L) {
                x_left = x1L;
            } else {
                x_left = x1L + (half - y1L) * (x2L - x1L) / (y2L - y1L);
            }

            // interpolate right
            const x1R = PhiArr[j-1];
            const x2R = PhiArr[j];
            const y1R = y_shift[j-1];
            const y2R = y_shift[j];
            if (y2R === y1R) {
                x_right = x2R;
            } else {
                x_right = x1R + (half - y1R) * (x2R - x1R) / (y2R - y1R);
            }

            fwhm_val = x_right - x_left;
        }
    }

    const data_peak = src_peak.data;
    const data_fwhm = src_fwhm.data;

    if (!isNaN(fwhm_val)) {
        data_peak['x']  = [x_peak, x_peak];
        data_peak['y']  = [0,      y_peak];

        data_fwhm['x']  = [x_left, x_right];
        data_fwhm['y']  = [half_level, half_level];
    } else {
        data_peak['x'] = [];
        data_peak['y'] = [];
        data_fwhm['x'] = [];
        data_fwhm['y'] = [];
    }

    src_peak.change.emit();
    src_fwhm.change.emit();

    // COM & FWHM from precomputed maps (moment-based)
    const com_val  = com_map[iy][ix];
    const fwhm_map_val = fwhm_map[iy][ix];

    const com_str  = (isNaN(com_val)      ? "n/a" : com_val.toExponential(2)  + " rad");
    const fwhm_str = (isNaN(fwhm_map_val) ? "n/a" : fwhm_map_val.toExponential(2) + " rad");

    title.text = `COM = ${com_str},  FWHM = ${fwhm_str}`;
""")

p_img.js_on_event(Tap, callback)

layout = row(p_img, p_curve)

output_file("rocking_viewer_ROI.html", title="Rocking Curve Viewer (Bokeh)")
save(layout)

print("Saved rocking_viewer.html – open it in a browser and click on the image.")













# def calc_fwhm_1d(image, u_range, u_steps):
#     """
#     Moment-based COM and FWHM along a 1D axis u.

#     Parameters
#     ----------
#     image : array-like, shape (u_steps,)
#         Intensity as function of u (e.g. rocking curve for one pixel).
#     u_range : float
#         Half-range of u (so u runs from -u_range to +u_range).
#         IMPORTANT: use radians if you want FWHM in radians.
#     u_steps : int
#         Number of u samples (must match image length).

#     Returns
#     -------
#     mean_u : float
#         Center of mass along u.
#     fwhm_u : float
#         FWHM along u, assuming a roughly Gaussian shape.
#         np.nan if it cannot be computed (zero or negative total intensity).
#     """
#     image = np.asarray(image, dtype=float).ravel()
#     if image.size != u_steps:
#         raise ValueError(f"image length {image.size} != u_steps {u_steps}")

#     # Coordinate axis
#     u = np.linspace(-u_range, u_range, u_steps)

#     total = image.sum()
#     if total <= 0 or not np.isfinite(total):
#         return np.nan, np.nan

#     # First moment (mean)
#     mean_u = np.sum(u * image) / total

#     # Second central moment (variance)
#     var_u = np.sum((u - mean_u) ** 2 * image) / total
#     if var_u <= 0:
#         return mean_u, np.nan

#     sigma = np.sqrt(var_u)
#     fwhm_u = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma

#     return mean_u, fwhm_u


# # Define the range and number of steps for phi and chi
# phi_range, phi_steps = 600e-6*180/np.pi, 401

# # Phi is redefined inside compute_rocking_curve_parallel, but you can keep this if you want
# rocking_curve, Phi = compute_rocking_curve_parallel(Hg, phi_range, phi_steps)

# data = rocking_curve          # shape (Nphi, Ny, Nx)
# Nphi, Ny, Nx = data.shape



# # careful: your phi_range variable is in DEGREES in your script
# phi_range_deg, phi_steps = 600e-6*180/np.pi, 401
# u_range = np.deg2rad(phi_range_deg)  # now in radians

# fwhm_phi = np.full((Ny, Nx), np.nan)
# com_phi_moment = np.full((Ny, Nx), np.nan)

# for iy in range(Ny):
#     for ix in range(Nx):
#         curve = data[:, iy, ix]  # shape (phi_steps,)
#         mean_u, fwhm_u = calc_fwhm_1d(curve, u_range=u_range, u_steps=phi_steps)
#         com_phi_moment[iy, ix] = mean_u
#         fwhm_phi[iy, ix] = fwhm_u


# data_ds    = data[:, ::2, ::2]   # (Nphi, Ny_ds, Nx_ds)
# com_phi_ds = com_phi_moment[::2, ::2]
# fwhm_phi_ds = fwhm_phi[::2, ::2]
# Ny, Nx = com_phi_ds.shape
# Nphi = Phi.shape[0]

# from bokeh.plotting import figure, output_file, save
# from bokeh.layouts import row
# from bokeh.models import ColumnDataSource, CustomJS
# from bokeh.events import Tap

# # ---------------------------------------------
# # Build Bokeh viewer: com_phi + rocking_curve
# # ---------------------------------------------

# # We already have:
# # data_ds      : (Nphi, Ny, Nx)
# # com_phi_ds   : (Ny, Nx)
# # fwhm_phi_ds  : (Ny, Nx)
# # Phi          : (Nphi,)

# Ny, Nx = com_phi_ds.shape
# Nphi = Phi.shape[0]

# # Flatten rocking_curve: (Npix, Nphi)
# rock_flat = data_ds.reshape(Nphi, Ny * Nx).T  # (Npix, Nphi)

# # Start at center pixel
# ix0, iy0 = Nx // 2, Ny // 2
# pix0 = iy0 * Nx + ix0
# curve0 = rock_flat[pix0]  # (Nphi,)

# # Image source
# source_img = ColumnDataSource(data=dict(
#     image=[com_phi_ds],
#     x=[0], y=[0], dw=[Nx], dh=[Ny],
# ))

# # Main curve
# source_curve = ColumnDataSource(data=dict(
#     x=Phi,
#     y=curve0,
# ))

# # FWHM helper lines (empty at start)
# source_peak = ColumnDataSource(data=dict(x=[], y=[]))   # vertical at peak
# source_fwhm = ColumnDataSource(data=dict(x=[], y=[]))   # horizontal at half max

# # WARNING: for large datasets this can be big even after downsampling.
# rock_flat_list    = rock_flat.tolist()
# com_map_list      = com_phi_ds.tolist()
# fwhm_map_list     = fwhm_phi_ds.tolist()
# Phi_list          = Phi.tolist()

# # ---------------------------------------------
# # Figures
# # ---------------------------------------------
# p_img = figure(
#     width=500, height=500,
#     x_range=(0, Nx), y_range=(0, Ny),
#     tools="tap,reset,pan,wheel_zoom",
#     title="COM φ map",
#     match_aspect=True,
# )
# p_img.image(image="image", x="x", y="y", dw="dw", dh="dh", source=source_img)

# p_curve = figure(
#     width=500, height=500,
#     title="Rocking curve",
#     x_axis_label="φ [rad]",
#     y_axis_label="Intensity",
# )
# # main curve
# p_curve.line("x", "y", source=source_curve)
# # vertical peak line
# p_curve.line("x", "y", source=source_peak,
#              line_color="red", line_dash="dashed", line_width=1)
# # horizontal FWHM line
# p_curve.line("x", "y", source=source_fwhm,
#              line_color="green", line_dash="dashed", line_width=1)

# # ---------------------------------------------
# # Callback: click image -> update curve + FWHM lines + title
# # ---------------------------------------------
# callback = CustomJS(args=dict(
#     src_curve=source_curve,
#     src_peak=source_peak,
#     src_fwhm=source_fwhm,
#     rock_flat=rock_flat_list,
#     nx=Nx,
#     ny=Ny,
#     Phi=Phi_list,
#     com_map=com_map_list,
#     fwhm_map=fwhm_map_list,
#     title=p_curve.title,
# ), code="""
#     // Tap event with x,y in data coords
#     const x = cb_obj.x;
#     const y = cb_obj.y;

#     const ix = Math.floor(x);
#     const iy = Math.floor(y);

#     if (ix < 0 || ix >= nx) { return; }
#     if (iy < 0 || iy >= ny) { return; }

#     const pix_index = iy * nx + ix;
#     const curve = rock_flat[pix_index];   // array length Nphi
#     const PhiArr = Phi;

#     // Update main curve
#     const data_curve = src_curve.data;
#     data_curve['y'] = curve;
#     src_curve.change.emit();

#     // ---- Compute FWHM in JS (same logic as Python) ----
#     const N = curve.length;
#     if (N < 2) {
#         src_peak.data = {x: [], y: []};
#         src_fwhm.data = {x: [], y: []};
#         src_peak.change.emit();
#         src_fwhm.change.emit();
#     } else {
#         // baseline = min(curve)
#         let baseline = curve[0];
#         for (let k = 1; k < N; k++) {
#             if (curve[k] < baseline) baseline = curve[k];
#         }
#         const y_shift = new Array(N);
#         for (let k = 0; k < N; k++) {
#             y_shift[k] = curve[k] - baseline;
#         }

#         // peak
#         let peak_idx = 0;
#         let peak_amp = y_shift[0];
#         for (let k = 1; k < N; k++) {
#             if (y_shift[k] > peak_amp) {
#                 peak_amp = y_shift[k];
#                 peak_idx = k;
#             }
#         }

#         let fwhm_val = NaN;
#         let x_left = NaN;
#         let x_right = NaN;
#         let x_peak = PhiArr[peak_idx];
#         let y_peak = curve[peak_idx];
#         let half_level = baseline;

#         if (peak_amp > 0) {
#             const half = peak_amp / 2.0;
#             half_level = baseline + half;

#             // left crossing
#             let i = peak_idx;
#             while (i > 0 && y_shift[i] >= half) {
#                 i -= 1;
#             }

#             // right crossing
#             let j = peak_idx;
#             while (j < N - 1 && y_shift[j] >= half) {
#                 j += 1;
#             }

#             if (i < peak_idx && j > peak_idx) {
#                 // interpolate left
#                 const x1L = PhiArr[i];
#                 const x2L = PhiArr[i+1];
#                 const y1L = y_shift[i];
#                 const y2L = y_shift[i+1];
#                 if (y2L === y1L) {
#                     x_left = x1L;
#                 } else {
#                     x_left = x1L + (half - y1L) * (x2L - x1L) / (y2L - y1L);
#                 }

#                 // interpolate right
#                 const x1R = PhiArr[j-1];
#                 const x2R = PhiArr[j];
#                 const y1R = y_shift[j-1];
#                 const y2R = y_shift[j];
#                 if (y2R === y1R) {
#                     x_right = x2R;
#                 } else {
#                     x_right = x1R + (half - y1R) * (x2R - x1R) / (y2R - y1R);
#                 }

#                 fwhm_val = x_right - x_left;
#             }
#         }

#         // update FWHM helper lines
#         const data_peak = src_peak.data;
#         const data_fwhm = src_fwhm.data;

#         if (!isNaN(fwhm_val)) {
#             data_peak['x']  = [x_peak, x_peak];
#             data_peak['y']  = [0,      y_peak];

#             data_fwhm['x']  = [x_left, x_right];
#             data_fwhm['y']  = [half_level, half_level];
#         } else {
#             data_peak['x'] = [];
#             data_peak['y'] = [];
#             data_fwhm['x'] = [];
#             data_fwhm['y'] = [];
#         }

#         src_peak.change.emit();
#         src_fwhm.change.emit();

#         // ---- Update title with COM and FWHM ----
#         const com_val  = com_map[iy][ix];
#         const fwhm_map_val = fwhm_map[iy][ix];  // from precomputed map (may be NaN)

#         let com_str  = isNaN(com_val)      ? "n/a" : com_val.toExponential(2)  + " rad";
#         let fwhm_str = isNaN(fwhm_map_val) ? "n/a" : fwhm_map_val.toExponential(2) + " rad";

#         title.text = `COM = ${com_str},  FWHM = ${fwhm_str}`;
#     }
# """)

# p_img.js_on_event(Tap, callback)

# layout = row(p_img, p_curve)

# output_file("rocking_viewer.html", title="Rocking Curve Viewer (Bokeh)")
# save(layout)

# print("Saved rocking_viewer.html – open it in a browser and click on the image.")





# # Phi is the rocking axis in radians
# num = (data * Phi[:, None, None]).sum(axis=0)  # (Ny, Nx)
# den = data.sum(axis=0)                         # (Ny, Nx)

# com_phi = np.full_like(num, np.nan, dtype=float)
# np.divide(num, den, out=com_phi, where=den != 0)   # radians


# from bokeh.plotting import figure, output_file, save
# from bokeh.layouts import row
# from bokeh.models import ColumnDataSource, CustomJS
# from bokeh.events import Tap

# # ---------------------------------------------
# # Build Bokeh viewer: com_phi + rocking_curve
# # ---------------------------------------------
# data_ds = data[:, ::2, ::2]
# com_phi_ds = com_phi[::2, ::2]
# Ny, Nx = com_phi_ds.shape
# Nphi = Phi.shape[0]

# # Flatten rocking_curve: (Npix, Nphi)
# rock_flat = data_ds.reshape(Nphi, Ny * Nx).T  # (Npix, Nphi)

# # Start at center pixel
# ix0, iy0 = Nx // 2, Ny // 2
# pix0 = iy0 * Nx + ix0
# curve0 = rock_flat[pix0]  # (Nphi,)

# source_img = ColumnDataSource(data=dict(
#     image=[com_phi_ds],  # if you want origin='lower' behaviour, you can use com_phi[::-1, :]
#     x=[0], y=[0], dw=[Nx], dh=[Ny],
# ))

# source_curve = ColumnDataSource(data=dict(
#     x=Phi,
#     y=curve0,
# ))

# # WARNING: for large datasets this can be big; for full-resolution 401×(510×170)
# # this will make a chunky HTML. Downsample / crop if needed.
# rock_flat_list = rock_flat.tolist()

# p_img = figure(
#     width=500, height=500,
#     x_range=(0, Nx), y_range=(0, Ny),
#     tools="tap,reset,pan,wheel_zoom",
#     title="COM φ map",
#     match_aspect=True,
# )
# p_img.image(image="image", x="x", y="y", dw="dw", dh="dh", source=source_img)

# p_curve = figure(
#     width=500, height=500,
#     title="Rocking curve",
#     x_axis_label="φ [rad]",
#     y_axis_label="Intensity",
# )
# p_curve.line("x", "y", source=source_curve)

# callback = CustomJS(args=dict(
#     src_curve=source_curve,
#     rock_flat=rock_flat_list,
#     nx=Nx,
#     ny=Ny,
# ), code="""
#     // Tap event with x,y in data coords
#     const x = cb_obj.x;
#     const y = cb_obj.y;

#     const ix = Math.floor(x);
#     const iy = Math.floor(y);

#     if (ix < 0 || ix >= nx) { return; }
#     if (iy < 0 || iy >= ny) { return; }

#     const pix_index = iy * nx + ix;
#     const curve = rock_flat[pix_index];

#     const data = src_curve.data;
#     data['y'] = curve;
#     src_curve.change.emit();
# """)

# p_img.js_on_event(Tap, callback)

# layout = row(p_img, p_curve)

# output_file("rocking_viewer.html", title="Rocking Curve Viewer (Bokeh)")
# save(layout)

# print("Saved rocking_viewer.html – open it in a browser and click on the image.")




# plt.imshow(com_phi.T, origin = 'lower', 
#             # vmin = -1.2e-4, vmax = 1.2e-4, 
#             extent=[xl_start, -xl_start,
#                     yl_start, -yl_start])
# plt.colorbar(format='%.0e')



# import tkinter as tk
# import numpy as np
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.figure import Figure


# class RockingCurveViewer(tk.Tk):
#     def __init__(self, data, com_phi, Phi, xl_start, yl_start):
#         super().__init__()

#         self.title("Rocking Curve Viewer: 50mu aperture")

#         # Store data
#         self.data = data  # shape (Nrock, Ny, Nx) = (51, 510, 170)
#         self.com_phi = com_phi  # shape (Ny, Nx) = (510, 170)
#         self.n_rock, self.ny, self.nx = data.shape
#         self.Phi = Phi  # shape (Nrock,)

#         if self.com_phi.shape != (self.ny, self.nx):
#             raise ValueError(
#                 f"com_phi shape {self.com_phi.shape} does not match data spatial shape ({self.ny}, {self.nx})"
#             )

#         # ---------- Top control panel: vmin / vmax ----------
#         ctrl = tk.Frame(self)
#         ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

#         self.vmin_init = -1.2e-4
#         self.vmax_init =  1.2e-4

#         tk.Label(ctrl, text="vmin [rad]:").pack(side=tk.LEFT)
#         self.vmin_var = tk.StringVar(value=f"{self.vmin_init:.2e}")
#         vmin_entry = tk.Entry(ctrl, textvariable=self.vmin_var, width=10)
#         vmin_entry.pack(side=tk.LEFT, padx=(0, 10))

#         tk.Label(ctrl, text="vmax [rad]:").pack(side=tk.LEFT)
#         self.vmax_var = tk.StringVar(value=f"{self.vmax_init:.2e}")
#         vmax_entry = tk.Entry(ctrl, textvariable=self.vmax_var, width=10)
#         vmax_entry.pack(side=tk.LEFT, padx=(0, 10))

#         update_btn = tk.Button(ctrl, text="Update colormap", command=self.update_clim)
#         update_btn.pack(side=tk.LEFT, padx=5)

#         vmin_entry.bind("<Return>", lambda e: self.update_clim())
#         vmax_entry.bind("<Return>", lambda e: self.update_clim())

#         # ---------- Figure + plots ----------
#         canvas_frame = tk.Frame(self)
#         canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

#         self.fig = Figure(figsize=(10, 5), dpi=100)
#         self.fig.subplots_adjust(wspace=0.4)
#         self.ax_img = self.fig.add_subplot(1, 2, 1)
#         self.ax_curve = self.fig.add_subplot(1, 2, 2)

#         # Left image: COM phi
#         im = self.ax_img.imshow(
#             self.com_phi.T,
#             origin="lower",
#             vmin=self.vmin_init,
#             vmax=self.vmax_init,
#             extent=[xl_start, -xl_start, yl_start, -yl_start],
#         )
#         cbar = self.fig.colorbar(im, ax=self.ax_img, format="%.0e")
#         cbar.set_label("COM φ")

#         self.ax_img.set_title("COM φ map")
#         self.ax_img.set_xlabel("x")
#         self.ax_img.set_ylabel("y")

#         # Right curve: main line + FWHM helpers
#         self.line, = self.ax_curve.plot([], [], "-")
#         self.peak_line, = self.ax_curve.plot([], [], "r--", linewidth=1)  # vertical at peak
#         self.fwhm_line, = self.ax_curve.plot([], [], "g--", linewidth=1)  # horizontal FWHM

#         self.ax_curve.set_xlabel("φ-value [radians]")
#         self.ax_curve.set_ylabel("Intensity")
#         self.ax_curve.set_xlim(self.Phi[0], self.Phi[-1])
#         self.ax_curve.set_title("Click a pixel on the left")

#         # Marker on the left image for clicked pixel
#         self.marker, = self.ax_img.plot([], [], "r+", ms=8, mew=1.5)

#         # Embed figure in Tk
#         self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
#         self.canvas.draw()
#         self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

#         # Store image + colorbar refs
#         self.im = im
#         self.cbar = cbar

#         # Store image geometry
#         self.img_array = im.get_array()       # com_phi.T, shape (Nx, Ny) = (170, 510)
#         self.ny_im, self.nx_im = self.img_array.shape
#         self.xmin, self.xmax, self.ymin, self.ymax = im.get_extent()

#         # Connect click event
#         self.fig.canvas.mpl_connect("button_press_event", self.on_click)

#     def update_clim(self):
#         try:
#             vmin = float(self.vmin_var.get())
#             vmax = float(self.vmax_var.get())
#         except ValueError:
#             print("Invalid vmin/vmax; use numeric values.")
#             return

#         if vmin >= vmax:
#             print("vmin must be smaller than vmax.")
#             return

#         self.im.set_clim(vmin=vmin, vmax=vmax)
#         self.cbar.update_normal(self.im)
#         self.canvas.draw_idle()

#     def _compute_fwhm(self, x, y):
#         """Return (x_peak, y_peak, x_left, x_right, half_level, fwhm) or None."""
#         x = np.asarray(x, dtype=float)
#         y = np.asarray(y, dtype=float)

#         if x.size == 0 or y.size == 0 or x.size != y.size:
#             return None
#         if not np.any(np.isfinite(y)):
#             return None

#         # Baseline: min value
#         baseline = np.nanmin(y)
#         y_shift = y - baseline

#         peak_idx = np.nanargmax(y_shift)
#         peak_amp = y_shift[peak_idx]
#         if peak_amp <= 0:
#             return None

#         half = peak_amp / 2.0
#         half_level = baseline + half
#         x_peak = x[peak_idx]
#         y_peak = y[peak_idx]

#         # ---- left side crossing ----
#         i = peak_idx
#         while i > 0 and y_shift[i] >= half:
#             i -= 1
#         if i == peak_idx:  # no crossing found
#             return None
#         x1, x2 = x[i], x[i + 1]
#         y1, y2 = y_shift[i], y_shift[i + 1]
#         if y2 == y1:
#             x_left = x1
#         else:
#             x_left = x1 + (half - y1) * (x2 - x1) / (y2 - y1)

#         # ---- right side crossing ----
#         j = peak_idx
#         n = len(y_shift)
#         while j < n - 1 and y_shift[j] >= half:
#             j += 1
#         if j == peak_idx:
#             return None
#         x1, x2 = x[j - 1], x[j]
#         y1, y2 = y_shift[j - 1], y_shift[j]
#         if y2 == y1:
#             x_right = x2
#         else:
#             x_right = x1 + (half - y1) * (x2 - x1) / (y2 - y1)

#         fwhm = x_right - x_left
#         return x_peak, y_peak, x_left, x_right, half_level, fwhm

#     def on_click(self, event):
#         if event.inaxes is not self.ax_img:
#             return
#         if event.xdata is None or event.ydata is None:
#             return

#         x = event.xdata
#         y = event.ydata

#         col = int((x - self.xmin) / (self.xmax - self.xmin) * self.nx_im)
#         row = int((y - self.ymin) / (self.ymax - self.ymin) * self.ny_im)

#         col = np.clip(col, 0, self.nx_im - 1)
#         row = np.clip(row, 0, self.ny_im - 1)

#         X = col  # 0..509 -> Ny
#         Y = row  # 0..169 -> Nx

#         curve = self.data[:, X, Y]

#         # Update main curve
#         self.line.set_data(self.Phi, curve)

#         # Compute FWHM
#         fwhm_info = self._compute_fwhm(self.Phi, curve)

#         if fwhm_info is not None:
#             x_peak, y_peak, x_left, x_right, half_level, fwhm = fwhm_info

#             # Vertical line at peak
#             self.peak_line.set_data([x_peak, x_peak], [0, y_peak])

#             # Horizontal FWHM line
#             self.fwhm_line.set_data([x_left, x_right], [half_level, half_level])

#             fwhm_text = f", FWHM = {fwhm:.2e} rad"
#         else:
#             # If FWHM can't be computed, clear the helper lines
#             self.peak_line.set_data([], [])
#             self.fwhm_line.set_data([], [])
#             fwhm_text = ", FWHM = n/a"

#         self.ax_curve.relim()
#         self.ax_curve.autoscale_view()
#         # Keep x-range fixed to full rocking curve
#         self.ax_curve.set_xlim(self.Phi[0], self.Phi[-1])

#         self.ax_curve.set_title(
#             f"COM value = {self.com_phi[X, Y]:.2e} rad{fwhm_text}"
#         )

#         # Update marker on left image
#         x_phys = self.xmin + (col + 0.5) * (self.xmax - self.xmin) / self.nx_im
#         y_phys = self.ymin + (row + 0.5) * (self.ymax - self.ymin) / self.ny_im
#         self.marker.set_data([x_phys], [y_phys])

#         self.canvas.draw_idle()
# if __name__ == "__main__":
#     app = RockingCurveViewer(data, com_phi, Phi, xl_start, yl_start)
#     app.mainloop()




# np.save('GUI/No_aperture_1dis/xl_start.npy', xl_start)
# np.save('GUI/No_aperture_1dis/yl_start.npy', yl_start)
# np.save('GUI/No_aperture_1dis/data.npy', data)
# np.save('GUI/No_aperture_1dis/Phi.npy', Phi)
# np.save('GUI/No_aperture_1dis/com_phi.npy', com_phi)



# # Define the filepath and filename prefix for the output images
# fpath = r'/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/borgi/Dislocation_Identification/Geometrical_Optics_BS/Geometrical_Optics_master/perfect_crystal_53_retry'
# fn_prefix = r'/mosa_test_0000_'
# ftype = ".npy"

# # Define the filepath and filename prefix for the output images
# fpath1 = r'/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/borgi/Dislocation_Identification/Geometrical_Optics_BS/Geometrical_Optics_master/disloc_crystal_53_retry'
# fn_prefix1 = r'/mosa_test_0000_'
# ftype1 = ".npy"

# # Generate arrays for phi and chi values
# Phi = np.linspace(-phi_range, phi_range, phi_steps)
# Chi = np.linspace(-chi_range, chi_range, chi_steps)


# # # Out comment to save images of new results
# # save_images(phi_range, phi_steps, 
# #             chi_range, chi_steps, 
# #             fpath, fn_prefix, ftype)

# # Out comment to save many images of new results
# save_images_parallel(np.zeros_like(Hg), phi_range, phi_steps, 
#                      chi_range, chi_steps, 
#                      fpath, fn_prefix, ftype)

# # Out comment to save many images of new results
# save_images_parallel(Hg, phi_range, phi_steps, 
#                      chi_range, chi_steps, 
#                      fpath1, fn_prefix, ftype)

# stack_no_dislocs, stack_reshape_no_dislocs, dim_1_no_dislocs, dim_2_no_dislocs = load_images(fpath, phi_steps, chi_steps)
# stack_dislocs, stack_reshape_dislocs, dim_1_dislocs, dim_2_dislocs = load_images(fpath1, phi_steps, chi_steps)


# plt.title('Perfect crystal: Example pixels angular spread')
# plt.imshow(40*stack_no_dislocs.reshape((chi_steps, phi_steps, dim_1_no_dislocs, dim_2_no_dislocs))[:,:,dim_1_no_dislocs//2+2,dim_2_no_dislocs//2+2], origin='lower', aspect='auto',
#            extent = [-np.deg2rad(phi_range), np.deg2rad(phi_range), 
#                      -np.deg2rad(chi_range), np.deg2rad(chi_range)])
# plt.colorbar()
# plt.xlabel(f'$\phi$ [radians]')
# plt.ylabel(f'$\chi$ [radians]')
# plt.show()

# plt.title('Single Dislocation in crystal: \n Example pixels angular spread')
# plt.imshow(40*stack_dislocs.reshape((chi_steps, phi_steps, dim_1_dislocs, dim_2_dislocs))[:,:,dim_1_dislocs//2+2,dim_2_dislocs//2+2], origin='lower', aspect='auto',
#            extent = [-np.deg2rad(phi_range), np.deg2rad(phi_range), 
#                      -np.deg2rad(chi_range), np.deg2rad(chi_range)])
# plt.colorbar()
# plt.xlabel(f'$\phi$ [radians]')
# plt.ylabel(f'$\chi$ [radians]')
# plt.show()
# plt.title('Phi: 192 µrad')
# plt.imshow(stack_reshape_dislocs[37,15], aspect='auto')
# plt.scatter(83, 253, s=25, c='black')
# plt.scatter(83, 253, s=3, c='orange')
# plt.xlabel('Pixels')
# plt.ylabel('Pixels')
# plt.show()


# pic = stack_dislocs.reshape((chi_steps, phi_steps, dim_1_dislocs, dim_2_dislocs))[15,15]
# reshaped_stack = stack_dislocs.reshape((chi_steps, phi_steps, dim_1_dislocs, dim_2_dislocs))
# reshaped_stack_clean = stack_no_dislocs.reshape((chi_steps, phi_steps, dim_1_no_dislocs, dim_2_no_dislocs))

# # Going to higher resolution, Chi can shift. Here it is corrected.
# com = center_of_mass(reshaped_stack_clean[:,:,-1,-1])
# plt.imshow(reshaped_stack_clean[:,:,-1,-1].T,)
# plt.scatter(*com, label = com)
# plt.legend()
# Chi_high = np.linspace(-chi_range,chi_range,chi_steps*100)
# shift = com[0]*100 - (chi_steps*100 / 2)
# shift_rads = Chi_high[int(abs(shift))]-Chi_high[0]

# shifted_Chi = np.linspace(-chi_range+ shift_rads,chi_range+shift_rads,chi_steps)



# com = center_of_mass(reshaped_stack[:,:,256,85])
# x, y = np.deg2rad(Phi[int(com[1])]), np.deg2rad(Chi[int(com[0])])

# fig, axs = plt.subplots(1, 2, figsize=(10, 4))


# axs[0].imshow(reshaped_stack[chi_steps//2, phi_steps//2].T, aspect='auto', origin='lower')
# axs[0].scatter(256,85, s=3, c='red')

# axs[1].imshow(reshaped_stack[:,:,256,85], aspect=1/4, origin='lower',
# extent = [-np.deg2rad(phi_range), np.deg2rad(phi_range), 
#                      -np.deg2rad(chi_range), np.deg2rad(chi_range)])
# axs[1].scatter(x,y, s=3, c='red', label = np.round((x,y),8))
# axs[1].set_xlabel(f'$\phi$ [radians]')
# axs[1].set_ylabel(f'$\chi$ [radians]')
# axs[1].legend()
# plt.tight_layout()
# plt.show()


# Chi_high = np.linspace(-chi_range,chi_range,chi_steps*100)


# phi_list = np.zeros((dim_1_dislocs, dim_2_dislocs))
# chi_list = np.zeros((dim_1_dislocs, dim_2_dislocs))
# shifted_Chi = np.deg2rad(np.linspace(-chi_range+ shift_rads,chi_range+shift_rads,chi_steps*20))
# Phi_high = np.deg2rad(np.linspace(-phi_range,phi_range,phi_steps*20))
# for i in tqdm(range(reshaped_stack.shape[2])):
#     for j in range(reshaped_stack.shape[3]):
#         x_ind, y_ind = center_of_mass(reshaped_stack[:,:,i,j])
#         phi_list[i,j] = Phi_high[np.round(y_ind*20).astype(int)]
#         chi_list[i,j] = shifted_Chi[np.round(x_ind*20).astype(int)]


# # Plot of two components of qi in (x, y, z=0) plane
# fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# # Plot qi_1
# im1 = axs[0].imshow((phi_list*-1).T, interpolation='none',
#                     extent=[xl_start, -xl_start,
#                             yl_start, -yl_start],
#                     vmin = -1e-4, vmax = 1e-4, cmap='viridis', origin = 'lower')
# axs[0].set_aspect('equal')
# axs[0].set_title('Extreme Phi')
# axs[0].set_xlabel('$y_{\ell}$ ($\mu$m)', fontsize=12)
# axs[0].set_ylabel('$x_{\ell}$ ($\mu$m)', fontsize=12)
# # axs[0].invert_yaxis()  # To match 'set(gca,'ydir','normal')' in MATLAB
# axs[0].grid(False)

# # Customize colorbar with scientific notation
# cbar1 = fig.colorbar(im1, ax=axs[0], format=ScalarFormatter(useMathText=True))
# cbar1.formatter.set_powerlimits((-2, 2))  # Adjust the power limits as needed
# cbar1.update_ticks()

# # Plot qi_2
# im2 = axs[1].imshow(chi_list.T*-1, 
#                     extent=[xl_start, -xl_start,
#                             yl_start, -yl_start],
#                     interpolation = 'none',
#                     vmin = -1e-4, vmax = 1e-4, cmap='viridis', origin = 'lower')
# axs[1].set_aspect('equal')
# axs[1].set_title('Extreme Chi')
# axs[1].set_xlabel('$y_{\ell}$ ($\mu$m)', fontsize=12)
# axs[1].set_ylabel('$x_{\ell}$ ($\mu$m)', fontsize=12)
# # axs[1].invert_yaxis()  # To match 'set(gca,'ydir','normal')' in MATLAB
# axs[1].grid(False)

# # Customize colorbar with scientific notation
# cbar2 = fig.colorbar(im2, ax=axs[1], format=ScalarFormatter(useMathText=True))
# cbar2.formatter.set_powerlimits((-2, 2))  # Adjust the power limits as needed
# cbar2.update_ticks()

# plt.tight_layout()
# plt.savefig('extrem_phi+chi2.svg')


# imp, qi_fieldp = forward(Hg, phi = 0, qi_return=True)

# X = np.linspace(-xl_start, xl_start, xl_steps) * 1E6  # rulers on the x-axis in µm
# Y = np.linspace(-yl_start, yl_start, yl_steps) * 1E6  # rulers on the y-axis in µm
# Z = np.linspace(-zl_start, zl_start, zl_steps) * 1E6  # rulers on the z-axis in µm

# # Plot of two components of qi in (x, y, z=0) plane
# fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# # Plot qi_1
# im1 = axs[0].imshow(qi_fieldp[0, :, :, zl_steps // 2].squeeze(), extent=[Y.min(), Y.max(), X.min(), X.max()],
#                     vmin=-1E-4, vmax=1E-4, cmap='viridis', origin = 'lower')
# # axs[0].set_xlim((-6,-4))
# # axs[0].set_ylim((1.5,3.5))
# axs[0].set_aspect('equal')
# axs[0].set_title('qi_1 for (x, y) plane, z=0')
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
# axs[1].set_title('qi_2 for (x, y) plane, z=0')
# axs[1].set_xlabel('$y_{\ell}$ ($\mu$m)', fontsize=12)
# axs[1].set_ylabel('$x_{\ell}$ ($\mu$m)', fontsize=12)
# # axs[1].invert_yaxis()  # To match 'set(gca,'ydir','normal')' in MATLAB
# axs[1].grid(False)

# # Customize colorbar with scientific notation
# cbar2 = fig.colorbar(im2, ax=axs[1], format=ScalarFormatter(useMathText=True))
# cbar2.formatter.set_powerlimits((-2, 2))  # Adjust the power limits as needed
# cbar2.update_ticks()

# plt.tight_layout()
# plt.savefig('qi1+qi2_fields1.svg')

