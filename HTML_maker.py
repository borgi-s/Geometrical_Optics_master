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
from bokeh.events import PanStart, PanEnd
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CustomJS, LinearColorMapper, ColorBar
from bokeh.palettes import Viridis256, Greys256



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

# Compute the rocking curve from the forward model
rocking_curve, Phi = compute_rocking_curve_parallel(Hg, phi_range, phi_steps)

data = (rocking_curve).astype(np.float16)*10          # shape (Nphi, Ny, Nx)

# ----- Central ROI instead of downsampling -----
Nphi, Ny_full, Nx_full = data.shape
ROI_scale = 3
roi_h, roi_w = Ny_full//ROI_scale, Nx_full//ROI_scale  # Ny, Nx in ROI
cy, cx = Ny_full // 2, Nx_full // 2

y0 = cy - roi_h // 2
y1 = y0 + roi_h
x0 = cx - roi_w // 2
x1 = x0 + roi_w

# safety, in case the image is smaller than requested ROI
y0 = max(y0, 0); x0 = max(x0, 0)
y1 = min(y1, Ny_full); x1 = min(x1, Nx_full)

data_roi    = data[:, y0:y1, x0:x1]     # (Nphi, Ny_roi, Nx_roi)
# com_phi_moment_roi = com_phi[y0:y1, x0:x1]     # (Ny_roi, Nx_roi)

Ny, Nx = data_roi.shape[1:]  # these are now ROI dims
Nphi = Phi.shape[0]



# ----- FWHM map on ROI -----

phi_steps = Nphi
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

# Value ranges for color mappers
com_min = np.nanmin(com_phi_moment_roi)
com_max = np.nanmax(com_phi_moment_roi)

img_min = np.nanmin(data_roi)
img_max = np.nanmax(data_roi)

# Color mappers:
# - COM map: Viridis
# - Image: Inferno
com_mapper = LinearColorMapper(palette=Viridis256, low=com_min, high=com_max)
img_mapper = LinearColorMapper(palette=Greys256, low=img_min, high=img_max)

# ---------------------------------------------
# Build Bokeh viewer: COM + rocking_curve + image (ROI)
# ---------------------------------------------
# data_roi     : (Nphi, Ny, Nx)
# com_phi_moment_roi  : (Ny, Nx)
# fwhm_phi_roi : (Ny, Nx)
# Phi          : (Nphi,)

Ny, Nx = com_phi_moment_roi.shape
Nphi = Phi.shape[0]
x_min, x_max = xl_start / ROI_scale * 1e6, -xl_start / ROI_scale * 1e6
y_min, y_max = yl_start / ROI_scale * 1e6, -yl_start / ROI_scale * 1e6


# Flatten rocking_curve: (Npix, Nphi)
rock_flat = data_roi.reshape(Nphi, Ny * Nx).T  # (Npix, Nphi)

# Start at center pixel of ROI
ix0, iy0 = Nx // 2, Ny // 2
pix0 = iy0 * Nx + ix0
curve0 = rock_flat[pix0]         # (Nphi,)

# For the image-at-phi panel, pick initial phi index (e.g. middle of scan)
phi_index0 = Nphi // 2
image0 = data_roi[phi_index0, :, :]  # (Ny, Nx)

# ----------------- Data sources -----------------
# Top-left COM map
source_img_com = ColumnDataSource(data=dict(
    image=[com_phi_moment_roi],
    x=[x_min], y=[y_min],
    dw=[x_max - x_min], dh=[y_max - y_min],
))

# Top-right rocking curve (main line)
source_curve = ColumnDataSource(data=dict(
    x=Phi,
    y=curve0,
))

# Line overlay on the bottom-left image
source_line_overlay = ColumnDataSource(data=dict(
    xs=[], ys=[]
))

# Line intensity profile for bottom-right plot
source_line_profile = ColumnDataSource(data=dict(
    s=[], intensity=[]
))

# FWHM helper lines on the curve
source_peak = ColumnDataSource(data=dict(x=[], y=[]))   # vertical at peak
source_fwhm = ColumnDataSource(data=dict(x=[], y=[]))   # horizontal at half max

# Bottom-left image at selected φ
source_img_phi = ColumnDataSource(data=dict(
    image=[image0],
    x=[0], y=[0], dw=[Nx], dh=[Ny],
))

# Convert arrays to JS-friendly lists
rock_flat_list   = rock_flat.tolist()        # (Npix, Nphi)
com_map_list     = com_phi_moment_roi.tolist()      # (Ny, Nx)
fwhm_map_list    = fwhm_phi_roi.tolist()     # (Ny, Nx)
Phi_list         = Phi.tolist()              # (Nphi,)
data_stack_list  = data_roi.tolist()         # [Nphi][Ny][Nx]

# ----------------- Figures -----------------
# Top-left: COM map
p_img_com = figure(
    width=600, height=500,
    x_range=(x_min, x_max),
    y_range=(y_min, y_max),
    tools="tap,reset,wheel_zoom",
    title="COM φ map (ROI)",
    x_axis_label="xl [µm]",
    y_axis_label="yl [µm]",
    match_aspect=True,
)
p_img_com.image(
    image="image", x="x", y="y", dw="dw", dh="dh",
    source=source_img_com,
    color_mapper=com_mapper,  
)
# Colorbar for COM map
colorbar_com = ColorBar(
    color_mapper=com_mapper,
    label_standoff=8,
    location=(0, 0),
)
p_img_com.add_layout(colorbar_com, 'right')
# Top-right: rocking curve
p_curve = figure(
    width=500, height=500,
    title="Rocking curve",
    x_axis_label="φ [rad]",
    y_axis_label="Intensity",
    tools="tap,reset,wheel_zoom",
)
p_curve.line("x", "y", source=source_curve)
p_curve.line("x", "y", source=source_peak,
             line_color="red", line_dash="dashed", line_width=1)
p_curve.line("x", "y", source=source_fwhm,
             line_color="green", line_dash="dashed", line_width=1)

# Bottom-left: image at selected φ
p_img_phi = figure(
    width=560, height=500,
    x_range=(0, Nx), y_range=(0, Ny),
    title="Raw image at selected φ",
    x_axis_label="xl [pixels]",
    y_axis_label="yl [pixels]",
    match_aspect=True,
    tools="wheel_zoom,reset"
)
p_img_phi.image(
    image="image", x="x", y="y", dw="dw", dh="dh",
    source=source_img_phi,
    color_mapper=img_mapper,        # <-- and here
)
# Colorbar for φ-image
colorbar_img = ColorBar(
    color_mapper=img_mapper,
    label_standoff=8,
    location=(0, 0),
)
p_img_phi.add_layout(colorbar_img, 'right')
p_img_phi.line(
    'xs', 'ys',
    source=source_line_overlay,
    line_color="yellow",
    line_width=2
)

# Bottom-right: line intensity profile
p_line = figure(
    width=500, height=500,
    title="Line intensity profile",
    x_axis_label="Distance [pixels]",
    y_axis_label="Intensity",
    tools="reset,wheel_zoom",
)

p_line.line('s', 'intensity', source=source_line_profile)


# ----------------- Callback 1: click COM map -> update curve + FWHM + title -----------------
callback_com = CustomJS(args=dict(
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
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
), code="""
    const x = cb_obj.x;
    const y = cb_obj.y;

    // Map physical coords -> pixel indices in ROI
    const dx = (x_max - x_min) / nx;
    const dy = (y_max - y_min) / ny;

    const ix = Math.floor((x - x_min) / dx);
    const iy = Math.floor((y - y_min) / dy);

    if (ix < 0 || ix >= nx) return;
    if (iy < 0 || iy >= ny) return;

    const pix_index = iy * nx + ix;
    const curve = rock_flat[pix_index];   // length Nphi
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

p_img_com.js_on_event("tap", callback_com)

# ----------------- Callback 2: click curve -> update image at φ -----------------
callback_curve = CustomJS(args=dict(
    src_img_phi=source_img_phi,
    Phi=Phi_list,
    data_stack=data_stack_list,   # [Nphi][Ny][Nx]
    title=p_img_phi.title,
    Nx=Nx,
    Ny=Ny,
), code="""
    const x = cb_obj.x;
    const PhiArr = Phi;
    const N = PhiArr.length;

    // find nearest phi index to click position
    let j = 0;
    let best = Math.abs(PhiArr[0] - x);
    for (let k = 1; k < N; k++) {
        const d = Math.abs(PhiArr[k] - x);
        if (d < best) {
            best = d;
            j = k;
        }
    }

    // 2D slice for this phi: shape (Ny, Nx) as nested JS arrays
    const im2d = data_stack[j];

    // Existing image buffer that Bokeh is already using
    const data_img = src_img_phi.data;
    const buf = data_img['image'][0];  // typically a 1D typed array of length Ny*Nx

    // Copy im2d into the flat buffer in row-major order
    let idx = 0;
    for (let iy = 0; iy < Ny; iy++) {
        const row = im2d[iy];
        for (let ix = 0; ix < Nx; ix++) {
            buf[idx++] = row[ix];
        }
    }

    // Tell Bokeh to re-render the image
    src_img_phi.change.emit();

    // update title with chosen φ
    const phi_val = PhiArr[j];
    title.text = `Raw image at φ = ${phi_val.toExponential(2)} rad`;
""")

p_curve.js_on_event("tap", callback_curve)

# ----------------- Callback 3: initiate draw line on image -> update line profile -----------------
callback_line_start = CustomJS(args=dict(
    src_line_overlay=source_line_overlay,
), code="""
    // Drag start on bottom-left image
    const x0 = cb_obj.x;
    const y0 = cb_obj.y;

    // Initialize overlay with a degenerate line (point)
    src_line_overlay.data = { xs: [x0, x0], ys: [y0, y0] };
    src_line_overlay.change.emit();
""")

p_img_phi.js_on_event(PanStart, callback_line_start)

# ----------------- Callback 4: finish line on image -> compute profile -----------------
callback_line_end = CustomJS(args=dict(
    src_img_phi=source_img_phi,
    src_line_overlay=source_line_overlay,
    src_line_profile=source_line_profile,
    Nx=Nx,
    Ny=Ny,
), code="""
    // Drag end on bottom-left image
    const x1 = cb_obj.x;
    const y1 = cb_obj.y;

    // Retrieve starting point from overlay source
    const overlay = src_line_overlay.data;
    let x0 = overlay.xs[0];
    let y0 = overlay.ys[0];

    // Update overlay to show the full line
    overlay.xs = [x0, x1];
    overlay.ys = [y0, y1];
    src_line_overlay.change.emit();

    // Current image data (flat array of length Ny*Nx)
    const img_data = src_img_phi.data;
    const img_flat = img_data.image[0];  // 1D flat buffer

    const Nx_val = Nx;
    const Ny_val = Ny;

    const dx = x1 - x0;
    const dy = y1 - y0;
    const steps_f = Math.max(Math.abs(dx), Math.abs(dy));
    const steps = Math.floor(steps_f);

    if (steps <= 0) {
        return;
    }

    const s = [];
    const intensity = [];

    // Euclidean length in "pixel units"
    const length = Math.sqrt(dx*dx + dy*dy);

    for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const x = x0 + t * dx;
        const y = y0 + t * dy;

        const ix = Math.round(x);
        const iy = Math.round(y);

        if (ix < 0 || ix >= Nx_val || iy < 0 || iy >= Ny_val) {
            continue;
        }

        const idx = iy * Nx_val + ix;
        const val = img_flat[idx];

        s.push(t * length);      // distance along line
        intensity.push(val);     // intensity at that pixel
    }

    src_line_profile.data = { s: s, intensity: intensity };
    src_line_profile.change.emit();
""")

p_img_phi.js_on_event(PanEnd, callback_line_end)

 


# ----------------- Layout and save -----------------
top_row = row(p_img_com, p_curve)
bottom_row = row(p_img_phi, p_line)
layout = column(top_row, bottom_row)

output_file("dist_0.25mu_25mu_aperture.html", title="Rocking Curve Viewer (Bokeh, 2x2)")
save(layout)

print("dist_0.25mu_25mu_aperture.html – open it in a browser and click around.")
