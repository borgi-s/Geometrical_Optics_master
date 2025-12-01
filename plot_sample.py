import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def euler_matrix(angles_deg, order="xyz"):
    """
    Rotation matrix from Euler angles (degrees).
    Default order 'xyz' = Rx @ Ry @ Rz (applied right-to-left to column vectors).
    """
    ax, ay, az = np.radians(angles_deg)
    Rx = np.array([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]])
    Ry = np.array([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]])
    Rz = np.array([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]])
    R = np.eye(3)
    for ch in order:
        if ch == "x": R = Rx @ R
        if ch == "y": R = Ry @ R
        if ch == "z": R = Rz @ R
    return R

def draw_cube(ax, side=1.2, center=(0,0,0), R=np.eye(3),
              facecolor=(0.6,0.75,1.0), edgecolor="k", alpha=0.65):
    """Axis-aligned cube (side length 'side') rotated by R and translated to center."""
    s = side/2
    V = np.array([[+s,+s,+s],[+s,+s,-s],[+s,-s,+s],[+s,-s,-s],
                  [-s,+s,+s],[-s,+s,-s],[-s,-s,+s],[-s,-s,-s]])
    V = (R @ V.T).T + np.asarray(center)
    faces_idx = [[0,2,3,1],[4,5,7,6],[0,1,5,4],[2,6,7,3],[0,4,6,2],[1,3,7,5]]
    faces = [[V[i] for i in idx] for idx in faces_idx]
    poly = Poly3DCollection(faces, facecolors=facecolor, edgecolors=edgecolor,
                            linewidths=1.0, alpha=alpha)
    ax.add_collection3d(poly)


# ------------- helpers -------------
def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v

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

def miller_body(hkl):
    """Return LaTeX body like [1 \\bar{2} 1] (no outer $...$)."""
    parts = []
    for n in np.rint(hkl).astype(int):
        parts.append(r"\bar{%d}" % -n if n < 0 else str(n))
    return "[" + " ".join(parts) + "]"


def draw_arrow(ax, origin, direction_lab, length=1.0, color="r", lw=2.5,
               label=None, text_shift=(0,0,0), arrow_length_ratio=0.15,
               text_color=None, zorder=None):
    if text_color is None:
        text_color = color
    o = np.asarray(origin, float)
    d = unit(direction_lab) * length
    ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2],
              arrow_length_ratio=arrow_length_ratio, linewidth=lw, color=color)
    if label:
        p = o + d + np.asarray(text_shift, float)

        bbox_args = None
        if zorder is not None:
            bbox_args = dict(facecolor="white", edgecolor="black",
                            alpha=0.95, boxstyle="square,pad=0.15")

        ax.text(
            p[0], p[1], p[2], label,
            fontsize=14,
            color=text_color,
            zorder=zorder if zorder is not None else 1,
            bbox=bbox_args
        )
    ax.set_box_aspect([1,1,1])

def set_equal_3d(ax, lim=1.0):
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)

# ------------- mappings (your definitions) -------------
def Theta_y(theta):
    """Rotation about y (lab y) by angle 'theta' (radians) used in r_s = Θ r_lab."""
    return np.array([[np.cos(theta),  0, np.sin(theta)],
                     [0,              1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])


def g_to_lab_matrix(theta_rad):
    """Return R_g2l = (U Θ)^(-1) mapping g → lab."""
    M = U.T @ Theta_y(theta_rad)     # lab → g
    return np.linalg.inv(M)       # g → lab

def v_g_to_lab(v_g, theta_rad):
    return g_to_lab_matrix(theta_rad) @ np.asarray(v_g, float)


def angle(v1, v2, acute=False):
    # Compute the angle between two vectors
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if acute:
        return angle
    else:
        return 2 * np.pi - angle

# ------------- CONFIG -------------
theta_deg = 17.953/2          # set your Θ angle (degrees)
# theta_deg = 20.76/2           # set your Θ angle (degrees)
theta = np.radians(theta_deg)

# Define vectors in g-coordinates (edit these!)
Q_g = np.array([1, 1, 1], float)
# Q_g = np.array([0, 0, 2], float)
y_g = np.array([1, 0, -1], float)
# y_g = np.array([1, -1, 0], float)
s_g = np.array([1, -2, 1], float)
# s_g = np.array([-1, -1, 0], float)

U = np.array([s_g/np.linalg.norm(s_g), y_g/np.linalg.norm(y_g), Q_g/np.linalg.norm(Q_g)])
# U = np.array([[1/np.sqrt(6),  -2/np.sqrt(6),  1/np.sqrt(6)],
#               [1/np.sqrt(2),   0.0,          -1/np.sqrt(2)],
#               [1/np.sqrt(3),   1/np.sqrt(3),  1/np.sqrt(3)]])


# U = np.array([[-1 / np.sqrt(2), -1 / np.sqrt(2),  0],
#               [ 1 / np.sqrt(2), -1 / np.sqrt(2),  0],
#               [ 0             ,  0             ,  2 / np.sqrt(4)]])



# Lab axes and incident beam convention
x_l = np.array([1,0,0], float)     # left→right
y_l = np.array([0,1,0], float)     # out-of-plane
z_l = np.array([0,0,1], float)     # bottom→top
k_i_dir = x_l.copy()               # incident along +x_l

# Compute transform and convert vectors to lab for plotting
R_g2l = g_to_lab_matrix(theta)
Q_l   = v_g_to_lab(Q_g, theta)
y_l_vec = -v_g_to_lab(y_g, theta)
s_l_vec = v_g_to_lab(s_g, theta)

# Diffracted direction (schematic): k_f ∥ k_i + Q̂ (in lab)




# k_f_dir = np.asarray([0.95106,0,0.30902])
k_f_dir = np.asarray([0.93507336,0,0.354454244])


# ------------- PLOT -------------
fig = plt.figure(figsize=(8, 6), dpi=400)

ax  = fig.add_subplot(111, projection="3d")


cube_euler_deg = (0, -theta_deg, 0)           # (Rx, Ry, Rz) degrees, order 'xyz'
R_sample = euler_matrix(cube_euler_deg, order="xyz")
# # Cube (sample)
draw_cube(ax, side=1.2, R=R_sample, alpha=0.3)

# Beams (red)
draw_arrow(ax, origin=(-1.7, 0, 0), direction_lab=k_i_dir, length=1.65,
           color="tab:red", label="Incident\n beam", text_shift=(-1.65,0.06,0.04))
draw_arrow(ax, origin=(0.05, 0, 0), direction_lab=k_f_dir, length=1.5,
           color="tab:red", label="Diffracted\n beam", text_shift=(-0.80,0.06,0.04))

# Q, y, s (black) — labels show g-indices
draw_arrow(ax, origin=(0,0,0), direction_lab=Q_l, length=1.2, color="k",
           label=r"$\mathbf{\hat{Q}}="+miller_body(Q_g)+r"$", text_shift=(0.05,0.02,0), arrow_length_ratio=0.2)
draw_arrow(ax, origin=(0.0,0.0,0.0), direction_lab=y_l_vec, length=1.7, color="k",
           label=r"$-\mathbf{\hat{y}}=-"+miller_body(y_g)+r"$", text_shift=(-0.15,-0.5,-0.12), arrow_length_ratio=0.2, zorder=999)
draw_arrow(ax, origin=(1.4/2,-0.61,0.2), direction_lab=s_l_vec, length=0.6, color="k",
           label=r"$\mathbf{\hat{s}}="+miller_body(s_g)+r"$", text_shift=(-0.35,0.02,-0.35), arrow_length_ratio=0.3)

# Lab triad OUTSIDE the cube (bottom-left)
lim = 1.5
triad_origin = np.array([-1.3*lim/1.7, -1.05*lim/1.7, -0.85*lim/1.7])
draw_arrow(ax, origin=triad_origin, direction_lab=x_l, length=0.45, color="k",
           label=r"$x_\ell$", text_shift=(0.02,0,0), arrow_length_ratio=0.25)
draw_arrow(ax, origin=triad_origin, direction_lab=-y_l, length=0.45, color="k",
           label=r"$-y_\ell$", text_shift=(-0.35,0.02,0.1), arrow_length_ratio=0.5)
draw_arrow(ax, origin=triad_origin, direction_lab=z_l, length=0.45, color="k",
           label=r"$z_\ell$", text_shift=(0,0,0.02), arrow_length_ratio=0.25)

# View so x is horizontal (right), z is vertical (up), y is depth (out-of-plane)
ax.view_init(elev=15, azim=-80)
# ax.view_init(elev=0, azim=0)
ax.set_axis_off()
set_equal_3d(ax, lim=lim)
plt.tight_layout()
# plt.savefig("dfxm_grain_coordinates.svg", bbox_inches="tight", pad_inches=0.01)

plt.show()
