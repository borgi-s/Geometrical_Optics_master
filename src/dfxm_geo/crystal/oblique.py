"""Oblique-angle DFXM geometry: crystal mount, Appendix-A solver, image-frame rotation.

Pure math, no I/O. See docs/superpowers/specs/2026-05-28-multi-reflection-oblique-angle-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from dfxm_geo.crystal.cell import _LATTICE_SYSTEMS, UnitCell


@dataclass(frozen=True, kw_only=True)
class CrystalMount:
    """Crystal lattice + mounting orientation.

    v3.0.0 (M4 Stage 4.1): any crystal system via the six cell parameters;
    constrained parameters are filled per ``UnitCell.from_lattice`` (e.g.
    ``lattice="hexagonal"`` needs only ``a`` and ``c``). Cell angles are
    degrees (CIF convention). The cubic path is bit-identical to v2.x.

    ``mount_x/y/z`` are Miller indices of the crystal *plane normals* aligned
    with the lab x/y/z axes; their crystal-Cartesian directions are
    ``B @ m`` (for cubic, ``B`` is proportional to the identity, so this
    reduces to the historical "Miller indices as directions" reading).
    The three directions must be mutually orthogonal in Cartesian space.
    Default for Al per paper §6.1: (1,0,0)/(0,1,0)/(0,0,1).
    """

    lattice: Literal[
        "cubic",
        "tetragonal",
        "orthorhombic",
        "hexagonal",
        "trigonal",
        "monoclinic",
        "triclinic",
    ]
    a: float
    b: float | None = None
    c: float | None = None
    alpha_deg: float | None = None
    beta_deg: float | None = None
    gamma_deg: float | None = None
    mount_x: tuple[int, int, int]
    mount_y: tuple[int, int, int]
    mount_z: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.lattice not in _LATTICE_SYSTEMS:
            raise ValueError(
                f"[crystal] lattice must be one of {_LATTICE_SYSTEMS}; got {self.lattice!r}."
            )
        # Build the cell now so invalid parameters fail at construction time.
        cell = self.cell
        for name, m in (
            ("mount_x", self.mount_x),
            ("mount_y", self.mount_y),
            ("mount_z", self.mount_z),
        ):
            if len(m) != 3:
                raise ValueError(f"{name} must have 3 components, got {m!r}.")
            if not all(isinstance(comp, int) and not isinstance(comp, bool) for comp in m):
                raise ValueError(f"{name} components must be integers (Miller indices), got {m!r}.")
        if cell.is_cubic:
            # Legacy v2.x check, kept verbatim: raw Miller vectors, tol 1e-12.
            vx = np.array(self.mount_x, dtype=float)
            vy = np.array(self.mount_y, dtype=float)
            vz = np.array(self.mount_z, dtype=float)
            for n1, v1, n2, v2 in (
                ("mount_x", vx, "mount_y", vy),
                ("mount_x", vx, "mount_z", vz),
                ("mount_y", vy, "mount_z", vz),
            ):
                dot = float(v1 @ v2)
                if abs(dot) > 1e-12:
                    raise ValueError(
                        f"crystal mount vectors must be mutually orthogonal: "
                        f"{n1}={v1.tolist()}, {n2}={v2.tolist()} have dot product = {dot}."
                    )
        else:
            # General cells: orthogonality of the Cartesian plane-normal
            # directions B @ m (normalized), tol 1e-9.
            B = cell.B
            units: dict[str, np.ndarray] = {}
            for name, m in (
                ("mount_x", self.mount_x),
                ("mount_y", self.mount_y),
                ("mount_z", self.mount_z),
            ):
                u = B @ np.array(m, dtype=float)
                units[name] = u / np.linalg.norm(u)
            for n1, n2 in (("mount_x", "mount_y"), ("mount_x", "mount_z"), ("mount_y", "mount_z")):
                dot = float(units[n1] @ units[n2])
                if abs(dot) > 1e-9:
                    angle_deg = float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
                    raise ValueError(
                        f"crystal mount plane normals must be mutually orthogonal in "
                        f"Cartesian space: {n1}={list(getattr(self, n1))} and "
                        f"{n2}={list(getattr(self, n2))} subtend {angle_deg:.4f} deg "
                        f"(lattice={self.lattice!r}). Pick Miller indices whose "
                        f"reciprocal vectors B@m are orthogonal."
                    )

    @cached_property
    def cell(self) -> UnitCell:
        """The six-parameter unit cell (constrained params filled per lattice)."""
        return UnitCell.from_lattice(
            self.lattice,
            a=self.a,
            b=self.b,
            c=self.c,
            alpha_deg=self.alpha_deg,
            beta_deg=self.beta_deg,
            gamma_deg=self.gamma_deg,
        )

    @cached_property
    def C_s(self) -> np.ndarray:
        """Real-space cell matrix.  (2π) C_s^{-T} G_hkl = Q_s^{(0)}, paper eq 24.

        Cubic: exactly ``a * eye(3)`` (bit-identical to v2.x).
        """
        return self.cell.A

    @cached_property
    def U_mount(self) -> np.ndarray:
        """Crystal → lab rotation matrix from the mount plane normals.

        Columns are the unit Cartesian directions of (mount_x, mount_y,
        mount_z). Cubic keeps the legacy raw-Miller normalization verbatim
        (bit-identical); general cells route through B @ m.
        For the paper Al setup this is the identity.
        """
        if self.cell.is_cubic:
            cols = []
            for m in (self.mount_x, self.mount_y, self.mount_z):
                v = np.array(m, dtype=float)
                cols.append(v / np.linalg.norm(v))
            return np.column_stack(cols)
        B = self.cell.B
        cols = []
        for m in (self.mount_x, self.mount_y, self.mount_z):
            u = B @ np.array(m, dtype=float)
            cols.append(u / np.linalg.norm(u))
        return np.column_stack(cols)


def _R_x(angle: float) -> np.ndarray:
    """Rotation around lab x̂ (beam axis) by `angle` rad."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _R_y(angle: float) -> np.ndarray:
    """Rotation around lab ŷ by `angle` rad."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _R_z(angle: float) -> np.ndarray:
    """Rotation around lab ẑ by `angle` rad."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def R_lab_to_image(eta: float, theta: float) -> np.ndarray:
    """Lab → image-detector-frame rotation.

    Generalized from the v2.2.0 simplified-geometry implicit rotation R_y(-2θ)
    to include the azimuthal rotation R_x(η) around the beam axis. At η=0
    this collapses bit-identically to v2.2.0.

    The image frame is defined such that the detector normal points along
    the diffracted-beam direction in the image frame.
    """
    return _R_x(eta) @ _R_y(-2.0 * theta)


def _solve_quadratic_in_tan_half(α0: float, α1: float, α2: float) -> tuple[float, float]:
    """Solve eq A.7: (α₂ - α₀)s² + 2α₁s + (α₀ + α₂) = 0 for s = tan(ω/2).

    Returns (s1, s2). When the discriminant < 0 both are NaN; when the
    quadratic degenerates (α₂ - α₀ = 0) the single root is in s1 and s2 is NaN.
    """
    A = α2 - α0
    B = 2.0 * α1
    C = α0 + α2

    if abs(A) < 1e-15:
        # Linear case: B s + C = 0
        if abs(B) < 1e-15:
            return float("nan"), float("nan")
        return -C / B, float("nan")

    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return float("nan"), float("nan")

    sqrt_disc = float(np.sqrt(disc))
    return (-B + sqrt_disc) / (2.0 * A), (-B - sqrt_disc) / (2.0 * A)


@dataclass(frozen=True)
class ReflectionGeometry:
    """One Laue solution for a (mount, hkl, keV) triple.

    Two ω-solutions per reflection per paper Appendix A; both stored.
    All angles in radians. NaN when no real ω exists.
    """

    hkl: tuple[int, int, int]
    keV: float
    omega_1: float
    eta_1: float
    theta_1: float
    omega_2: float
    eta_2: float
    theta_2: float


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric (cross-product) matrix K such that K @ u = v × u."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _eta_from_k_out(k_out: np.ndarray) -> float:
    """η = signed azimuth of k_out around lab x̂ (paper eq A.12, A.13).

    Sign convention matches darkmod (laue.get_eta_angle): η = -sign(k_y) · arccos(k_z / √(k_y² + k_z²)).
    Returns NaN when the y-z projection is zero (k_out parallel to x̂).
    """
    k_yz_norm = float(np.sqrt(k_out[1] ** 2 + k_out[2] ** 2))
    if k_yz_norm < 1e-15:
        return float("nan")
    return -float(np.sign(k_out[1])) * float(np.arccos(k_out[2] / k_yz_norm))


def compute_omega_eta(
    mount: CrystalMount,
    hkl: tuple[int, int, int],
    keV: float,
    *,
    rotation_axis: np.ndarray | None = None,
) -> ReflectionGeometry:
    """Solve paper Appendix A for (ω, η) at the given (mount, hkl, keV).

    Default rotation_axis = lab ẑ (paper §A: "selecting the rotation axis to be ẑ_l").
    Returns two ω-solutions per reflection (paper Table A.2 shows ω₁, ω₂);
    NaN-filled when no real ω solution exists at the keV.
    """
    if rotation_axis is None:
        rotation_axis = np.array([0.0, 0.0, 1.0])

    G_hkl = np.array(hkl, dtype=float)
    C_s_inv_T = np.linalg.inv(mount.C_s).T
    Q_s0 = 2.0 * np.pi * C_s_inv_T @ G_hkl  # paper eq 24, in 1/m
    Q_lab_static = mount.U_mount @ Q_s0  # mount-rotated, pre-ω

    wavelength = 1.239841984e-9 / keV  # m, hc/E
    k_l = (2.0 * np.pi / wavelength) * np.array([1.0, 0.0, 0.0])

    K = _skew(rotation_axis)
    K2 = K @ K

    α0 = float(-k_l @ K2 @ Q_lab_static)
    α1 = float(k_l @ K @ Q_lab_static)
    α2 = float(k_l @ (np.eye(3) + K2) @ Q_lab_static + (Q_lab_static @ Q_lab_static) / 2.0)

    s1, s2 = _solve_quadratic_in_tan_half(α0, α1, α2)

    results: list[tuple[float, float, float]] = []
    for s in (s1, s2):
        if np.isnan(s):
            results.append((float("nan"), float("nan"), float("nan")))
            continue
        omega_raw = 2.0 * float(np.arctan(s))
        omega = float(omega_raw % (2.0 * np.pi))  # wrap to [0, 2π)
        # Rotate Q_lab_static by ω around rotation_axis (Rodrigues).
        K_axis = K
        K_axis2 = K2
        R = np.eye(3) + np.sin(omega) * K_axis + (1.0 - np.cos(omega)) * K_axis2
        Q_lab = R @ Q_lab_static
        k_out = Q_lab + k_l
        eta = _eta_from_k_out(k_out)
        # θ from |Q| and wavelength: sin(θ) = |Q| · λ / (4π)
        sin_theta = float(np.linalg.norm(Q_lab) * wavelength / (4.0 * np.pi))
        theta = float("nan") if sin_theta > 1.0 else float(np.arcsin(sin_theta))
        results.append((omega, eta, theta))

    # Sort by ascending ω so that ω₁ < ω₂ (paper Table A.2 convention).
    # NaN omegas sort last (nan comparisons are well-defined in this context
    # because we use explicit checks; rely on Python's nan < x == False).
    results.sort(
        key=lambda t: t[0] if not (isinstance(t[0], float) and t[0] != t[0]) else float("inf")
    )
    (ω1, η1, θ1), (ω2, η2, θ2) = results
    return ReflectionGeometry(
        hkl=tuple(int(c) for c in hkl),  # type: ignore[arg-type]
        keV=float(keV),
        omega_1=ω1,
        eta_1=η1,
        theta_1=θ1,
        omega_2=ω2,
        eta_2=η2,
        theta_2=θ2,
    )


def find_reflections(
    mount: CrystalMount,
    keV: float,
    *,
    theta_range: tuple[float, float] = (0.0, float(np.deg2rad(16.25))),
    hkl_max: int = 5,
    eta_target: float | None = None,
    eta_tol: float = 1e-6,
) -> list[ReflectionGeometry]:
    """Enumerate Laue-satisfying reflections within |h|,|k|,|l| ≤ hkl_max.

    Returns a list sorted by primary η, then primary θ. When `eta_target` is
    given, only reflections whose η₁ OR η₂ is within `eta_tol` of the target
    are kept. NaN-filled rows (unreachable Laue at this keV) are dropped.

    Phase A: implemented and unit-tested. Phase B wires this into config-load
    validation and the dfxm-find-reflections CLI.
    """
    results: list[ReflectionGeometry] = []
    for h in range(-hkl_max, hkl_max + 1):
        for k in range(-hkl_max, hkl_max + 1):
            for l in range(-hkl_max, hkl_max + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                geom = compute_omega_eta(mount, (h, k, l), keV)
                if np.isnan(geom.omega_1) and np.isnan(geom.omega_2):
                    continue
                # θ-range filter (use whichever solution exists)
                theta = geom.theta_1 if not np.isnan(geom.theta_1) else geom.theta_2
                if not (theta_range[0] <= theta <= theta_range[1]):
                    continue
                if eta_target is not None:
                    e1 = geom.eta_1 if not np.isnan(geom.eta_1) else None
                    e2 = geom.eta_2 if not np.isnan(geom.eta_2) else None
                    match = any(e is not None and abs(e - eta_target) <= eta_tol for e in (e1, e2))
                    if not match:
                        continue
                results.append(geom)

    def _sort_key(g: ReflectionGeometry) -> tuple[float, float]:
        eta = g.eta_1 if not np.isnan(g.eta_1) else g.eta_2
        theta = g.theta_1 if not np.isnan(g.theta_1) else g.theta_2
        return (
            float(eta) if not np.isnan(eta) else np.inf,
            float(theta) if not np.isnan(theta) else np.inf,
        )

    return sorted(results, key=_sort_key)
