"""Oblique-angle DFXM geometry: crystal mount, Appendix-A solver, image-frame rotation.

Pure math, no I/O. See docs/superpowers/specs/2026-05-28-multi-reflection-oblique-angle-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np


@dataclass(frozen=True, kw_only=True)
class CrystalMount:
    """Crystal lattice + mounting orientation. v2.3.0 supports cubic only.

    `mount_x/y/z` are Miller indices of the crystal planes aligned with the
    lab x̂/ŷ/ẑ axes. Default for Al per paper §6.1: (1,0,0)/(0,1,0)/(0,0,1).
    """

    lattice: Literal["cubic"]
    a: float
    mount_x: tuple[int, int, int]
    mount_y: tuple[int, int, int]
    mount_z: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.lattice != "cubic":
            raise ValueError(
                f"v2.3.0 supports lattice='cubic' only; got {self.lattice!r}. "
                "See [[followups-cif-crystal-structures]]; .cif/non-cubic ships in v3.0.0."
            )
        for name, m in (
            ("mount_x", self.mount_x),
            ("mount_y", self.mount_y),
            ("mount_z", self.mount_z),
        ):
            if len(m) != 3:
                raise ValueError(f"{name} must have 3 components, got {m!r}.")
            if not all(isinstance(c, int) and not isinstance(c, bool) for c in m):
                raise ValueError(f"{name} components must be integers (Miller indices), got {m!r}.")
        # Orthogonality of the (normalised) vectors (cubic-only constraint).
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

    @cached_property
    def C_s(self) -> np.ndarray:
        """Cubic cell matrix C_s = a · I.  (2π) C_s^{-T} G_hkl = Q_s^{(0)}, paper eq 24."""
        return self.a * np.eye(3)

    @cached_property
    def U_mount(self) -> np.ndarray:
        """Crystal → lab rotation matrix from normalized mount Miller indices.

        Columns are the unit vectors (mount_x, mount_y, mount_z) in lab frame.
        For the paper Al setup this is the identity.
        """
        cols = []
        for m in (self.mount_x, self.mount_y, self.mount_z):
            v = np.array(m, dtype=float)
            cols.append(v / np.linalg.norm(v))
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
