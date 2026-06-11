"""General triclinic unit-cell geometry (M4 Stage 4.1).

Pure math, no I/O. Cartesian setting: a-vector along lab-independent
crystal x, b-vector in the x-y plane (standard crystallographic / gemmi
convention). Cell angles are DEGREES (CIF convention) — the repo's
"angles are radians" rule applies to scan/geometry angles, hence the
explicit ``_deg`` suffix on every angle field.

Cubic fast-paths are bit-identical to the legacy v2.x formulas
(``a * np.eye(3)`` cell matrix, ``d = a / sqrt(h^2 + k^2 + l^2)``);
this is the M4 Stage 4.1 regression guarantee — do not "simplify"
them into the general branch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

_LATTICE_SYSTEMS = (
    "cubic",
    "tetragonal",
    "orthorhombic",
    "hexagonal",
    "trigonal",
    "monoclinic",
    "triclinic",
)


@dataclass(frozen=True, kw_only=True)
class UnitCell:
    """Six-parameter unit cell. Lengths in metres, angles in degrees."""

    a: float
    b: float
    c: float
    alpha_deg: float
    beta_deg: float
    gamma_deg: float

    def __post_init__(self) -> None:
        for name in ("a", "b", "c"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"cell length {name} must be finite and > 0, got {v!r}.")
        for name in ("alpha_deg", "beta_deg", "gamma_deg"):
            v = getattr(self, name)
            if not (0.0 < v < 180.0):
                raise ValueError(f"cell angle {name} must be in (0, 180) degrees, got {v!r}.")
        if not self.is_cubic:
            ca, cb, cg = (
                math.cos(math.radians(x)) for x in (self.alpha_deg, self.beta_deg, self.gamma_deg)
            )
            vol2 = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
            if vol2 <= 0.0:
                raise ValueError(
                    f"cell angles (alpha={self.alpha_deg}, beta={self.beta_deg}, "
                    f"gamma={self.gamma_deg}) do not form a valid cell "
                    f"(squared volume factor {vol2:g} <= 0)."
                )

    @classmethod
    def cubic(cls, a: float) -> UnitCell:
        """Cubic cell — the v2.x-compatible constructor."""
        return cls(a=a, b=a, c=a, alpha_deg=90.0, beta_deg=90.0, gamma_deg=90.0)

    @property
    def is_cubic(self) -> bool:
        return (
            self.b == self.a
            and self.c == self.a
            and self.alpha_deg == 90.0
            and self.beta_deg == 90.0
            and self.gamma_deg == 90.0
        )

    @cached_property
    def A(self) -> np.ndarray:
        """Real-space cell matrix; columns are the lattice vectors a, b, c."""
        if self.is_cubic:
            return self.a * np.eye(3)
        ca = math.cos(math.radians(self.alpha_deg))
        cb = math.cos(math.radians(self.beta_deg))
        cg = math.cos(math.radians(self.gamma_deg))
        sg = math.sin(math.radians(self.gamma_deg))
        v = math.sqrt(1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg)
        return np.array(
            [
                [self.a, self.b * cg, self.c * cb],
                [0.0, self.b * sg, self.c * (ca - cb * cg) / sg],
                [0.0, 0.0, self.c * v / sg],
            ]
        )

    @cached_property
    def B(self) -> np.ndarray:
        """Reciprocal cell matrix, B = 2 pi inv(A)^T; columns are a*, b*, c*."""
        if self.is_cubic:
            return (2.0 * np.pi / self.a) * np.eye(3)
        return 2.0 * np.pi * np.linalg.inv(self.A).T

    def d_spacing(self, hkl: tuple[int, int, int]) -> float:
        """Interplanar spacing d_hkl (m), metric-tensor form d = 2 pi / |B G|."""
        h, k, l = hkl
        if self.is_cubic:
            return float(self.a / np.sqrt(h * h + k * k + l * l))
        g = self.B @ np.array([h, k, l], dtype=float)
        return float(2.0 * np.pi / np.linalg.norm(g))
