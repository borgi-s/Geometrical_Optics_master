"""Frank-equation GNB wall recipes + builder (gnb crystal mode)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TOL = 1e-9


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _cartesian(miller, cell) -> np.ndarray:
    """Crystal-Cartesian vector for a Miller index. Cubic: identity (recipes are FCC)."""
    v = np.asarray(miller, dtype=np.float64)
    if cell is None or cell.is_cubic:
        return v
    # Non-cubic recipes are out of scope; treat directions via real-space A.
    return cell.A @ v


def _in_plane_basis(n_hat) -> tuple[np.ndarray, np.ndarray]:
    n_hat = _unit(n_hat)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(n_hat @ seed) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = _unit(seed - (seed @ n_hat) * n_hat)
    e2 = np.cross(n_hat, e1)
    return e1, e2


@dataclass(frozen=True)
class DislocationSet:
    b: tuple[int, int, int]
    xi: tuple[int, int, int]
    slip_plane: tuple[int, int, int]
    rel_density: float


@dataclass(frozen=True)
class WallRecipe:
    name: str
    n: tuple[int, int, int]
    a: tuple[int, int, int]
    sets: tuple[DislocationSet, ...]
    structure: str = "fcc"
    frank_tol: float = 1e-6

    def validate(self, cell) -> None:
        n_hat = _unit(_cartesian(self.n, cell))
        if not self.sets:
            raise ValueError(f"{self.name}: recipe has no dislocation sets")
        for i, s in enumerate(self.sets):
            b = _unit(_cartesian(s.b, cell))
            xi = _unit(_cartesian(s.xi, cell))
            sp = _unit(_cartesian(s.slip_plane, cell))
            if abs(b @ sp) > _TOL:
                raise ValueError(f"{self.name} set {i}: b not in slip_plane (b·sp={b @ sp:.2e})")
            if abs(xi @ sp) > _TOL:
                raise ValueError(f"{self.name} set {i}: xi not in slip_plane")
            if abs(xi @ n_hat) > _TOL:
                raise ValueError(
                    f"{self.name} set {i}: xi not in boundary plane (xi·n={xi @ n_hat:.2e})"
                )
            if s.rel_density <= 0:
                raise ValueError(f"{self.name} set {i}: rel_density must be > 0")
