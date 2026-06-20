"""Frank-equation GNB wall recipes + builder (gnb crystal mode)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dfxm_geo.crystal.slip_systems import burgers_magnitude_of

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


def _frank_tensor(recipe, rho_hat, cell) -> np.ndarray:
    n_hat = _unit(_cartesian(recipe.n, cell))
    G = np.zeros((3, 3))
    for s, rho in zip(recipe.sets, rho_hat, strict=True):
        bhat = _unit(_cartesian(s.b, cell))
        xih = _unit(_cartesian(s.xi, cell))
        b_m = burgers_magnitude_of(s.b, cell, fraction=1.0) * 1e-6
        nxxi = np.cross(n_hat, xih)
        G += rho * np.outer(b_m * bhat, nxxi)
    return G


def _rhs_operator(recipe, theta_deg, cell) -> np.ndarray:
    a_hat = _unit(_cartesian(recipe.a, cell))
    ax = np.array(
        [[0.0, -a_hat[2], a_hat[1]], [a_hat[2], 0.0, -a_hat[0]], [-a_hat[1], a_hat[0], 0.0]]
    )
    return -2.0 * np.sin(np.deg2rad(theta_deg) / 2.0) * ax  # R: R@V == 2 sin(θ/2)(V×a)


def solve_density_scale(recipe, theta_deg, cell) -> tuple[np.ndarray, float]:
    n_hat = _unit(_cartesian(recipe.n, cell))
    rel = np.array([s.rel_density for s in recipe.sets], dtype=np.float64)
    G0 = _frank_tensor(recipe, rel, cell)  # tensor at rho0 = 1
    R = _rhs_operator(recipe, theta_deg, cell)
    e1, e2 = _in_plane_basis(n_hat)
    g = np.concatenate([G0 @ e1, G0 @ e2])
    r = np.concatenate([R @ e1, R @ e2])
    rho0 = float(g @ r / (g @ g))
    resid = float(np.linalg.norm(rho0 * g - r) / (np.linalg.norm(r) + 1e-300))
    return rel * rho0, resid


def frank_residual(recipe, rho_hat, theta_deg, cell, n_test=8, seed=0) -> float:
    n_hat = _unit(_cartesian(recipe.n, cell))
    a_hat = _unit(_cartesian(recipe.a, cell))
    e1, e2 = _in_plane_basis(n_hat)
    k = 2.0 * np.sin(np.deg2rad(theta_deg) / 2.0)
    rng = np.random.default_rng(seed)
    worst = 0.0
    G = _frank_tensor(recipe, rho_hat, cell)
    for _ in range(n_test):
        c = rng.standard_normal(2)
        V = c[0] * e1 + c[1] * e2
        rhs = k * np.cross(V, a_hat)
        worst = max(worst, float(np.linalg.norm(G @ V - rhs) / (np.linalg.norm(rhs) + 1e-300)))
    return worst


RECIPES: dict[str, WallRecipe] = {
    "leds_eq11": WallRecipe(
        name="leds_eq11",
        n=(1, 1, 1),
        a=(1, 1, 1),
        sets=(
            DislocationSet(b=(1, 0, -1), xi=(2, -1, -1), slip_plane=(1, 1, 1), rel_density=1.0),
            DislocationSet(b=(0, 1, -1), xi=(-1, 2, -1), slip_plane=(1, 1, 1), rel_density=1.0),
        ),
        frank_tol=1e-6,
    ),
    "leds_eq14": WallRecipe(
        name="leds_eq14",
        n=(1, 1, 1),
        a=(-1, 3, 1),
        sets=(
            DislocationSet(b=(1, 0, -1), xi=(1, 0, -1), slip_plane=(1, 1, 1), rel_density=1.0),
            DislocationSet(b=(0, 1, -1), xi=(1, 0, -1), slip_plane=(1, 1, 1), rel_density=1.0),
            DislocationSet(b=(1, 0, 1), xi=(1, -1, 0), slip_plane=(1, 1, -1), rel_density=3.0),
        ),
        frank_tol=1e-6,
    ),
    # frankus: APPROXIMATE per the paper (Sina's decision). In-plane B2/B5 candidate
    # (1:1:1) matches the paper's relaxed densities; the paper's ideal Eq.2 ratio is
    # 2:2:1 — documented discrepancy, flagged for G. Winther. See spike findings.
    "frankus": WallRecipe(
        name="frankus",
        n=(0, 1, 0),
        a=(0, 1, 0),
        sets=(
            DislocationSet(b=(1, -1, 0), xi=(1, 0, -1), slip_plane=(1, 1, 1), rel_density=1.0),
            DislocationSet(b=(0, 1, -1), xi=(1, 0, -1), slip_plane=(1, 1, 1), rel_density=1.0),
            DislocationSet(b=(1, 0, 1), xi=(1, 0, 1), slip_plane=(1, 1, -1), rel_density=1.0),
        ),
        frank_tol=2e-2,
    ),
}
