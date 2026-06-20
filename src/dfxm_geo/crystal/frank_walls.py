"""Frank-equation GNB wall recipes + builder (gnb crystal mode)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dfxm_geo.crystal.slip_systems import burgers_magnitude_of

_TOL = 1e-9


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    nrm = np.linalg.norm(v)
    if nrm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return v / nrm


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


def _edge_t(slip_plane, b) -> np.ndarray:
    """Input t vector for _ud_matrix_from_bnt (before the det-flip inside it).

    After the flip, Ud[:,2] = b×n (the kernel's reference edge t_0).
    We still pass slip_plane×b here as the 'raw' t so that the det-flip
    inside _ud_matrix_from_bnt yields the correct b×n column.
    """
    return np.cross(np.asarray(slip_plane, float), np.asarray(b, float))


def _signed_angle(t0: np.ndarray, target: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle (deg) rotating t0 -> target about axis. All args Cartesian."""
    t0h, th, ax = _unit(t0), _unit(target), _unit(axis)
    sin = float(np.cross(t0h, th) @ ax)
    cos = float(np.clip(t0h @ th, -1.0, 1.0))
    return float(np.degrees(np.arctan2(sin, cos)))


def build_wall_population(
    recipe,
    *,
    theta_deg,
    extent_um,
    cell,
    ny,
    crystal_to_lab,
    max_dislocations=None,
):
    """Assemble a DislocationPopulation for the gnb crystal mode.

    Parameters
    ----------
    recipe : WallRecipe
        Validated Frank-equation recipe (from RECIPES or custom).
    theta_deg : float
        Misorientation angle across the boundary (degrees).
    extent_um : float
        Total in-plane width of the wall in micrometres; dislocations are placed
        symmetrically over [-extent_um/2, +extent_um/2].
    cell : UnitCell
        Crystal unit cell (cubic for FCC recipes).
    ny : float
        Isotropic Poisson ratio.
    crystal_to_lab : array-like, shape (3, 3)
        Rotation from crystal frame to lab/sample frame.
    max_dislocations : int or None
        If not None, raise ValueError when the total dislocation count exceeds
        this limit (guard against accidental huge walls).

    Returns
    -------
    DislocationPopulation
    """
    if cell is not None and not cell.is_cubic:
        raise NotImplementedError(
            "gnb walls currently support cubic cells only (FCC/BCC); non-cubic "
            "(e.g. HCP) custom recipes are a follow-up because plane normals need "
            "reciprocal-space (B-matrix) handling. See docs/gnb-walls.md."
        )

    from dfxm_geo.direct_space.forward_model import (  # function-local: breaks import cycle  # type: ignore[attr-defined]
        DislocationPopulation,
        _ud_matrix_from_bnt,
    )

    recipe.validate(cell)
    rho_hat, resid = solve_density_scale(recipe, theta_deg, cell)
    if resid > recipe.frank_tol:
        raise ValueError(
            f"{recipe.name}: Frank residual {resid:.2e} exceeds tol {recipe.frank_tol:.2e}"
        )
    n_hat = _unit(_cartesian(recipe.n, cell))
    R_place = np.asarray(crystal_to_lab, dtype=np.float64)

    positions, Ud_list, rot_list, b_list = [], [], [], []
    for s, rho in zip(recipe.sets, rho_hat, strict=True):
        d_um = (1.0 / rho) * 1e6
        xih = _unit(_cartesian(s.xi, cell))
        u = _unit(np.cross(n_hat, xih))  # in-plane perpendicular (position offset direction)
        n_lines = int(np.floor(extent_um / d_um)) + 1
        ks = np.arange(n_lines) - (n_lines - 1) / 2.0
        # Build Ud FIRST; the det-flip inside _ud_matrix_from_bnt makes Ud[:,2] = b×n
        # (the kernel's reference edge t_0). Then measure rotation_deg from Ud[:,2] → xi
        # about Ud[:,1] (the slip-plane normal).
        Ud = _ud_matrix_from_bnt(s.b, s.slip_plane, _edge_t(s.slip_plane, s.b))
        rot = _signed_angle(
            Ud[:, 2], xih, Ud[:, 1]
        )  # POST-FLIP edge (b×n) → xi about slip-plane normal
        b_um = burgers_magnitude_of(s.b, cell, fraction=1.0)
        for k in ks:
            positions.append(R_place @ ((k * d_um) * u))
            Ud_list.append(Ud)
            rot_list.append(rot)
            b_list.append(b_um)

    if max_dislocations is not None and len(positions) > max_dislocations:
        raise ValueError(
            f"{recipe.name}: {len(positions)} dislocations exceeds max_dislocations="
            f"{max_dislocations}; increase theta_deg or reduce extent_um"
        )

    return DislocationPopulation(
        positions_um=np.asarray(positions, dtype=np.float64),
        Ud=np.asarray(Ud_list, dtype=np.float64),
        sidecar={
            "recipe": recipe.name,
            "theta_deg": theta_deg,
            "rho_hat_m_inv": rho_hat.tolist(),
            "frank_residual": resid,
        },
        rotation_deg=np.asarray(rot_list, dtype=np.float64),
        b_um_per=np.asarray(b_list, dtype=np.float64),
        ny=ny,
    )


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
