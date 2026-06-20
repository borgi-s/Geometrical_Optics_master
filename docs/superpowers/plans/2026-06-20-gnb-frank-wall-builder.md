# GNB Frank-equation wall builder (`gnb` mode) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `mode = "gnb"` crystal layout that builds a 2-D dislocation network (a geometrically necessary boundary) from a Frank-equation recipe and renders it through the existing forward model.

**Architecture:** A new pure-physics module `crystal/frank_walls.py` (recipe types, registry of 3 literature recipes + a custom hatch, a θ→density solver, a Frank-residual gate, and `build_wall_population`) populates the existing `DislocationPopulation`; a new `gnb` arm of `CrystalConfig` and a `gnb` branch in `build_dislocation_population` wire it in. No forward-engine changes. The three legacy modes (`centered`/`wall`/`random_dislocations`) stay byte-identical.

**Tech Stack:** Python 3, numpy, numba (existing kernels, untouched), pytest, mypy, TOML config.

## Global Constraints

- **venv interpreter only:** `C:\Users\borgi\Documents\GM-reworked\wt-gnb-walls\.venv-wt\Scripts\python.exe` (bash `python` is Python 2.7 — never use it). All `pytest`/`mypy`/`python` commands below use this interpreter.
- **Worktree:** all work on branch `feature/gnb-walls` in `C:\Users\borgi\Documents\GM-reworked\wt-gnb-walls`. Repo package root is `src/dfxm_geo`.
- **Byte-identity is sacred.** Do NOT modify `centered`/`wall`/`random_dislocations` code paths, the `_FCC_111_110_ORDERED` table, or `BURGERS_VECTOR`. The determinism gates (`tests/test_cubic_bit_identity.py`) must stay green.
- **mypy must stay at 0 errors** for `src/dfxm_geo` (current baseline 0/51).
- **Frank residual tolerance is per-recipe:** strict `1e-6` for `leds_eq11`/`leds_eq14`/exact custom; loose documented (`2e-2`) for `frankus` (approximate per the paper — Sina's decision).
- **No circular import:** `frank_walls.py` imports `forward_model` symbols ONLY via function-local imports inside `build_wall_population`; `forward_model`'s `gnb` branch imports `build_wall_population` via a function-local import. `frank_walls` may import `slip_systems` at module level (no cycle).
- **Every commit** ends with the two trailers from CLAUDE.md:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01Pnzi4PyKfqumRWHLzso7Md`.
- **Reactions excluded, forward-only, infinite straight dislocations, isotropic ν** (per spec scope).

## File Structure

- **Create** `src/dfxm_geo/crystal/frank_walls.py` — recipe types, registry, solver, residual gate, `build_wall_population`. The whole feature's physics.
- **Modify** `src/dfxm_geo/config.py` — add `GnbCrystalConfig`, `GnbCustomConfig`, `GnbSetConfig`; add `"gnb"` to the crystal mode union + `from_dict` parsing.
- **Modify** `src/dfxm_geo/direct_space/forward_model.py` — add the `gnb` branch in `build_dislocation_population` (function-local import; placement matrix).
- **Create** `tests/test_frank_walls.py` — unit tests (helpers, validate, solver, residual, recipes, build).
- **Create** `tests/test_gnb_config.py` — config resolution + parsing tests.
- **Create** `tests/test_gnb_e2e.py` — wiring, frame round-trip, e2e render, golden, g·b sanity.
- **Create** `src/dfxm_geo/data/configs/gnb_leds_eq11.toml`, `gnb_frankus.toml` — example configs.
- **Create** `docs/gnb-walls.md` — user docs.
- **Create** `tests/data/golden/gnb/*.npy` — golden snapshots (Task 8).

---

### Task 0: Worktree environment + green baseline

**Files:** none (environment only).

- [ ] **Step 1: Create the worktree venv**

Run:
```bash
"C:/Users/borgi/AppData/Local/Programs/Python/Python311/python.exe" -m venv "C:/Users/borgi/Documents/GM-reworked/wt-gnb-walls/.venv-wt" || py -3 -m venv "C:/Users/borgi/Documents/GM-reworked/wt-gnb-walls/.venv-wt"
```
(If the exact base-Python path differs, use the one that backs the existing `.venv`: `C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -c "import sys;print(sys.base_prefix)"`.)

- [ ] **Step 2: Editable install with dev extras**

Run:
```bash
cd "C:/Users/borgi/Documents/GM-reworked/wt-gnb-walls"
".venv-wt/Scripts/python.exe" -m pip install -e ".[dev]"
```
Expected: installs `dfxm_geo` editable + dev deps, exit 0.

- [ ] **Step 3: Clear any stale Fg caches (per CLAUDE.md, they cause spurious failures)**

Run:
```bash
rm -f "C:/Users/borgi/Documents/GM-reworked/wt-gnb-walls/direct_space/deformation_gradient_tensors/Fg_"*.npy 2>/dev/null; echo done
```

- [ ] **Step 4: Baseline gate — default suite + mypy**

Run:
```bash
".venv-wt/Scripts/python.exe" -m pytest -q
".venv-wt/Scripts/python.exe" -m mypy src/dfxm_geo
```
Expected: default suite green (≈1133 passed / 2 skip / 1 xfail per release records); mypy `Success: no issues found`. If anything unexpected fails, STOP and report before proceeding.

- [ ] **Step 5: No commit** (environment only).

---

### Task 1: Spike — pin crystal→lab placement + confirm frankus (exploratory, not TDD)

**Files:**
- Create (throwaway): `_spike_gnb.py` at the worktree root.
- Create: `docs/superpowers/notes/2026-06-20-gnb-spike-findings.md`

**Interfaces:**
- Produces: the constant `_CRYSTAL_TO_LAB` (3,3) placement matrix + the confirmed `frankus` density ratio + achieved Frank residual, recorded in the findings note for Tasks 4/7.

- [ ] **Step 1: Write the spike script**

Create `_spike_gnb.py` that (a) builds a minimal `leds_eq11` two-family net of infinite lines for one θ directly as a `DislocationPopulation`, renders it via the existing forward path for a simplified-geometry config, and checks where the boundary plane lands; (b) numerically confirms the frankus candidate (`b_B2=[1,-1,0]`, `b_B5=[0,1,-1]`, collinear `[1,0,1]`, ratio 1:1:1) reproduces the paper's near-in-plane `b_eff` (∝ `V×a`). Use the venv python. Reuse `dfxm_geo.config.load_config`/`forward` (or `build_dislocation_population` + `Find_Hg_from_population`) to render. Start with `_CRYSTAL_TO_LAB = np.eye(3)` and iterate.

- [ ] **Step 2: Determine the placement matrix**

Run `".venv-wt/Scripts/python.exe" _spike_gnb.py`. Inspect: do the rendered dislocation lines lie in the intended boundary plane? Adjust `_CRYSTAL_TO_LAB` (reasoning from the frame chain `M = Ud.T @ Us.T @ S.T @ Theta`: positions are lab-frame, so the placement that makes a line's lab geometry consistent with its `Ud`-derived direction is the rotation that inverts the orientation part the kernel applies). Iterate until the round-trip holds: for each set, the rendered line direction is ⊥ the boundary normal `n`.
Expected: a concrete 3×3 `_CRYSTAL_TO_LAB` (likely a permutation/`Us`-derived rotation, possibly `np.eye(3)`).

- [ ] **Step 3: Confirm frankus numerically**

In the spike, compute `b_eff(V) = Σ c_i (|b|·b̂_i)((n×ξ̂_i)·V)` for the frankus candidate at ratio 1:1:1, for `V=[0,0,-1]`, and the angle to `V×a=[1,0,0]`.
Expected: `b_eff ∝ [1,0,0]`, angle ≈ 0° (candidate is the exact in-plane net; matches the paper's relaxed 1:1:1, deviating from the paper's ideal Eq. 2 2:2:1).

- [ ] **Step 4: Record findings**

Write `docs/superpowers/notes/2026-06-20-gnb-spike-findings.md` with: the chosen `_CRYSTAL_TO_LAB` matrix (exact numbers), the round-trip evidence, the frankus ratio decision (1:1:1) + achieved residual, and the `rotation_deg` sign convention observed (whether `_signed_angle` needs negation to match the kernel).

- [ ] **Step 5: Commit the findings (delete the throwaway script)**

```bash
rm -f _spike_gnb.py
git add docs/superpowers/notes/2026-06-20-gnb-spike-findings.md
git commit -m "docs(gnb): spike findings — crystal->lab placement + frankus confirmation"
```

---

### Task 2: `frank_walls.py` — helpers, recipe types, `validate`

**Files:**
- Create: `src/dfxm_geo/crystal/frank_walls.py`
- Test: `tests/test_frank_walls.py`

**Interfaces:**
- Produces: `DislocationSet`, `WallRecipe` (frozen dataclasses), `_unit`, `_cartesian(v, cell)`, `_in_plane_basis(n_hat)`, `WallRecipe.validate(cell)`.

- [ ] **Step 1: Write failing tests for helpers + validate**

In `tests/test_frank_walls.py`:
```python
import numpy as np
import pytest
from dfxm_geo.crystal import frank_walls as fw
from dfxm_geo.crystal.cell import UnitCell  # existing UnitCell

CUBIC = UnitCell.cubic(4.05e-10)  # Al lattice param a in METRES (UnitCell.cubic takes metres; burgers_magnitude_of returns µm)

def test_unit_normalizes():
    assert np.allclose(np.linalg.norm(fw._unit([3.0, 0.0, 4.0])), 1.0)

def test_in_plane_basis_orthonormal_and_in_plane():
    n = fw._unit([1.0, 1.0, 1.0])
    e1, e2 = fw._in_plane_basis(n)
    assert abs(e1 @ n) < 1e-12 and abs(e2 @ n) < 1e-12
    assert abs(e1 @ e2) < 1e-12
    assert np.allclose([np.linalg.norm(e1), np.linalg.norm(e2)], 1.0)

def test_validate_accepts_good_recipe():
    r = fw.WallRecipe(
        name="t", n=(1, 1, 1), a=(1, 1, 1),
        sets=(fw.DislocationSet(b=(1, 0, -1), xi=(2, -1, -1), slip_plane=(1, 1, 1), rel_density=1.0),),
    )
    r.validate(CUBIC)  # no raise

def test_validate_rejects_b_not_in_slip_plane():
    r = fw.WallRecipe(
        name="bad", n=(1, 1, 1), a=(1, 1, 1),
        sets=(fw.DislocationSet(b=(1, 1, 1), xi=(2, -1, -1), slip_plane=(1, 1, 1), rel_density=1.0),),
    )
    with pytest.raises(ValueError, match="slip_plane"):
        r.validate(CUBIC)
```

- [ ] **Step 2: Run to verify failure**

Run: `".venv-wt/Scripts/python.exe" -m pytest tests/test_frank_walls.py -q`
Expected: FAIL (module `frank_walls` not found / attributes missing).

- [ ] **Step 3: Implement helpers + types + validate**

Create `src/dfxm_geo/crystal/frank_walls.py`:
```python
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
                raise ValueError(f"{self.name} set {i}: xi not in boundary plane (xi·n={xi @ n_hat:.2e})")
            if s.rel_density <= 0:
                raise ValueError(f"{self.name} set {i}: rel_density must be > 0")
```

- [ ] **Step 4: Run to verify pass**

Run: `".venv-wt/Scripts/python.exe" -m pytest tests/test_frank_walls.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/crystal/frank_walls.py tests/test_frank_walls.py
git commit -m "feat(gnb): frank_walls helpers, recipe types, validate"
```

---

### Task 3: θ→density solver + Frank-residual gate

**Files:**
- Modify: `src/dfxm_geo/crystal/frank_walls.py`
- Test: `tests/test_frank_walls.py`

**Interfaces:**
- Consumes: `_unit`, `_cartesian`, `_in_plane_basis`, `WallRecipe`, `burgers_magnitude_of`.
- Produces: `solve_density_scale(recipe, theta_deg, cell) -> tuple[np.ndarray, float]` (rho_hat in m⁻¹, residual); `frank_residual(recipe, rho_hat, theta_deg, cell, n_test=8, seed=0) -> float`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_frank_walls.py`:
```python
def _eq11():
    return fw.WallRecipe(
        name="leds_eq11", n=(1, 1, 1), a=(1, 1, 1),
        sets=(
            fw.DislocationSet(b=(1, 0, -1), xi=(2, -1, -1), slip_plane=(1, 1, 1), rel_density=1.0),
            fw.DislocationSet(b=(0, 1, -1), xi=(-1, 2, -1), slip_plane=(1, 1, 1), rel_density=1.0),
        ),
    )

def test_solver_residual_tiny_for_eq11():
    rho_hat, resid = fw.solve_density_scale(_eq11(), theta_deg=0.05, cell=CUBIC)
    assert resid < 1e-6
    assert np.all(rho_hat > 0)
    assert np.allclose(rho_hat[0], rho_hat[1])  # ratio 1:1

def test_spacing_matches_frank_relation():
    rho_hat, _ = fw.solve_density_scale(_eq11(), theta_deg=0.05, cell=CUBIC)
    b_m = burgers_magnitude_of((1, 0, -1), CUBIC, fraction=1.0) * 1e-6
    d_expected = b_m / (2 * np.sin(np.deg2rad(0.05) / 2))
    d_actual = 1.0 / rho_hat[0]
    assert d_actual == pytest.approx(d_expected, rel=0.05)

def test_frank_residual_matches_solver():
    r = _eq11()
    rho_hat, resid = fw.solve_density_scale(r, 0.05, CUBIC)
    assert fw.frank_residual(r, rho_hat, 0.05, CUBIC) < 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `".venv-wt/Scripts/python.exe" -m pytest tests/test_frank_walls.py -k "solver or spacing or frank_residual" -q`
Expected: FAIL (`solve_density_scale` not defined).

- [ ] **Step 3: Implement solver + residual**

Append to `frank_walls.py`:
```python
def _frank_tensor(recipe, rho_hat, cell) -> np.ndarray:
    n_hat = _unit(_cartesian(recipe.n, cell))
    G = np.zeros((3, 3))
    for s, rho in zip(recipe.sets, rho_hat):
        bhat = _unit(_cartesian(s.b, cell))
        xih = _unit(_cartesian(s.xi, cell))
        b_m = burgers_magnitude_of(s.b, cell, fraction=1.0) * 1e-6
        nxxi = np.cross(n_hat, xih)
        G += rho * np.outer(b_m * bhat, nxxi)
    return G


def _rhs_operator(recipe, theta_deg, cell) -> np.ndarray:
    a_hat = _unit(_cartesian(recipe.a, cell))
    ax = np.array([[0.0, -a_hat[2], a_hat[1]],
                   [a_hat[2], 0.0, -a_hat[0]],
                   [-a_hat[1], a_hat[0], 0.0]])
    return -2.0 * np.sin(np.deg2rad(theta_deg) / 2.0) * ax  # R: R@V == 2 sin(θ/2)(V×a)


def solve_density_scale(recipe, theta_deg, cell) -> tuple[np.ndarray, float]:
    n_hat = _unit(_cartesian(recipe.n, cell))
    rel = np.array([s.rel_density for s in recipe.sets], dtype=np.float64)
    G0 = _frank_tensor(recipe, rel, cell)            # tensor at rho0 = 1
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
```

- [ ] **Step 4: Run to verify pass**

Run: `".venv-wt/Scripts/python.exe" -m pytest tests/test_frank_walls.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/crystal/frank_walls.py tests/test_frank_walls.py
git commit -m "feat(gnb): theta->density solver + Frank residual gate"
```

---

### Task 4: Recipe registry (`leds_eq11`, `leds_eq14`, `frankus`)

**Files:**
- Modify: `src/dfxm_geo/crystal/frank_walls.py`
- Test: `tests/test_frank_walls.py`

**Interfaces:**
- Produces: `RECIPES: dict[str, WallRecipe]` with keys `"leds_eq11"`, `"leds_eq14"`, `"frankus"`.

- [ ] **Step 1: Write failing tests (verify each recipe against the Frank equation)**

Append:
```python
@pytest.mark.parametrize("name,strict", [("leds_eq11", True), ("leds_eq14", True), ("frankus", False)])
@pytest.mark.parametrize("theta", [0.02, 0.05, 0.2])
def test_registry_recipes_satisfy_frank(name, strict, theta):
    r = fw.RECIPES[name]
    r.validate(CUBIC)
    rho_hat, resid = fw.solve_density_scale(r, theta, CUBIC)
    tol = 1e-6 if strict else r.frank_tol
    assert resid < tol, f"{name} residual {resid:.2e} >= {tol:.2e}"
    assert fw.frank_residual(r, rho_hat, theta, CUBIC) < tol

def test_eq14_density_ratio_1_1_3():
    r = fw.RECIPES["leds_eq14"]
    rels = [s.rel_density for s in r.sets]
    assert rels == [1.0, 1.0, 3.0]

def test_frankus_documents_discrepancy():
    assert fw.RECIPES["frankus"].frank_tol >= 1e-3  # approximate per the paper
```

- [ ] **Step 2: Run to verify failure**

Run: `".venv-wt/Scripts/python.exe" -m pytest tests/test_frank_walls.py -k "registry or eq14_density or discrepancy" -q`
Expected: FAIL (`RECIPES` not defined).

- [ ] **Step 3: Implement the registry**

Append to `frank_walls.py` (vectors verified: eq11/eq14 exact to ~1e-16; frankus is the in-plane 1:1:1 candidate confirmed in Task 1):
```python
RECIPES: dict[str, WallRecipe] = {
    "leds_eq11": WallRecipe(
        name="leds_eq11", n=(1, 1, 1), a=(1, 1, 1),
        sets=(
            DislocationSet(b=(1, 0, -1), xi=(2, -1, -1), slip_plane=(1, 1, 1), rel_density=1.0),
            DislocationSet(b=(0, 1, -1), xi=(-1, 2, -1), slip_plane=(1, 1, 1), rel_density=1.0),
        ),
        frank_tol=1e-6,
    ),
    "leds_eq14": WallRecipe(
        name="leds_eq14", n=(1, 1, 1), a=(-1, 3, 1),
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
        name="frankus", n=(0, 1, 0), a=(0, 1, 0),
        sets=(
            DislocationSet(b=(1, -1, 0), xi=(1, 0, -1), slip_plane=(1, 1, 1), rel_density=1.0),
            DislocationSet(b=(0, 1, -1), xi=(1, 0, -1), slip_plane=(1, 1, 1), rel_density=1.0),
            DislocationSet(b=(1, 0, 1), xi=(1, 0, 1), slip_plane=(1, 1, -1), rel_density=1.0),
        ),
        frank_tol=2e-2,
    ),
}
```

- [ ] **Step 4: Run to verify pass**

Run: `".venv-wt/Scripts/python.exe" -m pytest tests/test_frank_walls.py -q`
Expected: PASS (all, incl. the parametrized Frank checks).

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/crystal/frank_walls.py tests/test_frank_walls.py
git commit -m "feat(gnb): recipe registry (leds_eq11, leds_eq14, frankus)"
```

---

### Task 5: `build_wall_population`

**Files:**
- Modify: `src/dfxm_geo/crystal/frank_walls.py`
- Test: `tests/test_frank_walls.py`

**Interfaces:**
- Consumes: solver, residual, `DislocationPopulation` + `_ud_matrix_from_bnt` (function-local import from `forward_model`).
- Produces: `build_wall_population(recipe, *, theta_deg, extent_um, cell, ny, crystal_to_lab, max_dislocations=None) -> DislocationPopulation`; `_edge_t(slip_plane, b)`; `_signed_angle(t0, xi, plane, cell)`.

- [ ] **Step 1: Write failing tests**

Append:
```python
def test_build_population_shapes_and_ratio():
    r = fw.RECIPES["leds_eq11"]
    pop = fw.build_wall_population(
        r, theta_deg=0.05, extent_um=10.0, cell=CUBIC, ny=0.334, crystal_to_lab=np.eye(3),
    )
    n = pop.positions_um.shape[0]
    assert pop.Ud.shape == (n, 3, 3)
    assert pop.rotation_deg.shape == (n,)
    assert pop.b_um_per.shape == (n,)
    assert pop.sidecar["recipe"] == "leds_eq11"
    assert pop.sidecar["frank_residual"] < 1e-6

def test_build_population_respects_max_dislocations():
    r = fw.RECIPES["leds_eq11"]
    with pytest.raises(ValueError, match="max_dislocations"):
        fw.build_wall_population(
            r, theta_deg=0.001, extent_um=50.0, cell=CUBIC, ny=0.334,
            crystal_to_lab=np.eye(3), max_dislocations=10,
        )

def test_lines_lie_in_boundary_plane_crystal():
    # With identity placement, in-plane perpendicular offsets ⊥ n in crystal frame.
    r = fw.RECIPES["leds_eq11"]
    pop = fw.build_wall_population(
        r, theta_deg=0.05, extent_um=10.0, cell=CUBIC, ny=0.334, crystal_to_lab=np.eye(3),
    )
    n_hat = fw._unit(fw._cartesian(r.n, CUBIC))
    # every position offset is in the boundary plane (⊥ n) under identity placement
    assert np.max(np.abs(pop.positions_um @ n_hat)) < 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `".venv-wt/Scripts/python.exe" -m pytest tests/test_frank_walls.py -k "build_population or lines_lie" -q`
Expected: FAIL (`build_wall_population` not defined).

- [ ] **Step 3: Implement**

Append to `frank_walls.py`:
```python
def _edge_t(slip_plane, b) -> np.ndarray:
    return np.cross(np.asarray(slip_plane, float), np.asarray(b, float))


def _signed_angle(t0, xi, plane, cell) -> float:
    """Signed angle (deg) rotating edge t0 -> xi about the slip-plane normal.

    Sign convention pinned by the Task-1 spike / Task-7 round-trip test; negate
    here if the round-trip test requires it.
    """
    t0h = _unit(_cartesian(t0, cell))
    xih = _unit(_cartesian(xi, cell))
    nrm = _unit(_cartesian(plane, cell))
    cos = float(np.clip(t0h @ xih, -1.0, 1.0))
    sin = float(np.cross(t0h, xih) @ nrm)
    return float(np.degrees(np.arctan2(sin, cos)))


def build_wall_population(recipe, *, theta_deg, extent_um, cell, ny, crystal_to_lab, max_dislocations=None):
    from dfxm_geo.direct_space.forward_model import (  # function-local: breaks import cycle
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
    for s, rho in zip(recipe.sets, rho_hat):
        d_um = (1.0 / rho) * 1e6
        xih = _unit(_cartesian(s.xi, cell))
        u = _unit(np.cross(n_hat, xih))                         # in-plane perpendicular
        n_lines = int(np.floor(extent_um / d_um)) + 1
        ks = np.arange(n_lines) - (n_lines - 1) / 2.0
        Ud = _ud_matrix_from_bnt(s.b, s.slip_plane, _edge_t(s.slip_plane, s.b))
        rot = _signed_angle(_edge_t(s.slip_plane, s.b), s.xi, s.slip_plane, cell)
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
```

- [ ] **Step 4: Run to verify pass**

Run: `".venv-wt/Scripts/python.exe" -m pytest tests/test_frank_walls.py -q`
Expected: PASS.

- [ ] **Step 5: mypy + commit**

```bash
".venv-wt/Scripts/python.exe" -m mypy src/dfxm_geo/crystal/frank_walls.py
git add src/dfxm_geo/crystal/frank_walls.py tests/test_frank_walls.py
git commit -m "feat(gnb): build_wall_population assembles the network population"
```
Expected: mypy clean. (If `_ud_matrix_from_bnt` is mypy-private-flagged, add a local `# type: ignore[attr-defined]` only on the import line.)

---

### Task 6: Config — `GnbCrystalConfig` + union + `from_dict`

**Files:**
- Modify: `src/dfxm_geo/config.py`
- Test: `tests/test_gnb_config.py`

**Interfaces:**
- Consumes: `WallRecipe`, `RECIPES`, `DislocationSet` (from `frank_walls`).
- Produces: `GnbCrystalConfig`, `GnbCustomConfig`, `GnbSetConfig`; `CrystalConfig.mode` accepts `"gnb"`; `GnbCrystalConfig.to_recipe() -> WallRecipe`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_gnb_config.py`:
```python
import pytest
from dfxm_geo.config import CrystalConfig

def test_gnb_named_recipe_resolves():
    c = CrystalConfig.from_dict({"mode": "gnb", "gnb": {"recipe": "frankus", "theta_deg": 0.05, "extent_um": 25.0}})
    assert c.mode == "gnb"
    assert c.gnb.recipe == "frankus"
    assert c.gnb.to_recipe().name == "frankus"

def test_gnb_custom_recipe_parses_list_of_tables():
    c = CrystalConfig.from_dict({
        "mode": "gnb",
        "gnb": {
            "recipe": "custom", "theta_deg": 0.05, "extent_um": 25.0,
            "custom": {
                "n": [1, 1, 1], "a": [1, 1, 1],
                "set": [
                    {"b": [1, 0, -1], "xi": [2, -1, -1], "slip_plane": [1, 1, 1], "rel_density": 1.0},
                    {"b": [0, 1, -1], "xi": [-1, 2, -1], "slip_plane": [1, 1, 1], "rel_density": 1.0},
                ],
            },
        },
    })
    r = c.gnb.to_recipe()
    assert len(r.sets) == 2 and r.n == (1, 1, 1)

def test_gnb_custom_required_when_recipe_custom():
    with pytest.raises(ValueError, match="custom"):
        CrystalConfig.from_dict({"mode": "gnb", "gnb": {"recipe": "custom", "theta_deg": 0.05, "extent_um": 25.0}})

def test_gnb_theta_must_be_positive():
    with pytest.raises(ValueError):
        CrystalConfig.from_dict({"mode": "gnb", "gnb": {"recipe": "frankus", "theta_deg": 0.0, "extent_um": 25.0}})
```

- [ ] **Step 2: Run to verify failure**

Run: `".venv-wt/Scripts/python.exe" -m pytest tests/test_gnb_config.py -q`
Expected: FAIL (`mode "gnb"` rejected / `gnb` attr missing).

- [ ] **Step 3: Implement config dataclasses + union wiring**

In `config.py` (exact anchors): add `"gnb"` to `_CRYSTAL_MODE_NAMES` (line 276); add `"gnb"` to the `CrystalConfig.mode` `Literal` (line 288); add a `gnb: GnbCrystalConfig | None = None` field on `CrystalConfig` (after line 291) — the existing `__post_init__` sibling-exclusion (lines 294–302) then covers it automatically via `_CRYSTAL_MODE_NAMES` + `getattr`; add a `elif mode == "gnb":` arm in `CrystalConfig.from_dict` (alongside the `centered`/`wall` dispatch at lines 358–365). Add the dataclasses below.
```python
@dataclass(frozen=True)
class GnbSetConfig:
    b: tuple[int, int, int]
    xi: tuple[int, int, int]
    slip_plane: tuple[int, int, int]
    rel_density: float

@dataclass(frozen=True)
class GnbCustomConfig:
    n: tuple[int, int, int]
    a: tuple[int, int, int]
    set: tuple[GnbSetConfig, ...]

@dataclass(frozen=True)
class GnbCrystalConfig:
    recipe: str                       # leds_eq11 | leds_eq14 | frankus | custom
    theta_deg: float
    extent_um: float
    max_dislocations: int | None = None
    custom: GnbCustomConfig | None = None

    def __post_init__(self) -> None:
        if self.theta_deg <= 0:
            raise ValueError("gnb.theta_deg must be > 0")
        if self.extent_um <= 0:
            raise ValueError("gnb.extent_um must be > 0")
        valid = {"leds_eq11", "leds_eq14", "frankus", "custom"}
        if self.recipe not in valid:
            raise ValueError(f"gnb.recipe must be one of {sorted(valid)}, got {self.recipe!r}")
        if self.recipe == "custom" and self.custom is None:
            raise ValueError("gnb.recipe='custom' requires a [crystal.gnb.custom] block")
        if self.recipe != "custom" and self.custom is not None:
            raise ValueError("gnb.custom is only allowed when recipe='custom'")

    def to_recipe(self):
        from dfxm_geo.crystal.frank_walls import RECIPES, WallRecipe, DislocationSet
        if self.recipe != "custom":
            return RECIPES[self.recipe]
        c = self.custom
        assert c is not None
        return WallRecipe(
            name="custom", n=c.n, a=c.a,
            sets=tuple(
                DislocationSet(b=s.b, xi=s.xi, slip_plane=s.slip_plane, rel_density=s.rel_density)
                for s in c.set
            ),
        )
```
And in `from_dict`, parse the nested list-of-tables:
```python
# inside CrystalConfig.from_dict, in the gnb branch:
gnb_d = d["gnb"]
custom = None
if "custom" in gnb_d:
    cu = gnb_d["custom"]
    custom = GnbCustomConfig(
        n=tuple(cu["n"]), a=tuple(cu["a"]),
        set=tuple(
            GnbSetConfig(b=tuple(s["b"]), xi=tuple(s["xi"]),
                        slip_plane=tuple(s["slip_plane"]), rel_density=float(s["rel_density"]))
            for s in cu["set"]
        ),
    )
gnb = GnbCrystalConfig(
    recipe=gnb_d["recipe"], theta_deg=float(gnb_d["theta_deg"]),
    extent_um=float(gnb_d["extent_um"]),
    max_dislocations=gnb_d.get("max_dislocations"), custom=custom,
)
```
Wire `gnb` into the discriminated-union construction the same way `wall` is, and add `"gnb"` to the allowed `mode` set.

- [ ] **Step 4: Run to verify pass**

Run: `".venv-wt/Scripts/python.exe" -m pytest tests/test_gnb_config.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Regression + mypy + commit**

```bash
".venv-wt/Scripts/python.exe" -m pytest tests/test_cubic_bit_identity.py -q
".venv-wt/Scripts/python.exe" -m mypy src/dfxm_geo
git add src/dfxm_geo/config.py tests/test_gnb_config.py
git commit -m "feat(gnb): GnbCrystalConfig + custom list-of-tables parsing"
```
Expected: byte-identity gates still green; mypy clean.

---

### Task 7: Wire `gnb` into `build_dislocation_population` + frame round-trip + e2e

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py`
- Test: `tests/test_gnb_e2e.py`

**Interfaces:**
- Consumes: `build_wall_population`, `GnbCrystalConfig.to_recipe()`, the spike's `_CRYSTAL_TO_LAB`.
- Produces: a `crystal.mode == "gnb"` branch in `build_dislocation_population` returning a `DislocationPopulation`.

- [ ] **Step 1: Write failing tests (round-trip + e2e, both geometries)**

Create `tests/test_gnb_e2e.py` — kernel-free **analytic backend + oblique FCC Al mount**, mirroring `tests/test_bcc_e2e.py` (analytic avoids needing a bootstrapped MC kernel; oblique provides the mount→cell). The render entry point is `run_simulation(cfg, out_dir)` which writes HDF5; read the image back:
```python
from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm_geo.crystal import frank_walls as fw
from dfxm_geo.crystal.cell import UnitCell
from dfxm_geo.crystal.oblique import CrystalMount, compute_omega_eta
from dfxm_geo.pipeline import SimulationConfig, run_simulation

_AL_A = 4.05e-10          # Al lattice param, METRES
_HKL = (-1, 1, -1)
_KEV = 17.0
CUBIC = UnitCell.cubic(_AL_A)


def _eta(hkl=_HKL) -> float:
    """Valid oblique eta for `hkl` at 17 keV, cubic identity mount.
    If CrystalMount(...) / compute_omega_eta(...) signatures differ, copy the
    exact construction from tests/test_hcp_e2e.py (it imports both)."""
    mount = CrystalMount(
        lattice="cubic", a=_AL_A,
        mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1),
        structure_type="fcc", material="Al",
    )
    res = compute_omega_eta(mount, hkl, _KEV)  # confirm tuple order (omega, eta, theta)
    return float(res[1])


def _gnb_toml(recipe="leds_eq11", hkl=_HKL, theta_deg=0.05, extent_um=12.0) -> str:
    eta = _eta(hkl)
    return f"""
[reciprocal]
hkl = [{hkl[0]}, {hkl[1]}, {hkl[2]}]
keV = {_KEV}
backend = "analytic"
beamstop = false
[geometry]
mode = "oblique"
eta = {eta!r}
[crystal]
lattice = "cubic"
a = {_AL_A!r}
structure_type = "fcc"
material = "Al"
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
mode = "gnb"
[crystal.gnb]
recipe = "{recipe}"
theta_deg = {theta_deg}
extent_um = {extent_um}
[scan.phi]
value = 0.0
[io]
include_perfect_crystal = false
write_strain_provenance = false
[postprocess]
enabled = false
"""


def _run(base: Path, body: str) -> np.ndarray:
    base.mkdir(parents=True, exist_ok=True)
    (base / "gnb.toml").write_text(body, encoding="utf-8")
    cfg = SimulationConfig.from_toml(base / "gnb.toml")
    out = base / "out"
    run_simulation(cfg, out)
    det = out / "scan0001" / "dfxm_sim_detector_0000.h5"
    with h5py.File(det, "r") as f:
        return f["/entry_0000/dfxm_sim_detector/image"][...]


def test_gnb_forward_renders_finite(tmp_path):
    img = _run(tmp_path / "eq11", _gnb_toml("leds_eq11"))
    assert img.ndim == 3 and img.shape[0] == 1
    assert np.isfinite(img).all()
    assert float(img.std()) > 0.0          # non-degenerate

def test_frame_round_trip_positions_in_boundary_plane():
    r = fw.RECIPES["leds_eq11"]
    pop = fw.build_wall_population(
        r, theta_deg=0.05, extent_um=12.0, cell=CUBIC, ny=0.334,
        crystal_to_lab=fw._CRYSTAL_TO_LAB,
    )
    n_lab = fw._CRYSTAL_TO_LAB @ fw._unit(fw._cartesian(r.n, CUBIC))
    assert np.max(np.abs(pop.positions_um @ n_lab)) < 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `".venv-wt/Scripts/python.exe" -m pytest tests/test_gnb_e2e.py -q`
Expected: FAIL (gnb branch raises / `_CRYSTAL_TO_LAB` missing).

- [ ] **Step 3: Add `_CRYSTAL_TO_LAB` (from spike) and the gnb branch**

In `frank_walls.py`, add the module-level constant from the Task-1 findings (example placeholder — replace with the spike's exact matrix):
```python
_CRYSTAL_TO_LAB = np.eye(3)  # set from the Task-1 spike findings
```
In `forward_model.py`, inside `build_dislocation_population`, add a branch parallel to the `wall` branch (do NOT alter existing branches). `cell` comes from the mount (oblique); in simplified mode (`mount is None`) synthesize a cubic FCC cell so `gnb` still runs:
```python
    if crystal.mode == "gnb":
        from dfxm_geo.crystal.cell import UnitCell  # local
        from dfxm_geo.crystal.frank_walls import (  # local: breaks import cycle
            _CRYSTAL_TO_LAB,
            build_wall_population,
        )
        g = crystal.gnb
        assert g is not None
        if mount is not None:
            cell_g = mount.cell
            ny = mount.resolved_poisson_ratio  # if it's a method not a property, add ()
        else:
            cell_g = UnitCell.cubic(4.05e-10)  # FCC Al default (m); oblique gives exact control
            ny = POISSON_RATIO
        return build_wall_population(
            g.to_recipe(),
            theta_deg=g.theta_deg,
            extent_um=g.extent_um,
            cell=cell_g,
            ny=ny,
            crystal_to_lab=_CRYSTAL_TO_LAB,
            max_dislocations=g.max_dislocations,
        )
```

- [ ] **Step 4: Run round-trip + e2e to verify pass; fix `_signed_angle` sign if needed**

Run: `".venv-wt/Scripts/python.exe" -m pytest tests/test_gnb_e2e.py -q`
Expected: PASS. The round-trip asserts positions lie in the boundary plane (placement); if `_CRYSTAL_TO_LAB` from the spike is wrong this fails — fix the matrix. (The `rotation_deg`/character sign is exercised by the g·b + golden tests in Task 8.)

- [ ] **Step 5: Simplified-mode coverage + regression + mypy + commit**

Add a simplified-mode unit test (covers the `mount=None` cell-synthesis path without an MC kernel). Match the real `build_dislocation_population` signature (crystal first, `mount` keyword; pass any other required args per its definition):
```python
def test_gnb_simplified_builds_population_without_mount():
    from dfxm_geo.config import CrystalConfig
    from dfxm_geo.direct_space.forward_model import build_dislocation_population
    crystal = CrystalConfig.from_dict(
        {"mode": "gnb", "gnb": {"recipe": "leds_eq11", "theta_deg": 0.05, "extent_um": 12.0}}
    )
    pop = build_dislocation_population(crystal, mount=None)
    assert pop.positions_um.shape[0] > 0
    assert pop.Ud.shape[0] == pop.positions_um.shape[0]
```
Then:
```bash
".venv-wt/Scripts/python.exe" -m pytest tests/test_cubic_bit_identity.py tests/test_gnb_e2e.py -q
".venv-wt/Scripts/python.exe" -m mypy src/dfxm_geo
git add src/dfxm_geo/direct_space/forward_model.py src/dfxm_geo/crystal/frank_walls.py tests/test_gnb_e2e.py
git commit -m "feat(gnb): wire gnb branch into build_dislocation_population + e2e"
```
Expected: byte-identity green; gnb e2e green; mypy clean.

---

### Task 8: Golden snapshots + g·b invisibility sanity

**Files:**
- Create: `tests/data/golden/gnb/leds_eq11_oblique.npy`, `frankus_oblique.npy`
- Test: `tests/test_gnb_e2e.py` (append)

**Interfaces:**
- Consumes: the forward entry point + the gnb configs.

- [ ] **Step 1: Write failing golden + g·b tests**

Append to `tests/test_gnb_e2e.py` (reusing `_gnb_toml` + `_run`):
```python
import os

@pytest.mark.parametrize("recipe", ["leds_eq11", "frankus"])
def test_gnb_golden_snapshot(tmp_path, recipe):
    img = _run(tmp_path / recipe, _gnb_toml(recipe))
    golden = f"tests/data/golden/gnb/{recipe}_oblique.npy"
    if not os.path.exists(golden):
        pytest.skip(f"golden {golden} missing — generate in Step 3")
    np.testing.assert_allclose(img, np.load(golden), rtol=1e-5, atol=1e-6)

def test_gb_zero_set_drops_contrast(tmp_path):
    # leds_eq11 set1 b=[1,0,-1]: g·b1 = h - l; set2 b=[0,1,-1]: g·b2 = k - l.
    # A reflection with g·b1=0 extinguishes set1. Compare BOTH-visible vs set1-extinct.
    # (_gnb_toml recomputes eta per hkl; if an hkl isn't a valid oblique Bragg
    #  condition, swap to one that is while keeping the g·b1=0 vs ≠0 contrast.)
    img_both = _run(tmp_path / "both", _gnb_toml("leds_eq11", hkl=(2, 2, 0)))  # g·b1=2, g·b2=2
    img_ext = _run(tmp_path / "ext", _gnb_toml("leds_eq11", hkl=(0, 2, 0)))    # g·b1=0 → set1 extinct
    assert float(img_ext.std()) < float(img_both.std())
```

- [ ] **Step 2: Run to verify failure/skip**

Run: `".venv-wt/Scripts/python.exe" -m pytest tests/test_gnb_e2e.py -k "golden or gb_zero" -q`
Expected: golden tests SKIP (no file yet); `test_gb_zero` may FAIL until the reflection is chosen.

- [ ] **Step 3: Generate goldens, verify, commit**

Generate the snapshots (the `_run` helper already returns the image array — save it with `np.save`), eyeball them for a sensible crossed-net wall pattern, then save under `tests/data/golden/gnb/`. Force-add past the repo's `*.npy` gitignore (per the showcase-golden convention):
```bash
".venv-wt/Scripts/python.exe" -m pytest tests/test_gnb_e2e.py -q   # now goldens load + assert
git add -f tests/data/golden/gnb/leds_eq11_oblique.npy tests/data/golden/gnb/frankus_oblique.npy
git add tests/test_gnb_e2e.py
git commit -m "test(gnb): golden snapshots + g·b invisibility sanity"
```
Expected: all gnb tests PASS.

---

### Task 9: Docs + example configs

**Files:**
- Create: `docs/gnb-walls.md`
- Create: `src/dfxm_geo/data/configs/gnb_leds_eq11.toml`, `gnb_frankus.toml`
- Modify: `docs/crystal-structures.md` (add a pointer to the new page)

- [ ] **Step 1: Write example configs**

Create `src/dfxm_geo/data/configs/gnb_leds_eq11.toml` and `gnb_frankus.toml` with a full runnable `[reciprocal]` + `[crystal] mode="gnb"` + `[crystal.gnb]` block (theta_deg=0.05, extent_um=25.0).

- [ ] **Step 2: Write `docs/gnb-walls.md`**

Document: the recipe table (the 3 recipes with their (n, a, sets)); the θ knob + `d≈|b|/(2 sin θ/2)` and the resolvability note (θ too small ⇒ sub-pixel); `extent_um` semantics; simplified vs oblique; the Schmid–Boas sign reconciliation; the **frankus approximation + Eq. 2 vs 1:1:1 discrepancy (flagged for G. Winther)**; the "reactions excluded" scope note; a `custom` example.

- [ ] **Step 3: Verify configs load + commit**

```bash
".venv-wt/Scripts/python.exe" -c "from dfxm_geo.config import load_config; load_config('src/dfxm_geo/data/configs/gnb_frankus.toml'); print('ok')"
git add docs/gnb-walls.md docs/crystal-structures.md src/dfxm_geo/data/configs/gnb_leds_eq11.toml src/dfxm_geo/data/configs/gnb_frankus.toml
git commit -m "docs(gnb): user docs + example configs"
```

- [ ] **Step 4: Final full gate**

```bash
rm -f direct_space/deformation_gradient_tensors/Fg_*.npy 2>/dev/null
".venv-wt/Scripts/python.exe" -m pytest -q
".venv-wt/Scripts/python.exe" -m mypy src/dfxm_geo
```
Expected: whole suite green (baseline + new gnb tests); mypy clean. This is the DoD gate.

---

## Self-Review

**Spec coverage:**
- `gnb` mode + 3 recipes + custom hatch → Tasks 4, 6. ✓
- θ primary knob + θ→density solver → Task 3. ✓
- Frank residual gate (per-recipe tol) → Tasks 3, 4. ✓
- Physical-width extent + max_dislocations cap + resolvability → Tasks 5, 9. ✓
- Frame placement (lab-frame positions, Us field orientation) + spike + round-trip → Tasks 1, 7. ✓
- frankus approximate per the paper + Eq.2 discrepancy documented → Tasks 1, 4, 9. ✓
- Both geometries (simplified + oblique) → Task 7. ✓
- Byte-identity of legacy modes → regression runs in Tasks 6, 7, 9. ✓
- e2e + golden + g·b sanity → Tasks 7, 8. ✓
- Circular-import avoidance (function-local imports) → Tasks 5, 7 + Global Constraints. ✓
- Docs + example configs → Task 9. ✓

**Placeholder scan:** `_CRYSTAL_TO_LAB = np.eye(3)` is an explicit, intentional spike output (Task 1 produces the real value; Task 7 wires it; the round-trip test in Task 7 gates it) — not a hand-wave. All other steps carry real code.

**Type consistency:** `build_wall_population(recipe, *, theta_deg, extent_um, cell, ny, crystal_to_lab, max_dislocations)` is used identically in Tasks 5 and 7. `solve_density_scale(recipe, theta_deg, cell)` and `frank_residual(recipe, rho_hat, theta_deg, cell, ...)` consistent across Tasks 3–5. `GnbCrystalConfig.to_recipe()` defined in Task 6, used in Task 7. ✓

**Verified during planning (settled):** `burgers_magnitude_of(b, cell, *, fraction)` returns |b| in **µm**; `UnitCell.cubic(a)` takes **metres**; forward entry point is `run_simulation(cfg, out_dir)` writing HDF5 (image at `out/scan0001/dfxm_sim_detector_0000.h5` → `/entry_0000/dfxm_sim_detector/image`, shape (n_frames,H,W)); `cell` comes from the mount (simplified mode synthesizes a cubic FCC cell); `CrystalConfig.from_dict` exists and dispatches on `mode` (lines 325/358). The e2e render uses the analytic backend + oblique FCC mount (kernel-free, mirrors `tests/test_bcc_e2e.py`); simplified is exercised at the build level (full simplified render would need a bootstrapped MC kernel — out of scope for fast tests).

**Implementation-time confirmations (flagged inline, non-blocking):** `compute_omega_eta` tuple order + `CrystalMount` ctor signature (mirror `tests/test_hcp_e2e.py`); whether `resolved_poisson_ratio` is a property or a method; the exact `build_dislocation_population` positional args for the simplified unit test; that the g·b-test reflections are valid oblique Bragg conditions.
