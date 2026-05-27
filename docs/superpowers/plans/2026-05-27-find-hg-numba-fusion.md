# Find_Hg numba fusion (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut `Find_Hg_from_population` (the ~80% lever per cluster profiling) by fusing its NumPy chain — per-dislocation coordinate transform, edge+screw field eval, `Ud @ Fdd @ Ud.T` rotation, cross-dislocation accumulation, and the `Fg → Hg` analytic inverse — into a single `@njit` kernel with an explicit dislocations×rays double loop.

**Architecture:** A module-private `@njit(cache=True, nogil=True, fastmath=False)` kernel in `crystal/dislocations.py` computes `Hg` directly per ray, looping dislocations inside. `Find_Hg_from_population` precomputes the per-dislocation collapsed transform `M_d = Ud_dᵀ · Usᵀ · Sᵀ · Θ` (3×3) + lab offset and hands flat arrays to the kernel. The existing NumPy `Fd_find_mixed` / `Fd_find_multi_dislocs_mixed` + `fast_inverse2` composition is preserved verbatim as the parity oracle. Correctness bar: `allclose(rtol=1e-12)` vs that oracle (tolerance chosen over bit-exact during brainstorming).

**Tech Stack:** Python 3, NumPy, numba (`@njit`), pytest. Venv python: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe`.

**Spec:** `docs/superpowers/specs/2026-05-27-forward-throughput-arc-design.md` (Phase 1).

**Phase boundary:** This plan is Phase 1 only. Phase 2 (float32 HDF5 + in-node fan-out launcher) gets its own plan after Phase 1 merges and the speedup is verified standalone.

---

## Math reference (the chain being fused)

From `Fd_find_mixed` (`crystal/dislocations.py:224`) for one dislocation, rl in **micrometres**:

```
rd = Ud.T @ Us.T @ S.T @ Theta @ (rl - offset_um)        # collapses to  M @ (rl - offset),  M = Ud.T @ Us.T @ S.T @ Theta
```

Pure-field gradient `G` (= `Fdd - I`, i.e. before the `+= identity`), with `alpha = 1e-20`,
`bf = b / (4π(1-ny))`, `nyf = 2·ny·(sqx+sqy)`, `denom = (sqx+sqy)² + alpha`,
`bf1 = b / (2π)`, `denom1 = sqz+sqy + alpha`, `c = cos(rotation_deg)`, `s = sin(rotation_deg)`:

```
sqx=rd0², sqy=rd1², sqz=rd2²
G00 = -rd1*(3*sqx+sqy-nyf)/denom * bf * c
G01 =  rd0*(3*sqx+sqy-nyf)/denom * bf * c   + (-rd2/denom1) * bf1 * s
G02 =  (rd1/denom1) * bf1 * s
G10 = -rd0*(3*sqy+sqx-nyf)/denom * bf * c
G11 =  rd1*(sqx-sqy+nyf)/denom   * bf * c
# G12, G20, G21, G22 = 0
```

Per `Fd_find_multi_dislocs_mixed`: `Fg = I + Σ_d Ud_d @ G_d @ Ud_d.T`.
Per `Find_Hg_from_population`: `Hg = transpose(fast_inverse2(Fg)) - I`, so `Hg[i,j] = inv(Fg)[j,i] - (i==j)`.

**Population note:** `Find_Hg_from_population` builds every `MixedDislocSpec` with `rotation_deg=0.0` ⇒ `c=1, s=0` (edge-only, screw terms vanish). The kernel still takes `cos_rot`/`sin_rot` arrays to remain a faithful general implementation of the chain.

---

## Task 1: Profiling repro + baseline (the profile-first gate)

**Files:**
- Create: `scripts/profile_find_hg.py`
- Create: `docs/superpowers/notes/2026-05-27-find-hg-baseline.md`

This task is investigative, not TDD. It confirms the spec's hypothesis (per-ray passes + inverse dominate, dislocation loop is cheap at low ndis) and records the baseline the final speedup is measured against. **Do not write the kernel until this is recorded.**

- [ ] **Step 1: Write the profiling script**

```python
# scripts/profile_find_hg.py
"""Profile Find_Hg_from_population in isolation (Phase 1 baseline).

Runs the random_dislocations population path used by configs/profile_rocking_random.toml
(ndis=4) and dumps a cProfile breakdown + a wall-clock median over repeats.
"""
from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import CrystalConfig, RandomDislocationsConfig


def _make_population(ndis: int = 4, seed: int = 42):
    crystal = CrystalConfig(
        mode="random_dislocations",
        random_dislocations=RandomDislocationsConfig(
            ndis=ndis, sigma=5.0, min_distance=4.0, seed=seed
        ),
    )
    rng = np.random.default_rng(seed)
    return fm.build_dislocation_population(
        crystal, fov_lateral_um=abs(fm.yl_start) * 2e6, rng=rng
    )


def main() -> None:
    pop = _make_population()
    # Warm up (numba/cache, allocation pools) before timing.
    fm.Find_Hg_from_population(pop, h=-1, k=1, l=-1)

    repeats = 5
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fm.Find_Hg_from_population(pop, h=-1, k=1, l=-1)
        times.append(time.perf_counter() - t0)
    print(f"median wall: {np.median(times)*1e3:.1f} ms  (n={repeats})")

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(repeats):
        fm.Find_Hg_from_population(pop, h=-1, k=1, l=-1)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(20)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it with the venv python**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" scripts/profile_find_hg.py`
Expected: prints a median wall time and a cProfile table; the top cumulative entries should be `Fd_find_mixed` / `Fd_find_multi_dislocs_mixed` / `fast_inverse2`, confirming the per-ray NumPy passes dominate over the (4-iteration) dislocation loop.

- [ ] **Step 3: Record the baseline**

Write `docs/superpowers/notes/2026-05-27-find-hg-baseline.md` with: the median wall time, the top ~10 cProfile rows, and a one-line confirmation of which functions dominate. **State the concrete speedup target here** (e.g. "target ≥Nx so Find_Hg drops below the forward-scan cost"), derived from the measured split — this is the gate the spec deferred to "after the profile".

- [ ] **Step 4: Commit**

```bash
git add scripts/profile_find_hg.py docs/superpowers/notes/2026-05-27-find-hg-baseline.md
git commit -m "Add Find_Hg profiling repro + recorded baseline (Phase 1 gate)"
```

---

## Task 2: Preserve the current implementation as a named reference oracle

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` (around `Find_Hg_from_population`, line ~921)

Pure refactor — no behavior change. Extract the current body of `Find_Hg_from_population` into a private `_find_hg_from_population_numpy` so the parity test in Task 3 has a stable oracle to compare against, then have `Find_Hg_from_population` delegate to it (still NumPy at this point).

- [ ] **Step 1: Extract the reference helper**

Add directly above `Find_Hg_from_population`:

```python
def _find_hg_from_population_numpy(
    population: "DislocationPopulation",
    h: int,
    k: int,
    l: int,
    *,
    S: np.ndarray,
    rl: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference NumPy implementation of the population Hg field.

    Kept verbatim as the parity oracle for the fused numba kernel
    (see docs/superpowers/plans/2026-05-27-find-hg-numba-fusion.md). Composes
    Fd_find_multi_dislocs_mixed (Fg) with fast_inverse2 (Fg -> Hg). Do not
    "optimize" this; its whole purpose is to be the slow, obviously-correct
    truth the kernel is checked against at rtol=1e-12.
    """
    from dfxm_geo.crystal.dislocations import Fd_find_multi_dislocs_mixed, MixedDislocSpec

    rl_eff = rl if rl is not None else globals()["rl"]
    Q_norm = np.sqrt(h * h + k * k + l * l)
    q_hkl = np.asarray([h, k, l]) / Q_norm
    crystals = [
        MixedDislocSpec(
            Ud_mix=population.Ud[i],
            rotation_deg=0.0,
            position_lab_um=(
                float(population.positions_um[i, 0]),
                float(population.positions_um[i, 1]),
                float(population.positions_um[i, 2]),
            ),
        )
        for i in range(len(population.positions_um))
    ]
    Fg = Fd_find_multi_dislocs_mixed(rl_eff * 1e6, Us, crystals, Theta, S=S)
    Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1])
    Hg -= np.identity(3)
    return Hg, q_hkl
```

- [ ] **Step 2: Delegate from the public function**

Replace the body of `Find_Hg_from_population` (keep its signature + docstring) with:

```python
    return _find_hg_from_population_numpy(population, h, k, l, S=S, rl=rl)
```

(`fast_inverse2` is already imported at `forward_model.py:25`; no new import needed.)

- [ ] **Step 3: Run the existing parity + smoke tests**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_population_rl_units.py tests/test_dislocations_smoke.py -q`
Expected: PASS (pure refactor, identical output).

- [ ] **Step 4: Commit**

```bash
git add src/dfxm_geo/direct_space/forward_model.py
git commit -m "Extract _find_hg_from_population_numpy reference oracle (no behavior change)"
```

---

## Task 3: Write the failing parity test for the fused kernel

**Files:**
- Test: `tests/test_find_hg_kernel_parity.py`

Compare the (not-yet-written) public function path against the reference oracle at `rtol=1e-12`, for both centered (ndis=1) and multi (ndis=4) populations.

- [ ] **Step 1: Write the failing test**

```python
"""Parity: fused Find_Hg numba kernel vs the NumPy reference oracle.

Correctness bar (brainstorming decision): allclose at rtol=1e-12. Allows tiny
FMA/reassociation drift from the fused kernel; tighter than the legacy
Fd_find_smoke golden (rtol=1e-10) deliberately, since this path is the new hot
loop and we want drift flagged early.
"""
from __future__ import annotations

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    CenteredCrystalConfig,
    CrystalConfig,
    RandomDislocationsConfig,
)


def _centered_pop():
    crystal = CrystalConfig(mode="centered", centered=CenteredCrystalConfig())
    return fm.build_dislocation_population(
        crystal, fov_lateral_um=abs(fm.yl_start) * 2e6, rng=None
    )


def _random_pop(ndis: int = 4, seed: int = 42):
    crystal = CrystalConfig(
        mode="random_dislocations",
        random_dislocations=RandomDislocationsConfig(
            ndis=ndis, sigma=5.0, min_distance=4.0, seed=seed
        ),
    )
    rng = np.random.default_rng(seed)
    return fm.build_dislocation_population(
        crystal, fov_lateral_um=abs(fm.yl_start) * 2e6, rng=rng
    )


@pytest.mark.parametrize("pop_factory", [_centered_pop, _random_pop])
def test_fused_matches_numpy_oracle(pop_factory):
    pop = pop_factory()
    Hg_fast, q_fast = fm.Find_Hg_from_population(pop, h=-1, k=1, l=-1)
    Hg_ref, q_ref = fm._find_hg_from_population_numpy(
        pop, -1, 1, -1, S=fm._S_IDENTITY, rl=None
    )
    np.testing.assert_allclose(q_fast, q_ref, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(Hg_fast, Hg_ref, rtol=1e-12, atol=1e-14)
```

Note: `fm._S_IDENTITY` is the module-level identity (`forward_model.py:34`) used as the `S` default in `Find_Hg_from_population`.

- [ ] **Step 2: Run to verify it fails**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_find_hg_kernel_parity.py -q`
Expected: FAIL — at this point `Find_Hg_from_population` still delegates to the NumPy oracle, so parity is *trivially* true. To make the test meaningful it must fail until the kernel exists. Therefore: write it now but expect it to **pass trivially**; it becomes a real guard after Task 5 swaps in the kernel. (If you prefer a hard red first, temporarily point the test at a stub kernel — optional.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_find_hg_kernel_parity.py
git commit -m "Add parity test: fused Find_Hg kernel vs NumPy oracle (rtol=1e-12)"
```

---

## Task 4: Implement the fused `@njit` kernel

**Files:**
- Modify: `src/dfxm_geo/crystal/dislocations.py` (add kernel + thin NumPy wrapper)

- [ ] **Step 1: Add the kernel and wrapper**

Append to `crystal/dislocations.py` (after `Fd_find_multi_dislocs_mixed`). Use `njit` (import it alongside the existing `from numba import jit`):

```python
from numba import njit  # add to existing numba import line

import math


@njit(cache=True, nogil=True, fastmath=False)
def _population_hg_kernel(
    rl_um: np.ndarray,    # (3, X) float64, MICROMETRES
    M: np.ndarray,        # (N, 3, 3) float64 = Ud.T @ Us.T @ S.T @ Theta
    offset: np.ndarray,   # (N, 3) float64, micrometres
    Ud: np.ndarray,       # (N, 3, 3) float64
    cos_rot: np.ndarray,  # (N,) float64
    sin_rot: np.ndarray,  # (N,) float64
    b: float,
    ny: float,
    Hg_out: np.ndarray,   # (X, 3, 3) float64  (written in place)
) -> None:
    """Fused population displacement-gradient kernel: rl -> Hg, in one pass.

    Replaces the NumPy composition
        Fd_find_multi_dislocs_mixed (per-dislocation field + Ud rotation + sum)
        -> fast_inverse2 -> transpose - I.
    Looping dislocations *inside* the ray loop keeps it flat in memory whether
    N (ndis) is 4 or 200 (a NumPy broadcast would materialize (N,X,3,3)).
    fastmath=False keeps reassociation tame so parity holds at rtol=1e-12.
    See math reference in the Phase 1 plan.
    """
    X = rl_um.shape[1]
    N = M.shape[0]
    alpha = 1e-20
    bf = b / (4.0 * math.pi * (1.0 - ny))
    bf1 = b / (2.0 * math.pi)

    # Per-ray scratch reused across iterations (no per-ray heap allocation).
    G = np.zeros((3, 3))
    Tmp = np.zeros((3, 3))

    for x in range(X):
        rx = rl_um[0, x]
        ry = rl_um[1, x]
        rz = rl_um[2, x]

        # Fg accumulator = I + sum_d Ud_d @ G_d @ Ud_d.T
        f00 = 1.0; f01 = 0.0; f02 = 0.0
        f10 = 0.0; f11 = 1.0; f12 = 0.0
        f20 = 0.0; f21 = 0.0; f22 = 1.0

        for d in range(N):
            dx = rx - offset[d, 0]
            dy = ry - offset[d, 1]
            dz = rz - offset[d, 2]
            rd0 = M[d, 0, 0] * dx + M[d, 0, 1] * dy + M[d, 0, 2] * dz
            rd1 = M[d, 1, 0] * dx + M[d, 1, 1] * dy + M[d, 1, 2] * dz
            rd2 = M[d, 2, 0] * dx + M[d, 2, 1] * dy + M[d, 2, 2] * dz

            sqx = rd0 * rd0
            sqy = rd1 * rd1
            sqz = rd2 * rd2
            denom = (sqx + sqy) * (sqx + sqy) + alpha
            nyf = 2.0 * ny * (sqx + sqy)
            c = cos_rot[d]
            s = sin_rot[d]
            denom1 = sqz + sqy + alpha

            # Pure-field gradient G (= Fdd - I); only 5 entries are nonzero.
            G[0, 0] = -rd1 * (3.0 * sqx + sqy - nyf) / denom * bf * c
            G[0, 1] = rd0 * (3.0 * sqx + sqy - nyf) / denom * bf * c + (-rd2 / denom1) * bf1 * s
            G[0, 2] = (rd1 / denom1) * bf1 * s
            G[1, 0] = -rd0 * (3.0 * sqy + sqx - nyf) / denom * bf * c
            G[1, 1] = rd1 * (sqx - sqy + nyf) / denom * bf * c
            G[1, 2] = 0.0
            G[2, 0] = 0.0
            G[2, 1] = 0.0
            G[2, 2] = 0.0

            # Tmp = G @ Ud_d.T  ; Tmp[a,col] = sum_j G[a,j] * Ud[d,col,j]
            for a in range(3):
                for col in range(3):
                    acc = 0.0
                    for j in range(3):
                        acc += G[a, j] * Ud[d, col, j]
                    Tmp[a, col] = acc

            # contribution = Ud_d @ Tmp ; accumulate into Fg
            f00 += Ud[d, 0, 0] * Tmp[0, 0] + Ud[d, 0, 1] * Tmp[1, 0] + Ud[d, 0, 2] * Tmp[2, 0]
            f01 += Ud[d, 0, 0] * Tmp[0, 1] + Ud[d, 0, 1] * Tmp[1, 1] + Ud[d, 0, 2] * Tmp[2, 1]
            f02 += Ud[d, 0, 0] * Tmp[0, 2] + Ud[d, 0, 1] * Tmp[1, 2] + Ud[d, 0, 2] * Tmp[2, 2]
            f10 += Ud[d, 1, 0] * Tmp[0, 0] + Ud[d, 1, 1] * Tmp[1, 0] + Ud[d, 1, 2] * Tmp[2, 0]
            f11 += Ud[d, 1, 0] * Tmp[0, 1] + Ud[d, 1, 1] * Tmp[1, 1] + Ud[d, 1, 2] * Tmp[2, 1]
            f12 += Ud[d, 1, 0] * Tmp[0, 2] + Ud[d, 1, 1] * Tmp[1, 2] + Ud[d, 1, 2] * Tmp[2, 2]
            f20 += Ud[d, 2, 0] * Tmp[0, 0] + Ud[d, 2, 1] * Tmp[1, 0] + Ud[d, 2, 2] * Tmp[2, 0]
            f21 += Ud[d, 2, 0] * Tmp[0, 1] + Ud[d, 2, 1] * Tmp[1, 1] + Ud[d, 2, 2] * Tmp[2, 1]
            f22 += Ud[d, 2, 0] * Tmp[0, 2] + Ud[d, 2, 1] * Tmp[1, 2] + Ud[d, 2, 2] * Tmp[2, 2]

        # Analytic 3x3 inverse of Fg (mirrors fast_inverse2), then Hg = inv.T - I.
        c00 = f11 * f22 - f12 * f21
        c10 = -(f10 * f22 - f12 * f20)
        c20 = f10 * f21 - f11 * f20
        idet = 1.0 / (f00 * c00 + f01 * c10 + f02 * c20)
        i00 = c00 * idet
        i01 = -idet * (f01 * f22 - f02 * f21)
        i02 = idet * (f01 * f12 - f02 * f11)
        i10 = c10 * idet
        i11 = idet * (f00 * f22 - f02 * f20)
        i12 = -idet * (f00 * f12 - f02 * f10)
        i20 = c20 * idet
        i21 = -idet * (f00 * f21 - f01 * f20)
        i22 = idet * (f00 * f11 - f01 * f10)

        # Hg = transpose(inv) - I  =>  Hg[i,j] = inv[j,i] - (i==j)
        Hg_out[x, 0, 0] = i00 - 1.0
        Hg_out[x, 0, 1] = i10
        Hg_out[x, 0, 2] = i20
        Hg_out[x, 1, 0] = i01
        Hg_out[x, 1, 1] = i11 - 1.0
        Hg_out[x, 1, 2] = i21
        Hg_out[x, 2, 0] = i02
        Hg_out[x, 2, 1] = i12
        Hg_out[x, 2, 2] = i22 - 1.0


def find_hg_population(
    rl_um: np.ndarray,
    M: np.ndarray,
    offset: np.ndarray,
    Ud: np.ndarray,
    cos_rot: np.ndarray,
    sin_rot: np.ndarray,
    *,
    b: float = BURGERS_VECTOR,
    ny: float = POISSON_RATIO,
) -> np.ndarray:
    """NumPy-facing wrapper around ``_population_hg_kernel``.

    Allocates the (X, 3, 3) output and ensures C-contiguous float64 inputs the
    kernel expects. ``rl_um`` is the lab-frame ray grid in MICROMETRES.
    """
    X = rl_um.shape[1]
    Hg_out = np.empty((X, 3, 3), dtype=np.float64)
    _population_hg_kernel(
        np.ascontiguousarray(rl_um, dtype=np.float64),
        np.ascontiguousarray(M, dtype=np.float64),
        np.ascontiguousarray(offset, dtype=np.float64),
        np.ascontiguousarray(Ud, dtype=np.float64),
        np.ascontiguousarray(cos_rot, dtype=np.float64),
        np.ascontiguousarray(sin_rot, dtype=np.float64),
        float(b),
        float(ny),
        Hg_out,
    )
    return Hg_out
```

- [ ] **Step 2: Sanity-import to trigger numba compile**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -c "from dfxm_geo.crystal.dislocations import find_hg_population; print('ok')"`
Expected: prints `ok` (numba compiles the kernel without type errors).

- [ ] **Step 3: Commit**

```bash
git add src/dfxm_geo/crystal/dislocations.py
git commit -m "Add fused _population_hg_kernel (rl -> Hg in one njit pass)"
```

---

## Task 5: Wire `Find_Hg_from_population` to the fused kernel

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py` (`Find_Hg_from_population`)

- [ ] **Step 1: Build the precomputed per-dislocation arrays and call the kernel**

Replace the body of `Find_Hg_from_population` (currently delegating to the oracle from Task 2) with:

```python
    from dfxm_geo.crystal.dislocations import find_hg_population

    rl_eff = rl if rl is not None else globals()["rl"]
    Q_norm = np.sqrt(h * h + k * k + l * l)
    q_hkl = np.asarray([h, k, l]) / Q_norm

    n = len(population.positions_um)
    # Collapse the per-dislocation transform to M_d = Ud_d.T @ Us.T @ S.T @ Theta.
    base = Us.T @ S.T @ Theta            # (3, 3), shared across dislocations
    M = np.empty((n, 3, 3))
    Ud = np.empty((n, 3, 3))
    offset = np.empty((n, 3))
    for i in range(n):
        Ud[i] = population.Ud[i]
        M[i] = population.Ud[i].T @ base
        offset[i] = population.positions_um[i]
    # Population dislocations are pure edge (rotation_deg = 0): cos=1, sin=0.
    cos_rot = np.ones(n)
    sin_rot = np.zeros(n)

    # rl is in metres; the field formula expects micrometres (b in µm) — *1e6,
    # exactly as the NumPy path and the reference disloc_identify.py do.
    Hg = find_hg_population(rl_eff * 1e6, M, offset, Ud, cos_rot, sin_rot)
    return Hg, q_hkl
```

- [ ] **Step 2: Run the parity test — now a real guard**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_find_hg_kernel_parity.py -q`
Expected: PASS — fused kernel matches the NumPy oracle at rtol=1e-12 for both centered and ndis=4 populations.

- [ ] **Step 3: Run the pre-existing population + smoke guards**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_population_rl_units.py tests/test_dislocations_smoke.py -q`
Expected: PASS (the rl-units oracle test and the physical-scale test still hold; `Find_Hg_from_population` output unchanged within tolerance).

- [ ] **Step 4: Commit**

```bash
git add src/dfxm_geo/direct_space/forward_model.py
git commit -m "Wire Find_Hg_from_population to the fused numba kernel"
```

---

## Task 6: Full suite + mypy + golden check

**Files:** none (verification only; regenerate a golden only if a test demands it).

- [ ] **Step 1: Run the full suite**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q`
Expected: PASS at the same pass/skip/xfail counts as `main` HEAD (the lone pre-existing xfail stays xfailed). If a forward-image golden shifts within rtol=1e-12, that is acceptable per the spec — but a *bit* shift in a bit-exact golden is not; investigate before regenerating.

- [ ] **Step 2: If (and only if) a golden legitimately moved within tolerance**

Regenerate via the test's own mechanism (the smoke test self-creates when the `.npy` is absent), document the tolerance and reason in the test/commit message. Do **not** regenerate to paper over an unexplained change.

- [ ] **Step 3: mypy**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: `Success: no issues found` (0 errors). Add precise types to the new wrapper/kernel signatures if mypy flags numba-decorated functions; mirror the typing style already used around `_mc_lut_forward`.

- [ ] **Step 4: Commit any test/golden updates**

```bash
git add -A
git commit -m "Phase 1 verification: full suite green + mypy clean"
```

---

## Task 7: Re-profile, quantify the win, update notes

**Files:**
- Modify: `docs/superpowers/notes/2026-05-27-find-hg-baseline.md` (append "after" numbers)

- [ ] **Step 1: Re-run the profiler from Task 1**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" scripts/profile_find_hg.py`
Expected: median wall time substantially below the Task 1 baseline; cProfile now dominated by the kernel call, not the NumPy chain.

- [ ] **Step 2: Record before/after + speedup factor**

Append the after-median, the speedup factor, and whether the Task-1 target was met to the baseline note.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/notes/2026-05-27-find-hg-baseline.md
git commit -m "Record Find_Hg fusion speedup (Phase 1 after-numbers)"
```

- [ ] **Step 4: Finish the branch**

Use the `superpowers:finishing-a-development-branch` skill to decide merge/PR. Per CLAUDE.md: confirm with the user before pushing or opening a PR; Sina is sole reviewer so a `--no-ff` merge to `main` after local verification is the established pattern. After merge, update `cleanup_session_state.md` / write a session-handoff memory and flip CLAUDE.md's "NEXT ARC" note to point at Phase 2.

---

## Self-review notes

- **Spec coverage:** Phase 1's three requirements — profile-first gate (Task 1), numba double-loop fusion to Hg (Tasks 4–5), 1e-12 parity + regen-golden discipline (Tasks 3, 6) — each map to a task. Phase 2 is intentionally out of this plan (separate plan after merge).
- **Type consistency:** kernel name `_population_hg_kernel` and wrapper `find_hg_population` are used identically in Tasks 4 and 5; oracle `_find_hg_from_population_numpy` defined in Task 2, consumed in Task 3.
- **Known wrinkle (Task 3):** the parity test passes trivially until Task 5 swaps in the kernel, then becomes a real guard. Called out explicitly so the executor doesn't mistake the early green for a finished kernel.
- **Math fidelity:** `M_d = Ud_d.T @ (Us.T @ S.T @ Theta)` reproduces the four chained matmuls in `Fd_find_mixed`; offset subtracted in µm before `M`; `*1e6` preserved at the `Find_Hg_from_population` boundary (the v2.0.0 rl-units bug guard); `Hg[i,j]=inv[j,i]-(i==j)` reproduces `transpose(fast_inverse2(Fg)) - I`.
