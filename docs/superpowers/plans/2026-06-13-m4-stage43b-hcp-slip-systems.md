# M4 Stage 4.3b — HCP slip systems Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the dislocation displacement stack work for HCP crystals (non-orthonormal frame, ⟨a⟩ + ⟨c+a⟩ Burgers vectors of different magnitudes, non-cubic q_hkl, 4-index Miller–Bravais) while keeping the FCC and BCC (cubic) paths byte-identical to current `main` (`b29872f`).

**Architecture:** Extend the data-driven slip-system registry from Stage 4.3a with a hexagonal enumerator (operates on 4-index indices: permute the (h,k,i) triple + l-sign + overall sign, convert to reduced 3-index; the glide check `h·U+k·V+l·W==0` is metric-free so it is unchanged). Convert Miller→Cartesian via `UnitCell.A`/`.B` only on the non-cubic path (cubic stays in the integer path → byte-identical). Make the per-dislocation Burgers magnitude `b` an array through the numba Hg kernels (FCC/BCC pass a uniform array == the scalar path → byte-identical). Route q_hkl through `cell.B` once, centrally in `build_forward_context`.

**Tech Stack:** Python 3.11+, numpy, numba (the fused Hg kernels), gemmi (optional, CIF route only), pytest. Venv python: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe`.

**Branch:** `feature/m4-stage43b-hcp` off `main` (`b29872f`). No tag (v3.0.0 ships after 4.3b + 4.4). Do NOT push without Sina's nod.

**Decisions locked with Sina (2026-06-13):**
- **Default HCP family set = all 5 families** (basal {0001}⟨11-2̄0⟩, prismatic {10-1̄0}⟨11-2̄0⟩, pyramidal-⟨a⟩ {10-1̄1}⟨11-2̄0⟩, 1st-order pyramidal-⟨c+a⟩ {10-1̄1}⟨11-2̄3̄⟩, 2nd-order pyramidal-⟨c+a⟩ {11-2̄2}⟨11-2̄3̄⟩). Counts **3 + 3 + 6 + 12 + 6 = 30 systems**. `slip_families` narrows the set per run.
- **DoD reference materials = Ti AND Mg** (Ti primary, Mg second e2e for c/a coverage).

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `src/dfxm_geo/crystal/slip_systems.py` | modify | + hexagonal enumerator, 5 HCP families, 4-index↔3-index converters, per-system \|b\|, count guards. Cubic enumerator untouched. |
| `src/dfxm_geo/crystal/dislocations.py` | modify | scalar `b` → per-dislocation `b_arr` through both numba kernels + wrappers + `Fd_find_multi_dislocs_mixed` + `find_hg_scene`/`find_hg_population`. Cubic byte-identical. |
| `src/dfxm_geo/direct_space/forward_model.py` | modify | cell-aware `_ud_matrix_from_bnt_cell`; `build_forward_context(cell=...)`; population builders HCP-aware + per-dislocation \|b\|. |
| `src/dfxm_geo/orchestrator.py` | modify | identify: thread `mount.cell`, per-candidate \|b\| → `find_hg_scene(b=...)`, Cartesian Ud + g·b. |
| `src/dfxm_geo/crystal/oblique.py` | modify | guard explicit `structure_type="hcp"` against a non-hexagonal lattice. |
| `src/dfxm_geo/config.py`, `src/dfxm_geo/reciprocal_space/kernel.py` | modify | accept 4-index `slip_plane_normal` / `[[crystal.slip_system]]` (length-4 → convert). |
| `src/dfxm_geo/io/hdf5.py` | modify | HCP provenance attrs (`c_over_a`, the resolved families). |
| `tests/test_slip_systems_hcp.py` | create | registry counts, glide, 4-index conversion, per-system \|b\|. |
| `tests/test_hcp_frame.py` | create | Cartesian Ud + q_hkl unit tests; cubic byte-identity of both. |
| `tests/test_kernel_b_array.py` | create | scalar-vs-uniform-array parity + per-dislocation \|b\| kernel parity. |
| `tests/test_cubic_bit_identity.py` | create | FCC **and** BCC forward+identify determinism / byte-identity gate. |
| `tests/test_hcp_e2e.py` | create | Ti + Mg forward + identify DoD (explicit + CIF routes). |
| `docs/crystal-structures.md`, `docs/output-format.md` | modify | HCP families, 4-index notation, c/a |b|, provenance attrs. |

**Bit-identity gate (the primary regression guard, every task):** any FCC or BCC run with the SAME config as on `main` must stay byte-identical. The mechanism is always the same — branch on `cell.is_cubic` and keep the existing integer/scalar path for cubic; only the `else` (non-cubic) branch is new. The anchors are the existing FCC/BCC suites plus the new `tests/test_cubic_bit_identity.py`.

**Crystallography reference (used throughout; verify in Task 1):**
- 3-index direction from 4-index `[u v t w]` (t = −(u+v)): `U=2u+v, V=u+2v, W=w`, then divide by gcd. 3-index plane from 4-index `(h k i l)` (i = −(h+k)): drop i → `(h, k, l)`.
- HCP a-axes (⟨a⟩, reduced 3-index): `[1,0,0]`, `[0,1,0]`, `[1,1,0]` (= −a₃); |b| = a.
- HCP ⟨c+a⟩ (reduced 3-index, from ⟨11-2̄3̄⟩): the six aᵢ ± c, e.g. `[1,0,1]`, `[1,0,-1]`, …; |b| = √(a²+c²).
- Glide is metric-free: for real direction `b=[u,v,w]` and reciprocal plane `n=(h,k,l)`, `g·r = 2π(hu+kv+lw)`, so the existing integer dot `b·n==0` is correct for **all** crystal systems.
- Cartesian orthogonality holds when `b·n==0`: `(A·b)·(B·n) = bᵀAᵀB n = 2π bᵀn = 0` (since `AᵀB = 2πI`). So `b̂ = norm(A·b)`, `n̂ = norm(B·n)` are orthonormal and `t̂ = n̂×b̂` completes a proper rotation (det +1).

---

### Task 1: `slip_systems.py` — hexagonal enumerator + 5 HCP families + 4-index converters

**Files:**
- Modify: `src/dfxm_geo/crystal/slip_systems.py`
- Test: `tests/test_slip_systems_hcp.py` (create)

- [ ] **Step 1: Write the failing tests.**

```python
# tests/test_slip_systems_hcp.py
"""M4 Stage 4.3b: HCP slip-system registry (hexagonal enumerator)."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm_geo.crystal.slip_systems import (
    hkil_to_hkl,
    plane_normals,
    slip_systems,
    uvtw_to_uvw,
)

# Expected per-family system counts (Hull & Bacon; Partridge 1967).
_HCP_FAMILY_COUNTS = {
    "{0001}<11-20>": 3,
    "{10-10}<11-20>": 3,
    "{10-11}<11-20>": 6,
    "{10-11}<11-23>": 12,
    "{11-22}<11-23>": 6,
}


def test_hkil_to_hkl_drops_redundant_index():
    assert hkil_to_hkl((1, 0, -1, 1)) == (1, 0, 1)
    assert hkil_to_hkl((0, 0, 0, 1)) == (0, 0, 1)
    with pytest.raises(ValueError):
        hkil_to_hkl((1, 0, 0, 1))  # i != -(h+k)


def test_uvtw_to_uvw_reduces_by_gcd():
    # a1 = 1/3<2-1-10> -> [100]; a1+a2 = <11-20> -> [110]
    assert uvtw_to_uvw((2, -1, -1, 0)) == (1, 0, 0)
    assert uvtw_to_uvw((1, 1, -2, 0)) == (1, 1, 0)
    # <c+a> 1/3<11-23bar> -> [1,1,-1]
    assert uvtw_to_uvw((1, 1, -2, -3)) == (1, 1, -1)
    with pytest.raises(ValueError):
        uvtw_to_uvw((1, 1, 0, 0))  # t != -(u+v)


def test_hcp_total_system_count_is_30():
    assert len(slip_systems("hcp")) == 30


def test_hcp_per_family_counts():
    sysl = slip_systems("hcp")
    for fam, n in _HCP_FAMILY_COUNTS.items():
        got = sum(1 for s in sysl if s.family == fam)
        assert got == n, f"family {fam}: expected {n}, got {got}"


def test_hcp_every_system_is_glide():
    for s in slip_systems("hcp"):
        assert s.b[0] * s.n[0] + s.b[1] * s.n[1] + s.b[2] * s.n[2] == 0, s


def test_hcp_basal_systems_are_the_three_a_axes_in_0001():
    basal = [s for s in slip_systems("hcp", families=["{0001}<11-20>"]) if True]
    assert len(basal) == 3
    assert all(s.n == (0, 0, 1) for s in basal)
    bset = {tuple(sorted(abs(c) for c in s.b)) for s in basal}
    assert bset == {(0, 0, 1)}  # every <a> reduces to a permutation of [1,0,0]/[1,1,0]


def test_hcp_plane_normals_distinct_counts():
    # basal 1, prismatic 3, 1st-order pyramidal 6 (shared by <a> and <c+a>),
    # 2nd-order pyramidal 6.  Distinct planes over all families:
    planes = plane_normals("hcp")
    assert (0, 0, 1) in planes
    assert len(planes) == 1 + 3 + 6 + 6  # = 16 distinct slip-plane normals


def test_hcp_families_subset_selection():
    only_basal = slip_systems("hcp", families=["{0001}<11-20>"])
    assert len(only_basal) == 3
    with pytest.raises(ValueError):
        slip_systems("hcp", families=["{nonsense}"])


def test_fcc_bcc_unchanged_by_hcp_addition():
    # The cubic enumerator must be untouched: counts identical to 4.3a.
    assert len(slip_systems("fcc")) == 12
    assert len(slip_systems("bcc")) == 24
```

- [ ] **Step 2: Run — FAIL** (`ImportError: cannot import name 'hkil_to_hkl'`, and `slip_systems("hcp")` raises "unknown structure").

Run: `...python.exe -m pytest tests/test_slip_systems_hcp.py -q`

- [ ] **Step 3: Implement** in `src/dfxm_geo/crystal/slip_systems.py`.

3a. Update the module docstring's "CUBIC ONLY" note to "cubic (FCC/BCC) + HCP (hexagonal enumerator)".

3b. Add the 4-index converters near the top (after `_canon`):

```python
import math  # add to the imports block


def _reduce(v: tuple[int, ...]) -> tuple[int, ...]:
    """Divide an integer index vector by the gcd of its components (sign kept)."""
    g = 0
    for c in v:
        g = math.gcd(g, abs(int(c)))
    if g == 0:
        return tuple(int(c) for c in v)
    return tuple(int(c) // g for c in v)


def hkil_to_hkl(hkil: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """4-index Miller–Bravais PLANE (h k i l), i = -(h+k), -> reduced 3-index (h k l)."""
    h, k, i, l = (int(x) for x in hkil)
    if i != -(h + k):
        raise ValueError(f"Miller–Bravais plane {hkil}: i must equal -(h+k) = {-(h + k)}.")
    return cast("tuple[int, int, int]", _reduce((h, k, l)))


def uvtw_to_uvw(uvtw: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """4-index DIRECTION [u v t w], t = -(u+v), -> reduced 3-index [U V W].

    U = 2u + v, V = u + 2v, W = w, then divide by gcd. (Weber symbols.)
    """
    u, v, t, w = (int(x) for x in uvtw)
    if t != -(u + v):
        raise ValueError(f"Miller–Bravais direction {uvtw}: t must equal -(u+v) = {-(u + v)}.")
    return cast("tuple[int, int, int]", _reduce((2 * u + v, u + 2 * v, w)))
```

3c. Add the hexagonal variant generator and family enumerator. Place after `_variants`:

```python
def _variants_hex_4(rep4: tuple[int, int, int, int]) -> set[tuple[int, int, int, int]]:
    """Hexagonal-point-group orbit of a 4-index rep, in 4-index.

    In Miller–Bravais notation the 6/mmm operations act as:
      * all permutations of the first three indices (the (h,k,i)/(u,v,t) triple),
      * an independent sign flip of the 4th index (l/w) — the σ_h mirror,
      * an overall sign flip — inversion.
    |S3| · 2 · 2 = 24 = |6/mmm|.  (Permuting the triple keeps t = -(u+v) valid.)
    """
    a, b, c, d = rep4
    out: set[tuple[int, int, int, int]] = set()
    for perm in set(itertools.permutations((a, b, c))):
        for ld in (d, -d):
            for sign in (1, -1):
                out.add(
                    cast(
                        "tuple[int, int, int, int]",
                        (sign * perm[0], sign * perm[1], sign * perm[2], sign * ld),
                    )
                )
    return out


def _enumerate_hex(fam: SlipFamily) -> list[SlipSystem]:
    """Glide-system enumeration for an HCP family (4-index reps in plane_family/burgers_family).

    Generates the hexagonal orbit of the plane and Burgers reps (4-index),
    converts each to reduced 3-index, pairs plane × Burgers, keeps b·n==0
    (metric-free glide), dedups ±(b, n).  ``t`` is the integer cross n×b — a
    deterministic placeholder; the PHYSICAL line direction is recomputed in
    Cartesian by the Ud builder for the non-cubic frame, so this integer t is
    never used for HCP geometry (only for provenance symmetry with the cubic
    SlipSystem layout).
    """
    plane4 = cast("tuple[int, int, int, int]", fam.plane_family)
    burg4 = cast("tuple[int, int, int, int]", fam.burgers_family)
    planes = sorted({_canon(hkil_to_hkl(p)) for p in _variants_hex_4(plane4)})
    burgers = sorted({uvtw_to_uvw(bb) for bb in _variants_hex_4(burg4)})
    seen: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    systems: list[SlipSystem] = []
    for n in planes:
        for b in burgers:
            if b[0] * n[0] + b[1] * n[1] + b[2] * n[2] != 0:
                continue
            key = (_canon(b), n)
            if key in seen:
                continue
            seen.add(key)
            t = cast("tuple[int, int, int]", tuple(int(x) for x in np.cross(n, b)))
            systems.append(SlipSystem(b=b, n=n, t=t, family=fam.name))
    return systems
```

3d. Extend `SlipFamily` with an enumerator discriminator and widen the index tuple type. Replace the dataclass:

```python
@dataclass(frozen=True)
class SlipFamily:
    name: str
    plane_family: tuple[int, ...]   # 3-index (cubic) or 4-index (hcp) representative
    burgers_family: tuple[int, ...]
    literal: bool = False  # when True: yield exactly this (b, n) pair, no symmetry expansion
    enumerator: str = "cubic"  # "cubic" (sign+perm) | "hex" (Miller–Bravais)
```

(The cubic families keep length-3 reps and the default `enumerator="cubic"`; `_variants`, `_canon`, `_enumerate_orbit` are untouched.)

3e. Dispatch in `_enumerate_family` — add an `enumerator == "hex"` branch at the very top (before the literal/FCC checks):

```python
def _enumerate_family(fam: SlipFamily) -> list[SlipSystem]:
    if fam.enumerator == "hex":
        return _enumerate_hex(fam)
    if fam.literal:
        ...  # unchanged
```

3f. Add the HCP registry entry (after the `bcc` entry):

```python
    "hcp": (
        SlipFamily("{0001}<11-20>", (0, 0, 0, 1), (2, -1, -1, 0), enumerator="hex"),
        SlipFamily("{10-10}<11-20>", (1, 0, -1, 0), (2, -1, -1, 0), enumerator="hex"),
        SlipFamily("{10-11}<11-20>", (1, 0, -1, 1), (2, -1, -1, 0), enumerator="hex"),
        SlipFamily("{10-11}<11-23>", (1, 0, -1, 1), (2, -1, -1, -3), enumerator="hex"),
        SlipFamily("{11-22}<11-23>", (1, 1, -2, 2), (2, -1, -1, -3), enumerator="hex"),
    ),
```

3g. Add an import-time count guard (mirrors `_assert_fcc_ordered_table_complete`), placed after the registry:

```python
def _assert_hcp_family_counts() -> None:
    """Guard (run at import): the hexagonal enumerator yields the textbook counts."""
    expected = {
        "{0001}<11-20>": 3,
        "{10-10}<11-20>": 3,
        "{10-11}<11-20>": 6,
        "{10-11}<11-23>": 12,
        "{11-22}<11-23>": 6,
    }
    for fam in _REGISTRY["hcp"]:
        got = len(_enumerate_family(fam))
        if got != expected[fam.name]:
            raise AssertionError(  # pragma: no cover - defensive import-time guard
                f"HCP family {fam.name}: enumerator produced {got} systems, expected {expected[fam.name]}."
            )


_assert_hcp_family_counts()
```

3h. Clean up the now-redundant `"hcp"` special case in `derive_structure_type` — change `if structure_type not in _REGISTRY and structure_type != "hcp":` to `if structure_type not in _REGISTRY:` and drop the "(hcp lands in 4.3b)" hint from the error.

- [ ] **Step 4: Run — PASS.**

Run: `...python.exe -m pytest tests/test_slip_systems_hcp.py -q` (all green)
Then the full registry suite: `...python.exe -m pytest tests/test_slip_systems.py -q` (FCC/BCC unchanged), and `...python.exe -m mypy src/dfxm_geo/crystal/slip_systems.py` (0 errors).

- [ ] **Step 5: Commit** — `feat(crystal): HCP slip families via a hexagonal Miller–Bravais enumerator`

---

### Task 2: `slip_systems.py` — per-system Burgers magnitude (⟨a⟩ vs ⟨c+a⟩)

**Files:**
- Modify: `src/dfxm_geo/crystal/slip_systems.py`
- Test: `tests/test_slip_systems_hcp.py` (append)

HCP mixes ⟨a⟩ (|b|=a) and ⟨c+a⟩ (|b|=√(a²+c²)), so a single family-level scalar is not enough for `random_dislocations`. Add a per-system magnitude helper (the integer Burgers reduced to a full lattice translation, fraction 1.0 for HCP).

- [ ] **Step 1: Append the failing tests.**

```python
def test_burgers_magnitude_of_int_hcp_a_and_ca():
    from dfxm_geo.crystal.cell import UnitCell
    from dfxm_geo.crystal.slip_systems import burgers_magnitude_of

    # alpha-Ti: a = 2.951 A, c = 4.684 A.
    cell = UnitCell.from_lattice("hexagonal", a=2.951e-10, c=4.684e-10)
    a_um = 2.951e-10 * 1e6
    ca_um = float(np.sqrt(2.951e-10**2 + 4.684e-10**2) * 1e6)
    assert np.isclose(burgers_magnitude_of((1, 0, 0), cell, fraction=1.0), a_um, rtol=1e-12)
    assert np.isclose(burgers_magnitude_of((1, 0, 1), cell, fraction=1.0), ca_um, rtol=1e-12)


def test_hcp_family_magnitudes_via_registry():
    from dfxm_geo.crystal.cell import UnitCell
    from dfxm_geo.crystal.slip_systems import burgers_magnitude

    cell = UnitCell.from_lattice("hexagonal", a=2.951e-10, c=4.684e-10)
    a_um = 2.951e-10 * 1e6
    ca_um = float(np.sqrt(2.951e-10**2 + 4.684e-10**2) * 1e6)
    assert np.isclose(burgers_magnitude("hcp", "{0001}<11-20>", cell), a_um, rtol=1e-12)
    assert np.isclose(burgers_magnitude("hcp", "{10-11}<11-23>", cell), ca_um, rtol=1e-12)
```

- [ ] **Step 2: Run — FAIL** (`cannot import name 'burgers_magnitude_of'`; `burgers_magnitude` raises "no lattice-translation fraction" for the HCP families).

- [ ] **Step 3: Implement.**

3a. Add the per-vector helper (after `burgers_magnitude`):

```python
def burgers_magnitude_of(b_int: tuple[int, int, int], cell: UnitCell, *, fraction: float) -> float:
    """|b| in micrometres for an explicit integer Burgers direction.

    |b| = fraction * |A . b_int| (A in metres -> result in um).  HCP stores the
    REDUCED full-translation integer direction (a1 = [100], c+a = [101], ...),
    so fraction = 1.0; the centered-lattice 1/2 only applies to the cubic
    centered translations (handled by the family-level ``burgers_magnitude``).
    """
    cart = cell.A @ np.array(b_int, dtype=float)  # metres
    return float(fraction * np.linalg.norm(cart) * 1e6)
```

3b. Register the HCP family fractions (extend `_BURGERS_FRACTION` — the reduced 3-index HCP Burgers are full translations, so 1.0):

```python
_BURGERS_FRACTION: dict[str, float] = {
    "{111}<110>": 0.5,
    "{110}<111>": 0.5,
    "{112}<111>": 0.5,
    # HCP: the registry stores the reduced full-translation integer direction
    # (a1 = [100], c+a = [101]), so no centered-lattice halving.
    "{0001}<11-20>": 1.0,
    "{10-10}<11-20>": 1.0,
    "{10-11}<11-20>": 1.0,
    "{10-11}<11-23>": 1.0,
    "{11-22}<11-23>": 1.0,
}
```

3c. Make the family-level `burgers_magnitude` use the reduced 3-index of the family rep (HCP reps are 4-index). Inside `burgers_magnitude`, after `fam = matches[0]` and the fraction resolution, replace the `b_int = np.array(fam.burgers_family, ...)` tail with:

```python
    if fam.enumerator == "hex":
        b_int_3 = uvtw_to_uvw(cast("tuple[int, int, int, int]", fam.burgers_family))
    else:
        b_int_3 = cast("tuple[int, int, int]", fam.burgers_family)
    cart = cell.A @ np.array(b_int_3, dtype=float)  # metres
    return float(frac * np.linalg.norm(cart) * 1e6)
```

- [ ] **Step 4: Run — PASS.** `...python.exe -m pytest tests/test_slip_systems_hcp.py -q`; mypy clean on the module.

- [ ] **Step 5: Commit** — `feat(crystal): per-system HCP Burgers magnitude (⟨a⟩=a, ⟨c+a⟩=√(a²+c²))`

---

### Task 3: `dislocations.py` — per-dislocation Burgers magnitude through the kernels

**Files:**
- Modify: `src/dfxm_geo/crystal/dislocations.py`
- Test: `tests/test_kernel_b_array.py` (create)

This is the kernel-signature change. Make `b` accept either a scalar (cubic, byte-identical) or a per-dislocation `(N,)` array (HCP). The two numba kernels move the `bf = b/(4π(1-ny))` / `bf1 = b/(2π)` computation INSIDE the per-dislocation `d` loop. For a uniform array every `bf` equals the old single scalar `bf` → byte-identical.

- [ ] **Step 1: Write the failing parity tests.**

```python
# tests/test_kernel_b_array.py
"""M4 Stage 4.3b: per-dislocation |b| through the fused Hg kernels.

scalar b == uniform array b (byte-identical); a 2-dislocation array with two
different |b| equals two independent single-dislocation renders at those |b|.
"""

from __future__ import annotations

import numpy as np

from dfxm_geo.crystal.dislocations import (
    MixedDislocSpec,
    find_hg_population,
    find_hg_scene,
)

_RNG = np.random.default_rng(0)


def _rl(n=64):
    return _RNG.standard_normal((3, n)) * 5.0


def _rot(seed):
    # any proper rotation via QR; deterministic per seed.
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_find_hg_population_scalar_equals_uniform_array():
    rl = _rl()
    M = np.stack([_rot(1), _rot(2), _rot(3)])
    Ud = np.stack([_rot(4), _rot(5), _rot(6)])
    offset = _RNG.standard_normal((3, 3))
    cos_rot = np.array([1.0, 0.5, -0.3])
    sin_rot = np.array([0.0, 0.86, 0.95])
    b = 2.862e-4
    hg_scalar = find_hg_population(rl, M, offset, Ud, cos_rot, sin_rot, b=b, ny=0.334)
    hg_array = find_hg_population(
        rl, M, offset, Ud, cos_rot, sin_rot, b=np.full(3, b), ny=0.334
    )
    assert np.array_equal(hg_scalar, hg_array)  # EXACT, not approx


def test_find_hg_scene_perdis_b_array_matches_independent_solos():
    rl = _rl()
    Us = _rot(10)
    Theta = _rot(11)
    specs = [
        MixedDislocSpec(Ud_mix=_rot(20), rotation_deg=0.0, position_lab_um=(1.0, 0.0, 0.0)),
        MixedDislocSpec(Ud_mix=_rot(21), rotation_deg=30.0, position_lab_um=(-1.0, 2.0, 0.0)),
    ]
    b_a, b_ca = 2.951e-4, 5.54e-4  # |a| vs |c+a| for Ti
    # Per-dislocation b array.
    _, solos = find_hg_scene(
        rl, Us, specs, Theta, per_dislocation=True, b=np.array([b_a, b_ca]), ny=0.32
    )
    # Independent single-spec renders at the matching |b|.
    solo_a, _ = find_hg_scene(rl, Us, [specs[0]], Theta, b=b_a, ny=0.32)
    solo_ca, _ = find_hg_scene(rl, Us, [specs[1]], Theta, b=b_ca, ny=0.32)
    assert np.allclose(solos[0], solo_a, rtol=1e-12, atol=1e-14)
    assert np.allclose(solos[1], solo_ca, rtol=1e-12, atol=1e-14)


def test_numpy_engine_b_array_parity():
    rl = _rl()
    Us, Theta = _rot(30), _rot(31)
    specs = [
        MixedDislocSpec(Ud_mix=_rot(40), rotation_deg=0.0),
        MixedDislocSpec(Ud_mix=_rot(41), rotation_deg=15.0),
    ]
    barr = np.array([3.0e-4, 5.0e-4])
    hg_numba, _ = find_hg_scene(rl, Us, specs, Theta, b=barr, ny=0.3, engine="numba")
    hg_numpy, _ = find_hg_scene(rl, Us, specs, Theta, b=barr, ny=0.3, engine="numpy")
    assert np.allclose(hg_numba, hg_numpy, rtol=1e-12, atol=1e-14)
```

- [ ] **Step 2: Run — FAIL** (passing an array `b` raises a numba typing error / TypeError on `float(b)`).

- [ ] **Step 3: Implement** in `dislocations.py`.

3a. `_population_hg_kernel`: change the signature `b: float` → `b: np.ndarray` (a `(N,)` array). Delete the two pre-loop lines `bf = b / (4.0*math.pi*(1.0-ny))` and `bf1 = b / (2.0*math.pi)`. Inside the `for d in range(N):` loop, immediately after reading `c = cos_rot[d]` / `s = sin_rot[d]`, add:

```python
            bf = b[d] / (4.0 * math.pi * (1.0 - ny))
            bf1 = b[d] / (2.0 * math.pi)
```

3b. `_scene_perdis_hg_kernel`: identical change (signature `b: np.ndarray`; drop the two pre-loop `bf`/`bf1`; recompute `bf`/`bf1` from `b[d]` inside the `d` loop).

3c. `find_hg_population`: accept `b: float | np.ndarray` and coerce to a contiguous `(N,)` float64 array before the kernel call:

```python
    N = M.shape[0]
    b_arr = np.ascontiguousarray(np.broadcast_to(np.asarray(b, dtype=np.float64), (N,)))
    _population_hg_kernel(
        np.ascontiguousarray(rl_um, dtype=np.float64),
        ...,
        b_arr,
        float(ny),
        Hg_out,
    )
```

(Update the signature `b: float = BURGERS_VECTOR` → `b: float | np.ndarray = BURGERS_VECTOR`. `np.broadcast_to` turns a scalar into a uniform `(N,)` view; `np.ascontiguousarray` materialises it for numba.)

3d. `_find_hg_scene_perdis_numba`: signature `b: float | np.ndarray`; coerce `b_arr = np.ascontiguousarray(np.broadcast_to(np.asarray(b, np.float64), (n,)))`; pass `b_arr` to `_scene_perdis_hg_kernel`.

3e. `find_hg_scene`: change `b: float = BURGERS_VECTOR` → `b: float | np.ndarray = BURGERS_VECTOR`. In the numba `not per_dislocation` branch, `find_hg_population` already handles the broadcast — pass `b` straight through. In the `per_dislocation` branch pass `b` to `_find_hg_scene_perdis_numba`. In the numpy branch pass `b` to `_find_hg_scene_numpy`.

3f. `_find_hg_scene_numpy`: signature `b: float | np.ndarray`. The oracle calls `Fd_find_mixed` once per spec; pick each spec's scalar |b|:

```python
    b_arr = np.broadcast_to(np.asarray(b, dtype=float), (len(specs),))
    ...
    # single-spec fast path:
    Fg = Fd_find_mixed(..., b=float(b_arr[0]), ny=ny, ...)
    ...
    # multi-spec accumulation: per spec
    parts = [
        Fd_find_mixed(rl_um, Us, Ud_mix=spec.Ud_mix, rotation_deg=spec.rotation_deg,
                      Theta=Theta, b=float(b_arr[i]), ny=ny,
                      position_lab_um=spec.position_lab_um, S=S)
        for i, spec in enumerate(specs)
    ]
```

3g. `Fd_find_multi_dislocs_mixed`: change `b: float = BURGERS_VECTOR` → `b: float | np.ndarray = BURGERS_VECTOR`; broadcast and index per crystal:

```python
    b_arr = np.broadcast_to(np.asarray(b, dtype=float), (len(crystals),))
    ...
    for i, spec in enumerate(crystals):
        Fg_one = Fd_find_mixed(rl, Us, Ud_mix=spec.Ud_mix, rotation_deg=spec.rotation_deg,
                               Theta=Theta, b=float(b_arr[i]), ny=ny,
                               position_lab_um=spec.position_lab_um, S=S)
        Fg_sum += Fg_one - I
```

(`Fd_find_mixed` and `Fd_find` keep their scalar `b: float` — they model a single dislocation / single-system wall.)

3h. Update the docstrings of the two kernels and `find_hg_population`/`find_hg_scene` to note `b` is now per-dislocation (a `(N,)` array; a scalar is broadcast — byte-identical to the old scalar path).

- [ ] **Step 4: Run — PASS.** `...python.exe -m pytest tests/test_kernel_b_array.py -q`. Then the existing dislocation/kernel parity suites MUST stay green: `...python.exe -m pytest tests/test_find_hg_kernel_parity.py tests/test_hg_scene.py tests/test_dislocations_mixed.py tests/test_identify_dedup.py tests/test_identification_multi_per_dis.py tests/test_population_rl_units.py -q`. mypy clean.

- [ ] **Step 5: Commit** — `refactor(crystal): per-dislocation |b| array through the Hg kernels (scalar path byte-identical)`

---

### Task 4: `forward_model.py` — Cartesian-frame Ud builder `_ud_matrix_from_bnt_cell`

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py`
- Test: `tests/test_hcp_frame.py` (create)

`_ud_matrix_from_bnt` treats Miller as Cartesian (cubic only). Add a cell-aware builder that converts via `A`/`B` for non-cubic and DELEGATES to the legacy builder for cubic (byte-identical, using the existing integer `t`).

- [ ] **Step 1: Write the failing tests.**

```python
# tests/test_hcp_frame.py
"""M4 Stage 4.3b: Cartesian-frame Ud + q_hkl; cubic byte-identity of both."""

from __future__ import annotations

import numpy as np

from dfxm_geo.crystal.cell import UnitCell
from dfxm_geo.crystal.slip_systems import slip_systems
from dfxm_geo.direct_space.forward_model import _ud_matrix_from_bnt, _ud_matrix_from_bnt_cell


def test_ud_cell_cubic_is_byte_identical_to_legacy():
    cell = UnitCell.cubic(4.0495e-10)
    for s in slip_systems("fcc") + slip_systems("bcc"):
        legacy = _ud_matrix_from_bnt(s.b, s.n, s.t)
        viacell = _ud_matrix_from_bnt_cell(s.b, s.n, cell, t_int=s.t)
        assert np.array_equal(legacy, viacell), s  # EXACT


def test_ud_cell_hcp_is_orthonormal_rotation():
    cell = UnitCell.from_lattice("hexagonal", a=2.951e-10, c=4.684e-10)
    for s in slip_systems("hcp"):
        Ud = _ud_matrix_from_bnt_cell(s.b, s.n, cell, t_int=s.t)
        # proper rotation
        assert np.isclose(np.linalg.det(Ud), 1.0, atol=1e-10)
        assert np.allclose(Ud @ Ud.T, np.eye(3), atol=1e-10)
        # column 0 (b̂) is the Cartesian slip direction; column 1 (n̂) the plane normal
        b_hat = cell.A @ np.array(s.b, float)
        b_hat /= np.linalg.norm(b_hat)
        n_hat = cell.B @ np.array(s.n, float)
        n_hat /= np.linalg.norm(n_hat)
        assert np.allclose(Ud[:, 0], b_hat, atol=1e-10)
        assert np.allclose(Ud[:, 1], n_hat, atol=1e-10)
        assert abs(b_hat @ n_hat) < 1e-10  # glide => orthogonal in Cartesian
```

- [ ] **Step 2: Run — FAIL** (`cannot import name '_ud_matrix_from_bnt_cell'`).

- [ ] **Step 3: Implement.** Add directly below `_ud_matrix_from_bnt` (keep that function UNCHANGED):

```python
def _ud_matrix_from_bnt_cell(
    b: tuple[int, int, int],
    n: tuple[int, int, int],
    cell: "UnitCell",
    *,
    t_int: tuple[int, int, int],
) -> np.ndarray:
    """Cell-aware Ud builder: [b̂ | n̂ | t̂] with the non-orthonormal frame handled.

    Cubic cells delegate to ``_ud_matrix_from_bnt`` (treats Miller as Cartesian)
    so FCC/BCC stay byte-identical — ``t_int`` is the existing integer line
    direction (registry ``s.t`` or the centered config ``c.t``).

    Non-cubic cells convert to Cartesian first: b̂ = norm(A·b) (real direction),
    n̂ = norm(B·n) (reciprocal plane normal), t̂ = norm(n̂ × b̂) (the pure-edge
    line; b̂ ⊥ n̂ holds because the system is glide, see the cell identity
    AᵀB = 2πI). The supplied ``t_int`` is ignored for non-cubic — the physical
    line direction must be computed in the Cartesian frame.
    """
    if cell.is_cubic:
        return _ud_matrix_from_bnt(b, n, t_int)
    b_hat = cell.A @ np.array(b, dtype=np.float64)
    b_hat /= np.linalg.norm(b_hat)
    n_hat = cell.B @ np.array(n, dtype=np.float64)
    n_hat /= np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, b_hat)
    t_hat /= np.linalg.norm(t_hat)
    Ud = np.column_stack([b_hat, n_hat, t_hat])
    if np.linalg.det(Ud) < 0:
        Ud[:, 2] = -Ud[:, 2]
    return Ud
```

Add `from dfxm_geo.crystal.cell import UnitCell` to the TYPE_CHECKING imports if not already present (it is imported lazily at line ~572; add a top-level `if TYPE_CHECKING:` import for the annotation).

- [ ] **Step 4: Run — PASS.** `...python.exe -m pytest tests/test_hcp_frame.py::test_ud_cell_cubic_is_byte_identical_to_legacy tests/test_hcp_frame.py::test_ud_cell_hcp_is_orthonormal_rotation -q`. mypy clean.

- [ ] **Step 5: Commit** — `feat(forward): cell-aware Ud builder for the HCP non-orthonormal frame (cubic byte-identical)`

---

### Task 5: `forward_model.py` — non-cubic q_hkl, centralized in `build_forward_context`

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py`, `src/dfxm_geo/orchestrator.py`, `src/dfxm_geo/io/migrate.py`
- Test: `tests/test_hcp_frame.py` (append)

`forward()` (`qs = Us @ Hg @ ctx.q_hkl`) and all three identify iterators read `ctx.q_hkl`. So q_hkl is fixed in ONE place — `build_forward_context` — and threaded everywhere via the context. The four local `q/|q|` recomputations in `Find_Hg` / `Find_Hg_from_population` / `_find_hg_from_population_numpy` are replaced by `ctx.q_hkl`.

- [ ] **Step 1: Append the failing tests.**

```python
def test_build_forward_context_qhkl_cubic_byte_identical():
    # cubic: q_hkl must stay EXACTLY q/sqrt(q@q) (no cell routing).
    from dfxm_geo.direct_space.forward_model import build_forward_context
    from dfxm_geo.crystal.cell import UnitCell
    import dfxm_geo.direct_space.forward_model as fm

    # Use the module's own resolution/instrument plumbing via a tiny helper is
    # heavy; instead test the q_hkl math directly through the public seam:
    hkl = (-1, 1, -1)
    q = np.asarray(hkl, dtype=float)
    expected = q / np.sqrt(float(q @ q))
    # cell=None and a cubic cell must both reproduce `expected` exactly.
    from dfxm_geo.direct_space.forward_model import _q_hkl_unit
    assert np.array_equal(_q_hkl_unit(hkl, None), expected)
    assert np.array_equal(_q_hkl_unit(hkl, UnitCell.cubic(4.0495e-10)), expected)


def test_q_hkl_unit_hcp_routes_through_B():
    from dfxm_geo.direct_space.forward_model import _q_hkl_unit
    from dfxm_geo.crystal.cell import UnitCell

    cell = UnitCell.from_lattice("hexagonal", a=2.951e-10, c=4.684e-10)
    hkl = (1, 0, -1)
    g = cell.B @ np.array(hkl, float)
    expected = g / np.linalg.norm(g)
    assert np.allclose(_q_hkl_unit(hkl, cell), expected, atol=1e-14)
    # and it differs from the naive cubic form (proves the cell matters)
    naive = np.asarray(hkl, float) / np.linalg.norm(np.asarray(hkl, float))
    assert not np.allclose(_q_hkl_unit(hkl, cell), naive)
```

- [ ] **Step 2: Run — FAIL** (`cannot import name '_q_hkl_unit'`).

- [ ] **Step 3: Implement.**

3a. Add a single q_hkl helper near `build_forward_context`:

```python
def _q_hkl_unit(hkl: "tuple[int, int, int]", cell: "UnitCell | None") -> np.ndarray:
    """Unit reciprocal-lattice direction for ``hkl`` in the crystal Cartesian frame.

    Cubic / no cell: ``q/|q|`` (B ∝ I → byte-identical to the v2.x form).
    Non-cubic: ``norm(B·[h,k,l])`` — the metric-correct direction (HCP, M4 4.3b).
    """
    q = np.asarray(hkl, dtype=float)
    if cell is None or cell.is_cubic:
        return q / np.sqrt(float(q @ q))
    g = cell.B @ q
    return g / np.linalg.norm(g)
```

3b. `build_forward_context`: add a `cell: "UnitCell | None" = None` parameter and use the helper:

```python
def build_forward_context(
    theta_run: float,
    resolution: "ResolutionContext",
    hkl: "tuple[int, int, int]",
    instrument: "InstrumentContext | None" = None,
    omega: float = 0.0,
    *,
    cell: "UnitCell | None" = None,
) -> "ForwardContext":
    ...
    instr = instrument if instrument is not None else build_instrument_context()
    geom = build_geometry_context(theta_run, instr, omega=omega)
    q_hkl_ = _q_hkl_unit(hkl, cell)  # B_0=I (cubic) or B·G (non-cubic, M4 4.3b)
    return ForwardContext(instrument=instr, geometry=geom, resolution=resolution, q_hkl=q_hkl_)
```

3c. `Find_Hg`: delete `Q_norm = ...` / `q_hkl = np.asarray([h, k, l]) / Q_norm`; replace with `q_hkl = ctx.q_hkl`. (h, k, l stay as parameters — they drive the cache filename + the `_vars.txt` sidecar; q_hkl now comes from the context, which already carries the cell-correct value.)

3d. `Find_Hg_from_population`: same — replace the local `Q_norm`/`q_hkl` with `q_hkl = ctx.q_hkl`.

3e. `_find_hg_from_population_numpy`: same — `q_hkl = ctx.q_hkl`.

3f. Thread the resolved cell into every `build_forward_context` call. The mount/cell is available where the context is built. Edit each call site:
- `orchestrator.py` `_context_for_run` (~260): `return fm.build_forward_context(run.theta, res, run.hkl, omega=run.omega, cell=_mount_cell(config))`.
- `orchestrator.py` other sites (~379, ~572, ~1831): add `cell=_mount_cell(config)`.
- Add a tiny helper in `orchestrator.py`:

```python
def _mount_cell(config) -> "UnitCell | None":
    """The resolved UnitCell when an oblique mount is present, else None (cubic q_hkl)."""
    mount = config.geometry.mount
    return mount.cell if mount is not None else None
```

- `io/migrate.py` (~131): the migrate path builds a context for a fixed cubic `_HKL`; pass `cell=None` explicitly (or leave the default — cubic byte-identical). Add a one-line comment that migrate is cubic-only.

- [ ] **Step 4: Run — PASS.** `tests/test_hcp_frame.py` green. Then the FCC/BCC forward + identify suites (the bit-identity anchors) MUST stay green: `...python.exe -m pytest tests/test_forward_model_backlog.py tests/test_pipeline_crystal_modes.py tests/test_oblique_forward_contrast.py tests/test_identification_oblique_e2e.py -q`. mypy clean.

- [ ] **Step 5: Commit** — `feat(forward): non-cubic q_hkl via cell.B, threaded through ForwardContext (cubic byte-identical)`

---

### Task 6: `forward_model.py` — HCP-aware population builders + per-dislocation |b|

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py`
- Test: `tests/test_hcp_e2e.py` (Task 11 covers forward e2e); `tests/test_forward_model_backlog.py` (FCC anchors stay green)

Wire the HCP slip systems + per-dislocation |b| into `centered` / `wall` / `random_dislocations`. The Ud comes from `_ud_matrix_from_bnt_cell`; `random_dislocations` gets a per-dislocation |b| array (⟨a⟩ vs ⟨c+a⟩); `centered`/`wall` keep the scalar.

- [ ] **Step 1:** No new unit test here (Task 11's Ti/Mg e2e is the gate; the FCC anchors guard bit-identity). Capture the FCC baseline: `...python.exe -m pytest tests/test_pipeline_crystal_modes.py tests/test_forward_model_backlog.py -q`.

- [ ] **Step 2: Extend `DislocationPopulation`** with an optional per-dislocation |b| array (default None → uniform `b_um`, byte-identical):

```python
    # Per-dislocation Burgers magnitude (µm), HCP only (⟨a⟩ vs ⟨c+a⟩ differ).
    # None → uniform ``b_um`` for the whole population (FCC/BCC/centered/wall) →
    # byte-identical scalar path.  Shape (N,) when set.
    b_um_per: np.ndarray | None = None
```

- [ ] **Step 3: Resolve the cell once** in `_resolve_structure_systems_b`. Change its return to include the cell, OR (lower-churn) keep its signature and add a sibling that the builders use. Simplest: have the builders read `mount.cell` directly. In `build_dislocation_population`, where `structure, systems, b_um, ny = _resolve_structure_systems_b(mount)` is called, also bind:

```python
    cell = mount.cell if mount is not None else None
```

- [ ] **Step 4: `centered`** — swap the Ud builder to the cell-aware one (cubic unchanged):

```python
        c = crystal.centered
        assert c is not None
        positions = np.zeros((1, 3), dtype=np.float64)
        if cell is None or cell.is_cubic:
            Ud = _ud_matrix_from_bnt(c.b, c.n, c.t)[np.newaxis, :, :]
        else:
            Ud = _ud_matrix_from_bnt_cell(c.b, c.n, cell, t_int=c.t)[np.newaxis, :, :]
        return DislocationPopulation(positions_um=positions, Ud=Ud, sidecar=None, b_um=b_um, ny=ny)
```

(`b_um` for HCP centered is the family |b| of the resolved structure's first system — already produced by `_resolve_structure_systems_b` via `burgers_magnitude`. For an HCP centered run the user's `[crystal.centered] b` direction picks the system; resolve its |b| — see Step 6.)

- [ ] **Step 5: `wall`** — same cell-aware swap for `Ud_single` (wall is one system, scalar |b| fine):

```python
        s0 = systems[0]
        if cell is None or cell.is_cubic:
            Ud_single = _ud_matrix_from_bnt(s0.b, s0.n, s0.t)
        else:
            Ud_single = _ud_matrix_from_bnt_cell(s0.b, s0.n, cell, t_int=s0.t)
        Ud = np.broadcast_to(Ud_single, (w.ndis, 3, 3)).copy()
        # |b| for the wall's single system (HCP: that system's family magnitude).
        b_wall = b_um if (cell is None or cell.is_cubic) else burgers_magnitude_of(
            s0.b, cell, fraction=1.0
        )
        return DislocationPopulation(positions_um=positions, Ud=Ud, sidecar=None, b_um=b_wall, ny=ny)
```

- [ ] **Step 6: `random_dislocations`** — per-dislocation Ud AND |b|:

```python
        slip_indices = rng.integers(0, len(systems), size=rd.ndis)
        Ud = np.zeros((rd.ndis, 3, 3), dtype=np.float64)
        is_cubic = cell is None or cell.is_cubic
        b_per = None if is_cubic else np.empty(rd.ndis, dtype=np.float64)
        sidecar_dislocations: list[dict] = []
        for i in range(rd.ndis):
            s = systems[slip_indices[i]]
            b, n, t = s.b, s.n, s.t
            if is_cubic:
                Ud[i] = _ud_matrix_from_bnt(b, n, t)
            else:
                Ud[i] = _ud_matrix_from_bnt_cell(b, n, cell, t_int=t)
                b_per[i] = burgers_magnitude_of(b, cell, fraction=1.0)
            sidecar_dislocations.append({... "b": list(b), "n": list(n), "t": list(t),
                                         "family": s.family, ...})
        ...
        return DislocationPopulation(
            positions_um=positions, Ud=Ud, sidecar=sidecar, b_um=b_um, ny=ny, b_um_per=b_per
        )
```

(`b_um` stays the representative/provenance scalar; `b_um_per` drives the kernel for HCP. Add `"family": s.family` to the sidecar dislocation dict so the per-dislocation slip mode is recorded.)

- [ ] **Step 7: Use `b_um_per` in the Hg call.** In `Find_Hg_from_population` and `_find_hg_from_population_numpy`, pass the per-dislocation array when present:

```python
    b_eff = population.b_um_per if population.b_um_per is not None else population.b_um
    ...
    Hg = find_hg_population(rl_eff * 1e6, M, offset, Ud, cos_rot, sin_rot, b=b_eff, ny=population.ny)
```

and in the numpy oracle:

```python
    Fg = Fd_find_multi_dislocs_mixed(rl_eff * 1e6, Us_, crystals, Theta_, b=b_eff, ny=population.ny, S=S)
```

- [ ] **Step 8:** Add the imports `from dfxm_geo.crystal.slip_systems import burgers_magnitude_of` (alongside the existing `burgers_magnitude` import) in forward_model.py. For the HCP centered |b|, resolve the centered system's own |b| (Step 4) — replace the `b_um` passed to the centered population with:

```python
        b_centered = b_um if (cell is None or cell.is_cubic) else burgers_magnitude_of(
            c.b, cell, fraction=1.0
        )
```

and use `b_um=b_centered` in the centered return.

- [ ] **Step 9: Run — PASS.** FCC anchors byte-identical: `...python.exe -m pytest tests/test_pipeline_crystal_modes.py tests/test_forward_model_backlog.py tests/test_bcc_e2e.py -q`. mypy clean.

- [ ] **Step 10: Commit** — `feat(forward): HCP-aware population builders with per-dislocation |b| (⟨a⟩/⟨c+a⟩)`

---

### Task 7: `orchestrator.py` — HCP identify (per-candidate |b| + Cartesian g·b/Ud)

**Files:**
- Modify: `src/dfxm_geo/orchestrator.py`
- Test: `tests/test_hcp_e2e.py` (Task 11 identify e2e); FCC/BCC identify suites stay green

The identify sweep builds `MixedDislocSpec` lists from (plane, b_idx) candidates and calls `find_hg_scene(..., ny=_ny)` WITHOUT `b` (defaults to the FCC constant). For HCP each candidate's |b| differs (⟨a⟩ vs ⟨c+a⟩), so the per-candidate |b| must be computed and passed; the Ud and g·b must be built in the Cartesian frame.

- [ ] **Step 1:** Gate on Task 11's HCP identify e2e + the existing FCC/BCC identify tests. Capture the FCC/BCC baseline: `...python.exe -m pytest tests/test_identification_scan_modes.py tests/test_bcc_e2e.py -q`.

- [ ] **Step 2: Extend `_resolve_identify_planes_and_burgers`** to also return a per-candidate |b| resolver and surface the cell. Add to its return tuple a `burgers_mag_fn(plane, b_idx) -> float`:
- FCC / cubic: returns the existing scalar `BURGERS_VECTOR` (byte-identical — identify currently uses the default).
- non-cubic (HCP): `burgers_magnitude_of(burgers_in_plane_int(structure, plane, families=fams)[b_idx], cell, fraction=1.0)`.

```python
    if structure == "fcc":
        ...
        def _fcc_mag(plane, b_idx):
            return BURGERS_VECTOR
        return _ALL_111_PLANES, _burgers_vectors, _fcc_int, _fcc_mag
    ...
    # non-FCC
    cell = mount.cell  # mount is not None here
    is_cubic = cell.is_cubic

    def _nonfcc_mag(plane, b_idx):
        if is_cubic:
            return BURGERS_VECTOR  # BCC identify kept the v2.x default magnitude
        b_int = _burgers_in_plane_int(structure, plane, families=fams)[b_idx]
        return _burgers_magnitude_of(tuple(int(x) for x in b_int), cell, fraction=1.0)

    return planes, _nonfcc_burgers, _nonfcc_int, _nonfcc_mag
```

(Import `burgers_magnitude_of` as `_burgers_magnitude_of`. Keeping BCC at the default magnitude preserves the BCC e2e byte-for-byte; HCP is the only path that changes — the magnitude actually matters there.)

**Update ALL THREE unpack sites** of `_resolve_identify_planes_and_burgers` to take the new 4th return — `orchestrator.py:860` (single), `:1216` (multi), `:1483` (zscan):

```python
    all_planes, _burgers_fn, _burgers_int_fn, _burgers_mag_fn = _resolve_identify_planes_and_burgers(
        config.geometry.mount
    )
```

The multi/zscan iterators draw via `_draw_dislocation`; thread the resolved magnitude into its output dict (add `"b_um": _burgers_mag_fn(plane, b_idx)`) so Step 4 can read each dislocation's |b|.

- [ ] **Step 3: Build the Cartesian Ud + g·b for non-cubic.** Resolve the cell once per iterator (next to `_ny = _resolve_identify_ny(...)`):

```python
    _cell = config.geometry.mount.cell if config.geometry.mount is not None else None
    _is_cubic = _cell is None or _cell.is_cubic
```

In the plane loop, when `not _is_cubic`:
- `n_arr` (plane normal): use the Cartesian normal `n̂ = norm(cell.B @ plane)` instead of `norm(plane)`.
- the Burgers directions feeding `_rotated_t_vectors` must be Cartesian `norm(cell.A @ b_int)`. The simplest seam: have `_burgers_fn(plane)` already return Cartesian unit directions for HCP. Update the non-FCC `_nonfcc_burgers` to return Cartesian units when non-cubic:

```python
    def _nonfcc_burgers(plane):
        units_miller = _burgers_in_plane(structure, plane, families=fams)  # (m,3) Miller units +/-
        if is_cubic:
            return units_miller
        ints = _burgers_in_plane_int(structure, plane, families=fams)  # aligned integers
        cart = (cell.A @ ints.T).T
        return cart / np.linalg.norm(cart, axis=1, keepdims=True)
```

- the g·b query: `q_hkl` is already Cartesian (Task 5 made `ctx.q_hkl = norm(B·hkl)`). The Burgers fed to `_gb_cos` must also be Cartesian. Since `_nonfcc_burgers` now returns Cartesian units for HCP, `_gb_cos(q_hkl, b_table[b_idx])` is correct. **Audit:** every `_gb_cos`/`_gb_visible` call uses `b_table[b_idx]` (the `_burgers_fn` output) — confirm none use the raw integer `burgers_int` for g·b (the integer is only the HDF5 label).

- [ ] **Step 4: Pass per-candidate |b| to `find_hg_scene`.** At each of the three `find_hg_scene(...)` call sites (single ~924, multi ~1284, zscan ~1567), add `b=`:
- single: `b=_burgers_mag_fn(plane, b_idx)`.
- multi / zscan (two specs): build a 2-element array of the two candidates' |b|: `b=np.array([_mag(d1), _mag(d2)])` where each `_mag` resolves from the dislocation's (plane, b_idx). `_draw_dislocation` already records `plane` + `b_idx`; thread the resolved magnitude into its output dict (add `"b_um"`) and read it here. For FCC/BCC this is `BURGERS_VECTOR` → byte-identical (find_hg_scene's default was the same constant).

- [ ] **Step 5: HCP Ud_mix construction.** `_rotated_t_vectors` builds Ud_mix from `n_arr` + a Burgers unit + the rotation. With `n_arr` and the Burgers units now Cartesian (Steps 3), the resulting Ud_mix is the correct HCP rotation. Verify `_rotated_t_vectors` only uses the passed (already-Cartesian) vectors and does no further Miller assumption (it builds t = b×n from the inputs — Cartesian-correct once inputs are Cartesian).

- [ ] **Step 6: Run — PASS.** FCC/BCC identify byte-identical: `...python.exe -m pytest tests/test_identification_scan_modes.py tests/test_bcc_e2e.py tests/test_identification_oblique_e2e.py -q`. mypy clean.

- [ ] **Step 7: Commit** — `feat(identify): HCP per-candidate |b| + Cartesian g·b/Ud (FCC/BCC byte-identical)`

---

### Task 8: config + kernel — accept 4-index `slip_plane_normal` / `[[crystal.slip_system]]`

**Files:**
- Modify: `src/dfxm_geo/config.py`, `src/dfxm_geo/reciprocal_space/kernel.py`, `src/dfxm_geo/crystal/oblique.py`
- Test: `tests/test_crystal_structure_config.py` (append) or a new `tests/test_hcp_config.py`

Let users specify HCP planes/Burgers in 4-index (length-4 lists) and convert to 3-index internally; guard `structure_type="hcp"` against a non-hexagonal lattice.

- [ ] **Step 1: Write the failing tests** (`tests/test_hcp_config.py`):

```python
"""M4 Stage 4.3b: 4-index Miller–Bravais config acceptance + HCP lattice guard."""

from __future__ import annotations

import pytest

from dfxm_geo.config import IdentificationConfig, IdentificationCrystalConfig, GeometryConfig
from dfxm_geo.crystal.oblique import CrystalMount


def test_slip_plane_normal_accepts_4index():
    # (0001) basal in 4-index -> (0,0,1) 3-index, accepted for an HCP mount.
    mount = CrystalMount(lattice="hexagonal", a=2.951e-10, c=4.684e-10,
                         structure_type="hcp",
                         mount_x=(2, -1, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1))
    cfg = IdentificationConfig(
        crystal=IdentificationCrystalConfig(slip_plane_normal=(0, 0, 0, 1),
                                            sweep_all_slip_planes=False),
        geometry=GeometryConfig(mode="oblique", eta=0.1, mount=mount),
    )
    assert cfg.crystal.slip_plane_normal == (0, 0, 1)


def test_structure_type_hcp_requires_hexagonal_lattice():
    with pytest.raises(ValueError, match="hcp"):
        CrystalMount(lattice="cubic", a=3.0e-10, structure_type="hcp")
```

- [ ] **Step 2: Run — FAIL** (a 4-tuple `slip_plane_normal` is not converted; cubic+hcp does not raise).

- [ ] **Step 3: Implement.**

3a. In `config.py`, normalise a length-4 `slip_plane_normal` in `IdentificationCrystalConfig.__post_init__` (add one if absent) — convert via `hkil_to_hkl`:

```python
    def __post_init__(self) -> None:
        spn = self.slip_plane_normal
        if len(spn) == 4:
            from dfxm_geo.crystal.slip_systems import hkil_to_hkl
            object.__setattr__(self, "slip_plane_normal", hkil_to_hkl(tuple(int(x) for x in spn)))
        elif len(spn) != 3:
            raise ValueError(f"slip_plane_normal must be 3- or 4-index; got {spn!r}.")
```

3b. In `kernel.py` `_crystal_mount_from_toml`, the `[[crystal.slip_system]]` hatch: accept 4-index `plane`/`burgers`. In `register_custom` (slip_systems.py) or at the parse site, convert length-4 → 3-index (`hkil_to_hkl` for plane, `uvtw_to_uvw` for burgers) before validation. Do it in `register_custom`:

```python
    for i, s in enumerate(systems):
        plane_raw = tuple(int(x) for x in s["plane"])
        b_raw = tuple(int(x) for x in s["burgers"])
        plane = hkil_to_hkl(plane_raw) if len(plane_raw) == 4 else plane_raw
        b = uvtw_to_uvw(b_raw) if len(b_raw) == 4 else b_raw
        ...
```

3c. In `oblique.py` `CrystalMount.__post_init__`, guard explicit `structure_type="hcp"` against a non-hexagonal lattice (only when no space group, since the space group already governs that path):

```python
        if self.structure_type == "hcp" and self.space_group is None and self.lattice not in (
            "hexagonal", "trigonal",
        ):
            raise ValueError(
                f"structure_type='hcp' requires a hexagonal/trigonal lattice; got "
                f"lattice={self.lattice!r}. (Set lattice='hexagonal' with a and c.)"
            )
```

- [ ] **Step 4: Run — PASS.** New config tests + the existing config suites (`...python.exe -m pytest tests/test_crystal_structure_config.py tests/test_crystal_mount_from_toml_cif.py tests/test_kernel_cli_crystal_block.py -q`). mypy clean.

- [ ] **Step 5: Commit** — `feat(config): 4-index Miller–Bravais slip planes/Burgers + HCP lattice guard`

---

### Task 9: provenance — HCP attrs (c/a + resolved families)

**Files:**
- Modify: `src/dfxm_geo/io/hdf5.py` (the structure-provenance dict from Task 10 of 4.3a)
- Test: `tests/test_hcp_e2e.py` asserts the attrs (Task 11)

- [ ] **Step 1:** Add to the structure-provenance assembly (where 4.3a writes `structure_type`/`burgers_magnitude_um`/`poisson_ratio`/`material`/`space_group`): when the resolved structure is `hcp`, also emit `c_over_a = cell.c / cell.a` and `slip_families` (the resolved family-name list — `mount.slip_families or [f.name for f in _REGISTRY[structure]]`). Provenance never includes a cif path (4.2 rule).

- [ ] **Step 2:** Extend the Task 11 e2e to assert `/1.1` carries `c_over_a` (≈ 1.587 for Ti) and a non-empty `slip_families`.

- [ ] **Step 3: Run — PASS.** mypy clean.

- [ ] **Step 4: Commit** — `feat(io): HCP provenance (c/a, resolved slip families)`

---

### Task 10: cubic (FCC + BCC) bit-identity regression gate

**Files:**
- Create: `tests/test_cubic_bit_identity.py`

The single most important guard: prove FCC AND BCC forward+identify are byte-identical after the whole 4.3b change. (4.3a shipped an FCC gate; 4.3b's kernel-signature + q_hkl changes touch the BCC path too, so add BCC determinism here.)

- [ ] **Step 1: Write the test** — run each config TWICE and assert exact equality, and assert the FCC random-dislocation Ud sequence still equals the legacy `_SLIP_SYSTEM_111` order (via `slip_systems("fcc")` order, the algebraic proof from 4.3a):

```python
# tests/test_cubic_bit_identity.py
"""FCC + BCC default paths stay byte-identical after the HCP (4.3b) changes."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

# Reuse the BCC forward TOML from tests/test_bcc_e2e.py and an FCC simplified TOML.
from tests.test_bcc_e2e import _bcc_forward_toml  # noqa


def _run_forward(tmp_path, toml_text, name):
    from dfxm_geo.pipeline import SimulationConfig, run_simulation
    p = tmp_path / f"{name}.toml"
    p.write_text(toml_text, encoding="utf-8")
    out = tmp_path / name
    run_simulation(SimulationConfig.from_toml(p), out)
    det = next(out.glob("scan*/dfxm_sim_detector_0000.h5"))
    with h5py.File(det, "r") as f:
        return f["/entry_0000/dfxm_sim_detector/image"][...]


@pytest.mark.slow
def test_bcc_forward_deterministic(tmp_path):
    a = _run_forward(tmp_path / "a", _bcc_forward_toml(), "bcc")
    b = _run_forward(tmp_path / "b", _bcc_forward_toml(), "bcc")
    assert np.array_equal(a, b)


@pytest.mark.slow
def test_fcc_random_forward_deterministic(tmp_path):
    # Minimal FCC random_dislocations simplified forward, fixed seed.
    toml = (
        '[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\nbackend = "analytic"\nbeamstop = false\n\n'
        '[crystal]\nmode = "random_dislocations"\n\n'
        "[crystal.random_dislocations]\nndis = 5\nseed = 12345\nsigma = 10.0\n\n"
        "[scan.phi]\nvalue = 0.0\n\n"
        "[io]\ninclude_perfect_crystal = false\nwrite_strain_provenance = false\n\n"
        "[postprocess]\nenabled = false\n"
    )
    a = _run_forward(tmp_path / "a", toml, "fcc")
    b = _run_forward(tmp_path / "b", toml, "fcc")
    assert np.array_equal(a, b)


def test_fcc_slip_order_unchanged():
    from dfxm_geo.crystal.slip_systems import _FCC_111_110_ORDERED, slip_systems
    sysl = slip_systems("fcc")
    assert [(s.b, s.n, s.t) for s in sysl] == list(_FCC_111_110_ORDERED)
```

- [ ] **Step 2:** Run twice each; assert identical.

- [ ] **Step 3: Run — PASS.** `...python.exe -m pytest tests/test_cubic_bit_identity.py -q`.

- [ ] **Step 4: Commit** — `test: FCC+BCC byte-identity gate across the HCP changes`

---

### Task 11: HCP end-to-end DoD (Ti + Mg, forward + identify, explicit + CIF)

**Files:**
- Create: `tests/test_hcp_e2e.py`

The Stage-4.3b integration gate. Runs on the analytic backend (oblique mode, beamstop=false) so no MC kernel is bootstrapped — the same kernel-free trick the BCC e2e uses. Orthonormal HCP mount: `mount_x=(2,-1,0)`, `mount_y=(0,1,0)`, `mount_z=(0,0,1)` (their `B·m` are mutually orthogonal for a hexagonal cell). The reflection's η is computed by `compute_omega_eta` and fed back to the config (the oblique validator requires the exact η; η=0 is rejected).

- [ ] **Step 1: Write the failing tests.**

```python
# tests/test_hcp_e2e.py
"""M4 Stage 4.3b DoD: HCP forward + identify end-to-end (Ti and Mg)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm_geo.crystal.oblique import CrystalMount, compute_omega_eta
from dfxm_geo.crystal.slip_systems import plane_normals
from dfxm_geo.pipeline import (
    SimulationConfig,
    load_identification_config,
    run_identification,
    run_simulation,
)

# alpha-Ti and Mg (P6_3/mmc), lengths in metres.
_TI = dict(a=2.951e-10, c=4.684e-10, material="Ti", keV=17.0)
_MG = dict(a=3.209e-10, c=5.211e-10, material="Mg", keV=17.0)
_HKL = (0, 0, 2)  # (0002) basal reflection (paper's ⟨c+a⟩ filter)
_MOUNT_KW = dict(mount_x=(2, -1, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1))


def _eta_for(metal, hkl):
    mount = CrystalMount(lattice="hexagonal", a=metal["a"], c=metal["c"],
                         structure_type="hcp", **_MOUNT_KW)
    geom = compute_omega_eta(mount, hkl, metal["keV"])
    # pick the finite solution
    eta = geom.eta_1 if not np.isnan(geom.eta_1) else geom.eta_2
    assert not np.isnan(eta), f"({hkl}) unreachable at {metal['keV']} keV for {metal['material']}"
    return float(eta)


def _hcp_forward_toml(metal, hkl, eta):
    return (
        "[reciprocal]\n"
        f"hkl = [{hkl[0]}, {hkl[1]}, {hkl[2]}]\n"
        f"keV = {metal['keV']}\n"
        'backend = "analytic"\nbeamstop = false\n\n'
        '[geometry]\nmode = "oblique"\n'
        f"eta = {eta!r}\n\n"
        "[crystal]\n"
        'lattice = "hexagonal"\n'
        f"a = {metal['a']!r}\n"
        f"c = {metal['c']!r}\n"
        'structure_type = "hcp"\n'
        f'material = "{metal["material"]}"\n'
        "mount_x = [2, -1, 0]\nmount_y = [0, 1, 0]\nmount_z = [0, 0, 1]\n"
        'mode = "random_dislocations"\n\n'
        "[crystal.random_dislocations]\nndis = 6\nseed = 7\nsigma = 8.0\n\n"
        "[scan.phi]\nvalue = 0.0\n\n"
        "[io]\ninclude_perfect_crystal = false\nwrite_strain_provenance = false\n\n"
        "[postprocess]\nenabled = false\n"
    )


@pytest.mark.slow
@pytest.mark.parametrize("metal", [_TI, _MG], ids=["Ti", "Mg"])
def test_hcp_forward_runs(tmp_path, metal):
    eta = _eta_for(metal, _HKL)
    cfg_path = tmp_path / "fwd.toml"
    cfg_path.write_text(_hcp_forward_toml(metal, _HKL, eta), encoding="utf-8")
    cfg = SimulationConfig.from_toml(cfg_path)
    assert cfg.geometry.mount.resolved_structure_type == "hcp"
    out = tmp_path / "out"
    run_simulation(cfg, out)
    det = next(out.glob("scan*/dfxm_sim_detector_0000.h5"))
    with h5py.File(det, "r") as f:
        img = f["/entry_0000/dfxm_sim_detector/image"][...]
    assert np.isfinite(img).all() and float(img.max()) > 0.0
    # provenance: HCP structure + c/a + a mix of |b| (⟨a⟩ vs ⟨c+a⟩) in the sidecar
    with h5py.File(out / "dfxm_geo.h5", "r") as f:
        attrs = dict(f["/1.1"].attrs)
    assert attrs["structure_type"] == "hcp"
    assert np.isclose(float(attrs["c_over_a"]), metal["c"] / metal["a"], rtol=1e-9)


def _hcp_identify_toml(metal, hkl, eta):
    return (
        'mode = "single"\n\n'
        "[crystal]\n"
        "slip_plane_normal = [0, 0, 1]\n"      # basal; sweep_all covers the rest
        "angle_start_deg = 0.0\nangle_stop_deg = 0.0\nangle_step_deg = 10.0\n"
        "sweep_all_slip_planes = true\nexclude_invisibility = false\n"
        'lattice = "hexagonal"\n'
        f"a = {metal['a']!r}\n"
        f"c = {metal['c']!r}\n"
        'structure_type = "hcp"\n'
        f'material = "{metal["material"]}"\n'
        "mount_x = [2, -1, 0]\nmount_y = [0, 1, 0]\nmount_z = [0, 0, 1]\n\n"
        '[geometry]\nmode = "oblique"\n'
        f"eta = {eta!r}\n\n"
        "[reciprocal]\n"
        f"hkl = [{hkl[0]}, {hkl[1]}, {hkl[2]}]\n"
        f"keV = {metal['keV']}\n"
        f"lattice_a = {metal['a']!r}\nbeamstop = false\naperture = false\n\n"
        "[scan.phi]\nvalue = 1e-4\n"
    )


@pytest.mark.slow
@pytest.mark.parametrize("metal", [_TI, _MG], ids=["Ti", "Mg"])
def test_hcp_identify_has_ca_and_a_labels(tmp_path, metal):
    eta = _eta_for(metal, _HKL)
    cfg_path = tmp_path / "id.toml"
    cfg_path.write_text(_hcp_identify_toml(metal, _HKL, eta), encoding="utf-8")
    cfg = load_identification_config(cfg_path)
    assert cfg.geometry.mount.resolved_structure_type == "hcp"
    out = tmp_path / "out"
    run_identification(cfg, out)
    hcp_planes = {p for p in plane_normals("hcp")}
    master = out / "dfxm_identify.h5"
    with h5py.File(master, "r") as f:
        scan_ids = sorted(k for k in f if k != "dfxm_geo")
        assert scan_ids
        burgers_lens = set()
        for sid in scan_ids:
            scan = f[sid]
            assert scan.attrs["structure_type"] == "hcp"
            spn = tuple(int(round(float(c))) for c in scan["sample"]["slip_plane_normal"][()])
            from dfxm_geo.crystal.slip_systems import _canon
            assert _canon(spn) in hcp_planes
            b = scan["sample"]["burgers"][()]
            burgers_lens.add(int(round(float(np.dot(b, b)))))  # |b_int|^2 proxy: 1 (a) vs 3 (c+a)
        # both an <a> (e.g. [100], |.|^2=1) and a <c+a> (e.g. [101], |.|^2=2) appear
        assert len(burgers_lens) >= 2, f"only one Burgers class swept: {burgers_lens}"


@pytest.mark.slow
def test_hcp_via_ti_cif(tmp_path):
    pytest.importorskip("gemmi")
    cif = (
        "data_Ti\n"
        "_cell_length_a 2.951\n_cell_length_b 2.951\n_cell_length_c 4.684\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 120\n"
        "_symmetry_space_group_name_H-M 'P 63/m m c'\n"
        "_space_group_IT_number 194\n"
        "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Ti1 0.3333 0.6667 0.25\n"
    )
    (tmp_path / "ti.cif").write_text(cif, encoding="utf-8")
    eta = _eta_for(_TI, _HKL)
    toml = (
        "[reciprocal]\n"
        f"hkl = [{_HKL[0]}, {_HKL[1]}, {_HKL[2]}]\nkeV = 17.0\n"
        'backend = "analytic"\nbeamstop = false\n\n'
        '[geometry]\nmode = "oblique"\n'
        f"eta = {eta!r}\n\n"
        "[crystal]\n"
        'cif = "ti.cif"\n'   # structure derived from P6_3/mmc -> hcp
        "mount_x = [2, -1, 0]\nmount_y = [0, 1, 0]\nmount_z = [0, 0, 1]\n"
        'mode = "centered"\n\n'
        "[crystal.centered]\n"
        "b = [1, 0, 0]\nn = [0, 0, 1]\nt = [0, 1, 0]\n\n"  # basal <a>
        "[scan.phi]\nvalue = 0.0\n\n"
        "[io]\ninclude_perfect_crystal = false\nwrite_strain_provenance = false\n\n"
        "[postprocess]\nenabled = false\n"
    )
    cfg_path = tmp_path / "ti.toml"
    cfg_path.write_text(toml, encoding="utf-8")
    cfg = SimulationConfig.from_toml(cfg_path)
    assert cfg.geometry.mount.resolved_structure_type == "hcp"
    out = tmp_path / "out"
    run_simulation(cfg, out)
    det = next(out.glob("scan*/dfxm_sim_detector_0000.h5"))
    with h5py.File(det, "r") as f:
        img = f["/entry_0000/dfxm_sim_detector/image"][...]
    assert np.isfinite(img).all() and float(img.max()) > 0.0
```

- [ ] **Step 2: Run — FAIL** until all wiring lands (this task runs last among the code tasks).

- [ ] **Step 3:** No new impl — integration gate over Tasks 1–9. Fix wiring gaps surfaced here. **If `_HKL=(0,0,2)` is not reachable at 17 keV for the chosen mount, switch to a reachable HCP reflection** (use `find_reflections(mount, 17.0)` interactively to pick one with finite η; e.g. a `(1,0,-1)`-type). Document the chosen reflection in the test docstring.

- [ ] **Step 4: Run — PASS.** Full suite + slow. mypy 0.

- [ ] **Step 5: Commit** — `test: HCP forward + identify end-to-end DoD (Ti + Mg, explicit + CIF)`

---

### Task 12: docs

**Files:**
- Modify: `docs/crystal-structures.md`, `docs/output-format.md`

- [ ] **Step 1:** `docs/crystal-structures.md` — add an HCP section: the 5 families with counts (3/3/6/12/6 = 30) and 4-index names; how to select with `slip_families`; the 4-index Miller–Bravais acceptance (`slip_plane_normal`, `[[crystal.slip_system]]`); the c/a-dependent |b| (⟨a⟩=a, ⟨c+a⟩=√(a²+c²)); the orthonormal-mount requirement for hexagonal cells (default `(100)/(010)/(001)` is NOT orthogonal — use e.g. `(2,-1,0)/(0,1,0)/(0,0,1)`); and the prominent **isotropic-elasticity-only** caveat (unchanged from 4.3a). Remove the "HCP coming in 4.3b" note.
- [ ] **Step 2:** `docs/output-format.md` — document the new `/1.1` HCP provenance attrs (`c_over_a`, `slip_families`).
- [ ] **Step 3:** Full suite + mypy green; `...python.exe -m compileall src/dfxm_geo` clean.
- [ ] **Step 4: Commit** — `docs: HCP slip systems, 4-index notation, orthonormal-mount note`

---

### Task 13: final gates

- [ ] `...python.exe -m pytest -q` — full suite green. Compare the failure SET to the pre-existing baseline (the lone xfail is the hdf5 bit-equiv pair; 1022-region default count at the 4.3a merge + the new HCP/kernel/bit-identity tests).
- [ ] `...python.exe -m pytest -q -m slow` — slow green (HCP e2e Ti+Mg, FCC+BCC bit-identity).
- [ ] `...python.exe -m mypy src/dfxm_geo/` — 0 errors (was 0/43 at the 4.3a merge).
- [ ] `grep -rn "B_0 = I\|B_0=I" src/dfxm_geo/` — the 4 q_hkl sites are gone (only `_q_hkl_unit` remains; cubic branch documented).
- [ ] Sanity: an HCP `random_dislocations` forward sidecar records a MIX of `family` values (⟨a⟩ and ⟨c+a⟩) and `b_um_per` is set; an FCC run has `b_um_per is None`.
- [ ] Review `git log --oneline main..HEAD`; update CLAUDE.md + auto-memory; hand off via `superpowers:finishing-a-development-branch` (no tag; no push without Sina's nod). Note the open 4.3a follow-up #4 (consolidate the two FCC orderings) — re-examine now that q_hkl/Burgers went non-cubic, but do NOT merge the orderings without re-proving both FCC byte-identity gates.

---

## Self-review notes (for the executor)

- **Spec coverage (spec §4):** (1) non-orthonormal frame → Task 4 (Ud) + Task 5 (q_hkl); (2) per-dislocation |b| → Task 3 (kernels) + Task 6 (forward) + Task 7 (identify); (3) non-cubic q_hkl → Task 5; (4) 4-index Miller–Bravais → Task 1 (converters/enumerator) + Task 8 (config); (5) c/a |b| → Task 2. DoD (Ti/Mg forward+identify, FCC/BCC byte-identical) → Tasks 10–11.
- **The bit-identity risk is everywhere cubic touches a generalized seam.** Mitigation pattern is uniform: branch on `cell.is_cubic` and keep the legacy integer/scalar path. The proofs are: kernel scalar==uniform-array (Task 3, `np.array_equal`), `_ud_matrix_from_bnt_cell` cubic==legacy for every registry system (Task 4, `np.array_equal`), `_q_hkl_unit` cubic==`q/|q|` (Task 5, `np.array_equal`), and the FCC+BCC forward determinism gate (Task 10). All four use EXACT equality, not tolerances.
- **The numba kernels recompile** when `b: float` → `b: np.ndarray` (signature change). `cache=True` rebuilds the on-disk cache on first run; the parity tests (Task 3) trigger the rebuild. No action needed beyond running the tests.
- **Identify never passed `b` before** (defaulted to `BURGERS_VECTOR`), so BCC identify implicitly used the FCC |b|. Task 7 keeps BCC at that default magnitude (byte-identical) and only changes HCP, where the |b| genuinely differs by family. If a future task wants the physically-correct BCC identify |b|, that is a separate, intentionally-non-byte-identical change.
- **Orthonormal HCP mount:** the cubic default `(100)/(010)/(001)` mount is REJECTED by `CrystalMount.__post_init__` for a hexagonal cell (`a*`,`b*` subtend 60°). Every HCP test/config must use an orthogonal mount, e.g. `(2,-1,0)/(0,1,0)/(0,0,1)`. This is called out in Tasks 11–12.
- **`SlipSystem.t` is a placeholder for HCP** (integer `n×b`), never used for HCP geometry — the Cartesian line direction is `n̂×b̂`, computed in `_ud_matrix_from_bnt_cell`. Do not "use" the integer t in any HCP Ud path.
- **Reflection reachability (Task 11):** if `(0,0,2)` is not Laue-reachable at 17 keV for the chosen mount, `compute_omega_eta` returns NaN η — switch to a reachable reflection via `find_reflections`. The test asserts non-NaN η before running, so this fails loudly, not silently.
- **4.3a landmines preserved:** the two FCC orderings (`_FCC_111_110_ORDERED` vs identify's `_ALL_111_PLANES`/`_ORDERED_BASES`), the FCC |b| constant `BURGERS_VECTOR` (not cell-derived), and the FCC √2 integer reconstruction all stay untouched on the FCC path. HCP adds NEW branches; it never edits the FCC ones.
