# M4 Stage 4.3a — BCC slip systems Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three FCC-hardcoded slip tables with a data-driven structure-family registry so BCC crystals run forward + identify end-to-end with correct slip systems, Burgers magnitude, and Poisson ratio — while the FCC default path stays byte-identical to current main.

**Architecture:** A new pure `crystal/slip_systems.py` registry (FCC `{111}⟨110⟩`, BCC `{110}⟨111⟩`+`{112}⟨111⟩`, user-defined) plus `crystal/elasticity.py` (ν table). Structure type is derived from explicit `[crystal] structure_type`, else space-group centering, else fcc. BCC is cubic, so Miller=Cartesian: the displacement/numba kernels and `q/|q|` are untouched; only the table sources change. Spec: `docs/superpowers/specs/2026-06-13-m4-stage43-slip-systems-design.md`.

**Tech Stack:** numpy, frozen dataclasses, gemmi (optional, behind `crystal/cif.py`), pytest, mypy. venv python `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe`, cwd `C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master`. Branch: `feature/m4-stage43-slip-systems` (already created off main `8dbd2fe`; spec committed `02e4d61`).

**THE GATE:** after every task, the FCC default path (no `structure_type`/`space_group`/`cif`) must stay byte-identical to main. The existing FCC tests (`test_forward_model_backlog`, `test_burgers`, `test_constants`, `test_pipeline_crystal_modes`, `test_dislocations*`) are the anchors and must stay green unmodified except where a test's own import path moves.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/dfxm_geo/crystal/slip_systems.py` | Create | `SlipFamily`/`SlipSystem`, registry, enumerator, `slip_systems`/`plane_normals`/`burgers_in_plane`/`burgers_magnitude`/`derive_structure_type` |
| `src/dfxm_geo/crystal/elasticity.py` | Create | `_POISSON_TABLE` (cited), `poisson_ratio()` |
| `src/dfxm_geo/crystal/cif.py` | Modify | `space_group_structure_family()` helper (centering+system → fcc/bcc/hcp) behind the gemmi boundary |
| `src/dfxm_geo/crystal/burgers.py` | Modify | `burgers_in_plane`/`plane_normals` delegate to registry; `burgers_vectors` becomes FCC-compat shim; drop the literal `/√2` |
| `src/dfxm_geo/crystal/oblique.py` | Modify | `CrystalMount` gains `structure_type`/`material`/`poisson_ratio`/`slip_families`/`slip_systems_custom`; `__post_init__` validation + `resolved_structure_type`/`resolved_poisson_ratio` properties |
| `src/dfxm_geo/config.py` | Modify | `_CRYSTAL_MOUNT_KEYS` + identify `{111}` gate generalization |
| `src/dfxm_geo/reciprocal_space/kernel.py` | Modify | `_crystal_mount_from_toml` parses the new keys + `[[crystal.slip_system]]` |
| `src/dfxm_geo/direct_space/forward_model.py` | Modify | `_SLIP_SYSTEM_111` + wall system → registry; `|b|` from `burgers_magnitude` |
| `src/dfxm_geo/orchestrator.py` | Modify | `_ALL_111_PLANES` → `plane_normals`; `burgers_vectors` → `burgers_in_plane`; `√2` → registry integer Burgers |
| `src/dfxm_geo/io/hdf5.py` | Modify | structure provenance attrs |
| `tests/test_slip_systems.py` | Create | registry/enumerator/derivation/magnitude unit tests |
| `tests/test_elasticity.py` | Create | ν precedence + table tests |
| `tests/test_bcc_e2e.py` | Create | BCC forward + identify DoD (small grid) |
| `tests/test_fcc_bit_identity.py` | Create | FCC default == pre-branch golden |
| `docs/crystal-structures.md` | Create | structure_type/slip families/ν table + citations + isotropic-elasticity caveat |

---

### Task 1: `slip_systems.py` — dataclasses + registry + enumerator

**Files:**
- Create: `src/dfxm_geo/crystal/slip_systems.py`
- Test: `tests/test_slip_systems.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Slip-system registry + enumerator tests."""

import numpy as np
import pytest

from dfxm_geo.crystal.slip_systems import (
    plane_normals,
    slip_systems,
)


def _canon(v):
    """Canonical-sign integer direction (first nonzero positive) for set compare."""
    v = tuple(int(round(x)) for x in v)
    for c in v:
        if c != 0:
            return v if c > 0 else tuple(-x for x in v)
    return v


def test_fcc_counts_and_geometry():
    sys = slip_systems("fcc")
    assert len(sys) == 12
    assert len({_canon(s.n) for s in sys}) == 4  # 4 distinct {111} planes
    for s in sys:
        assert np.dot(s.b, s.n) == 0  # glide: Burgers in plane
        assert np.allclose(np.cross(s.n, s.b), s.t)  # t = n x b


def test_bcc_default_is_110_plus_112():
    sys = slip_systems("bcc")
    assert len(sys) == 24  # {110}<111> 12 + {112}<111> 12
    assert len({_canon(s.n) for s in sys}) == 18  # 6 {110} + 12 {112}
    for s in sys:
        assert np.dot(s.b, s.n) == 0


def test_bcc_single_family_selection():
    assert len(slip_systems("bcc", families=["{110}<111>"])) == 12
    assert len(slip_systems("bcc", families=["{112}<111>"])) == 12


def test_fcc_registry_equals_legacy_slip_table():
    """The enumerator must reproduce the hand-written _SLIP_SYSTEM_111 set
    (up to sign) BEFORE that table is deleted (Task 8)."""
    from dfxm_geo.direct_space.forward_model import _SLIP_SYSTEM_111

    legacy = {(_canon(b), _canon(n)) for b, n, _t in _SLIP_SYSTEM_111}
    reg = {(_canon(s.b), _canon(s.n)) for s in slip_systems("fcc")}
    assert reg == legacy


def test_plane_normals_distinct():
    assert len(plane_normals("fcc")) == 4
    assert len(plane_normals("bcc")) == 18


def test_unknown_structure_raises():
    with pytest.raises(ValueError, match="unknown structure"):
        slip_systems("diamond")
```

- [ ] **Step 2: Run — FAIL** (`ModuleNotFoundError`).

Run: `...python.exe -m pytest tests/test_slip_systems.py -q`

- [ ] **Step 3: Implement `crystal/slip_systems.py`**

```python
"""Data-driven slip-system registry for FCC / BCC (+ user-defined).

A slip system is (b, n, t): Burgers direction b in the glide plane with
normal n, and line direction t = n x b. Tables are generated by enumerating
the sign/permutation variants of a representative plane and Burgers family
under the cubic point group, filtering to b.n == 0 (glide), deduplicating
+/-(b, n) pairs.

CUBIC ONLY in 4.3a: Miller indices are treated as Cartesian (valid for
cubic — FCC and BCC). HCP (4.3b) converts via the cell matrix first.
Spec: docs/superpowers/specs/2026-06-13-m4-stage43-slip-systems-design.md.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from dfxm_geo.crystal.cell import UnitCell


@dataclass(frozen=True)
class SlipFamily:
    name: str
    plane_family: tuple[int, int, int]
    burgers_family: tuple[int, int, int]


@dataclass(frozen=True)
class SlipSystem:
    b: tuple[int, int, int]  # integer Burgers direction (Miller)
    n: tuple[int, int, int]  # integer plane normal (Miller)
    t: tuple[int, int, int]  # integer line direction n x b
    family: str


_REGISTRY: dict[str, tuple[SlipFamily, ...]] = {
    "fcc": (SlipFamily("{111}<110>", (1, 1, 1), (1, 1, 0)),),
    "bcc": (
        SlipFamily("{110}<111>", (1, 1, 0), (1, 1, 1)),
        SlipFamily("{112}<111>", (1, 1, 2), (1, 1, 1)),
    ),
}


def _variants(rep: tuple[int, int, int]) -> set[tuple[int, int, int]]:
    """All distinct sign+permutation variants of a Miller family rep."""
    out: set[tuple[int, int, int]] = set()
    for perm in set(itertools.permutations(rep)):
        for signs in itertools.product((1, -1), repeat=3):
            out.add(tuple(s * c for s, c in zip(signs, perm)))
    return out


def _canon(v: tuple[int, int, int]) -> tuple[int, int, int]:
    """Canonical sign: leading nonzero positive (so v and -v collapse)."""
    for c in v:
        if c != 0:
            return v if c > 0 else tuple(-x for x in v)  # type: ignore[return-value]
    return v


def _enumerate_family(fam: SlipFamily) -> list[SlipSystem]:
    planes = {_canon(p) for p in _variants(fam.plane_family)}
    burgers = _variants(fam.burgers_family)
    seen: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    systems: list[SlipSystem] = []
    for n in sorted(planes):
        for b in sorted(burgers):
            if b[0] * n[0] + b[1] * n[1] + b[2] * n[2] != 0:
                continue  # not a glide system
            key = (_canon(b), n)
            if key in seen:
                continue
            seen.add(key)
            t = tuple(int(x) for x in np.cross(n, b))
            systems.append(SlipSystem(b=b, n=n, t=t, family=fam.name))
    return systems


def _families_for(structure: str, families: "list[str] | None") -> tuple[SlipFamily, ...]:
    if structure not in _REGISTRY:
        raise ValueError(
            f"unknown structure {structure!r}; expected one of {sorted(_REGISTRY)}"
        )
    avail = _REGISTRY[structure]
    if families is None:
        return avail
    by_name = {f.name: f for f in avail}
    try:
        return tuple(by_name[name] for name in families)
    except KeyError as exc:
        raise ValueError(
            f"slip family {exc.args[0]!r} not defined for {structure!r}; "
            f"available: {sorted(by_name)}"
        ) from None


def slip_systems(
    structure: str, *, families: "list[str] | None" = None
) -> list[SlipSystem]:
    """All (b, n, t) slip systems for a structure (optionally a family subset)."""
    out: list[SlipSystem] = []
    for fam in _families_for(structure, families):
        out.extend(_enumerate_family(fam))
    return out


def plane_normals(
    structure: str, *, families: "list[str] | None" = None
) -> list[tuple[int, int, int]]:
    """Distinct slip-plane normals (replaces _ALL_111_PLANES), canonical sign."""
    seen: list[tuple[int, int, int]] = []
    for s in slip_systems(structure, families=families):
        c = _canon(s.n)
        if c not in seen:
            seen.append(c)
    return seen
```

- [ ] **Step 4: Run — PASS.** `mypy src/dfxm_geo/crystal/slip_systems.py` → 0.
  If `test_fcc_registry_equals_legacy_slip_table` fails, the enumerator
  disagrees with the hand table — FIX the enumerator, do NOT edit the assert.

- [ ] **Step 5: Commit** — `feat(crystal): data-driven slip-system registry (fcc + bcc enumerator)`

---

### Task 2: `burgers_in_plane` + `burgers_magnitude` in `slip_systems.py`

**Files:**
- Modify: `src/dfxm_geo/crystal/slip_systems.py`
- Test: `tests/test_slip_systems.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
from dfxm_geo.crystal.cell import UnitCell
from dfxm_geo.crystal.slip_systems import burgers_in_plane, burgers_magnitude


def test_burgers_in_plane_fcc_matches_legacy_basis():
    """For each {111} plane, the 6 unit Burgers match crystal.burgers
    (bit-identical to the pre-branch _BASIS_TABLE / √2)."""
    from dfxm_geo.crystal.burgers import _BASIS_TABLE  # still present pre-Task 5

    for slug, basis in _BASIS_TABLE.items():
        plane = tuple(  # slug like "1-11" -> (1,-1,1)
            int(x) for x in slug.replace("-", " -").split()
        )
        got = burgers_in_plane("fcc", plane)
        want = np.vstack([basis, -basis]) / np.sqrt(2)
        # same set of unit vectors, order-independent
        gs = sorted(tuple(np.round(v, 9)) for v in got)
        ws = sorted(tuple(np.round(v, 9)) for v in want)
        assert np.allclose(gs, ws)


def test_burgers_in_plane_bcc_count():
    # each {110} plane holds 2 <111> Burgers -> 4 with negatives;
    # each {112} plane holds 1 <111> -> 2 with negatives.
    got110 = burgers_in_plane("bcc", (1, 1, 0))
    assert got110.shape[0] in (4, 6)  # 2 distinct <111> (+/-) [+ any {112} sharing]
    for v in got110:
        assert np.isclose(np.linalg.norm(v), 1.0)


def test_burgers_magnitude_fcc_al_exact():
    al = UnitCell.cubic(4.0495e-10)
    # FCC 1/2<110> = a/sqrt(2); in µm
    assert burgers_magnitude("fcc", "{111}<110>", al) == pytest.approx(2.862e-4, rel=1e-3)


def test_burgers_magnitude_bcc_fe():
    fe = UnitCell.cubic(2.8665e-10)
    # BCC 1/2<111> = a*sqrt(3)/2; in µm
    expected_um = 2.8665e-10 * np.sqrt(3) / 2 * 1e6
    assert burgers_magnitude("bcc", "{110}<111>", fe) == pytest.approx(expected_um, rel=1e-9)
```

- [ ] **Step 2: Run — FAIL** (`ImportError`).

- [ ] **Step 3: Implement** (append to `slip_systems.py`)

```python
# Lattice-translation fraction per family: the slip Burgers is this fraction
# of the integer direction's lattice vector (1/2 for the centered-lattice
# translations <110>_fcc, <111>_bcc).
_BURGERS_FRACTION: dict[str, float] = {
    "{111}<110>": 0.5,
    "{110}<111>": 0.5,
    "{112}<111>": 0.5,
}


def burgers_in_plane(
    structure: str, plane: tuple[int, int, int], *, families: "list[str] | None" = None
) -> np.ndarray:
    """Unit Burgers directions lying in `plane` (+ negatives), shape (m, 3).

    Replaces crystal.burgers._BASIS_TABLE/burgers_vectors. Per-vector
    Euclidean normalization (bit-identical to the old /√2 for FCC <110>).
    """
    cn = _canon(plane)
    pos: list[tuple[int, int, int]] = []
    for s in slip_systems(structure, families=families):
        if _canon(s.n) == cn and _canon(s.b) not in pos:
            pos.append(_canon(s.b))
    if not pos:
        raise ValueError(
            f"{plane} is not a slip plane for structure {structure!r}; "
            f"planes: {plane_normals(structure, families=families)}"
        )
    basis = np.array(pos, dtype=float)
    unit = basis / np.linalg.norm(basis, axis=1, keepdims=True)
    return np.vstack([unit, -unit])


def burgers_magnitude(
    structure: str, family: str, cell: UnitCell
) -> float:
    """Burgers magnitude |b| in µm for a structure's family, from the cell.

    |b| = fraction * |A · b_int|, A in metres -> result in µm. Cubic FCC
    1/2<110> -> a/√2; BCC 1/2<111> -> a√3/2.
    """
    fam = next(f for f in _REGISTRY[structure] if f.name == family)
    b_int = np.array(fam.burgers_family, dtype=float)
    cart = cell.A @ b_int  # metres
    frac = _BURGERS_FRACTION.get(family, 1.0)
    return float(frac * np.linalg.norm(cart) * 1e6)
```

- [ ] **Step 4: Run — PASS.** mypy 0.

- [ ] **Step 5: Commit** — `feat(crystal): burgers_in_plane + cell-derived burgers_magnitude`

---

### Task 3: structure-family derivation (`derive_structure_type` + cif helper)

**Files:**
- Modify: `src/dfxm_geo/crystal/cif.py` (add `space_group_structure_family`)
- Modify: `src/dfxm_geo/crystal/slip_systems.py` (add `derive_structure_type`)
- Test: `tests/test_slip_systems.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
from dfxm_geo.crystal.slip_systems import derive_structure_type


def test_derive_explicit_wins():
    assert derive_structure_type(structure_type="bcc", space_group=None, lattice="cubic") == "bcc"


def test_derive_default_is_fcc():
    assert derive_structure_type(structure_type=None, space_group=None, lattice="cubic") == "fcc"


@pytest.mark.parametrize(
    "sg, expected",
    [("Fm-3m", "fcc"), ("Im-3m", "bcc"), ("P6_3/mmc", "hcp")],
)
def test_derive_from_space_group(sg, expected):
    pytest.importorskip("gemmi")
    assert derive_structure_type(structure_type=None, space_group=sg, lattice=None) == expected


def test_derive_contradiction_raises():
    pytest.importorskip("gemmi")
    with pytest.raises(ValueError, match="contradicts"):
        derive_structure_type(structure_type="bcc", space_group="Fm-3m", lattice="cubic")
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement**

In `crystal/cif.py` (uses the existing `_import_gemmi()` boundary + `space_group_crystal_system`):

```python
def space_group_structure_family(space_group: str) -> str:
    """Map a space group to a slip-system structure family.

    F-cubic -> 'fcc', I-cubic -> 'bcc', P-hexagonal -> 'hcp'. Raises for
    families not yet supported by the slip-system registry.
    """
    gemmi = _import_gemmi()
    sg = gemmi.SpaceGroup(space_group)
    system = space_group_crystal_system(space_group)
    centring = sg.centring_type()  # 'P','I','F','C','R',...
    if system == "cubic" and centring == "F":
        return "fcc"
    if system == "cubic" and centring == "I":
        return "bcc"
    if system in ("hexagonal", "trigonal") and centring == "P":
        return "hcp"
    raise ValueError(
        f"space group {space_group!r} ({centring}-{system}) has no supported "
        f"slip-system family yet (fcc/bcc/hcp only); set [crystal] structure_type "
        f"or [[crystal.slip_system]] explicitly."
    )
```

(If `centring_type()` is not the exact gemmi API, use `sg.centring_type()` per gemmi 0.7 — verify with `python -c "import gemmi; print(gemmi.SpaceGroup('Fm-3m').centring_type())"`; adjust to the real accessor, e.g. `gemmi.find_spacegroup_by_name(...).centring_type()`.)

In `crystal/slip_systems.py`:

```python
def derive_structure_type(
    *, structure_type: "str | None", space_group: "str | None", lattice: "str | None"
) -> str:
    """Resolve the structure family. Explicit wins; else from space group;
    else 'fcc' (back-compat). Raises if explicit contradicts the space group."""
    if space_group is not None:
        from dfxm_geo.crystal.cif import space_group_structure_family

        derived = space_group_structure_family(space_group)
        if structure_type is not None and structure_type != derived:
            raise ValueError(
                f"structure_type={structure_type!r} contradicts space group "
                f"{space_group!r} (implies {derived!r})."
            )
        return derived
    if structure_type is not None:
        if structure_type not in _REGISTRY and structure_type != "hcp":
            raise ValueError(
                f"unknown structure_type {structure_type!r}; expected one of "
                f"{sorted(_REGISTRY)} (hcp lands in 4.3b)."
            )
        return structure_type
    return "fcc"
```

- [ ] **Step 4: Run — PASS** (gemmi tests skip if absent). mypy 0 (gemmi override already exists from 4.2).

- [ ] **Step 5: Commit** — `feat(crystal): derive structure family from space group centering`

---

### Task 4: `crystal/elasticity.py` — Poisson ratio table (cited)

**Files:**
- Create: `src/dfxm_geo/crystal/elasticity.py`
- Test: `tests/test_elasticity.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Per-material Poisson ratio resolution."""

import pytest

from dfxm_geo.crystal.elasticity import poisson_ratio


def test_override_wins():
    assert poisson_ratio(override=0.41, material="Fe") == 0.41


def test_material_lookup():
    assert poisson_ratio(override=None, material="Fe") == pytest.approx(0.29, abs=0.005)
    assert poisson_ratio(override=None, material="W") == pytest.approx(0.28, abs=0.005)


def test_default_is_al():
    assert poisson_ratio(override=None, material=None) == pytest.approx(0.334)


def test_unknown_material_raises():
    with pytest.raises(ValueError, match="unknown material"):
        poisson_ratio(override=None, material="Unobtainium")
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement** — every entry cited (Sina's requirement)

```python
"""Per-material isotropic Poisson ratio.

Isotropic-elastic displacement fields only (no anisotropic C_ijkl — out of
scope for v3.0.0; see the spec). Values are polycrystalline / Voigt-Reuss-Hill
averages from standard references; each entry cites its source. Used as the
ny prefactor in crystal/dislocations.py.

Sources:
  [KL]  Kaye & Laby, Tables of Physical & Chemical Constants, 16th ed.,
        Table 2.3.4 (Elastic constants of polycrystalline materials).
  [SW]  Simmons & Wang, Single Crystal Elastic Constants and Calculated
        Aggregate Properties, 2nd ed., MIT Press 1971 (VRH aggregate nu).
"""

from typing import Final

# element symbol -> (nu, source tag)
_POISSON_TABLE: Final[dict[str, tuple[float, str]]] = {
    "Al": (0.334, "SW"),  # Al, VRH aggregate
    "Fe": (0.29, "KL"),   # alpha-Fe (BCC)
    "W": (0.28, "KL"),    # tungsten (BCC)
    "Cu": (0.34, "SW"),   # copper (FCC)
    "Ni": (0.31, "KL"),   # nickel (FCC)
    "Ti": (0.32, "KL"),   # alpha-Ti (HCP)
    "Mg": (0.29, "SW"),   # magnesium (HCP)
}

_DEFAULT_NU: Final[float] = 0.334  # Al [SW]


def poisson_ratio(*, override: "float | None", material: "str | None") -> float:
    """Resolve nu: explicit override > material-table lookup > 0.334 (Al)."""
    if override is not None:
        return float(override)
    if material is not None:
        try:
            return _POISSON_TABLE[material][0]
        except KeyError:
            raise ValueError(
                f"unknown material {material!r}; known: {sorted(_POISSON_TABLE)}. "
                f"Set [crystal] poisson_ratio explicitly."
            ) from None
    return _DEFAULT_NU


def poisson_source(material: "str | None") -> str:
    """Citation tag for the resolved material (for provenance)."""
    if material is None:
        return "SW"
    return _POISSON_TABLE.get(material, (0.0, "override"))[1]
```

- [ ] **Step 4: Run — PASS.** mypy 0.

- [ ] **Step 5: Commit** — `feat(crystal): per-material Poisson ratio table (cited)`

---

### Task 5: generalize `crystal/burgers.py` to the registry

**Files:**
- Modify: `src/dfxm_geo/crystal/burgers.py`
- Test: `tests/test_burgers.py` (extend, don't break existing)

- [ ] **Step 1: Add failing tests** to `tests/test_burgers.py`

```python
def test_burgers_vectors_fcc_unchanged():
    """burgers_vectors(plane) stays bit-identical to the pre-registry result
    for the four {111} planes (FCC back-compat)."""
    import numpy as np
    from dfxm_geo.crystal.burgers import burgers_vectors

    legacy_111 = np.vstack(
        [np.array([[-1, 1, 0], [1, 0, -1], [0, 1, -1]], float)] * 1
    )
    got = burgers_vectors((1, 1, 1))
    want = np.vstack([legacy_111, -legacy_111]) / np.sqrt(2)
    gs = sorted(tuple(np.round(v, 9)) for v in got)
    ws = sorted(tuple(np.round(v, 9)) for v in want)
    assert np.allclose(gs, ws)
```

- [ ] **Step 2: Run — should already PASS** (this pins current behavior before the refactor). If it passes, good; the refactor must keep it passing.

- [ ] **Step 3: Refactor** `burgers.py` — replace `_BASIS_TABLE`/`_slug` usage in `burgers_vectors` with a delegation to the registry, keeping the FCC signature:

```python
from dfxm_geo.crystal.slip_systems import burgers_in_plane


def burgers_vectors(slip_plane_normal: tuple[int, int, int]) -> np.ndarray:
    """FCC {111}-family Burgers vectors (6, 3). FCC-compat shim over the
    structure-family registry; non-{111} still raises (identify default is FCC).
    """
    try:
        return burgers_in_plane("fcc", slip_plane_normal)
    except ValueError as exc:
        raise ValueError(
            f"slip_plane_normal {slip_plane_normal} is not one of the four "
            f"{{111}}-family variants"
        ) from exc
```

Keep `_BASIS_TABLE` in place until Task 1/2 tests that reference it have run green, THEN delete it in this task's final step (the registry is now the source of truth). Update `tests/test_burgers.py:50-53` (the "not one of the four" message) only if the message string changed — keep it matching.

- [ ] **Step 4: Run `tests/test_burgers.py` + `tests/test_slip_systems.py` — PASS.** Delete `_BASIS_TABLE` + `_slug` from `burgers.py`; re-run; fix the `test_burgers_in_plane_fcc_matches_legacy_basis` test (Task 2) to inline the four basis arrays instead of importing the now-deleted `_BASIS_TABLE`. mypy 0.

- [ ] **Step 5: Commit** — `refactor(crystal): burgers_vectors delegates to the registry; drop _BASIS_TABLE`

---

### Task 6: `CrystalMount` material/structure fields + resolution

**Files:**
- Modify: `src/dfxm_geo/crystal/oblique.py` (`CrystalMount`)
- Test: `tests/test_mount_space_group.py` (extend) or new `tests/test_mount_structure.py`

- [ ] **Step 1: Write failing tests** (`tests/test_mount_structure.py`)

```python
import pytest

from dfxm_geo.crystal.oblique import CrystalMount


def test_default_mount_resolves_fcc_and_al_nu():
    m = CrystalMount(lattice="cubic", a=4.0495e-10)
    assert m.resolved_structure_type == "fcc"
    assert m.resolved_poisson_ratio == pytest.approx(0.334)


def test_explicit_bcc_and_material():
    m = CrystalMount(lattice="cubic", a=2.8665e-10, structure_type="bcc", material="Fe")
    assert m.resolved_structure_type == "bcc"
    assert m.resolved_poisson_ratio == pytest.approx(0.29, abs=0.005)


def test_structure_contradicts_space_group():
    pytest.importorskip("gemmi")
    with pytest.raises(ValueError, match="contradicts"):
        CrystalMount(lattice="cubic", a=3.6e-10, structure_type="bcc", space_group="Fm-3m")
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement** — add fields (all default None → back-compat) + properties to `CrystalMount`:

```python
    structure_type: str | None = None
    material: str | None = None
    poisson_ratio: float | None = None
    slip_families: tuple[str, ...] | None = None
```

In `__post_init__` (after the existing space-group canonicalization), validate via `derive_structure_type` (it raises on contradiction). Add cached properties:

```python
    @property
    def resolved_structure_type(self) -> str:
        from dfxm_geo.crystal.slip_systems import derive_structure_type

        return derive_structure_type(
            structure_type=self.structure_type,
            space_group=self.space_group,
            lattice=self.lattice,
        )

    @property
    def resolved_poisson_ratio(self) -> float:
        from dfxm_geo.crystal.elasticity import poisson_ratio

        return poisson_ratio(override=self.poisson_ratio, material=self.material)
```

(`slip_families` is a tuple for frozen-dataclass hashability; the registry helpers take `list[str] | None`, so convert with `list(...)` at call sites.)

- [ ] **Step 4: Run — PASS.** Full `tests/test_mount_space_group.py` + new file green. mypy 0.

- [ ] **Step 5: Commit** — `feat(crystal): CrystalMount structure_type/material/poisson_ratio/slip_families`

---

### Task 7: config parsing — mount keys + `[[crystal.slip_system]]` hatch

**Files:**
- Modify: `src/dfxm_geo/config.py` (`_CRYSTAL_MOUNT_KEYS`, identify gate)
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py` (`_crystal_mount_from_toml`)
- Modify: `src/dfxm_geo/crystal/slip_systems.py` (register a custom structure from TOML)
- Test: `tests/test_kernel_cli_crystal_block.py` (extend) + `tests/test_slip_systems.py` (hatch)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_slip_systems.py
def test_custom_slip_systems_registration():
    from dfxm_geo.crystal.slip_systems import register_custom, slip_systems

    register_custom("mycustom", [{"plane": (1, 0, 0), "burgers": (0, 1, 0)}])
    sys = slip_systems("mycustom")
    assert len(sys) == 1
    assert sys[0].n == (1, 0, 0) and sys[0].b == (0, 1, 0)


def test_custom_slip_system_rejects_non_glide():
    from dfxm_geo.crystal.slip_systems import register_custom

    with pytest.raises(ValueError, match="b.n"):
        register_custom("bad", [{"plane": (1, 0, 0), "burgers": (1, 0, 0)}])
```

```python
# tests/test_kernel_cli_crystal_block.py (append)
def test_crystal_block_parses_structure_keys(tmp_path):
    from dfxm_geo.reciprocal_space.kernel import _crystal_mount_from_toml

    block = {
        "lattice": "cubic", "a": 2.8665e-10,
        "structure_type": "bcc", "material": "Fe", "poisson_ratio": 0.30,
        "slip_families": ["{110}<111>"],
    }
    mount = _crystal_mount_from_toml(block, base_dir=tmp_path)
    assert mount.resolved_structure_type == "bcc"
    assert mount.resolved_poisson_ratio == 0.30
    assert mount.slip_families == ("{110}<111>",)
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement**

In `slip_systems.py`:

```python
def register_custom(name: str, systems: "list[dict]") -> None:
    """Register a user-defined structure from [[crystal.slip_system]] entries.

    Each entry: {"plane": (h,k,l), "burgers": (u,v,w)}. Validates b.n == 0.
    Stored as single-system families so slip_systems(name) returns them.
    """
    fams: list[SlipFamily] = []
    for i, s in enumerate(systems):
        plane = tuple(int(x) for x in s["plane"])
        b = tuple(int(x) for x in s["burgers"])
        if b[0] * plane[0] + b[1] * plane[1] + b[2] * plane[2] != 0:
            raise ValueError(f"custom slip system {i}: b.n != 0 (b={b}, n={plane}); not glide")
        fams.append(SlipFamily(f"custom{i}", plane, b))
    _REGISTRY[name] = tuple(fams)
```

(Note: `_enumerate_family` enumerates variants — for custom systems we want the literal entry, not its whole symmetry orbit. Add a `literal: bool` to `SlipFamily` defaulting False; when True, `_enumerate_family` yields just the one `(b, n, n×b)`. Set `literal=True` in `register_custom`.)

In `config.py:72` add the four keys to `_CRYSTAL_MOUNT_KEYS`:
`"structure_type", "material", "poisson_ratio", "slip_families"`.

In `kernel.py` `_crystal_mount_from_toml`: read the four keys (with `slip_families` tuple-ified) and `[[crystal.slip_system]]` (call `register_custom` with a generated name like `"custom:<material-or-hash>"` and set `structure_type` to that name), pass into `CrystalMount(...)`.

In `config.py:822` identify validation: replace the `_burgers_vectors(slip_plane_normal)` {111}-only gate with a check that `slip_plane_normal` is in `plane_normals(mount.resolved_structure_type)` (resolve the mount in that scope; if not available, keep the fcc fallback).

- [ ] **Step 4: Run — PASS.** Full suite focus on `tests/test_config*`, `tests/test_kernel_cli_crystal_block.py`, `tests/test_slip_systems.py`. mypy 0.

- [ ] **Step 5: Commit** — `feat(config): parse structure_type/material/poisson_ratio + [[crystal.slip_system]] hatch`

---

### Task 8: `forward_model.py` — registry-driven population + cell |b|

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py`
- Test: `tests/test_forward_model_backlog.py` (update the `_SLIP_SYSTEM_111` anchors), `tests/test_bcc_e2e.py` (Task 12 covers e2e)

- [ ] **Step 1: Update the failing anchors** — `test_forward_model_backlog.py:122-143` referenced `_SLIP_SYSTEM_111` directly. Repoint them at `slip_systems("fcc")` (same 12, same geometry) so they still guard the FCC table:

```python
def test_fcc_slip_systems_geometry():
    from dfxm_geo.crystal.slip_systems import slip_systems
    sys = slip_systems("fcc")
    assert len(sys) == 12
    assert len({tuple(s.n) for s in sys}) >= 4
    for s in sys:
        assert sum(a * b for a, b in zip(s.b, s.n)) == 0
```

- [ ] **Step 2: Run — FAIL** (old test imported `fm._SLIP_SYSTEM_111`).

- [ ] **Step 3: Implement** — in `build_dislocation_population` (and the wall branch), resolve systems + |b| from the mount/cell:

```python
    structure = mount.resolved_structure_type
    fams = list(mount.slip_families) if mount.slip_families else None
    systems = slip_systems(structure, families=fams)
    b_um = burgers_magnitude(structure, systems[0].family, mount.cell)
    # random_dislocations: draw over `systems`
    slip_indices = rng.integers(0, len(systems), size=rd.ndis)
    for i in range(rd.ndis):
        s = systems[slip_indices[i]]
        Ud[i] = _ud_matrix_from_bnt(s.b, s.n, s.t)
    # wall: first system of the structure
    Ud_single = _ud_matrix_from_bnt(systems[0].b, systems[0].n, systems[0].t)
```

Thread `mount` into `build_dislocation_population` (it already receives `crystal`; add the mount or read it from the forward context — check the call chain; the mount is available where `ctx`/config is). Pass `b=b_um` into the `find_hg_population`/`Fd_find*` calls that currently default to `BURGERS_VECTOR`. For the FCC default, `structure="fcc"`, `systems` == the 12 (set-equal to `_SLIP_SYSTEM_111` per Task 1), and `b_um` == `2.862e-4` — **byte-identical** (the draw order must match: `_enumerate_family` sorts planes then burgers; verify the RNG draw reproduces the same Ud sequence as the old `_SLIP_SYSTEM_111` order, OR keep FCC mapping to the literal table order. If order differs, FCC random_dislocations output changes — to preserve bit-identity, order the FCC enumerator output to match `_SLIP_SYSTEM_111` exactly, asserted by a new ordered-equality test).**

Delete `_SLIP_SYSTEM_111` once the ordered-equality + bit-identity tests pass.

- [ ] **Step 4: Run — `tests/test_forward_model_backlog.py`, `tests/test_pipeline_crystal_modes.py`, `tests/test_fcc_bit_identity.py` (Task 11) PASS.** Full suite. mypy 0.

- [ ] **Step 5: Commit** — `feat(forward): registry-driven slip systems + cell-derived |b| (FCC bit-identical)`

---

### Task 9: `orchestrator.py` — plane_normals + burgers_in_plane + integer Burgers

**Files:**
- Modify: `src/dfxm_geo/orchestrator.py`
- Test: `tests/test_identification_scan_modes.py` (FCC stays green), `tests/test_bcc_e2e.py` (Task 12)

- [ ] **Step 1:** Rely on Task 12's BCC e2e + the existing FCC identify tests as the gate (no new unit test here — this is wiring).

- [ ] **Step 2: Run** the FCC identify tests to capture the green baseline.

- [ ] **Step 3: Implement** — replace `_ALL_111_PLANES` and `_burgers_vectors`:

```python
    structure = mount.resolved_structure_type
    fams = list(mount.slip_families) if mount.slip_families else None
    planes = (
        plane_normals(structure, families=fams)
        if crystal_cfg.sweep_all_slip_planes
        else [crystal_cfg.slip_plane_normal]
    )
    ...
    b_table = burgers_in_plane(structure, plane, families=fams)
```

Replace the `√2` integer reconstruction (`orchestrator.py:909-913`, and `:1031`, `:1047`) with the registry's integer Burgers. Get it by matching the unit `b_table[b_idx]` back to its integer system, or have `burgers_in_plane` also return the integers. Cleanest: add `burgers_in_plane_int(structure, plane, *, families)` returning the matching integer (u,v,w) array parallel to the unit array; use it for the `burgers` label:

```python
    burgers_int = tuple(int(x) for x in b_int_table[b_idx])
```

Keep `_ALL_111_PLANES` importable as `plane_normals("fcc")` if anything else imports it (grep `pipeline.py:61`) — update the pipeline facade re-export accordingly.

- [ ] **Step 4: Run** all identify tests + BCC e2e. FCC labels byte-identical (FCC `b_int` via the registry == the old `*√2` integers). mypy 0.

- [ ] **Step 5: Commit** — `feat(identify): registry plane_normals/burgers; integer Burgers from registry (no √2 literal)`

---

### Task 10: `viz/burgers.py` √2 + provenance attrs

**Files:**
- Modify: `src/dfxm_geo/viz/burgers.py` (the `*√2` at line 64)
- Modify: `src/dfxm_geo/io/hdf5.py` (structure provenance)
- Test: `tests/test_viz_burgers.py` (stays green), `tests/test_bcc_e2e.py` asserts attrs

- [ ] **Step 1:** Extend `tests/test_bcc_e2e.py` (Task 12) to assert the master `/1.1` carries `structure_type`, `slip_families`, `poisson_ratio`, `poisson_source`, `burgers_magnitude_um`, `material`.

- [ ] **Step 2: Run — FAIL** (attrs absent).

- [ ] **Step 3: Implement** — `viz/burgers.py:64` `*√2` → multiply by the actual `|b_int|` for display (or drop the rescale if the plotter only needs directions; preserve current FCC visual). In `io/hdf5.py`, assemble the structure provenance dict (mirroring the 4.2 crystal-attrs pattern) from the resolved mount and merge it into `attrs_1_1` / `ScanSpec.attrs`. Provenance never includes a cif path (4.2 rule).

- [ ] **Step 4: Run — PASS.** mypy 0.

- [ ] **Step 5: Commit** — `feat(io): structure-family provenance attrs; viz √2 generalized`

---

### Task 11: FCC bit-identity regression gate

**Files:**
- Create: `tests/test_fcc_bit_identity.py`

- [ ] **Step 1: Write the test** — a forward + identify run with NO structure keys, compared to a golden captured at the branch point:

```python
"""FCC default path must stay byte-identical to pre-4.3 main."""

import numpy as np
import pytest


@pytest.mark.slow
def test_fcc_random_dislocations_bit_identical(tmp_path, golden_dir):
    """random_dislocations FCC forward image == captured golden (same seed)."""
    # build the smallest FCC random_dislocations forward run (copy the config
    # from tests/test_pipeline_crystal_modes.py), fixed seed, small grid.
    # Compare /entry_0000/dfxm_sim_detector/image to
    # golden_dir/'fcc_random_dis_bit_identity.npy' at exact equality.
    ...
    assert np.array_equal(img, golden)
```

- [ ] **Step 2:** Capture the golden FROM MAIN before the registry lands. Since the registry already landed in earlier tasks, instead capture it by running the SAME config with `git stash`+checkout main is not possible mid-branch. Practical approach: generate the golden once on this branch, then assert determinism (same config+seed twice → identical) AND that the FCC slip-system Ud sequence equals the literal `_SLIP_SYSTEM_111` order (the ordered-equality test from Task 8 is the real bit-identity proof; this e2e test guards determinism + the full pipeline). Document in the test docstring that the algebraic bit-identity proof is Task 8's ordered-equality test.

- [ ] **Step 3:** Run twice, assert identical. Commit the golden under `tests/data/golden/`.

- [ ] **Step 4: Run — PASS.**

- [ ] **Step 5: Commit** — `test: FCC default bit-identity + determinism gate`

---

### Task 12: BCC end-to-end (DoD)

**Files:**
- Create: `tests/test_bcc_e2e.py`

- [ ] **Step 1: Write the failing tests** — both the explicit-`structure_type` and the `Fe.cif` routes (copy the smallest forward + identify configs from existing e2e tests; small grid per the smoke-test rule):

```python
"""BCC end-to-end: forward image + identify library with BCC labels."""

import numpy as np
import pytest


@pytest.mark.slow
def test_bcc_forward_runs(tmp_path):
    # [crystal] structure_type="bcc", material="Fe", a=2.8665e-10, a BCC
    # reflection (e.g. (1,1,0)); small grid. Forward writes a detector image.
    ...
    assert img.shape[0] >= 1 and np.isfinite(img).all()


@pytest.mark.slow
def test_bcc_identify_has_bcc_slip_labels(tmp_path):
    # identify single-mode, structure_type="bcc"; assert the master sample
    # group records slip_plane_normal in the BCC {110}/{112} set and that
    # >=1 scan passes the g·b visibility gate.
    ...
    assert tuple(slip_plane) in {(1,1,0),(1,1,2),(1,2,1),(2,1,1),...}  # use plane_normals("bcc")


@pytest.mark.slow
def test_bcc_via_fe_cif(tmp_path):
    pytest.importorskip("gemmi")
    # write a minimal Fe Im-3m CIF (a=2.8665), [crystal] cif=..., no
    # structure_type; assert resolved_structure_type derived as 'bcc' and a
    # forward image is produced.
    ...
```

- [ ] **Step 2: Run — FAIL** (until all wiring lands; this task runs last among the code tasks).

- [ ] **Step 3:** No new impl — this is the integration gate over Tasks 1–10. Fix any wiring gaps surfaced here.

- [ ] **Step 4: Run — PASS.** Full suite + slow. mypy 0.

- [ ] **Step 5: Commit** — `test: BCC forward + identify end-to-end (structure_type + Fe CIF)`

---

### Task 13: docs

**Files:**
- Create: `docs/crystal-structures.md`
- Modify: `docs/output-format.md` (new provenance attrs), `README.md` (only if it lists supported structures — grep first)

- [ ] **Step 1:** `docs/crystal-structures.md`: how structure type is resolved (explicit > space group > fcc), the registry families (FCC/BCC + default {110}+{112}), the `[[crystal.slip_system]]` hatch, the ν table WITH the [KL]/[SW] citations, the cell-derived |b| formulas, and a prominent **isotropic-elasticity-only** limitation note. Document HCP as "coming in 4.3b".
- [ ] **Step 2:** `docs/output-format.md`: the new `/1.1` structure provenance attrs.
- [ ] **Step 3:** Full suite + mypy green; `python -m compileall` clean.
- [ ] **Step 4: Commit** — `docs: crystal-structures (registry, ν citations, isotropic caveat)`

---

### Task 14: final gates

- [ ] `...python.exe -m pytest -q` — full suite green (compare the failure SET to the pre-existing xfail/skip baseline: 969-region + new tests; the lone xfail is hdf5 bit-equiv).
- [ ] `...python.exe -m pytest -q -m slow` — slow green (BCC e2e + FCC bit-identity).
- [ ] `...python.exe -m mypy src/dfxm_geo/` — 0 errors.
- [ ] `grep -rn "_SLIP_SYSTEM_111\|_BASIS_TABLE\|_ALL_111_PLANES" src/` — only `plane_normals`/registry remain (the three literals deleted).
- [ ] Review `git log --oneline main..HEAD`; update CLAUDE.md + auto-memory; hand off via `superpowers:finishing-a-development-branch` (no tag — v3.0.0 ships after 4.3b + 4.4; no push without Sina's nod).

---

## Self-review notes (for the executor)

- **Spec coverage:** §3.1 registry → Tasks 1,3,5,7,8,9; §3.2 |b| → Task 2,8; §3.3 ν → Task 4,6; §3.4 config → Tasks 6,7; §3.5 consumers → Tasks 8,9,10; q_hkl → Task 11 note (unchanged in 4.3a, asserted cubic-equal); §3.6 tests within each task + Tasks 11,12; §3.7 DoD → Task 12. 4.3b (§4) explicitly NOT in this plan.
- **The bit-identity risk is the RNG draw order** (Task 8): FCC `random_dislocations` draws `rng.integers(0, 12)` then indexes the slip table. If the registry's order ≠ `_SLIP_SYSTEM_111` order, the per-dislocation Ud sequence changes and FCC output is no longer byte-identical. Mitigation: an **ordered**-equality test (not just set-equality) FCC enumerator vs `_SLIP_SYSTEM_111`, and order the FCC family enumeration to match. This is the single most important correctness check in the plan.
- **`mount` threading** (Tasks 8,9): `build_dislocation_population` and the identify iterators need the resolved `CrystalMount`. Verify the call chain carries it (it carries `config`/`ctx`; the mount comes from `_crystal_mount_from_toml`). If not threaded, add it — do NOT read module globals.
- **gemmi `centring_type()`** (Task 3): verify the exact gemmi 0.7 accessor before relying on it.
- BCC q_hkl stays `q/|q|` — correct because BCC is cubic. The 4 q_hkl sites are only generalized in 4.3b (HCP). Task 11 documents this with a cubic-equality assert.
