# M4 Stage 4.2 — CIF Ingestion + Extinction Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `[crystal] cif = "path.cif"` populates cell + space group via gemmi (TOML keys override), and space-group systematic absences are filtered from `find_reflections()` / hard-rejected for explicit hkl — with an Al₂O₃ ceramic acceptance case.

**Architecture:** Approach A from the approved spec (`docs/superpowers/specs/2026-06-12-m4-stage42-cif-ingestion-design.md`): a new `crystal/cif.py` is the **only** module that touches gemmi (lazily — `[cif]` optional extra); `CrystalMount` gains `space_group: str | None`; `UnitCell` is untouched (bit-identity gate). No space group anywhere → behavior identical to v2.x (opt-in filtering, hard error on explicit extinct hkl, no escape hatch).

**Tech Stack:** Python 3.11, gemmi ≥0.7 (already in the venv: 0.7.5), pytest, mypy. Venv python: `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe`. Repo root (= cwd for all commands): `C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master`. Branch: `feature/m4-stage42-cif-ingestion`.

**Verified facts baked into this plan (from the 2026-06-12 spike, gemmi 0.7.5):**
- `gemmi.read_small_structure(path)` is the small-molecule entry point. `gemmi.read_structure` SILENTLY returns a 1 Å placeholder cell on these CIFs — never use it. A CIF without cell tags also yields the placeholder `(1,1,1,90,90,90)` cell.
- `SmallStructure.spacegroup_hm` is `''` (empty string) when the CIF has no SG tag.
- `gemmi.SpaceGroup(name)` accepts `'Fm-3m'`, `'F m -3 m'`, `'P63/mmc'`, `'P6_3/mmc'`; raises `ValueError: Unknown space-group name: ...` otherwise. `.hm` is the canonical symbol (`'F m -3 m'`), `.crystal_system_str()` returns e.g. `'cubic'`/`'trigonal'`.
- `sg.operations().is_systematically_absent([h,k,l])` — textbook-verified: Fm-3m absent {100,110,210,211}, present {111,200,220,311}; P6₃/mmc (0,0,l) absent for odd l; **R-3c (hexagonal axes)** absent {(0,0,3),(0,0,9),(1,0,0),(1,0,1)}, present {(0,0,6),(0,0,12),(0,1,2),(1,0,4),(1,1,0),(1,1,3),(0,2,4),(1,1,6),(2,0,2)}.
- `CrystalMount` is `@dataclass(frozen=True, kw_only=True)` → in-place normalization needs `object.__setattr__`.
- `UnitCell.from_lattice` `_fill` uses **exact** equality for constrained params — fine for CIF values (identical strings parse identically), and a sloppy CIF fails with a clear error.

---

## Subagent dispatch map (waves, parallelism, models)

Per the user's standing rules: dispatch independent tasks **in one message, parallel, `run_in_background: true`**; set `model` explicitly on every dispatch. Tasks in the same wave touch disjoint files and may run concurrently. Never run two tasks that modify the same file in parallel.

| Wave | Tasks (parallel within wave) | Model | Why |
|---|---|---|---|
| 1 | Task 1 (`crystal/cif.py` + fixtures + unit tests) | **Sonnet** | new module, API given verbatim |
| 1 | Task 2 (packaging: pyproject extras + mypy) | **Haiku** | mechanical TOML edits + one test |
| 2 | Task 3 (`CrystalMount.space_group` + `find_reflections` filter) | **Sonnet** | one file, subtle frozen-dataclass detail spelled out |
| 3 | Task 4 (`_crystal_mount_from_toml` cif/space_group/base_dir) | **Sonnet** | parsing + override precedence |
| 3 | Task 5 (`[[reflections]]` extinct rejection) | **Sonnet** | small, but must read `_resolve_entry` |
| 4 | Task 6 (config.py: extinct checks, cif trigger, lattice_a inheritance, identify loader) | **Opus** | many interacting config paths; the subtle heart of the stage |
| 4 | Task 7 (bootstrap CLI extinct rejection + base_dir) | **Sonnet** | kernel.py only (disjoint from Task 6's config.py) |
| 4 | Task 8 (find-reflections CLI: base_dir + SG header) | **Haiku** | two-line change + CLI test |
| 5 | Task 9 (provenance serializers emit `space_group`) | **Sonnet** | config.py — must wait for Task 6 (same file) |
| 5 | Task 10 (ceramic acceptance e2e) | **Sonnet** | new test file only; needs 3+4+8 merged |
| 6 | Task 11 (docs: architecture.md + spec status + CLAUDE.md note) | **Haiku** | prose given below |
| 6 | Task 12 (final gates + cleanup) | controller or **Sonnet** | full suite + slow + mypy |

Dependency edges: 1→3→{4,5}; 4→{6,7,8}; 6→9; {3,4,8}→10; everything→12. Task 2 is independent of all (pyproject only).

---

### Task 1: `crystal/cif.py` — the gemmi boundary [Wave 1, parallel with Task 2, model: Sonnet]

**Files:**
- Create: `src/dfxm_geo/crystal/cif.py`
- Create: `tests/data/cif/al_fm3m.cif`, `tests/data/cif/mg_p63mmc.cif`, `tests/data/cif/al2o3_r3c.cif`, `tests/data/cif/nocell.cif`, `tests/data/cif/nospacegroup.cif`
- Create: `tests/test_cif_loader.py`, `tests/test_extinction_rules.py`

- [ ] **Step 1: Write the CIF fixtures** (exact content; `tests/data/cif/` is a new directory)

`tests/data/cif/al_fm3m.cif`:
```
data_Al
_cell_length_a    4.0495
_cell_length_b    4.0495
_cell_length_c    4.0495
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M   'F m -3 m'
_space_group_IT_number           225
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Al1 0 0 0
```

`tests/data/cif/mg_p63mmc.cif`:
```
data_Mg
_cell_length_a    3.2094
_cell_length_b    3.2094
_cell_length_c    5.2108
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 120
_symmetry_space_group_name_H-M   'P 63/m m c'
```

`tests/data/cif/al2o3_r3c.cif` (corundum, hexagonal axes — the ceramic):
```
data_Al2O3
_cell_length_a    4.7602
_cell_length_b    4.7602
_cell_length_c    12.9933
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 120
_symmetry_space_group_name_H-M   'R -3 c'
_space_group_IT_number           167
```

`tests/data/cif/nocell.cif`:
```
data_X
_symmetry_space_group_name_H-M   'F m -3 m'
```

`tests/data/cif/nospacegroup.cif`:
```
data_X
_cell_length_a    4.0
_cell_length_b    4.0
_cell_length_c    4.0
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
```

- [ ] **Step 2: Write the failing unit tests**

`tests/test_cif_loader.py`:
```python
"""M4 Stage 4.2: crystal/cif.py loader unit tests (spec 2026-06-12)."""

from pathlib import Path

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.crystal.cif import CifCell, load_cif

DATA = Path(__file__).parent / "data" / "cif"


class TestLoadCif:
    def test_al_cubic(self) -> None:
        cc = load_cif(DATA / "al_fm3m.cif")
        assert cc.lattice == "cubic"
        assert cc.a == pytest.approx(4.0495e-10)  # Angstrom -> metres
        assert cc.b == pytest.approx(4.0495e-10)
        assert cc.alpha_deg == 90.0 and cc.gamma_deg == 90.0
        assert cc.space_group == "F m -3 m"  # canonical H-M

    def test_mg_hexagonal(self) -> None:
        cc = load_cif(DATA / "mg_p63mmc.cif")
        assert cc.lattice == "hexagonal"
        assert cc.a == pytest.approx(3.2094e-10)
        assert cc.c == pytest.approx(5.2108e-10)
        assert cc.gamma_deg == 120.0
        assert cc.space_group == "P 63/m m c"

    def test_corundum_trigonal_sg_maps_to_hexagonal_lattice(self) -> None:
        # R-3c is crystal-system 'trigonal' but the CIF is in hexagonal axes
        # (gamma=120) -> lattice='hexagonal' per the Stage 4.1 convention.
        cc = load_cif(DATA / "al2o3_r3c.cif")
        assert cc.lattice == "hexagonal"
        assert cc.space_group == "R -3 c"
        assert cc.c == pytest.approx(12.9933e-10)

    def test_no_spacegroup_is_none_not_error(self) -> None:
        cc = load_cif(DATA / "nospacegroup.cif")
        assert cc.space_group is None
        assert cc.lattice == "cubic"  # inferred from cell parameters

    def test_no_cell_raises(self) -> None:
        with pytest.raises(ValueError, match="no cell parameters"):
            load_cif(DATA / "nocell.cif")

    def test_missing_file_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            load_cif(DATA / "does_not_exist.cif")


class TestGemmiMissing:
    def test_import_error_mentions_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import dfxm_geo.crystal.cif as cif_mod

        def _boom() -> None:
            raise ImportError(
                "CIF/space-group support requires gemmi — install with "
                "`pip install dfxm-geo[cif]` (or `conda install -c conda-forge gemmi`)."
            )

        monkeypatch.setattr(cif_mod, "_import_gemmi", _boom)
        with pytest.raises(ImportError, match=r"dfxm-geo\[cif\]"):
            cif_mod.load_cif(DATA / "al_fm3m.cif")
```

`tests/test_extinction_rules.py`:
```python
"""M4 Stage 4.2: extinction-rule wrapper vs textbook tables (NOT vs gemmi itself)."""

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.crystal.cif import (
    check_space_group_lattice,
    is_systematically_absent,
    reject_extinct,
    validate_space_group,
)

# Textbook systematic absences (International Tables / standard XRD refs).
FCC_ABSENT = [(1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 1, 1)]
FCC_PRESENT = [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1)]
BCC_ABSENT = [(1, 0, 0), (1, 1, 1), (2, 1, 0)]  # h+k+l odd
BCC_PRESENT = [(1, 1, 0), (2, 0, 0), (2, 1, 1)]
HCP_ABSENT = [(0, 0, 1), (0, 0, 3)]  # 000l: l = 2n (6_3 screw)
HCP_PRESENT = [(0, 0, 2), (1, 0, 0), (1, 0, 1)]
R3C_ABSENT = [(0, 0, 3), (0, 0, 9), (1, 0, 0), (1, 0, 1)]  # hex axes
R3C_PRESENT = [(0, 0, 6), (0, 1, 2), (1, 0, 4), (1, 1, 0), (1, 1, 3), (1, 1, 6)]


@pytest.mark.parametrize(
    "sg,absent,present",
    [
        ("Fm-3m", FCC_ABSENT, FCC_PRESENT),
        ("Im-3m", BCC_ABSENT, BCC_PRESENT),
        ("P63/mmc", HCP_ABSENT, HCP_PRESENT),
        ("R-3c", R3C_ABSENT, R3C_PRESENT),
    ],
)
def test_textbook_absences(sg, absent, present) -> None:
    for hkl in absent:
        assert is_systematically_absent(sg, hkl), f"{sg} {hkl} should be absent"
    for hkl in present:
        assert not is_systematically_absent(sg, hkl), f"{sg} {hkl} should be present"


def test_validate_space_group_canonicalizes_and_rejects() -> None:
    assert validate_space_group("Fm-3m") == "F m -3 m"
    assert validate_space_group("P6_3/mmc") == "P 63/m m c"
    with pytest.raises(ValueError, match="space-group"):
        validate_space_group("Fm-3x")


def test_check_space_group_lattice() -> None:
    assert check_space_group_lattice("Fm-3m", "cubic") == "F m -3 m"
    # Trigonal SG on a hexagonal-axes lattice is the R-3c corundum case: allowed.
    assert check_space_group_lattice("R-3c", "hexagonal") == "R -3 c"
    with pytest.raises(ValueError, match="incompatible"):
        check_space_group_lattice("Fm-3m", "hexagonal")


def test_reject_extinct() -> None:
    reject_extinct(None, (1, 0, 0), "[reciprocal] hkl")  # no SG -> no-op
    reject_extinct("Fm-3m", (1, 1, 1), "[reciprocal] hkl")  # allowed -> no-op
    with pytest.raises(ValueError, match="systematically absent"):
        reject_extinct("Fm-3m", (1, 0, 0), "[reciprocal] hkl")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_cif_loader.py tests/test_extinction_rules.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'dfxm_geo.crystal.cif'`

- [ ] **Step 4: Write the implementation**

`src/dfxm_geo/crystal/cif.py`:
```python
"""CIF ingestion + space-group extinction rules (M4 Stage 4.2).

The ONLY module that touches gemmi, and it imports it lazily: the `[cif]`
optional extra stays optional, and no gemmi objects escape this boundary.
Spec: docs/superpowers/specs/2026-06-12-m4-stage42-cif-ingestion-design.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ANGSTROM = 1e-10

# Crystal systems a CrystalMount `lattice` label may carry. A hexagonal-axes
# lattice legitimately hosts trigonal space groups (e.g. corundum R-3c) — the
# Stage 4.1 convention maps hexagonal-setting trigonal cells to
# lattice='hexagonal'.
_COMPATIBLE_SYSTEMS: dict[str, frozenset[str]] = {
    "cubic": frozenset({"cubic"}),
    "tetragonal": frozenset({"tetragonal"}),
    "orthorhombic": frozenset({"orthorhombic"}),
    "hexagonal": frozenset({"hexagonal", "trigonal"}),
    "trigonal": frozenset({"trigonal"}),
    "monoclinic": frozenset({"monoclinic"}),
    "triclinic": frozenset({"triclinic"}),
}


def _import_gemmi() -> Any:
    try:
        import gemmi
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "CIF/space-group support requires gemmi — install with "
            "`pip install dfxm-geo[cif]` (or `conda install -c conda-forge gemmi`)."
        ) from exc
    return gemmi


@dataclass(frozen=True)
class CifCell:
    """Cell + symmetry read from a CIF. Lengths in METRES, angles in degrees."""

    lattice: str
    a: float
    b: float
    c: float
    alpha_deg: float
    beta_deg: float
    gamma_deg: float
    space_group: str | None  # canonical H-M symbol, or None if the CIF has none
    source: str


def validate_space_group(name: str) -> str:
    """Resolve a (forgivingly spelled) H-M symbol; return the canonical form."""
    gemmi = _import_gemmi()
    try:
        sg = gemmi.SpaceGroup(name)
    except ValueError:
        raise ValueError(f"unknown space-group symbol: {name!r}.") from None
    return str(sg.hm)


def space_group_crystal_system(name: str) -> str:
    gemmi = _import_gemmi()
    return str(gemmi.SpaceGroup(name).crystal_system_str())


def check_space_group_lattice(space_group: str, lattice: str) -> str:
    """Validate the symbol AND its compatibility with a lattice label.

    Returns the canonical H-M symbol.
    """
    canonical = validate_space_group(space_group)
    system = space_group_crystal_system(canonical)
    if system not in _COMPATIBLE_SYSTEMS.get(lattice, frozenset()):
        raise ValueError(
            f"space group {canonical!r} belongs to the {system} crystal system, "
            f"which is incompatible with lattice={lattice!r}."
        )
    return canonical


def is_systematically_absent(space_group: str, hkl: tuple[int, int, int]) -> bool:
    """True if `hkl` is a systematic absence of `space_group` (centering + glide/screw)."""
    gemmi = _import_gemmi()
    ops = gemmi.SpaceGroup(space_group).operations()
    return bool(ops.is_systematically_absent(list(hkl)))


def absence_checker(space_group: str) -> Callable[[tuple[int, int, int]], bool]:
    """Build-once checker for hkl loops (avoids re-parsing the symbol per hkl)."""
    gemmi = _import_gemmi()
    ops = gemmi.SpaceGroup(space_group).operations()

    def _absent(hkl: tuple[int, int, int]) -> bool:
        return bool(ops.is_systematically_absent(list(hkl)))

    return _absent


def reject_extinct(
    space_group: str | None, hkl: tuple[int, int, int], context: str
) -> None:
    """Shared hard-error guard for explicitly-configured reflections.

    No-op when no space group is known (filtering is opt-in by design).
    """
    if space_group is None:
        return
    if is_systematically_absent(space_group, hkl):
        raise ValueError(
            f"{context}: reflection {tuple(hkl)} is systematically absent in "
            f"space group {space_group!r} — run 'dfxm-find-reflections' to "
            f"list allowed reflections."
        )


def _infer_lattice(
    system: str | None,
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> str:
    """Map a crystal system (or, lacking one, the cell shape) to a lattice label."""
    if system == "trigonal":
        # Hexagonal-axes setting (corundum R-3c etc.) -> 'hexagonal' (4.1 rule);
        # rhombohedral setting keeps 'trigonal'.
        return "hexagonal" if abs(gamma - 120.0) < 1e-6 else "trigonal"
    if system is not None:
        return system
    # No space group in the CIF: classify from the cell parameters.
    def eq(x: float, y: float) -> bool:
        return abs(x - y) <= 1e-9 * max(abs(x), abs(y), 1e-30)

    right = eq(alpha, 90.0) and eq(beta, 90.0)
    if right and eq(gamma, 90.0):
        if eq(a, b) and eq(b, c):
            return "cubic"
        if eq(a, b):
            return "tetragonal"
        return "orthorhombic"
    if right and eq(gamma, 120.0) and eq(a, b):
        return "hexagonal"
    if eq(a, b) and eq(b, c) and eq(alpha, beta) and eq(beta, gamma):
        return "trigonal"
    if eq(alpha, 90.0) and eq(gamma, 90.0):
        return "monoclinic"
    return "triclinic"


def load_cif(path: str | Path) -> CifCell:
    """Read cell parameters + space group from a small-molecule CIF.

    Uses gemmi.read_small_structure — NOT read_structure, whose macromolecular
    parser silently returns a placeholder 1 Å cell on small-molecule CIFs.
    """
    gemmi = _import_gemmi()
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"[crystal] cif file not found: {p}")
    try:
        small = gemmi.read_small_structure(str(p))
    except Exception as exc:
        raise ValueError(f"could not parse CIF {p}: {exc}") from exc
    cell = small.cell
    # gemmi signals "no cell tags" with its placeholder unit cell.
    if (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma) == (
        1.0, 1.0, 1.0, 90.0, 90.0, 90.0,
    ):
        raise ValueError(f"CIF {p} carries no cell parameters (_cell_length_a …).")

    hm = small.spacegroup_hm  # '' when the CIF has no space-group tag
    space_group: str | None = None
    system: str | None = None
    if hm:
        space_group = validate_space_group(hm)
        system = space_group_crystal_system(space_group)

    lattice = _infer_lattice(
        system, cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma
    )
    return CifCell(
        lattice=lattice,
        a=cell.a * _ANGSTROM,
        b=cell.b * _ANGSTROM,
        c=cell.c * _ANGSTROM,
        alpha_deg=float(cell.alpha),
        beta_deg=float(cell.beta),
        gamma_deg=float(cell.gamma),
        space_group=space_group,
        source=str(p),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_cif_loader.py tests/test_extinction_rules.py -q`
Expected: all pass (≈11 tests)

- [ ] **Step 6: mypy the new module**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: 0 errors (if mypy complains about the gemmi import: Task 2 adds the override; coordinate — if Task 2 hasn't merged yet, note it and move on, the gate re-runs in Task 12)

- [ ] **Step 7: Commit**

```bash
git add src/dfxm_geo/crystal/cif.py tests/data/cif tests/test_cif_loader.py tests/test_extinction_rules.py
git commit -m "feat: crystal/cif.py gemmi boundary — CIF loader + extinction rules (Stage 4.2 Task 1)"
```

---

### Task 2: Packaging — `[cif]` extra + mypy override [Wave 1, parallel with Task 1, model: Haiku]

**Files:**
- Modify: `pyproject.toml` (`[project.optional-dependencies]` at line ~42; `[tool.mypy]` overrides module list at line ~123)
- Create: `tests/test_pyproject_cif_extra.py`

- [ ] **Step 1: Write the failing test** (mirror the existing `tests/test_pyproject_hdf5_dep.py` pattern)

`tests/test_pyproject_cif_extra.py`:
```python
"""Stage 4.2 packaging: [cif] optional extra + gemmi in dev + mypy override."""

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def _load() -> dict:
    with open(PYPROJECT, "rb") as fh:
        return tomllib.load(fh)


def test_cif_extra_pins_gemmi() -> None:
    extras = _load()["project"]["optional-dependencies"]
    assert "cif" in extras
    assert any(d.startswith("gemmi") for d in extras["cif"])


def test_dev_extra_includes_gemmi() -> None:
    extras = _load()["project"]["optional-dependencies"]
    assert any(d.startswith("gemmi") for d in extras["dev"])


def test_mypy_ignores_gemmi_imports() -> None:
    overrides = _load()["tool"]["mypy"]["overrides"]
    modules = [m for o in overrides for m in o.get("module", [])]
    assert "gemmi" in modules
```

- [ ] **Step 2: Run it to verify it fails**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_pyproject_cif_extra.py -q`
Expected: 3 failures (KeyError/assert)

- [ ] **Step 3: Edit pyproject.toml**

In `[project.optional-dependencies]`: append `"gemmi>=0.7",` to the `dev` list, and add after the `beamstop-wire` block:
```toml
cif = [
    "gemmi>=0.7",
]
```
In the `[tool.mypy]` overrides `module = [...]` list (line ~123, the one ending with `ignore_missing_imports = true`): add `"gemmi",` keeping alphabetical-ish placement next to `"fabio"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_pyproject_cif_extra.py tests/test_pyproject_version.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_pyproject_cif_extra.py
git commit -m "build: [cif] optional extra (gemmi), dev extra + mypy override (Stage 4.2 Task 2)"
```

---

### Task 3: `CrystalMount.space_group` + `find_reflections` filtering [Wave 2, after Task 1, model: Sonnet]

**Files:**
- Modify: `src/dfxm_geo/crystal/oblique.py` (field at ~line 51, `__post_init__` end ~line 107, `find_reflections` ~line 333)
- Create: `tests/test_mount_space_group.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_mount_space_group.py`:
```python
"""Stage 4.2: space_group on CrystalMount + opt-in extinction filtering."""

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.crystal.oblique import CrystalMount, find_reflections

AL = dict(lattice="cubic", a=4.0495e-10, mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1))


def test_space_group_default_none() -> None:
    m = CrystalMount(**AL)
    assert m.space_group is None


def test_space_group_canonicalized() -> None:
    m = CrystalMount(**AL, space_group="Fm-3m")
    assert m.space_group == "F m -3 m"


def test_space_group_lattice_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        CrystalMount(**AL, space_group="P63/mmc")


def test_bad_symbol_rejected() -> None:
    with pytest.raises(ValueError, match="space-group"):
        CrystalMount(**AL, space_group="Fm-3x")


def test_find_reflections_filters_absences() -> None:
    bare = CrystalMount(**AL)
    fcc = CrystalMount(**AL, space_group="Fm-3m")
    keV = 17.0
    all_refl = find_reflections(bare, keV, hkl_max=2)
    filtered = find_reflections(fcc, keV, hkl_max=2)
    assert len(filtered) < len(all_refl)
    for g in filtered:
        h, k, l = g.hkl
        parities = {h % 2, k % 2, l % 2}
        assert len(parities) == 1, f"mixed-parity {g.hkl} survived Fm-3m filtering"
    # The classic FCC reflection family is still there.
    assert any(set(map(abs, g.hkl)) == {1} for g in filtered)


def test_find_reflections_unchanged_without_space_group() -> None:
    bare = CrystalMount(**AL)
    # Regression guard: no SG -> identical to the pre-4.2 enumeration.
    refl = find_reflections(bare, 17.0, hkl_max=1)
    assert len(refl) > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_mount_space_group.py -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'space_group'`

- [ ] **Step 3: Implement**

In `CrystalMount` (frozen, kw_only), after `mount_z: tuple[int, int, int]`:
```python
    # M4 Stage 4.2: optional space group (canonical H-M after __post_init__).
    # None = no symmetry knowledge; extinction filtering is opt-in.
    space_group: str | None = None
```
At the END of `__post_init__` (after the orthogonality checks):
```python
        if self.space_group is not None:
            from dfxm_geo.crystal.cif import check_space_group_lattice

            canonical = check_space_group_lattice(self.space_group, self.lattice)
            object.__setattr__(self, "space_group", canonical)
```
(`object.__setattr__` because the dataclass is frozen. The import is function-local but cheap — `crystal.cif` itself never imports gemmi at module level.)

In `find_reflections`, before the triple loop:
```python
    is_absent = None
    if mount.space_group is not None:
        from dfxm_geo.crystal.cif import absence_checker

        is_absent = absence_checker(mount.space_group)
```
Inside the loop, immediately after the `if h == 0 and k == 0 and l == 0: continue` guard (BEFORE `compute_omega_eta`, so absent hkl skip the solver entirely):
```python
                if is_absent is not None and is_absent((h, k, l)):
                    continue
```
Also extend the `find_reflections` docstring with one line: `When the mount carries a space_group, systematically-absent reflections are skipped (M4 Stage 4.2).`

- [ ] **Step 4: Run tests**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_mount_space_group.py tests/test_oblique_crystal_mount.py tests/test_oblique_find_reflections.py -q`
Expected: all PASS (the two pre-existing files prove no regression)

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/crystal/oblique.py tests/test_mount_space_group.py
git commit -m "feat: CrystalMount.space_group + opt-in extinction filter in find_reflections (Stage 4.2 Task 3)"
```

---

### Task 4: `_crystal_mount_from_toml` — `cif` / `space_group` keys + `base_dir` [Wave 3, after Task 3, parallel with Task 5, model: Sonnet]

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py:37-64` (`_crystal_mount_from_toml`)
- Create: `tests/test_crystal_mount_from_toml_cif.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_crystal_mount_from_toml_cif.py`:
```python
"""Stage 4.2: [crystal] cif + space_group parsing with per-key TOML override."""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.reciprocal_space.kernel import _crystal_mount_from_toml

DATA = Path(__file__).parent / "data" / "cif"
MOUNTS = {"mount_x": [1, 0, 0], "mount_y": [0, 1, 0], "mount_z": [0, 0, 1]}


def test_cif_populates_cell_and_space_group() -> None:
    m = _crystal_mount_from_toml({"cif": str(DATA / "al_fm3m.cif"), **MOUNTS})
    assert m.lattice == "cubic"
    assert m.a == pytest.approx(4.0495e-10)
    assert m.space_group == "F m -3 m"


def test_explicit_toml_key_overrides_cif() -> None:
    m = _crystal_mount_from_toml(
        {"cif": str(DATA / "al_fm3m.cif"), "a": 4.05e-10, **MOUNTS}
    )
    assert m.a == 4.05e-10  # TOML wins
    assert m.space_group == "F m -3 m"  # untouched keys still come from the CIF


def test_space_group_toml_overrides_cif() -> None:
    m = _crystal_mount_from_toml(
        {"cif": str(DATA / "al_fm3m.cif"), "space_group": "Pm-3m", **MOUNTS}
    )
    assert m.space_group == "P m -3 m"


def test_space_group_without_cif() -> None:
    m = _crystal_mount_from_toml(
        {"lattice": "cubic", "a": 4.0495e-10, "space_group": "Fm-3m", **MOUNTS}
    )
    assert m.space_group == "F m -3 m"


def test_relative_cif_path_resolves_against_base_dir(tmp_path: Path) -> None:
    shutil.copy(DATA / "al_fm3m.cif", tmp_path / "al.cif")
    m = _crystal_mount_from_toml({"cif": "al.cif", **MOUNTS}, base_dir=tmp_path)
    assert m.a == pytest.approx(4.0495e-10)


def test_cif_still_requires_mount_keys() -> None:
    with pytest.raises(ValueError, match="missing key: mount_x"):
        _crystal_mount_from_toml({"cif": str(DATA / "al_fm3m.cif")})


def test_no_cif_no_lattice_still_errors() -> None:
    with pytest.raises(ValueError, match="missing key: lattice"):
        _crystal_mount_from_toml({"a": 4.0495e-10, **MOUNTS})


def test_hexagonal_cif_end_to_end() -> None:
    # Hexagonal needs an orthogonal plane-normal triple: a*, a*-2b* (.L c*).
    m = _crystal_mount_from_toml(
        {
            "cif": str(DATA / "mg_p63mmc.cif"),
            "mount_x": [1, 0, 0],
            "mount_y": [1, -2, 0],
            "mount_z": [0, 0, 1],
        }
    )
    assert m.lattice == "hexagonal"
    assert m.c == pytest.approx(5.2108e-10)
    assert m.space_group == "P 63/m m c"
```

- [ ] **Step 2: Run to verify failure**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_crystal_mount_from_toml_cif.py -q`
Expected: FAIL — `TypeError: _crystal_mount_from_toml() got an unexpected keyword argument 'base_dir'` / KeyError-driven ValueErrors

- [ ] **Step 3: Implement** — replace the body of `_crystal_mount_from_toml` (kernel.py:37-64) with:

```python
def _crystal_mount_from_toml(
    data: dict | None, base_dir: Path | None = None
) -> CrystalMount:
    """Build a CrystalMount from a `[crystal]` TOML block (or None → default Al).

    M4 Stage 4.2: an optional ``cif`` key reads cell parameters + space group
    from a CIF file (relative paths resolve against ``base_dir``, the config
    TOML's directory); explicit TOML keys override CIF values per-key. An
    optional ``space_group`` key works with or without a CIF. ``mount_x/y/z``
    are always TOML-only (the CIF cannot know the experimental mounting).
    """
    if data is None:
        return _DEFAULT_AL_CRYSTAL

    cif_vals: dict = {}
    if "cif" in data:
        from dfxm_geo.crystal.cif import load_cif

        cif_path = Path(str(data["cif"]))
        if not cif_path.is_absolute() and base_dir is not None:
            cif_path = base_dir / cif_path
        cc = load_cif(cif_path)
        cif_vals = {
            "lattice": cc.lattice,
            "a": cc.a,
            "b": cc.b,
            "c": cc.c,
            "alpha_deg": cc.alpha_deg,
            "beta_deg": cc.beta_deg,
            "gamma_deg": cc.gamma_deg,
            "space_group": cc.space_group,
        }

    def _opt(key: str) -> float | None:
        if key in data:
            return float(data[key])
        val = cif_vals.get(key)
        return float(val) if val is not None else None

    try:
        lattice = data.get("lattice", cif_vals.get("lattice"))
        if lattice is None:
            raise KeyError("lattice")
        a_val = data.get("a", cif_vals.get("a"))
        if a_val is None:
            raise KeyError("a")
        return CrystalMount(
            lattice=lattice,
            a=float(a_val),
            b=_opt("b"),
            c=_opt("c"),
            alpha_deg=_opt("alpha_deg"),
            beta_deg=_opt("beta_deg"),
            gamma_deg=_opt("gamma_deg"),
            space_group=data.get("space_group", cif_vals.get("space_group")),
            mount_x=tuple(int(x) for x in data["mount_x"]),  # type: ignore[arg-type]
            mount_y=tuple(int(x) for x in data["mount_y"]),  # type: ignore[arg-type]
            mount_z=tuple(int(x) for x in data["mount_z"]),  # type: ignore[arg-type]
        )
    except KeyError as exc:
        raise ValueError(f"[crystal] block missing key: {exc.args[0]}") from None
```
(Check kernel.py already has `from pathlib import Path` at top — it does, `--config` is `type=Path`.)

NOTE for the hexagonal/cubic CIF case: the CIF supplies all six parameters and `UnitCell.from_lattice` fills/checks constrained ones with exact equality — identical CIF strings parse to identical floats, so this passes; an internally inconsistent CIF fails with `from_lattice`'s clear message. Do not add tolerances here.

- [ ] **Step 4: Run tests** (new + the existing kernel-CLI crystal-block suite)

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_crystal_mount_from_toml_cif.py tests/test_kernel_cli_crystal_block.py tests/test_kernel_cli_geometry_block.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/kernel.py tests/test_crystal_mount_from_toml_cif.py
git commit -m "feat: [crystal] cif + space_group keys in _crystal_mount_from_toml, base_dir-aware (Stage 4.2 Task 4)"
```

---

### Task 5: `[[reflections]]` extinct-hkl rejection [Wave 3, after Task 3, parallel with Task 4, model: Sonnet]

**Files:**
- Modify: `src/dfxm_geo/crystal/reflections.py` (`_resolve_entry`, called from `resolve_reflections` at line ~117)
- Create: `tests/test_reflections_extinct.py`

- [ ] **Step 1: Read `_resolve_entry`** in `src/dfxm_geo/crystal/reflections.py` to find where the entry's `hkl` is first parsed into a tuple (it is the function `resolve_reflections` maps over entries at line 117).

- [ ] **Step 2: Write the failing tests**

`tests/test_reflections_extinct.py`:
```python
"""Stage 4.2: [[reflections]] entries hard-reject systematically-absent hkl."""

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.crystal.oblique import CrystalMount
from dfxm_geo.crystal.reflections import resolve_reflections, resolve_reflections_auto

FCC = CrystalMount(
    lattice="cubic", a=4.0495e-10, space_group="Fm-3m",
    mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1),
)


def test_extinct_entry_rejected() -> None:
    with pytest.raises(ValueError, match="systematically absent"):
        resolve_reflections([{"hkl": [1, 0, 0], "eta": 0.0}], FCC, 17.0)


def test_allowed_entry_passes() -> None:
    runs = resolve_reflections([{"hkl": [-1, 1, -1]}], FCC, 17.0)
    assert len(runs) == 1


def test_auto_inherits_filtering() -> None:
    # [reflections_auto] expands via find_reflections, which already filters
    # (Task 3); smoke-assert no mixed-parity hkl comes back for FCC.
    runs = resolve_reflections_auto({"eta_target": 0.615}, FCC, 17.0)
    for r in runs:
        h, k, l = r.hkl
        assert len({h % 2, k % 2, l % 2}) == 1
```
(If `resolve_reflections_auto({"eta_target": 0.615}, ...)` returns an empty list for this mount, pick the eta of the Al -1,1,-1 reflection instead: compute it in the test via `compute_omega_eta(FCC, (-1, 1, -1), 17.0).eta_1`. An empty-but-no-error result is acceptable for the smoke assertion as long as `test_extinct_entry_rejected` exercises the hard error.)

- [ ] **Step 3: Run to verify the first test fails**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_reflections_extinct.py -q`
Expected: `test_extinct_entry_rejected` FAILS (no rejection yet — it errors differently or passes resolution); others may pass already.

- [ ] **Step 4: Implement** — in `_resolve_entry`, immediately after the entry's `hkl` tuple is parsed, add:

```python
    reject_extinct(mount.space_group, hkl, "[[reflections]]")
```
with the module-top import (safe — `crystal.cif` only imports gemmi lazily inside functions):
```python
from dfxm_geo.crystal.cif import reject_extinct
```
(Adapt the variable name to whatever `_resolve_entry` calls its parsed hkl tuple; if `_resolve_entry` does not receive `mount`, do the loop in `resolve_reflections` right before `resolved = [_resolve_entry(...)]` instead — same one-liner per entry.)

- [ ] **Step 5: Run tests**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_reflections_extinct.py tests/test_reflections_resolver.py tests/test_reflections_config.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/dfxm_geo/crystal/reflections.py tests/test_reflections_extinct.py
git commit -m "feat: reject systematically-absent hkl in [[reflections]] entries (Stage 4.2 Task 5)"
```

---

### Task 6: config.py — extinct checks, cif trigger, lattice_a inheritance, identify loader [Wave 4, after Task 4, parallel with Tasks 7+8, model: **Opus**]

This is the subtle one: four config paths interact. Read `_build_geometry_config` (config.py:442-513), `SimulationConfig.from_toml` (:579-605), and `load_identification_config` (:738-800) before editing.

**Files:**
- Modify: `src/dfxm_geo/config.py`
- Create: `tests/test_config_cif.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_config_cif.py`:
```python
"""Stage 4.2 config semantics: cif trigger, lattice_a inheritance, extinct rejection."""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.config import SimulationConfig, load_identification_config

DATA = Path(__file__).parent / "data" / "cif"


def _write(tmp_path: Path, body: str) -> Path:
    shutil.copy(DATA / "al_fm3m.cif", tmp_path / "al.cif")
    shutil.copy(DATA / "mg_p63mmc.cif", tmp_path / "mg.cif")
    p = tmp_path / "config.toml"
    p.write_text(body, encoding="utf-8")
    return p


MOUNT_LINES = """
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
"""


class TestSimplifiedMode:
    def test_cubic_cif_inherits_lattice_a(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            f'[crystal]\ncif = "al.cif"\n{MOUNT_LINES}\n'
            '[geometry]\nmode = "simplified"\n'
            "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n",
        )
        cfg = SimulationConfig.from_toml(p)
        assert cfg.reciprocal.lattice_a == pytest.approx(4.0495e-10)

    def test_explicit_lattice_a_wins_over_cif(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            f'[crystal]\ncif = "al.cif"\n{MOUNT_LINES}\n'
            '[geometry]\nmode = "simplified"\n'
            "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\nlattice_a = 4.05e-10\n",
        )
        cfg = SimulationConfig.from_toml(p)
        assert cfg.reciprocal.lattice_a == 4.05e-10

    def test_noncubic_cif_rejected_in_simplified(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            '[crystal]\ncif = "mg.cif"\n'
            "mount_x = [1, 0, 0]\nmount_y = [1, -2, 0]\nmount_z = [0, 0, 1]\n"
            '[geometry]\nmode = "simplified"\n',
        )
        with pytest.raises(ValueError, match="non-cubic"):
            SimulationConfig.from_toml(p)

    def test_extinct_hkl_rejected_simplified(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            f'[crystal]\ncif = "al.cif"\n{MOUNT_LINES}\n'
            '[geometry]\nmode = "simplified"\n'
            "[reciprocal]\nhkl = [1, 0, 0]\nkeV = 17.0\n",
        )
        with pytest.raises(ValueError, match="systematically absent"):
            SimulationConfig.from_toml(p)

    def test_bare_space_group_key_rejects_extinct(self, tmp_path: Path) -> None:
        # No CIF, no mount keys — just symmetry knowledge on a default config.
        p = _write(
            tmp_path,
            '[crystal]\nspace_group = "Fm-3m"\n'
            "[reciprocal]\nhkl = [1, 0, 0]\nkeV = 17.0\n",
        )
        with pytest.raises(ValueError, match="systematically absent"):
            SimulationConfig.from_toml(p)

    def test_bare_space_group_allowed_hkl_passes(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            '[crystal]\nspace_group = "Fm-3m"\n'
            "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n",
        )
        cfg = SimulationConfig.from_toml(p)
        assert cfg.geometry.mode == "simplified"


class TestObliqueMode:
    def test_extinct_hkl_rejected_oblique(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            f'[crystal]\ncif = "al.cif"\n{MOUNT_LINES}\n'
            '[geometry]\nmode = "oblique"\neta = 0.0\n'
            "[reciprocal]\nhkl = [1, 0, 0]\nkeV = 17.0\n",
        )
        with pytest.raises(ValueError, match="systematically absent"):
            SimulationConfig.from_toml(p)

    def test_mount_carries_space_group(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            f'[crystal]\ncif = "al.cif"\n{MOUNT_LINES}\n'
            '[geometry]\nmode = "oblique"\neta = 0.0\n'
            "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n",
        )
        cfg = SimulationConfig.from_toml(p)
        assert cfg.geometry.mount is not None
        assert cfg.geometry.mount.space_group == "F m -3 m"


class TestIdentificationConfig:
    def test_identify_loader_strips_new_keys_and_checks_extinct(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            f'[crystal]\ncif = "al.cif"\n{MOUNT_LINES}\n'
            '[geometry]\nmode = "oblique"\neta = 0.0\n'
            "[reciprocal]\nhkl = [1, 0, 0]\nkeV = 17.0\n",
        )
        with pytest.raises(ValueError, match="systematically absent"):
            load_identification_config(p)

    def test_identify_loader_happy_path(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            f'[crystal]\ncif = "al.cif"\n{MOUNT_LINES}\n'
            '[geometry]\nmode = "oblique"\neta = 0.0\n'
            "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n",
        )
        cfg = load_identification_config(p)
        assert cfg.geometry.mount is not None
        assert cfg.geometry.mount.space_group == "F m -3 m"
```

- [ ] **Step 2: Run to verify failures**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_config_cif.py -q`
Expected: multiple FAILs (lattice_a stays at the Al default; no extinct rejection; identify loader chokes on `cif`/`space_group` reaching `IdentificationCrystalConfig`)

- [ ] **Step 3: Implement, in five sub-edits**

**(a)** New helper above `SimulationConfig` (near `_build_geometry_config`):
```python
def _maybe_inherit_cif_lattice_a(raw: dict, base_dir: Path | None) -> None:
    """[crystal] cif + cubic cell + no explicit [reciprocal] lattice_a → inherit a.

    Mutates ``raw`` in place BEFORE ReciprocalConfig.from_dict so the Bragg
    validity check and the analytic backend see the CIF lattice parameter.
    Explicit [reciprocal] lattice_a wins (per-key override rule). Non-cubic
    cells don't inherit — lattice_a is the CUBIC simplified-geometry knob.
    """
    crystal_raw = raw.get("crystal") or {}
    if "cif" not in crystal_raw:
        return
    if "lattice_a" in (raw.get("reciprocal") or {}):
        return
    from dfxm_geo.reciprocal_space.kernel import _crystal_mount_from_toml

    mount = _crystal_mount_from_toml(crystal_raw, base_dir=base_dir)
    if mount.cell.is_cubic:
        raw.setdefault("reciprocal", {})["lattice_a"] = mount.a
```

**(b)** `_build_geometry_config`: new signature
```python
def _build_geometry_config(
    raw: dict,
    reciprocal: ReciprocalConfig,
    multi_reflection: bool = False,
    base_dir: Path | None = None,
) -> GeometryConfig:
```
Add `reject_extinct` to the existing function-local import block:
```python
    from dfxm_geo.crystal.cif import reject_extinct
```
Simplified branch — replace the `if "lattice" in crystal_raw:` block body and add the bare-space_group case:
```python
    if mode == "simplified":
        crystal_raw = raw.get("crystal") or {}
        # A "lattice" or "cif" key signals an explicit bootstrap-style mount;
        # parse it (which may surface mount errors simplified mode previously
        # ignored) so non-cubic cells cannot slip through the cubic-only path.
        if "lattice" in crystal_raw or "cif" in crystal_raw:
            mount = _crystal_mount_from_toml(crystal_raw, base_dir=base_dir)
            if not mount.cell.is_cubic:
                raise ValueError(
                    "non-cubic [crystal] cells require [geometry] mode='oblique' "
                    "(simplified mode hardwires the cubic symmetric geometry)."
                )
            if not multi_reflection:
                reject_extinct(mount.space_group, reciprocal.hkl, "[reciprocal] hkl")
        elif "space_group" in crystal_raw and not multi_reflection:
            # Bare symmetry knowledge on an otherwise-default mount (no CIF,
            # no mount keys) still gates the configured reflection.
            reject_extinct(
                str(crystal_raw["space_group"]), reciprocal.hkl, "[reciprocal] hkl"
            )
        return GeometryConfig(mode="simplified")
```
Oblique branch — pass `base_dir` and add the single-reflection check right after the existing non-cubic rejection:
```python
    mount = _crystal_mount_from_toml(raw.get("crystal"), base_dir=base_dir)

    if not mount.cell.is_cubic:
        raise ValueError(...)  # UNCHANGED existing message

    if not multi_reflection:
        reject_extinct(mount.space_group, reciprocal.hkl, "[reciprocal] hkl")
```
(Multi-reflection entries are guarded per-entry by Task 5; do NOT check `reciprocal.hkl` for multi — it's the unused default there.)

**(c)** `SimulationConfig.from_toml`: after `raw = tomllib.load(fh)` add
```python
        base_dir = Path(path).parent
        _maybe_inherit_cif_lattice_a(raw, base_dir)
```
and change the geometry call to
```python
        geometry = _build_geometry_config(raw, reciprocal, multi_reflection=multi, base_dir=base_dir)
```

**(d)** `load_identification_config`: same two-line insertion after `data = tomllib.load(fh)` (using `data` instead of `raw`), same `base_dir=` on its `_build_geometry_config` call (line ~782), and extend the mount-key strip list (line ~765) to:
```python
    for _mount_key in (
        "lattice", "a", "b", "c", "alpha_deg", "beta_deg", "gamma_deg",
        "cif", "space_group", "mount_x", "mount_y", "mount_z",
    ):
        crystal_data.pop(_mount_key, None)
```
(Check first whether `b`/`c`/`alpha_deg`/`beta_deg`/`gamma_deg` are already stripped — Stage 4.1 may have extended this list. Keep whatever is there and add `cif` + `space_group`; if the optional cell keys are missing AND `IdentificationCrystalConfig` would reject them, add them too.)

**(e)** Check `CrystalConfig.from_dict` (forward-side `[crystal]` parser): it "picks known fields", so `cif`/`space_group` flow past it harmlessly — verify by running `tests/test_pipeline_crystal_modes.py`; if it errors on unknown keys, apply the same pop-strip there.

- [ ] **Step 4: Run tests**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_config_cif.py tests/test_configs_load_under_new_schema.py tests/test_empty_toml_runs.py tests/test_partial_reciprocal_override.py tests/test_reciprocal_lattice_a.py tests/test_identification_oblique_wiring.py tests/test_pipeline_crystal_modes.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/config.py tests/test_config_cif.py
git commit -m "feat: config-level CIF wiring — cif trigger, lattice_a inheritance, extinct rejection (Stage 4.2 Task 6)"
```

---

### Task 7: `dfxm-bootstrap` extinct rejection + base_dir [Wave 4, after Task 4, parallel with Tasks 6+8, model: Sonnet]

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py` — `cli_main` (mount parse ~line 777, `_MOUNT_KEYS` ~757, hkl validation ~806-812)
- Create: `tests/test_kernel_cli_cif.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_kernel_cli_cif.py`:
```python
"""Stage 4.2: dfxm-bootstrap rejects extinct hkl + resolves relative cif paths."""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.reciprocal_space.kernel import cli_main

DATA = Path(__file__).parent / "data" / "cif"


def _config(tmp_path: Path, hkl: str, extra_crystal: str = "") -> Path:
    shutil.copy(DATA / "al_fm3m.cif", tmp_path / "al.cif")
    p = tmp_path / "config.toml"
    p.write_text(
        '[crystal]\ncif = "al.cif"\n'  # relative on purpose: base_dir test
        "mount_x = [1, 0, 0]\nmount_y = [0, 1, 0]\nmount_z = [0, 0, 1]\n"
        f"{extra_crystal}"
        '[geometry]\nmode = "oblique"\neta = 0.0\n'
        f"[reciprocal]\nhkl = {hkl}\nkeV = 17.0\n",
        encoding="utf-8",
    )
    return p


def test_bootstrap_rejects_extinct_hkl(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rc = cli_main(["--config", str(_config(tmp_path, "[1, 0, 0]"))])
    assert rc == 1
    assert "systematically absent" in capsys.readouterr().err


def test_bootstrap_resolves_relative_cif_and_validates(tmp_path: Path, capsys) -> None:
    # Allowed hkl gets PAST mount parsing and extinction; we don't need the
    # full (minutes-long) kernel run to prove the parse path, so point
    # --output at an existing file WITHOUT --force: the overwrite guard
    # exits 1 AFTER validation. (Same trick as the existing kernel CLI tests
    # — check tests/test_kernel_cli.py for the established pattern and reuse
    # it if it differs.)
    out = tmp_path / "kernel.npz"
    out.write_bytes(b"placeholder")
    rc = cli_main(["--config", str(_config(tmp_path, "[-1, 1, -1]")), "--output", str(out)])
    err = capsys.readouterr().err
    assert "systematically absent" not in err
    assert "cif file not found" not in err
    assert rc == 1  # refused overwrite — but only after the parse succeeded
```

- [ ] **Step 2: Run to verify failure**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_kernel_cli_cif.py -q`
Expected: first test FAILS (no rejection; the run proceeds or dies on the relative path)

- [ ] **Step 3: Implement** — three edits in `cli_main`:

1. `_MOUNT_KEYS` (line ~757): add `"cif"`:
```python
    _MOUNT_KEYS = ("lattice", "a", "mount_x", "mount_y", "mount_z", "cif")
```
2. Mount parse (line ~777): thread the config dir:
```python
        mount = _crystal_mount_from_toml(
            mount_block if is_mount_block else None, base_dir=args.config.parent
        )
```
3. After `theta = _validate_reflection(hkl_tuple, float(raw_keV), cell)` (line ~809), inside the same `try`:
```python
            from dfxm_geo.crystal.cif import reject_extinct

            sg = mount.space_group
            if sg is None and mount_block:
                # Bare [crystal] space_group on a forward-layout block (no
                # mount keys) — still gate the reflection.
                raw_sg = mount_block.get("space_group")
                sg = str(raw_sg) if raw_sg is not None else None
            reject_extinct(sg, hkl_tuple, "[reciprocal] hkl")
```
(The surrounding `except ValueError` already prints `error: {exc}` and returns 1 — the test asserts that.)

- [ ] **Step 4: Run tests** (new + the kernel CLI regression set)

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_kernel_cli_cif.py tests/test_kernel_cli.py tests/test_kernel_cli_crystal_block.py tests/test_kernel_cli_geometry_block.py tests/test_kernel_cli_both_validated.py tests/test_kernel_cli_filename_and_metadata.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/kernel.py tests/test_kernel_cli_cif.py
git commit -m "feat: dfxm-bootstrap rejects extinct hkl; cif key detected as mount block (Stage 4.2 Task 7)"
```

---

### Task 8: `dfxm-find-reflections` — base_dir + space-group header [Wave 4, after Task 4, parallel with Tasks 6+7, model: Haiku]

**Files:**
- Modify: `src/dfxm_geo/find_reflections_cmd.py:62` (mount parse) and `:85-87` (header print)
- Modify: `tests/test_find_reflections_cli.py` (append one test)

- [ ] **Step 1: Write the failing test** (append to `tests/test_find_reflections_cli.py`; match its existing invocation style)

```python
def test_cif_config_relative_path_and_sg_header(tmp_path, capsys):
    import shutil
    from pathlib import Path as _P

    import pytest as _pytest

    _pytest.importorskip("gemmi")
    data = _P(__file__).parent / "data" / "cif"
    shutil.copy(data / "al_fm3m.cif", tmp_path / "al.cif")
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[crystal]\ncif = "al.cif"\n'
        "mount_x = [1, 0, 0]\nmount_y = [0, 1, 0]\nmount_z = [0, 0, 1]\n"
        "[reciprocal]\nkeV = 17.0\n",
        encoding="utf-8",
    )
    from dfxm_geo.find_reflections_cmd import cli_main

    rc = cli_main(["--config", str(cfg), "--hkl-max", "2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "space_group=F m -3 m" in out
    assert " 1 0 0 " not in out  # forbidden FCC reflection filtered
```

- [ ] **Step 2: Run to verify it fails**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_find_reflections_cli.py -q`
Expected: new test FAILS (relative cif path unresolved → `cif file not found`)

- [ ] **Step 3: Implement** — two edits in `find_reflections_cmd.py`:

Line 62:
```python
    mount = _crystal_mount_from_toml(raw.get("crystal"), base_dir=config_path.parent)
```
Header print (line ~85-87):
```python
    sg_note = f"  space_group={mount.space_group}" if mount.space_group else ""
    print(
        f"# mount: x={mount.mount_x} y={mount.mount_y} z={mount.mount_z}  lattice={mount.lattice}  a={mount.a:g} m  keV={keV:g}{sg_note}"
    )
```

- [ ] **Step 4: Run tests**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_find_reflections_cli.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/find_reflections_cmd.py tests/test_find_reflections_cli.py
git commit -m "feat: dfxm-find-reflections resolves relative cif paths + prints space group (Stage 4.2 Task 8)"
```

---

### Task 9: Provenance — emit `space_group` (resolved values only, never the cif path) [Wave 5, after Task 6 (same file), model: Sonnet]

**Files:**
- Modify: `src/dfxm_geo/config.py` — `_dataclass_to_toml_str` `[crystal]` block (~line 886-899) and `_identification_config_to_toml_str` (~line 994-1013)
- Create: `tests/test_provenance_space_group.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_provenance_space_group.py`:
```python
"""Stage 4.2: provenance TOML emits space_group; cubic no-SG output byte-identical."""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.config import SimulationConfig, _dataclass_to_toml_str

DATA = Path(__file__).parent / "data" / "cif"

MOUNT_LINES = "mount_x = [1, 0, 0]\nmount_y = [0, 1, 0]\nmount_z = [0, 0, 1]\n"


def _load(tmp_path: Path, crystal_extra: str) -> SimulationConfig:
    shutil.copy(DATA / "al_fm3m.cif", tmp_path / "al.cif")
    p = tmp_path / "config.toml"
    p.write_text(
        f"[crystal]\n{crystal_extra}{MOUNT_LINES}"
        '[geometry]\nmode = "oblique"\neta = 0.0\n'
        "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\n",
        encoding="utf-8",
    )
    return SimulationConfig.from_toml(p)


def test_space_group_emitted_and_cif_path_not(tmp_path: Path) -> None:
    cfg = _load(tmp_path, 'cif = "al.cif"\n')
    toml_str = _dataclass_to_toml_str(cfg)
    assert 'space_group = "F m -3 m"' in toml_str
    assert "cif" not in toml_str.split("[crystal]")[1].split("[")[0].replace(
        "space_group", ""
    )  # resolved values only; the cif key itself is never echoed


def test_no_space_group_line_when_none(tmp_path: Path) -> None:
    cfg = _load(tmp_path, 'lattice = "cubic"\na = 4.0495e-10\n')
    assert "space_group" not in _dataclass_to_toml_str(cfg)


def test_provenance_round_trips_space_group(tmp_path: Path) -> None:
    cfg = _load(tmp_path, 'cif = "al.cif"\n')
    echo = tmp_path / "echo.toml"
    echo.write_text(_dataclass_to_toml_str(cfg), encoding="utf-8")
    cfg2 = SimulationConfig.from_toml(echo)
    assert cfg2.geometry.mount is not None
    assert cfg2.geometry.mount.space_group == "F m -3 m"
```

- [ ] **Step 2: Run to verify failure**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_provenance_space_group.py -q`
Expected: first + third FAIL (`space_group` never emitted)

- [ ] **Step 3: Implement** — in `_dataclass_to_toml_str`, after the non-cubic block and BEFORE the `mount_x` line (config.py ~896), insert:

```python
        if mount.space_group is not None:
            lines.append(f'space_group = "{mount.space_group}"')
```
Mirror the same two lines in `_identification_config_to_toml_str` at the equivalent spot (after the non-cubic `gamma_deg` block, before `mount_x`, ~line 1008):
```python
        if mount.space_group is not None:
            lines.append(f'space_group = "{mount.space_group}"')
```

- [ ] **Step 4: Run tests + the provenance regression set**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_provenance_space_group.py tests/test_hdf5_provenance.py tests/test_pipeline_writes_oblique_provenance.py tests/test_hdf5_geometry_provenance_helper.py tests/test_sidecar.py -q`
Expected: PASS (the pre-existing files prove cubic no-SG output is unchanged)

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/config.py tests/test_provenance_space_group.py
git commit -m "feat: provenance TOML emits space_group (resolved values, no cif path) (Stage 4.2 Task 9)"
```

---

### Task 10: Ceramic acceptance — Al₂O₃ end-to-end [Wave 5, after Tasks 3+4+8, model: Sonnet]

**Files:**
- Create: `tests/test_ceramic_acceptance_al2o3.py`

- [ ] **Step 1: Write the test** (this is acceptance, not TDD — everything it needs already shipped in Tasks 1-8; it must pass on first run)

`tests/test_ceramic_acceptance_al2o3.py`:
```python
"""Stage 4.2 ceramic acceptance: Al2O3 (corundum, R-3c) CIF end-to-end.

Pins the 'ceramics planning works' claim: the CIF loads, the mount builds,
dfxm-find-reflections enumerates only symmetry-allowed reflections, and the
textbook R-3c extinction conditions hold for every emitted row:
  - R-centering (hexagonal axes): -h + k + l = 3n
  - c-glide on (0,0,l): l = 6n
"""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.find_reflections_cmd import cli_main

DATA = Path(__file__).parent / "data" / "cif"


@pytest.fixture
def alumina_config(tmp_path: Path) -> Path:
    shutil.copy(DATA / "al2o3_r3c.cif", tmp_path / "al2o3.cif")
    p = tmp_path / "alumina.toml"
    # Orthogonal plane-normal triple for hexagonal axes: a*, a*-2b*, c*.
    p.write_text(
        '[crystal]\ncif = "al2o3.cif"\n'
        "mount_x = [1, 0, 0]\nmount_y = [1, -2, 0]\nmount_z = [0, 0, 1]\n"
        "[reciprocal]\nkeV = 17.0\n",
        encoding="utf-8",
    )
    return p


def _rows(out: str) -> list[tuple[int, int, int]]:
    rows = []
    for line in out.splitlines():
        if line.startswith("#") or line.strip().startswith("hkl"):
            continue
        parts = line.split()
        if len(parts) >= 7:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return rows


def test_alumina_reflection_table_obeys_textbook_extinctions(
    alumina_config: Path, capsys: pytest.CaptureFixture
) -> None:
    rc = cli_main(["--config", str(alumina_config), "--hkl-max", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "space_group=R -3 c" in out
    rows = _rows(out)
    assert len(rows) > 0, "no reachable reflections enumerated for alumina"
    for h, k, l in rows:
        assert (-h + k + l) % 3 == 0, f"R-centering violated by {(h, k, l)}"
        if h == 0 and k == 0:
            assert l % 6 == 0, f"c-glide 000l condition violated by {(0, 0, l)}"
    # Named textbook absences never appear.
    for forbidden in [(0, 0, 3), (0, 0, 9), (1, 0, 0), (1, 0, 1)]:
        assert forbidden not in rows


def test_alumina_explicit_forbidden_reflection_hard_errors(
    alumina_config: Path, tmp_path: Path
) -> None:
    from dfxm_geo.config import SimulationConfig

    body = alumina_config.read_text(encoding="utf-8")
    p = tmp_path / "alumina_bad.toml"
    p.write_text(
        body.replace(
            "[reciprocal]\nkeV = 17.0\n",
            '[geometry]\nmode = "oblique"\neta = 0.0\n[reciprocal]\nhkl = [0, 0, 3]\nkeV = 17.0\n',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="systematically absent"):
        SimulationConfig.from_toml(p)
```
NOTE: if the second test trips the oblique-mode "non-cubic cells are not yet supported in the forward/identify pipeline" guard BEFORE the extinction check, that is correct Stage 4.1 behavior — reorder the checks in `_build_geometry_config` so `reject_extinct` runs BEFORE the non-cubic rejection (the extinction error is the more specific diagnosis), and keep both tests. Coordinate with Task 6's merged code; this is a one-line move.

- [ ] **Step 2: Run it**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_ceramic_acceptance_al2o3.py -q`
Expected: PASS (possibly after the check-reorder noted above)

- [ ] **Step 3: Commit**

```bash
git add tests/test_ceramic_acceptance_al2o3.py src/dfxm_geo/config.py
git commit -m "test: Al2O3 ceramic acceptance — CIF e2e + textbook R-3c extinctions (Stage 4.2 Task 10)"
```

---

### Task 11: Docs [Wave 6, after Task 10, model: Haiku]

**Files:**
- Modify: `docs/architecture.md` — "Crystal cell and mount" subsection (added in Stage 4.1)
- Modify: `docs/superpowers/specs/2026-06-12-m4-stage42-cif-ingestion-design.md` — status line

- [ ] **Step 1: Append to the "Crystal cell and mount" subsection of `docs/architecture.md`:**

```markdown
### CIF ingestion and extinction rules (M4 Stage 4.2)

`crystal/cif.py` is the only module that touches **gemmi** (lazy import; the
`[cif]` optional extra — `pip install dfxm-geo[cif]`, or `conda install -c
conda-forge gemmi` since conda recipes carry no optional deps). A
`[crystal] cif = "path.cif"` key populates the six cell parameters (Å → m)
and the space group; explicit TOML keys override CIF values per-key, and a
plain `[crystal] space_group = "Fm-3m"` works without any CIF.
`mount_x/y/z` remain TOML-only. Relative CIF paths resolve against the
config file's directory.

With a space group present, `find_reflections()` skips systematic absences
(centering + glide/screw, via gemmi's symmetry operations) and every site
that accepts an explicit hkl — config load, `[[reflections]]` entries,
`dfxm-bootstrap` — hard-errors on a forbidden reflection. No space group →
behavior identical to v2.x (opt-in by design). Provenance emits the
*resolved* cell values plus `space_group`, never the cif path.
```

- [ ] **Step 2: Update the spec status line** in `docs/superpowers/specs/2026-06-12-m4-stage42-cif-ingestion-design.md` from "implementation on `feature/m4-stage42-cif-ingestion`" to "**Status: implemented on `feature/m4-stage42-cif-ingestion`, 2026-06-XX (fill in date).**"

- [ ] **Step 3: Commit**

```bash
git add docs/architecture.md docs/superpowers/specs/2026-06-12-m4-stage42-cif-ingestion-design.md
git commit -m "docs: architecture + spec status for Stage 4.2 CIF ingestion (Task 11)"
```

NOTE for the controller (not the subagent): also update the out-of-repo `CLAUDE.md` release-checklist bullet — at next PyPI publish, the conda-forge recipe must NOT gain gemmi as a `run` dep (optional extra; document `conda install gemmi` instead), alongside the existing `dfxm-find-reflections` entry-point mirror note. And the roadmap `roadmap202606audit.md` Stage 4.2 block gets its ✅ at merge time.

---

### Task 12: Final gates + cleanup [Wave 6, last, controller-run or model: Sonnet]

- [ ] **Step 1: Full suite**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q`
Expected: 0 failures. Baseline at branch point: 898 passed / 1 skipped / 22 deselected / 1 xfailed (the skip = legacy pickle absent, the xfail = hdf5 bit-equiv — both pre-existing; compare the failure SET, not the green count). New tests add ≈35 to the pass count.

- [ ] **Step 2: Slow markers**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q -m slow`
Expected: 13 passed / 0 failures (same as the 4.1 gate)

- [ ] **Step 3: mypy**

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/`
Expected: `Success: no issues found in 41 source files` (40 + the new cif.py)

- [ ] **Step 4: Bit-identity subset** (the Stage 4.1 regression gate still holds)

Run: `& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q tests/test_forward_model_smoke.py tests/test_hdf5_bit_equiv.py tests/test_oblique_compute_omega_eta.py tests/test_run_theta.py tests/test_dislocations_smoke.py tests/test_find_hg_kernel_parity.py`
Expected: same pass/xfail set as at branch point (23 passed / 1 xfailed)

- [ ] **Step 5: Cleanup session intermediates**

Delete the spike scratch files (created during brainstorming, not plan tasks):
```bash
rm tmp/spike_gemmi_cif.py tmp/spike_al.cif tmp/spike_mg.cif tmp/spike_nospg.cif
```
Check for any session-created file >10 MB (none expected — this stage writes no kernels): `git status` + a glance at `direct_space/deformation_gradient_tensors/`.

- [ ] **Step 6: Commit any stragglers, then hand to superpowers:finishing-a-development-branch**

Merge decision (merge to local main / push / keep) is Sina's call per CLAUDE.md. NO version tag — v3.0.0 ships at the END of M4 (after 4.3/4.4).

---

## Self-review notes (already applied)

- Spec coverage: goals 1-3 → Tasks 4+6 (cif key + overrides), 3+5+6+7 (filtering + rejection at all three sites), 10 (ceramic). Packaging → Task 2. Provenance → Task 9. Error handling → Tasks 1, 4, 6, 7. Simplified-mode lattice_a inheritance → Task 6. Docs → Task 11.
- Check-order wrinkle surfaced in Task 10 (extinction vs non-cubic rejection order) is called out explicitly rather than left to be discovered.
- Type consistency: `reject_extinct(space_group: str | None, hkl, context)`, `absence_checker(space_group) -> Callable`, `CifCell` field names, and `_crystal_mount_from_toml(data, base_dir=None)` are used with identical signatures across Tasks 1, 3, 4, 5, 6, 7.
