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
    except ValueError as exc:
        raise ValueError(f"unknown space-group symbol: {name!r}.") from exc
    return str(sg.hm)


def space_group_crystal_system(name: str) -> str:
    """Return the crystal system string for a canonical H-M space-group symbol."""
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


def reject_extinct(space_group: str | None, hkl: tuple[int, int, int], context: str) -> None:
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
        1.0,
        1.0,
        1.0,
        90.0,
        90.0,
        90.0,
    ):
        raise ValueError(f"CIF {p} carries no cell parameters (_cell_length_a …).")

    hm = small.spacegroup_hm  # '' when the CIF has no space-group tag
    space_group: str | None = None
    system: str | None = None
    if hm:
        space_group = validate_space_group(hm)
        system = space_group_crystal_system(space_group)

    lattice = _infer_lattice(system, cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)
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
