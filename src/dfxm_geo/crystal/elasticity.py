"""Per-material isotropic Poisson ratio.

Isotropic-elastic displacement fields only (no anisotropic C_ijkl — out of
scope for v3.0.0; see the M4.3 spec). Values are polycrystalline /
Voigt-Reuss-Hill aggregate averages from standard references; each entry
cites its source. Used as the `ny` prefactor in crystal/dislocations.py.

Sources:
  [KL]  Kaye & Laby, Tables of Physical & Chemical Constants, 16th ed.,
        Sec. 2.3.4 (elastic properties of polycrystalline solids).
  [SW]  Simmons & Wang, Single Crystal Elastic Constants and Calculated
        Aggregate Properties, 2nd ed., MIT Press 1971 (VRH aggregate nu).
"""

from typing import Final

# element symbol -> (nu, source tag). Each value is cited (Sina requirement).
_POISSON_TABLE: Final[dict[str, tuple[float, str]]] = {
    "Al": (0.334, "SW"),  # aluminium (FCC), VRH aggregate
    "Fe": (0.29, "KL"),  # alpha-iron (BCC)
    "W": (0.28, "KL"),  # tungsten (BCC)
    "Cu": (0.34, "SW"),  # copper (FCC)
    "Ni": (0.31, "KL"),  # nickel (FCC)
    "Ti": (0.32, "KL"),  # alpha-titanium (HCP)
    "Mg": (0.29, "SW"),  # magnesium (HCP)
}

_DEFAULT_NU: Final[float] = _POISSON_TABLE["Al"][0]
_DEFAULT_SOURCE: Final[str] = _POISSON_TABLE["Al"][1]


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


def poisson_source(*, override: "float | None", material: "str | None") -> str:
    """Citation tag for the resolved nu (for provenance), mirroring poisson_ratio's
    precedence: explicit override -> 'override'; material -> its table tag; else
    the Al default tag. Raises for an unknown material (consistent with poisson_ratio).
    """
    if override is not None:
        return "override"
    if material is not None:
        try:
            return _POISSON_TABLE[material][1]
        except KeyError:
            raise ValueError(
                f"unknown material {material!r}; known: {sorted(_POISSON_TABLE)}."
            ) from None
    return _DEFAULT_SOURCE
