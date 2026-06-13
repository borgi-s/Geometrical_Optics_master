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

_DEFAULT_NU: Final[float] = 0.334  # Al [SW]
_DEFAULT_SOURCE: Final[str] = "SW"


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
    """Citation tag for the resolved material's nu (for provenance)."""
    if material is None:
        return _DEFAULT_SOURCE
    return _POISSON_TABLE.get(material, (0.0, "override"))[1]
