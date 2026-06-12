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
