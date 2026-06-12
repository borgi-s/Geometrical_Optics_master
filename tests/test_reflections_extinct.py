"""Stage 4.2: [[reflections]] entries hard-reject systematically-absent hkl."""

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.crystal.oblique import CrystalMount, compute_omega_eta
from dfxm_geo.crystal.reflections import resolve_reflections, resolve_reflections_auto

FCC = CrystalMount(
    lattice="cubic",
    a=4.0495e-10,
    space_group="Fm-3m",
    mount_x=(1, 0, 0),
    mount_y=(0, 1, 0),
    mount_z=(0, 0, 1),
)


def test_extinct_entry_rejected() -> None:
    with pytest.raises(ValueError, match="systematically absent"):
        resolve_reflections([{"hkl": [1, 0, 0], "eta": 0.0}], FCC, 17.0)


def test_allowed_entry_passes() -> None:
    runs = resolve_reflections([{"hkl": [-1, 1, -1]}], FCC, 17.0)
    assert len(runs) == 1


def test_auto_inherits_filtering() -> None:
    # [reflections_auto] expands via find_reflections, which already filters
    # (Task 3). Use the eta of the Al -1,1,-1 reflection as a real target.
    eta = compute_omega_eta(FCC, (-1, 1, -1), 17.0).eta_1
    runs = resolve_reflections_auto({"eta_target": float(eta)}, FCC, 17.0)
    for r in runs:
        h, k, l = r.hkl
        assert len({h % 2, k % 2, l % 2}) == 1, f"mixed-parity {r.hkl} survived"
