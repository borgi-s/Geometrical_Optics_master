"""Task 9: structure-aware identify plane/Burgers sweep.

Unit-level coverage of ``_resolve_identify_planes_and_burgers`` and the non-FCC
branch of ``_draw_dislocation``. The FCC path must REUSE the exact v2.x objects
(``_ALL_111_PLANES`` + ``_burgers_vectors`` + the ``*√2`` integer
reconstruction) so the identify byte-identity gate (test_identification_*,
test_gb_labels_in_identify) keeps passing; the BCC path must come from the
registry with ⟨111⟩ integer Burgers (NOT ⟨110⟩×√2).
"""

import numpy as np

from dfxm_geo.constants import BURGERS_VECTOR
from dfxm_geo.crystal.burgers import burgers_vectors as _burgers_vectors
from dfxm_geo.crystal.oblique import CrystalMount
from dfxm_geo.orchestrator import (
    _ALL_111_PLANES,
    _draw_dislocation,
    _identify_structure_is_fcc,
    _resolve_identify_planes_and_burgers,
)


def test_fcc_path_reuses_exact_v2x_objects():
    """mount None and explicit fcc both reuse _ALL_111_PLANES + _burgers_vectors
    (same object identity → no behavioural drift) and the *√2 integer label."""
    for mount in (None, CrystalMount(lattice="cubic", a=4.0495e-10, structure_type="fcc")):
        assert _identify_structure_is_fcc(mount) is True
        planes, burgers_fn, burgers_int_fn, burgers_mag_fn = _resolve_identify_planes_and_burgers(
            mount
        )
        assert planes is _ALL_111_PLANES  # same order + signs (load-bearing)
        assert burgers_fn is _burgers_vectors
        # integer label == the historical int(round(unit * √2)) reconstruction
        for plane in _ALL_111_PLANES:
            bt = _burgers_vectors(plane)
            for k in range(len(bt)):
                want = np.asarray(
                    [int(round(bt[k, c] * np.sqrt(2))) for c in range(3)], dtype=np.int32
                )
                assert np.array_equal(burgers_int_fn(plane, k), want)
                # FCC identify keeps the default |b| (byte-identical, M4 4.3b).
                assert burgers_mag_fn(plane, k) == BURGERS_VECTOR


def test_bcc_path_is_registry_driven_111():
    """BCC routes to the non-FCC branch: plane_normals/burgers_in_plane from the
    registry, integer Burgers are ⟨111⟩ aligned with the unit table (× √3)."""
    mount = CrystalMount(lattice="cubic", a=2.8665e-10, structure_type="bcc", material="Fe")
    assert _identify_structure_is_fcc(mount) is False
    planes, burgers_fn, burgers_int_fn, burgers_mag_fn = _resolve_identify_planes_and_burgers(mount)
    assert planes is not _ALL_111_PLANES
    assert (1, 1, 0) in planes  # BCC {110} present
    unit = burgers_fn((1, 1, 0))
    for k in range(unit.shape[0]):
        bi = burgers_int_fn((1, 1, 0), k)
        assert sorted(abs(int(c)) for c in bi) == [1, 1, 1]  # ⟨111⟩
        assert np.allclose(bi.astype(float), unit[k] * np.sqrt(3))  # index alignment
        # BCC (cubic) identify keeps the default |b| (byte-identical, M4 4.3b).
        assert burgers_mag_fn((1, 1, 0), k) == BURGERS_VECTOR


def test_bcc_slip_family_subset_filters_planes():
    """slip_families restricts the plane sweep to the requested family ({110})."""
    mount = CrystalMount(
        lattice="cubic", a=2.8665e-10, structure_type="bcc", slip_families=["{110}<111>"]
    )
    planes, _, _, _ = _resolve_identify_planes_and_burgers(mount)
    # 6 distinct {110} normals, no {112}.
    assert len(planes) == 6
    for p in planes:
        assert sorted(abs(c) for c in p) == [0, 1, 1]


def test_draw_dislocation_nonfcc_branch_reachable():
    """The non-FCC _draw_dislocation branch draws an in-sweep plane and yields a
    ⟨111⟩ integer Burgers (registry), not a ⟨110⟩×√2 vector."""
    mount = CrystalMount(lattice="cubic", a=2.8665e-10, structure_type="bcc", material="Fe")
    planes, burgers_fn, burgers_int_fn, _ = _resolve_identify_planes_and_burgers(mount)
    rng = np.random.default_rng(0)
    d = _draw_dislocation(
        rng, 1.0, planes=planes, burgers_fn=burgers_fn, burgers_int_fn=burgers_int_fn
    )
    assert tuple(d["plane"]) in [tuple(p) for p in planes]
    assert d["Ud"].shape == (3, 3)
    bvi = [int(round(c)) for c in d["b_vec"]]  # registry integer Burgers
    assert sorted(abs(c) for c in bvi) == [1, 1, 1]


def test_draw_dislocation_fcc_default_is_v2x_byte_path():
    """A bare _draw_dislocation (no triple) keeps the v2.x FCC stream: it draws
    from _ALL_111_PLANES (modulus 4) and b_vec = unit × √2 EXACTLY (not rounded)."""
    rng_a = np.random.default_rng(123)
    d = _draw_dislocation(rng_a, 0.0)
    assert tuple(d["plane"]) in [tuple(p) for p in _ALL_111_PLANES]
    bt = _burgers_vectors(d["plane"])
    assert np.array_equal(d["b_vec"], bt[d["b_idx"]] * np.sqrt(2))  # exact float, not int-rounded
