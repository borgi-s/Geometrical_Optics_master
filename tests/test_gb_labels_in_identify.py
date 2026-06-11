"""Task 5: g·b labels (gb_cos, gb_visible) in identify HDF5.

Tests:
  (a) Single-mode tiny run: every /N.1/dfxm_geo has gb_cos and gb_visible;
      gb_cos matches a recomputation from the stored burgers vector + q_hkl.
  (b) 2-reflection run: labels DIFFER across reflection masters for the same
      scan index (different q_hkl → different gb_cos).

Analytic backend throughout (no kernel npz needed → CI-safe).
Smoke scale: 1 slip plane, 2 b-vectors, 1 alpha point = 2 scans per reflection.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from dfxm_geo.crystal.burgers import gb_cos as _gb_cos
from dfxm_geo.crystal.burgers import gb_visible as _gb_visible
from dfxm_geo.pipeline import load_identification_config, run_identification

# ---------------------------------------------------------------------------
# Single-reflection TOML (no [[reflections]], hkl explicit)
# ---------------------------------------------------------------------------
_SINGLE_TOML = """
[reciprocal]
keV = 19.1
backend = "analytic"
beamstop = false
hkl = [1, 1, 3]

[geometry]
mode = "simplified"

[crystal]
lattice = "cubic"
a       = 4.0493e-10
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
slip_plane_normal   = [1, 1, 1]
angle_start_deg     = 0.0
angle_stop_deg      = 0.0
angle_step_deg      = 10.0
b_vector_indices    = [0, 1]
sweep_all_slip_planes = false
exclude_invisibility  = false

[noise]
poisson_noise = false

[identification]

[scan]
[scan.phi]
value = 0.0
range = 1.25e-4
steps = 3
"""

# ---------------------------------------------------------------------------
# Multi-reflection TOML (two [[reflections]], same tiny sweep)
# ---------------------------------------------------------------------------
_MULTI_REFLECT_TOML = """
[reciprocal]
keV = 19.1
backend = "analytic"
beamstop = false

[geometry]
mode = "oblique"

[crystal]
lattice = "cubic"
a       = 4.0493e-10
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
slip_plane_normal   = [1, 1, 1]
angle_start_deg     = 0.0
angle_stop_deg      = 0.0
angle_step_deg      = 10.0
b_vector_indices    = [0, 1]
sweep_all_slip_planes = false
exclude_invisibility  = false

[noise]
poisson_noise = false

[identification]

[scan]
[scan.phi]
value = 0.0
range = 1.25e-4
steps = 3

[[reflections]]
hkl = [1, 1, 3]
[[reflections]]
hkl = [-1, -1, 3]
eta = 0.3531
"""


def _write(tmp_path, content: str):
    p = tmp_path / "config.toml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# (a) Single-reflection: labels present + recomputable
# ---------------------------------------------------------------------------


def test_single_mode_gb_labels_present(tmp_path):
    """Every /N.1/dfxm_geo must contain gb_cos and gb_visible datasets."""
    cfg = load_identification_config(_write(tmp_path, _SINGLE_TOML))
    out = tmp_path / "out"
    run_identification(cfg, out)

    with h5py.File(out / "dfxm_identify.h5", "r") as fh:
        scans = sorted(k for k in fh if k[0].isdigit())
        assert scans, "No scan groups found"
        for s in scans:
            grp = fh[s]["dfxm_geo"]
            assert "gb_cos" in grp, f"{s}/dfxm_geo missing gb_cos"
            assert "gb_visible" in grp, f"{s}/dfxm_geo missing gb_visible"


def test_single_mode_gb_cos_recomputable(tmp_path):
    """gb_cos stored in HDF5 == recomputed from the scan's stored q_hkl + burgers."""
    cfg = load_identification_config(_write(tmp_path, _SINGLE_TOML))
    out = tmp_path / "out"
    run_identification(cfg, out)

    with h5py.File(out / "dfxm_identify.h5", "r") as fh:
        scans = sorted(k for k in fh if k[0].isdigit())
        for s in scans:
            stored_gb_cos = float(fh[s]["dfxm_geo"]["gb_cos"][()])
            # q_hkl is stored in dfxm_geo; burgers (integer indices) in sample
            q_hkl = fh[s]["dfxm_geo"]["q_hkl"][...]
            burgers = fh[s]["sample"]["burgers"][...]  # integer [h, k, l] * sqrt(2)
            recomputed = _gb_cos(q_hkl, burgers.astype(float))
            assert stored_gb_cos == pytest.approx(recomputed, abs=1e-12), (
                f"{s}: stored gb_cos {stored_gb_cos} != recomputed {recomputed}"
            )


def test_single_mode_gb_visible_consistent_with_threshold(tmp_path):
    """gb_visible matches gb_visible(q_hkl, burgers, threshold_deg)."""
    cfg = load_identification_config(_write(tmp_path, _SINGLE_TOML))
    out = tmp_path / "out"
    run_identification(cfg, out)
    threshold = cfg.crystal.invisibility_threshold_deg

    with h5py.File(out / "dfxm_identify.h5", "r") as fh:
        scans = sorted(k for k in fh if k[0].isdigit())
        for s in scans:
            stored_gb_visible = int(fh[s]["dfxm_geo"]["gb_visible"][()])
            q_hkl = fh[s]["dfxm_geo"]["q_hkl"][...]
            burgers = fh[s]["sample"]["burgers"][...]
            expected = int(_gb_visible(q_hkl, burgers.astype(float), threshold))
            assert stored_gb_visible == expected, (
                f"{s}: stored gb_visible {stored_gb_visible} != expected {expected}"
            )


# ---------------------------------------------------------------------------
# (b) 2-reflection: labels differ across reflections for same scan index
# ---------------------------------------------------------------------------


def test_two_reflection_gb_cos_differs_across_reflections(tmp_path):
    """For the same scan index, gb_cos must differ between reflection masters
    because q_hkl differs ([1,1,3] vs [-1,-1,3]).
    """
    cfg = load_identification_config(_write(tmp_path, _MULTI_REFLECT_TOML))
    out = tmp_path / "out"
    run_identification(cfg, out)

    gb_cos_per_reflection: list[list[float]] = []
    for idx in (1, 2):
        master = out / f"reflection_{idx:03d}" / "dfxm_identify.h5"
        assert master.is_file(), f"Missing {master}"
        with h5py.File(master, "r") as fh:
            scans = sorted(k for k in fh if k[0].isdigit())
            vals = [float(fh[s]["dfxm_geo"]["gb_cos"][()]) for s in scans]
            gb_cos_per_reflection.append(vals)

    # Both reflections must have the same number of scans (aligned grid)
    assert len(gb_cos_per_reflection[0]) == len(gb_cos_per_reflection[1])

    # For at least one scan, gb_cos must differ between the two reflections
    # (q_hkl=[1,1,3] vs q_hkl=[-1,-1,3] may give same |cos| for symmetric b —
    #  but with offset eta the projections differ enough at least in sign; we
    #  test the vectorial value from q_hkl after normalisation).
    # A stronger check: labels are recomputed from the run's q_hkl, which IS
    # different. We verify that both reflections store the CORRECT q_hkl and
    # then their gb_cos values reflect it.
    q_hkl_per_reflection: list[list[np.ndarray]] = []
    for idx in (1, 2):
        master = out / f"reflection_{idx:03d}" / "dfxm_identify.h5"
        with h5py.File(master, "r") as fh:
            scans = sorted(k for k in fh if k[0].isdigit())
            qs = [fh[s]["dfxm_geo"]["q_hkl"][...] for s in scans]
            q_hkl_per_reflection.append(qs)

    # The q_hkl stored in reflection_001 vs reflection_002 must differ
    for s_idx in range(len(q_hkl_per_reflection[0])):
        q1 = q_hkl_per_reflection[0][s_idx]
        q2 = q_hkl_per_reflection[1][s_idx]
        # Unit vectors from [1,1,3] and [-1,-1,3] must differ
        assert not np.allclose(q1, q2), (
            f"Scan {s_idx}: q_hkl is identical across reflections ({q1} == {q2})"
        )

    # And gb_cos, being derived from the specific q_hkl of each reflection,
    # must also differ for at least one b-vector direction.
    any_differ = any(
        abs(v1 - v2) > 1e-10
        for v1, v2 in zip(gb_cos_per_reflection[0], gb_cos_per_reflection[1], strict=True)
    )
    assert any_differ, (
        "gb_cos is identical across both reflections for all scans — "
        "expected at least one scan to differ (different q_hkl)"
    )
