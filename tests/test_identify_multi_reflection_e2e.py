"""2-reflection identify e2e smoke: per-reflection masters, aligned grids, RNG policy.

Smoke scale: angle_stop_deg=0 -> 1 angle point; b_vector_indices=[0,1] -> 2 b-vectors;
1 slip plane -> 2 total scans per reflection. analytic backend, no kernel npz needed.
"""

from __future__ import annotations

import h5py
import pytest

from dfxm_geo.pipeline import load_identification_config, run_identification

# ---------------------------------------------------------------------------
# Minimal multi-reflection identify TOML (analytic backend, tiny sweep)
# ---------------------------------------------------------------------------
_MULTI_IDENTIFY_TOML = """
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

# Same TOML but with exclude_invisibility=true
_MULTI_IDENTIFY_EXCL_TOML = _MULTI_IDENTIFY_TOML.replace(
    "exclude_invisibility  = false", "exclude_invisibility  = true"
)


def _write(tmp_path, content: str):
    p = tmp_path / "config.toml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_two_reflection_identify_layout(tmp_path):
    """result n_reflections==2, both reflection_NNN/dfxm_identify.h5 exist,
    super-master dfxm_identify_multi.h5 exists."""
    cfg = load_identification_config(_write(tmp_path, _MULTI_IDENTIFY_TOML))
    out = tmp_path / "out"
    result = run_identification(cfg, out)
    assert result["n_reflections"] == 2
    for idx in (1, 2):
        master = out / f"reflection_{idx:03d}" / "dfxm_identify.h5"
        assert master.is_file(), f"Missing {master}"
    assert (out / "dfxm_identify_multi.h5").is_file()


def test_same_scan_count_across_reflections(tmp_path):
    """Both reflection masters must contain the same number of scans (aligned grid)."""
    cfg = load_identification_config(_write(tmp_path, _MULTI_IDENTIFY_TOML))
    out = tmp_path / "out"
    run_identification(cfg, out)
    scan_counts = []
    for idx in (1, 2):
        with h5py.File(out / f"reflection_{idx:03d}" / "dfxm_identify.h5", "r") as fh:
            scans = [k for k in fh if k[0].isdigit()]
            scan_counts.append(len(scans))
    assert scan_counts[0] == scan_counts[1], (
        f"Scan counts differ: reflection_001 has {scan_counts[0]}, "
        f"reflection_002 has {scan_counts[1]}"
    )


def test_identical_sample_content_across_reflections(tmp_path):
    """Per-scan sample metadata must be IDENTICAL across both reflection masters
    (shared param RNG -> same crystal realization in every reflection).

    We compare slip_plane_normal, burgers, and rotation_deg for each scan index.
    """
    cfg = load_identification_config(_write(tmp_path, _MULTI_IDENTIFY_TOML))
    out = tmp_path / "out"
    run_identification(cfg, out)

    per_reflection_samples: list[list[dict]] = []
    for idx in (1, 2):
        master = out / f"reflection_{idx:03d}" / "dfxm_identify.h5"
        scans_meta: list[dict] = []
        with h5py.File(master, "r") as fh:
            scans = sorted(k for k in fh if k[0].isdigit())
            for s in scans:
                samp = fh[s]["sample"]
                entry = {
                    "slip_plane_normal": samp["slip_plane_normal"][...].tolist(),
                    "burgers": samp["burgers"][...].tolist(),
                    "rotation_deg": float(samp["rotation_deg"][()]),
                }
                scans_meta.append(entry)
        per_reflection_samples.append(scans_meta)

    # Both reflections must have identical sample params (same crystal)
    assert len(per_reflection_samples[0]) == len(per_reflection_samples[1])
    for k, (s0, s1) in enumerate(
        zip(per_reflection_samples[0], per_reflection_samples[1], strict=True)
    ):
        assert s0["slip_plane_normal"] == s1["slip_plane_normal"], (
            f"scan {k}: slip_plane_normal differs"
        )
        assert s0["burgers"] == s1["burgers"], f"scan {k}: burgers differs"
        assert s0["rotation_deg"] == pytest.approx(s1["rotation_deg"]), (
            f"scan {k}: rotation_deg differs"
        )


def test_per_scan_reflection_attrs_present(tmp_path):
    """Each scan's /N.1 group must carry reflection_index and n_reflections attrs."""
    cfg = load_identification_config(_write(tmp_path, _MULTI_IDENTIFY_TOML))
    out = tmp_path / "out"
    run_identification(cfg, out)

    for idx in (1, 2):
        with h5py.File(out / f"reflection_{idx:03d}" / "dfxm_identify.h5", "r") as fh:
            scans = sorted(k for k in fh if k[0].isdigit())
            assert scans, f"reflection_{idx:03d} has no scans"
            for s in scans:
                grp_attrs = dict(fh[s].attrs)
                assert "reflection_index" in grp_attrs, f"{s}: missing reflection_index"
                assert "n_reflections" in grp_attrs, f"{s}: missing n_reflections"
                assert grp_attrs["reflection_index"] == idx, f"{s}: wrong reflection_index"
                assert grp_attrs["n_reflections"] == 2, f"{s}: wrong n_reflections"


def test_super_master_links_and_table(tmp_path):
    """Super-master must link to each reflection's root and carry the table."""
    cfg = load_identification_config(_write(tmp_path, _MULTI_IDENTIFY_TOML))
    out = tmp_path / "out"
    run_identification(cfg, out)
    super_master = out / "dfxm_identify_multi.h5"
    assert super_master.is_file()
    with h5py.File(super_master, "r") as fh:
        # ExternalLinks resolve to each reflection's root
        for key in ("reflection_001", "reflection_002"):
            assert key in fh, f"Missing {key} in super-master"
        # Table present
        assert "reflections" in fh
        assert fh["reflections"]["hkl"].shape == (2, 3)
        assert fh.attrs["n_reflections"] == 2


def test_exclude_invisibility_same_scan_count_across_reflections(tmp_path):
    """With exclude_invisibility=true, the all-reflections gate must produce
    the SAME scan count in both reflection masters (aligned grid invariant).

    In a 2-reflection config, a (plane,b) pair is kept if it is visible to
    AT LEAST ONE reflection, so both masters always contain the same set.
    """
    cfg = load_identification_config(_write(tmp_path, _MULTI_IDENTIFY_EXCL_TOML))
    out = tmp_path / "out"
    run_identification(cfg, out)
    scan_counts = []
    for idx in (1, 2):
        with h5py.File(out / f"reflection_{idx:03d}" / "dfxm_identify.h5", "r") as fh:
            scans = [k for k in fh if k[0].isdigit()]
            scan_counts.append(len(scans))
    assert scan_counts[0] == scan_counts[1], (
        f"All-reflections invisibility gate produced mismatched grids: "
        f"reflection_001={scan_counts[0]}, reflection_002={scan_counts[1]}"
    )
