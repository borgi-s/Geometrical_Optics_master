"""2-reflection identify e2e smoke: per-reflection masters, aligned grids, RNG policy.

Smoke scale: angle_stop_deg=0 -> 1 angle point; b_vector_indices=[0,1] -> 2 b-vectors;
1 slip plane -> 2 total scans per reflection. analytic backend, no kernel npz needed.
"""

from __future__ import annotations

import h5py
import numpy as np
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

[detector]
model = "ideal"

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

# Same TOML but with the detector model enabled (fixed seed for determinism).
# Dropping the model="ideal" line lets the default (noisy) detector model apply.
_MULTI_IDENTIFY_NOISY_TOML = _MULTI_IDENTIFY_TOML.replace(
    'model = "ideal"',
    "rng_seed = 42",
)


def _write(tmp_path, content: str):
    p = tmp_path / "config.toml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cli_main_identify_multi_reflection_summary(tmp_path, capsys):
    """dfxm-identify must exit 0 on a [[reflections]] config and print an
    aggregate summary (regression: the CLI print assumed the single-reflection
    return shape and crashed with KeyError('n_images') AFTER all per-reflection
    masters were already written — caught on the cluster, 2026-06-11)."""
    from dfxm_geo.pipeline import cli_main_identify

    p = _write(tmp_path, _MULTI_IDENTIFY_TOML)
    out = tmp_path / "out"
    rc = cli_main_identify(["--config", str(p), "--output", str(out)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "2 reflections" in captured
    # the per-reflection masters really exist (the CLI didn't just exit early)
    for idx in (1, 2):
        assert (out / f"reflection_{idx:03d}" / "dfxm_identify.h5").is_file()


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
    assert scan_counts[0] > 0, "all configs excluded - gate may be over-aggressive"
    assert scan_counts[0] == scan_counts[1], (
        f"All-reflections invisibility gate produced mismatched grids: "
        f"reflection_001={scan_counts[0]}, reflection_002={scan_counts[1]}"
    )


@pytest.mark.slow
def test_detector_noise_determinism_and_per_reflection_independence(tmp_path):
    """Detector-model noise is deterministic (fixed seed) and independent per reflection.

    Three assertions at the e2e level:

      1. Two noisy runs with the SAME rng_seed produce bit-for-bit identical
         uint16 detector frames for every reflection (seed → reproducibility).

      2. The two reflections within ONE noisy run produce DIFFERENT uint16
         frames for scan0001 — i.e. reflection_001 and reflection_002 receive
         independent noise draws. This follows from the
         ``default_rng([seed, reflection_index])`` contract: each reflection's
         noise RNG is seeded with a distinct tuple and cannot be identical to
         another reflection's stream.  The per-stream unit test lives in
         ``tests/test_apply_detector_model.py``.

      3. (Sanity) The ideal (noiseless) run produces float32 frames; both
         noisy runs produce uint16 frames — confirming the detector model was
         actually applied.

    Note: this test does NOT compare uint16 ADU to float32 normalized intensity
    (different scales) — all comparisons are between same-dtype uint16 arrays.
    """
    det_path = "scan0001/dfxm_sim_detector_0000.h5"
    det_internal = "/entry_0000/dfxm_sim_detector/image"

    def _read_frame(out_dir, reflection_idx):
        refl = f"reflection_{reflection_idx:03d}"
        with h5py.File(out_dir / refl / det_path, "r") as fh:
            return fh[det_internal][...]

    # --- noisy run A (seed=42) ---
    p_a = tmp_path / "noisy_a.toml"
    p_a.write_text(_MULTI_IDENTIFY_NOISY_TOML, encoding="utf-8")
    out_a = tmp_path / "out_a"
    run_identification(load_identification_config(p_a), out_a)

    # --- noisy run B (same TOML, same seed) ---
    p_b = tmp_path / "noisy_b.toml"
    p_b.write_text(_MULTI_IDENTIFY_NOISY_TOML, encoding="utf-8")
    out_b = tmp_path / "out_b"
    run_identification(load_identification_config(p_b), out_b)

    # Assertion 3: noisy frames are uint16; sanity-check ideal is float32.
    ideal_cfg = load_identification_config(_write(tmp_path, _MULTI_IDENTIFY_TOML))
    out_ideal = tmp_path / "out_ideal"
    run_identification(ideal_cfg, out_ideal)
    assert _read_frame(out_ideal, 1).dtype == np.float32, "ideal model must produce float32"
    assert _read_frame(out_a, 1).dtype == np.uint16, "noisy model must produce uint16"

    # Assertion 1: determinism — run A and run B are bit-for-bit identical.
    for idx in (1, 2):
        frame_a = _read_frame(out_a, idx)
        frame_b = _read_frame(out_b, idx)
        assert np.array_equal(frame_a, frame_b), (
            f"reflection_{idx:03d}: same-seed runs produced different uint16 frames — "
            "detector noise RNG is not deterministic"
        )

    # Assertion 2: per-reflection independence — refl_001 != refl_002 within run A.
    frame_r1 = _read_frame(out_a, 1)
    frame_r2 = _read_frame(out_a, 2)
    assert not np.array_equal(frame_r1, frame_r2), (
        "reflection_001 and reflection_002 produced identical uint16 frames — "
        "per-reflection RNG streams are not independent "
        "(expected default_rng([seed, reflection_index]) to differ)"
    )
