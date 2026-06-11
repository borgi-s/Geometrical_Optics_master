"""G1.1/G1.2 (roadmap M2): oblique identify integration tests, all three modes.

Real-compute e2e on the ANALYTIC resolution backend (beamstop=false, no MC
kernel npz needed — the same trick as Gate C test_oblique_forward_contrast).
Geometry: the paper Fig 3B reflection (-1,-1,3) @ 19.1 keV, eta=0.353140 rad,
theta ~ 15.417 deg (Table A.2). Marked `slow` (module-default grid, ~tens of
seconds per frame); run with `pytest -m slow`.

Each test asserts the M2 DoD: the run completes, the master + per-scan layout
exists, and the oblique geometry round-trips through the per-scan /N.1 attrs
(geometry_mode/eta/theta/mount — written by _identify_geometry_attrs since M2).

Discriminating oblique from simplified fallback
-----------------------------------------------
The theta assertion (~15.417 deg) is a sanity check that ctx threading worked
end-to-end (the value differs from the import-time default), NOT a fallback
guard: run_theta returns ~15.417 deg for this reflection in BOTH oblique and
simplified modes (same physics, different coordinate treatment).

The actual oblique-vs-simplified discriminators are:
  - ``geometry_mode == "oblique"`` in the /N.1 attrs
  - ``eta == 0.353140`` (eta is 0.0 in simplified mode)

Both are asserted by ``_assert_oblique_scan_attrs``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm_geo.pipeline import (
    load_identification_config,
    run_identification,
)

_OBLIQUE_ETA = 0.353140
_OBLIQUE_THETA = np.deg2rad(15.417)

# Identify-config TOML for the paper oblique geometry. Landmines (recon
# 2026-06-11): mount keys live in [crystal] top level (stripped before
# IdentificationCrystalConfig); lattice_a must match the mount's a;
# beamstop/aperture false -> analytic backend (no kernel npz on disk needed).
_OBLIQUE_E2E_TOML_SINGLE = (
    'mode = "single"\n'
    "\n"
    "[crystal]\n"
    "slip_plane_normal = [1, 1, -1]\n"
    "angle_start_deg = 0.0\n"
    "angle_stop_deg = 0.0\n"
    "angle_step_deg = 10.0\n"
    "b_vector_indices = [0]\n"
    "sweep_all_slip_planes = false\n"
    "exclude_invisibility = false\n"
    'lattice = "cubic"\n'
    "a = 4.0493e-10\n"
    "mount_x = [1, 0, 0]\n"
    "mount_y = [0, 1, 0]\n"
    "mount_z = [0, 0, 1]\n"
    "\n"
    "[geometry]\n"
    'mode = "oblique"\n'
    f"eta = {_OBLIQUE_ETA}\n"
    "\n"
    "[reciprocal]\n"
    "hkl = [-1, -1, 3]\n"
    "keV = 19.1\n"
    "lattice_a = 4.0493e-10\n"
    "beamstop = false\n"
    "aperture = false\n"
    "\n"
    "[scan.phi]\n"
    "value = 0.46e-3\n"
    "\n"
    "[scan.chi]\n"
    "value = 0.067e-3\n"
    "\n"
    "[scan.two_dtheta]\n"
    "value = -0.42e-3\n"
)


def _assert_oblique_scan_attrs(master_path: Path) -> list[str]:
    """Shared DoD assertion: every /N.1 carries the oblique geometry attrs.

    Checks three things:
    1. ``geometry_mode == "oblique"`` — primary fallback guard; simplified
       mode would write "simplified" here.
    2. ``eta == 0.353140`` (atol=1e-6) — secondary fallback guard; simplified
       mode uses eta=0.0. The tight tolerance confirms exact pass-through of
       the config value with no lossy conversion.
    3. ``theta ~= 15.417 deg`` — sanity check that ctx threading reached the
       attr writer (theta differs from the import-time default); NOT a
       fallback discriminator (simplified mode produces the same theta for
       this reflection).
    4. All three mount vectors and lattice/a — confirm crystal ctx threading.

    Returns the scan ids for further per-mode assertions."""
    with h5py.File(master_path, "r") as f:
        scan_ids = [k for k in f if k != "dfxm_geo"]
        assert scan_ids, "identify run produced no scans"
        for sid in scan_ids:
            attrs = f[sid].attrs
            # --- primary oblique-vs-simplified discriminators ---
            assert attrs["geometry_mode"] == "oblique"
            assert np.isclose(float(attrs["eta"]), _OBLIQUE_ETA, atol=1e-6)
            # --- ctx-threading sanity (same value in both modes for this hkl) ---
            assert np.isclose(float(attrs["theta"]), _OBLIQUE_THETA, atol=1e-3)
            # --- crystal ctx threading: all mount vectors + lattice ---
            np.testing.assert_array_equal(attrs["mount_x"], [1, 0, 0])
            np.testing.assert_array_equal(attrs["mount_y"], [0, 1, 0])
            np.testing.assert_array_equal(attrs["mount_z"], [0, 0, 1])
            assert attrs["lattice"] == "cubic"
            assert np.isclose(float(attrs["a"]), 4.0493e-10)
    return scan_ids


@pytest.mark.slow
def test_identify_single_oblique_e2e(tmp_path: Path) -> None:
    """G1.1 (single): oblique identify runs end-to-end on the analytic
    backend and round-trips oblique provenance (attrs + embedded TOML)."""
    import tomllib

    cfg_path = tmp_path / "oblique_identify.toml"
    cfg_path.write_text(_OBLIQUE_E2E_TOML_SINGLE)
    cfg = load_identification_config(cfg_path)
    assert cfg.geometry.mode == "oblique"  # precondition
    assert cfg.reciprocal.eta == pytest.approx(_OBLIQUE_ETA, abs=1e-6)

    out = tmp_path / "out"
    run_identification(cfg, out)

    master = out / "dfxm_identify.h5"
    assert master.is_file()
    # 1 plane x 1 b x 1 angle = 1 scan
    assert (out / "scan0001" / "dfxm_sim_detector_0000.h5").is_file()
    scan_ids = _assert_oblique_scan_attrs(master)
    assert scan_ids == ["1.1"]

    with h5py.File(master, "r") as f:
        # detector frame actually computed (non-degenerate physics)
        img = f["/1.1/instrument/dfxm_sim_detector/data"][0].astype(np.float64)
        assert img.max() > 0.0
        assert np.isfinite(img).all()
        assert img.min() >= 0.0
        # embedded config TOML round-trips oblique (pre-M2 the ONLY provenance)
        toml_str = f["/dfxm_geo/config_toml"][()].decode()
    parsed = tomllib.loads(toml_str)
    assert parsed["geometry"]["mode"] == "oblique"
    assert np.isclose(parsed["geometry"]["eta"], _OBLIQUE_ETA, atol=1e-6)
