"""M4 Stage 4.3a Task 12: BCC end-to-end DoD — the Stage-4.3a integration gate.

These tests actually RUN the pipeline (forward + identify) on a BCC structure
and prove the whole stage composes: the [crystal] structure_type / cif route
reaches the population builder, the registry-driven slip systems + cell-derived
|b| flow through to the detector image, and the BCC provenance round-trips to
HDF5.

Runs on the ANALYTIC resolution backend (oblique mode, beamstop=false) so no MC
kernel npz needs bootstrapping for a BCC reflection — the same kernel-free trick
the M2 oblique e2e tests use (test_oblique_forward_contrast.py,
test_identification_oblique_e2e.py). Centered / single-mode + a single frame
keeps each run ~1 s even at the module-default grid.

Geometry: Fe BCC (a = 2.8665 Å, Im-3m), reflection (1, 1, 0) — NOT extinct for
Im-3m (h+k+l even) — at 17 keV with the cubic identity mount.  For this mount
the reflection's oblique angle is η = ±π/2 (from compute_omega_eta); the config
must pass that exact η or the [geometry] validator rejects it.

The DoD: BCC forward AND BCC identify both run end-to-end and carry BCC-specific
physics (|b| = a√3/2, NOT the FCC a/√2; a ⟨111⟩ Burgers; a {110}/{112} slip
plane from the registry).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm_geo.crystal.slip_systems import plane_normals
from dfxm_geo.pipeline import (
    SimulationConfig,
    load_identification_config,
    run_identification,
    run_simulation,
)

# Fe BCC lattice parameter (m) and the valid oblique angle for (1,1,0) at 17 keV
# with the cubic identity mount: η₂ = +π/2 (from compute_omega_eta).
_FE_A = 2.8665e-10
_BCC_HKL = (1, 1, 0)
_BCC_KEV = 17.0
_BCC_ETA = float(np.pi / 2)  # = 1.5707963267948966, the reflection's computed η₂

# Expected BCC {110}<111> Burgers magnitude in µm: |b| = a·√3/2.
_EXPECTED_B_UM = _FE_A * np.sqrt(3) / 2 * 1e6  # ≈ 2.482e-4
# The historical FCC constant (a/√2 for Al) the BCC value must NOT collapse to.
_FCC_B_UM = 2.862e-4


def _canon(v: tuple[int, int, int]) -> tuple[int, int, int]:
    """Canonical sign: leading nonzero positive (so v and -v collapse).

    Matches dfxm_geo.crystal.slip_systems._canon so HDF5-read normals compare
    equal to the registry's plane_normals output.
    """
    for c in v:
        if c != 0:
            return v if c > 0 else tuple(-x for x in v)  # type: ignore[return-value]
    return v


_BCC_PLANES_CANON = {_canon(p) for p in plane_normals("bcc")}


def _bcc_forward_toml() -> str:
    """Centered single-frame BCC forward on the analytic backend (no kernel)."""
    return (
        "[reciprocal]\n"
        f"hkl = [{_BCC_HKL[0]}, {_BCC_HKL[1]}, {_BCC_HKL[2]}]\n"
        f"keV = {_BCC_KEV}\n"
        'backend = "analytic"\n'
        "beamstop = false\n"
        "\n"
        "[geometry]\n"
        'mode = "oblique"\n'
        f"eta = {_BCC_ETA!r}\n"
        "\n"
        "[crystal]\n"
        'lattice = "cubic"\n'
        f"a = {_FE_A!r}\n"
        'structure_type = "bcc"\n'
        'material = "Fe"\n'
        "mount_x = [1, 0, 0]\n"
        "mount_y = [0, 1, 0]\n"
        "mount_z = [0, 0, 1]\n"
        'mode = "centered"\n'
        "\n"
        "[crystal.centered]\n"
        # BCC {110}<111> glide system: b·n = 1·1 + (-1)·1 + 1·0 = 0 ✓
        "b = [1, -1, 1]\n"
        "n = [1, 1, 0]\n"
        "t = [1, -1, -2]\n"
        "\n"
        "[scan.phi]\n"
        "value = 0.0\n"
        "\n"
        "[io]\n"
        "include_perfect_crystal = false\n"
        "write_strain_provenance = false\n"
        "\n"
        "[postprocess]\n"
        "enabled = false\n"
    )


@pytest.mark.slow
def test_bcc_forward_runs(tmp_path: Path) -> None:
    """BCC structure_type forward runs end-to-end at a non-extinct reflection.

    DoD half 1: the [crystal] structure_type='bcc' mount reaches the
    population builder + detector writer; a finite image is produced and the
    run used the cell-derived BCC |b| (a√3/2), NOT the FCC constant.
    """
    cfg_path = tmp_path / "bcc_forward.toml"
    cfg_path.write_text(_bcc_forward_toml(), encoding="utf-8")
    cfg = SimulationConfig.from_toml(cfg_path)
    assert cfg.geometry.mount is not None
    assert cfg.geometry.mount.resolved_structure_type == "bcc"  # precondition

    out = tmp_path / "out"
    run_simulation(cfg, out)

    # Detector image written, finite, correct shape.
    det = out / "scan0001" / "dfxm_sim_detector_0000.h5"
    assert det.is_file()
    with h5py.File(det, "r") as f:
        img = f["/entry_0000/dfxm_sim_detector/image"][...]
    assert img.ndim == 3 and img.shape[0] == 1  # (n_frames, H, W)
    assert np.isfinite(img).all()
    assert float(img.max()) > 0.0

    # Provenance proves BCC physics: |b| = a√3/2, structure_type=bcc.
    with h5py.File(out / "dfxm_geo.h5", "r") as f:
        attrs = dict(f["/1.1"].attrs)
    assert attrs["structure_type"] == "bcc"
    assert attrs["material"] == "Fe"
    b_um = float(attrs["burgers_magnitude_um"])
    assert np.isclose(b_um, _EXPECTED_B_UM, rtol=1e-4), (
        f"BCC |b|={b_um} should be a√3/2={_EXPECTED_B_UM}"
    )
    assert not np.isclose(b_um, _FCC_B_UM, rtol=1e-3), (
        f"BCC |b|={b_um} collapsed to the FCC constant {_FCC_B_UM}"
    )


def _bcc_identify_toml() -> str:
    """Single-mode BCC identify on the analytic backend (no kernel)."""
    return (
        'mode = "single"\n'
        "\n"
        "[crystal]\n"
        f"slip_plane_normal = [{_BCC_HKL[0]}, {_BCC_HKL[1]}, {_BCC_HKL[2]}]\n"
        "angle_start_deg = 0.0\n"
        "angle_stop_deg = 0.0\n"
        "angle_step_deg = 10.0\n"
        "b_vector_indices = [0]\n"
        "sweep_all_slip_planes = false\n"
        "exclude_invisibility = false\n"
        'lattice = "cubic"\n'
        f"a = {_FE_A!r}\n"
        'structure_type = "bcc"\n'
        'material = "Fe"\n'
        "mount_x = [1, 0, 0]\n"
        "mount_y = [0, 1, 0]\n"
        "mount_z = [0, 0, 1]\n"
        "\n"
        "[geometry]\n"
        'mode = "oblique"\n'
        f"eta = {_BCC_ETA!r}\n"
        "\n"
        "[reciprocal]\n"
        f"hkl = [{_BCC_HKL[0]}, {_BCC_HKL[1]}, {_BCC_HKL[2]}]\n"
        f"keV = {_BCC_KEV}\n"
        f"lattice_a = {_FE_A!r}\n"
        "beamstop = false\n"
        "aperture = false\n"
        "\n"
        "[scan.phi]\n"
        "value = 1e-4\n"
    )


@pytest.mark.slow
def test_bcc_identify_has_bcc_slip_labels(tmp_path: Path) -> None:
    """BCC identify single-mode runs end-to-end and records BCC slip labels.

    DoD half 2: the master records a slip_plane_normal that is one of
    plane_normals('bcc') (canonicalized for the compare), structure_type='bcc'
    provenance is present, and >=1 scan was written.
    """
    cfg_path = tmp_path / "bcc_identify.toml"
    cfg_path.write_text(_bcc_identify_toml(), encoding="utf-8")
    cfg = load_identification_config(cfg_path)
    assert cfg.geometry.mount is not None
    assert cfg.geometry.mount.resolved_structure_type == "bcc"  # precondition

    out = tmp_path / "out"
    run_identification(cfg, out)

    master = out / "dfxm_identify.h5"
    assert master.is_file()
    with h5py.File(master, "r") as f:
        scan_ids = sorted(k for k in f if k != "dfxm_geo")
        assert len(scan_ids) >= 1, "BCC identify produced no scans"
        for sid in scan_ids:
            scan = f[sid]
            assert scan.attrs["structure_type"] == "bcc"
            assert scan.attrs["geometry_mode"] == "oblique"
            samp = scan["sample"]
            spn = samp["slip_plane_normal"][()]
            spn_int = tuple(int(round(float(c))) for c in spn)
            assert _canon(spn_int) in _BCC_PLANES_CANON, (
                f"slip_plane_normal {spn_int} (canon {_canon(spn_int)}) is not a BCC slip plane"
            )
            # Burgers label is a registry ⟨111⟩ direction (not ⟨110⟩×√2).
            b_vec = samp["burgers"][()]
            b_int = sorted(abs(int(round(float(c)))) for c in b_vec)
            assert b_int == [1, 1, 1], f"BCC Burgers {b_vec} should be ⟨111⟩"


@pytest.mark.slow
def test_bcc_via_fe_cif(tmp_path: Path) -> None:
    """BCC derived from a minimal Fe Im-3m CIF (no explicit structure_type).

    Optional CIF-route DoD: a [crystal] cif pointing at an Fe Im-3m CIF, with
    NO structure_type asserted, must derive resolved_structure_type='bcc' from
    the space group and still produce a finite forward image.
    """
    pytest.importorskip("gemmi")

    cif_text = (
        "data_Fe\n"
        "_cell_length_a    2.8665\n"
        "_cell_length_b    2.8665\n"
        "_cell_length_c    2.8665\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta  90\n"
        "_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M   'I m -3 m'\n"
        "_space_group_IT_number           229\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "Fe1 0 0 0\n"
    )
    (tmp_path / "fe.cif").write_text(cif_text, encoding="utf-8")

    cfg_toml = (
        "[reciprocal]\n"
        f"hkl = [{_BCC_HKL[0]}, {_BCC_HKL[1]}, {_BCC_HKL[2]}]\n"
        f"keV = {_BCC_KEV}\n"
        'backend = "analytic"\n'
        "beamstop = false\n"
        "\n"
        "[geometry]\n"
        'mode = "oblique"\n'
        f"eta = {_BCC_ETA!r}\n"
        "\n"
        "[crystal]\n"
        'cif = "fe.cif"\n'  # NO explicit structure_type — derived from the space group
        "mount_x = [1, 0, 0]\n"
        "mount_y = [0, 1, 0]\n"
        "mount_z = [0, 0, 1]\n"
        'mode = "centered"\n'
        "\n"
        "[crystal.centered]\n"
        "b = [1, -1, 1]\n"
        "n = [1, 1, 0]\n"
        "t = [1, -1, -2]\n"
        "\n"
        "[scan.phi]\n"
        "value = 0.0\n"
        "\n"
        "[io]\n"
        "include_perfect_crystal = false\n"
        "write_strain_provenance = false\n"
        "\n"
        "[postprocess]\n"
        "enabled = false\n"
    )
    cfg_path = tmp_path / "fe.toml"
    cfg_path.write_text(cfg_toml, encoding="utf-8")
    cfg = SimulationConfig.from_toml(cfg_path)

    # The space group drives the structure family — no structure_type asserted.
    assert cfg.geometry.mount is not None
    assert cfg.geometry.mount.resolved_structure_type == "bcc"

    out = tmp_path / "out"
    run_simulation(cfg, out)

    det = out / "scan0001" / "dfxm_sim_detector_0000.h5"
    assert det.is_file()
    with h5py.File(det, "r") as f:
        img = f["/entry_0000/dfxm_sim_detector/image"][...]
    assert np.isfinite(img).all()
    assert float(img.max()) > 0.0

    with h5py.File(out / "dfxm_geo.h5", "r") as f:
        attrs = dict(f["/1.1"].attrs)
    assert attrs["structure_type"] == "bcc"
    assert np.isclose(float(attrs["burgers_magnitude_um"]), _EXPECTED_B_UM, rtol=1e-4)
    assert "space_group" in attrs
    assert attrs["space_group"]  # non-empty
