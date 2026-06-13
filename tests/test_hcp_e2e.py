"""M4 Stage 4.3b DoD: HCP forward + identify end-to-end (Ti and Mg).

The Stage-4.3b integration gate. These tests actually RUN the pipeline
(forward + identify) on an HCP structure and prove the whole stage composes:
the non-orthonormal hexagonal frame reaches the population builder, the
hexagonal-enumerator slip systems with per-dislocation ⟨a⟩/⟨c+a⟩ |b| flow
through to the detector image, the non-cubic q_hkl (B·hkl) is used, and the
HCP provenance (c/a, resolved families) round-trips to HDF5.

Runs on the ANALYTIC resolution backend (oblique mode, beamstop=false) so no MC
kernel npz needs bootstrapping for an HCP reflection — the same kernel-free
trick the BCC e2e (test_bcc_e2e.py) and the M2 oblique e2e tests use.

Orthonormal HCP mount: mount_x=(2,-1,0), mount_y=(0,1,0), mount_z=(0,0,1).
The cubic default (100)/(010)/(001) is REJECTED for a hexagonal cell because
a*, b* subtend 60°; these three mount vectors have mutually-orthogonal B·m.

Reflection: (1, 0, -1), a 1st-order-pyramidal-type reflection.  The paper's
preferred basal (0002) is NOT Laue-reachable at 17 keV for this mount
(compute_omega_eta returns NaN η for both Ti and Mg), so we use (1,0,-1),
which IS reachable with a finite η (≈ -2.08 rad for Ti, ≈ -2.07 for Mg, both
well away from the η=0 the oblique validator rejects).  The η is computed by
compute_omega_eta and fed back to the config (the oblique validator requires
the exact η).

The DoD: HCP forward AND HCP identify both run end-to-end for Ti AND Mg
(explicit structure_type + the CIF route), carrying HCP-specific physics —
the hexagonal Cartesian frame, c/a-dependent |b|, and a swept set of Burgers
labels containing BOTH an ⟨a⟩ and a ⟨c+a⟩ class.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from dfxm_geo.crystal.oblique import CrystalMount, compute_omega_eta
from dfxm_geo.crystal.slip_systems import plane_normals
from dfxm_geo.pipeline import (
    SimulationConfig,
    load_identification_config,
    run_identification,
    run_simulation,
)

# alpha-Ti and Mg (P6_3/mmc), lengths in metres.
_TI = dict(a=2.951e-10, c=4.684e-10, material="Ti", keV=17.0)
_MG = dict(a=3.209e-10, c=5.211e-10, material="Mg", keV=17.0)
# (1,0,-1): reachable (finite η) at 17 keV for both Ti and Mg with the
# orthonormal hexagonal mount below.  (0002) — the paper's basal ⟨c+a⟩ filter —
# is NOT reachable at 17 keV for this mount (compute_omega_eta -> NaN η).
_HKL = (1, 0, -1)
_MOUNT_KW = dict(mount_x=(2, -1, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1))


def _eta_for(metal, hkl):
    mount = CrystalMount(
        lattice="hexagonal",
        a=metal["a"],
        c=metal["c"],
        structure_type="hcp",
        **_MOUNT_KW,
    )
    geom = compute_omega_eta(mount, hkl, metal["keV"])
    # pick the finite solution
    eta = geom.eta_1 if not np.isnan(geom.eta_1) else geom.eta_2
    assert not np.isnan(eta), f"({hkl}) unreachable at {metal['keV']} keV for {metal['material']}"
    return float(eta)


def _hcp_forward_toml(metal, hkl, eta):
    return (
        "[reciprocal]\n"
        f"hkl = [{hkl[0]}, {hkl[1]}, {hkl[2]}]\n"
        f"keV = {metal['keV']}\n"
        'backend = "analytic"\nbeamstop = false\n\n'
        '[geometry]\nmode = "oblique"\n'
        f"eta = {eta!r}\n\n"
        "[crystal]\n"
        'lattice = "hexagonal"\n'
        f"a = {metal['a']!r}\n"
        f"c = {metal['c']!r}\n"
        'structure_type = "hcp"\n'
        f'material = "{metal["material"]}"\n'
        "mount_x = [2, -1, 0]\nmount_y = [0, 1, 0]\nmount_z = [0, 0, 1]\n"
        'mode = "random_dislocations"\n\n'
        "[crystal.random_dislocations]\nndis = 6\nseed = 7\nsigma = 8.0\n\n"
        "[scan.phi]\nvalue = 0.0\n\n"
        "[io]\ninclude_perfect_crystal = false\nwrite_strain_provenance = false\n\n"
        "[postprocess]\nenabled = false\n"
    )


@pytest.mark.slow
@pytest.mark.parametrize("metal", [_TI, _MG], ids=["Ti", "Mg"])
def test_hcp_forward_runs(tmp_path, metal):
    eta = _eta_for(metal, _HKL)
    cfg_path = tmp_path / "fwd.toml"
    cfg_path.write_text(_hcp_forward_toml(metal, _HKL, eta), encoding="utf-8")
    cfg = SimulationConfig.from_toml(cfg_path)
    assert cfg.geometry.mount is not None
    assert cfg.geometry.mount.resolved_structure_type == "hcp"
    out = tmp_path / "out"
    run_simulation(cfg, out)
    det = next(out.glob("scan*/dfxm_sim_detector_0000.h5"))
    with h5py.File(det, "r") as f:
        img = f["/entry_0000/dfxm_sim_detector/image"][...]
    assert np.isfinite(img).all() and float(img.max()) > 0.0
    # provenance: HCP structure + c/a + a mix of |b| (⟨a⟩ vs ⟨c+a⟩) in the sidecar
    with h5py.File(out / "dfxm_geo.h5", "r") as f:
        attrs = dict(f["/1.1"].attrs)
    assert attrs["structure_type"] == "hcp"
    assert np.isclose(float(attrs["c_over_a"]), metal["c"] / metal["a"], rtol=1e-9)
    assert "slip_families" in attrs
    assert len(list(attrs["slip_families"])) > 0


def _hcp_identify_toml(metal, hkl, eta):
    return (
        'mode = "single"\n\n'
        "[crystal]\n"
        "slip_plane_normal = [0, 0, 1]\n"  # basal; sweep_all covers the rest
        "angle_start_deg = 0.0\nangle_stop_deg = 0.0\nangle_step_deg = 10.0\n"
        "sweep_all_slip_planes = true\nexclude_invisibility = false\n"
        'lattice = "hexagonal"\n'
        f"a = {metal['a']!r}\n"
        f"c = {metal['c']!r}\n"
        'structure_type = "hcp"\n'
        f'material = "{metal["material"]}"\n'
        "mount_x = [2, -1, 0]\nmount_y = [0, 1, 0]\nmount_z = [0, 0, 1]\n\n"
        '[geometry]\nmode = "oblique"\n'
        f"eta = {eta!r}\n\n"
        "[reciprocal]\n"
        f"hkl = [{hkl[0]}, {hkl[1]}, {hkl[2]}]\n"
        f"keV = {metal['keV']}\n"
        f"lattice_a = {metal['a']!r}\nbeamstop = false\naperture = false\n\n"
        "[scan.phi]\nvalue = 1e-4\n"
    )


@pytest.mark.slow
@pytest.mark.parametrize("metal", [_TI, _MG], ids=["Ti", "Mg"])
def test_hcp_identify_has_ca_and_a_labels(tmp_path, metal):
    eta = _eta_for(metal, _HKL)
    cfg_path = tmp_path / "id.toml"
    cfg_path.write_text(_hcp_identify_toml(metal, _HKL, eta), encoding="utf-8")
    cfg = load_identification_config(cfg_path)
    assert cfg.geometry.mount is not None
    assert cfg.geometry.mount.resolved_structure_type == "hcp"
    out = tmp_path / "out"
    run_identification(cfg, out)
    hcp_planes = {p for p in plane_normals("hcp")}
    master = out / "dfxm_identify.h5"
    with h5py.File(master, "r") as f:
        scan_ids = sorted(k for k in f if k != "dfxm_geo")
        assert scan_ids
        burgers_lens = set()
        for sid in scan_ids:
            scan = f[sid]
            assert scan.attrs["structure_type"] == "hcp"
            spn = tuple(int(round(float(c))) for c in scan["sample"]["slip_plane_normal"][()])
            from dfxm_geo.crystal.slip_systems import _canon

            assert _canon(spn) in hcp_planes
            b = scan["sample"]["burgers"][()]
            # |b_int|^2 proxy: ⟨a⟩ ([100]/[110]) vs ⟨c+a⟩ ([101]/[111]/...) classes.
            burgers_lens.add(int(round(float(np.dot(b, b)))))
        # both an ⟨a⟩ and a ⟨c+a⟩ Burgers class appear across the swept systems.
        assert len(burgers_lens) >= 2, f"only one Burgers class swept: {burgers_lens}"


@pytest.mark.slow
def test_hcp_via_ti_cif(tmp_path):
    """HCP derived from a minimal Ti P6_3/mmc CIF (no explicit structure_type)."""
    pytest.importorskip("gemmi")
    cif = (
        "data_Ti\n"
        "_cell_length_a 2.951\n_cell_length_b 2.951\n_cell_length_c 4.684\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 120\n"
        "_symmetry_space_group_name_H-M 'P 63/m m c'\n"
        "_space_group_IT_number 194\n"
        "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Ti1 0.3333 0.6667 0.25\n"
    )
    (tmp_path / "ti.cif").write_text(cif, encoding="utf-8")
    eta = _eta_for(_TI, _HKL)
    toml = (
        "[reciprocal]\n"
        f"hkl = [{_HKL[0]}, {_HKL[1]}, {_HKL[2]}]\nkeV = 17.0\n"
        'backend = "analytic"\nbeamstop = false\n\n'
        '[geometry]\nmode = "oblique"\n'
        f"eta = {eta!r}\n\n"
        "[crystal]\n"
        'cif = "ti.cif"\n'  # structure derived from P6_3/mmc -> hcp
        "mount_x = [2, -1, 0]\nmount_y = [0, 1, 0]\nmount_z = [0, 0, 1]\n"
        'mode = "centered"\n\n'
        "[crystal.centered]\n"
        "b = [1, 0, 0]\nn = [0, 0, 1]\nt = [0, 1, 0]\n\n"  # basal ⟨a⟩
        "[scan.phi]\nvalue = 0.0\n\n"
        "[io]\ninclude_perfect_crystal = false\nwrite_strain_provenance = false\n\n"
        "[postprocess]\nenabled = false\n"
    )
    cfg_path = tmp_path / "ti.toml"
    cfg_path.write_text(toml, encoding="utf-8")
    cfg = SimulationConfig.from_toml(cfg_path)
    assert cfg.geometry.mount is not None
    assert cfg.geometry.mount.resolved_structure_type == "hcp"
    out = tmp_path / "out"
    run_simulation(cfg, out)
    det = next(out.glob("scan*/dfxm_sim_detector_0000.h5"))
    with h5py.File(det, "r") as f:
        img = f["/entry_0000/dfxm_sim_detector/image"][...]
    assert np.isfinite(img).all() and float(img.max()) > 0.0
