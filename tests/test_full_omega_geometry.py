"""Full-omega forward path: the goniometer rotation omega must rotate the probed
volume (ray grid ``rl``) about lab z, not only the diffraction vector (B').

Physics (verified via the offset oracle): the goniometer rotates the SAMPLE by
R_z(omega) about lab z; the lab beam/detector are fixed. To evaluate the field
at a fixed detector ray ``rl_lab`` we look up the material coordinate
``R_z(omega).T @ rl_lab``. So ``build_geometry_context`` must counter-rotate the
ray grid by ``R_z(omega).T`` when omega != 0 (z preserved -> prob_z unchanged),
while ``precompute_forward_static`` keeps the existing ``R_z(omega) @ Us`` on the
q-path. Together that is full-omega. omega == 0 stays bit-identical.
"""

from __future__ import annotations

import numpy as np

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.crystal.oblique import _R_z
from dfxm_geo.reciprocal_space.analytic_resolution import AnalyticResolution


def _instr():
    return fm.build_instrument_context_from_config(psize=4e-8, zl_rms=fm.zl_rms, Npixels=63, Nsub=1)


def test_omega_zero_rl_is_plain_mgrid():
    """omega == 0 must leave rl as the un-rotated lab mgrid (bit-identical guard)."""
    instr = _instr()
    th = 0.3
    geom = fm.build_geometry_context(th, instr, omega=0.0)
    # rebuild the documented mgrid expression independently
    xl_range = -(instr.yl_start / np.tan(2 * th) / 3)
    yl_range = -instr.yl_start
    zl_range = 0.5 * instr.zl_rms * 6
    rl = np.vstack(
        np.mgrid[
            -xl_range : xl_range : complex(instr.xl_steps),
            -yl_range : yl_range : complex(instr.yl_steps),
            -zl_range : zl_range : complex(instr.zl_steps),
        ]
    ).reshape(3, -1)
    assert np.array_equal(geom.rl, rl)


def test_omega_counter_rotates_rl_about_lab_z():
    """omega != 0 must counter-rotate rl by R_z(omega).T relative to omega == 0."""
    instr = _instr()
    th, omega = 0.3, 0.7
    geom0 = fm.build_geometry_context(th, instr, omega=0.0)
    geomw = fm.build_geometry_context(th, instr, omega=omega)
    expected = _R_z(omega).T @ geom0.rl
    np.testing.assert_allclose(geomw.rl, expected, rtol=1e-12, atol=1e-18)


def test_omega_preserves_z_and_prob_z():
    """R_z is about lab z, so the z-row and the beam profile prob_z are unchanged."""
    instr = _instr()
    th, omega = 0.3, 1.1
    geom0 = fm.build_geometry_context(th, instr, omega=0.0)
    geomw = fm.build_geometry_context(th, instr, omega=omega)
    np.testing.assert_allclose(geomw.rl[2], geom0.rl[2], rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(geomw.prob_z, geom0.prob_z, rtol=1e-12, atol=1e-18)


def test_full_omega_places_core_at_Rz_offset():
    """End-to-end: a centered dislocation's field-singular core (max |Hg|) must
    appear in the detector at R_z(omega) @ offset under full-omega (the overlay
    formula), whereas at omega == 0 it sits at offset."""
    import dfxm_geo.crystal.frank_walls as fw
    from dfxm_geo.crystal.cell import UnitCell

    instr = fm.build_instrument_context_from_config(
        psize=4e-8, zl_rms=fm.zl_rms, Npixels=151, Nsub=1
    )
    cell = UnitCell.cubic(4.05e-10)
    theta, omega = 0.2575, 1.0427  # ~ (2,2,0) FCC Al 17 keV (theta 14.75, omega 59.75 deg)
    res = fm.ResolutionContext(
        Resq_i=None,
        qi1_start=0.0,
        qi1_step=0.0,
        qi2_start=0.0,
        qi2_step=0.0,
        qi3_start=0.0,
        qi3_step=0.0,
        npoints1=None,
        npoints2=None,
        npoints3=None,
        analytic_eval=AnalyticResolution(
            theta=theta,
            eta=-1.5708,
            zeta_v_fwhm=0.00053,
            zeta_h_fwhm=0.0,
            NA_rms=0.00031106382978723403,
            eps_rms=6e-05,
            zeta_v_clip=0.00014,
        ),
        loaded_kernel_path=None,
    )
    s = fw.RECIPES["leds_eq14"].sets[0]
    Ud = fm._ud_matrix_from_bnt(s.b, s.slip_plane, fw._edge_t(s.slip_plane, s.b))
    xih = fw._unit(fw._cartesian(s.xi, cell))
    rot = fw._signed_angle(Ud[:, 2], xih, Ud[:, 1])
    b_um = fw.burgers_magnitude_of(s.b, cell, fraction=1.0)
    O = np.array([0.7, 0.4, 0.0])
    pop = fm.DislocationPopulation(
        positions_um=np.array([O]),
        Ud=np.array([Ud]),
        sidecar=None,
        rotation_deg=np.array([rot]),
        b_um_per=np.array([b_um]),
        ny=0.334,
    )

    ctx0 = fm.build_forward_context(theta, res, (2, 2, 0), instrument=instr, omega=0.0, cell=cell)
    ctxw = fm.build_forward_context(theta, res, (2, 2, 0), instrument=instr, omega=omega, cell=cell)
    rl_lab = ctx0.geometry.rl  # detector lab coords (fixed)
    # Restrict to the mid-z (z ~ 0) layer: the core line tilts through z, so the
    # representative crossing is at z = 0 (omega rotates only lab x/y).
    zvals = np.unique(rl_lab[2])
    sel = np.isclose(rl_lab[2], zvals[len(zvals) // 2])

    def core_xy(ctx):
        Hg, _ = fm.Find_Hg_from_population(pop, ctx=ctx)
        hn = (Hg**2).sum(axis=(1, 2))[sel]
        j = int(np.argmax(hn))
        return rl_lab[:2, sel][:, j] * 1e6  # detector lab position of max-|Hg| ray (um)

    xy0 = core_xy(ctx0)
    xyw = core_xy(ctxw)
    expected_w = (_R_z(omega) @ O)[:2]
    # omega == 0: core at the offset; full-omega: core at R_z(omega) @ offset.
    # tolerance ~ a couple of detector pixels (40 nm) + the line's z-tilt.
    np.testing.assert_allclose(xy0, O[:2], atol=0.12)
    np.testing.assert_allclose(xyw, expected_w, atol=0.12)
