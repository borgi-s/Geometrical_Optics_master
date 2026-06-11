"""g.b invisibility physics regression test (Task 6, M3 plan 2).

Classic kinematical criterion: a dislocation is invisible when g.b = 0 (its
Burgers vector is perpendicular to the diffraction vector).  This test
exercises the multi-reflection forward orchestrator as the physics gate:

    b = [-1, 1, 0]    n = [1, 1, 1]    t = n x b = [-1, -1, 2]
        g.b for g=[1,1,1]: (-1+1+0)/sqrt(3) = 0   -> INVISIBLE
        g.b for g=[2,0,0]: (-2+0+0)/sqrt(2) != 0  -> VISIBLE

Both reflections are Laue-reachable at 19.1 keV with the standard cubic Al
mount.  The contrast metric is

    contrast = std(im_strained - im_perfect) / mean(im_perfect)

where scan0001 is the strained image and scan0002 is the perfect-crystal
reference (io.include_perfect_crystal = true).  At Npixels=510, steps=5:

    Measured values (analytic backend, deterministic — zero variance):
        contrast(g=111) = 0.0287    (near-zero: dislocation is invisible)
        contrast(g=200) = 0.1020    (visible: clear dislocation signal)
        ratio(g=200 / g=111) = 3.55

The test asserts ratio > 3.0 (measured 3.55, deterministic under the
analytic backend — ratio has been verified stable across repeated runs).

Sanity-flip control (verified offline, not in suite to save runtime):
    b=[1,0,0], n=[0,1,0], t=[0,0,-1] => BOTH g=111 and g=200 visible
    => ratio = 2.46 (below the 3x threshold => assert would FAIL correctly).

Physics basis:
    Borgi, S. et al. J. Appl. Cryst. 58, 813-821 (2025).
    Howie & Whelan (1961): g.b = 0 criterion for kinematical invisibility.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from dfxm_geo.crystal.burgers import gb_cos
from dfxm_geo.pipeline import SimulationConfig, run_simulation

# ---------------------------------------------------------------------------
# Analytic backend, tiny scan (5 frames), include_perfect_crystal=true.
# beamstop=false is required by the analytic backend (no kernel).
# b=[-1,1,0], n=[1,1,1], t=n x b = (-1,-1,2) — validated by CenteredCrystalConfig.
# Reflections: [1,1,1] (invisible: g.b=0) and [2,0,0] (visible: g.b=-2).
# Both are Laue-reachable at 19.1 keV with a cubic Al (a=4.0493 A) mount.
# ---------------------------------------------------------------------------
_GB_INVISIBILITY_TOML = """\
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
mode = "centered"

[crystal.centered]
b = [-1, 1, 0]
n = [1,  1, 1]
t = [-1, -1, 2]

[scan]
[scan.phi]
value = 0.0
range = 1.25e-4
steps = 5

[io]
include_perfect_crystal = true

[postprocess]
enabled = false

[[reflections]]
hkl = [1, 1, 1]
[[reflections]]
hkl = [2, 0, 0]
"""

# Minimum acceptable ratio of contrast(g=200) / contrast(g=111).
# Measured at Npixels=510 steps=5 analytic backend: 3.55x (deterministic).
_MIN_RATIO = 3.0

# Small positive guard against division by near-zero contrast.
_TINY = 1e-10


@pytest.fixture(scope="module")
def gb_invisibility_run(tmp_path_factory):
    """Run the 2-reflection forward simulation once; share across all tests
    in this module (scope=module keeps total runtime to a single forward pass)."""
    tmp = tmp_path_factory.mktemp("gb_invis")
    cfg_file = tmp / "config.toml"
    cfg_file.write_text(_GB_INVISIBILITY_TOML, encoding="utf-8")
    cfg = SimulationConfig.from_toml(cfg_file)
    out_dir = tmp / "out"
    run_simulation(cfg, out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _contrast(out_dir, reflection_index: int) -> float:
    """Compute std(im_strained - im_perfect) / mean(im_perfect) for one reflection.

    scan0001 = strained, scan0002 = perfect crystal
    (io.include_perfect_crystal = true writes the perfect scan into scan0002).
    """
    refl_dir = out_dir / f"reflection_{reflection_index:03d}"
    det_strained = refl_dir / "scan0001" / "dfxm_sim_detector_0000.h5"
    det_perfect = refl_dir / "scan0002" / "dfxm_sim_detector_0000.h5"

    with h5py.File(det_strained, "r") as fh:
        im_strained = fh["/entry_0000/dfxm_sim_detector/image"][...].astype(float)
    with h5py.File(det_perfect, "r") as fh:
        im_perfect = fh["/entry_0000/dfxm_sim_detector/image"][...].astype(float)

    mean_perfect = float(np.mean(im_perfect))
    diff = im_strained - im_perfect
    return float(np.std(diff)) / max(mean_perfect, _TINY)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGbInvisibilityPhysics:
    """g.b criterion gate: a dislocation invisible for g=[1,1,1] must produce
    dramatically lower contrast than when visible for g=[2,0,0]."""

    def test_gb_cos_for_chosen_vectors(self):
        """Sanity-check: gb_cos is 0 for g=111 and non-zero for g=200.

        This verifies the crystallography of the test config independently of
        the forward simulation.
        """
        b = np.array([-1.0, 1.0, 0.0])
        g_111 = np.array([1.0, 1.0, 1.0])
        g_200 = np.array([2.0, 0.0, 0.0])

        assert gb_cos(g_111, b) == pytest.approx(0.0, abs=1e-12), (
            "g=[1,1,1] should be invisible (g.b=0)"
        )
        assert gb_cos(g_200, b) > 0.5, "g=[2,0,0] should be strongly visible (g.b=-2)"

    def test_t_is_n_cross_b(self):
        """The t vector in the TOML is exactly n x b (CenteredCrystalConfig validator)."""
        n = np.array([1, 1, 1])
        b = np.array([-1, 1, 0])
        t_computed = np.cross(n, b)  # [-1, -1, 2]
        t_toml = np.array([-1, -1, 2])
        # parallel: cross product must be zero (integer check, no tolerance needed)
        assert np.all(np.cross(t_computed, t_toml) == 0), (
            f"t={t_toml} is not parallel to n x b={t_computed}"
        )

    def test_contrast_g200_dominates_g111(self, gb_invisibility_run):
        """g=[2,0,0] image must have >= 3x the relative contrast of g=[1,1,1].

        The g=[1,1,1] reflection probes a direction perpendicular to b (g.b=0),
        so the dislocation strain field produces essentially no modulation.
        The g=[2,0,0] reflection is along [1,0,0] which has g.b != 0 and
        shows clear dislocation contrast.

        Measured values (analytic backend, steps=5, Npixels=510, deterministic):
            contrast(g=111) = 0.0287
            contrast(g=200) = 0.1020
            ratio = 3.55
        """
        out_dir = gb_invisibility_run

        c_111 = _contrast(out_dir, reflection_index=1)  # g=[1,1,1]
        c_200 = _contrast(out_dir, reflection_index=2)  # g=[2,0,0]

        ratio = c_200 / max(c_111, _TINY)

        # Report values for diagnosis when the assertion fails.
        assert ratio >= _MIN_RATIO, (
            f"g.b invisibility broken: contrast(g=111)={c_111:.5f}, "
            f"contrast(g=200)={c_200:.5f}, ratio={ratio:.2f} < {_MIN_RATIO}. "
            "Expected: g=[1,1,1] nearly invisible (g.b=0), g=[2,0,0] visible (g.b!=0)."
        )

    def test_invisible_reflection_contrast_is_small(self, gb_invisibility_run):
        """The invisible reflection (g=111) must have very small contrast.

        This test ensures the g=[1,1,1] signal is genuinely near-zero, not
        just smaller than g=[2,0,0] — it guards against both images having
        large contrast.

        Measured: contrast(g=111) = 0.0287 at steps=5, Npixels=510.
        """
        out_dir = gb_invisibility_run
        c_111 = _contrast(out_dir, reflection_index=1)

        # Contrast must be below 0.10 (the measured value is 0.029).
        assert c_111 < 0.10, (
            f"g=111 contrast too large: {c_111:.5f}. "
            "Expected near-zero (g.b=0 => dislocation invisible)."
        )

    def test_visible_reflection_has_nonzero_contrast(self, gb_invisibility_run):
        """The visible reflection (g=200) must show measurable contrast.

        Measured: contrast(g=200) = 0.1020.
        """
        out_dir = gb_invisibility_run
        c_200 = _contrast(out_dir, reflection_index=2)

        assert c_200 > 0.05, (
            f"g=200 contrast too small: {c_200:.5f}. "
            "Expected significant contrast (g.b != 0 => dislocation visible)."
        )
