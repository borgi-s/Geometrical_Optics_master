"""Regression test for the rl-units bug in the population strain path.

`Find_Hg_from_population` must pass the lab-frame ray grid `rl` to
`Fd_find_multi_dislocs_mixed` in MICROMETRES (matching `b = BURGERS_VECTOR`
in µm), exactly as the wall path does via `Fd_find(rl * 1e6, ...)` and as the
reference `disloc_identify.py` does via `Fd_find_mixed(rl * 1e6, ...)`.

The bug (shipped in sub-project C / v1.2.0): the population path passed `rl`
in METRES, making |rd| 1e6x too small, the 1/r field 1e6x too large, and the
weak-beam contrast collapse to the singular core. The wall path was unaffected
(it uses the legacy Fd_find with the correct *1e6), which is why the
Fd_find_smoke golden never caught it.
"""

from __future__ import annotations

import numpy as np

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.crystal.dislocations import Fd_find_mixed
from dfxm_geo.crystal.rotations import fast_inverse2
from dfxm_geo.pipeline import CenteredCrystalConfig, CrystalConfig


def _default_ctx() -> fm.ForwardContext:
    """A ForwardContext for the default reflection with a no-op resolution.

    #16 Slice 5: rl/Us/Theta are sourced from this ctx (geometry/instrument)
    rather than module globals; the resolution backend is a stub since the
    population strain path doesn't touch it.
    """
    instr = fm.build_instrument_context()
    geom = fm.build_geometry_context(0.30, instr)
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
        analytic_eval=None,
        loaded_kernel_path=None,
    )
    return fm.ForwardContext(
        instrument=instr,
        geometry=geom,
        resolution=res,
        q_hkl=np.array([-1.0, 1.0, -1.0]) / np.sqrt(3),
    )


def _hg_from_fg(Fg: np.ndarray) -> np.ndarray:
    """Hg = transpose(Fg^-1) - I, the displacement-gradient convention."""
    Hg = np.transpose(fast_inverse2(Fg), [0, 2, 1])
    return Hg - np.identity(3)


class TestPopulationRlUnits:
    def test_centered_field_matches_micrometre_oracle(self) -> None:
        """Find_Hg_from_population for a single centered dislocation must equal
        the field computed by feeding rl in micrometres directly to
        Fd_find_mixed (the reference convention)."""
        crystal = CrystalConfig(mode="centered", centered=CenteredCrystalConfig())
        pop = fm.build_dislocation_population(
            crystal, fov_lateral_um=abs(fm.yl_start) * 2e6, rng=None
        )

        # Pipeline path (uses ctx.geometry.rl internally):
        ctx = _default_ctx()
        Hg_pop, _ = fm.Find_Hg_from_population(pop, h=-1, k=1, l=-1, ctx=ctx)

        # Oracle: the SAME rl (in micrometres) into the same kernel.
        Ud = pop.Ud[0]
        Fg_oracle = Fd_find_mixed(
            ctx.geometry.rl * 1e6, ctx.instrument.Us, Ud, 0.0, ctx.geometry.Theta
        )
        Hg_oracle = _hg_from_fg(Fg_oracle)

        # Must match the micrometre oracle, NOT the (buggy) metre-scale field.
        assert np.allclose(Hg_pop, Hg_oracle, atol=1e-9), (
            "Find_Hg_from_population field does not match the micrometre oracle; "
            "rl is likely being passed in metres (missing *1e6)."
        )

    def test_centered_field_is_physically_scaled(self) -> None:
        """The peak displacement gradient of a single dislocation should be
        order ~0.1-3 (physical near-core strain), not ~1e6x inflated."""
        crystal = CrystalConfig(mode="centered", centered=CenteredCrystalConfig())
        pop = fm.build_dislocation_population(
            crystal, fov_lateral_um=abs(fm.yl_start) * 2e6, rng=None
        )
        Hg_pop, _ = fm.Find_Hg_from_population(pop, h=-1, k=1, l=-1, ctx=_default_ctx())
        peak = float(np.abs(Hg_pop - np.identity(3)).max())
        assert 0.05 < peak < 5.0, (
            f"peak |Hg-I| = {peak:.3e} is outside the physical range; "
            "a metre-scale rl inflates this well above 1."
        )
