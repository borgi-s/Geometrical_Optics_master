"""Regression test for the rl-units bug in the IDENTIFICATION strain path.

The three identification spec generators (`_iter_identification_single`,
`_iter_identification_multi`, `_iter_identification_zscan`) build each
dislocation's displacement-gradient field `Hg` by calling `Fd_find_mixed` /
`Fd_find_multi_dislocs_mixed`. Those kernels expect the lab-frame ray grid in
MICROMETRES (matching `b = BURGERS_VECTOR` in µm), exactly as the forward path
does via `Fd_find(rl * 1e6, ...)` (strain_cache.py:61) and
`find_hg_population(rl_eff * 1e6, ...)` (forward_model.py:1139).

The bug: the identification generators passed `rl_eff` / `rl_shifted` in
METRES (`fm.rl` and `fm.Z_shift(...)` are both metres) with no `* 1e6`, so
the coordinates were 1e6x too small, the 1/r field 1e6x too large, the
deformation gradient saturated, and `Hg = (Fg^-1)^T - I` collapsed to ~-I
everywhere (bulk |Hg| ~ O(1) instead of the physical ~1e-5). Every
identification detector image collapsed to the singular core / went empty.

The v2.0.0 rl-units fix patched only `Find_Hg_from_population` (the forward
population path); the identification call sites in `pipeline.py` were never
patched. This went undetected across releases because the identification
tests asserted only file existence + array shape, never intensity / field
magnitude.

A physically-correct single-dislocation field has bulk (99th-percentile)
|Hg| of order 1e-5 (audit-measured 2.14e-5 for the corrected path); the
buggy metre-scale field measured ~1.46. The band ``1e-8 < p99 < 1e-2``
cleanly separates the two.
"""

from __future__ import annotations

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    IdentificationConfig,
    IdentificationCrystalConfig,
    IdentificationMonteCarloConfig,
    IdentificationNoiseConfig,
    IdentificationZScanConfig,
    ReciprocalConfig,
    ScanConfig,
    _iter_identification_multi,
    _iter_identification_single,
    _iter_identification_zscan,
)

# Physical band for the bulk displacement-gradient magnitude of a single
# dislocation. Correct path ~2e-5; the metre-scale bug inflates it to ~O(1).
_PHYS_LO = 1e-8
_PHYS_HI = 1e-2


@pytest.fixture
def q_hkl_set():
    """Set the module-global reflection `fm.q_hkl` as the kernel loader does.

    The spec generators read `fm.q_hkl` (via `precompute_forward_static`),
    which the on-demand kernel load normally sets. `Hg`'s magnitude — the
    quantity under test — does NOT depend on `q_hkl` (it only feeds the
    scan-invariant `base_qc`), so this is faithful setup, not a mock of the
    code under test. Saved/restored to avoid leaking into other tests.
    """
    saved = fm.q_hkl
    q = np.asarray([-1.0, 1.0, -1.0])
    fm.q_hkl = q / np.linalg.norm(q)
    try:
        yield
    finally:
        fm.q_hkl = saved


def _single_b0_alpha0_crystal() -> IdentificationCrystalConfig:
    """One plane, one Burgers vector, one rotation angle → exactly one spec."""
    return IdentificationCrystalConfig(
        slip_plane_normal=(1, 1, 1),
        angle_start_deg=0.0,
        angle_stop_deg=0.0,
        angle_step_deg=10.0,
        b_vector_indices=[0],
        sweep_all_slip_planes=False,
        exclude_invisibility=False,
    )


def _assert_physical_field(Hg: np.ndarray, mode: str) -> None:
    p99 = float(np.percentile(np.abs(Hg), 99))
    assert _PHYS_LO < p99 < _PHYS_HI, (
        f"{mode}: 99th-percentile |Hg| = {p99:.3e} is outside the physical band "
        f"[{_PHYS_LO:.0e}, {_PHYS_HI:.0e}]; a metre-scale rl (missing *1e6) "
        f"inflates the field ~1e6x and collapses the identify image."
    )


class TestIdentificationRlUnits:
    def test_single_field_is_physically_scaled(self, q_hkl_set: None) -> None:
        cfg = IdentificationConfig(
            mode="single",
            crystal=_single_b0_alpha0_crystal(),
            scan=ScanConfig(),
            reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
        )
        spec = next(iter(_iter_identification_single(cfg, fm._context_from_globals())))
        _assert_physical_field(spec.dfxm_geo["Hg"], "single")

    def test_multi_field_is_physically_scaled(self, q_hkl_set: None) -> None:
        cfg = IdentificationConfig(
            mode="multi",
            # pos_std_um=0.0 pins both dislocations to the grid centre so the
            # near-core field is ON the ray grid; with random ±µm offsets the
            # metre-scale bug pushes both cores off the (1e6x-too-small) grid,
            # leaving a tiny field that hides the bug from a magnitude check.
            multi=IdentificationMonteCarloConfig(
                n_samples=1, pos_std_um=0.0, render_per_dislocation=False
            ),
            scan=ScanConfig(),
            noise=IdentificationNoiseConfig(poisson_noise=False, rng_seed=0),
            reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
        )
        spec = next(iter(_iter_identification_multi(cfg, fm._context_from_globals())))
        _assert_physical_field(spec.dfxm_geo["Hg"], "multi")

    def test_zscan_field_is_physically_scaled(self, q_hkl_set: None) -> None:
        cfg = IdentificationConfig(
            mode="z-scan",
            crystal=_single_b0_alpha0_crystal(),
            zscan=IdentificationZScanConfig(z_offsets_um=[0.0], include_secondary=False),
            scan=ScanConfig(),
            reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
        )
        spec = next(iter(_iter_identification_zscan(cfg, fm._context_from_globals())))
        _assert_physical_field(spec.dfxm_geo["Hg"], "z-scan")
