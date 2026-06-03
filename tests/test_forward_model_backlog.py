"""Backlog fixes for direct_space/forward_model.py (lane A).

Covers four deferred items:
  * #8  find_hg_population must honour per-dislocation rotation_deg (mixed
        edge+screw character), not silently force pure edge.
  * #9  _SLIP_SYSTEM_111 must span all 12 FCC slip systems (4 {111} planes x
        3 <110> Burgers each), every (b, n) glide-orthogonal (b.n == 0).
  * #11 _load_analytic_resolution must derive theta from the config's lattice
        parameter, not a hardcoded Al value.
  * #14 docstring-only; asserted via an import-level substring check.
"""

from __future__ import annotations

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm


# --------------------------------------------------------------------------- #
# #8 — find_hg_population honours rotation_deg (mixed edge+screw)
# --------------------------------------------------------------------------- #
def _single_disloc_population(rotation_deg: float) -> fm.DislocationPopulation:
    """A 1-dislocation population at the origin with the given rotation_deg.

    Uses the canonical wall slip system so only the screw/edge mixing
    (rotation_deg) varies between the two populations under test.
    """
    Ud = fm._ud_matrix_from_bnt((1, -1, 0), (1, 1, 1), (1, 1, -2))[np.newaxis, :, :]
    return fm.DislocationPopulation(
        positions_um=np.zeros((1, 3), dtype=np.float64),
        Ud=Ud,
        sidecar=None,
        rotation_deg=np.array([rotation_deg], dtype=np.float64),
    )


def test_population_rotation_deg_changes_hg_field() -> None:
    """A pure-screw (rotation_deg=90) population must produce a DIFFERENT Hg
    field than a pure-edge (rotation_deg=0) one. Before the fix the kernel
    hardcoded cos=1/sin=0 (pure edge), so the two were bit-identical."""
    pop_edge = _single_disloc_population(0.0)
    pop_screw = _single_disloc_population(90.0)

    Hg_edge, _ = fm.Find_Hg_from_population(pop_edge, h=-1, k=1, l=-1)
    Hg_screw, _ = fm.Find_Hg_from_population(pop_screw, h=-1, k=1, l=-1)

    assert not np.allclose(Hg_edge, Hg_screw), (
        "pure-screw and pure-edge populations gave identical Hg fields; "
        "rotation_deg is being ignored (item #8 regression)."
    )


def test_population_default_rotation_is_pure_edge() -> None:
    """Populations with rotation_deg=None (the build_dislocation_population
    default) must keep the legacy pure-edge field — identical to an explicit
    rotation_deg=0 array. Guards backward compatibility / the parity oracle."""
    Ud = fm._ud_matrix_from_bnt((1, -1, 0), (1, 1, 1), (1, 1, -2))[np.newaxis, :, :]
    pop_none = fm.DislocationPopulation(
        positions_um=np.zeros((1, 3), dtype=np.float64), Ud=Ud, sidecar=None
    )
    pop_zero = _single_disloc_population(0.0)

    Hg_none, _ = fm.Find_Hg_from_population(pop_none, h=-1, k=1, l=-1)
    Hg_zero, _ = fm.Find_Hg_from_population(pop_zero, h=-1, k=1, l=-1)

    np.testing.assert_allclose(Hg_none, Hg_zero, rtol=1e-12, atol=1e-14)


def test_population_numpy_oracle_honours_rotation_deg() -> None:
    """The NumPy parity oracle must apply the SAME rotation_deg as the fused
    kernel, so the rtol=1e-12 parity test keeps holding for mixed populations."""
    pop_screw = _single_disloc_population(90.0)
    # S4 (#16): pass explicit ctx to both sides — same geometry, guaranteed parity.
    ctx = fm._context_from_globals()
    Hg_fast, _ = fm.Find_Hg_from_population(pop_screw, h=-1, k=1, l=-1, ctx=ctx)
    Hg_ref, _ = fm._find_hg_from_population_numpy(
        pop_screw, -1, 1, -1, S=fm._S_IDENTITY, rl=None, ctx=ctx
    )
    np.testing.assert_allclose(Hg_fast, Hg_ref, rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------- #
# #9 — _SLIP_SYSTEM_111 covers all 12 FCC slip systems
# --------------------------------------------------------------------------- #
def test_slip_system_table_has_twelve_systems() -> None:
    """FCC has 12 slip systems: 4 {111} planes x 3 <110> Burgers each."""
    assert len(fm._SLIP_SYSTEM_111) == 12


def test_slip_system_table_spans_four_distinct_planes() -> None:
    """The 12 systems must span all 4 distinct {111} plane normals."""
    planes = {tuple(n) for _, n, _ in fm._SLIP_SYSTEM_111}
    assert len(planes) == 4


def test_every_slip_system_is_glide_orthogonal() -> None:
    """Every (b, n) pair must satisfy b.n == 0 (Burgers lies in the glide
    plane) and t must be (anti)parallel to n x b (a valid line direction)."""
    for b, n, t in fm._SLIP_SYSTEM_111:
        b_arr = np.array(b, dtype=np.float64)
        n_arr = np.array(n, dtype=np.float64)
        t_arr = np.array(t, dtype=np.float64)
        assert int(np.dot(b_arr, n_arr)) == 0, f"b.n != 0 for b={b}, n={n}"
        cross = np.cross(n_arr, b_arr)
        # t parallel or antiparallel to n x b.
        assert np.allclose(np.cross(cross, t_arr), 0.0), (
            f"t={t} is not collinear with n x b for b={b}, n={n}"
        )


# --------------------------------------------------------------------------- #
# #11 — _load_analytic_resolution derives theta from the config lattice param
# --------------------------------------------------------------------------- #
class _StubReciprocalConfig:
    """Minimal duck-typed ReciprocalConfig for analytic-backend tests."""

    def __init__(self, lattice_a: float | None = None) -> None:
        self.hkl = (-1, 1, -1)
        self.keV = 17.0
        self.eta = 0.0
        self.zeta_v_fwhm = 5.3e-4
        self.zeta_h_fwhm = 0.0
        self.NA_rms = 7.31e-4 / 2.35
        self.eps_rms = 1.41e-4 / 2.35
        self.zeta_v_clip = 1.4e-4
        if lattice_a is not None:
            self.lattice_a = lattice_a


def test_analytic_resolution_theta_depends_on_lattice_param(monkeypatch) -> None:
    """A non-Al lattice parameter `a` must yield a correspondingly different
    Bragg theta in the analytic backend, instead of the hardcoded Al value."""
    captured: dict[str, float] = {}

    class _SpyAnalyticResolution:
        def __init__(self, *, theta: float, **_kw) -> None:
            captured["theta"] = theta

    monkeypatch.setattr(
        "dfxm_geo.reciprocal_space.analytic_resolution.AnalyticResolution",
        _SpyAnalyticResolution,
    )
    # Keep the Hg/q_hkl side-effect inert (don't trigger a kernel build).
    monkeypatch.setattr(fm, "Hg", np.zeros((1, 3, 3)))
    # _load_analytic_resolution sets the fm._analytic_eval module global as a
    # side effect; register it with monkeypatch so the spy is restored on
    # teardown. Otherwise it leaks and breaks forward() in later tests (the
    # exact fm-globals hazard the #16 ForwardContext refactor targets).
    monkeypatch.setattr(fm, "_analytic_eval", None)

    fm._load_analytic_resolution(_StubReciprocalConfig(lattice_a=4.0495e-10))
    theta_al = captured["theta"]

    fm._load_analytic_resolution(_StubReciprocalConfig(lattice_a=5.0e-10))
    theta_big = captured["theta"]

    assert theta_al != pytest.approx(theta_big), (
        "theta did not change with the lattice parameter; _load_analytic_resolution "
        "is still hardcoding the Al `a` (item #11 regression)."
    )
    # Larger a -> larger d_hkl -> smaller sin(theta) -> smaller theta.
    assert theta_big < theta_al


def test_analytic_resolution_defaults_to_al_lattice(monkeypatch) -> None:
    """When the config carries no lattice_a, the backend must fall back to the
    legacy Al value (backward compatibility for v2.2.0-era configs)."""
    captured: dict[str, float] = {}

    class _SpyAnalyticResolution:
        def __init__(self, *, theta: float, **_kw) -> None:
            captured["theta"] = theta

    monkeypatch.setattr(
        "dfxm_geo.reciprocal_space.analytic_resolution.AnalyticResolution",
        _SpyAnalyticResolution,
    )
    monkeypatch.setattr(fm, "Hg", np.zeros((1, 3, 3)))
    # _load_analytic_resolution sets the fm._analytic_eval module global as a
    # side effect; register it with monkeypatch so the spy is restored on
    # teardown. Otherwise it leaks and breaks forward() in later tests (the
    # exact fm-globals hazard the #16 ForwardContext refactor targets).
    monkeypatch.setattr(fm, "_analytic_eval", None)

    fm._load_analytic_resolution(_StubReciprocalConfig(lattice_a=None))
    theta_default = captured["theta"]

    fm._load_analytic_resolution(_StubReciprocalConfig(lattice_a=4.0495e-10))
    theta_explicit_al = captured["theta"]

    assert theta_default == pytest.approx(theta_explicit_al)


# --------------------------------------------------------------------------- #
# #14 — contrast-formation docstring at the forward()/_mc_lut_forward site
# --------------------------------------------------------------------------- #
def test_contrast_formation_docstring_present() -> None:
    """The contrast-formation explanation (depth-integrated 2theta projection,
    bright in Res_q / dark outside, deep-weak-beam regime) must be documented
    with the Poulsen 2021 + Borgi 2024 citations."""
    doc = (fm.forward.__doc__ or "") + (fm._mc_lut_forward.__doc__ or "")
    lower = doc.lower()
    assert "depth-integrated" in lower or "depth integrated" in lower
    assert "2theta" in lower or "2θ" in doc or "two-theta" in lower
    assert "poulsen" in lower
    assert "borgi" in lower
