"""Single-reflection OBLIQUE runs must render at the solver-resolved goniometer
omega, not omega=0.

For an oblique mount the config resolves (theta, omega) from compute_omega_eta and
stores them on GeometryConfig. The forward render must thread that omega into the
ForwardContext (full-omega), exactly as the multi-reflection [[reflections]] path
does. Previously the single-reflection path dropped it (built the context with the
default omega=0), so off-axis reflections were imaged with the wrong sample
orientation. SIMPLIFIED mode keeps omega=0 (its resolved omega IS 0), so that path
stays byte-identical.
"""

from __future__ import annotations

import numpy as np

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.crystal.oblique import CrystalMount, compute_omega_eta
from dfxm_geo.pipeline import SimulationConfig, run_simulation

_AL_A, _KEV = 4.05e-10, 17.0


def _eta(hkl) -> float:
    m = CrystalMount(
        lattice="cubic",
        a=_AL_A,
        mount_x=(1, 0, 0),
        mount_y=(0, 1, 0),
        mount_z=(0, 0, 1),
        structure_type="fcc",
        material="Al",
    )
    g = compute_omega_eta(m, hkl, _KEV)
    return float(g.eta_1 if not np.isnan(g.eta_1) else g.eta_2)


def _oblique_toml(hkl) -> str:
    return f"""
[reciprocal]
hkl = [{hkl[0]}, {hkl[1]}, {hkl[2]}]
keV = {_KEV}
backend = "analytic"
beamstop = false
[geometry]
mode = "oblique"
eta = {_eta(hkl)!r}
[crystal]
lattice = "cubic"
a = {_AL_A!r}
structure_type = "fcc"
material = "Al"
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
mode = "gnb"
[crystal.gnb]
recipe = "leds_eq14"
theta_deg = 0.05
extent_um = 12.0
[scan.phi]
value = 0.0
[detector]
model = "ideal"
[io]
include_perfect_crystal = false
write_strain_provenance = false
[postprocess]
enabled = false
"""


def test_single_reflection_oblique_threads_resolved_omega(tmp_path, monkeypatch):
    hkl = (2, 2, 0)  # off-axis: solver omega ~ 59.75 deg (NOT a free azimuth)
    (tmp_path / "c.toml").write_text(_oblique_toml(hkl), encoding="utf-8")
    cfg = SimulationConfig.from_toml(tmp_path / "c.toml")
    assert cfg.geometry.omega != 0.0  # oblique resolves a non-zero omega

    seen: list[float] = []
    orig = fm.build_forward_context

    def spy(*args, **kwargs):
        seen.append(float(kwargs.get("omega", 0.0)))
        return orig(*args, **kwargs)

    monkeypatch.setattr(fm, "build_forward_context", spy)
    run_simulation(cfg, tmp_path / "out")

    assert seen, "build_forward_context was never called"
    assert any(abs(o - cfg.geometry.omega) < 1e-9 for o in seen), (
        f"single-reflection render did not thread the resolved omega "
        f"{cfg.geometry.omega}: contexts built with omega={seen}"
    )
