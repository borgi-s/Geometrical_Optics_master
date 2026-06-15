"""M4 Stage 4.3a C1: the [[crystal.slip_system]] custom escape hatch end-to-end.

`register_custom` registers user slip systems as LITERAL single-system families
(``custom0/custom1/...``). The Task-2 hardening made ``burgers_magnitude`` REQUIRE
the family in ``_BURGERS_FRACTION``, which crashed every custom/non-FCC structure
in ``_resolve_structure_systems_b`` (forward) and ``structure_provenance_attrs``
→ ``burgers_magnitude`` (HDF5 provenance). The fix: a literal family uses
fraction 1.0 (the user supplied the actual integer Burgers vector). This test
runs a forward through the custom hatch and asserts a finite image + a sane,
cell-derived ``burgers_magnitude_um``.

Geometry mirrors test_bcc_e2e: analytic backend (no MC kernel), oblique mode,
reflection (1,1,0) at 17 keV with the cubic identity mount (η = +π/2), a single
centered dislocation, single frame.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm_geo.pipeline import SimulationConfig, run_simulation

_A = 3.0e-10  # custom cubic lattice parameter (m)
_HKL = (1, 1, 0)
_KEV = 17.0
_ETA = float(np.pi / 2)  # the (1,1,0) reflection's computed η₂ for the identity mount

# Literal-family |b|: fraction 1.0 × |A·b_int|, b_int = (1, -1, 1) → |b| = √3·a.
_EXPECTED_B_UM = _A * np.sqrt(3) * 1e6


def _custom_forward_toml() -> str:
    """Centered single-frame forward via the [[crystal.slip_system]] hatch."""
    return (
        "[reciprocal]\n"
        f"hkl = [{_HKL[0]}, {_HKL[1]}, {_HKL[2]}]\n"
        f"keV = {_KEV}\n"
        'backend = "analytic"\n'
        "beamstop = false\n"
        "\n"
        "[geometry]\n"
        'mode = "oblique"\n'
        f"eta = {_ETA!r}\n"
        "\n"
        "[crystal]\n"
        'lattice = "cubic"\n'
        f"a = {_A!r}\n"
        # A custom (non-FCC) structure needs an explicit Poisson source
        # (repo-audit #2 nu-gate, oblique.py); custom structures have no
        # material name, so give an explicit value.
        "poisson_ratio = 0.3\n"
        "mount_x = [1, 0, 0]\n"
        "mount_y = [0, 1, 0]\n"
        "mount_z = [0, 0, 1]\n"
        'mode = "centered"\n'
        "\n"
        # The custom escape hatch: one user-defined glide system.
        # b·n = 1·1 + (-1)·1 + 1·0 = 0 ✓
        "[[crystal.slip_system]]\n"
        "plane = [1, 1, 0]\n"
        "burgers = [1, -1, 1]\n"
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


@pytest.mark.slow
def test_custom_slip_system_forward_runs(tmp_path: Path) -> None:
    """A [[crystal.slip_system]] custom structure runs forward end-to-end.

    Regression for C1: before the literal-family fix this raised
    ``no lattice-translation fraction registered for family 'custom0'`` in both
    the population builder and the HDF5 provenance writer.
    """
    cfg_path = tmp_path / "custom_forward.toml"
    cfg_path.write_text(_custom_forward_toml(), encoding="utf-8")
    cfg = SimulationConfig.from_toml(cfg_path)

    # The hatch defines its own structure_type ("custom:user").
    assert cfg.geometry.mount is not None
    assert cfg.geometry.mount.resolved_structure_type.startswith("custom:")

    out = tmp_path / "out"
    run_simulation(cfg, out)

    # Detector image written, finite, non-trivial.
    det = out / "scan0001" / "dfxm_sim_detector_0000.h5"
    assert det.is_file()
    with h5py.File(det, "r") as f:
        img = f["/entry_0000/dfxm_sim_detector/image"][...]
    assert img.ndim == 3 and img.shape[0] == 1
    assert np.isfinite(img).all()
    assert float(img.max()) > 0.0

    # Provenance carries a sane, cell-derived |b| (literal family → fraction 1.0).
    with h5py.File(out / "dfxm_geo.h5", "r") as f:
        attrs = dict(f["/1.1"].attrs)
    assert attrs["structure_type"].startswith("custom:")
    b_um = float(attrs["burgers_magnitude_um"])
    assert np.isfinite(b_um) and b_um > 0.0
    assert np.isclose(b_um, _EXPECTED_B_UM, rtol=1e-4), (
        f"custom-hatch |b|={b_um} should be √3·a (literal, no ½)={_EXPECTED_B_UM}"
    )
