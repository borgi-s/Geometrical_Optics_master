"""Gate C (v2.3.0 oblique ship-gate): the oblique forward produces sensible output.

At/near the rocking peak the bulk crystal diffracts, so an oblique single-image
forward must yield a broadly illuminated FIELD (not the sparse dots a wrong
setpoint/geometry produced), with the run's oblique geometry recorded in the HDF5
provenance. This is a qualitative sanity gate -- NOT pixel-fidelity to the paper's
Fig 3 (that needs darkmod + a GPU; see
docs/superpowers/specs/2026-05-29-v230-oblique-ship-gate-rescope.md).

Marked `slow`: runs a real analytic oblique forward (no kernel needed) at the
module-default grid (~tens of seconds). Run with `pytest -m slow`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm_geo.data import configs_root
from dfxm_geo.pipeline import ScanConfig, SimulationConfig, run_simulation


def _find_detector(grp: h5py.Group) -> h5py.Dataset:
    for key in grp:
        item = grp[key]
        if isinstance(item, h5py.Dataset) and item.ndim >= 2 and "detector" in item.name:
            return item
        if isinstance(item, h5py.Group):
            found = _find_detector(item)
            if found is not None:
                return found
    return None


@pytest.mark.slow
def test_oblique_forward_at_peak_is_a_full_field(tmp_path: Path) -> None:
    """Oblique forward at the rocking peak -> broadly illuminated field + oblique
    provenance (geometry_mode/eta/theta), not sparse dots."""
    cfg = SimulationConfig.from_toml(configs_root() / "al_oblique_figure3.toml")
    # Rocking peak (default ScanConfig = all axes fixed at 0) so the bulk diffracts;
    # keep the output small (no strain dump, no perfect-crystal scan).
    cfg = replace(
        cfg,
        scan=ScanConfig(),
        io=replace(cfg.io, write_strain_provenance=False, include_perfect_crystal=False),
    )

    run_simulation(cfg, tmp_path)

    with h5py.File(tmp_path / "dfxm_geo.h5", "r") as f:
        attrs = f["/1.1"].attrs
        assert attrs["geometry_mode"] == "oblique"
        assert np.isclose(float(attrs["eta"]), 0.353140, atol=1e-4)
        # theta must be the run reflection's Bragg angle (15.416 deg), not the
        # 8.98 deg module default (the theta_0 staleness this arc fixed).
        assert np.isclose(float(attrs["theta"]), np.deg2rad(15.416), atol=1e-3)
        img = _find_detector(f["/1.1"])[0].astype(np.float64)

    nonzero_fraction = float((img > 0).mean())
    bright_fraction = float((img > 0.3 * img.max()).mean())
    assert img.max() > 0.0
    assert nonzero_fraction > 0.5, (
        f"expected a full field, got nonzero fraction {nonzero_fraction:.3f}"
    )
    assert bright_fraction > 0.3, (
        f"expected broad illumination, got bright fraction {bright_fraction:.3f}"
    )
