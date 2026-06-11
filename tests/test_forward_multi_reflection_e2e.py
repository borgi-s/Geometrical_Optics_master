"""2-reflection forward e2e smoke (analytic backend, tiny grid).

Tests the Task 3 per-reflection orchestrator: run_simulation loops over
config.reflections, writes per-reflection subdirs + a super-master.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from dfxm_geo.pipeline import SimulationConfig, run_simulation

# Analytic backend, tiny scan (3 frames), no kernel needed — CI-safe.
# [crystal] carries both mount keys AND dislocation-layout schema (mode +
# [crystal.centered]) so SimulationConfig.from_toml succeeds (see
# test_reflections_config.py for the dual-block rationale).
# [postprocess] enabled = false — skip postprocessing (requires /2.1 + Hg).
MULTI_FORWARD_TOML = """
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
b = [1, -1, 0]
n = [1,  1, -1]
t = [1,  1,  2]

[scan]
[scan.phi]
value = 0.0
range = 1.25e-4
steps = 3

[io]
include_perfect_crystal = false

[postprocess]
enabled = false

[[reflections]]
hkl = [1, 1, 3]
[[reflections]]
hkl = [-1, -1, 3]
eta = 0.3531
"""


@pytest.fixture
def multi_cfg(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(MULTI_FORWARD_TOML, encoding="utf-8")
    return SimulationConfig.from_toml(p)


def test_two_reflection_forward_writes_per_reflection_masters(multi_cfg, tmp_path):
    out = tmp_path / "out"
    result = run_simulation(multi_cfg, out)
    assert result["n_reflections"] == 2
    for idx in (1, 2):
        master = out / f"reflection_{idx:03d}" / "dfxm_geo.h5"
        assert master.is_file(), f"master missing: {master}"
        with h5py.File(master, "r") as fh:
            assert "1.1" in fh, f"scan group /1.1 missing in {master}"
            attrs = dict(fh["1.1"].attrs)
            assert attrs["reflection_index"] == idx
            assert attrs["n_reflections"] == 2
            assert "omega" in attrs


def test_super_master_links_and_table(multi_cfg, tmp_path):
    out = tmp_path / "out"
    run_simulation(multi_cfg, out)
    super_master = out / "dfxm_geo_multi.h5"
    assert super_master.is_file(), f"super-master missing: {super_master}"
    with h5py.File(super_master, "r") as fh:
        # ExternalLinks resolve to each reflection's root
        assert "1.1" in fh["reflection_001"], "ExternalLink reflection_001 broken"
        assert "1.1" in fh["reflection_002"], "ExternalLink reflection_002 broken"
        table = fh["reflections"]
        assert table["hkl"].shape == (2, 3)
        assert table["omega"].shape == (2,)
        assert table["eta"].shape == (2,)
        assert table["theta"].shape == (2,)
        assert fh.attrs["n_reflections"] == 2


def test_per_reflection_images_differ(multi_cfg, tmp_path):
    """Different q_hkl / omega must produce different contrast (B-prime is live)."""
    out = tmp_path / "out"
    run_simulation(multi_cfg, out)
    stacks = []
    for idx in (1, 2):
        det = out / f"reflection_{idx:03d}" / "scan0001" / "dfxm_sim_detector_0000.h5"
        assert det.is_file(), f"detector file missing: {det}"
        with h5py.File(det, "r") as fh:
            stacks.append(fh["/entry_0000/dfxm_sim_detector/image"][...])
    assert not np.allclose(stacks[0], stacks[1]), (
        "reflection_001 and reflection_002 produced identical images — "
        "B-prime rotation is not being applied"
    )
