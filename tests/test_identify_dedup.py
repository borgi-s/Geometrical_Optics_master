"""W2 dedup contract at the orchestrator level.

The orchestrators must delegate each scene (combined + optional solos) to
exactly ONE `find_hg_scene` call with `per_dislocation` mirroring the
config flag — the pre-W2 code instead made one combined call plus one
recompute per solo render. The single-evaluation-per-dislocation property
*inside* the seam is proven engine-independently in tests/test_hg_scene.py;
spying on the seam (not on Fd_find_mixed) keeps this test valid after the
W3 engine flip routes the math through the numba kernel.

Needs the bootstrapped kernel npz (ctx construction loads the resolution LUT).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.crystal.dislocations import find_hg_scene as _real_scene

kernel_dir = Path(fm.pkl_fpath)
pytestmark = pytest.mark.skipif(
    not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")),
    reason="No bootstrapped kernel npz found.",
)

_MULTI_TOML = """\
mode = "multi"

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0

[scan.phi]
value = 1.25e-4
range = 1.25e-4
steps = 2

[detector]
rng_seed = 7

[multi]
n_samples = 1
pos_std_um = 5.0
render_per_dislocation = true

[io]
include_perfect_crystal = false
write_strain_provenance = false
"""

_MULTI_TOML_NO_SOLOS = _MULTI_TOML.replace(
    "render_per_dislocation = true",
    "render_per_dislocation = false",
)


def _spy_scene_calls(monkeypatch, toml_text: str, tmp_path: Path) -> list[dict]:
    """Run one identify config with find_hg_scene wrapped in a recording spy."""
    import dfxm_geo.orchestrator as orch
    import dfxm_geo.pipeline as pipeline

    calls: list[dict] = []

    def spy(rl_um, Us, specs, Theta, **kwargs):
        calls.append(
            {
                "n_specs": len(specs),
                "per_dislocation": kwargs.get("per_dislocation", False),
            }
        )
        return _real_scene(rl_um, Us, specs, Theta, **kwargs)

    # The orchestrators call the name imported into the orchestrator namespace
    # (refactor gate); patch it there. run_identification is still driven through
    # the pipeline facade below.
    monkeypatch.setattr(orch, "find_hg_scene", spy)

    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(toml_text, encoding="utf-8")
    cfg = pipeline.load_identification_config(cfg_path)
    pipeline.run_identification(cfg, tmp_path / "out")
    return calls


def test_multi_render_per_dislocation_is_one_scene_call(monkeypatch, tmp_path):
    # 1 sample × 2 dislocations, render_per_dislocation=True → exactly ONE
    # find_hg_scene call carrying both specs and per_dislocation=True.
    # Pre-W2 the orchestrator made 1 combined + 2 solo Fd computations and
    # never touched find_hg_scene (this test fails with AttributeError there).
    calls = _spy_scene_calls(monkeypatch, _MULTI_TOML, tmp_path)
    assert calls == [{"n_specs": 2, "per_dislocation": True}]


def test_multi_without_solo_renders_is_one_combined_call(monkeypatch, tmp_path):
    # Same scene, solos off -> still exactly ONE seam call, per_dislocation=False.
    calls = _spy_scene_calls(monkeypatch, _MULTI_TOML_NO_SOLOS, tmp_path)
    assert calls == [{"n_specs": 2, "per_dislocation": False}]


_ZSCAN_TOML = """\
mode = "z-scan"

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0

[scan.phi]
value = 1.25e-4
range = 1.25e-4
steps = 2

[detector]
rng_seed = 7

[crystal]
sweep_all_slip_planes = false
slip_plane_normal = [1, 1, 1]
b_vector_indices = [0]
angle_start_deg = 0.0
angle_stop_deg = 0.0
angle_step_deg = 30.0

[zscan]
z_offsets_um = [0.0]
include_secondary = true
render_per_dislocation = true

[io]
include_perfect_crystal = false
write_strain_provenance = false
"""


_ZSCAN_TOML_NO_SECONDARY = _ZSCAN_TOML.replace(
    "include_secondary = true",
    "include_secondary = false",
).replace(
    "render_per_dislocation = true",
    "render_per_dislocation = false",
)


def test_zscan_render_per_dislocation_is_one_scene_call(monkeypatch, tmp_path):
    # 1 z × 1 plane × 1 b × 1 angle, secondary on, render_per_dislocation on
    # → exactly ONE find_hg_scene call with both specs and per_dislocation=True.
    # Pre-W2: 1 combined + 2 solo Fd computations, zero seam calls.
    calls = _spy_scene_calls(monkeypatch, _ZSCAN_TOML, tmp_path)
    assert calls == [{"n_specs": 2, "per_dislocation": True}]


def test_zscan_without_secondary_is_one_single_spec_call(monkeypatch, tmp_path):
    # Secondary off -> the primary alone still routes through the seam.
    # The else-branch omits per_dislocation kwarg; spy records False via
    # kwargs.get default.
    # Note: render_per_dislocation=True is invalid with include_secondary=False
    # (the validator rejects it), so _ZSCAN_TOML_NO_SECONDARY sets both to false.
    calls = _spy_scene_calls(monkeypatch, _ZSCAN_TOML_NO_SECONDARY, tmp_path)
    assert calls == [{"n_specs": 1, "per_dislocation": False}]
