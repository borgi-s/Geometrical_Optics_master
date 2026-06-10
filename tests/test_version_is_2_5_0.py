"""Pin the project version to 2.5.0 (ForwardContext refactor).

v2.5.0 retires all per-reflection module globals in
``direct_space/forward_model.py`` (#16): kernel loaders return a
``ResolutionContext``, every forward/identify entry point threads an explicit
``ForwardContext``, and simplified-geometry runs use each reflection's own
Bragg angle (``run_theta``) instead of the import-time ``theta_0`` constant.
Also ships: the multi-mode ``render_per_dislocation`` position fix (instance
labels were rendered at the origin), memory-bounded kernel generation
(``batch_size``), and the M1 Phase 2a identify-profiling toolkit
(``scripts/profile_identify.py`` + ``fanout.py --timing-json``) with its
measured laptop baseline.

No dependency, entry-point, or ``requires-python`` changes from 2.4.x, so the
conda-forge autotick bot's version-only bump suffices.
"""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_version_is_2_5_0() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == "2.5.0"
