"""Pin the project version to 2.5.1 (identify-throughput patch: pool + dedup + fused Hg).

v2.5.1 is the M1 Phase 2b performance arc, shipped as a patch release because
the >=5x throughput DoD awaits cluster validation (laptop measured 3.87x):

- W2: all three identify orchestrators route Hg through the single
  ``find_hg_scene`` seam; each dislocation's deformation field is computed
  once per scene (``engine="numpy"`` is bit-identical to the legacy path).
- W3: fused numba kernels (``_scene_perdis_hg_kernel`` + the Phase-1
  population kernel) are the default engine; outputs differ from the NumPy
  oracle only at ~1e-15.
- W1: ``scripts/fanout.py`` runs a persistent worker pool by default
  (``--isolate`` restores subprocess-per-config); pool and isolate outputs
  are bit-identical.

No dependency, entry-point, or ``requires-python`` changes from 2.5.0, so the
conda-forge autotick bot's version-only bump suffices (if PyPI publish is
approved later; v2.5.1 is tag+push only for now).
"""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_version_is_2_5_1() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == "2.5.1"
