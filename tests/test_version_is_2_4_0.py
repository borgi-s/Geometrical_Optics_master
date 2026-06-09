"""Pin the project version to 2.4.0 (identify fan-out + colleague notebooks).

v2.4.0 ships the identification fan-out path and a colleague-ready example
suite:

- ``dfxm-identify --seed INT`` for reproducible / shardable identification runs;
- ``scripts/fanout.py --mode {forward,identify}`` and a fixed
  ``lsf/identify_array.bsub`` per-task seed (the array previously drew identical
  samples on every task);
- ``__version__`` now derives from the installed package metadata instead of a
  stale hardcoded string;
- ``examples/identification_ml_tutorial/`` (self-contained ML tutorial) and
  ``examples/cluster_showcase/`` (real cluster-sweep showcase), both with
  rendered HTML exports for read-only viewing.

No dependency, entry-point, or ``requires-python`` changes from 2.3.x, so the
conda-forge autotick bot's version-only bump suffices.
"""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_version_is_2_4_0() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == "2.4.0"
