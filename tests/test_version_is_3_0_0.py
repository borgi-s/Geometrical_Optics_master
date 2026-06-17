"""Pin the project version to 3.0.0 (arbitrary crystal structures + realistic detector).

v3.0.0 is a MAJOR release (see docs/release-notes-3.0.0.md):

- Crystal structures: CIF ingestion (lazy gemmi), FCC/BCC/HCP slip-system
  registry, general (triclinic-capable) cell geometry, structure-factor
  extinction, per-material elasticity.
- Reflections: multi-reflection ``[[reflections]]`` + ``dfxm-find-reflections``.
- Geometry: symmetric + oblique; per-axis scan primitives; centered/random/wall.
- Optics: MC vs analytic resolution backends; beam-stop + apertures in the BFP.
- Detector: measured uint16-ADU PCO-edge model REPLACES ``[noise]`` (breaking,
  ``DetectorConfig`` replaces ``IdentificationNoiseConfig``); ``exposure_time`` is
  a user-tunable brightness knob; ``model="ideal"`` for raw float32.
- Identification pipeline + ``dfxm_geo.scoring`` cross-correlation.
- BLISS HDF5 outputs + migration tools; fan-out / drafted LSF cluster scripts.

CONDA-FORGE SYNC at publish: the feedstock ``build.python.entry_points`` must
list all SEVEN scripts (note ``dfxm-find-reflections`` added since 2.5.x), and
``gemmi`` is an OPTIONAL ``[cif]`` extra (NOT in ``requirements: run`` — keep the
package ``noarch: python``). ``counts_scale`` ships provisional (``1.0e4``); see
docs/calibration/counts_scale_rocking_study_2026-06-16.md.
"""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_version_is_3_0_0() -> None:
    with (REPO / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == "3.0.0"
