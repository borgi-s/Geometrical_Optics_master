"""M4 Stage 4.4 golden regression: lock the per-structure forward render.

Renders one pure-edge dislocation per crystal system (FCC Al / BCC W / HCP Ti)
on the analytic backend and compares the raw float32 detector image against a
committed golden. The recipe is shared with the docs/paper figure via
scripts/render_structure_showcase.py, so this test guards against silent
forward-physics drift across crystal systems.

Slow (three full-grid analytic renders); not run in the default CI suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# scripts/ is not on sys.path by default; the shared recipe module lives there.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import render_structure_showcase as rss  # noqa: E402

_GOLDEN = Path(__file__).resolve().parent / "data" / "golden" / "structure_showcase"


@pytest.mark.slow
@pytest.mark.parametrize("tag", rss.TAGS)
def test_structure_render_matches_golden(tag: str, tmp_path: Path) -> None:
    golden = np.load(_GOLDEN / f"{tag}.npy").astype(np.float64)
    img = rss.render_raw(tag, tmp_path)

    assert img.shape == golden.shape, f"{tag}: shape {img.shape} != golden {golden.shape}"
    assert np.isfinite(img).all(), f"{tag}: render has non-finite pixels"
    assert float(img.std()) > 0.0, f"{tag}: render has no contrast"

    # Analytic backend is pure-numpy deterministic; rtol guards platform float
    # noise only. atol scaled to the golden's dynamic range so near-zero
    # background pixels don't fail on relative tolerance alone.
    atol = 1e-9 + 1e-6 * float(np.abs(golden).max())
    assert np.allclose(img, golden, rtol=1e-6, atol=atol), (
        f"{tag}: render drifted from golden "
        f"(max abs diff {float(np.abs(img - golden).max()):.3e}, atol {atol:.3e})"
    )


@pytest.mark.slow
def test_fcc_render_is_deterministic(tmp_path: Path) -> None:
    """Same recipe -> bit-identical pixels on the same machine (independent of
    the stored golden's tolerance)."""
    a = rss.render_raw("fcc", tmp_path / "a")
    b = rss.render_raw("fcc", tmp_path / "b")
    assert np.array_equal(a, b), "FCC render is non-deterministic across two runs"
