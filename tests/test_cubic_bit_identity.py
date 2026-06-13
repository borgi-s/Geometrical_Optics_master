"""M4 Stage 4.3b Task 10: FCC + BCC byte-identity regression gate.

The FCC full-pipeline determinism test (seeded random_dislocations, MC kernel)
already lives in ``tests/test_fcc_bit_identity.py``
(``test_fcc_random_dislocations_forward_deterministic``).  This file covers the
complementary cases that 4.3b adds or could break:

  1. **BCC forward determinism** — run the BCC centered TOML (analytic backend,
     no kernel needed) TWICE into separate tmp dirs; assert the detector image is
     byte-identical (``np.array_equal``).  Exercises the per-dislocation |b|
     array path introduced in Task 3 on the cubic side of the ``is_cubic``
     branch guard.

  2. **FCC slip-system order** — assert that ``slip_systems("fcc")`` returns
     exactly the same (b, n, t) triples as ``_FCC_111_110_ORDERED``, in the
     same order.  This is the algebraic bit-identity proof: the RNG draw order
     (``rng.integers(0, 12)``) maps to the same dislocation geometry as before,
     so any FCC random_dislocations run is byte-identical to the 4.3a/pre-4.3b
     result.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm_geo.crystal.slip_systems import (
    _FCC_111_110_ORDERED,
    slip_systems,
)

# ---------------------------------------------------------------------------
# Import the canonical BCC TOML helper from the existing BCC e2e suite so we
# exercise exactly the same physics config that the DoD tests use.
# ---------------------------------------------------------------------------
from tests.test_bcc_e2e import _bcc_forward_toml

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_forward(tmp_path: Path, toml_text: str, name: str) -> np.ndarray:
    """Write *toml_text* to a config file, run forward into *tmp_path/name*,
    and return the detector image array.

    Uses ``SimulationConfig.from_toml`` + ``run_simulation`` — the same
    pattern as test_bcc_e2e — so the whole pipeline is exercised.
    """
    from dfxm_geo.pipeline import SimulationConfig, run_simulation

    # tmp_path is already a unique directory created by pytest; place the
    # config file directly in it and write HDF5 outputs to a sub-directory.
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_path / f"{name}.toml"
    cfg_path.write_text(toml_text, encoding="utf-8")
    cfg = SimulationConfig.from_toml(cfg_path)

    out = tmp_path / "out"
    run_simulation(cfg, out)

    # Locate the per-scan detector h5.  The BCC TOML writes exactly one scan
    # (scan0001) and one frame, matching test_bcc_e2e.test_bcc_forward_runs.
    det_files = sorted((out / "scan0001").glob("dfxm_sim_detector_*.h5"))
    assert det_files, f"no detector h5 found under {out / 'scan0001'}"
    det_path = det_files[0]

    with h5py.File(det_path, "r") as f:
        return f["/entry_0000/dfxm_sim_detector/image"][...]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_bcc_forward_deterministic(tmp_path: Path) -> None:
    """A BCC centered forward run is byte-identical across two executions.

    Runs the canonical BCC Fe (analytic backend, no kernel npz required) config
    twice into separate output directories and asserts ``np.array_equal``.

    This gates the per-dislocation |b| array introduced in Task 3: on the cubic
    branch the uniform-scalar ``b`` shortcut must still produce an identical
    result every call (no hidden state mutation, no floating-point ordering
    surprise from the new array plumbing).

    The BCC physics: |b| = a√3/2 for {110}<111>.  If the array path silently
    mutated |b| between runs the images would differ.
    """
    toml = _bcc_forward_toml()
    img_a = _run_forward(tmp_path / "run_a", toml, "bcc_fwd_a")
    img_b = _run_forward(tmp_path / "run_b", toml, "bcc_fwd_b")

    assert img_a.shape == img_b.shape, (
        f"BCC forward produced different shapes: {img_a.shape} vs {img_b.shape}"
    )
    assert np.isfinite(img_a).all(), "BCC forward image contains non-finite values"
    assert float(img_a.max()) > 0.0, "BCC forward image is all zeros — no contrast"
    assert np.array_equal(img_a, img_b), (
        "BCC forward is NOT byte-identical across two runs — "
        f"max diff = {np.abs(img_a.astype(float) - img_b.astype(float)).max()}"
    )


def test_fcc_slip_order_unchanged() -> None:
    """FCC slip-system order from the registry equals ``_FCC_111_110_ORDERED``.

    This is the algebraic bit-identity proof for FCC after the HCP changes:
    ``slip_systems("fcc")`` must return the SAME (b, n, t) triples in the SAME
    order as the pinned ``_FCC_111_110_ORDERED`` table, because the
    ``rng.integers(0, 12)`` draw in the population builder indexes into that
    order.  If the order drifted, every FCC random_dislocations run would
    produce a different dislocation geometry — breaking byte-identity without
    any per-pixel diff showing up until an e2e comparison.

    NOT marked slow: pure in-memory registry check, runs in <1 ms.
    """
    fcc_systems = slip_systems("fcc")
    assert len(fcc_systems) == len(_FCC_111_110_ORDERED), (
        f"slip_systems('fcc') returned {len(fcc_systems)} systems, "
        f"expected {len(_FCC_111_110_ORDERED)} from _FCC_111_110_ORDERED"
    )
    for idx, (sys_, legacy) in enumerate(zip(fcc_systems, _FCC_111_110_ORDERED, strict=True)):
        b_leg, n_leg, t_leg = legacy
        assert sys_.b == b_leg, (
            f"FCC slip system #{idx}: b={sys_.b} != legacy b={b_leg} — order drifted"
        )
        assert sys_.n == n_leg, (
            f"FCC slip system #{idx}: n={sys_.n} != legacy n={n_leg} — order drifted"
        )
        assert sys_.t == t_leg, (
            f"FCC slip system #{idx}: t={sys_.t} != legacy t={t_leg} — order drifted"
        )
