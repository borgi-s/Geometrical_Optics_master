"""Verify all three identify sub-modes thread eta to AnalyticResolution (Task 17).

All three identify sub-modes (single/multi/z-scan) call run_identification(),
which calls _load_resolution(config.reciprocal) before dispatching to the
sub-runner.  Task 16 already wired _load_resolution → fm._load_analytic_resolution
→ AnalyticResolution(eta=...).  This test confirms that the same single gate
covers every identify sub-mode — i.e., Task 17 is a no-op with respect to
production code.

Verification strategy (per Task 17 spec):
  - No HDF5 goldens generated (multi-minute LUT bootstrap avoided).
  - Monkeypatch fm.Find_Hg + fm.Hg so no kernel is needed.
  - For each sub-mode, call run_identification() with a minimal config that
    carries eta=0.3 in reciprocal.  Short-circuit the sub-runner so it
    returns immediately without doing any simulation work.
  - Assert fm._analytic_eval.eta == 0.3 after the call.

The test also independently verifies the shared gate directly, matching the
style of test_forward_threads_eta_to_analytic.py (Task 16).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    IdentificationConfig,
    IdentificationMonteCarloConfig,
    IdentificationZScanConfig,
    ReciprocalConfig,
    _load_resolution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reciprocal_analytic(eta: float) -> ReciprocalConfig:
    """Analytic-backend ReciprocalConfig with the given eta, no beamstop."""
    return ReciprocalConfig(
        hkl=(-1, 1, -1),
        keV=17.0,
        backend="analytic",
        beamstop=False,
        eta=eta,
    )


def _patch_fm_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch fm so _load_analytic_resolution doesn't need a kernel on disk."""
    dummy_Hg = np.zeros((1, 3, 3))
    dummy_q = np.array([1.0, 0.0, 0.0])
    monkeypatch.setattr(fm, "Find_Hg", lambda *a, **kw: (dummy_Hg, dummy_q))
    monkeypatch.setattr(fm, "Hg", dummy_Hg)


# ---------------------------------------------------------------------------
# Direct gate test — mirrors test_forward_threads_eta_to_analytic.py
# ---------------------------------------------------------------------------


def test_identify_reciprocal_config_eta_threads_to_analytic_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_load_resolution(IdentificationConfig.reciprocal) propagates eta to _analytic_eval."""
    _patch_fm_kernel(monkeypatch)
    cfg = IdentificationConfig(reciprocal=_reciprocal_analytic(eta=0.3))
    _load_resolution(cfg.reciprocal)
    assert fm._analytic_eval is not None
    assert fm._analytic_eval.eta == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Per-sub-mode tests via run_identification entry point
# ---------------------------------------------------------------------------


def _make_identify_config(mode: str, eta: float) -> IdentificationConfig:
    """Build a minimal IdentificationConfig for the given mode."""
    kwargs: dict[str, Any] = dict(
        mode=mode,
        reciprocal=_reciprocal_analytic(eta=eta),
    )
    if mode == "multi":
        kwargs["multi"] = IdentificationMonteCarloConfig(n_samples=1)
    elif mode == "z-scan":
        kwargs["zscan"] = IdentificationZScanConfig(z_offsets_um=[0.0])
    return IdentificationConfig(**kwargs)


@pytest.mark.parametrize("mode", ["single", "multi", "z-scan"])
def test_run_identification_threads_eta_for_each_sub_mode(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """run_identification() with eta=0.3 sets fm._analytic_eval.eta=0.3 for every sub-mode.

    The sub-runner is short-circuited so no actual simulation runs.
    We only need to verify that _load_resolution is called (and thus eta is
    threaded) before the sub-runner is ever reached — which is guaranteed by
    the code structure at pipeline.py line ~1841.

    fm._analytic_eval is captured at sub-runner dispatch (inside the guard),
    since the state guard restores fm._analytic_eval to None on exit.
    """
    _patch_fm_kernel(monkeypatch)

    # Short-circuit the sub-runner dispatch so no kernel/HDF5 work runs.
    # run_identification() calls _load_resolution *before* dispatching, so
    # fm._analytic_eval is already set when this stub is invoked.
    stub_result: dict[str, Any] = {"n_images": 0, "output_dir": tmp_path}

    runner_map = {
        "single": "dfxm_geo.pipeline._run_identification_single",
        "multi": "dfxm_geo.pipeline._run_identification_multi",
        "z-scan": "dfxm_geo.pipeline._run_identification_zscan",
    }

    captured: dict[str, Any] = {}

    def _capture_stub(*args: Any, **kwargs: Any) -> dict[str, Any]:
        # Runs inside run_identification's state guard, AFTER _load_resolution
        # has threaded eta → fm._analytic_eval and BEFORE the guard restores it.
        captured["eval"] = fm._analytic_eval
        return stub_result

    from dfxm_geo.pipeline import run_identification

    with patch(runner_map[mode], side_effect=_capture_stub):
        run_identification(_make_identify_config(mode, eta=0.3), tmp_path)

    assert captured["eval"] is not None, (
        f"fm._analytic_eval was None at sub-runner dispatch (mode={mode!r}); "
        "_load_resolution did not run or did not select the analytic path."
    )
    assert captured["eval"].eta == pytest.approx(0.3), (
        f"Expected eta=0.3 in _analytic_eval, got {captured['eval'].eta} (mode={mode!r})"
    )
