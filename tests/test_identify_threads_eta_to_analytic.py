"""Verify all three identify sub-modes thread eta to AnalyticResolution (Task 17).

All three identify sub-modes (single/multi/z-scan) call run_identification(),
which calls _load_resolution(config.reciprocal) before dispatching to the
sub-runner.  Task 16 already wired _load_resolution → fm._load_analytic_resolution
→ AnalyticResolution(eta=...).  This test confirms that the same single gate
covers every identify sub-mode — i.e., Task 17 is a no-op with respect to
production code.

Verification strategy (per Task 17 spec):
  - No HDF5 goldens generated (multi-minute LUT bootstrap avoided).
  - For each sub-mode, call run_identification() with a minimal config that
    carries eta=0.3 in reciprocal.  Short-circuit the sub-runner so it
    returns immediately without doing any simulation work.
  - Assert the ForwardContext handed to the sub-runner carries
    ctx.resolution.analytic_eval.eta == 0.3.

#16 Slice 5: _load_analytic_resolution no longer sets a module global; it
RETURNS a ResolutionContext threaded into the ForwardContext that
run_identification() passes to the sub-runner. We capture that ctx in the
stub rather than reading a (deleted) fm._analytic_eval global.

The test also independently verifies the shared gate directly, matching the
style of test_forward_threads_eta_to_analytic.py (Task 16).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

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


# ---------------------------------------------------------------------------
# Direct gate test — mirrors test_forward_threads_eta_to_analytic.py
# ---------------------------------------------------------------------------


def test_identify_reciprocal_config_eta_threads_to_analytic_eval() -> None:
    """_load_resolution(IdentificationConfig.reciprocal) propagates eta to analytic_eval.

    #16 Slice 5: _load_resolution RETURNS a ResolutionContext; the analytic
    backend lives on res.analytic_eval (no module global).
    """
    cfg = IdentificationConfig(reciprocal=_reciprocal_analytic(eta=0.3))
    res = _load_resolution(cfg.reciprocal)
    assert res.analytic_eval is not None
    assert res.analytic_eval.eta == pytest.approx(0.3)


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
    tmp_path: Path,
) -> None:
    """run_identification() with eta=0.3 threads eta=0.3 into the sub-runner ctx.

    The sub-runner is short-circuited so no actual simulation runs.
    We only need to verify that _load_resolution is called (and thus eta is
    threaded into the ForwardContext) before the sub-runner is ever reached.

    #16 Slice 5: run_identification builds a ForwardContext from the
    _load_resolution return value and passes it as the 3rd positional arg to
    each sub-runner. We capture that ctx in the stub and assert on
    ctx.resolution.analytic_eval.eta (no module global).
    """
    # Short-circuit the sub-runner dispatch so no kernel/HDF5 work runs.
    # run_identification() calls _load_resolution + build_forward_context
    # *before* dispatching, so ctx is already built when this stub is invoked.
    stub_result: dict[str, Any] = {"n_images": 0, "output_dir": tmp_path}

    runner_map = {
        "single": "dfxm_geo.orchestrator._run_identification_single",
        "multi": "dfxm_geo.orchestrator._run_identification_multi",
        "z-scan": "dfxm_geo.orchestrator._run_identification_zscan",
    }

    captured: dict[str, Any] = {}

    def _capture_stub(*args: Any, **kwargs: Any) -> dict[str, Any]:
        # Sub-runner signature is (config, output_dir, ctx); ctx is args[2].
        ctx = kwargs.get("ctx", args[2] if len(args) > 2 else None)
        captured["eval"] = ctx.resolution.analytic_eval if ctx is not None else None
        return stub_result

    from dfxm_geo.pipeline import run_identification

    with patch(runner_map[mode], side_effect=_capture_stub):
        run_identification(_make_identify_config(mode, eta=0.3), tmp_path)

    assert captured["eval"] is not None, (
        f"ctx.resolution.analytic_eval was None at sub-runner dispatch (mode={mode!r}); "
        "_load_resolution did not run or did not select the analytic path."
    )
    assert captured["eval"].eta == pytest.approx(0.3), (
        f"Expected eta=0.3 in analytic_eval, got {captured['eval'].eta} (mode={mode!r})"
    )
