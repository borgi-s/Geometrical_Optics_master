"""scripts/profile_identify.py — per-stage timing breakdown for identify runs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import dfxm_geo.direct_space.forward_model as fm

# Load scripts/profile_identify.py as a module (scripts/ is not a package).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile_identify.py"
_spec = importlib.util.spec_from_file_location("profile_identify", _SCRIPT)
profile_identify = importlib.util.module_from_spec(_spec)
sys.modules["profile_identify"] = profile_identify
_spec.loader.exec_module(profile_identify)


_TINY_MULTI_TOML = """\
mode = "multi"

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0

[scan.phi]
value = 1.25e-4
range = 1.25e-4
steps = 2

[noise]
poisson_noise = true
rng_seed = 1

[multi]
n_samples = 1
pos_std_um = 5.0

[io]
include_perfect_crystal = false
write_strain_provenance = false
"""


def test_profile_one_reports_stage_breakdown(tmp_path: Path) -> None:
    """profile_one must run a real identify config single-threaded and return
    the M1 Phase 2a stage split: kernel load vs Hg vs frames vs write vs noise,
    with the buckets consistent with the measured total."""
    kernel_dir = Path(fm.pkl_fpath)
    if not sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz")):
        pytest.skip("No bootstrapped kernel npz found; skipping profile run.")

    cfg = tmp_path / "tiny_multi.toml"
    cfg.write_text(_TINY_MULTI_TOML, encoding="utf-8")
    out = tmp_path / "prof"

    result = profile_identify.profile_one(cfg, out, skip_cprofile_dump=False)

    for key in (
        "total_s",
        "cold_start_s",
        "kernel_load_s",
        "hg_s",
        "frames_s",
        "poisson_s",
        "io_other_s",
        "unattributed_s",
        "stage_calls",
    ):
        assert key in result, key
    assert result["total_s"] > 0
    # The tiny config really computed Hg and frames.
    assert result["hg_s"] > 0
    assert result["frames_s"] > 0
    assert result["stage_calls"]["frames"] == 2  # 2 phi frames, 1 sample
    # Buckets are derived by subtraction — they must not go negative and must
    # sum (with the residual) back to the total.
    assert result["io_other_s"] >= 0
    assert result["unattributed_s"] >= 0
    recomposed = (
        result["kernel_load_s"]
        + result["hg_s"]
        + result["frames_s"]
        + result["poisson_s"]
        + result["io_other_s"]
        + result["unattributed_s"]
    )
    assert recomposed == pytest.approx(result["total_s"], rel=0.05)
    # Artifacts for the docs baseline.
    assert (out / "stage_timings.json").is_file()
    assert (out / "profile_summary.txt").is_file()
    assert (out / "identify_single_thread.prof").is_file()
