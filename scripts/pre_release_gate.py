#!/usr/bin/env python
"""Pre-release gate for dfxm_geo.

Run this before tagging a release. It runs the two correctness gates that the
default `pytest -q` run does NOT cover, and exits nonzero if either fails:

  (a) The @slow MC-vs-analytic OBLIQUE parity test
      (tests/test_mc_vs_analytic_oblique_parity.py). The default pyproject.toml
      addopts deselect `slow`, so this never runs in `pytest -q` or CI. It
      guards that the analytic resolution backend matches the MC LUT at
      arbitrary (theta, eta) -- i.e. the oblique geometry is wired correctly.

  (b) The Fd_find golden + forward bit-equivalence goldens
      (tests/test_dislocations_smoke.py::test_Fd_find_matches_golden and
      tests/test_hdf5_bit_equiv.py::test_hdf5_writer_bit_equivalent_to_legacy_npy_golden).
      These are the numerical safety net under the whole cleanup: the Fd_find
      smoke golden and the legacy-writer forward golden.

Usage (run with the venv python -- bare `python` is Python 2.7 on this box):

    C:\\Users\\borgi\\Documents\\GM-reworked\\.venv\\Scripts\\python.exe \\
        scripts/pre_release_gate.py

    # or, with the venv already active:
    python scripts/pre_release_gate.py

    python scripts/pre_release_gate.py --help   # show this and exit

Exit codes:
    0  both gates passed
    1  one or both gates failed (see the per-gate output above the summary)

Notes:
    - This intentionally re-uses the running interpreter (sys.executable), so
      invoking it with the venv python runs the gates under the venv python.
    - `-m slow` is passed explicitly to override the default `-m 'not ... slow'`
      deselection for gate (a). Gate (b) selects its two goldens by node id.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Repo root = parent of this scripts/ directory.
_REPO = Path(__file__).resolve().parents[1]

# Gate (a): the @slow MC-vs-analytic oblique parity test. Run with `-m slow`
# to override the default deselection of the slow marker.
_OBLIQUE_PARITY_TEST = "tests/test_mc_vs_analytic_oblique_parity.py"

# Gate (b): the Fd_find golden + the forward bit-equivalence golden. Selected
# by explicit node id so the gate is robust to other tests in those files.
_GOLDEN_TESTS = [
    "tests/test_dislocations_smoke.py::test_Fd_find_matches_golden",
    "tests/test_hdf5_bit_equiv.py::test_hdf5_writer_bit_equivalent_to_legacy_npy_golden",
]


def _run_pytest(label: str, args: list[str]) -> int:
    """Run pytest with `args` under the current interpreter; return its rc."""
    cmd = [sys.executable, "-m", "pytest", *args]
    print(f"\n=== {label} ===")
    print("  $ " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=_REPO)
    rc = completed.returncode
    print(f"  -> {label}: {'PASS' if rc == 0 else f'FAIL (rc={rc})'}")
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-release gate: run the @slow oblique MC-vs-analytic parity "
            "test and the Fd_find / forward bit-equiv goldens. Exits nonzero "
            "if either fails."
        ),
    )
    # No options beyond --help; parse so `--help` works and unknown args error.
    parser.parse_args(argv)

    # Gate (a): @slow oblique parity. `-m slow` overrides the default
    # `-m 'not bench and not slow'` deselection from pyproject.toml.
    rc_parity = _run_pytest(
        "Gate A: @slow MC-vs-analytic oblique parity",
        ["-q", "-m", "slow", _OBLIQUE_PARITY_TEST],
    )

    # Gate (b): Fd_find golden + forward bit-equiv golden, by node id.
    rc_golden = _run_pytest(
        "Gate B: Fd_find golden + forward bit-equiv golden",
        ["-q", *_GOLDEN_TESTS],
    )

    overall = 0 if (rc_parity == 0 and rc_golden == 0) else 1
    print("\n=== pre-release gate summary ===")
    print(f"  Gate A (oblique parity): {'PASS' if rc_parity == 0 else 'FAIL'}")
    print(f"  Gate B (goldens):        {'PASS' if rc_golden == 0 else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if overall == 0 else 'FAIL'}")
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
