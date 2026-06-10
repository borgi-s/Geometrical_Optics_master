"""In-process config runner for `scripts/fanout.py` pool mode (v2.6.0 W1).

Lives inside the package (not in scripts/) so `ProcessPoolExecutor` can
pickle `run_one` by module path under Windows spawn. Keep this module's
import light: `dfxm_geo.pipeline` (the heavy import) is loaded lazily
inside `run_one` so its cost is measured and attributed, exactly like the
subprocess path's DFXM_TIMING contract.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

_import_timed = False


def _timed_pipeline_import() -> float:
    """Import dfxm_geo.pipeline; return the seconds it took.

    Returns the measured time on this process's first call and exactly 0.0
    afterwards — so per-config DFXM_TIMING rows show the worker's one-time
    import cost on its first config and 0.0 on the warm ones.
    """
    global _import_timed
    if _import_timed:
        return 0.0
    t0 = time.perf_counter()
    import dfxm_geo.pipeline  # noqa: F401  (heavy: numpy/numba/h5py chain)

    _import_timed = True
    return time.perf_counter() - t0


def _resolve_cli(mode: str) -> tuple[Callable[[list[str] | None], int], list[str]]:
    """Map fanout mode -> (CLI callable, extra argv). Mirrors fanout.build_cmd."""
    from dfxm_geo.pipeline import cli_main, cli_main_identify

    if mode == "forward":
        return cli_main, ["--no-postprocess"]
    if mode == "identify":
        return cli_main_identify, []
    raise ValueError(f"mode must be 'forward' or 'identify', got {mode!r}")


def run_one(mode: str, config: str, output_dir: str, log_path: str) -> dict[str, Any]:
    """Run one config in-process; the pool-mode analogue of one subprocess.

    Contract (mirrors `scripts/fanout.py` `_CHILD_SNIPPET` + `_default_runner`):
    - stdout AND stderr (tqdm writes to stderr) go to `log_path`;
    - a `DFXM_TIMING {json}` line with import_s/run_s is printed into the log
      so `fanout.parse_timing_log` works unchanged in pool mode;
    - every exception after the log file opens is contained: returns
      ``{"returncode": -1}`` with the traceback in the log (batch resilience
      — one bad config never kills the sweep). SystemExit codes (argparse
      errors) are propagated as rc. A failure to open the log itself (e.g.
      PermissionError) propagates to the caller — the pool runner contains it.

    Args are strings (not Path) so the cross-process pickle stays trivial.
    Returns ``{"returncode": int, "wall_s": float}``.
    """
    t_wall = time.perf_counter()
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as fh, redirect_stdout(fh), redirect_stderr(fh):
        try:
            import_s = _timed_pipeline_import()
            cli, extra = _resolve_cli(mode)
            # argv is always a concrete list here; the CLIs' None default
            # (sys.argv mode) is never exercised from the pool path.
            argv = ["--config", config, "--output", output_dir] + extra
            t0 = time.perf_counter()
            rc = int(cli(argv) or 0)
            run_s = time.perf_counter() - t0
            print(
                "DFXM_TIMING "
                + json.dumps({"import_s": round(import_s, 3), "run_s": round(run_s, 3)})
            )
        except SystemExit as exc:  # argparse exits with code 2 on bad args
            code = exc.code
            rc = code if isinstance(code, int) else (0 if code is None else 1)
            if rc != 0:
                traceback.print_exc()
        except Exception:
            traceback.print_exc()
            rc = -1
    return {"returncode": rc, "wall_s": time.perf_counter() - t_wall}
