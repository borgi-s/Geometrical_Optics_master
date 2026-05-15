"""Run a DFXM forward simulation from a TOML config file.

Usage:
    python scripts/run_forward.py --config configs/default.toml --output ./out

After `pip install -e .`, the same entry point is available as `dfxm-forward`.
"""

import sys

from dfxm_geo.pipeline import cli_main

if __name__ == "__main__":
    sys.exit(cli_main())
