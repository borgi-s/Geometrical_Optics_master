"""Bundled data shipped inside the dfxm_geo wheel (config templates, etc.).

`configs_root()` and `iter_config_files()` are the single source of truth for
where the bundled TOML templates live. Both the `dfxm-init` CLI and the test
suite consume them so they cannot drift from what actually ships.
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Iterator
from pathlib import Path


def configs_root() -> Path:
    """Filesystem path to the bundled `configs/` template directory."""
    return Path(str(importlib.resources.files("dfxm_geo.data").joinpath("configs")))


def iter_config_files() -> Iterator[tuple[str, Path]]:
    """Yield `(relative_posix_path, absolute_path)` for every bundled `.toml`.

    Recurses into subdirectories (e.g. `variants/`). The relative path uses
    forward slashes regardless of platform.
    """
    root = configs_root()
    for path in sorted(root.rglob("*.toml")):
        rel = path.relative_to(root).as_posix()
        yield rel, path
