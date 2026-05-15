"""Importing the reciprocal_space modules must be side-effect-free.

Before this round, ``dfxm_geo.reciprocal_space.kernel`` ran a 1e8-ray Monte
Carlo at import time and ``.exposure`` ran a 1e6-ray loop; ``.resolution``
silently created a ``pkl_files/`` directory in the CWD. All three are now
gated behind ``if __name__ == "__main__":`` / function bodies — importing
should do nothing observable.
"""

from __future__ import annotations

import importlib
import sys


def test_kernel_module_import_is_side_effect_free(tmp_path, monkeypatch):
    """Importing dfxm_geo.reciprocal_space.kernel must not run the Monte Carlo."""
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("dfxm_geo.reciprocal_space.kernel", None)
    mod = importlib.import_module("dfxm_geo.reciprocal_space.kernel")
    assert hasattr(mod, "generate_kernel")
    assert not (tmp_path / "pkl_files").exists(), "kernel.py must not create pkl_files/ on import"


def test_exposure_module_import_is_side_effect_free(tmp_path, monkeypatch):
    """Importing dfxm_geo.reciprocal_space.exposure must not run the ray loop."""
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("dfxm_geo.reciprocal_space.exposure", None)
    mod = importlib.import_module("dfxm_geo.reciprocal_space.exposure")
    assert hasattr(mod, "run_exposure_simulation")
    assert not list(tmp_path.iterdir()), "exposure.py must not write any files on import"


def test_resolution_module_import_no_longer_creates_pkl_files(tmp_path, monkeypatch):
    """Importing dfxm_geo.reciprocal_space.resolution must not mkdir pkl_files/."""
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("dfxm_geo.reciprocal_space.resolution", None)
    importlib.import_module("dfxm_geo.reciprocal_space.resolution")
    assert not (tmp_path / "pkl_files").exists(), (
        "resolution.py must not create pkl_files/ as an import-time side effect"
    )
