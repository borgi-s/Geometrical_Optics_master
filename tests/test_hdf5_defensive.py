"""Layer 2 defensive: guard against silent regressions of the v1.1 cutover."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_h5py_imported_in_hdf5_module() -> None:
    """Catch rollback to a non-HDF5 writer."""
    src = (REPO / "src" / "dfxm_geo" / "io" / "hdf5.py").read_text(encoding="utf-8")
    assert "import h5py" in src, "io/hdf5.py must import h5py"


def test_pipeline_does_not_use_np_save_for_image_stack() -> None:
    """Catch accidental return-to-.npy in the forward-sim image write path.

    Note: pipeline.py still contains np.save calls in the identification
    runners (`_run_identification_single`, `_run_identification_multi`).
    Identification stays on .npy until v1.2 (per S3 decision). We assert
    that np.save calls are NOT inside run_simulation or run_postprocess.
    """
    src = (REPO / "src" / "dfxm_geo" / "pipeline.py").read_text(encoding="utf-8")
    # Find the run_simulation and run_postprocess function bodies and check
    # they don't contain `np.save(`.
    import ast

    tree = ast.parse(src)
    forward_funcs = ("run_simulation", "run_postprocess")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in forward_funcs:
            body_src = ast.get_source_segment(src, node) or ""
            assert "np.save(" not in body_src, (
                f"{node.name} contains np.save (forward-sim should write only HDF5)"
            )


def test_public_io_does_not_expose_load_images() -> None:
    """B3: legacy load_images is removed from the public package surface."""
    with pytest.raises(ImportError):
        from dfxm_geo.io.images import load_images  # noqa: F401


def test_load_images_legacy_is_internal_only() -> None:
    """The legacy helper lives in migrate, NOT re-exported from io."""
    from dfxm_geo.io import migrate

    assert hasattr(migrate, "_load_images_legacy")


def test_pipeline_does_not_import_legacy_images_module_for_reading() -> None:
    """pipeline.py only imports save_images_parallel from io.images (z-scan)."""
    src = (REPO / "src" / "dfxm_geo" / "pipeline.py").read_text(encoding="utf-8")
    # No import of load_images / load_image / load_images_parallel.
    for forbidden in ("load_images", "load_image", "load_images_parallel"):
        # Note: avoid false positive on `load_h5_scan` matching `load_image`
        # substring — check for whole-word imports.
        import re

        pattern = rf"from dfxm_geo\.io\.images import [^#\n]*\b{forbidden}\b"
        assert not re.search(pattern, src), f"pipeline.py imports {forbidden} from io.images"
