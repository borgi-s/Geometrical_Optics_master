"""Pytest fixtures shared across the test suite."""

from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent / "data" / "golden"


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    """Directory containing reference outputs for smoke tests."""
    return GOLDEN_DIR


@pytest.fixture(autouse=True)
def _bound_kernel_cache_memory():
    """Clear pipeline._KERNEL_CTX_CACHE after each test to bound process memory.

    #16 Slice 5 deleted ``_forward_state_guard``, which used to snapshot/restore
    (and thus free) the loaded resolution kernel after every run. The idempotent
    kernel cache now *retains* the loaded LUT (~100+ MB for the canonical npz)
    for the whole process — correct for production (one kernel per worker), but
    in a single pytest process running the full px510 suite the retained LUT
    starves later memory-heavy tests and spawned subprocesses (fanout). Clearing
    between tests restores the pre-deletion memory profile; production is
    untouched (it never clears).
    """
    yield
    import dfxm_geo.pipeline as _pipeline

    _pipeline._KERNEL_CTX_CACHE.clear()


@pytest.fixture(autouse=True)
def _restore_slip_registry():
    """Snapshot and restore slip_systems._REGISTRY around every test.

    Prevents custom structures registered in one test from leaking into others
    (e.g. register_custom calls in test_crystal_structure_config.py).
    """
    from dfxm_geo.crystal import slip_systems as _ss

    snapshot = dict(_ss._REGISTRY)
    yield
    _ss._REGISTRY.clear()
    _ss._REGISTRY.update(snapshot)
