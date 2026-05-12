"""Pytest fixtures shared across the test suite."""
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent / "data" / "golden"


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    """Directory containing reference outputs for smoke tests."""
    return GOLDEN_DIR
