"""Performance benchmarks for hot paths in dfxm_geo.

These are skipped by default (`-m "not bench"` in pyproject.toml). To run:

    pytest -m bench

Pickle-dependent paths (`forward`, `Find_Hg`, `load_or_generate_Hg`) are
deferred until a fixture kernel exists — see plan Phase 7.x.
"""

import numpy as np
import pytest

from dfxm_geo.crystal.dislocations import Fd_find
from dfxm_geo.crystal.rotations import fast_inverse2, rotatedU


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture
def random_matrix_stack(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(size=(10_000, 3, 3))


@pytest.fixture
def Ud() -> np.ndarray:
    return np.array(
        [
            [1 / np.sqrt(2), 1 / np.sqrt(3), 1 / np.sqrt(6)],
            [-1 / np.sqrt(2), 1 / np.sqrt(3), 1 / np.sqrt(6)],
            [0, -1 / np.sqrt(3), 2 / np.sqrt(6)],
        ]
    )


@pytest.fixture
def Us() -> np.ndarray:
    return np.array(
        [
            [1 / np.sqrt(2), -1 / np.sqrt(6), -1 / np.sqrt(3)],
            [0, -2 / np.sqrt(6), 1 / np.sqrt(3)],
            [-1 / np.sqrt(2), -1 / np.sqrt(6), -1 / np.sqrt(3)],
        ]
    ).T


@pytest.fixture
def Theta() -> np.ndarray:
    theta_0 = 17.953 / 2 * np.pi / 180
    return np.array(
        [
            [np.cos(theta_0), 0, np.sin(theta_0)],
            [0, 1, 0],
            [-np.sin(theta_0), 0, np.cos(theta_0)],
        ]
    )


@pytest.fixture
def rl_small(rng: np.random.Generator) -> np.ndarray:
    # ~5k lab-frame points — enough for the timing to be meaningful without
    # exercising the parallel path (which kicks in at ndis > 100).
    return rng.standard_normal(size=(3, 5_000)) * 1e-6


@pytest.mark.bench
def test_fast_inverse2_bench(benchmark, random_matrix_stack: np.ndarray) -> None:
    # Trigger the numba JIT compile outside the timed region so the first
    # call's compile cost doesn't pollute the benchmark.
    fast_inverse2(random_matrix_stack[:10].copy())
    benchmark(fast_inverse2, random_matrix_stack)


@pytest.mark.bench
def test_rotatedU_bench(benchmark, Us: np.ndarray) -> None:
    axis = np.array([1.0, 0.0, 0.0])
    benchmark(rotatedU, axis, 1e-3, Us, "sample")


@pytest.mark.bench
def test_Fd_find_small_bench(
    benchmark,
    rl_small: np.ndarray,
    Ud: np.ndarray,
    Us: np.ndarray,
    Theta: np.ndarray,
) -> None:
    # Single dislocation: hits the sequential path, not the parallel branch.
    benchmark(Fd_find, rl_small * 1e6, Ud, Us, Theta, 1.0, 1)


@pytest.mark.bench
def test_Fd_find_bipolar_wall_bench(
    benchmark,
    rl_small: np.ndarray,
    Ud: np.ndarray,
    Us: np.ndarray,
    Theta: np.ndarray,
) -> None:
    """ndis=50 exercises the numba-JIT bipolar wall accumulator.

    The first Fd_find call here compiles the JIT; we trigger that before the
    timed region so the compile cost doesn't pollute the benchmark.
    """
    Fd_find(rl_small * 1e6, Ud, Us, Theta, 1.0, 2)  # warmup JIT
    benchmark(Fd_find, rl_small * 1e6, Ud, Us, Theta, 1.0, 50)
