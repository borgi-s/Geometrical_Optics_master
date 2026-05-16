"""Performance benchmarks for hot paths in dfxm_geo.

These are skipped by default (`-m "not bench"` in pyproject.toml). To run:

    pytest -m bench

Pickle-dependent paths (`forward`, `Find_Hg`, `load_or_generate_Hg`) are
deferred until a fixture kernel exists — see plan Phase 7.x.
"""

from pathlib import Path

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


@pytest.mark.bench
def test_Fd_find_parallel_bench(
    benchmark,
    rl_small: np.ndarray,
    Ud: np.ndarray,
    Us: np.ndarray,
    Theta: np.ndarray,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ndis=151 (the production default) routes through the parallel branch.

    Workers share the same numba-JIT'd `_accumulate_bipolar_walls` as the
    sequential branch. We warm the JIT and stash the chunk debug print
    before the timed region.
    """
    Fd_find(rl_small * 1e6, Ud, Us, Theta, 1.0, 151)  # warmup JIT + threads
    capsys.readouterr()  # drop the "print(chunks)" output from the warmup
    benchmark(Fd_find, rl_small * 1e6, Ud, Us, Theta, 1.0, 151)


@pytest.fixture
def _scatter_inputs(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Production-shaped inputs for the forward() scatter benchmark.

    Detector image is 510 x 170 (the post-Nsub output of forward()). The
    valid-ray count (200k) is a representative idx.sum() in a real run —
    most NN1*NN2*NN3 grid points fall outside the Resq_i lookup mask.
    """
    H, W = 510, 170
    n_valid = 200_000
    flat_indices = rng.integers(0, H * W, size=n_valid, dtype=np.int64)
    weights = rng.standard_normal(n_valid).astype(np.float32)
    return flat_indices, weights, np.zeros((H, W))


@pytest.mark.bench
def test_bincount_scatter_bench(
    benchmark,
    _scatter_inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """The Phase 8 bincount-based scatter replacing np.add.at in forward()."""
    flat_indices, weights, _ = _scatter_inputs

    def _run() -> None:
        contribution = np.bincount(flat_indices, weights=weights, minlength=510 * 170)
        contribution.reshape((510, 170))

    benchmark(_run)


@pytest.mark.bench
def test_np_add_at_scatter_bench(
    benchmark,
    _scatter_inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """The pre-Phase 8 np.add.at scatter, kept as a regression reference.

    Should be ~5-10x slower than `test_bincount_scatter_bench` on the same
    inputs. If they ever come within 2x of each other, numpy made np.add.at
    a lot faster and we can consider reverting (the old code was simpler).
    """
    flat_indices, weights, im = _scatter_inputs

    def _run() -> None:
        rows, cols = np.divmod(flat_indices, 170)
        np.add.at(im, (rows, cols), weights)

    benchmark(_run)


@pytest.mark.bench
def test_load_default_kernel_bench(benchmark, tmp_path: Path) -> None:
    """Regression baseline for _load_default_kernel on the canonical npz size.

    The canonical kernel is (400, 200, 200) float64 = 128 MB; cold np.load
    typically lands at 100-300 ms on Sina's laptop. `pytest -m bench
    --benchmark-compare=baseline_v1_0_1` will flag regressions beyond the
    saved baseline's std-dev threshold.
    """
    import dfxm_geo.direct_space.forward_model as fm

    rng = np.random.default_rng(0)
    dst = tmp_path / "Resq_i_bench.npz"
    np.savez(
        dst,
        Resq_i=rng.random((400, 200, 200), dtype=np.float64),
        qi1_range=np.float64(5e-4),
        qi2_range=np.float64(0.75e-2),
        qi3_range=np.float64(0.75e-2),
        npoints1=np.int64(400),
        npoints2=np.int64(200),
        npoints3=np.int64(200),
    )

    # Save all 15 globals that _load_default_kernel mutates for clean state restoration
    saved = {
        "Resq_i": fm.Resq_i,
        "qi1_range": fm.qi1_range,
        "qi2_range": fm.qi2_range,
        "qi3_range": fm.qi3_range,
        "npoints1": fm.npoints1,
        "npoints2": fm.npoints2,
        "npoints3": fm.npoints3,
        "qi1_start": fm.qi1_start,
        "qi2_start": fm.qi2_start,
        "qi3_start": fm.qi3_start,
        "qi1_step": fm.qi1_step,
        "qi2_step": fm.qi2_step,
        "qi3_step": fm.qi3_step,
        "qi_starts": fm.qi_starts,
        "qi_steps": fm.qi_steps,
    }
    try:
        benchmark(fm._load_default_kernel, pkl_path=str(dst), compute_Hg=False)
    finally:
        # Restore all 15 globals to prevent state leakage to subsequent tests
        for key, val in saved.items():
            setattr(fm, key, val)
