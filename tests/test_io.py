"""Unit tests for dfxm_geo.io.images (round-trip save/load) and io.strain_cache."""

import os
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from unittest.mock import patch

import fabio
import numpy as np
import pytest

from dfxm_geo.io import check_folder
from dfxm_geo.io.images import load_image, load_images, load_images_parallel, save_edfs
from dfxm_geo.io.strain_cache import load_or_generate_Hg


@pytest.fixture
def small_image_stack(tmp_path):
    """Create 6 small images on disk in tmp_path.

    Arranged in a u_steps=2, v_steps=3 grid (so 6 total). Each image is a
    distinct 8x4 array, so we can verify the load function preserves order
    and content.
    """
    rng = np.random.default_rng(0)
    u_steps, v_steps = 2, 3
    dim_1, dim_2 = 8, 4
    images = []
    for k in range(u_steps * v_steps):
        img = rng.normal(size=(dim_1, dim_2))
        # Filenames are sorted; use a deterministic numeric prefix.
        fname = tmp_path / f"img_{k:04d}.npy"
        np.save(fname, img)
        images.append(img)
    return tmp_path, images, u_steps, v_steps, dim_1, dim_2


def test_load_image_round_trip(tmp_path):
    """load_image(p) returns exactly what np.save wrote to p."""
    rng = np.random.default_rng(1)
    arr = rng.normal(size=(16, 8))
    path = tmp_path / "test.npy"
    np.save(path, arr)
    loaded = load_image(str(path))
    np.testing.assert_array_equal(loaded, arr)


def test_load_images_returns_expected_shapes(small_image_stack):
    """load_images reconstructs the stack and reshape with the right dims."""
    fpath, images, u_steps, v_steps, dim_1, dim_2 = small_image_stack
    stack, stack_reshape, d1, d2 = load_images(str(fpath), u_steps, v_steps)
    assert stack.shape == (u_steps * v_steps, dim_1, dim_2)
    assert stack_reshape.shape == (u_steps, v_steps, dim_1, dim_2)
    assert d1 == dim_1
    assert d2 == dim_2


def test_load_images_preserves_content(small_image_stack):
    """Stack entries match what was saved, in sorted-filename order."""
    fpath, images, u_steps, v_steps, _, _ = small_image_stack
    stack, _, _, _ = load_images(str(fpath), u_steps, v_steps)
    for i, expected in enumerate(images):
        np.testing.assert_array_equal(stack[i], expected)


def test_load_images_parallel_matches_sequential(small_image_stack):
    """The parallel loader produces the same stack as the serial one."""
    fpath, _, u_steps, v_steps, _, _ = small_image_stack
    stack_serial, _, _, _ = load_images(str(fpath), u_steps, v_steps)
    stack_parallel, _, _, _ = load_images_parallel(str(fpath), u_steps, v_steps)
    np.testing.assert_array_equal(stack_serial, stack_parallel)


def test_load_images_raises_for_missing_dir(tmp_path):
    """A clear ValueError is raised when the directory doesn't exist."""
    with pytest.raises(ValueError, match="does not exist"):
        load_images(str(tmp_path / "no_such_dir"), 2, 2)


def test_load_images_raises_for_empty_dir(tmp_path):
    """A clear ValueError is raised when the directory has no .npy files."""
    # tmp_path is empty by default.
    with pytest.raises(ValueError, match="does not contain"):
        load_images(str(tmp_path), 2, 2)


def test_load_images_parallel_raises_for_missing_dir(tmp_path):
    """Parallel loader carries the same directory-missing guard as the serial one."""
    with pytest.raises(ValueError, match="does not exist"):
        load_images_parallel(str(tmp_path / "no_such_dir"), 2, 2)


def test_load_images_parallel_raises_for_empty_dir(tmp_path):
    """Parallel loader carries the same empty-dir guard as the serial one."""
    with pytest.raises(ValueError, match="does not contain"):
        load_images_parallel(str(tmp_path), 2, 2)


# ---------------------------------------------------------------------------
# load_or_generate_Hg
# ---------------------------------------------------------------------------


@pytest.fixture
def rotation_matrices():
    """ID06-style Ud / Us / Theta matrices for strain-field tests."""
    theta_0 = 17.953 / 2 * np.pi / 180
    Ud = np.array(
        [
            [1 / np.sqrt(2), 1 / np.sqrt(3), 1 / np.sqrt(6)],
            [-1 / np.sqrt(2), 1 / np.sqrt(3), 1 / np.sqrt(6)],
            [0, -1 / np.sqrt(3), 2 / np.sqrt(6)],
        ]
    )
    Us = np.array(
        [
            [1 / np.sqrt(2), -1 / np.sqrt(6), -1 / np.sqrt(3)],
            [0, -2 / np.sqrt(6), 1 / np.sqrt(3)],
            [-1 / np.sqrt(2), -1 / np.sqrt(6), -1 / np.sqrt(3)],
        ]
    ).T
    Theta = np.array(
        [
            [np.cos(theta_0), 0, np.sin(theta_0)],
            [0, 1, 0],
            [-np.sin(theta_0), 0, np.cos(theta_0)],
        ]
    )
    return Ud, Us, Theta


def test_load_or_generate_Hg_ndis_zero_is_identity_minus_I(rotation_matrices):
    """For ndis=0 the strain field is identity, so Hg = I^T - I = 0."""
    Ud, Us, Theta = rotation_matrices
    rl = np.zeros((3, 100))
    rl[0] = np.linspace(-1, 1, 100) * 1e-6
    Hg = load_or_generate_Hg(rl, Ud, Us, Theta, dis=1.0, ndis=0)
    assert Hg.shape == (100, 3, 3)
    np.testing.assert_allclose(Hg, 0, atol=1e-12)


def test_load_or_generate_Hg_caches_to_disk(rotation_matrices, tmp_path):
    """First call computes + saves Fg; second call loads from disk."""
    Ud, Us, Theta = rotation_matrices
    rng = np.random.default_rng(0)
    rl = rng.normal(size=(3, 200)) * 1e-6
    cache = tmp_path / "Fg_cache.npy"

    Hg_a = load_or_generate_Hg(rl, Ud, Us, Theta, dis=1.0, ndis=1, file_path=str(cache))
    assert cache.exists(), "cache file should exist after first call"

    Hg_b = load_or_generate_Hg(rl, Ud, Us, Theta, dis=1.0, ndis=1, file_path=str(cache))
    np.testing.assert_array_equal(Hg_a, Hg_b)


def test_load_or_generate_Hg_ndis_zero_with_missing_cache_returns_zeros(
    rotation_matrices, tmp_path
):
    """When file_path points to a non-existent file and ndis=0, the function
    falls through the FileNotFoundError branch and returns the identity-minus-I
    short-circuit (Hg = 0) without invoking Fd_find."""
    Ud, Us, Theta = rotation_matrices
    rl = np.zeros((3, 50))
    rl[0] = np.linspace(-1, 1, 50) * 1e-6
    missing = tmp_path / "does_not_exist_yet.npy"
    assert not missing.exists()

    Hg = load_or_generate_Hg(rl, Ud, Us, Theta, dis=1.0, ndis=0, file_path=str(missing))

    assert Hg.shape == (50, 3, 3)
    np.testing.assert_allclose(Hg, 0, atol=1e-12)
    assert missing.exists(), "cache file should be written after generation"


def test_load_or_generate_Hg_in_memory_path_matches_cached(rotation_matrices, tmp_path):
    """file_path=None (in-memory) and file_path=<cache> produce identical Hg."""
    Ud, Us, Theta = rotation_matrices
    rng = np.random.default_rng(1)
    rl = rng.normal(size=(3, 150)) * 1e-6
    cache = tmp_path / "Fg_cache.npy"

    Hg_in_memory = load_or_generate_Hg(rl, Ud, Us, Theta, dis=2.0, ndis=1)
    Hg_cached = load_or_generate_Hg(rl, Ud, Us, Theta, dis=2.0, ndis=1, file_path=str(cache))
    np.testing.assert_allclose(Hg_in_memory, Hg_cached, atol=1e-12)


# ---------------------------------------------------------------------------
# check_folder
# ---------------------------------------------------------------------------


def test_check_folder_creates_missing_dir(tmp_path):
    """check_folder mkdirs the target if it does not exist."""
    target = "new_subfolder"
    assert not (tmp_path / target).exists()
    check_folder(str(tmp_path), target)
    assert (tmp_path / target).is_dir()


def test_check_folder_no_op_when_exists(tmp_path):
    """check_folder is idempotent — calling on an existing dir does not raise."""
    target = "preexisting"
    (tmp_path / target).mkdir()
    # Should not raise.
    check_folder(str(tmp_path), target)
    assert (tmp_path / target).is_dir()


# ---------------------------------------------------------------------------
# save_edfs (round-trip through fabio)
# ---------------------------------------------------------------------------


def test_save_edfs_writes_one_file_per_uv_pair(tmp_path):
    """save_edfs writes u_steps * v_steps .edf files into the target directory."""
    u_steps, v_steps, h, w = 2, 3, 5, 4
    rng = np.random.default_rng(0)
    # imstack is indexed as imstack[j, i] for j in [0, u_steps), i in [0, v_steps).
    imstack = rng.integers(0, 1000, size=(u_steps, v_steps, h, w)).astype(np.float64)
    v = np.linspace(0.1, 0.2, v_steps)
    u = np.linspace(-0.05, 0.05, u_steps)

    fpath = str(tmp_path)
    fn_prefix = "/sim_"
    result = save_edfs(imstack, v, u, fpath, fn_prefix)
    assert result is True

    edf_files = sorted(f for f in os.listdir(tmp_path) if f.endswith(".edf"))
    assert len(edf_files) == u_steps * v_steps


def test_save_edfs_round_trips_pixel_data(tmp_path):
    """A saved EDF can be read back with fabio and yields the same pixel values."""
    h, w = 4, 3
    expected_img = np.arange(h * w).reshape(h, w).astype(np.float64)
    imstack = expected_img[np.newaxis, np.newaxis, :, :]
    v = np.array([0.1])
    u = np.array([0.05])

    save_edfs(imstack, v, u, str(tmp_path), "/sim_")
    written = next(f for f in os.listdir(tmp_path) if f.endswith(".edf"))
    loaded = fabio.open(str(tmp_path / written)).data
    np.testing.assert_array_equal(loaded.astype(np.float64), expected_img)


def test_save_images_parallel_uses_explicit_max_workers(tmp_path, monkeypatch):
    """The max_workers kwarg overrides env var and auto-default."""
    import dfxm_geo.io.images as images_mod

    # Mock forward() so we don't need the real kernel.
    monkeypatch.setattr(images_mod, "forward", lambda Hg, phi=0, chi=0: np.zeros((4, 4)))

    captured = {}

    class _SpyExecutor(_ThreadPoolExecutor):
        def __init__(self, max_workers=None, **kwargs):
            captured["max_workers"] = max_workers
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setenv("DFXM_MAX_WORKERS", "99")  # env var should be ignored
    with patch.object(images_mod, "ThreadPoolExecutor", _SpyExecutor):
        images_mod.save_images_parallel(
            Hg=np.zeros((1, 3, 3)),
            phi_range=0.01,
            phi_steps=2,
            chi_range=0.01,
            chi_steps=2,
            fpath=str(tmp_path),
            fn_prefix="/x_",
            ftype=".npy",
            max_workers=3,
        )
    assert captured["max_workers"] == 3


def test_save_images_parallel_falls_back_to_env_var(tmp_path, monkeypatch):
    """When max_workers is None, DFXM_MAX_WORKERS env var is honored."""
    import dfxm_geo.io.images as images_mod

    monkeypatch.setattr(images_mod, "forward", lambda Hg, phi=0, chi=0: np.zeros((4, 4)))

    captured = {}

    class _SpyExecutor(_ThreadPoolExecutor):
        def __init__(self, max_workers=None, **kwargs):
            captured["max_workers"] = max_workers
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setenv("DFXM_MAX_WORKERS", "5")
    with patch.object(images_mod, "ThreadPoolExecutor", _SpyExecutor):
        images_mod.save_images_parallel(
            Hg=np.zeros((1, 3, 3)),
            phi_range=0.01,
            phi_steps=2,
            chi_range=0.01,
            chi_steps=2,
            fpath=str(tmp_path),
            fn_prefix="/x_",
            ftype=".npy",
        )
    assert captured["max_workers"] == 5


def test_auto_max_workers_returns_positive_int(monkeypatch):
    """_auto_max_workers must always return a positive int regardless of psutil presence."""
    from dfxm_geo.io.images import _auto_max_workers

    monkeypatch.delenv("DFXM_MAX_WORKERS", raising=False)
    val = _auto_max_workers()
    assert isinstance(val, int)
    assert val >= 1


def test_auto_max_workers_env_var_overrides(monkeypatch):
    """DFXM_MAX_WORKERS env var takes precedence in _auto_max_workers."""
    from dfxm_geo.io.images import _auto_max_workers

    monkeypatch.setenv("DFXM_MAX_WORKERS", "7")
    assert _auto_max_workers() == 7

    # Invalid values fall back to the auto-computed default.
    monkeypatch.setenv("DFXM_MAX_WORKERS", "not-an-int")
    val = _auto_max_workers()
    assert isinstance(val, int) and val >= 1

    monkeypatch.setenv("DFXM_MAX_WORKERS", "0")
    val = _auto_max_workers()
    assert isinstance(val, int) and val >= 1  # zero/negative falls back


def test_save_image_does_not_misunpack_forward(tmp_path, monkeypatch):
    """save_image unpacked forward() as a 2-tuple; this regression pins the single-array contract.

    Mocks forward() to return a single np.ndarray (the default) and confirms
    save_image writes the file without raising ValueError.
    """
    import dfxm_geo.io.images as images_mod

    expected_array = np.arange(12, dtype=float).reshape(3, 4)

    def fake_forward(Hg, phi=0.0, chi=0.0, *args, **kwargs):
        return expected_array

    monkeypatch.setattr(images_mod, "forward", fake_forward)
    args = (
        np.zeros((1, 3, 3)),  # Hg (unused by fake_forward)
        0.0,  # phi
        0.0,  # chi
        0,  # j
        0,  # i
        str(tmp_path),  # fpath
        "/test_",  # fn_prefix
        ".npy",  # ftype
    )
    images_mod.save_image(args)

    out_file = tmp_path / "test_0000_0000.npy"
    assert out_file.is_file()
    loaded = np.load(out_file)
    np.testing.assert_array_equal(loaded, expected_array)
