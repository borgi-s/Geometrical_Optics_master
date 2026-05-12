"""Unit tests for dfxm_geo.io.images (round-trip save/load)."""

import numpy as np
import pytest

from dfxm_geo.io.images import load_image, load_images, load_images_parallel


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
