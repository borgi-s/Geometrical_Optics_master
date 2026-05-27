"""Unit tests for dfxm_geo.io.images (save_images_parallel writer) and io.strain_cache.

Note: load_image, load_images, load_images_parallel, and save_edfs were removed
in v1.1.0.  Their tests are deleted here accordingly.  The HDF5 equivalents live
in io/hdf5.py; the migration helper is in io/migrate.py.
"""

from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest

from dfxm_geo.io import check_folder
from dfxm_geo.io.strain_cache import load_or_generate_Hg

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


def test_load_or_generate_Hg_regenerates_on_shape_mismatch(rotation_matrices, tmp_path):
    """A cached Fg whose shape[0] mismatches rl.shape[1] is discarded.

    Regression guard: the on-disk cache key in `forward_model.Find_Hg` is keyed
    by physics params (dis/psize/zl_rms) plus the ray-grid params (Npixels/Nsub).
    If a stale cache from a different ray grid is found anyway — e.g. the user
    deletes one of the new key tokens, or a path collision — the shape guard
    inside `load_or_generate_Hg` must regenerate rather than silently load wrong
    shape data.
    """
    Ud, Us, Theta = rotation_matrices
    cache = tmp_path / "Fg_cache.npy"

    # Plant a stale cache shaped for 200 rays.
    rng = np.random.default_rng(2)
    stale_Fg = rng.normal(size=(200, 3, 3))
    np.save(cache, stale_Fg)
    assert cache.exists()

    # Call with rl of a *different* size (300 rays). The shape guard must
    # discard the stale cache and regenerate so the returned Hg has the right
    # leading dimension.
    rl = rng.normal(size=(3, 300)) * 1e-6
    Hg = load_or_generate_Hg(rl, Ud, Us, Theta, dis=1.0, ndis=1, file_path=str(cache))
    assert Hg.shape == (300, 3, 3), "shape must match rl, not stale cache"

    # The cache on disk should now be the regenerated 300-ray Fg, not the
    # stale 200-ray one — so a second call with the same rl loads cleanly.
    Hg2 = load_or_generate_Hg(rl, Ud, Us, Theta, dis=1.0, ndis=1, file_path=str(cache))
    np.testing.assert_array_equal(Hg, Hg2)


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


def test_save_images_parallel_uses_explicit_max_workers(tmp_path, monkeypatch):
    """The max_workers kwarg overrides env var and auto-default."""
    import dfxm_geo.direct_space.forward_model as fm_mod
    import dfxm_geo.io.images as images_mod

    # Mock the compute so we don't need the real kernel. save_images_parallel
    # precomputes base_qc via precompute_forward_static(Hg) and save_image calls
    # forward_from_static(base_qc, ...) — patch both source module attributes.
    monkeypatch.setattr(fm_mod, "precompute_forward_static", lambda Hg: np.zeros((3, 1)))
    monkeypatch.setattr(
        fm_mod, "forward_from_static", lambda base_qc, phi=0, chi=0: np.zeros((4, 4))
    )

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
    import dfxm_geo.direct_space.forward_model as fm_mod
    import dfxm_geo.io.images as images_mod

    monkeypatch.setattr(fm_mod, "precompute_forward_static", lambda Hg: np.zeros((3, 1)))
    monkeypatch.setattr(
        fm_mod, "forward_from_static", lambda base_qc, phi=0, chi=0: np.zeros((4, 4))
    )

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

    Mocks forward_from_static() to return a single np.ndarray (the default) and
    confirms save_image writes the file without raising ValueError.
    """
    import dfxm_geo.direct_space.forward_model as fm_mod
    import dfxm_geo.io.images as images_mod

    expected_array = np.arange(12, dtype=float).reshape(3, 4)

    def fake_forward_from_static(base_qc, phi=0.0, chi=0.0, *args, **kwargs):
        return expected_array

    monkeypatch.setattr(fm_mod, "forward_from_static", fake_forward_from_static)
    args = (
        np.zeros((3, 1)),  # base_qc (unused by fake_forward_from_static)
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


class TestLoadOrGenerateHgSampleRemount:
    """The S kwarg threads through to Fd_find, producing different Fg."""

    def _rl(self, n: int = 8) -> np.ndarray:
        lin = np.linspace(-1.0, 1.0, n)
        grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"))
        return grid.reshape(3, -1)

    def test_S_kwarg_default_matches_omitted(self, rotation_matrices, tmp_path) -> None:
        """Calling with S=identity must equal calling without S."""
        from dfxm_geo.io.strain_cache import load_or_generate_Hg

        rl = self._rl()
        Ud, Us, Theta = rotation_matrices

        path_a = str(tmp_path / "fg_a.npy")
        path_b = str(tmp_path / "fg_b.npy")

        Hg_omitted = load_or_generate_Hg(rl, Ud, Us, Theta, dis=1.0, ndis=1, file_path=path_a)
        Hg_with_I = load_or_generate_Hg(
            rl, Ud, Us, Theta, dis=1.0, ndis=1, file_path=path_b, S=np.identity(3)
        )
        np.testing.assert_array_equal(Hg_omitted, Hg_with_I)

    def test_S2_yields_distinct_Hg(self, rotation_matrices, tmp_path) -> None:
        """S=S2 must produce a different Hg than S=identity on the same inputs."""
        from dfxm_geo.crystal.remount import S2
        from dfxm_geo.io.strain_cache import load_or_generate_Hg

        rl = self._rl()
        Ud, Us, Theta = rotation_matrices

        Hg_I = load_or_generate_Hg(
            rl, Ud, Us, Theta, dis=1.0, ndis=3, file_path=str(tmp_path / "fg_I.npy")
        )
        Hg_S2 = load_or_generate_Hg(
            rl,
            Ud,
            Us,
            Theta,
            dis=1.0,
            ndis=3,
            file_path=str(tmp_path / "fg_S2.npy"),
            S=S2,
        )
        assert not np.allclose(Hg_I, Hg_S2)
