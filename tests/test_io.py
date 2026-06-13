"""Unit tests for dfxm_geo.io.images (_auto_max_workers) and io.strain_cache.

Note: load_image, load_images, load_images_parallel, and save_edfs were removed
in v1.1.0.  save_image and save_images_parallel were removed in S4 of the
ForwardContext refactor (#16) — identification migrated to HDF5 output in
v1.2.0.  The HDF5 equivalents live in io/hdf5.py; the migration helper is in
io/migrate.py.
"""

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


def _fake_vm(available_gb):
    """A psutil.virtual_memory() stand-in reporting `available_gb` free."""
    from types import SimpleNamespace

    return lambda: SimpleNamespace(available=int(available_gb * (1024**3)))


def test_auto_max_workers_uses_all_cpus_when_memory_ample(monkeypatch):
    """With abundant RAM the cap is CPU-bound, not memory-throttled."""
    import psutil

    from dfxm_geo.io import images

    monkeypatch.delenv("DFXM_MAX_WORKERS", raising=False)
    monkeypatch.setattr(images.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(psutil, "virtual_memory", _fake_vm(64.0))
    assert images._auto_max_workers() == 8


def test_auto_max_workers_ray_proportional_beats_fixed_slab(monkeypatch):
    """At Nsub=1 a modest RAM budget still permits several workers.

    The old fixed 1 GiB/worker reservation collapsed to a single worker at
    ~2 GiB usable; the ray-proportional estimate (~0.13 GiB/worker at
    px510/Nsub=1) permits more, which is the whole point of the change.
    """
    import psutil

    from dfxm_geo.io import images

    monkeypatch.delenv("DFXM_MAX_WORKERS", raising=False)
    monkeypatch.setattr(images.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(psutil, "virtual_memory", _fake_vm(2.0))
    workers = images._auto_max_workers()
    assert 2 <= workers <= 8  # old slab logic returned 1 here


def test_auto_max_workers_shared_between_io_modules():
    """hdf5 re-exports the single images definition (no divergent copy)."""
    from dfxm_geo.io import hdf5, images

    assert hdf5._auto_max_workers is images._auto_max_workers


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


class TestLoadOrGenerateHgBParam:
    """The b kwarg must be forwarded to Fd_find (regression for the missing b=b bug)."""

    def test_b_forwarded_to_Fd_find(self, rotation_matrices, monkeypatch) -> None:
        """Monkeypatch Fd_find to capture the b it receives and verify forwarding."""
        import dfxm_geo.io.strain_cache as sc
        from dfxm_geo.constants import BURGERS_VECTOR

        captured: dict[str, float] = {}

        original_Fd_find = sc.Fd_find

        def fake_Fd_find(rl, Ud, Us, Theta, dis, ndis, b=BURGERS_VECTOR, **kwargs):
            captured["b"] = b
            return original_Fd_find(rl, Ud, Us, Theta, dis, ndis, b=b, **kwargs)

        monkeypatch.setattr(sc, "Fd_find", fake_Fd_find)

        Ud, Us, Theta = rotation_matrices
        rng = np.random.default_rng(42)
        rl = rng.normal(size=(3, 20)) * 1e-6
        custom_b = 9.999e-4  # deliberately non-default

        sc.load_or_generate_Hg(rl, Ud, Us, Theta, dis=1.0, ndis=1, b=custom_b)

        assert "b" in captured, "Fd_find was never called"
        assert captured["b"] == custom_b, (
            f"load_or_generate_Hg did not forward b to Fd_find: "
            f"expected {custom_b}, got {captured['b']}"
        )

    def test_distinct_b_values_yield_distinct_Hg(self, rotation_matrices, tmp_path) -> None:
        """Two distinct b magnitudes must produce different Hg arrays."""
        from dfxm_geo.io.strain_cache import load_or_generate_Hg

        Ud, Us, Theta = rotation_matrices
        rng = np.random.default_rng(7)
        rl = rng.normal(size=(3, 16)) * 1e-6

        b_fcc = 2.862e-4  # typical FCC Al |b| in µm
        b_bcc = 2.476e-4  # typical BCC Fe |b| in µm

        Hg_fcc = load_or_generate_Hg(
            rl, Ud, Us, Theta, dis=1.0, ndis=1, file_path=str(tmp_path / "fg_fcc.npy"), b=b_fcc
        )
        Hg_bcc = load_or_generate_Hg(
            rl, Ud, Us, Theta, dis=1.0, ndis=1, file_path=str(tmp_path / "fg_bcc.npy"), b=b_bcc
        )

        assert not np.allclose(Hg_fcc, Hg_bcc), (
            "Different b values must produce different Hg; "
            "if they are equal, b is not being forwarded to Fd_find"
        )


class TestLoadOrGenerateHgNyParam:
    """The ny (Poisson) kwarg must reach Fd_find (M4 4.3a I2 regression)."""

    def test_ny_forwarded_to_Fd_find(self, rotation_matrices, monkeypatch) -> None:
        """Monkeypatch Fd_find to capture the ny it receives and verify forwarding."""
        import dfxm_geo.io.strain_cache as sc
        from dfxm_geo.constants import POISSON_RATIO

        captured: dict[str, float] = {}
        original_Fd_find = sc.Fd_find

        def fake_Fd_find(rl, Ud, Us, Theta, dis, ndis, *, ny=POISSON_RATIO, **kwargs):
            captured["ny"] = ny
            return original_Fd_find(rl, Ud, Us, Theta, dis, ndis, ny=ny, **kwargs)

        monkeypatch.setattr(sc, "Fd_find", fake_Fd_find)

        Ud, Us, Theta = rotation_matrices
        rng = np.random.default_rng(42)
        rl = rng.normal(size=(3, 20)) * 1e-6
        custom_ny = 0.29  # BCC Fe, deliberately non-default

        sc.load_or_generate_Hg(rl, Ud, Us, Theta, dis=1.0, ndis=1, ny=custom_ny)

        assert "ny" in captured, "Fd_find was never called"
        assert captured["ny"] == custom_ny, (
            f"load_or_generate_Hg did not forward ny to Fd_find: "
            f"expected {custom_ny}, got {captured['ny']}"
        )

    def test_default_ny_is_poisson_ratio(self, rotation_matrices, monkeypatch) -> None:
        """Omitting ny forwards POISSON_RATIO (0.334, Al) — the byte-identical default."""
        import dfxm_geo.io.strain_cache as sc
        from dfxm_geo.constants import POISSON_RATIO

        captured: dict[str, float] = {}
        original_Fd_find = sc.Fd_find

        def fake_Fd_find(rl, Ud, Us, Theta, dis, ndis, *, ny=POISSON_RATIO, **kwargs):
            captured["ny"] = ny
            return original_Fd_find(rl, Ud, Us, Theta, dis, ndis, ny=ny, **kwargs)

        monkeypatch.setattr(sc, "Fd_find", fake_Fd_find)
        Ud, Us, Theta = rotation_matrices
        rng = np.random.default_rng(1)
        rl = rng.normal(size=(3, 12)) * 1e-6

        sc.load_or_generate_Hg(rl, Ud, Us, Theta, dis=1.0, ndis=1)

        assert captured["ny"] == POISSON_RATIO

    def test_distinct_ny_values_yield_distinct_Hg(self, rotation_matrices, tmp_path) -> None:
        """Two distinct Poisson ratios must produce different Hg arrays."""
        from dfxm_geo.io.strain_cache import load_or_generate_Hg

        Ud, Us, Theta = rotation_matrices
        rng = np.random.default_rng(7)
        rl = rng.normal(size=(3, 16)) * 1e-6

        Hg_al = load_or_generate_Hg(
            rl, Ud, Us, Theta, dis=1.0, ndis=1, file_path=str(tmp_path / "fg_al.npy"), ny=0.334
        )
        Hg_fe = load_or_generate_Hg(
            rl, Ud, Us, Theta, dis=1.0, ndis=1, file_path=str(tmp_path / "fg_fe.npy"), ny=0.29
        )
        assert not np.allclose(Hg_al, Hg_fe), (
            "Different ny values must produce different Hg; "
            "if they are equal, ny is not being forwarded to Fd_find"
        )
