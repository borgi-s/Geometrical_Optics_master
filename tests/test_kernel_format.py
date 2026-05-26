"""Tests for the .npz kernel format that replaces the legacy pickle format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

GOLDEN_DIR = Path(__file__).resolve().parent / "data" / "golden"


class TestNpzRoundTrip:
    """Layer 1: round-trip and param-bundling correctness."""

    def test_savez_round_trip_preserves_resq_i(self, tmp_path: Path) -> None:
        """np.savez -> np.load preserves the resolution array bit-for-bit."""
        rng = np.random.default_rng(42)
        arr = rng.uniform(0, 1, size=(8, 6, 4)).astype(np.float64)

        dst = tmp_path / "round_trip.npz"
        np.savez(dst, Resq_i=arr, qi1_range=np.float64(5e-4))

        loaded = np.load(dst)
        assert np.array_equal(loaded["Resq_i"], arr)

    def test_bundled_scalars_extract_to_correct_types(self, tmp_path: Path) -> None:
        """Scalar params bundled into npz extract back to the right Python types."""
        dst = tmp_path / "bundled.npz"
        np.savez(
            dst,
            Resq_i=np.zeros((2, 2, 2)),
            qi1_range=np.float64(5e-4),
            qi2_range=np.float64(7.5e-3),
            qi3_range=np.float64(7.5e-3),
            npoints1=np.int64(400),
            npoints2=np.int64(200),
            npoints3=np.int64(200),
            Nrays=np.int64(100_000_000),
        )

        loaded = np.load(dst)
        # 0-d numpy arrays — extract with .item() or float() / int()
        assert float(loaded["qi1_range"]) == pytest.approx(5e-4)
        assert int(loaded["npoints1"]) == 400
        assert int(loaded["Nrays"]) == 100_000_000


class TestGenerateKernelWritesNpz:
    """The generate_kernel function writes a .npz with bundled params."""

    def test_generate_kernel_writes_npz_with_bundled_meta(self, tmp_path: Path) -> None:
        """generate_kernel(output_path=tmp_path/foo.npz) produces a valid npz."""
        from dfxm_geo.reciprocal_space.kernel import generate_kernel

        dst = tmp_path / "Resq_i_test.npz"
        generate_kernel(
            output_path=dst,
            Nrays=1000,  # tiny; just exercise the write path
            npoints1=4,
            npoints2=4,
            npoints3=4,
        )

        assert dst.is_file()
        loaded = np.load(dst)
        assert "Resq_i" in loaded.files
        assert loaded["Resq_i"].shape == (4, 4, 4)
        assert int(loaded["Nrays"]) == 1000
        assert int(loaded["npoints1"]) == 4
        assert float(loaded["qi1_range"]) == pytest.approx(5e-4)
        # Round-trip the write-only audit fields too — these aren't read by
        # _load_default_kernel but must travel intact for reproducibility.
        assert float(loaded["theta"]) > 0  # Bragg angle, set by _default_theta_al_111
        assert float(loaded["D"]) == pytest.approx(2 * np.sqrt(50e-6 * 1.6e-3))
        assert bool(loaded["beamstop"]) is True
        assert bool(loaded["aperture"]) is True

        # Sidecar must NOT exist
        sidecar = dst.with_name(dst.stem + "_vars.txt")
        assert not sidecar.exists(), f"unexpected sidecar at {sidecar}"


class TestLoadDefaultKernel:
    """Layer 1 + 2: the loader resolves npz paths and populates module state."""

    def test_load_default_kernel_loads_npz(self, tmp_path: Path) -> None:
        """_load_default_kernel reads an .npz and sets module globals correctly."""
        import dfxm_geo.direct_space.forward_model as fm

        # Build a tiny synthetic npz
        dst = tmp_path / "Resq_i_test.npz"
        np.savez(
            dst,
            Resq_i=np.ones((4, 4, 4), dtype=np.float64),
            qi1_range=np.float64(1e-3),
            qi2_range=np.float64(2e-3),
            qi3_range=np.float64(3e-3),
            npoints1=np.int64(4),
            npoints2=np.int64(4),
            npoints3=np.int64(4),
        )

        # Snapshot all globals that _load_default_kernel mutates, so the test
        # doesn't leak state into subsequent tests in the same process.
        saved_state = {
            name: getattr(fm, name)
            for name in (
                "Resq_i",
                "qi1_range",
                "qi2_range",
                "qi3_range",
                "npoints1",
                "npoints2",
                "npoints3",
                "qi1_start",
                "qi2_start",
                "qi3_start",
                "qi1_step",
                "qi2_step",
                "qi3_step",
                "qi_starts",
                "qi_steps",
            )
        }
        try:
            fm._load_default_kernel(pkl_path=str(dst), compute_Hg=False)
            assert fm.Resq_i is not None
            assert fm.Resq_i.shape == (4, 4, 4)
            assert np.array_equal(fm.Resq_i, np.ones((4, 4, 4)))
            assert fm.qi1_range == pytest.approx(1e-3)
            assert fm.npoints1 == 4
        finally:
            for name, value in saved_state.items():
                setattr(fm, name, value)


class TestLegacyPickleRejection:
    """The loader refuses .pkl paths with a clear migration message."""

    def test_load_raises_on_pkl_path(self, tmp_path: Path) -> None:
        """A .pkl path raises RuntimeError mentioning dfxm-bootstrap."""
        import dfxm_geo.direct_space.forward_model as fm

        # Create a non-empty file with the wrong extension
        fake_pkl = tmp_path / "Resq_i_legacy.pkl"
        fake_pkl.write_bytes(b"\x80\x04\x95")  # pickle magic; never read

        with pytest.raises(RuntimeError, match="pickle support was removed"):
            fm._load_default_kernel(pkl_path=str(fake_pkl), compute_Hg=False)


@pytest.mark.xfail(
    reason=(
        "Requires the 2023 legacy pickle Resq_i_20230913_1308.pkl to build the "
        "comparison npz; that artifact is not in the repo and cannot be "
        "reconstructed, so the test SKIPS when it is absent. Independently, the "
        "snapshot was captured at Nsub=2 while the default is now Nsub=1. Not an "
        "RNG-seeding issue -- the wall path Find_Hg -> Fd_find is deterministic. "
        "(Kernel MC generation is now seedable via dfxm-bootstrap --seed, but "
        "this snapshot predates that.) Un-xfailing needs the legacy pickle plus a "
        "golden regenerated at the current Nsub default."
    ),
    strict=False,
)
class TestForwardOutputBitEquivalence:
    """Layer 2: forward() output using npz-loaded kernel must match the
    pickle-era snapshot bit-for-bit.

    Skipped if either the existing pickle (to regenerate the npz from) or the
    golden snapshot is missing.
    """

    def test_forward_output_matches_pickle_era_snapshot(self, tmp_path: Path) -> None:
        import dfxm_geo.direct_space.forward_model as fm

        snapshot_path = GOLDEN_DIR / "forward_snapshot_pickle_era.npy"
        if not snapshot_path.exists():
            pytest.skip(f"snapshot not present at {snapshot_path}")

        # Build an npz from the existing pickle on disk by reading it once via
        # a deliberately-allowed pickle import in the *test*, then writing npz.
        # This isolates pickle.load to the test harness, not production code.
        # Task 9's import-audit is scoped to src/ files (via mod.__file__), so
        # this test-side pickle import does NOT trip the defensive guard.
        import pickle as _pkl

        legacy_pkl = Path(fm.pkl_fpath) / "Resq_i_20230913_1308.pkl"
        if not legacy_pkl.exists():
            pytest.skip(f"legacy pickle not present at {legacy_pkl}; cannot build comparison npz")

        with open(legacy_pkl, "rb") as f:
            Resq_i_from_pickle = _pkl.load(f)

        # Read the existing _vars.txt one last time (also via eval — test-only)
        vars_txt = legacy_pkl.with_name(legacy_pkl.stem + "_vars.txt")
        var_d = eval(vars_txt.read_text())  # noqa: S307

        dst = tmp_path / "Resq_i_from_pickle.npz"
        np.savez(
            dst,
            Resq_i=Resq_i_from_pickle,
            qi1_range=np.float64(var_d["qi1_range"]),
            qi2_range=np.float64(var_d["qi2_range"]),
            qi3_range=np.float64(var_d["qi3_range"]),
            npoints1=np.int64(var_d["npoints1"]),
            npoints2=np.int64(var_d["npoints2"]),
            npoints3=np.int64(var_d["npoints3"]),
        )

        # Snapshot all globals mutated by _load_default_kernel so this test
        # doesn't leak state into subsequent tests in the same process.
        # compute_Hg=True is used (Option A) so Hg is computed from the tmp npz
        # — this avoids needing fm.Hg to be pre-set, which it won't be on a
        # clean checkout where the default .npz doesn't exist yet.
        saved_state = {
            name: getattr(fm, name)
            for name in (
                "Resq_i",
                "qi1_range",
                "qi2_range",
                "qi3_range",
                "npoints1",
                "npoints2",
                "npoints3",
                "qi1_start",
                "qi2_start",
                "qi3_start",
                "qi1_step",
                "qi2_step",
                "qi3_step",
                "qi_starts",
                "qi_steps",
                "Hg",
                "q_hkl",
            )
        }
        try:
            # compute_Hg=True: _load_default_kernel calls Find_Hg which loads
            # the cached Fg from disk (direct_space/deformation_gradient_tensors/).
            # This sets fm.Hg and fm.q_hkl, which forward() requires.
            fm._load_default_kernel(pkl_path=str(dst), compute_Hg=True)
            out = fm.forward(fm.Hg, phi=0.0, chi=0.0)
            if isinstance(out, tuple):
                out = out[0]
            golden = np.load(snapshot_path)
            assert np.array_equal(out, golden), "forward() output differs from pickle-era snapshot"
        finally:
            for name, value in saved_state.items():
                setattr(fm, name, value)


class TestCrossPlatformSyntheticKernel:
    """Layer 3: the committed synthetic golden loads regardless of platform."""

    def test_load_synthetic_kernel_committed_golden(self) -> None:
        """The 4x4x4 synthetic golden npz loads with all expected keys."""
        golden = GOLDEN_DIR / "synthetic_kernel.npz"
        assert golden.exists(), f"missing committed golden at {golden}"

        loaded = np.load(golden)
        expected_keys = {
            "Resq_i",
            "qi1_range",
            "qi2_range",
            "qi3_range",
            "npoints1",
            "npoints2",
            "npoints3",
            "Nrays",
        }
        assert set(loaded.files) == expected_keys

        assert loaded["Resq_i"].shape == (4, 4, 4)
        assert loaded["Resq_i"].dtype == np.float64
        assert float(loaded["qi1_range"]) == pytest.approx(5e-4)
        assert int(loaded["npoints1"]) == 4


class TestNoPickleImportsInRuntime:
    """Defensive guard: runtime modules must not import pickle.

    Audit scope: src/ files only — accessed via `mod.__file__`, which resolves
    to the installed/editable src path, never tests/. Test files MAY use
    pickle (e.g. TestForwardOutputBitEquivalence) without tripping this guard.
    """

    def test_forward_model_does_not_import_pickle(self) -> None:
        """forward_model.py (src) must not have `import pickle` or `from pickle`."""
        import dfxm_geo.direct_space.forward_model as fm

        src = Path(fm.__file__).read_text(encoding="utf-8")
        assert "import pickle" not in src, "pickle import re-introduced in forward_model.py"
        assert "from pickle" not in src

    def test_kernel_module_does_not_import_pickle(self) -> None:
        """kernel.py and resolution.py (src) must not import pickle."""
        from dfxm_geo.reciprocal_space import kernel, resolution

        for mod in (kernel, resolution):
            src = Path(mod.__file__).read_text(encoding="utf-8")
            assert "import pickle" not in src, f"pickle import re-introduced in {mod.__file__}"
            assert "from pickle" not in src, f"pickle import re-introduced in {mod.__file__}"
