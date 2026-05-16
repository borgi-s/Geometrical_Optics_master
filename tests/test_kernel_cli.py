"""Tests for `dfxm_geo.reciprocal_space.kernel.generate_kernel` and `cli_main`.

Uses Nrays=1000 + a 20**3 grid to keep each test under ~1 s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class TestGenerateKernelOutputPath:
    def test_writes_to_explicit_path(self, tmp_path: Path) -> None:
        from dfxm_geo.reciprocal_space.kernel import generate_kernel

        out = tmp_path / "subdir" / "Resq_i_test.npz"
        result_path = generate_kernel(
            Nrays=1000,
            npoints1=20,
            npoints2=20,
            npoints3=20,
            output_path=out,
        )
        assert Path(result_path) == out
        assert out.is_file()
        loaded = np.load(out)
        arr = loaded["Resq_i"]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (20, 20, 20)

    def test_no_sidecar_written(self, tmp_path: Path) -> None:
        """Params are bundled into the .npz; no `<stem>_vars.txt` sidecar is written."""
        from dfxm_geo.reciprocal_space.kernel import generate_kernel

        out = tmp_path / "Resq_i_explicit.npz"
        generate_kernel(Nrays=1000, npoints1=20, npoints2=20, npoints3=20, output_path=out)
        sidecar = tmp_path / "Resq_i_explicit_vars.txt"
        assert not sidecar.exists(), f"unexpected sidecar at {sidecar}"
        # Defensive: no sidecar in CWD/pkl_files either.
        assert not (Path.cwd() / "pkl_files" / "Resq_i_explicit_vars.txt").exists()


class TestDefaultConfigReciprocalBlock:
    def test_default_toml_has_reciprocal_block(self) -> None:
        """configs/default.toml must include a `[reciprocal]` block that
        dfxm-bootstrap can drive without any extra args.
        """
        import tomllib

        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.toml"
        with cfg_path.open("rb") as f:
            data = tomllib.load(f)
        assert "reciprocal" in data, "configs/default.toml missing [reciprocal] block"
        recip = data["reciprocal"]
        # CDD_inc canonical recipe (spec §1 + kernel.py defaults).
        assert recip["Nrays"] == int(1e8)
        assert recip["beamstop"] is True
        assert recip["aperture"] is True
        assert recip["knife_edge"] is False
        assert recip["bs_height"] == 25e-3
        # qi ranges (units consistent with generate_kernel kwargs).
        assert recip["qi1_range"] == 5e-4
        assert recip["qi2_range"] == 0.75e-2
        assert recip["qi3_range"] == 0.75e-2

    def test_toml_values_match_generate_kernel_defaults(self) -> None:
        """Pin the [reciprocal] TOML block to generate_kernel's defaults.

        Catches drift: if anyone bumps a default in kernel.py and forgets to
        update the TOML, this test trips. Conversely if a TOML value is
        edited without intent, this test surfaces it.
        """
        import inspect
        import tomllib

        from dfxm_geo.reciprocal_space.kernel import generate_kernel

        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.toml"
        with cfg_path.open("rb") as f:
            recip = tomllib.load(f)["reciprocal"]

        sig = inspect.signature(generate_kernel)
        for key, val in recip.items():
            assert key in sig.parameters, f"unknown kwarg {key} in [reciprocal]"
            default = sig.parameters[key].default
            assert default == val, (
                f"[reciprocal].{key} = {val!r} drifted from generate_kernel default {default!r}"
            )


class TestCliMain:
    """Unit tests for `dfxm_geo.reciprocal_space.kernel.cli_main`.

    We mock `generate_kernel` itself (the underlying Monte Carlo is exercised
    elsewhere); the goal here is to pin the CLI surface — flags, defaults,
    overwrite-guard, and TOML parsing.
    """

    def _make_config(self, tmp_path: Path) -> Path:
        cfg = tmp_path / "tiny.toml"
        cfg.write_text(
            "[reciprocal]\n"
            "Nrays = 1000\n"
            "npoints1 = 20\n"
            "npoints2 = 20\n"
            "npoints3 = 20\n"
            "qi1_range = 5e-4\n"
            "qi2_range = 7.5e-3\n"
            "qi3_range = 7.5e-3\n"
            "beamstop = true\n"
            "bs_height = 25e-3\n"
            "aperture = true\n"
            "knife_edge = false\n",
            encoding="utf-8",
        )
        return cfg

    def test_requires_config(self, capsys: pytest.CaptureFixture[str]) -> None:
        from dfxm_geo.reciprocal_space.kernel import cli_main

        with pytest.raises(SystemExit) as excinfo:
            cli_main([])
        assert excinfo.value.code == 2  # argparse usage error

    def test_invokes_generate_kernel_with_toml_params(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dfxm_geo.reciprocal_space import kernel as kmod

        cfg = self._make_config(tmp_path)
        captured: dict[str, object] = {}

        def fake_generate(**kwargs: object) -> Path:
            captured.update(kwargs)
            out = kwargs.get("output_path")
            assert isinstance(out, Path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"")
            return out

        monkeypatch.setattr(kmod, "generate_kernel", fake_generate)

        out = tmp_path / "canonical.npz"
        rc = kmod.cli_main(["--config", str(cfg), "--output", str(out)])
        assert rc == 0
        assert out.is_file()
        # TOML fields are forwarded as kwargs.
        assert captured["Nrays"] == 1000
        assert captured["npoints1"] == 20
        assert captured["beamstop"] is True
        assert captured["bs_height"] == 25e-3
        assert captured["output_path"] == out

    def test_default_output_matches_forward_model_canonical_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no --output, write to `<fm.pkl_fpath>/<fm.pkl_fn>`."""
        import os

        import dfxm_geo.direct_space.forward_model as fm
        from dfxm_geo.reciprocal_space import kernel as kmod

        # Redirect the canonical path into tmp_path so we don't trip the
        # overwrite-guard on a checkout that already has the real kernel npz.
        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path) + os.sep)
        monkeypatch.setattr(fm, "pkl_fn", "Resq_i_canonical.npz")

        cfg = self._make_config(tmp_path)
        seen: dict[str, object] = {}

        def fake_generate(**kwargs: object) -> Path:
            seen.update(kwargs)
            out = kwargs["output_path"]
            assert isinstance(out, Path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"")
            return out

        monkeypatch.setattr(kmod, "generate_kernel", fake_generate)

        rc = kmod.cli_main(["--config", str(cfg)])
        assert rc == 0
        expected = Path(fm.pkl_fpath) / fm.pkl_fn
        assert seen["output_path"] == expected

    def test_refuses_to_overwrite_without_force(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from dfxm_geo.reciprocal_space import kernel as kmod

        cfg = self._make_config(tmp_path)
        existing = tmp_path / "existing.pkl"
        existing.write_bytes(b"prior contents")
        called = False

        def fake_generate(**kwargs: object) -> Path:
            nonlocal called
            called = True
            return Path()

        monkeypatch.setattr(kmod, "generate_kernel", fake_generate)

        rc = kmod.cli_main(["--config", str(cfg), "--output", str(existing)])
        assert rc != 0
        assert not called, "generate_kernel must not run when output exists and --force absent"
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "--force" in out

    def test_force_flag_overwrites(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from dfxm_geo.reciprocal_space import kernel as kmod

        cfg = self._make_config(tmp_path)
        existing = tmp_path / "existing.pkl"
        existing.write_bytes(b"prior contents")

        def fake_generate(**kwargs: object) -> Path:
            out = kwargs["output_path"]
            assert isinstance(out, Path)
            out.write_bytes(b"new contents")
            return out

        monkeypatch.setattr(kmod, "generate_kernel", fake_generate)

        rc = kmod.cli_main(["--config", str(cfg), "--output", str(existing), "--force"])
        assert rc == 0
        assert existing.read_bytes() == b"new contents"

    def test_missing_reciprocal_block_errors_clearly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from dfxm_geo.reciprocal_space.kernel import cli_main

        bad = tmp_path / "no_recip.toml"
        bad.write_text("[scan]\nphi_range = 0.1\n", encoding="utf-8")
        rc = cli_main(["--config", str(bad)])
        assert rc == 1
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "[reciprocal]" in out

    def test_missing_config_file_errors_clearly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Non-existent --config path → clean CLI error, not a traceback."""
        from dfxm_geo.reciprocal_space.kernel import cli_main

        missing = tmp_path / "does_not_exist.toml"
        rc = cli_main(["--config", str(missing)])
        assert rc == 1
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "not found" in out
        assert str(missing) in out

    def test_unknown_reciprocal_key_errors_clearly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Unknown TOML key in [reciprocal] → clean CLI error, not a TypeError."""
        from dfxm_geo.reciprocal_space.kernel import cli_main

        cfg = tmp_path / "bogus.toml"
        cfg.write_text("[reciprocal]\nNrays = 1000\ntotally_invented_key = 99\n", encoding="utf-8")
        rc = cli_main(["--config", str(cfg)])
        assert rc == 1
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "totally_invented_key" in out

    def test_if_missing_exits_zero_when_kernel_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--if-missing skips silently (exit 0) when the destination already exists.

        Lets cluster batch templates call `dfxm-bootstrap --if-missing` as an
        idempotent guard without parsing pkl_fn or hardcoding the filename.
        """
        from dfxm_geo.reciprocal_space import kernel as kmod

        cfg = self._make_config(tmp_path)
        existing = tmp_path / "existing.pkl"
        existing.write_bytes(b"prior contents")
        called = False

        def fake_generate(**kwargs: object) -> Path:
            nonlocal called
            called = True
            return Path()

        monkeypatch.setattr(kmod, "generate_kernel", fake_generate)

        rc = kmod.cli_main(["--config", str(cfg), "--output", str(existing), "--if-missing"])
        assert rc == 0, "must exit 0 when kernel npz exists under --if-missing"
        assert not called, "generate_kernel must not run under --if-missing when output exists"
        assert existing.read_bytes() == b"prior contents", "existing kernel npz must not be touched"

    def test_if_missing_generates_when_kernel_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--if-missing falls through to normal generation when the destination is absent."""
        from dfxm_geo.reciprocal_space import kernel as kmod

        cfg = self._make_config(tmp_path)
        target = tmp_path / "fresh.npz"
        assert not target.exists()

        def fake_generate(**kwargs: object) -> Path:
            out = kwargs["output_path"]
            assert isinstance(out, Path)
            out.write_bytes(b"freshly generated")
            return out

        monkeypatch.setattr(kmod, "generate_kernel", fake_generate)

        rc = kmod.cli_main(["--config", str(cfg), "--output", str(target), "--if-missing"])
        assert rc == 0
        assert target.read_bytes() == b"freshly generated"

    def test_force_and_if_missing_are_mutually_exclusive(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--force and --if-missing can't be combined — the semantics conflict."""
        from dfxm_geo.reciprocal_space.kernel import cli_main

        cfg = self._make_config(tmp_path)
        rc = cli_main(
            ["--config", str(cfg), "--output", str(tmp_path / "x.npz"), "--force", "--if-missing"]
        )
        assert rc == 1
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "mutually exclusive" in out
