# Bootstrap multi-reflection + Bragg validity implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `dfxm-bootstrap` accept arbitrary `(hkl, keV)` from the `[reciprocal]` TOML block, validate the Bragg geometry, and write a per-reflection kernel npz whose filename encodes the reflection — preparing for sub-project D's multi-reflection forward/identify lookup.

**Architecture:** Single change point: `src/dfxm_geo/reciprocal_space/kernel.py`. Add two helpers (`_validate_reflection`, `_build_kernel_filename`); extend `cli_main` to pop new `hkl`/`keV` TOML keys, compute θ, build the per-reflection filename, and pass `theta=θ` + `output_path=<built>` to the (unchanged) `generate_kernel`. Also: update `pkl_fn` in `forward_model.py:61` to the new filename pattern; add `hkl` + `keV` to `configs/default.toml`.

**Tech Stack:** Python 3.11+, `numpy`, `tomllib` (stdlib), `pytest` with `capsys`/`monkeypatch`, `mypy --strict`.

**Spec:** `docs/superpowers/specs/2026-05-20-bootstrap-multi-reflection-design.md` (commit `ed4f51d`).

---

## File Structure

**Modify:**
- `src/dfxm_geo/reciprocal_space/kernel.py`
  - Add `_validate_reflection(hkl, keV, a) -> float` (~30 LOC including warnings)
  - Add `_build_kernel_filename(hkl, keV, date) -> str` (~3 LOC)
  - Extend `cli_main`: pop `hkl`/`keV` from TOML, allow them in unknown-key check, build output path, compute θ, echo θ to stdout (~30 LOC)
- `src/dfxm_geo/direct_space/forward_model.py`
  - Line 61: update `pkl_fn` default to new pattern (~1 LOC)
  - Update comment on lines 60–61
- `configs/default.toml`
  - Add `hkl = [-1, 1, -1]` and `keV = 17.0` under `[reciprocal]` (~2 LOC)
- `tests/test_kernel_cli.py`
  - Add `TestValidateReflection` (~50 LOC)
  - Add `TestBuildKernelFilename` (~20 LOC)
  - Add `TestCliMainMultiReflection` (~70 LOC)
  - Add `TestPklFnRegression` (~10 LOC)
  - Extend existing `TestDefaultConfigReciprocalBlock` (~10 LOC)

**Total**: ~225 LOC (well within the ~200 LOC spec estimate; tests dominate).

**Working directory:** `C:/Users/borgi/Documents/GM-reworked/Geometrical_Optics_master/`
**Branch:** `chore/spec-bootstrap-multi-reflection` (current). Code commits land on this branch; spec commit `ed4f51d` already on it.
**Python interpreter:** `C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe` (bash `python` is 2.7 — do NOT use it).

---

## Task 1: `_validate_reflection` helper

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py` (add after existing `_default_theta_al_111` at line 29)
- Test: `tests/test_kernel_cli.py` (append new class)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_kernel_cli.py`:

```python
class TestValidateReflection:
    def test_len_not_three_raises(self) -> None:
        from dfxm_geo.reciprocal_space.kernel import _validate_reflection

        with pytest.raises(ValueError, match=r"hkl must have 3 components, got 2\."):
            _validate_reflection((1, 1), 17.0, 4.0495e-10)

    def test_non_int_component_raises(self) -> None:
        from dfxm_geo.reciprocal_space.kernel import _validate_reflection

        with pytest.raises(ValueError, match=r"hkl components must be int"):
            _validate_reflection((1.5, 1, -1), 17.0, 4.0495e-10)

    def test_zero_hkl_raises(self) -> None:
        from dfxm_geo.reciprocal_space.kernel import _validate_reflection

        with pytest.raises(ValueError, match=r"hkl=\(0,0,0\) is not a valid reflection"):
            _validate_reflection((0, 0, 0), 17.0, 4.0495e-10)

    def test_nonpositive_keV_raises(self) -> None:
        from dfxm_geo.reciprocal_space.kernel import _validate_reflection

        with pytest.raises(ValueError, match=r"keV must be > 0, got 0"):
            _validate_reflection((-1, 1, -1), 0.0, 4.0495e-10)
        with pytest.raises(ValueError, match=r"keV must be > 0, got -5"):
            _validate_reflection((-1, 1, -1), -5.0, 4.0495e-10)

    def test_unsatisfiable_bragg_raises(self) -> None:
        from dfxm_geo.reciprocal_space.kernel import _validate_reflection

        # hkl=(10,10,10) at 1 keV: λ ≈ 12.4 Å, d ≈ 0.234 Å → sin θ ≈ 26 > 1.
        with pytest.raises(ValueError, match=r"Bragg condition unsatisfiable"):
            _validate_reflection((10, 10, 10), 1.0, 4.0495e-10)

    def test_default_al_111_matches_existing_helper(self) -> None:
        from dfxm_geo.reciprocal_space.kernel import (
            _default_theta_al_111,
            _validate_reflection,
        )

        theta_new = _validate_reflection((-1, 1, -1), 17.0, 4.0495e-10)
        theta_old = _default_theta_al_111(17)
        assert theta_new == theta_old  # bit-equal

    def test_known_reflection_200_at_17keV(self) -> None:
        from dfxm_geo.reciprocal_space.kernel import _validate_reflection

        # Al (2,0,0): d_200 = a/2 = 2.02475 Å; λ at 17 keV ≈ 0.7293 Å.
        # sin θ = λ / (2d) = 0.18012; θ = 10.376° = 0.18112 rad.
        theta = _validate_reflection((2, 0, 0), 17.0, 4.0495e-10)
        assert theta == pytest.approx(np.deg2rad(10.376), abs=1e-3)

    def test_low_theta_warns(self, capsys: pytest.CaptureFixture[str]) -> None:
        from dfxm_geo.reciprocal_space.kernel import _validate_reflection

        # Al (1,1,1) at 80 keV: very short λ → very small θ.
        _validate_reflection((1, 1, 1), 80.0, 4.0495e-10)
        captured = capsys.readouterr()
        assert "is very low" in captured.err
        assert "θ" in captured.err or "theta" in captured.err.lower()

    def test_high_theta_warns(self, capsys: pytest.CaptureFixture[str]) -> None:
        from dfxm_geo.reciprocal_space.kernel import _validate_reflection

        # Al (3,2,1) at 5.74 keV: λ ≈ 2.16 Å, d ≈ 1.082 Å,
        # sin θ ≈ 0.998 → θ ≈ 86.3° (solidly > 85°, well under 90°).
        _validate_reflection((3, 2, 1), 5.74, 4.0495e-10)
        captured = capsys.readouterr()
        assert "near back-reflection" in captured.err
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_cli.py::TestValidateReflection -v
```
Expected: all 9 tests FAIL with `ImportError: cannot import name '_validate_reflection'`.

- [ ] **Step 3: Implement `_validate_reflection` in `kernel.py`**

Insert immediately after `_default_theta_al_111` (currently ends at line 29):

```python
def _validate_reflection(
    hkl: tuple[int, int, int],
    keV: float,
    a: float,
) -> float:
    """Compute and validate the Bragg angle θ for an arbitrary cubic reflection.

    Args:
        hkl: Miller indices (must be ints, length 3, not all zero).
        keV: beam energy in keV (must be > 0).
        a: cubic lattice parameter in metres.

    Returns:
        Bragg angle θ in radians.

    Raises:
        ValueError: on structural input errors or unsatisfiable Bragg geometry.

    Emits warnings to stderr when θ ∉ [5°, 85°] (unusual but valid reflections).
    """
    import sys

    if len(hkl) != 3:
        raise ValueError(f"hkl must have 3 components, got {len(hkl)}.")
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in hkl):
        raise ValueError(f"hkl components must be int, got {hkl}.")
    if hkl == (0, 0, 0):
        raise ValueError("hkl=(0,0,0) is not a valid reflection (no diffraction).")
    if keV <= 0:
        raise ValueError(f"keV must be > 0, got {keV}.")

    h, k, l = hkl
    d_hkl = a / np.sqrt(h * h + k * k + l * l)
    wavelength = 1.239841984e-9 / keV  # hc/E, metres
    sin_theta = wavelength / (2 * d_hkl)
    if sin_theta > 1:
        lam_A = wavelength * 1e10
        two_d_A = 2 * d_hkl * 1e10
        raise ValueError(
            f"Bragg condition unsatisfiable: λ={lam_A:.4f} Å, "
            f"2·d_hkl={two_d_A:.4f} Å, sin θ = {sin_theta:.4f} > 1 for "
            f"hkl={hkl} at {keV} keV. Pick a lower-order reflection or "
            f"higher beam energy."
        )

    theta = float(np.arcsin(sin_theta))
    theta_deg = float(np.degrees(theta))
    if theta_deg < 5.0:
        print(
            f"warning: θ = {theta_deg:.2f}° is very low (< 5°); "
            f"reflection unusual but valid.",
            file=sys.stderr,
        )
    elif theta_deg > 85.0:
        print(
            f"warning: θ = {theta_deg:.2f}° near back-reflection (> 85°); "
            f"reflection unusual but valid.",
            file=sys.stderr,
        )
    return theta
```

Note: `isinstance(x, int) and not isinstance(x, bool)` rejects Python `True`/`False` (which are technically `int` subclasses). `numpy` integer scalars from `tomllib` are not a concern — TOML int parses to Python `int`.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_cli.py::TestValidateReflection -v
```
Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/kernel.py tests/test_kernel_cli.py
git commit -m "feat(kernel): add _validate_reflection helper with Bragg validity checks"
```

---

## Task 2: `_build_kernel_filename` helper

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py` (add after `_validate_reflection`)
- Test: `tests/test_kernel_cli.py` (append new class)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_kernel_cli.py`:

```python
class TestBuildKernelFilename:
    def test_basic_default_reflection(self) -> None:
        from dfxm_geo.reciprocal_space.kernel import _build_kernel_filename

        result = _build_kernel_filename((-1, 1, -1), 17.0, "20260520_2100")
        assert result == "Resq_i_h-1_k1_l-1_17keV_20260520_2100.npz"

    def test_g_format_drops_trailing_zero(self) -> None:
        from dfxm_geo.reciprocal_space.kernel import _build_kernel_filename

        # 17.0 → "17"
        assert "_17keV_" in _build_kernel_filename((-1, 1, -1), 17.0, "20260520_2100")

    def test_g_format_keeps_decimals_when_needed(self) -> None:
        from dfxm_geo.reciprocal_space.kernel import _build_kernel_filename

        # 17.5 → "17.5"
        assert "_17.5keV_" in _build_kernel_filename((-1, 1, -1), 17.5, "20260520_2100")

    def test_int_keV_renders_as_int(self) -> None:
        from dfxm_geo.reciprocal_space.kernel import _build_kernel_filename

        # int 17 → "17" (not "17.0")
        assert "_17keV_" in _build_kernel_filename((-1, 1, -1), 17, "20260520_2100")

    def test_mixed_signs(self) -> None:
        from dfxm_geo.reciprocal_space.kernel import _build_kernel_filename

        result = _build_kernel_filename((2, -1, 3), 8.0, "20260520_2100")
        assert result == "Resq_i_h2_k-1_l3_8keV_20260520_2100.npz"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_cli.py::TestBuildKernelFilename -v
```
Expected: all 5 tests FAIL with `ImportError: cannot import name '_build_kernel_filename'`.

- [ ] **Step 3: Implement `_build_kernel_filename` in `kernel.py`**

Insert immediately after `_validate_reflection`:

```python
def _build_kernel_filename(
    hkl: tuple[int, int, int],
    keV: float,
    date: str,
) -> str:
    """Per-reflection kernel npz basename: `Resq_i_h{h}_k{k}_l{l}_{keV}keV_{date}.npz`.

    `:g` formatting drops trailing zeros on keV (17.0 → "17", 17.5 → "17.5").
    Negative hkl components render naturally (-1 → "h-1").
    """
    h, k, l = hkl
    return f"Resq_i_h{h}_k{k}_l{l}_{keV:g}keV_{date}.npz"
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_cli.py::TestBuildKernelFilename -v
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/kernel.py tests/test_kernel_cli.py
git commit -m "feat(kernel): add _build_kernel_filename helper for per-reflection paths"
```

---

## Task 3: `cli_main` happy path — hkl + keV → θ → filename → generate_kernel

**Files:**
- Modify: `src/dfxm_geo/reciprocal_space/kernel.py` (extend `cli_main`, currently lines 146–255)
- Test: `tests/test_kernel_cli.py` (append new class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kernel_cli.py`:

```python
class TestCliMainMultiReflection:
    @staticmethod
    def _write_minimal_toml(tmp_path: Path, body: str) -> Path:
        cfg = tmp_path / "config.toml"
        cfg.write_text(body)
        return cfg

    def test_happy_path_hkl_keV(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """TOML with hkl + keV → exit 0, expected filename, computed θ echoed."""
        from dfxm_geo.reciprocal_space import kernel as kmod

        captured_theta: list[float] = []
        captured_output_path: list[Path] = []

        def fake_generate_kernel(
            date: str | None = None,
            *,
            output_path: Path | None = None,
            theta: float = 0.0,
            **kwargs: object,
        ) -> Path:
            captured_theta.append(theta)
            assert output_path is not None
            captured_output_path.append(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"")
            return output_path

        monkeypatch.setattr(kmod, "generate_kernel", fake_generate_kernel)

        cfg = self._write_minimal_toml(
            tmp_path,
            "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\nNrays = 1000\n",
        )
        out_dir = tmp_path / "out"
        # cli_main needs --output to write outside pkl_files
        rc = kmod.cli_main(["--config", str(cfg), "--output", str(out_dir / "kernel.npz")])
        assert rc == 0

        # θ computed correctly (matches default Al 111 helper)
        assert captured_theta[0] == kmod._default_theta_al_111(17)

        # stdout echo
        out = capsys.readouterr().out
        assert "reflection: hkl=(-1, 1, -1)" in out
        assert "keV=17" in out
        assert "θ" in out or "theta" in out.lower()

    def test_happy_path_filename_built_when_no_output_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No --output → cli_main builds <pkl_fpath>/Resq_i_h<h>_k<k>_l<l>_<keV>keV_<date>.npz."""
        from dfxm_geo.reciprocal_space import kernel as kmod

        captured_output_path: list[Path] = []

        def fake_generate_kernel(
            date: str | None = None,
            *,
            output_path: Path | None = None,
            theta: float = 0.0,
            **kwargs: object,
        ) -> Path:
            assert output_path is not None
            captured_output_path.append(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"")
            return output_path

        monkeypatch.setattr(kmod, "generate_kernel", fake_generate_kernel)

        # Redirect pkl_fpath to tmp_path so we don't pollute the repo
        import dfxm_geo.direct_space.forward_model as fm
        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path) + os.sep, raising=True)

        cfg = self._write_minimal_toml(
            tmp_path,
            "[reciprocal]\nhkl = [2, 0, 0]\nkeV = 17.0\nNrays = 1000\n",
        )
        rc = kmod.cli_main(["--config", str(cfg)])
        assert rc == 0
        assert captured_output_path[0].name.startswith("Resq_i_h2_k0_l0_17keV_")
        assert captured_output_path[0].suffix == ".npz"
```

Also add to the imports section at the top of `tests/test_kernel_cli.py` (only `os` is new):
```python
import os
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_cli.py::TestCliMainMultiReflection::test_happy_path_hkl_keV tests/test_kernel_cli.py::TestCliMainMultiReflection::test_happy_path_filename_built_when_no_output_flag -v
```
Expected: FAIL — `cli_main` does not yet allow `hkl` or `keV` in the `[reciprocal]` block (gets rejected by the unknown-key check).

- [ ] **Step 3: Modify `cli_main` to handle `hkl` + `keV`**

In `src/dfxm_geo/reciprocal_space/kernel.py`, modify the body of `cli_main` between the `data = tomllib.load(f)` block and the existing `valid_recip_keys` check. The full updated section (replacing roughly the current lines 220–253):

```python
    import inspect

    # Bind to the original `generate_kernel` (not the module-level name) so
    # that test monkeypatches replacing kmod.generate_kernel don't shrink the
    # valid-key set to just the fake's **kwargs.
    sig = inspect.signature(_generate_kernel_original)
    valid_params = set(sig.parameters)
    # `date` and `output_path` are CLI-managed kwargs, not TOML-driven.
    # `hkl` and `keV` are cli_main-scope reflection inputs, not
    # `generate_kernel` kwargs — added explicitly to the allow-list.
    valid_recip_keys = (valid_params - {"date", "output_path"}) | {"hkl", "keV"}
    unknown = set(data["reciprocal"]) - valid_recip_keys
    if unknown:
        print(
            f"error: unknown [reciprocal] keys: {sorted(unknown)}; "
            f"valid keys are {sorted(valid_recip_keys)}.",
            file=sys.stderr,
        )
        return 1

    # Pop hkl/keV — they are cli_main-scope, not generate_kernel kwargs.
    reciprocal_kwargs = dict(data["reciprocal"])
    raw_hkl = reciprocal_kwargs.pop("hkl", None)
    raw_keV = reciprocal_kwargs.pop("keV", None)

    if (raw_hkl is None) != (raw_keV is None):
        print(
            "error: must provide both `hkl` and `keV`, or neither.",
            file=sys.stderr,
        )
        return 1

    if (raw_hkl is not None or raw_keV is not None) and "theta" in reciprocal_kwargs:
        print(
            "error: cannot specify both `theta` and `hkl`+`keV`; pick one.",
            file=sys.stderr,
        )
        return 1

    a_lattice = 4.0495e-10  # Al lattice parameter, m
    if raw_hkl is not None and raw_keV is not None:
        try:
            hkl_tuple: tuple[int, int, int] = tuple(raw_hkl)  # type: ignore[assignment]
            theta = _validate_reflection(hkl_tuple, float(raw_keV), a_lattice)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        keV_for_filename: float = float(raw_keV)
    else:
        print(
            "warning: [reciprocal] has no `hkl`/`keV`; defaulting to Al "
            "(-1, 1, -1) @ 17 keV.",
            file=sys.stderr,
        )
        hkl_tuple = (-1, 1, -1)
        keV_for_filename = 17.0
        theta = _default_theta_al_111(17)

    # Inject the computed theta so generate_kernel uses our value, not its
    # module-load default. (Skip if the TOML already set theta — that path
    # was rejected above when hkl/keV also present, so it only fires when
    # neither hkl/keV were given AND theta was.)
    if "theta" not in reciprocal_kwargs:
        reciprocal_kwargs["theta"] = theta

    # Echo computed θ for sanity (Q4).
    theta_deg = float(np.degrees(theta))
    print(
        f"reflection: hkl={hkl_tuple}, keV={keV_for_filename:g} "
        f"→ θ = {theta_deg:.4f}°"
    )

    # Build output path.
    if args.output is not None:
        output_path = args.output
    else:
        date = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = Path(fm.pkl_fpath) / _build_kernel_filename(
            hkl_tuple, keV_for_filename, date
        )

    if output_path.exists():
        if args.if_missing:
            print(f"kernel already present at {output_path}; skipping.")
            return 0
        if not args.force:
            print(
                f"refusing to overwrite existing kernel npz at {output_path}; "
                f"pass --force to regenerate.",
                file=sys.stderr,
            )
            return 1

    written = generate_kernel(output_path=output_path, **reciprocal_kwargs)
    print(f"wrote {written}")
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_cli.py::TestCliMainMultiReflection::test_happy_path_hkl_keV tests/test_kernel_cli.py::TestCliMainMultiReflection::test_happy_path_filename_built_when_no_output_flag -v
```
Expected: both PASS.

Also run the existing kernel CLI tests to confirm no regression:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_cli.py -v
```
Expected: all previously-passing tests still PASS (no regressions in `TestGenerateKernelOutputPath`, `TestDefaultConfigReciprocalBlock`, etc.).

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/reciprocal_space/kernel.py tests/test_kernel_cli.py
git commit -m "feat(kernel): plumb hkl/keV through cli_main with per-reflection filename"
```

---

## Task 4: `cli_main` soft default + WARN

**Files:**
- Modify: (no further code changes — already implemented in Task 3)
- Test: `tests/test_kernel_cli.py` (append two more tests to `TestCliMainMultiReflection`)

- [ ] **Step 1: Write the failing tests**

Append to the `TestCliMainMultiReflection` class:

```python
    def test_no_hkl_no_keV_warns_and_uses_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """[reciprocal] without hkl/keV → exit 0, WARN to stderr, default reflection used."""
        from dfxm_geo.reciprocal_space import kernel as kmod

        captured: dict[str, object] = {}

        def fake_generate_kernel(
            date: str | None = None,
            *,
            output_path: Path | None = None,
            theta: float = 0.0,
            **kwargs: object,
        ) -> Path:
            captured["theta"] = theta
            assert output_path is not None
            captured["output_path"] = output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"")
            return output_path

        monkeypatch.setattr(kmod, "generate_kernel", fake_generate_kernel)

        import dfxm_geo.direct_space.forward_model as fm
        monkeypatch.setattr(fm, "pkl_fpath", str(tmp_path) + os.sep, raising=True)

        cfg = tmp_path / "config.toml"
        cfg.write_text("[reciprocal]\nNrays = 1000\n")
        rc = kmod.cli_main(["--config", str(cfg)])
        assert rc == 0

        err = capsys.readouterr().err
        assert "[reciprocal] has no" in err
        assert "defaulting to Al" in err

        # Default theta == _default_theta_al_111(17)
        assert captured["theta"] == kmod._default_theta_al_111(17)

        # Filename uses default reflection
        out_path = captured["output_path"]
        assert isinstance(out_path, Path)
        assert out_path.name.startswith("Resq_i_h-1_k1_l-1_17keV_")
```

- [ ] **Step 2: Run test to verify it passes (Task 3 already implemented the branch)**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_cli.py::TestCliMainMultiReflection::test_no_hkl_no_keV_warns_and_uses_default -v
```
Expected: PASS. (The Task 3 implementation already handles this; Task 4 just adds the test that locks it in.)

If it fails, debug — the soft-default branch was specified in Section 4 of the design and added in Task 3 Step 3. Do not skip ahead to the next task without a passing test here.

- [ ] **Step 3: Commit**

```bash
git add tests/test_kernel_cli.py
git commit -m "test(kernel): lock in soft-default-with-WARN branch for missing hkl/keV"
```

---

## Task 5: `cli_main` input-coherence errors

**Files:**
- Modify: (no further code changes — already implemented in Task 3)
- Test: `tests/test_kernel_cli.py` (append five more tests to `TestCliMainMultiReflection`)

- [ ] **Step 1: Write the failing tests**

Append to `TestCliMainMultiReflection`:

```python
    def test_hkl_only_no_keV_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from dfxm_geo.reciprocal_space import kernel as kmod

        cfg = tmp_path / "config.toml"
        cfg.write_text("[reciprocal]\nhkl = [-1, 1, -1]\nNrays = 1000\n")
        rc = kmod.cli_main(["--config", str(cfg)])
        assert rc == 1
        assert "must provide both" in capsys.readouterr().err

    def test_keV_only_no_hkl_errors(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from dfxm_geo.reciprocal_space import kernel as kmod

        cfg = tmp_path / "config.toml"
        cfg.write_text("[reciprocal]\nkeV = 17.0\nNrays = 1000\n")
        rc = kmod.cli_main(["--config", str(cfg)])
        assert rc == 1
        assert "must provide both" in capsys.readouterr().err

    def test_theta_plus_hkl_errors(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from dfxm_geo.reciprocal_space import kernel as kmod

        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[reciprocal]\nhkl = [-1, 1, -1]\nkeV = 17.0\ntheta = 0.165\nNrays = 1000\n"
        )
        rc = kmod.cli_main(["--config", str(cfg)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "cannot specify both" in err
        assert "`theta`" in err

    def test_zero_hkl_errors_cleanly(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """ValueError from _validate_reflection surfaces as exit 1 + stderr, not a traceback."""
        from dfxm_geo.reciprocal_space import kernel as kmod

        cfg = tmp_path / "config.toml"
        cfg.write_text("[reciprocal]\nhkl = [0, 0, 0]\nkeV = 17.0\nNrays = 1000\n")
        rc = kmod.cli_main(["--config", str(cfg)])
        assert rc == 1
        assert "not a valid reflection" in capsys.readouterr().err

    def test_unsatisfiable_bragg_errors_cleanly(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from dfxm_geo.reciprocal_space import kernel as kmod

        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[reciprocal]\nhkl = [10, 10, 10]\nkeV = 1.0\nNrays = 1000\n"
        )
        rc = kmod.cli_main(["--config", str(cfg)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Bragg condition unsatisfiable" in err
        assert "λ=" in err
```

- [ ] **Step 2: Run tests to verify they pass (Task 3 already implemented these branches)**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_cli.py::TestCliMainMultiReflection -v
```
Expected: all `TestCliMainMultiReflection` tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_kernel_cli.py
git commit -m "test(kernel): cover cli_main input-coherence errors and Bragg unsatisfiability"
```

---

## Task 6: Update `configs/default.toml` with `hkl` + `keV`

**Files:**
- Modify: `configs/default.toml`
- Test: `tests/test_kernel_cli.py` (extend existing `TestDefaultConfigReciprocalBlock`)

- [ ] **Step 1: Inspect current `configs/default.toml` to find the `[reciprocal]` block**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_cli.py::TestDefaultConfigReciprocalBlock -v
```
Then read the file to see the current `[reciprocal]` block layout:
```
type configs\default.toml
```

- [ ] **Step 2: Write failing tests in `TestDefaultConfigReciprocalBlock`**

Append two methods to the existing `TestDefaultConfigReciprocalBlock` class in `tests/test_kernel_cli.py`:

```python
    def test_default_toml_has_hkl_key(self) -> None:
        """configs/default.toml must declare hkl explicitly for the
        post-sub-project-A bootstrap to record the reflection in the filename."""
        import tomllib

        with open("configs/default.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["reciprocal"]["hkl"] == [-1, 1, -1]

    def test_default_toml_has_keV_key(self) -> None:
        import tomllib

        with open("configs/default.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["reciprocal"]["keV"] == 17.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_cli.py::TestDefaultConfigReciprocalBlock::test_default_toml_has_hkl_key tests/test_kernel_cli.py::TestDefaultConfigReciprocalBlock::test_default_toml_has_keV_key -v
```
Expected: FAIL with `KeyError: 'hkl'` and `KeyError: 'keV'`.

- [ ] **Step 4: Add `hkl` and `keV` to `configs/default.toml`**

Inside the `[reciprocal]` block, add (at the top of the block to make them the lead settings):

```toml
hkl = [-1, 1, -1]
keV = 17.0
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_cli.py::TestDefaultConfigReciprocalBlock -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/default.toml tests/test_kernel_cli.py
git commit -m "config: add hkl=[-1,1,-1] and keV=17.0 to default [reciprocal] block"
```

---

## Task 7: Update `pkl_fn` in `forward_model.py` + regression test

**Context:** `pkl_fn` (line 61) is the filename `dfxm-forward` looks for at import time. The module is *lazy-load*: the file is only opened on demand, so the value can point at a not-yet-existing file safely. We update the value to a pattern-conforming placeholder; the user runs `dfxm-bootstrap` post-PR to produce the real file and updates the value to match (operational follow-up).

**Files:**
- Modify: `src/dfxm_geo/direct_space/forward_model.py:60–61`
- Test: `tests/test_kernel_cli.py` (append new class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kernel_cli.py`:

```python
class TestPklFnRegression:
    def test_pkl_fn_matches_new_pattern(self) -> None:
        """pkl_fn must follow the per-reflection pattern from sub-project A.

        After bootstrap regen, the user updates the literal value to the
        produced filename; this regression guard ensures it stays
        pattern-conforming and doesn't silently revert to the old
        Resq_i_<date>.npz format.
        """
        import re

        import dfxm_geo.direct_space.forward_model as fm

        assert re.fullmatch(
            r"Resq_i_h-1_k1_l-1_17keV_\d{8}_\d{4}\.npz",
            fm.pkl_fn,
        ), f"pkl_fn={fm.pkl_fn!r} does not match the new per-reflection pattern"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_cli.py::TestPklFnRegression -v
```
Expected: FAIL — current `pkl_fn = "Resq_i_20260516_2100.npz"` does not match the new pattern.

- [ ] **Step 3: Update `pkl_fn` to a pattern-conforming default**

In `src/dfxm_geo/direct_space/forward_model.py`, replace line 61:

Old:
```python
pkl_fn = "Resq_i_20260516_2100.npz"  # Update after `dfxm-bootstrap` regen
```

New:
```python
pkl_fn = "Resq_i_h-1_k1_l-1_17keV_20260520_2100.npz"  # Update after `dfxm-bootstrap` regen (per-reflection pattern)
```

Also update the surrounding comment block (lines 59–60) to reflect the new pattern:

Old:
```python
# Constant name preserved (`pkl_fn`) so import-time monkeypatches in tests
# and the dfxm-bootstrap CLI don't break. Value is now the npz canonical.
```

New:
```python
# Constant name preserved (`pkl_fn`) so import-time monkeypatches in tests
# and the dfxm-bootstrap CLI don't break. Value follows the
# Resq_i_h{h}_k{k}_l{l}_{keV}keV_{date}.npz pattern introduced in
# sub-project A; the date stamp is a placeholder until the next bootstrap
# regen produces the real file on each host.
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest tests/test_kernel_cli.py::TestPklFnRegression -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dfxm_geo/direct_space/forward_model.py tests/test_kernel_cli.py
git commit -m "fix(forward): update pkl_fn default to per-reflection pattern"
```

---

## Task 8: Full verification — pytest + mypy + smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the full pytest suite**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m pytest -q
```
Expected: 0 failures. Any new test count should equal the sum of:
- `TestValidateReflection`: 9
- `TestBuildKernelFilename`: 5
- `TestCliMainMultiReflection`: 7
- `TestPklFnRegression`: 1
- Plus 2 added to `TestDefaultConfigReciprocalBlock`
- = 24 new tests.

The `tests/data/golden/Fd_find_smoke.npy` smoke test must still pass — that's the safety net per CLAUDE.md.

- [ ] **Step 2: Run mypy**

Run:
```
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m mypy src/dfxm_geo/
```
Expected: 0 errors. The `# type: ignore[assignment]` on `hkl_tuple = tuple(raw_hkl)` is acceptable — `raw_hkl` is `Any` (from tomllib) and the cast is verified by `_validate_reflection`'s `len(hkl) != 3` check.

If mypy complains about anything else, fix inline.

- [ ] **Step 3: Smoke-test the CLI end-to-end with a tiny config**

Create a temporary config and run `dfxm-bootstrap` against it with monkey-patched `Nrays`/grid to keep it fast.

Run:
```
echo [reciprocal] > nul && (echo [reciprocal] & echo hkl = [-1, 1, -1] & echo keV = 17.0 & echo Nrays = 1000 & echo npoints1 = 20 & echo npoints2 = 20 & echo npoints3 = 20) > tmp_bootstrap_config.toml
C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe -m dfxm_geo.reciprocal_space.kernel --config tmp_bootstrap_config.toml --output tmp_kernel.npz
```

Expected output (the date stamp will reflect runtime):
```
reflection: hkl=(-1, 1, -1), keV=17 → θ = 9.4517°
wrote tmp_kernel.npz
```

Then clean up:
```
del tmp_bootstrap_config.toml tmp_kernel.npz
```

- [ ] **Step 4: Commit any inline fixes from Steps 1–3 (if needed)**

If pytest or mypy required fixups, commit them:
```bash
git add -p
git commit -m "fix(spec-followup): inline corrections from full verification"
```

If no fixes needed, skip this step.

- [ ] **Step 5: Push the branch (after confirming with Sina per CLAUDE.md)**

```bash
git push -u origin chore/spec-bootstrap-multi-reflection
```

Open the PR with `gh pr create` only after Sina explicitly approves. Use this body skeleton:

```
## Summary
- Adds hkl + keV TOML keys to [reciprocal]; cli_main computes θ from
  them via new _validate_reflection helper.
- New _build_kernel_filename helper for the per-reflection filename
  pattern Resq_i_h{h}_k{k}_l{l}_{keV}keV_{date}.npz.
- Updates pkl_fn default in forward_model.py to the new pattern.

## Spec
- docs/superpowers/specs/2026-05-20-bootstrap-multi-reflection-design.md

## Test plan
- [ ] pytest -q (24 new tests pass; smoke + golden untouched)
- [ ] mypy src/dfxm_geo/ (0 errors)
- [ ] Manual smoke: dfxm-bootstrap with the new config writes the
      pattern-conforming filename and echoes θ in degrees on stdout.

## Operational follow-up (post-merge)
- Run `dfxm-bootstrap --config configs/default.toml` on laptop +
  cluster to produce the real per-reflection kernel file.
- Update `pkl_fn` in forward_model.py:61 to match the actually produced
  filename (replacing the 20260520_2100 placeholder timestamp).
```

---

## Operational follow-ups (after PR merge)

These are deliberately out of the PR scope:

- Run `dfxm-bootstrap --config configs/default.toml` on each host (laptop, DTU cluster) to produce the real per-reflection kernel file. ~50 s per host with the default `Nrays=1e8`.
- Update `pkl_fn` in `forward_model.py:61` to match the produced filename (replacing the `20260520_2100` placeholder timestamp). Follow-up commit on `main`.
- Update `cleanup_session_state.md` in auto-memory with the new round summary.

These follow the same operational pattern as the PR #6 post-bootstrap regen step Sina already knows.
