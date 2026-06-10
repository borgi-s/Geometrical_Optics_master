"""dfxm-bootstrap [[reflections]] loop: one kernel per unique (theta, eta, keV) group.

Physics caution verified 2026-06-10:
  compute_omega_eta at 19.1 keV for Al cubic identity mount gives:
    (1,1,3):    eta_1=-0.353125, eta_2=+0.353125
    (-1,-1,3):  eta_1=+0.353125, eta_2=-0.353125
  Solution 1 (default) gives OPPOSITE signed etas → 2 groups, not 1.
  Fix: add `eta = 0.3531` to [geometry] so the resolver matches the +0.3531
  branch for both entries (solution 2 for (1,1,3), solution 1 for (-1,-1,3)).
  Both land at eta=+0.353125 rad → single group, single generate_kernel call.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import dfxm_geo.reciprocal_space.kernel as kernel_mod

# eta = 0.3531 rad (≈ 20.23°) is the shared branch that both (1,1,3) and
# (-1,-1,3) reach with this Al cubic mount at 19.1 keV. Without this override
# the default solution-1 branch gives opposite-signed etas and they'd land in
# different groups. See module docstring for the full derivation.
GROUP1_TOML = """
[reciprocal]
keV = 19.1

[geometry]
mode = "oblique"
eta  = 0.3531

[crystal]
lattice = "cubic"
a       = 4.0493e-10
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]

[[reflections]]
hkl = [1, 1, 3]
[[reflections]]
hkl = [-1, -1, 3]
"""

MIXED_TOML = (
    GROUP1_TOML
    + """
[[reflections]]
hkl = [1, 1, 1]
omega_solution = 1
"""
)


@pytest.fixture
def fake_generate(monkeypatch, tmp_path):
    calls: list[dict] = []

    def _fake(output_path=None, **kwargs):
        calls.append({"output_path": output_path, **kwargs})
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake-npz")
        return p

    monkeypatch.setattr(kernel_mod, "generate_kernel", _fake)
    return calls


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_same_group_bootstraps_one_kernel(fake_generate, tmp_path):
    rc = kernel_mod.cli_main(
        ["--config", str(_write(tmp_path, GROUP1_TOML)), "--output", str(tmp_path / "k")]
    )
    assert rc in (None, 0)
    assert len(fake_generate) == 1  # ONE kernel for the shared (theta, eta) group


def test_mixed_groups_bootstrap_per_group(fake_generate, tmp_path):
    kernel_mod.cli_main(
        ["--config", str(_write(tmp_path, MIXED_TOML)), "--output", str(tmp_path / "k")]
    )
    assert len(fake_generate) == 2  # {113}-group + 111


def test_manifest_written_next_to_kernels(fake_generate, tmp_path):
    kernel_mod.cli_main(
        ["--config", str(_write(tmp_path, MIXED_TOML)), "--output", str(tmp_path / "k")]
    )
    manifest = tmp_path / "k" / "kernel_manifest.toml"
    assert manifest.is_file()
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    assert len(data["kernels"]) == 2
    hkls = {tuple(h) for entry in data["kernels"] for h in entry["reflections"]}
    assert (1, 1, 3) in hkls and (1, 1, 1) in hkls
    for entry in data["kernels"]:
        assert {"group", "theta", "eta", "keV", "filename", "reflections", "omegas"} <= set(entry)


def test_if_missing_skips_existing_kernel(fake_generate, tmp_path):
    out = tmp_path / "k"
    out.mkdir()
    # Pre-create a kernel whose name matches the (theta, eta, keV) glob pattern for
    # GROUP1_TOML.  Verified: resolve_reflections for (1,1,3)+(-1,-1,3) at 19.1 keV
    # with Al cubic identity mount + eta=0.3531 gives theta≈0.2691, eta≈0.3531 (4dp).
    pre = out / "Resq_i_theta0.2691rad_eta0.3531rad_19.1keV_20260101_0000.npz"
    pre.write_bytes(b"existing")
    rc = kernel_mod.cli_main(
        ["--config", str(_write(tmp_path, GROUP1_TOML)), "--output", str(out), "--if-missing"]
    )
    assert rc in (None, 0)
    assert len(fake_generate) == 0  # skipped — kernel already on disk
    manifest = tomllib.loads((out / "kernel_manifest.toml").read_text(encoding="utf-8"))
    assert manifest["kernels"][0]["filename"] == pre.name


def test_output_naming_existing_file_errors(fake_generate, tmp_path):
    f = tmp_path / "afile.npz"
    f.write_bytes(b"x")
    rc = kernel_mod.cli_main(["--config", str(_write(tmp_path, GROUP1_TOML)), "--output", str(f)])
    assert rc == 1
    assert len(fake_generate) == 0


def test_missing_keV_warns(fake_generate, tmp_path, capsys):
    no_kev_toml = """
[reciprocal]

[geometry]
mode = "oblique"
eta  = 0.3531

[crystal]
lattice = "cubic"
a       = 4.0493e-10
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]

[[reflections]]
hkl = [1, 1, 3]
"""
    kernel_mod.cli_main(
        ["--config", str(_write(tmp_path, no_kev_toml)), "--output", str(tmp_path / "k")]
    )
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    assert "17.0 keV" in captured.err


def test_hkl_in_reciprocal_with_reflections_errors(fake_generate, tmp_path):
    conflict_toml = GROUP1_TOML.replace(
        "[reciprocal]\nkeV = 19.1", "[reciprocal]\nkeV = 19.1\nhkl = [1, 1, 1]"
    )
    rc = kernel_mod.cli_main(
        ["--config", str(_write(tmp_path, conflict_toml)), "--output", str(tmp_path / "k")]
    )
    assert rc == 1
    assert len(fake_generate) == 0
