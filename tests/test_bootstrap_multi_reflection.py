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
