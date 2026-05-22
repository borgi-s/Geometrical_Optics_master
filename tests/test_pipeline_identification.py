"""Unit tests for dfxm-identify pipeline shape."""

from pathlib import Path

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    AxisScanConfig,
    IdentificationConfig,
    IdentificationCrystalConfig,
    IdentificationMonteCarloConfig,
    IdentificationNoiseConfig,
    IOConfig,
    ReciprocalConfig,
    ScanConfig,
    _run_identification_multi,
    _run_identification_single,
    cli_main_identify,
    load_identification_config,
    run_identification,
)


def test_identification_crystal_config_defaults():
    cfg = IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1))
    assert cfg.slip_plane_normal == (1, 1, 1)
    assert cfg.angle_start_deg == 0.0
    assert cfg.angle_stop_deg == 350.0
    assert cfg.angle_step_deg == 10.0
    assert cfg.b_vector_indices is None
    assert cfg.sweep_all_slip_planes is True
    assert cfg.exclude_invisibility is True
    assert cfg.invisibility_threshold_deg == 10.0


def test_identification_scan_config_defaults():
    """IdentificationNoiseConfig holds the old IdentificationScanConfig noise fields."""
    noise = IdentificationNoiseConfig()
    assert noise.poisson_noise is True
    assert noise.rng_seed == 0
    assert noise.intensity_scale == 7.0


def test_identification_montecarlo_config_defaults():
    cfg = IdentificationMonteCarloConfig()
    assert cfg.n_samples == 1000
    assert cfg.pos_std_um == 5.0
    assert cfg.n_png_previews == 50


def test_identification_crystal_config_is_frozen():
    from dataclasses import FrozenInstanceError

    cfg = IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1))
    with pytest.raises(FrozenInstanceError):
        cfg.angle_step_deg = 5.0  # type: ignore[misc]


def _make_io_config():
    return IOConfig(
        fn_prefix="/mosa_test_0000_",
        ftype=".npy",
        dislocs_dirname="identify",
        perfect_dirname="ignored",
        include_perfect_crystal=False,
    )


def _make_scan_noise(rng_seed: int = 0, intensity_scale: float = 7.0):
    """Return a (ScanConfig, IdentificationNoiseConfig) pair replacing old IdentificationScanConfig."""
    scan = ScanConfig(phi=AxisScanConfig(value=150e-6))
    noise = IdentificationNoiseConfig(rng_seed=rng_seed, intensity_scale=intensity_scale)
    return scan, noise


def test_identification_config_mode_single_ok():
    scan, noise = _make_scan_noise()
    cfg = IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=scan,
        noise=noise,
        io=_make_io_config(),
    )
    assert cfg.mode == "single"
    assert cfg.multi is None


def test_identification_config_mode_multi_ok():
    scan, noise = _make_scan_noise()
    cfg = IdentificationConfig(
        mode="multi",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=scan,
        noise=noise,
        multi=IdentificationMonteCarloConfig(),
        io=_make_io_config(),
    )
    assert cfg.mode == "multi"
    assert cfg.multi is not None


def test_identification_config_mode_multi_requires_multi_block():
    scan, noise = _make_scan_noise()
    with pytest.raises(ValueError, match="mode='multi'"):
        IdentificationConfig(
            mode="multi",
            crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
            scan=scan,
            noise=noise,
            multi=None,
            io=_make_io_config(),
        )


def test_identification_config_invalid_slip_plane_raises():
    scan, noise = _make_scan_noise()
    with pytest.raises(ValueError, match="not one of the four"):
        IdentificationConfig(
            mode="single",
            crystal=IdentificationCrystalConfig(slip_plane_normal=(2, 0, 0)),
            scan=scan,
            noise=noise,
            io=_make_io_config(),
        )


def test_identification_config_invalid_mode_raises():
    scan, noise = _make_scan_noise()
    with pytest.raises(ValueError, match="mode must be"):
        IdentificationConfig(
            mode="bogus",  # type: ignore[arg-type]
            crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
            scan=scan,
            noise=noise,
            io=_make_io_config(),
        )


def test_load_identification_config_single(tmp_path):
    """Round-trip a single-mode TOML file → IdentificationConfig."""
    toml_text = """
mode = "single"

[crystal]
slip_plane_normal = [1, 1, 1]
angle_start_deg = 0.0
angle_stop_deg = 350.0
angle_step_deg = 10.0
sweep_all_slip_planes = true
exclude_invisibility = true
invisibility_threshold_deg = 10.0

[scan.phi]
value = 1.5e-4

[noise]
poisson_noise = true
rng_seed = 0
intensity_scale = 7.0

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify"
perfect_dirname = "ignored"
include_perfect_crystal = false

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0
"""
    cfg_path = tmp_path / "identification_single.toml"
    cfg_path.write_text(toml_text, encoding="utf-8")

    cfg = load_identification_config(cfg_path)
    assert cfg.mode == "single"
    assert cfg.crystal.slip_plane_normal == (1, 1, 1)
    assert cfg.scan.phi.value == pytest.approx(1.5e-4)
    assert cfg.multi is None


def test_load_identification_config_multi(tmp_path):
    """Multi-mode TOML round-trips, including the [multi] block."""
    toml_text = """
mode = "multi"

[crystal]
slip_plane_normal = [1, 1, 1]

[scan.phi]
value = 1.5e-4

[noise]
rng_seed = 42

[multi]
n_samples = 100
pos_std_um = 3.0
n_png_previews = 10

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify_multi"
perfect_dirname = "ignored"
include_perfect_crystal = false

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0
"""
    cfg_path = tmp_path / "identification_multi.toml"
    cfg_path.write_text(toml_text, encoding="utf-8")

    cfg = load_identification_config(cfg_path)
    assert cfg.mode == "multi"
    assert cfg.multi is not None
    assert cfg.multi.n_samples == 100
    assert cfg.noise.rng_seed == 42


def test_load_identification_config_missing_mode_raises(tmp_path):
    """A TOML missing the top-level `mode = ...` field raises."""
    cfg_path = tmp_path / "bad.toml"
    cfg_path.write_text("[crystal]\nslip_plane_normal = [1, 1, 1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing top-level 'mode'"):
        load_identification_config(cfg_path)


def _tiny_single_config(tmp_path):
    scan = ScanConfig(phi=AxisScanConfig(value=150e-6))
    noise = IdentificationNoiseConfig(rng_seed=0, intensity_scale=1.0)
    return IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=(1, 1, 1),
            angle_start_deg=0.0,
            angle_stop_deg=90.0,
            angle_step_deg=90.0,  # only 2 angles: 0 and 90
            b_vector_indices=[0, 1],  # only 2 Burgers vectors
            sweep_all_slip_planes=False,  # just one plane
            exclude_invisibility=False,  # don't filter
        ),
        scan=scan,
        noise=noise,
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
        io=_make_io_config(),
    )


def test_run_identification_single_writes_expected_count(tmp_path, monkeypatch):
    """Tiny sweep (1 slip plane × 2 b × 2 angles = 4 configs) writes a master
    HDF5 with 4 /N.1 scans and 4 per-scan detector files (v1.2.0 layout).
    """
    import h5py

    expected_image = np.ones((170, 510))
    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: expected_image)
    # Writer falls back to fm._loaded_kernel_path for provenance when
    # kernel_npz is None. Point it at a dummy file; no real kernel IO.
    fake_kernel = tmp_path / "fake_kernel.npz"
    np.savez(fake_kernel, Resq_i=np.zeros(1), hkl=np.array([-1, 1, -1]), keV=17.0)
    monkeypatch.setattr(fm, "_loaded_kernel_path", fake_kernel)

    output_dir = tmp_path / "out"
    cfg = _tiny_single_config(tmp_path)

    result = _run_identification_single(cfg, output_dir)

    master = output_dir / "dfxm_identify.h5"
    assert master.is_file()
    # 4 scan dirs, one detector file each
    for k in range(1, 5):
        det = output_dir / f"scan{k:04d}" / "dfxm_sim_detector_0000.h5"
        assert det.is_file(), f"missing {det}"
    # No legacy sidecars
    assert not (output_dir / "manifest.csv").exists()
    assert not (output_dir / "n_1_1_1").exists()
    with h5py.File(master, "r") as f:
        scan_ids = sorted(k for k in f if k != "dfxm_geo")
        assert scan_ids == ["1.1", "2.1", "3.1", "4.1"]
    assert result["n_images"] == 4


def _tiny_multi_config():
    scan = ScanConfig(phi=AxisScanConfig(value=150e-6))
    noise = IdentificationNoiseConfig(rng_seed=0, intensity_scale=1.0)
    return IdentificationConfig(
        mode="multi",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=scan,
        noise=noise,
        multi=IdentificationMonteCarloConfig(n_samples=3, pos_std_um=2.0, n_png_previews=2),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
        io=_make_io_config(),
    )


def test_run_identification_multi_writes_samples_and_manifest(tmp_path, monkeypatch):
    """n_samples=3 writes 3 .npy + 2 PNGs (n_png_previews=2) + manifest with 3 rows."""
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm

    expected_image = np.ones((170, 510))
    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: expected_image)

    output_dir = tmp_path / "out"
    cfg = _tiny_multi_config()

    result = _run_identification_multi(cfg, output_dir)

    npys = sorted((output_dir / "im_data").glob("*.npy"))
    pngs = sorted((output_dir / "images").glob("*.png"))
    assert len(npys) == 3
    assert len(pngs) == 2
    manifest = output_dir / "manifest.csv"
    assert manifest.is_file()
    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4  # header + 3 rows
    assert result["n_samples"] == 3


def test_run_identification_multi_is_deterministic_for_seed(tmp_path, monkeypatch):
    """Two runs at the same seed produce identical manifests."""
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm

    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    counter = {"n": 0}

    def fake_forward(*args, **kwargs):
        counter["n"] += 1
        return np.full((170, 510), float(counter["n"]))

    monkeypatch.setattr(fm, "forward", fake_forward)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    cfg = _tiny_multi_config()

    _run_identification_multi(cfg, out1)
    counter["n"] = 0
    _run_identification_multi(cfg, out2)

    m1 = (out1 / "manifest.csv").read_text(encoding="utf-8")
    m2 = (out2 / "manifest.csv").read_text(encoding="utf-8")
    assert m1 == m2


def test_run_identification_dispatches_to_single(tmp_path, monkeypatch):
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo import pipeline

    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: np.ones((170, 510)))
    monkeypatch.setattr(pipeline, "_lookup_and_load_kernel", lambda *args, **kwargs: None)
    # Stand in for the kernel provenance the writer reads.
    fake_kernel = tmp_path / "fake_kernel.npz"
    np.savez(fake_kernel, Resq_i=np.zeros(1), hkl=np.array([-1, 1, -1]), keV=17.0)
    monkeypatch.setattr(fm, "_loaded_kernel_path", fake_kernel)

    cfg = _tiny_single_config(tmp_path)
    result = run_identification(cfg, tmp_path / "out")
    assert "n_images" in result


def test_run_identification_dispatches_to_multi(tmp_path, monkeypatch):
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo import pipeline

    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: np.ones((170, 510)))
    monkeypatch.setattr(pipeline, "_lookup_and_load_kernel", lambda *args, **kwargs: None)

    cfg = _tiny_multi_config()
    result = run_identification(cfg, tmp_path / "out")
    assert "n_samples" in result


def test_cli_main_identify_parses_args(tmp_path, monkeypatch):
    """Smoke test: CLI parses --config and --output, calls run_identification, returns 0."""
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo import pipeline

    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: np.ones((170, 510)))
    monkeypatch.setattr(pipeline, "_lookup_and_load_kernel", lambda *args, **kwargs: None)
    fake_kernel = tmp_path / "fake_kernel.npz"
    np.savez(fake_kernel, Resq_i=np.zeros(1), hkl=np.array([-1, 1, -1]), keV=17.0)
    monkeypatch.setattr(fm, "_loaded_kernel_path", fake_kernel)

    toml_text = """
mode = "single"

[crystal]
slip_plane_normal = [1, 1, 1]
angle_start_deg = 0.0
angle_stop_deg = 0.0
angle_step_deg = 10.0
b_vector_indices = [0]
sweep_all_slip_planes = false
exclude_invisibility = false

[scan.phi]
value = 1.5e-4

[noise]
poisson_noise = false
rng_seed = 0
intensity_scale = 1.0

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify"
perfect_dirname = "ignored"
include_perfect_crystal = false

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0
"""
    cfg_path = tmp_path / "id.toml"
    cfg_path.write_text(toml_text, encoding="utf-8")
    out_dir = tmp_path / "out"

    exit_code = cli_main_identify(["--config", str(cfg_path), "--output", str(out_dir)])
    assert exit_code == 0
    assert (out_dir / "dfxm_identify.h5").is_file()


def test_example_single_config_loads():
    """configs/identification_single.toml parses and validates."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_identification_config(repo_root / "configs" / "identification_single.toml")
    assert cfg.mode == "single"
    assert cfg.crystal.sweep_all_slip_planes is True
    assert cfg.crystal.exclude_invisibility is True


def test_example_multi_config_loads():
    """configs/identification_multi.toml parses and validates."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_identification_config(repo_root / "configs" / "identification_multi.toml")
    assert cfg.mode == "multi"
    assert cfg.multi is not None
    assert cfg.multi.n_samples == 1000


def test_dfxm_identify_cli_end_to_end(tmp_path):
    """Invoke `dfxm-identify` via subprocess on a tiny config; confirm exit 0.

    Skipped if the default Resq_i kernel npz is missing (CI runners
    without the npz just skip; local dev with the npz exercises the
    full forward call).
    """
    import subprocess
    import sys as _sys

    import dfxm_geo.direct_space.forward_model as fm

    kernel_dir = Path(fm.pkl_fpath)
    matches = sorted(kernel_dir.glob("Resq_i_h-1_k1_l-1_17keV_*.npz"))
    if not matches:
        pytest.skip(f"no kernel npz found in {kernel_dir}")

    toml_text = """
mode = "single"

[crystal]
slip_plane_normal = [1, 1, 1]
angle_start_deg = 0.0
angle_stop_deg = 0.0
angle_step_deg = 10.0
b_vector_indices = [0]
sweep_all_slip_planes = false
exclude_invisibility = false

[scan.phi]
value = 1.5e-4

[noise]
poisson_noise = false
rng_seed = 0
intensity_scale = 1.0

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify"
perfect_dirname = "ignored"
include_perfect_crystal = false

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0
"""
    cfg_path = tmp_path / "smoke.toml"
    cfg_path.write_text(toml_text, encoding="utf-8")
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [
            _sys.executable,
            "-c",
            "from dfxm_geo.pipeline import cli_main_identify; "
            f"raise SystemExit(cli_main_identify(['--config', r'{cfg_path}', '--output', r'{out_dir}']))",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    master = out_dir / "dfxm_identify.h5"
    assert master.is_file(), f"missing master HDF5 in {out_dir}"
    # 1 plane × 1 b × 1 angle (0 to 0 inclusive @ step 10) = 1 scan
    det = out_dir / "scan0001" / "dfxm_sim_detector_0000.h5"
    assert det.is_file(), f"missing per-scan detector file: {det}"
    # No legacy sidecar layout
    assert not (out_dir / "n_1_1_1").exists()
    assert not (out_dir / "manifest.csv").exists()


def test_identification_zscan_config_defaults():
    from dfxm_geo.pipeline import IdentificationZScanConfig

    cfg = IdentificationZScanConfig(
        z_offsets_um=[0.0],
        phi_range_deg=0.03,
        phi_steps=11,
        chi_range_deg=0.1,
        chi_steps=11,
    )
    assert cfg.z_offsets_um == [0.0]
    assert cfg.phi_steps == 11
    assert cfg.chi_steps == 11
    assert cfg.include_secondary is True
    assert cfg.secondary_rng_offset == 1


def test_identification_zscan_config_is_frozen():
    from dataclasses import FrozenInstanceError

    from dfxm_geo.pipeline import IdentificationZScanConfig

    cfg = IdentificationZScanConfig(
        z_offsets_um=[0.0],
        phi_range_deg=0.03,
        phi_steps=11,
        chi_range_deg=0.1,
        chi_steps=11,
    )
    with pytest.raises(FrozenInstanceError):
        cfg.phi_steps = 21  # type: ignore[misc]


def _tiny_zscan_config(slip_plane=(1, 1, 1)):
    from dfxm_geo.pipeline import IdentificationZScanConfig

    scan = ScanConfig(phi=AxisScanConfig(value=150e-6))
    noise = IdentificationNoiseConfig(rng_seed=0, intensity_scale=1.0)
    return IdentificationConfig(
        mode="z-scan",
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=slip_plane,
            angle_start_deg=0.0,
            angle_stop_deg=0.0,
            angle_step_deg=10.0,
            b_vector_indices=[0],
            sweep_all_slip_planes=False,
            exclude_invisibility=False,
        ),
        scan=scan,
        noise=noise,
        zscan=IdentificationZScanConfig(
            z_offsets_um=[0.0],
            phi_range_deg=0.03,
            phi_steps=2,
            chi_range_deg=0.1,
            chi_steps=2,
            include_secondary=False,
        ),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
        io=_make_io_config(),
    )


def test_identification_config_mode_zscan_ok():
    cfg = _tiny_zscan_config()
    assert cfg.mode == "z-scan"
    assert cfg.zscan is not None
    assert cfg.multi is None


def test_identification_config_mode_zscan_requires_zscan_block():
    from dfxm_geo.pipeline import IdentificationZScanConfig  # noqa: F401

    scan, noise = _make_scan_noise()
    with pytest.raises(ValueError, match="mode='z-scan'"):
        IdentificationConfig(
            mode="z-scan",
            crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
            scan=scan,
            noise=noise,
            zscan=None,
            io=_make_io_config(),
        )


def test_identification_config_mode_single_rejects_zscan_block():
    """Passing a zscan block in single mode is a config error (clarity)."""
    from dfxm_geo.pipeline import IdentificationZScanConfig

    scan, noise = _make_scan_noise()
    with pytest.raises(ValueError, match="single|multi.*zscan"):
        IdentificationConfig(
            mode="single",
            crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
            scan=scan,
            noise=noise,
            zscan=IdentificationZScanConfig(
                z_offsets_um=[0.0],
                phi_range_deg=0.03,
                phi_steps=2,
                chi_range_deg=0.1,
                chi_steps=2,
            ),
            io=_make_io_config(),
        )


def test_load_identification_config_zscan(tmp_path):
    """A mode='z-scan' TOML round-trips, including the [zscan] block."""
    toml_text = """
mode = "z-scan"

[crystal]
slip_plane_normal = [1, 1, 1]

[scan.phi]
value = 1.5e-4

[noise]
rng_seed = 7

[zscan]
z_offsets_um = [-1.0, 0.0, 1.0]
phi_range_deg = 0.03
phi_steps = 21
chi_range_deg = 0.1
chi_steps = 21
include_secondary = true
secondary_rng_offset = 2

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify_zscan"
perfect_dirname = "ignored"
include_perfect_crystal = false

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0
"""
    cfg_path = tmp_path / "id_zscan.toml"
    cfg_path.write_text(toml_text, encoding="utf-8")

    cfg = load_identification_config(cfg_path)
    assert cfg.mode == "z-scan"
    assert cfg.zscan is not None
    assert cfg.zscan.z_offsets_um == [-1.0, 0.0, 1.0]
    assert cfg.zscan.phi_steps == 21
    assert cfg.zscan.include_secondary is True
    assert cfg.zscan.secondary_rng_offset == 2
    assert cfg.noise.rng_seed == 7


def test_run_identification_zscan_writes_per_config_rocking_grid(tmp_path, monkeypatch):
    """Tiny z-scan: 1 layer x 1 plane x 1 b x 1 alpha = 1 configuration ->
    1 config directory with phi_steps*chi_steps .npy files (4 here)."""
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo.pipeline import _run_identification_zscan

    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: np.ones((170, 510)))

    output_dir = tmp_path / "out"
    cfg = _tiny_zscan_config()
    result = _run_identification_zscan(cfg, output_dir)

    layer_dirs = sorted(output_dir.glob("layer_*"))
    assert len(layer_dirs) == 1
    config_dirs = list((layer_dirs[0] / "n_1_1_1").glob("b*"))
    assert len(config_dirs) == 1
    npys = list(config_dirs[0].glob("*.npy"))
    assert len(npys) == 4  # 2 phi * 2 chi
    manifest = output_dir / "manifest.csv"
    assert manifest.is_file()
    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # header + 1 row
    assert result["n_configurations"] == 1


def test_run_identification_zscan_is_deterministic_for_seed(tmp_path, monkeypatch):
    """Two runs at same seed produce identical manifests (same secondary draws)."""
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo.pipeline import IdentificationZScanConfig, _run_identification_zscan

    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: np.ones((170, 510)))

    scan = ScanConfig(phi=AxisScanConfig(value=150e-6))
    noise = IdentificationNoiseConfig(rng_seed=0, intensity_scale=1.0)
    cfg = IdentificationConfig(
        mode="z-scan",
        crystal=IdentificationCrystalConfig(
            slip_plane_normal=(1, 1, 1),
            angle_start_deg=0.0,
            angle_stop_deg=90.0,
            angle_step_deg=90.0,
            b_vector_indices=[0, 1],
            sweep_all_slip_planes=False,
            exclude_invisibility=False,
        ),
        scan=scan,
        noise=noise,
        zscan=IdentificationZScanConfig(
            z_offsets_um=[0.0],
            phi_range_deg=0.03,
            phi_steps=2,
            chi_range_deg=0.1,
            chi_steps=2,
            include_secondary=True,  # exercise the secondary draw
        ),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
        io=_make_io_config(),
    )

    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    _run_identification_zscan(cfg, out1)
    _run_identification_zscan(cfg, out2)

    m1 = (out1 / "manifest.csv").read_text(encoding="utf-8")
    m2 = (out2 / "manifest.csv").read_text(encoding="utf-8")
    assert m1 == m2


def test_run_identification_dispatches_to_zscan(tmp_path, monkeypatch):
    """run_identification dispatches mode='z-scan' to _run_identification_zscan."""
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo import pipeline

    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: np.ones((170, 510)))
    monkeypatch.setattr(pipeline, "_lookup_and_load_kernel", lambda *args, **kwargs: None)

    cfg = _tiny_zscan_config()
    result = run_identification(cfg, tmp_path / "out")
    assert "n_configurations" in result


def test_cli_main_identify_zscan_mode(tmp_path, monkeypatch):
    """The CLI accepts --mode z-scan and produces a manifest."""
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo import pipeline

    monkeypatch.setattr(fm, "Hg", np.zeros((100, 3, 3)))
    monkeypatch.setattr(fm, "q_hkl", np.array([-1, 1, -1]) / np.sqrt(3))
    monkeypatch.setattr(fm, "forward", lambda *args, **kwargs: np.ones((170, 510)))
    monkeypatch.setattr(pipeline, "_lookup_and_load_kernel", lambda *args, **kwargs: None)

    toml_text = """
mode = "z-scan"

[crystal]
slip_plane_normal = [1, 1, 1]
angle_start_deg = 0.0
angle_stop_deg = 0.0
angle_step_deg = 10.0
b_vector_indices = [0]
sweep_all_slip_planes = false
exclude_invisibility = false

[scan.phi]
value = 1.5e-4

[noise]
poisson_noise = false
rng_seed = 0
intensity_scale = 1.0

[zscan]
z_offsets_um = [0.0]
phi_range_deg = 0.03
phi_steps = 2
chi_range_deg = 0.1
chi_steps = 2
include_secondary = false

[io]
fn_prefix = "/mosa_test_0000_"
ftype = ".npy"
dislocs_dirname = "identify_zscan"
perfect_dirname = "ignored"
include_perfect_crystal = false

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0
"""
    cfg_path = tmp_path / "zscan.toml"
    cfg_path.write_text(toml_text, encoding="utf-8")
    out_dir = tmp_path / "out"

    exit_code = cli_main_identify(["--config", str(cfg_path), "--output", str(out_dir)])
    assert exit_code == 0
    assert (out_dir / "manifest.csv").is_file()


def test_example_zscan_config_loads():
    """configs/identification_zscan.toml parses and validates."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_identification_config(repo_root / "configs" / "identification_zscan.toml")
    assert cfg.mode == "z-scan"
    assert cfg.zscan is not None
    assert len(cfg.zscan.z_offsets_um) >= 1


class TestIdentificationConfigScanReusesSharedShape:
    def test_phi_value_from_scan_phi_value(self, tmp_path: Path) -> None:
        toml_text = (
            'mode = "single"\n'
            "\n"
            "[reciprocal]\n"
            "hkl = [-1, 1, -1]\n"
            "keV = 17.0\n"
            "\n"
            "[crystal]\n"
            "slip_plane_normal = [1, 1, 1]\n"
            "sweep_all_slip_planes = true\n"
            "exclude_invisibility = true\n"
            "\n"
            "[scan.phi]\n"
            "value = 0.00015\n"
            "\n"
            "[noise]\n"
            "poisson_noise = true\n"
            "rng_seed = 0\n"
            "intensity_scale = 7.0\n"
            "\n"
            "[io]\n"
            'fn_prefix = "/mosa_test_0000_"\n'
            'ftype = ".npy"\n'
            'dislocs_dirname = "identify"\n'
            'perfect_dirname = "ignored"\n'
            "include_perfect_crystal = false\n"
        )
        cfg_path = tmp_path / "id.toml"
        cfg_path.write_text(toml_text)
        cfg = load_identification_config(cfg_path)
        # New shape: scan is the shared ScanConfig
        assert isinstance(cfg.scan, ScanConfig)
        assert cfg.scan.phi.value == 1.5e-4
        # Noise lives in its own block
        assert cfg.noise.poisson_noise is True
        assert cfg.noise.intensity_scale == 7.0


class TestIdentificationConfigZScanForbidsScanZ:
    def test_zscan_mode_with_scan_z_rejected(self, tmp_path: Path) -> None:
        # When mode='z-scan', the [zscan].z_offsets_um drives z; [scan.z] is forbidden.
        toml_text = (
            'mode = "z-scan"\n'
            "\n"
            "[reciprocal]\n"
            "hkl = [-1, 1, -1]\n"
            "keV = 17.0\n"
            "\n"
            "[crystal]\n"
            "slip_plane_normal = [1, 1, 1]\n"
            "\n"
            "[scan.phi]\n"
            "value = 0.00015\n"
            "[scan.z]\n"
            "range = 1e-6\n"
            "steps = 3\n"
            "\n"
            "[zscan]\n"
            "z_offsets_um = [-1.0, 0.0, 1.0]\n"
            "phi_range_deg = 0.03\n"
            "phi_steps = 5\n"
            "chi_range_deg = 0.1\n"
            "chi_steps = 5\n"
            "\n"
            "[io]\n"
            "include_perfect_crystal = false\n"
        )
        cfg_path = tmp_path / "id.toml"
        cfg_path.write_text(toml_text)
        with pytest.raises(ValueError, match=r"mode='z-scan'.*\[scan.z\] is forbidden"):
            load_identification_config(cfg_path)
