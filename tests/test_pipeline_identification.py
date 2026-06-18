"""Unit tests for dfxm-identify pipeline shape."""

from pathlib import Path

import numpy as np
import pytest

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import (
    AxisScanConfig,
    DetectorConfig,
    IdentificationConfig,
    IdentificationCrystalConfig,
    IdentificationMonteCarloConfig,
    IOConfig,
    ReciprocalConfig,
    ScanConfig,
    _run_identification_multi,
    _run_identification_single,
    cli_main_identify,
    load_identification_config,
    run_identification,
)


def _stub_resolution() -> fm.ResolutionContext:
    """A no-IO ResolutionContext with an analytic_eval sentinel so the writer's
    "backend loaded" guard passes and kernel_npz stays None. #16 Slice 5: the
    render itself is stubbed (forward_from_static), so the sentinel is never
    actually evaluated."""
    return fm.ResolutionContext(
        Resq_i=None,
        qi1_start=0.0,
        qi1_step=0.0,
        qi2_start=0.0,
        qi2_step=0.0,
        qi3_start=0.0,
        qi3_step=0.0,
        npoints1=None,
        npoints2=None,
        npoints3=None,
        analytic_eval=object(),
        loaded_kernel_path=None,
    )


def _ident_ctx(cfg: IdentificationConfig) -> fm.ForwardContext:
    """Build the run's ForwardContext directly (replaces the deleted
    fm._context_from_globals() shim); resolution is a no-IO sentinel."""
    from dfxm_geo.pipeline import run_theta

    return fm.build_forward_context(run_theta(cfg), _stub_resolution(), cfg.reciprocal.hkl)


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
    """DetectorConfig holds the detector model + calibration fields."""
    detector = DetectorConfig()
    assert detector.model == "pco_edge_4.2_id03"
    assert detector.rng_seed == 0
    assert detector.counts_scale == 1.0e4 / 15


def test_identification_montecarlo_config_defaults():
    cfg = IdentificationMonteCarloConfig()
    assert cfg.n_samples == 1000
    assert cfg.pos_std_um == 5.0
    assert cfg.render_per_dislocation is False


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


def _make_scan_noise(rng_seed: int = 0):
    """Return a (ScanConfig, DetectorConfig) pair replacing old IdentificationScanConfig."""
    scan = ScanConfig(phi=AxisScanConfig(value=150e-6))
    detector = DetectorConfig(rng_seed=rng_seed)
    return scan, detector


def test_identification_config_mode_single_ok():
    scan, noise = _make_scan_noise()
    cfg = IdentificationConfig(
        mode="single",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=scan,
        detector=noise,
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
        detector=noise,
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
            detector=noise,
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
            detector=noise,
            io=_make_io_config(),
        )


def test_identification_config_invalid_mode_raises():
    scan, noise = _make_scan_noise()
    with pytest.raises(ValueError, match="mode must be"):
        IdentificationConfig(
            mode="bogus",  # type: ignore[arg-type]
            crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
            scan=scan,
            detector=noise,
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

[detector]
rng_seed = 0

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

[detector]
rng_seed = 42

[multi]
n_samples = 100
pos_std_um = 3.0
render_per_dislocation = false

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
    assert cfg.detector.rng_seed == 42


def test_load_identification_config_missing_mode_defaults_to_single(tmp_path):
    """Sub-project F: a TOML with no top-level `mode` field defaults to 'single'."""
    cfg_path = tmp_path / "no_mode.toml"
    cfg_path.write_text("[crystal]\nslip_plane_normal = [1, 1, 1]\n", encoding="utf-8")
    cfg = load_identification_config(cfg_path)
    assert cfg.mode == "single"


def _tiny_single_config(tmp_path):
    scan = ScanConfig(phi=AxisScanConfig(value=150e-6))
    noise = DetectorConfig(model="ideal", rng_seed=0)
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
        detector=noise,
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
        io=_make_io_config(),
    )


def test_run_identification_single_writes_expected_count(tmp_path, monkeypatch):
    """Tiny sweep (1 slip plane × 2 b × 2 angles = 4 configs) writes a master
    HDF5 with 4 /N.1 scans and 4 per-scan detector files (v1.2.0 layout).
    """
    import h5py

    expected_image = np.ones((170, 510))
    monkeypatch.setattr(fm, "forward_from_static", lambda *args, **kwargs: expected_image)

    output_dir = tmp_path / "out"
    cfg = _tiny_single_config(tmp_path)

    result = _run_identification_single(cfg, output_dir, _ident_ctx(cfg))

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
    noise = DetectorConfig(model="ideal", rng_seed=0)
    return IdentificationConfig(
        mode="multi",
        crystal=IdentificationCrystalConfig(slip_plane_normal=(1, 1, 1)),
        scan=scan,
        detector=noise,
        multi=IdentificationMonteCarloConfig(n_samples=3, pos_std_um=2.0),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
        io=_make_io_config(),
    )


def _collect_dislocations(master_path: Path, scan_id: str) -> list[tuple]:
    """Pull (plane, burgers, rotation_deg, position_um) tuples from a scan."""
    import h5py

    out: list[tuple] = []
    with h5py.File(master_path, "r") as f:
        disl = f[f"/{scan_id}/sample/dislocations"]
        for idx in sorted(disl):
            d = disl[idx]
            out.append(
                (
                    tuple(d["slip_plane_normal"][...].tolist()),
                    tuple(d["burgers"][...].tolist()),
                    float(d["rotation_deg"][()]),
                    tuple(d["position_um"][...].tolist()),
                )
            )
    return out


def test_run_identification_multi_writes_master_plus_scan_dirs(tmp_path, monkeypatch):
    """n_samples=3 writes master + 3 scan dirs (v1.2.0 layout), no legacy sidecars."""
    import h5py
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm

    expected_image = np.ones((170, 510))
    monkeypatch.setattr(fm, "forward_from_static", lambda *args, **kwargs: expected_image)

    output_dir = tmp_path / "out"
    cfg = _tiny_multi_config()

    result = _run_identification_multi(cfg, output_dir, _ident_ctx(cfg))

    master = output_dir / "dfxm_identify.h5"
    assert master.is_file()
    for k in range(1, 4):
        det = output_dir / f"scan{k:04d}" / "dfxm_sim_detector_0000.h5"
        assert det.is_file(), f"missing {det}"
    # No legacy sidecars
    assert not (output_dir / "manifest.csv").exists()
    assert not (output_dir / "im_data").exists()
    assert not (output_dir / "images").exists()
    with h5py.File(master, "r") as f:
        scan_ids = sorted(k for k in f if k != "dfxm_geo")
        assert scan_ids == ["1.1", "2.1", "3.1"]
    assert result["n_samples"] == 3


def test_run_identification_multi_is_deterministic_for_seed(tmp_path, monkeypatch):
    """Two runs at the same seed produce identical dislocation draws."""
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm

    monkeypatch.setattr(fm, "forward_from_static", lambda *args, **kwargs: np.ones((170, 510)))

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    cfg = _tiny_multi_config()

    _run_identification_multi(cfg, out1, _ident_ctx(cfg))
    _run_identification_multi(cfg, out2, _ident_ctx(cfg))

    for scan_id in ("1.1", "2.1", "3.1"):
        d1 = _collect_dislocations(out1 / "dfxm_identify.h5", scan_id)
        d2 = _collect_dislocations(out2 / "dfxm_identify.h5", scan_id)
        assert d1 == d2, f"scan {scan_id}: dislocation draws differ"


def test_run_identification_dispatches_to_single(tmp_path, monkeypatch):
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo import orchestrator

    monkeypatch.setattr(fm, "forward_from_static", lambda *args, **kwargs: np.ones((170, 510)))
    monkeypatch.setattr(
        orchestrator, "_load_resolution", lambda *args, **kwargs: _stub_resolution()
    )

    cfg = _tiny_single_config(tmp_path)
    result = run_identification(cfg, tmp_path / "out")
    assert "n_images" in result


def test_run_identification_dispatches_to_multi(tmp_path, monkeypatch):
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo import orchestrator

    monkeypatch.setattr(fm, "forward_from_static", lambda *args, **kwargs: np.ones((170, 510)))
    monkeypatch.setattr(
        orchestrator, "_load_resolution", lambda *args, **kwargs: _stub_resolution()
    )

    cfg = _tiny_multi_config()
    result = run_identification(cfg, tmp_path / "out")
    assert "n_samples" in result


def test_cli_main_identify_parses_args(tmp_path, monkeypatch):
    """Smoke test: CLI parses --config and --output, calls run_identification, returns 0."""
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo import orchestrator

    monkeypatch.setattr(fm, "forward_from_static", lambda *args, **kwargs: np.ones((170, 510)))
    monkeypatch.setattr(
        orchestrator, "_load_resolution", lambda *args, **kwargs: _stub_resolution()
    )

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

[detector]
model = "ideal"
rng_seed = 0

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
    from dfxm_geo.data import configs_root

    cfg = load_identification_config(configs_root() / "identification_single.toml")
    assert cfg.mode == "single"
    assert cfg.crystal.sweep_all_slip_planes is True
    assert cfg.crystal.exclude_invisibility is True


def test_example_multi_config_loads():
    """configs/identification_multi.toml parses and validates."""
    from dfxm_geo.data import configs_root

    cfg = load_identification_config(configs_root() / "identification_multi.toml")
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

[detector]
model = "ideal"
rng_seed = 0

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

    cfg = IdentificationZScanConfig(z_offsets_um=[0.0])
    assert cfg.z_offsets_um == [0.0]
    assert cfg.include_secondary is True
    assert cfg.secondary_rng_offset == 0


def test_identification_zscan_config_is_frozen():
    from dataclasses import FrozenInstanceError

    from dfxm_geo.pipeline import IdentificationZScanConfig

    cfg = IdentificationZScanConfig(z_offsets_um=[0.0])
    with pytest.raises(FrozenInstanceError):
        cfg.secondary_rng_offset = 2  # type: ignore[misc]


def _tiny_zscan_config(slip_plane=(1, 1, 1)):
    """Tiny z-scan config: 1 layer x 1 plane x 1 b x 1 alpha = 1 scan,
    with a phi x chi rocking grid coming from the shared [scan.<axis>]
    schema (B+C). include_secondary=False to skip the random draw.
    """
    from dfxm_geo.pipeline import IdentificationZScanConfig

    scan = ScanConfig(
        phi=AxisScanConfig(range=0.034377, steps=2),
        chi=AxisScanConfig(range=0.114, steps=2),
    )
    noise = DetectorConfig(model="ideal", rng_seed=0)
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
        detector=noise,
        zscan=IdentificationZScanConfig(
            z_offsets_um=[0.0],
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
            detector=noise,
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
            detector=noise,
            zscan=IdentificationZScanConfig(z_offsets_um=[0.0]),
            io=_make_io_config(),
        )


def test_load_identification_config_zscan(tmp_path):
    """A mode='z-scan' TOML round-trips under the post-Phase-5 schema:
    [zscan] carries only z_offsets_um / include_secondary / secondary_rng_offset,
    and the rocking grid lives in [scan.phi] / [scan.chi].
    """
    toml_text = """
mode = "z-scan"

[crystal]
slip_plane_normal = [1, 1, 1]

[scan.phi]
range = 0.03
steps = 21
[scan.chi]
range = 0.1
steps = 21

[detector]
rng_seed = 7

[zscan]
z_offsets_um = [-1.0, 0.0, 1.0]
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
    assert cfg.zscan.include_secondary is True
    assert cfg.zscan.secondary_rng_offset == 2
    assert cfg.scan.phi.steps == 21
    assert cfg.scan.chi.steps == 21
    assert cfg.detector.rng_seed == 7


def test_run_identification_zscan_writes_per_config_rocking_grid(tmp_path, monkeypatch):
    """Tiny z-scan: 1 layer x 1 plane x 1 b x 1 alpha = 1 configuration -> 1 scan
    dir under the v1.2.0 master+per-scan HDF5 layout, with phi_steps*chi_steps frames.
    """
    import h5py
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo.pipeline import _run_identification_zscan

    monkeypatch.setattr(fm, "forward_from_static", lambda *args, **kwargs: np.ones((170, 510)))

    output_dir = tmp_path / "out"
    cfg = _tiny_zscan_config()
    result = _run_identification_zscan(cfg, output_dir, _ident_ctx(cfg))

    master = output_dir / "dfxm_identify.h5"
    assert master.is_file()
    det = output_dir / "scan0001" / "dfxm_sim_detector_0000.h5"
    assert det.is_file()
    # No legacy sidecars / per-config dirs
    assert not (output_dir / "manifest.csv").exists()
    assert not (output_dir / "layer_0000").exists()
    with h5py.File(master, "r") as f:
        scan_ids = sorted(k for k in f if k != "dfxm_geo")
        assert scan_ids == ["1.1"]
        scan = f["/1.1"]
        assert scan.attrs["identify_mode"] == "z-scan"
        # 2 phi * 2 chi = 4 frames
        assert f["/1.1/instrument/dfxm_sim_detector/data"].shape[0] == 4
        samp = scan["sample"]
        assert "z_offset_um" in samp
        assert "primary" in samp
        assert "secondary" not in samp  # include_secondary=False
    assert result["n_configurations"] == 1


def _collect_zscan_secondaries(master_path: Path) -> list[tuple]:
    """Pull each scan's secondary-dislocation draw from the master HDF5."""
    import h5py

    out: list[tuple] = []
    with h5py.File(master_path, "r") as f:
        scan_ids = sorted((k for k in f if k != "dfxm_geo"), key=lambda s: int(s.split(".")[0]))
        for sid in scan_ids:
            samp = f[f"/{sid}/sample"]
            sec = samp["secondary"]
            out.append(
                (
                    tuple(sec["slip_plane_normal"][...].tolist()),
                    tuple(sec["burgers"][...].tolist()),
                    float(sec["rotation_deg"][()]),
                    tuple(sec["position_um"][...].tolist()),
                )
            )
    return out


def test_run_identification_zscan_is_deterministic_for_seed(tmp_path, monkeypatch):
    """Two runs at the same seed produce identical secondary-dislocation draws
    in every scan (compared via the master HDF5's /N.1/sample/secondary group).
    """
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo.pipeline import IdentificationZScanConfig, _run_identification_zscan

    monkeypatch.setattr(fm, "forward_from_static", lambda *args, **kwargs: np.ones((170, 510)))

    scan = ScanConfig(
        phi=AxisScanConfig(range=0.034377, steps=2),
        chi=AxisScanConfig(range=0.114, steps=2),
    )
    noise = DetectorConfig(model="ideal", rng_seed=0)
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
        detector=noise,
        zscan=IdentificationZScanConfig(
            z_offsets_um=[0.0],
            include_secondary=True,  # exercise the secondary draw
        ),
        reciprocal=ReciprocalConfig(hkl=(-1, 1, -1), keV=17.0),
        io=_make_io_config(),
    )

    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    _run_identification_zscan(cfg, out1, _ident_ctx(cfg))
    _run_identification_zscan(cfg, out2, _ident_ctx(cfg))

    s1 = _collect_zscan_secondaries(out1 / "dfxm_identify.h5")
    s2 = _collect_zscan_secondaries(out2 / "dfxm_identify.h5")
    assert s1 == s2
    assert len(s1) == 4  # 2 b * 2 alpha = 4 scans


def test_run_identification_dispatches_to_zscan(tmp_path, monkeypatch):
    """run_identification dispatches mode='z-scan' to _run_identification_zscan."""
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo import orchestrator

    monkeypatch.setattr(fm, "forward_from_static", lambda *args, **kwargs: np.ones((170, 510)))
    monkeypatch.setattr(
        orchestrator, "_load_resolution", lambda *args, **kwargs: _stub_resolution()
    )

    cfg = _tiny_zscan_config()
    result = run_identification(cfg, tmp_path / "out")
    assert "n_configurations" in result


def test_cli_main_identify_zscan_mode(tmp_path, monkeypatch):
    """The CLI accepts --mode z-scan and produces the v1.2.0 master HDF5."""
    import numpy as np

    import dfxm_geo.direct_space.forward_model as fm
    from dfxm_geo import orchestrator

    monkeypatch.setattr(fm, "forward_from_static", lambda *args, **kwargs: np.ones((170, 510)))
    monkeypatch.setattr(
        orchestrator, "_load_resolution", lambda *args, **kwargs: _stub_resolution()
    )

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
range = 0.034377
steps = 2
[scan.chi]
range = 0.114
steps = 2

[detector]
model = "ideal"
rng_seed = 0

[zscan]
z_offsets_um = [0.0]
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
    assert (out_dir / "dfxm_identify.h5").is_file()


def test_example_zscan_config_loads():
    """configs/identification_zscan.toml parses and validates."""
    from dfxm_geo.data import configs_root

    cfg = load_identification_config(configs_root() / "identification_zscan.toml")
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
            "[detector]\n"
            "rng_seed = 0\n"
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
        # Detector config lives in its own block
        assert cfg.detector.model == "pco_edge_4.2_id03"
        assert cfg.detector.rng_seed == 0


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
            "\n"
            "[io]\n"
            "include_perfect_crystal = false\n"
        )
        cfg_path = tmp_path / "id.toml"
        cfg_path.write_text(toml_text)
        with pytest.raises(ValueError, match=r"mode='z-scan'.*\[scan.z\] is forbidden"):
            load_identification_config(cfg_path)
