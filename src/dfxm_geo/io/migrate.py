"""Migration: convert legacy .npy output dirs to dfxm_geo.h5.

CLI entry points:

- ``dfxm-migrate-output``: legacy ``.npy`` directory -> v1.2.0 master+per-scan.
- ``dfxm-migrate-h5``: v1.1.0 single-file ``dfxm_geo.h5`` -> v1.2.0 master+per-scan.

This module holds the only remaining reader for legacy .npy stacks
(``_load_images_legacy``). It is NOT exported from the package public API.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TypedDict

import h5py
import numpy as np

import dfxm_geo.direct_space.forward_model as _fm
from dfxm_geo.crystal.remount import SAMPLE_REMOUNT_OPTIONS


def _load_images_legacy(
    fpath: str,
    u_steps: int,
    v_steps: int,
    file_ext: str = ".npy",
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Verbatim of the old `dfxm_geo.io.images.load_images` (deleted in v1.1).

    Used only by the migration script. Not part of the public API.
    """
    if not os.path.isdir(fpath):
        raise ValueError(f"Directory does not exist: {fpath}")
    file_list = [f for f in os.listdir(fpath) if f.endswith(file_ext)]
    if not file_list:
        raise ValueError(f"Empty directory: {fpath}")
    file_list.sort()
    stack = np.empty(
        (len(file_list), *np.load(os.path.join(fpath, file_list[0])).shape),
        dtype=np.float64,
    )
    for i, fname in enumerate(file_list):
        stack[i] = np.load(os.path.join(fpath, fname))
    dim_1, dim_2 = stack.shape[1], stack.shape[2]
    stack_reshape = stack.reshape((u_steps, v_steps, dim_1, dim_2))
    return stack, stack_reshape, dim_1, dim_2


def _phi_per_frame(phi_steps: int, chi_steps: int, phi_range_deg: float) -> np.ndarray:
    Phi = np.linspace(-np.deg2rad(phi_range_deg), np.deg2rad(phi_range_deg), phi_steps)
    out = np.empty(phi_steps * chi_steps, dtype=np.float64)
    for chi_idx in range(chi_steps):
        for phi_idx in range(phi_steps):
            out[chi_idx * phi_steps + phi_idx] = Phi[phi_idx]
    return out


def _chi_per_frame(phi_steps: int, chi_steps: int, chi_range_deg: float) -> np.ndarray:
    Chi = np.linspace(-np.deg2rad(chi_range_deg), np.deg2rad(chi_range_deg), chi_steps)
    out = np.empty(phi_steps * chi_steps, dtype=np.float64)
    for chi_idx in range(chi_steps):
        for phi_idx in range(phi_steps):
            out[chi_idx * phi_steps + phi_idx] = Chi[chi_idx]
    return out


def migrate_npy_dir_to_h5(
    npy_dir: Path,
    h5_path: Path,
    *,
    phi_steps: int,
    chi_steps: int,
    phi_range_deg: float,
    chi_range_deg: float,
    dis: float,
    ndis: int,
    sample_remount: str,
    dislocs_dirname: str = "images10",
    perfect_dirname: str = "images10_perf_crystal",
) -> None:
    """Read legacy .npy stacks under `npy_dir` and write v1.2.0 master+per-scan."""
    from dfxm_geo.io.hdf5 import (
        DETECTOR_FILE_FMT,
        DETECTOR_INTERNAL_PATH,
        SCAN_DIR_FMT,
        MasterWriter,
        _scan_title,
        _write_detector_file,
    )

    S = SAMPLE_REMOUNT_OPTIONS[sample_remount]
    Hg, q_hkl = _fm.Find_Hg(dis, ndis, _fm.psize, _fm.zl_rms, S=S, remount_name=sample_remount)

    dislocs = _load_images_legacy(
        str(npy_dir / dislocs_dirname),
        u_steps=phi_steps,
        v_steps=chi_steps,
    )[0]

    perfect_path = npy_dir / perfect_dirname
    has_perfect = perfect_path.is_dir()
    perfect = (
        _load_images_legacy(
            str(perfect_path),
            u_steps=phi_steps,
            v_steps=chi_steps,
        )[0]
        if has_perfect
        else None
    )

    config_toml = (
        f'[crystal]\nmode = "wall"\n[crystal.wall]\ndis = {dis}\n'
        f'ndis = {ndis}\nsample_remount = "{sample_remount}"\n\n'
        f"[scan.phi]\nrange = {phi_range_deg}\nsteps = {phi_steps}\n\n"
        f"[scan.chi]\nrange = {chi_range_deg}\nsteps = {chi_steps}\n"
    )
    title = _scan_title(phi_range_deg, phi_steps, chi_range_deg, chi_steps)
    phi_pf = _phi_per_frame(phi_steps, chi_steps, phi_range_deg)
    chi_pf = _chi_per_frame(phi_steps, chi_steps, chi_range_deg)
    out_dir = h5_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    kernel_npz = _fm._loaded_kernel_path
    if kernel_npz is None:
        raise RuntimeError("no kernel loaded — migration requires a loaded kernel for provenance.")

    with MasterWriter(
        h5_path,
        cli="dfxm-migrate-output (legacy import)",
        config_toml=config_toml,
        kernel_npz=kernel_npz,
    ) as master:
        # /1.1
        scan1_dir_rel = Path(SCAN_DIR_FMT.format(1))
        det1_file = out_dir / scan1_dir_rel / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector")
        _write_detector_file(det1_file, dislocs)
        master.add_scan(
            scan_id="1.1",
            title=title,
            start_time="legacy",
            end_time="legacy",
            sample={
                "name": "simulated, dislocations",
                "dis": float(dis),
                "ndis": int(ndis),
                "sample_remount": sample_remount,
            },
            positioners={"phi": phi_pf, "chi": chi_pf},
            detector_links={
                "dfxm_sim_detector": (
                    scan1_dir_rel / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector"),
                    DETECTOR_INTERNAL_PATH,
                )
            },
            dfxm_geo={
                "Hg": Hg,
                "q_hkl": q_hkl,
                "theta": float(_fm.theta),
                "psize": float(_fm.psize),
                "zl_rms": float(_fm.zl_rms),
            },
            attrs={},
        )
        if has_perfect and perfect is not None:
            scan2_dir_rel = Path(SCAN_DIR_FMT.format(2))
            det2_file = out_dir / scan2_dir_rel / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector")
            _write_detector_file(det2_file, perfect)
            master.add_scan(
                scan_id="2.1",
                title=title,
                start_time="legacy",
                end_time="legacy",
                sample={
                    "name": "simulated, perfect crystal",
                    "dis": float(dis),
                    "ndis": int(ndis),
                    "sample_remount": sample_remount,
                },
                positioners={"phi": phi_pf, "chi": chi_pf},
                detector_links={
                    "dfxm_sim_detector": (
                        scan2_dir_rel / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector"),
                        DETECTOR_INTERNAL_PATH,
                    )
                },
                dfxm_geo={
                    "Hg": np.zeros_like(Hg),
                    "q_hkl": q_hkl,
                    "theta": float(_fm.theta),
                    "psize": float(_fm.psize),
                    "zl_rms": float(_fm.zl_rms),
                },
                attrs={},
            )


def migrate_h5_master_to_master(src: Path, dst_dir: Path) -> None:
    """Convert a v1.1.0 single-file dfxm_geo.h5 to the v1.2.0 master+per-scan layout.

    Pixel data is moved to LIMA-style per-scan detector files under
    dst_dir/scan{N:04d}/dfxm_sim_detector_0000.h5; the new master at
    dst_dir/dfxm_geo.h5 ExternalLinks to them. All non-pixel-data nodes
    (/dfxm_geo/, /N.1/sample/, /N.1/instrument/positioners/, /N.1/dfxm_geo/,
    title/start_time/end_time) are copied losslessly.
    """
    from dfxm_geo.io.hdf5 import (
        DETECTOR_FILE_FMT,
        DETECTOR_INTERNAL_PATH,
        SCAN_DIR_FMT,
        _write_detector_file,
    )

    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    new_master = dst_dir / "dfxm_geo.h5"

    with h5py.File(src, "r") as fin, h5py.File(new_master, "w") as fout:
        # /dfxm_geo/ provenance — copy verbatim
        if "dfxm_geo" in fin:
            fin.copy("dfxm_geo", fout)

        # All /N.1 scan entries
        scan_ids = sorted(k for k in fin if k != "dfxm_geo" and "." in k)
        for idx, scan_id in enumerate(scan_ids, start=1):
            src_scan = fin[scan_id]
            new_scan = fout.create_group(scan_id)
            new_scan.attrs.update({k: v for k, v in src_scan.attrs.items()})
            # Copy title / start_time / end_time
            for name in ("title", "start_time", "end_time"):
                if name in src_scan:
                    fin.copy(f"{scan_id}/{name}", new_scan)
            # Copy sample/, dfxm_geo/ (and analysis/) subtrees
            for sub in ("sample", "dfxm_geo"):
                if sub in src_scan:
                    fin.copy(f"{scan_id}/{sub}", new_scan)
            # Copy positioners
            if "instrument/positioners" in src_scan:
                new_instr = new_scan.create_group("instrument")
                new_instr.attrs["NX_class"] = "NXinstrument"
                fin.copy(f"{scan_id}/instrument/positioners", new_instr)
            else:
                new_instr = new_scan.require_group("instrument")
                new_instr.attrs["NX_class"] = "NXinstrument"
            # Extract pixel data into a new per-scan detector file
            pix_path = f"{scan_id}/instrument/dfxm_sim_detector/data"
            if pix_path in fin:
                stack = fin[pix_path][...]
                scan_dir_rel = Path(SCAN_DIR_FMT.format(idx))
                det_file = (
                    dst_dir / scan_dir_rel / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector")
                )
                _write_detector_file(det_file, stack)
                # ExternalLink in new master
                new_det = new_instr.create_group("dfxm_sim_detector")
                new_det.attrs["NX_class"] = "NXdetector"
                new_det["data"] = h5py.ExternalLink(
                    str(scan_dir_rel / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector")).replace(
                        "\\", "/"
                    ),
                    DETECTOR_INTERNAL_PATH,
                )
            # measurement softlinks
            meas = new_scan.create_group("measurement")
            meas.attrs["NX_class"] = "NXcollection"
            for det_name in new_instr:
                if det_name == "positioners":
                    continue
                meas[det_name] = h5py.SoftLink(f"/{scan_id}/instrument/{det_name}/data")
            if "instrument/positioners" in src_scan:
                for axis in src_scan["instrument/positioners"]:
                    meas[axis] = h5py.SoftLink(f"/{scan_id}/instrument/positioners/{axis}")


def cli_main_h5_to_h5(argv: list[str] | None = None) -> int:
    """Entry point for `dfxm-migrate-h5`."""
    p = argparse.ArgumentParser(
        description="Convert v1.1.0 single-file dfxm_geo.h5 to v1.2.0 master+per-scan layout."
    )
    p.add_argument("input_h5", type=Path, help="v1.1.0 dfxm_geo.h5")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <input_h5>.v120/)",
    )
    args = p.parse_args(argv)
    dst = args.output or args.input_h5.with_suffix(args.input_h5.suffix + ".v120")
    migrate_h5_master_to_master(args.input_h5, dst)
    print(f"Wrote {dst}/")
    return 0


class _MigrateParams(TypedDict):
    phi_steps: int
    chi_steps: int
    phi_range_deg: float
    chi_range_deg: float
    dis: float
    ndis: int
    sample_remount: str


# IUCrJ-2024 defaults: 61x61 with the canonical Borgi 2024 ranges. Used when
# the user runs `dfxm-migrate-output <old_dir>` with no --config.
_IUCRJ_2024_DEFAULTS: _MigrateParams = {
    "phi_steps": 61,
    "chi_steps": 61,
    "phi_range_deg": 0.0006 * 180 / np.pi,
    "chi_range_deg": 0.002 * 180 / np.pi,
    "dis": 4.0,
    "ndis": 151,
    "sample_remount": "S1",
}


def cli_main(argv: list[str] | None = None) -> int:
    """Entry point for `dfxm-migrate-output`."""
    p = argparse.ArgumentParser(
        description="Migrate a legacy .npy output directory to dfxm_geo.h5."
    )
    p.add_argument(
        "input_dir", type=Path, help="Old output dir (contains images10/, images10_perf_crystal/)."
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .h5 path (default: <input_dir>/dfxm_geo.h5).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML config used to generate the original sim. If omitted, uses IUCrJ-2024 defaults.",
    )
    args = p.parse_args(argv)
    out = args.output or (args.input_dir / "dfxm_geo.h5")

    params: _MigrateParams
    if args.config is None:
        params = _IUCRJ_2024_DEFAULTS
        print(f"No --config given; using IUCrJ-2024 defaults: {params}", file=sys.stderr)
    else:
        import tomllib

        with args.config.open("rb") as f:
            raw = tomllib.load(f)
        params = {
            "phi_steps": int(raw["scan"]["phi_steps"]),
            "chi_steps": int(raw["scan"]["chi_steps"]),
            "phi_range_deg": float(raw["scan"]["phi_range"]),
            "chi_range_deg": float(raw["scan"]["chi_range"]),
            "dis": float(raw["crystal"]["dis"]),
            "ndis": int(raw["crystal"]["ndis"]),
            "sample_remount": str(raw["crystal"]["sample_remount"]),
        }

    migrate_npy_dir_to_h5(npy_dir=args.input_dir, h5_path=out, **params)
    print(f"Wrote {out}")
    return 0


def cli_main_npy_to_h5(argv: list[str] | None = None) -> int:
    """Renamed alias for cli_main; matches pyproject entry-point name."""
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(cli_main())
