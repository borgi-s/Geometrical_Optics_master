"""Migration: convert legacy .npy output dirs to dfxm_geo.h5.

CLI: `dfxm-migrate-output <input_dir> --output <out.h5> [--config <toml>]`

This module holds the only remaining reader for legacy .npy stacks
(`_load_images_legacy`). It is NOT exported from the package public API.
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
from dfxm_geo.io.hdf5 import _scan_title, _write_provenance, write_h5_scan


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
    """Read legacy .npy stacks under `npy_dir` and write a v1.1 HDF5."""
    # Reproduce Hg from the same inputs so the new file's /1.1/dfxm_geo/Hg
    # matches what the original sim used. q_hkl mirrors the kernel.
    S = SAMPLE_REMOUNT_OPTIONS[sample_remount]
    Hg, q_hkl = _fm.Find_Hg(dis, ndis, _fm.psize, _fm.zl_rms, S=S, remount_name=sample_remount)

    dislocs = _load_images_legacy(
        str(npy_dir / dislocs_dirname),
        u_steps=phi_steps,
        v_steps=chi_steps,
    )[0]  # [0] = the flat stack

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
        f'[crystal]\ndis = {dis}\nndis = {ndis}\nsample_remount = "{sample_remount}"\n'
        f"[scan]\nphi_range = {phi_range_deg}\nphi_steps = {phi_steps}\n"
        f"chi_range = {chi_range_deg}\nchi_steps = {chi_steps}\n"
    )

    title = _scan_title(phi_range_deg, phi_steps, chi_range_deg, chi_steps)
    phi_pf = _phi_per_frame(phi_steps, chi_steps, phi_range_deg)
    chi_pf = _chi_per_frame(phi_steps, chi_steps, chi_range_deg)

    write_h5_scan(
        h5_path,
        scan_id="1.1",
        images=dislocs,
        phi=phi_pf,
        chi=chi_pf,
        title=title,
        sample_name="simulated, dislocations",
        sample_dis=dis,
        sample_ndis=ndis,
        sample_remount=sample_remount,
        Hg=Hg,
        q_hkl=q_hkl,
        theta=float(_fm.theta),
        psize=float(_fm.psize),
        zl_rms=float(_fm.zl_rms),
    )
    if has_perfect and perfect is not None:
        write_h5_scan(
            h5_path,
            scan_id="2.1",
            images=perfect,
            phi=phi_pf,
            chi=chi_pf,
            title=title,
            sample_name="simulated, perfect crystal",
            sample_dis=dis,
            sample_ndis=ndis,
            sample_remount=sample_remount,
            Hg=np.zeros_like(Hg),
            q_hkl=q_hkl,
            theta=float(_fm.theta),
            psize=float(_fm.psize),
            zl_rms=float(_fm.zl_rms),
        )

    kernel_npz = _fm._loaded_kernel_path
    if kernel_npz is None:
        raise RuntimeError(
            "no kernel loaded — migration requires a loaded kernel for provenance recording."
        )
    with h5py.File(h5_path, "a") as f:
        _write_provenance(
            f,
            cli="dfxm-migrate-output (legacy import)",
            kernel_npz=kernel_npz,
            config_toml=config_toml,
        )


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


if __name__ == "__main__":
    sys.exit(cli_main())
