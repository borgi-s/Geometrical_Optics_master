"""Cross-machine HDF5 comparison for the v1.1.0 PR #10 validation.

Compares the dfxm_geo.h5 produced by an identical config on two machines:
the laptop and the DTU cluster. Reports schema/provenance agreement plus
several granularities of numerical comparison on the detector data and
postprocess analysis.

Expected outcome at Nsub=1, with non-seeded Find_Hg + independent MC kernels:
- Schema/provenance: match exactly
- Detector data: bit-wise differ; statistically close
- phi/chi positioners: exact match (just the rocking grid)
- com_chi/com_phi (if present): close but not bit-equal

Run from the repo root:
    python scripts/validate_cross_machine.py
or with explicit paths:
    python scripts/validate_cross_machine.py <laptop.h5> <cluster.h5>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np


def find_default_paths() -> tuple[Path, Path]:
    laptops = sorted(Path("output").glob("laptop_hdf5_nsub1_*/dfxm_geo.h5"))
    if not laptops:
        raise FileNotFoundError("no laptop_hdf5_nsub1_*/dfxm_geo.h5 under output/")
    clusters = sorted(Path("output").glob("cluster_*/dfxm_geo.h5"))
    if not clusters:
        raise FileNotFoundError("no cluster_*/dfxm_geo.h5 under output/")
    return laptops[-1], clusters[-1]


def summarize(p: Path) -> dict:
    with h5py.File(p, "r") as f:
        return dict(
            path=str(p),
            version=f["/dfxm_geo/version"][()].decode(),
            git_sha=f["/dfxm_geo/git_sha"][()].decode(),
            git_dirty=bool(f["/dfxm_geo/git_dirty"][()]),
            hostname=f["/dfxm_geo/hostname"][()].decode(),
            kernel_pkl=f["/dfxm_geo/kernel/pkl_fn"][()].decode(),
            kernel_sha=f["/dfxm_geo/kernel/sha256"][()].decode()[:12],
            scans=sorted(k for k in f if k.endswith(".1")),
            data_shape=tuple(f["/1.1/instrument/dfxm_sim_detector/data"].shape),
            data_dtype=str(f["/1.1/instrument/dfxm_sim_detector/data"].dtype),
            title=f["/1.1/title"][()].decode(),
            phi_n=int(f["/1.1/instrument/positioners/phi"].shape[0]),
            chi_n=int(f["/1.1/instrument/positioners/chi"].shape[0]),
            analysis=sorted(f["/1.1/dfxm_geo/analysis"]) if "/1.1/dfxm_geo/analysis" in f else None,
        )


def compare_array(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    """Bit-wise + statistical comparison of two in-memory arrays of the same shape."""
    assert a.shape == b.shape, f"{name}: shape mismatch {a.shape} vs {b.shape}"

    diff = a - b
    abs_diff = np.abs(diff)
    scale = np.maximum(np.abs(a), np.abs(b))
    rel = np.where(scale > 0, abs_diff / scale, 0.0)

    return {
        "shape": a.shape,
        "dtype": str(a.dtype),
        "array_equal": bool(np.array_equal(a, b)),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rms_diff": float(np.sqrt(np.mean(diff**2))),
        "max_rel_diff": float(rel.max()),
        "mean_rel_diff": float(rel.mean()),
        "allclose_rtol_1e-7": bool(np.allclose(a, b, rtol=1e-7, atol=0)),
        "allclose_rtol_1e-5": bool(np.allclose(a, b, rtol=1e-5, atol=0)),
        "allclose_rtol_1e-3": bool(np.allclose(a, b, rtol=1e-3, atol=0)),
        "allclose_rtol_1e-2": bool(np.allclose(a, b, rtol=1e-2, atol=0)),
        "allclose_rtol_1e-1": bool(np.allclose(a, b, rtol=1e-1, atol=0)),
        "value_range_a": (float(a.min()), float(a.max())),
        "value_range_b": (float(b.min()), float(b.max())),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
    }


def compare_detector_streaming(a_dset: h5py.Dataset, b_dset: h5py.Dataset, chunk: int = 32) -> dict:
    """Stream over the leading axis to compare large detector stacks without
    loading both 2.4 GiB arrays at once. Accumulates: max abs diff, sum/sum2
    for mean+rms, max rel diff, min/max/sum for value ranges, and the worst-case
    allclose verdict per chunk (AND-reduced)."""
    assert a_dset.shape == b_dset.shape
    n = a_dset.shape[0]

    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    max_rel = 0.0
    sum_rel = 0.0
    rel_count = 0
    min_a = np.inf
    max_a = -np.inf
    sum_a = 0.0
    min_b = np.inf
    max_b = -np.inf
    sum_b = 0.0
    n_total = a_dset.size

    array_equal = True
    allclose = {tol: True for tol in (1e-7, 1e-5, 1e-3, 1e-2, 1e-1)}

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        a = a_dset[start:stop].astype(np.float64, copy=False)
        b = b_dset[start:stop].astype(np.float64, copy=False)

        if not np.array_equal(a, b):
            array_equal = False

        diff = a - b
        abs_diff = np.abs(diff)
        max_abs = max(max_abs, float(abs_diff.max()))
        sum_abs += float(abs_diff.sum())
        sum_sq += float((diff * diff).sum())

        scale = np.maximum(np.abs(a), np.abs(b))
        nz = scale > 0
        if nz.any():
            rel_chunk = abs_diff[nz] / scale[nz]
            max_rel = max(max_rel, float(rel_chunk.max()))
            sum_rel += float(rel_chunk.sum())
            rel_count += int(nz.sum())

        min_a = min(min_a, float(a.min()))
        max_a = max(max_a, float(a.max()))
        sum_a += float(a.sum())
        min_b = min(min_b, float(b.min()))
        max_b = max(max_b, float(b.max()))
        sum_b += float(b.sum())

        for tol in allclose:
            if allclose[tol] and not np.allclose(a, b, rtol=tol, atol=0):
                allclose[tol] = False

        if start % (chunk * 32) == 0:
            print(f"    chunk {start}/{n}...", flush=True)

    return {
        "shape": a_dset.shape,
        "dtype": str(a_dset.dtype),
        "array_equal": array_equal,
        "max_abs_diff": max_abs,
        "mean_abs_diff": sum_abs / n_total,
        "rms_diff": float(np.sqrt(sum_sq / n_total)),
        "max_rel_diff": max_rel,
        "mean_rel_diff": sum_rel / max(rel_count, 1),
        **{f"allclose_rtol_{tol:g}": allclose[tol] for tol in allclose},
        "value_range_a": (min_a, max_a),
        "value_range_b": (min_b, max_b),
        "mean_a": sum_a / n_total,
        "mean_b": sum_b / n_total,
    }


def main() -> int:
    if len(sys.argv) == 3:
        L_path, C_path = Path(sys.argv[1]), Path(sys.argv[2])
    else:
        L_path, C_path = find_default_paths()

    print(f"LAPTOP : {L_path}")
    print(f"CLUSTER: {C_path}")
    print()

    # --- 1. Schema + provenance --------------------------------------
    l_sum = summarize(L_path)
    c_sum = summarize(C_path)
    print("--- SCHEMA + PROVENANCE ---")
    print(json.dumps({"laptop": l_sum, "cluster": c_sum}, indent=2))

    schema_ok = True
    if l_sum["version"] != c_sum["version"] or l_sum["version"] != "1.1.0":
        print(f"FAIL: version mismatch ({l_sum['version']} vs {c_sum['version']})")
        schema_ok = False
    if l_sum["git_sha"] != c_sum["git_sha"]:
        print(f"FAIL: git_sha mismatch ({l_sum['git_sha']} vs {c_sum['git_sha']})")
        schema_ok = False
    if l_sum["scans"] != c_sum["scans"]:
        print(f"FAIL: scan layout mismatch ({l_sum['scans']} vs {c_sum['scans']})")
        schema_ok = False
    if l_sum["data_shape"] != c_sum["data_shape"]:
        print(f"FAIL: data_shape mismatch ({l_sum['data_shape']} vs {c_sum['data_shape']})")
        schema_ok = False
    if l_sum["analysis"] != c_sum["analysis"]:
        print(f"FAIL: analysis arrays differ ({l_sum['analysis']} vs {c_sum['analysis']})")
        schema_ok = False

    if schema_ok:
        print("\nSCHEMA + PROVENANCE: PASS")
    else:
        print("\nSCHEMA + PROVENANCE: FAIL — stop here, fix schema before comparing data")
        return 1

    # --- 2. Positioner exactness -------------------------------------
    print("\n--- POSITIONERS (phi, chi) - expect EXACT match ---")
    with h5py.File(L_path) as fl, h5py.File(C_path) as fc:
        for motor in ("phi", "chi"):
            a = fl[f"/1.1/instrument/positioners/{motor}"][:]
            b = fc[f"/1.1/instrument/positioners/{motor}"][:]
            ok = np.array_equal(a, b)
            print(
                f"  {motor:5s}: array_equal={ok}  shape={a.shape}  range=[{a.min():.6f}, {a.max():.6f}]"
            )

    # --- 3. Detector data (streaming — 2.4 GiB each, can't fit both in RAM) ----
    print("\n--- DETECTOR DATA (/1.1/instrument/dfxm_sim_detector/data) ---")
    print("  streaming chunks of 32 frames; ~2.4 GiB total per side...")
    with h5py.File(L_path) as fl, h5py.File(C_path) as fc:
        stats = compare_detector_streaming(
            fl["/1.1/instrument/dfxm_sim_detector/data"],
            fc["/1.1/instrument/dfxm_sim_detector/data"],
        )
    print(json.dumps(stats, indent=2))

    # --- 4. Postprocess analysis arrays ------------------------------
    print("\n--- POSTPROCESS ANALYSIS ---")
    with h5py.File(L_path) as fl, h5py.File(C_path) as fc:
        if "/1.1/dfxm_geo/analysis" not in fl:
            print("  no analysis group present; skipping")
        else:
            for key in sorted(fl["/1.1/dfxm_geo/analysis"]):
                la = fl[f"/1.1/dfxm_geo/analysis/{key}"][()]
                lc = fc[f"/1.1/dfxm_geo/analysis/{key}"][()]

                if np.isscalar(la) or (hasattr(la, "shape") and la.shape == ()):
                    la_v = float(la) if np.issubdtype(type(la), np.number) else la
                    lc_v = float(lc) if np.issubdtype(type(lc), np.number) else lc
                    print(
                        f"  {key:20s} (scalar): laptop={la_v}  cluster={lc_v}  equal={la_v == lc_v}"
                    )
                elif hasattr(la, "shape"):
                    s = compare_array(key, la, lc)
                    print(
                        f"  {key:20s} (array {la.shape}): "
                        f"equal={s['array_equal']}  "
                        f"max|d|={s['max_abs_diff']:.3e}  "
                        f"rms={s['rms_diff']:.3e}  "
                        f"allclose@1e-3={s['allclose_rtol_1e-3']}"
                    )

    # --- 5. Summary ---------------------------------------------------
    print("\n--- INTERPRETATION ---")
    if stats["array_equal"]:
        print("  Detector data: BIT-WISE IDENTICAL. Surprising.")
    else:
        print("  Detector data: NOT bit-wise identical (expected).")
        print(f"    max abs diff       : {stats['max_abs_diff']:.3e}")
        print(f"    rms diff           : {stats['rms_diff']:.3e}")
        print(f"    max rel diff       : {stats['max_rel_diff']:.3e}")
        print(f"    mean abs diff      : {stats['mean_abs_diff']:.3e}")
        print(f"    laptop mean        : {stats['mean_a']:.3e}")
        print(f"    cluster mean       : {stats['mean_b']:.3e}")
        for tol_key in sorted(k for k in stats if k.startswith("allclose_")):
            print(f"    {tol_key:28s}: {stats[tol_key]}")
        print()
        print("  The two main sources of disagreement:")
        print("  (1) Reciprocal-space kernel: independent MC seeds on each machine")
        print("      (Resq_i_20260516_2100.npz built separately on each host).")
        print("  (2) Direct-space Fg cache: Find_Hg uses non-seeded")
        print("      np.random.default_rng() so dislocation positions differ.")
        print()
        print("  For PR #10 the gate is: schema+provenance agree (checked above) AND")
        print("  darfix+darling open both files. Numerical bit-equivalence is not")
        print("  required and is not testable without seeding Find_Hg + sharing Fg.")

    return 0 if schema_ok else 1


if __name__ == "__main__":
    sys.exit(main())
