# dfxm-geo HDF5 output format (v1.1.0)

`dfxm-forward` writes one HDF5 file per simulation: `<output_dir>/dfxm_geo.h5`. The layout follows the **ESRF BLISS** convention (compatible with [darfix](https://darfix.readthedocs.io/) and [darling](https://github.com/AxelHenningsson/darling)) and adds a sim-specific `/dfxm_geo/` group for provenance.

## High-level structure

```
dfxm_geo.h5
├── /dfxm_geo/                     ← global provenance (one per file)
│   ├── version                    "1.1.0"
│   ├── git_sha                    "..." (40-char) or "unknown"
│   ├── git_dirty                  bool
│   ├── generated_at               "2026-05-17T10:00:00+00:00"
│   ├── hostname                   "borgi-laptop" / "n-62-12-15"
│   ├── python_version, numpy_version
│   ├── cli                        full command line
│   ├── config_toml                entire TOML config as a UTF-8 string
│   └── kernel/
│       ├── pkl_fn                 "Resq_i_2026-05-16_2100.npz"
│       ├── sha256                 64-char hex digest
│       └── qi1_range...Nrays      mirrored from the kernel npz
│
├── /1.1/                          ← BLISS scan: dislocated crystal
│   ├── title                      "fscan2d phi ... chi ... 1.0"
│   ├── start_time, end_time       ISO-8601
│   ├── instrument/
│   │   ├── dfxm_sim_detector/data (N_frames, H, W) float64, chunks (1, H, W), gzip-4 + shuffle
│   │   └── positioners/
│   │       ├── phi                (N_frames,) degrees, attrs["units"] = "degree"
│   │       └── chi                (N_frames,) degrees, attrs["units"] = "degree"
│   ├── measurement/               ← BLISS soft-links
│   │   ├── dfxm_sim_detector → /1.1/instrument/dfxm_sim_detector/data
│   │   ├── phi                  → /1.1/instrument/positioners/phi
│   │   └── chi                  → /1.1/instrument/positioners/chi
│   ├── sample/                    ← NXsample
│   │   ├── name                   "simulated, dislocations"
│   │   ├── dis, ndis, sample_remount
│   └── dfxm_geo/                  ← per-scan sim-specific
│       ├── Hg                     (npixels, 3, 3) float64
│       ├── q_hkl                  (3,)
│       ├── theta, psize, zl_rms
│       └── analysis/              ← present after `dfxm-forward` (postprocess stage)
│           ├── phi_list, chi_list (H, W) COM maps
│           ├── qi_field
│           └── chi_shift_deg      scalar
│
└── /2.1/                          ← optional perfect-crystal scan
    └── (same shape as /1.1/, Hg=0, sample/name="simulated, perfect crystal")
```

## Frame ordering

fscan2d convention: phi inner, chi outer. The k-th frame corresponds to:

```python
phi_idx = k % phi_steps
chi_idx = k // phi_steps
```

(Equivalently, `k = chi_idx * phi_steps + phi_idx`.)

`load_h5_scan` returns the data both as a flat `(N_frames, H, W)` stack and as a `(phi_steps, chi_steps, H, W)` reshape.

## NX_class attributes

The following groups carry `NX_class` attrs so silx, NeXpy, and other NeXus-aware viewers can recognize the structure:

| Path                                  | NX_class       |
| ------------------------------------- | -------------- |
| `/1.1`                                | `NXentry`      |
| `/1.1/instrument`                     | `NXinstrument` |
| `/1.1/instrument/dfxm_sim_detector`   | `NXdetector`   |
| `/1.1/instrument/positioners`         | `NXcollection` |
| `/1.1/sample`                         | `NXsample`     |

We do not claim full NeXus compliance — these attrs are a free interop bonus, not a contract.

## Reading the file

```python
from dfxm_geo.io.hdf5 import load_h5_scan

stack, stack_reshape, h, w = load_h5_scan(
    "output/dfxm_geo.h5", scan_id="1.1",
    # phi_steps / chi_steps inferred from /dfxm_geo/config_toml if omitted
)
```

Or via h5py directly:

```python
import h5py
with h5py.File("output/dfxm_geo.h5", "r") as f:
    images = f["/1.1/instrument/dfxm_sim_detector/data"][...]
    phi_deg = f["/1.1/instrument/positioners/phi"][...]
    chi_deg = f["/1.1/instrument/positioners/chi"][...]
```

## Opening in darfix / darling

Darfix:

```bash
darfix output/dfxm_geo.h5
```

Darling:

```python
import darling
dset = darling.DataSet("output/dfxm_geo.h5", scan_id="1.1")
print(dset.data.shape, dset.motors)
```

## Compression

Image data is chunked `(1, H, W)` with gzip-4 + shuffle. Typical compression ratio is ~3-5× for sim images (which are sparse — most pixels are near zero). Read latency for one full stack on a laptop SSD is ~1-2 seconds.

## Migrating old `.npy` outputs

```bash
dfxm-migrate-output <old_output_dir>
# or:
dfxm-migrate-output <old_output_dir> --config <toml> --output out.h5
```

Without `--config`, defaults to IUCrJ-2024 params (61×61, dis=4, ndis=151, S1) which match the canonical Borgi 2024 paper-reproduction runs.
