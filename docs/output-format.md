# dfxm-geo HDF5 output format (v1.3.0)

## Overview

Starting in v1.2.0, both `dfxm-forward` and `dfxm-identify` write a **master
HDF5 file** at the top of the output directory, with per-scan **LIMA-style**
detector files in `scan0001/`, `scan0002/`, … subdirectories. The master
holds positioners, sample identity, and provenance; the per-scan files hold
only pixel data. The two are joined by HDF5 `ExternalLink`s, so reading is
transparent — h5py follows the link automatically and the consumer sees one
logical scan tree.

This layout is a faithful mirror of an ESRF BLISS dataset on disk (e.g.
`Al1050_dislocations_wireRocking_..._w1_50ticks.h5` master alongside
`scan0001/pco_ff_0000.h5`, `scan0002/pco_ff_0000.h5`, …). Simulated data is
therefore drop-in interoperable with real beamline data: the same loaders
(`silx`, `darfix`, `darling`) work on both, and processing flows do not need
a "real vs. sim" branch.

In v1.1.0 the same metadata was packed into a single `dfxm_geo.h5` with
pixel data inline. v1.2.0 preserves the metadata schema but splits the
pixel data out into per-scan directories. Use `dfxm-migrate-h5` to convert
v1.1.0 files to the new layout (see [Migration from v1.1.0](#migration-from-v110)).

## High-level structure

```
out_dir/
  dfxm_geo.h5                                    ← master (forward) OR
  dfxm_identify.h5                               ← master (identify)
    /dfxm_geo/                                   ← global provenance (one per master)
      version, git_sha, git_dirty, hostname,
      python_version, numpy_version, generated_at,
      cli, config_toml
      kernel/{pkl_fn, sha256, ...}               ← bundled kernel metadata
    /1.1/                                        ← BLISS scan entry, metadata-only in master
      @scan_mode, @scanned_axes, @crystal_mode (forward) | @identify_mode (identify)
      title, start_time, end_time
      sample/                                    ← per-scan identity (see per-mode section)
      instrument/
        dfxm_sim_detector/
          @NX_class = "NXdetector"
          data → ExternalLink('scan0001/dfxm_sim_detector_0000.h5',
                              '/entry_0000/dfxm_sim_detector/image')
        positioners/
          @NX_class = "NXcollection"
          phi, chi, two_dtheta, z  ← (N_frames,) for scanned, scalar for fixed
                                     phi/chi/two_dtheta in degrees (@units="degree");
                                     z in micrometers (@units="micrometer")
      measurement/                               ← BLISS soft-links
        @NX_class = "NXcollection"
        dfxm_sim_detector, phi, chi, two_dtheta, z   ← SoftLink to instrument/...
      dfxm_geo/                                  ← per-scan sim-specific
        Hg, q_hkl, theta, psize, zl_rms
        analysis/...                             ← present only after forward postprocess
    /2.1/                                        ← next scan; same structure
    /3.1/, /4.1/, ...                            ← only identification masters have >2
  scan0001/
    dfxm_sim_detector_0000.h5                    ← LIMA-style detector file, pixels only
      / @NX_class="NXroot", @creator="dfxm-geo", @default="entry_0000"
      /entry_0000/
        @NX_class = "NXentry"
        title, start_time, end_time
        dfxm_sim_detector/
          @NX_class = "NXdetector"
          image  (N_frames, H, W) float32
            chunks=(1, H, W), gzip-4 + shuffle
            @interpretation = "image"
        plot/  @NX_class="NXdata", @signal="image"
          image → SoftLink(/entry_0000/dfxm_sim_detector/image)
        measurement/ → SoftLink(/entry_0000/dfxm_sim_detector/image)
  scan0002/dfxm_sim_detector_0000.h5
  ...
  scanNNNN/dfxm_sim_detector_0000.h5
```

**External link targets are stored as relative paths** (forward-slashes,
even on Windows — required for HDF5 portability), so the entire `out_dir/`
tree is portable. Moving or copying the directory preserves all links.

## Master file (`dfxm_geo.h5` / `dfxm_identify.h5`)

The master is a single HDF5 file at the top of the output directory. It
carries two kinds of content: the global `/dfxm_geo/` provenance group
(written once per run) and one BLISS-style `/N.1/` scan entry per scan.

### `/dfxm_geo/` provenance group

Written exactly once, on master close. Captures everything needed to
reproduce the run:

```
/dfxm_geo/
  version              "1.2.0"
  git_sha              "..." (40-char) or "unknown"
  git_dirty            bool
  generated_at         "2026-05-21T10:00:00+00:00"
  hostname             "borgi-laptop" / "n-62-12-15"
  python_version, numpy_version
  cli                  full command line
  config_toml          entire TOML config as a UTF-8 string
  kernel/
    pkl_fn             "Resq_i_2026-05-20_2014.npz"
    sha256             64-char hex digest
    qi1_range...Nrays  mirrored from the kernel npz
```

This group is identical to the v1.1.0 provenance group; only its location
(now always in the master, regardless of how many scan directories the run
produced) is different.

### `/N.1/` BLISS scan entries

Each scan in the run becomes one `/N.1/` entry under the master root.
Forward mode writes `/1.1/` (dislocations) and optionally `/2.1/` (perfect
crystal, when `[io].include_perfect_crystal = true`). Identification mode
writes one `/N.1/` per config drawn from the runner generator. For
`mode="single"`: `n_z × n_planes × n_b × n_alpha` entries (where `n_z` =
`[scan.z].steps` if set, else 1). For `mode="multi"`: `n_z × n_samples`.
For `mode="z-scan"`: `len(z_offsets_um) × phi_chi_configs` (legacy z-scan
mode, distinct from `[scan.z]` axis scanning). New in v1.3.0: `[scan.z]`
in single/multi multiplies the scan count by `n_z` and recomputes the
strain field at each z-slice. New in v1.3.1: `[scan.two_dtheta]` is a
within-scan axis in identification (multiplies `n_frames`-per-scan, not
the scan count) — it differs from `[scan.z]` because the
deformation-gradient `Hg` is two_dtheta-independent, so `forward()`
shifts the Bragg angle internally without recomputing `Hg`.

The metadata schema of `/N.1/` is identical to v1.1.0 except that
`instrument/dfxm_sim_detector/data` is now an `ExternalLink` to the
sibling scan directory's LIMA file, not an inline dataset. Readers that
walk `f["/1.1/instrument/dfxm_sim_detector/data"][...]` continue to work
unchanged because h5py follows external links transparently.

The per-scan `/N.1/dfxm_geo/` subgroup carries the strain provenance
(`Hg`, `q_hkl`, `theta`, `psize`, `zl_rms`). `/N.1/dfxm_geo/Hg` (the large
per-ray displacement-gradient array, ~106 MB/config) is written only when
`io.write_strain_provenance` is true (default); disable it for batch/ML runs
to drop ~106 MB/config — the tiny scalars are always kept.

## Per-`/N.1` attrs

Each scan entry carries attributes that summarize the scan mode and
crystal/identify mode used to generate it. These are written verbatim from
the B+C `[scan.<axis>]` schema for `scan_mode` / `scanned_axes` and from
the C / E discriminators for `crystal_mode` / `identify_mode`.

| Attr            | Forward | Identify | Example values                                                                                                            |
| --------------- | :-----: | :------: | ------------------------------------------------------------------------------------------------------------------------- |
| `scan_mode`     |    ✓    |    ✓     | `"single"`, `"rocking"`, `"rolling"`, `"strain"`, `"layer"`, `"mosa"`, `"mosa_strain"`, `"mosa_layer"`, `"mosa_strain_layer"` |
| `scanned_axes`  |    ✓    |    ✓     | `[]`, `["phi"]`, `["chi"]`, `["phi", "chi"]`, `["phi", "chi", "two_dtheta", "z"]` |
| `crystal_mode`  |    ✓    |    —     | `"centered"`, `"wall"`, `"random_dislocations"`                                                                           |
| `identify_mode` |    —    |    ✓     | `"single"`, `"multi"`, `"z-scan"`                                                                                         |

`scan_mode` and `scanned_axes` are derived from
`config.scan.derived_mode_name()` and `config.scan.scanned_axes()`.
`crystal_mode` is set by sub-project C's discriminated-union layout.
`identify_mode` is new in v1.2.0 (sub-project E).

> **Known gap (deferred to v1.3.0):** `/2.1` currently does **not** carry
> `scan_mode` / `scanned_axes` / `crystal_mode` attrs even though it
> shares the same configuration as `/1.1`. Only `/1.1` has them at
> present. Tracked as a B+C follow-up.

### Structure provenance attrs (M4 Stage 4.3a)

For **structure-aware** runs (oblique geometry or an explicit `[crystal]
structure_type` / `cif` / `space_group`), each `/N.1/` entry is extended
with the following attributes written by
`dfxm_geo.io.hdf5.structure_provenance_attrs`:

| Attr | Type | Example | Notes |
| ---- | ---- | ------- | ----- |
| `structure_type` | string | `"bcc"` | Resolved structure family (`"fcc"` or `"bcc"`) |
| `poisson_ratio` | float64 | `0.29` | Resolved ν used in the displacement-field calculation |
| `poisson_source` | string | `"KL"` | Citation tag: `"KL"` (Kaye & Laby), `"SW"` (Simmons & Wang), or `"override"` |
| `burgers_magnitude_um` | float64 | `2.485e-4` | \|b\| in µm for the primary slip family |
| `material` | string | `"Fe"` | Only present when `[crystal] material` was set |
| `slip_families` | list[string] | `["{110}<111>"]` | Only present when `[crystal] slip_families` was set |
| `space_group` | string | `"Im-3m"` | Only present when a space group was derived from CIF or set explicitly |

**FCC simplified path:** when `mount` is `None` (the default for a symmetric
FCC run without `[crystal] cif`, `structure_type`, or `space_group`), none of
these attrs are written. Outputs from this path remain byte-identical to v2.5.x.
See `docs/crystal-structures.md` for the full provenance rules and the
resolution precedence for `structure_type` and `poisson_ratio`.

## Per-mode `sample/` layouts

The `/N.1/sample/` group's contents depend on which pipeline wrote the
master and (for identification) which sub-mode produced the scan.

### Forward mode

Unchanged from v1.1.0; only the file location has moved (single-file to
master):

```
/N.1/sample/
  @NX_class = "NXsample"
  name                "simulated, dislocations" | "simulated, perfect crystal"
  dis                 scalar float64    (wall mode only; sentinel/absent for centered + random)
  ndis                scalar int64
  sample_remount      scalar string     (wall mode only; "N/A" otherwise)
```

> **Known gap (deferred to v1.3.0):** `dis` is written as `-1.0` for
> centered + random_dislocations crystal modes, which is a meaningless
> sentinel rather than `None` or an omitted field. Tracked as a B+C
> follow-up.

### Identify single mode

One dislocation per scan, drawn from the `[single]` config block:

```
/N.1/sample/
  @NX_class = "NXsample"
  name                "simulated, dislocation identification (single)"
  slip_plane_normal   (3,) int32          e.g. [1, 1, 1]
  burgers             (3,) int32          e.g. [1, 0, 1]  (scaled by √2 to integers)
  rotation_deg        scalar float64      e.g. 45.0
```

### Identify multi mode

Two dislocations per scan (Monte Carlo, drawn per sample). Layout uses an
`NXcollection` of `NXsample` sub-groups, one per dislocation index:

```
/N.1/sample/
  @NX_class = "NXsample"
  name                "simulated, dislocation identification (multi)"
  dislocations/
    @NX_class = "NXcollection"
    0/
      @NX_class = "NXsample"
      slip_plane_normal, burgers, rotation_deg
      position_um       (3,) float64      lab-coord (x, y, z=0)
    1/
      @NX_class = "NXsample"
      slip_plane_normal, burgers, rotation_deg
      position_um       (3,) float64
```

Dislocation index (`0`, `1`) is draw order, not a spatial ordering. The
trainer correlates pixel features to which dislocation via `position_um`
projected through the detector geometry.

### Identify z-scan mode

A primary dislocation always present; an optional secondary one if
`[zscan].include_secondary = true`:

```
/N.1/sample/
  @NX_class = "NXsample"
  name                "simulated, dislocation identification (z-scan)"
  z_offset_um         scalar float64
  primary/
    @NX_class = "NXsample"
    slip_plane_normal, burgers, rotation_deg
    position_um       (3,) float64       (always [0, 0, 0] currently)
  secondary/                              ← only present if include_secondary=True
    @NX_class = "NXsample"
    slip_plane_normal, burgers, rotation_deg
    position_um       (3,) float64       (always [0, 0, 0] currently — pos_std_um=0 in current draw)
```

## Per-scan LIMA-style detector files

Each `scanNNNN/dfxm_sim_detector_0000.h5` is a standalone HDF5 file in the
ESRF LIMA style. It carries only pixel data and the minimum NeXus
scaffolding around it. The internal layout is:

```
scan0001/dfxm_sim_detector_0000.h5
  / @NX_class="NXroot", @creator="dfxm-geo", @default="entry_0000"
  /entry_0000/
    @NX_class = "NXentry"
    dfxm_sim_detector/
      @NX_class = "NXdetector"
      image  (N_frames, H, W) float32
        chunks=(1, H, W), gzip-4 + shuffle
        @interpretation = "image"
    plot/
      @NX_class = "NXdata"
      @signal = "image"
      image → SoftLink(/entry_0000/dfxm_sim_detector/image)
    measurement → SoftLink(/entry_0000/dfxm_sim_detector/image)
```

The internal path `/entry_0000/dfxm_sim_detector/image` is the canonical
ExternalLink target referenced from the master, so a master entry
`/N.1/instrument/dfxm_sim_detector/data` resolves transparently to the
linked dataset.

The `plot/` (NXdata) and root-level `measurement` soft-link are
BLISS-compatible aliases for the same pixel dataset; they are what
`silx view` and `darfix` use to auto-discover the "default plot" without
extra hints.

## Multi-mode opt-in: `render_per_dislocation`

By default, multi mode writes **one** detector file per scan dir — the
combined detector image with both dislocations' contributions summed and
Poisson noise applied. Setting `[multi].render_per_dislocation = true`
opts in to a three-detector pattern instead:

```
scan0001/
  dfxm_sim_detector_0000.h5            ← both dislocations summed (canonical detector, with Poisson noise)
  dfxm_sim_detector_dis0_0000.h5       ← first dislocation only (noiseless)
  dfxm_sim_detector_dis1_0000.h5       ← second dislocation only (noiseless)
```

In this mode, the master's `/N.1/instrument/` carries three `NXdetector`
groups (`dfxm_sim_detector`, `dfxm_sim_detector_dis0`,
`dfxm_sim_detector_dis1`), each with its own `ExternalLink` pointing at
the corresponding LIMA file. Each per-dislocation file holds exactly the
same `(N_frames, H, W)` stack shape as the combined detector, and the
master's `/N.1/measurement/` group acquires one SoftLink per detector for
BLISS-compatible discovery.

**Noise policy:** The combined detector receives the canonical Poisson
noise stream (so it's a realistic detector image). Per-dislocation files
are **noiseless** (`image_arr * intensity_scale` only) — they exist to
give ML training pipelines deterministic per-instance ground truth, and
adding independent Poisson draws to each would corrupt the signal/
background balance vs. the combined detector. The per-dis files sum
exactly to the noiseless underlying image of the combined detector.

Total compute cost: N_samples × 3 forward() calls (3× the default).
Real beamlines do exactly this pattern when scanning multiple detectors
simultaneously, so the layout stays BLISS-faithful.

## Frame ordering

Starting in v1.3.0, the canonical scan trajectory walks 4 axes:
phi-innermost, then chi, then `two_dtheta`, then z-outermost.
Phi/chi-only scans remain v1.1.0/v1.2.0-compatible (the outer two loop
levels collapse to length-1 when their axis is fixed). The k-th frame
in any detector file corresponds to:

```python
phi_idx        = k % phi_steps
chi_idx        = (k // phi_steps) % chi_steps
two_dtheta_idx = (k // (phi_steps * chi_steps)) % two_dtheta_steps
z_idx          = k // (phi_steps * chi_steps * two_dtheta_steps)
```

`N_frames` per `/N.1/` depends on which axes are scanned:

| scan_mode           | scanned_axes                              | N_frames                                                |
| ------------------- | ----------------------------------------- | ------------------------------------------------------- |
| `single`            | `[]`                                      | 1                                                       |
| `rocking`           | `["phi"]`                                 | `phi_steps`                                             |
| `rolling`           | `["chi"]`                                 | `chi_steps`                                             |
| `strain`            | `["two_dtheta"]`                          | `two_dtheta_steps`                                      |
| `layer`             | `["z"]`                                   | `z_steps`                                               |
| `mosa`              | `["phi", "chi"]`                          | `phi_steps × chi_steps`                                 |
| `mosa_strain`       | `["phi", "chi", "two_dtheta"]`            | `phi_steps × chi_steps × two_dtheta_steps`              |
| `mosa_layer`        | `["phi", "chi", "z"]`                     | `phi_steps × chi_steps × z_steps`                       |
| `mosa_strain_layer` | `["phi", "chi", "two_dtheta", "z"]`       | `phi_steps × chi_steps × two_dtheta_steps × z_steps`   |

Positioners follow the same rule: scanned axes get `(N_frames,)` 1-D
arrays; fixed axes get scalar 0-D datasets equal to `axis.value`. Units:
`phi`, `chi`, `two_dtheta` are stored in degrees (`@units="degree"`);
`z` is stored in micrometers (`@units="micrometer"`).

`load_h5_scan` returns the data both as a flat `(N_frames, H, W)` stack
and as a `(phi_steps, chi_steps, H, W)` reshape for convenience.
For 4-axis scans, `load_h5_scan`'s `stack_reshape` may not match the
higher-rank trajectory — fall back to indexing `stack` directly using
the formula above when consuming `mosa_strain_layer` / `mosa_strain` /
`mosa_layer` data.

## NX_class table

Quick reference of which groups carry which `NX_class` attribute, so
silx, NeXpy, and other NeXus-aware viewers can navigate the tree:

| Path                                                              | NX_class       |
| ----------------------------------------------------------------- | -------------- |
| `dfxm_geo.h5` / `dfxm_identify.h5` root                           | `NXroot`       |
| `/N.1`                                                            | `NXentry`      |
| `/N.1/sample`                                                     | `NXsample`     |
| `/N.1/sample/dislocations` (multi mode)                           | `NXcollection` |
| `/N.1/sample/dislocations/<i>` (multi mode)                       | `NXsample`     |
| `/N.1/sample/primary` / `/N.1/sample/secondary` (z-scan mode)     | `NXsample`     |
| `/N.1/instrument`                                                 | `NXinstrument` |
| `/N.1/instrument/dfxm_sim_detector` (and `_dis0`, `_dis1`)        | `NXdetector`   |
| `/N.1/instrument/positioners`                                     | `NXcollection` |
| `/N.1/measurement`                                                | `NXcollection` |
| `scanNNNN/dfxm_sim_detector_0000.h5` root                         | `NXroot`       |
| `scanNNNN/.../entry_0000`                                         | `NXentry`      |
| `scanNNNN/.../entry_0000/dfxm_sim_detector`                       | `NXdetector`   |
| `scanNNNN/.../entry_0000/plot`                                    | `NXdata`       |

We do not claim full NeXus compliance — these attrs are a free interop
bonus, not a contract.

## Reading the data

### Via `load_h5_scan`

The recommended high-level reader. Follows external links transparently;
returns flat and reshaped views in one call:

```python
from dfxm_geo.io.hdf5 import load_h5_scan

stack, stack_reshape, h, w = load_h5_scan(
    "out_dir/dfxm_geo.h5", scan_id="1.1",
    # phi_steps / chi_steps inferred from /dfxm_geo/config_toml if omitted
)
# stack:         (N_frames, H, W) float64
# stack_reshape: (phi_steps, chi_steps, H, W) float64 — same memory, different view
```

### Via h5py directly

h5py follows `ExternalLink`s automatically, so reads against the master
look exactly like v1.1.0 reads:

```python
import h5py
with h5py.File("out_dir/dfxm_geo.h5", "r") as f:
    images = f["/1.1/instrument/dfxm_sim_detector/data"][...]   # resolved via ExternalLink
    phi_deg = f["/1.1/instrument/positioners/phi"][...]
    chi_deg = f["/1.1/instrument/positioners/chi"][...]
```

To inspect the link itself instead of the linked dataset, use
`get(..., getlink=True)`:

```python
with h5py.File("out_dir/dfxm_geo.h5", "r") as f:
    link = f["/1.1/instrument/dfxm_sim_detector"].get("data", getlink=True)
    assert isinstance(link, h5py.ExternalLink)
    print(link.filename, link.path)
    # → scan0001/dfxm_sim_detector_0000.h5  /entry_0000/dfxm_sim_detector/image
```

You can also bypass the master entirely and open the per-scan detector
file directly:

```python
with h5py.File("out_dir/scan0001/dfxm_sim_detector_0000.h5", "r") as f:
    images = f["/entry_0000/dfxm_sim_detector/image"][...]
```

### Via darfix

```bash
darfix out_dir/dfxm_geo.h5
```

Darfix's `ImageDataset` (via `silx.io.DataUrl`) discovers the BLISS
scan navigator automatically. Note that darfix transitively depends on
`scikit-image`; install it alongside if you don't already have it.

### Via darling

Use the `DarlingReader` shipped in `dfxm_geo`:

```python
import darling
from dfxm_geo.io.darling_reader import DarlingReader

dset = darling.DataSet(DarlingReader("out_dir/dfxm_geo.h5"))
dset.load_scan("1.1")
print(dset.data.shape, dset.motors.shape)
```

**Why a custom reader is required (v1.2.0+):** darling 2.0.0 discovers the
detector dataset with `h5py.visititems`, which does **not** traverse external
links. Since v1.2.0 the per-scan detector data lives in a separate LIMA-style
file linked into the master via an `ExternalLink`, so darling's built-in
readers cannot find it (and pointing darling at the per-scan file fails too —
it lacks the BLISS scan structure). `DarlingReader` opens the master, reads the
stack through the explicit linked path, and hands darling pre-resolved
`(a, b, m, n)` data plus `(k, m, n)` motor meshgrids — exactly darling's
`Reader` contract. It has no import-time dependency on darling, so importing
`dfxm_geo` never requires darling to be installed.

For a plain in-memory stack without darling, use
`dfxm_geo.io.darling_reader.resolve_detector_data(path, scan_id)`, which returns
the `(N_frames, H, W)` array with the external link resolved.

`DarlingReader` handles rectilinear scans over one or two varying motors (the
case darling targets). For >2 scanned axes use `load_h5_scan` or silx directly.

## Compression

Image data is chunked `(1, H, W)` (one frame per chunk) with gzip-4 plus
the shuffle filter. Detector images are stored as `float32` — lossless for
the simulated intensities, and it halves file size; the forward model
computes in `float64`. Typical compression ratio is
~3-5× for simulated images (which are sparse — most pixels are near
zero); read latency for one full stack on a laptop SSD is ~1-2 seconds.

Chunks-of-one means frame-by-frame access is fast (no need to decompress
unrelated frames), at the cost of slightly worse overall compression
than larger chunks would give. This matches the BLISS convention and
keeps darfix's "load N frames at a time" pattern efficient.

## Migration from v1.1.0

A v1.1.0 single-file `dfxm_geo.h5` can be converted to the v1.2.0
master + per-scan-dirs layout in place with `dfxm-migrate-h5`:

```bash
dfxm-migrate-h5 path/to/v110_dfxm_geo.h5
# → path/to/v110_dfxm_geo.h5.v120/
#     ├── dfxm_geo.h5            ← new master
#     ├── scan0001/dfxm_sim_detector_0000.h5
#     └── scan0002/dfxm_sim_detector_0000.h5

# or with an explicit output directory:
dfxm-migrate-h5 path/to/v110_dfxm_geo.h5 --output path/to/new_out_dir/
```

The migration preserves all non-pixel-data nodes byte-for-byte:
`/dfxm_geo/` provenance, `/N.1/sample/`, `/N.1/dfxm_geo/` (including
`analysis/` if present), positioners, measurement soft-links, title /
start_time / end_time. Only the pixel datasets at
`/N.1/instrument/dfxm_sim_detector/data` are moved out into the
per-scan LIMA files and replaced with `ExternalLink`s.

A round-trip check is run as part of the package's test suite:
`load_h5_scan(migrated_master, scan_id="1.1")` returns byte-identical
pixels to `load_h5_scan(original, scan_id="1.1")`.

To convert legacy `.npy` output directories (pre-v1.1.0), use
`dfxm-migrate-output` — which now also emits the v1.2.0 layout
directly:

```bash
dfxm-migrate-output <old_output_dir>
# or with an explicit config:
dfxm-migrate-output <old_output_dir> --config configs/default.toml
```

## External link path caveat (symlinks)

Relative `ExternalLink` targets in HDF5 resolve from the **master
file's directory on disk** — that is, from whatever directory contains
the file h5py actually opens. This is what makes the `out_dir/` tree
portable: copy or move the entire directory and the links continue to
work without rewriting.

There is one subtle edge case to be aware of: if you create a symlink
to the master file and open the symlink, h5py follows the symlink and
resolves the external links from the **symlink target's** directory,
not from the symlink's own directory. Concretely:

```
/data/runs/run-2026-05-21/
  dfxm_geo.h5                         ← the real master
  scan0001/dfxm_sim_detector_0000.h5
  scan0002/dfxm_sim_detector_0000.h5

/home/borgi/latest.h5  →  /data/runs/run-2026-05-21/dfxm_geo.h5   (symlink)
```

Opening `/home/borgi/latest.h5` resolves the `ExternalLink` targets
relative to `/data/runs/run-2026-05-21/`, not to `/home/borgi/`. The
links continue to work, so this is usually invisible — but if you also
symlink the scan directories into `/home/borgi/` and expect h5py to
discover those, it will not: only the target directory is searched.

Best practice: don't symlink across the master/scan-dir boundary.
Either symlink the whole `out_dir/` tree as one unit (which works
correctly because relative paths are preserved inside the tree), or
copy it to its destination.

## Implementation notes

### Fg disk cache (wall mode, v1.3.0)

`Find_Hg` writes a disk cache for the per-z dislocation-wall strain
tensor (`direct_space/deformation_gradient_tensors/Fg_<dis>_<...>.npy`).
When `z_offset_um` is non-zero (forward-mode `[scan.z]` outer loop), the
filename gains a `_z{nm}nm` suffix encoding `round(z_offset_um * 1000)`.
Example: `z_offset_um = 12.5` → `Fg_...remountS1_z12500nm.npy`. The
`z = 0` filename is unchanged from v1.2.0, so existing caches remain
hit-eligible for non-z-scan runs. `Find_Hg_from_population` (centered +
random_dislocations modes) does not write a disk cache and is therefore
unaffected by this suffix.

### z positioner unit handling (v1.3.0)

The z positioner is a translation axis stored in **micrometers**
(`@units="micrometer"`) — the raw value from `[scan.z].range / .value`
is written without conversion. The three angular axes (`phi`, `chi`,
`two_dtheta`) are radians upstream and stored in degrees on disk
(`@units="degree"`). `MasterWriter.add_scan` special-cases `z` to skip
the `np.degrees()` conversion that would otherwise produce a meaningless
value with a misleading unit attribute.
