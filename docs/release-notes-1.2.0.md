# Release notes — v1.2.0

**Release date:** TBD (release branch)
**Highlights:** Sub-projects A + B + C + D + E land together. Pipeline-features arc complete.

## Breaking changes

- **Output layout (forward + identify): single-file → master + per-scan-dirs.**
  v1.1.0 wrote one `dfxm_geo.h5` containing pixel data inline. v1.2.0 writes
  `dfxm_geo.h5` as a *master* that ExternalLinks to per-scan LIMA-style
  detector files in `scan0001/`, `scan0002/`, … subdirectories of the output
  directory. The shape on disk now mirrors an ESRF BLISS dataset.
  - **Migration tool:** `dfxm-migrate-h5 <v110_dfxm_geo.h5> [--output <dir>]`.
- **Identification mode: `.npy` + `manifest.csv` + `images/*.png` → HDF5.**
  All three sub-modes (`single`, `multi`, `z-scan`) write `dfxm_identify.h5`
  + per-scan detector files in the same shape as forward. Ground-truth labels
  live in `/N.1/sample/` (per-mode layout — see `docs/output-format.md`).
  Manifest/preview sidecars are dropped.
- **`IdentificationZScanConfig`: dropped `phi_range_deg / phi_steps /
  chi_range_deg / chi_steps`.** The rocking-curve grid is now read from the
  shared `[scan.phi]` / `[scan.chi]` (B+C schema). Update existing configs.
- **`IdentificationMonteCarloConfig`: dropped `n_png_previews`; added
  `render_per_dislocation: bool = False`.** When `true`, each multi-mode
  scan dir also writes per-dislocation detector files
  (`dfxm_sim_detector_dis0_0000.h5`, `..._dis1_0000.h5`) for unambiguous
  instance labels. Per-dis files are noiseless; the combined detector
  receives Poisson noise as before.

## New features (A + B + C + D, untagged on main since v1.1.0)

- **A — multi-reflection bootstrap + Bragg validity.** `[reciprocal]` block in
  TOML carries hkl + keV; kernel lookup validates Bragg-satisfiability before
  load.
- **D — multi-reflection kernel lookup in forward + identify.** Kernels
  bundle their `hkl`/`keV` metadata; runtime picks the matching one and
  loads it on the fly.
- **B — per-axis `[scan.<axis>]` schema + derived mode names.** `single`,
  `rocking`, `rolling`, `mosa`, etc. derived from which axes carry
  range+steps.
- **C — crystal layouts: discriminated union.** `[crystal] mode = "centered"
  | "wall" | "random_dislocations"` with matching sub-block.
- **E — identification → HDF5 (this release).**

## Out of scope / deferred (v1.3.0+)

- Wiring `[scan.two_dtheta]` / `[scan.z]` into forward + identification
  kernels (eager `ValueError` for now).
- z-scan mode consolidation into `single + [scan.z]`.
- Pixel-level segmentation masks for multi mode.
- `render_per_dislocation` analogue for z-scan's primary/secondary pair.
- `_SLIP_SYSTEM_111` table extension (6/12 → 12/12 FCC slip systems).
- `sample_dis = -1.0` sentinel cleanup (centered + random_dislocations).
- `/2.1` HDF5 attrs `scan_mode`/`scanned_axes`/`crystal_mode` (currently
  only `/1.1` carries them).

## Migration checklist for existing users

1. **HDF5 outputs from v1.1.0:** run `dfxm-migrate-h5 <old.h5>` to convert.
2. **TOML configs for z-scan mode:** move `[zscan].phi_*/chi_*` to
   `[scan.phi]/[scan.chi]`. See `configs/identification_zscan.toml`.
3. **TOML configs for multi mode:** remove `n_png_previews`. To enable
   per-dis rendering, add `render_per_dislocation = true`.
4. **Cluster bootstrap:** `git pull && pip install -e ".[dev]" &&
   dfxm-bootstrap --config configs/default.toml` to refresh the kernel
   with the v1.2.0 metadata.
