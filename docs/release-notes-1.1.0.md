# v1.1.0 — HDF5 output format

**Release date:** 2026-05-17

## Highlights

- `dfxm-forward` now writes a single BLISS-style HDF5 file (`<output_dir>/dfxm_geo.h5`) instead of thousands of per-frame `.npy` files. ~3.5× space savings + full embedded provenance.
- Compatible out of the box with [darfix](https://darfix.readthedocs.io/) and [darling](https://github.com/AxelHenningsson/darling) — the actual analysis tools at ESRF ID03 / ID06.
- New `dfxm-migrate-output` CLI converts legacy `.npy` output directories to the new format.

## Breaking changes (B3 cutover)

- `dfxm_geo.io.images.load_images` is **removed** from the public API. New code uses `dfxm_geo.io.hdf5.load_h5_scan`. Legacy `.npy` reading is preserved internally inside `dfxm_geo.io.migrate` for the migration script.
- `<output_dir>/images10/` and `<output_dir>/images10_perf_crystal/` are no longer created. The corresponding `IOConfig.dislocs_dirname` / `perfect_dirname` / `fn_prefix` / `ftype` fields are now ignored.
- `dfxm_geo.io.images` trimmed: `save_edfs`, `load_image`, `load_images`, `load_images_parallel` removed. `save_images_parallel` retained because identification z-scan (`dfxm-identify` mode=z-scan) still uses it.

## Out of scope (deferred)

- `dfxm-identify` output format stays on `.npy` + CSV. HDF5 migration deferred to v1.2.0 after laptop+cluster feedback validates the schema.
- Durability hardening (HDF5 flush / atomic rename) — only revisit if jobs grow past ~3 hours.

## Migration

```bash
dfxm-migrate-output <old_output_dir>
# or with explicit config:
dfxm-migrate-output <old_output_dir> --config configs/default.toml
```

## Provenance

Every output file embeds:
- `dfxm-geo` package version
- git HEAD SHA and dirty flag
- Hostname, Python version, numpy version, ISO timestamp
- Full TOML config used to generate the file
- SHA-256 of the reciprocal-space kernel npz used

See [docs/output-format.md](output-format.md) for the full schema.

## Design history

The HDF5 format was designed through a structured /grill-me interview (Q1-Q10) covering drivers, schema fidelity, file granularity, two-stack representation, placement patterns, write patterns, scope, backwards-compat, compression, and test plan. The resolved decisions are saved in `memory/followups_hdf5_output_format.md`.
