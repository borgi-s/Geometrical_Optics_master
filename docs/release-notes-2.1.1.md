# DFXM v2.1.1 — Follow-ups: reproducible kernels, darling interop, z-scan instance labels

Released: 2026-05-26.

A batch of three deferred follow-ups, all backward-compatible additions
(no breaking changes), plus the README header figure fix and a conda-forge
recipe.

## Reproducible Monte Carlo kernels (`dfxm-bootstrap --seed`)

The reciprocal-space resolution kernel is a Monte Carlo integral, but its
RNG was never seedable from `generate_kernel` or the `dfxm-bootstrap` CLI,
so two bootstraps of identical config produced slightly different kernels.

- `generate_kernel(seed=None)` now seeds `np.random.default_rng(seed)`,
  threads it through `reciprocal_res_func`, and records `seed` in the npz
  metadata (`-1` sentinel = unseeded entropy run) for provenance.
- `dfxm-bootstrap --seed N` exposes it on the CLI and overrides any `seed`
  in the `[reciprocal]` TOML block. The optional key is documented in
  `configs/default.toml`.

Also corrects two long-stale bit-equivalence `xfail` reasons that wrongly
blamed "non-seeded `Find_Hg`" — the wall path `Find_Hg → Fd_find` is fully
deterministic; the real blockers are a missing 2023 legacy pickle and a
golden captured at `Nsub=2` vs the now-default `Nsub=1`.

## darling interop: `DarlingReader`

darling 2.0.0 discovers detector datasets with `h5py.visititems`, which
does not traverse external links — so since the v1.2.0 master+per-scan
layout (detector data in a LIMA-style file linked via `ExternalLink`),
darling's built-in readers could not find our detector data.

- New `dfxm_geo.io.darling_reader.DarlingReader`: honours darling's
  `Reader` protocol, opening the master and reading the stack through the
  explicit linked path, returning pre-resolved `(a,b,m,n)` data +
  `(k,m,n)` motor meshgrids. No import-time dependency on darling.
- `resolve_detector_data(path, scan_id)` returns the plain in-memory
  `(N_frames, H, W)` stack with the external link resolved.

```python
import darling
from dfxm_geo.io.darling_reader import DarlingReader
dset = darling.DataSet(DarlingReader("out_dir/dfxm_geo.h5"))
dset.load_scan("1.1")
```

## z-scan instance labels (`render_per_dislocation`)

z-scan identification gains multi mode's `render_per_dislocation`. When a
secondary dislocation is present (`include_secondary=True`),
`render_per_dislocation=True` writes two extra **noiseless** detectors per
scan — `dfxm_sim_detector_primary` and `dfxm_sim_detector_secondary` —
each holding one dislocation rendered in isolation (ground-truth instance
labels that bypass the post-write Poisson pass). The flag requires a
secondary and is rejected otherwise.

## Packaging

- README header schematic re-exported from the vector source: the previous
  raster was clipped at the bottom edge (the "Condenser"/"Goniometer"
  labels were cut off); the full figure now renders with margins.
- A conda-forge recipe (`packaging/conda/recipe.yaml`) pulling the v2.1.0
  PyPI sdist, with submission notes in `packaging/conda/README.md`.

## Tests

Suite: 530 passed / 1 skipped / 12 deselected / 1 xfailed; mypy clean.
New: `TestKernelSeed` (6), `test_darling_reader.py` (6, incl. a real
`darling.DataSet` round-trip), `test_identification_zscan_per_dis.py` (4).

## Upgrade

```bash
pip install --upgrade dfxm-geo==2.1.1
```

No API or config changes required; all additions are opt-in.
