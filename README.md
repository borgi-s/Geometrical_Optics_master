# DFXM Geometrical-Optics Forward Model

A Python implementation of the geometrical-optics forward model for Dark Field
X-ray Microscopy (DFXM), as published in:

> Borgi, S. et al. *J. Appl. Cryst.* (2024).
> DOI: [10.1107/S1600576724001183](https://doi.org/10.1107/S1600576724001183)
> [Article on IUCr](https://journals.iucr.org/j/issues/2024/02/00/nb5370/)

The default beamline configuration matches ID06 at the European Synchrotron
Radiation Facility (ESRF).

## What this code does

Given a crystal containing dislocations, this code simulates the DFXM images
that would be recorded on a detector under a defined beam and goniometer
geometry. It models both the direct-space deformation field around dislocations
and the reciprocal-space resolution function of the microscope.

This is not a generic optics simulator — it is specifically a *forward model*
for dark-field X-ray microscopy at synchrotron sources, used to interpret
images of strain fields and crystal defects.

## Status

This repository is undergoing a structural cleanup (branch
`cleanup/main-modernization`). The physics is stable; the surrounding
engineering is being modernized. See `docs/superpowers/plans/2026-05-12-codebase-cleanup.md`
for the full roadmap.

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/borgi-s/Geometrical_Optics_master.git
cd Geometrical_Optics_master
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                             # smoke tests should pass
```

## Running a simulation

Both entry points expect a pickled reciprocal-space resolution function at
`reciprocal_space/pkl_files/Resq_i_<timestamp>.pkl`, generated once with:

```bash
python reciprocal_space/generate_Resq_i.py
```

### Recommended: config-driven CLI

```bash
dfxm-forward --config configs/default.toml --output ./out
```

The CLI runs the forward simulation only (rocking sweep → image stacks
on disk). See [`docs/reproducibility.md`](docs/reproducibility.md) for the
config schema, pre-built variants for different dislocation densities,
and what is not yet configurable.

### Legacy demo: `init_forward.py`

```bash
python init_forward.py
```

Still in place as a runnable demo that produces the per-pixel COM /
mosaicity maps and SVG paper-style figures. The simulation portion of
the demo is equivalent to `dfxm-forward --config configs/default.toml`;
the analysis + plotting will move into `dfxm_geo.analysis` in a future
phase (see plan, Phase 9).

## Project structure

```
.
├── functions.py                  Crystal mechanics and dislocation fields
├── image_processor.py            Image I/O, moment/FWHM analysis, parallel rendering
├── init_forward.py               Main simulation entry script
├── direct_space/
│   └── forward_model.py          Direct-space forward simulator
├── reciprocal_space/
│   ├── generate_Resq_i.py        Resolution-function generator (run first)
│   ├── recspace_res.py           Monte Carlo reciprocal-space resolution
│   └── exposure_time.py          Exposure-time helper
├── tests/                        Pytest smoke tests
└── docs/                         Architecture, physics, reproducibility guides (planned)
```

A future refactor (plan Phase 4) moves the physics modules into
`src/dfxm_geo/{crystal,direct_space,reciprocal_space,analysis,io,viz}`.

## Reproducing the paper figures

See [`docs/reproducibility.md`](docs/reproducibility.md) for the current
recipe (CLI for the simulation, `init_forward.py` for the post-processing,
pending the Phase 9 port of the analysis into `dfxm_geo.analysis`).
Reference datasets are scheduled for Zenodo deposit; until then, contact
the corresponding author.

## Citing

See `CITATION.cff`.

```bibtex
@article{borgi2024dfxm,
  title  = {Geometrical Optics: forward modelling of Dark Field X-ray Microscopy},
  author = {Borgi, Sina and others},
  journal= {Journal of Applied Crystallography},
  year   = {2024},
  doi    = {10.1107/S1600576724001183},
}
```

(Title and full author list need verification against the published paper —
see the `TODO(borgi)` note in `CITATION.cff`.)

## Contributing

PRs welcome. Run `pre-commit run --all-files` before pushing.

## License

MIT. See `LICENSE`.
