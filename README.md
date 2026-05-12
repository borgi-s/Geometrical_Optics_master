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

The current entry point is `init_forward.py`. It expects a pickled
reciprocal-space resolution function at
`reciprocal_space/pkl_files/Resq_i_<timestamp>.pkl`, which you must generate
first:

```bash
# 1. Generate the reciprocal-space resolution kernel
python reciprocal_space/generate_Resq_i.py

# 2. Run the forward simulation
python init_forward.py
```

A future cleanup phase (see plan, Phase 6) replaces `init_forward.py` with a
config-driven CLI:

```bash
# Planned, not yet available:
dfxm-forward --config configs/default.toml --output ./out
```

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

A dedicated notebook for this is being prepared
(`notebooks/99_paper_figures.ipynb`, plan Phase 9). For now, the figures in
the *J. Appl. Cryst.* 2024 article were produced by running `init_forward.py`
against a specific set of input dislocation geometries — contact the
corresponding author for the reference dataset.

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
