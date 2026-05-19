# DFXM Geometrical-Optics Forward Model

> A physics-based forward model that simulates Dark-Field X-ray Microscopy images of crystals with dislocations — published in the Journal of Applied Crystallography (2024).

## What this is

Dark-Field X-ray Microscopy (DFXM) is a synchrotron technique for imaging crystal defects with sub-micron resolution deep inside bulk materials. This package implements the geometrical-optics forward model described in Borgi et al., *Journal of Applied Crystallography*, 2024 (DOI: [10.1107/S1600576724001183](https://doi.org/10.1107/S1600576724001183)). Given a dislocation configuration and beamline geometry, it computes the expected detector image by convolving the direct-space deformation field around each defect with the reciprocal-space resolution function of the microscope. The default configuration targets the ID06 beamline at ESRF.

## Stack

- **Language:** Python 3.11+
- **Key libraries:** NumPy, SciPy (TOML-configured parameter sweeps)
- **HPC:** Batch templates for LSF (DTU Sophia) and SLURM (ESRF clusters)
- **Testing:** pytest with reference datasets
- **License:** MIT

## How to run

```bash
git clone https://github.com/borgi-s/Geometrical_Optics_master.git
cd Geometrical_Optics_master
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest                          # smoke tests against reference data

# One-time bootstrap (generates reciprocal-space resolution kernel, ~50 s):
dfxm-bootstrap configs/id06_edge.toml

# Forward simulation (run as many times as needed against the cached kernel):
dfxm-forward configs/id06_edge.toml
```

For large parameter sweeps, see `lsf/` and `slurm/` for ready-to-submit batch scripts.

## Background

Developed during PhD research at DTU Physics in collaboration with ESRF (European Synchrotron Radiation Facility). The accompanying paper is open-access via the IUCr. Reference datasets for full reproducibility are planned for Zenodo deposit (tracked in the Roadmap below).

## Roadmap

Next: package as a `pip install`-able PyPI distribution under a name TBD. Planned work includes refactoring to a `src/` layout, a `pyproject.toml`-based build, expanded test coverage with Zenodo reference data, and a GitHub Actions CI/CD release pipeline.
