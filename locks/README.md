# Lock files

  Exact-pin environment captures for fast cluster bring-up.
  Generated via
  `conda env export --no-builds`. Platform-specific (Linux-64
  only).

  - `environment-dtu-linux-64.yml` — DTU HPC working env,
  captured 2026-05-15

  Usage (skips the ~30-min solver):

      conda env create -f locks/environment-dtu-linux-64.yml

  For multi-platform reproducibility (paper runs, Zenodo
  deposit), generate
  a proper conda-lock file instead:

      pip install conda-lock
      conda-lock --file environment.yml -p linux-64 -p osx-64 -p
   win-64
