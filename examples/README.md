# Examples

Two complementary notebooks for the DFXM identification → ML workflow. To *run*
either, clone this repo and install the package (`pip install -e ".[dev]"`) — the
`.ipynb` alone is not enough. The notebooks are committed **output-stripped**
(`nbstripout`); rendered HTML exports (figures included, for read-only viewing
without running) are large and regenerable, so they are not committed — generate
them yourself (below) or ask for a copy.

## `identification_ml_tutorial/` — start here

`dfxm_identify_ml_tutorial.ipynb` is a self-contained tutorial: it builds a
resolution kernel, generates a tiny labeled dataset with `dfxm-identify`, shows
the HDF5 layout, the per-image labels your model trains on, the images
themselves, and how to scale to 100k+ images on a cluster. It regenerates all of
its own inputs, so a fresh clone runs it end to end (~2 min headless).

```bash
jupyter lab examples/identification_ml_tutorial/dfxm_identify_ml_tutorial.ipynb
# or render to HTML with outputs:
jupyter nbconvert --to html --execute \
  examples/identification_ml_tutorial/dfxm_identify_ml_tutorial.ipynb
```

## `cluster_showcase/` — a real sweep

`showcase.ipynb` visualizes the output of an actual cluster identify fan-out
(`scripts/fanout.py --mode identify` / `lsf/identify_array.bsub`). It loads a
`presentation_assets/` directory — detector frames, the per-frame rotations, and
a `summary.json` produced by the sweep — and plots them.

`presentation_assets/` is **gitignored** (a large, machine-specific cluster
output), so it is not in the repo. To reproduce the figures, run your own sweep
and drop the resulting `presentation_assets/` next to the notebook or at the repo
root — the notebook searches CWD then walks up. To render without re-running
(keeping the outputs already in the notebook):

```bash
jupyter nbconvert --to html examples/cluster_showcase/showcase.ipynb
```
