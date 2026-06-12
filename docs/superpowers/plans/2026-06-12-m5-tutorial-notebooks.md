# M5 Arc 1: Tutorial Notebooks 01–05 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship tutorial notebooks 01–05, `docs/references.md`, and a standalone notebook-CI workflow per the approved spec `docs/superpowers/specs/2026-06-12-m5-tutorial-notebooks-design.md`.

**Architecture:** Five flat, output-stripped `.ipynb` files under `examples/`, each self-contained (own out dir, idempotent smoke-kernel bootstrap where needed), each saving one committed preview PNG to `examples/img/`. References live in `docs/references.md` with stable anchors. CI executes 01–03 via `nbmake` in a new `.github/workflows/notebooks.yml`.

**Tech Stack:** dfxm_geo public API (`SimulationConfig`/`run_simulation`/`run_postprocess`/`load_identification_config`-via-CLI), h5py, matplotlib, nbformat (builder), nbmake (execution), GitHub Actions.

---

## Working rules (read first, apply to every task)

- **Workspace:** `C:\Users\borgi\Documents\GM-reworked\wt-m5-tutorial` on branch `feature/m5-tutorial-notebooks`. Python = `C:\Users\borgi\Documents\GM-reworked\wt-m5-tutorial\.venv\Scripts\python.exe` (call it `$PY` below; always use the full path).
- **NEVER touch:** `pyproject.toml`, `.github/workflows/ci.yml`, anything under `src/dfxm_geo/`, the main tree at `Geometrical_Optics_master\` (an M4 session owns it), or the shared `GM-reworked\.venv`. If a notebook exposes a production bug, STOP and report — do not fix on this branch.
- **Builder scripts** live OUTSIDE the repo in `C:\Users\borgi\Documents\GM-reworked\m5_scratch\` (created in Task 1). They are scratch, never committed. The committed artifact is the generated `.ipynb`.
- **Commits:** conventional prefixes (`docs:`, `feat:`, `ci:`, `chore:`). Pre-commit runs nbstripout; if a hook modifies a file, `git add` it again and re-commit.
- **All notebooks assume cwd = `examples/`** (`nbmake` starts the kernel in the notebook's own directory, so this holds in CI; locally run `jupyter lab` from `examples/`).
- Verified API facts used throughout (do not re-derive): empty TOML is a valid config; `run_simulation(cfg, out_dir)` returns `{"h5_path", "Hg", "q_hkl", "include_perfect_crystal"}` (single-reflection) or `{"n_reflections", "reflections"}` (multi); detector data at `/N.1/instrument/dfxm_sim_detector/data`; kernel lookup dir is `fm.pkl_fpath` (module attr, settable); scan ranges are RADIANS; analytic backend requires `beamstop = false` and needs NO kernel file; forward centered mode validates `t ∥ n×b` (edge-only).

### Builder pattern (used by Tasks 3–7)

Each notebook task writes a script `m5_scratch\build_NN.py` of this exact shape, then runs it:

```python
import sys
sys.path.insert(0, r"C:\Users\borgi\Documents\GM-reworked\m5_scratch")
from nbbuild import build

CELLS = [
    ("md", '''# Title ...'''),
    ("code", '''print("hello")'''),
]
build(CELLS, r"C:\Users\borgi\Documents\GM-reworked\wt-m5-tutorial\examples\NN_name.ipynb")
```

Convention: builder cell strings use `'''` delimiters; notebook code inside them uses `"""` for its own multi-line strings. Never nest same-style triple quotes.

---

### Task 1: Tooling + scaffolding

**Files:**
- Create: `C:\Users\borgi\Documents\GM-reworked\m5_scratch\nbbuild.py` (outside repo, not committed)
- Create: `examples/.gitignore`
- Create dir: `examples/img/` (materialized when first preview lands)

- [ ] **Step 1: Install nbmake into the worktree venv only** (NOT pyproject)

Run: `$PY -m pip install nbmake`
Expected: `Successfully installed nbmake-...` (pytest plugin; pulls nbclient).

- [ ] **Step 2: Write the builder helper**

Write `C:\Users\borgi\Documents\GM-reworked\m5_scratch\nbbuild.py`:

```python
"""Build a stripped .ipynb from (kind, source) tuples. Scratch tool — never committed."""
import nbformat as nbf


def build(cells, out_path):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    for kind, src in cells:
        c = nbf.v4.new_markdown_cell(src.strip()) if kind == "md" else nbf.v4.new_code_cell(src.strip())
        nb.cells.append(c)
    nbf.validate(nb)
    nbf.write(nb, out_path)
    print("wrote", out_path)
```

- [ ] **Step 3: Write `examples/.gitignore`**

```
out_0*/
kernel/
```

(Root `.gitignore` already ignores `*.h5` / `*.npz` / `*.npy` globally; this hides the notebooks' scratch dirs and the shared tutorial kernel. `examples/img/*.png` is intentionally NOT ignored — previews are committed.)

- [ ] **Step 4: Sanity-check nbbuild + nbmake round-trip**

Run (from the worktree root):
```
$PY -c "import sys; sys.path.insert(0, r'C:\Users\borgi\Documents\GM-reworked\m5_scratch'); from nbbuild import build; build([('code', 'print(1+1)')], 'examples/_smoke.ipynb')"
$PY -m pytest --nbmake examples/_smoke.ipynb -q
```
Expected: `1 passed`. Then delete: `rm examples/_smoke.ipynb`.

- [ ] **Step 5: Commit**

```bash
git add examples/.gitignore
git commit -m "chore: gitignore tutorial scratch dirs under examples/"
```

---

### Task 2: `docs/references.md` + two spec amendments

**Files:**
- Create: `docs/references.md`
- Modify: `docs/superpowers/specs/2026-06-12-m5-tutorial-notebooks-design.md` (§3 row for 03, §4)

- [ ] **Step 1: Verify the four uncertain citations by web lookup**

Use WebSearch/WebFetch to confirm exact fields (title, journal, volume, pages, year, DOI) for:
1. **Borgi 2025** — known: Borgi, Winther, Poulsen, "Individual dislocation identification in dark-field X-ray microscopy", J. Appl. Cryst. 58, 813–821 (2025). Find the DOI.
2. **Poulsen 2021** — Poulsen et al., J. Appl. Cryst. 54, 1555 (2021) — geometrical-optics DFXM formalism. Confirm full author list, title, page range, DOI.
3. **Poulsen 2017** — Poulsen et al., J. Appl. Cryst. (2017) — DFXM resolution-function/optics paper (basis of `reciprocal_res_func`; see header of `src/dfxm_geo/reciprocal_space/resolution.py` for the exact reference — read that file header first, it may pin volume/pages). Confirm DOI.
4. **darkmod paper** — arXiv:2503.22022 (GO-family DFXM code `darkmod`; quoted in `docs/recovery-plan-darkmod-vs-go.md:19`). Get authors + title from arXiv.

Already-verified entries (use as-is): **Borgi 2024** = Borgi, S., Ræder, T. M., Carlsen, M. A., Detlefs, C., Winther, G., & Poulsen, H. F. (2024). Simulations of dislocation contrast in dark-field X-ray microscopy. J. Appl. Cryst. 57, 358–368. doi:10.1107/S1600576724001183. **Hirth & Lothe** = Theory of Dislocations, 2nd ed., Wiley (1982). **Carlsen 2022** = arXiv:2201.07549 (dynamical/wave-optics DFXM model; also the source for the ±140 µrad condenser-aperture clip — see `docs/model-taxonomy.md:39,68`).

- [ ] **Step 2: Write `docs/references.md`**

Structure (fill the verified fields from Step 1; BibTeX keys = anchor slugs):

```markdown
# References

Every physics claim in the tutorial notebooks (`examples/01–05`) links here.

## Cited works

### <a id="borgi-2024"></a>Borgi 2024 — the forward model
Role: the dislocation-contrast forward model this package implements (two-stage GO + resolution kernel).
```bibtex
@article{borgi2024,
  author  = {Borgi, S. and R{\ae}der, T. M. and Carlsen, M. A. and Detlefs, C. and Winther, G. and Poulsen, H. F.},
  title   = {Simulations of dislocation contrast in dark-field X-ray microscopy},
  journal = {Journal of Applied Crystallography},
  volume  = {57},
  pages   = {358--368},
  year    = {2024},
  doi     = {10.1107/S1600576724001183},
}
```

### <a id="borgi-2025"></a>Borgi 2025 — identification
[same pattern]

### <a id="poulsen-2021"></a>Poulsen 2021 — the original MATLAB GO model
[same pattern]

### <a id="poulsen-2017"></a>Poulsen 2017 — DFXM optics & resolution function
[same pattern]

### <a id="hirth-lothe-1982"></a>Hirth & Lothe 1982 — dislocation displacement fields
[@book entry]

### <a id="carlsen-2022"></a>Carlsen 2022 — wave-optics DFXM (and the condenser aperture)
[@misc arXiv entry]

### <a id="darkmod"></a>darkmod — independent GO-family implementation
[@misc arXiv:2503.22022 entry]

## Where each is used
| Reference | Notebooks | Docs |
|---|---|---|
| Borgi 2024 | 01, 02, 03, 04 | README, physics.md |
| Borgi 2025 | 04 (g·b criterion), 05 | README |
| Poulsen 2021 | 02 | README |
| Poulsen 2017 | 02 | physics.md |
| Hirth & Lothe 1982 | 03 | physics.md |
| Carlsen 2022 | 02 | model-taxonomy.md, resolution-backend-comparison.md |
| darkmod | — | model-taxonomy.md, recovery-plan-darkmod-vs-go.md |
```

(The bracketed "[same pattern]" lines above are instructions to YOU, the executor: write a complete entry for each, in the identical role-line + bibtex-block form, using Step 1's verified fields. No placeholders may survive into the committed file.)

- [ ] **Step 3: Amend the spec (two fixes)**

In `docs/superpowers/specs/2026-06-12-m5-tutorial-notebooks-design.md`:
1. §4: replace "and the darkmod comparison source (Carlsen et al.)" with "Carlsen et al. 2022 (arXiv:2201.07549, wave-optics model + condenser-aperture provenance) and the darkmod paper (arXiv:2503.22022, the GO-family comparison code)" — the original line conflated two different works.
2. §3 row for 03: append to the Content cell: "mixed character via the identification engine's α sweep (forward centered mode is edge-only by validation; α↔edge/screw endpoint naming deliberately left open — pending decision)".

- [ ] **Step 4: Commit**

```bash
git add docs/references.md docs/superpowers/specs/2026-06-12-m5-tutorial-notebooks-design.md
git commit -m "docs: references page with BibTeX + anchors; spec fixes (darkmod/Carlsen split, 03 mixed-character note)"
```

---

### Task 3: Notebook 01 — quickstart

**Files:**
- Create: `m5_scratch\build_01.py` (scratch)
- Create: `examples/01_quickstart.ipynb`
- Create: `examples/img/01_quickstart_preview.png` (by executing the notebook)

- [ ] **Step 1: Write `m5_scratch\build_01.py`** with these CELLS (builder pattern from the preamble):

```python
CELLS = [
    ("md", '''
# 01 · Quickstart — a first DFXM image in five lines

Dark-field X-ray microscopy (DFXM) maps lattice distortion deep inside crystals by imaging a single Bragg reflection through an objective lens. `dfxm_geo` simulates it in two stages: a reciprocal-space **resolution kernel** (which local q-deviations the instrument accepts) and a direct-space **geometrical-optics pass** (what each detector pixel collects through that acceptance). Physics: [Borgi 2024](../docs/references.md#borgi-2024) §2; all citations resolve on the [references page](../docs/references.md).
'''),
    ("md", '''
![DFXM optics schematic](../docs/img/fig_3_1_microscope_schematic.png)

Stage 1 runs once per reflection (`dfxm-bootstrap`: Monte-Carlo ray sampling → an `.npz` lookup table); stage 2 runs per frame (`dfxm-forward` / `run_simulation`). The setup cell builds a small tutorial kernel (2 M rays, seconds; production uses 10⁸) into `examples/kernel/` and points the in-process lookup there.
'''),
    ("code", '''
%matplotlib inline
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

import dfxm_geo
import dfxm_geo.direct_space.forward_model as fm

HERE = Path.cwd()
assert HERE.name == "examples", "Run this notebook from the examples/ folder"
IMG, OUT, KERNEL_DIR = HERE / "img", HERE / "out_01", HERE / "kernel"
for d in (IMG, OUT, KERNEL_DIR):
    d.mkdir(exist_ok=True)
fm.pkl_fpath = KERNEL_DIR.resolve().as_posix() + "/"  # kernel lookup -> examples/kernel/

KERNEL_FILE = KERNEL_DIR / "Resq_i_h-1_k1_l-1_17keV_tutorial.npz"
if not KERNEL_FILE.exists():
    boot = OUT / "bootstrap.toml"
    boot.write_text("[reciprocal]\\nNrays = 2_000_000\\nseed = 0\\nbeamstop = false\\n", encoding="utf-8")
    subprocess.run(
        [sys.executable, "-c",
         "from dfxm_geo.reciprocal_space.kernel import cli_main; cli_main()",
         "--config", str(boot), "--output", str(KERNEL_FILE)],
        check=True,
    )
print("dfxm_geo", dfxm_geo.__version__, "| kernel:", KERNEL_FILE.name)
'''),
    ("md", '''
## An empty TOML is a valid experiment

Since v2.0.0 every config key has a physically sensible default: one edge dislocation centred in single-crystal aluminium, imaged on the (−1, 1, −1) reflection at 17 keV. An empty file runs out of the box; any key you write overrides exactly one default.
'''),
    ("code", '''
from dfxm_geo.pipeline import SimulationConfig, run_simulation

(OUT / "quickstart.toml").write_text("", encoding="utf-8")
cfg = SimulationConfig.from_toml(OUT / "quickstart.toml")
result = run_simulation(cfg, OUT)
print("wrote", result["h5_path"])
'''),
    ("md", '''
## What got written

`run_simulation` produced a BLISS-style master file with one NXentry per scan — `/1.1` is the dislocated crystal, `/2.1` the perfect-crystal reference — with detector frames in per-scan files reached through HDF5 external links. Layout details: [docs/output-format.md](../docs/output-format.md).
'''),
    ("code", '''
with h5py.File(result["h5_path"], "r") as f:
    disloc = f["/1.1/instrument/dfxm_sim_detector/data"][0]
    perfect = f["/2.1/instrument/dfxm_sim_detector/data"][0]

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, im, title in [(axes[0], disloc, "centred edge dislocation"),
                      (axes[1], perfect, "perfect crystal")]:
    ax.imshow(im, cmap="magma", norm=PowerNorm(0.45))
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(IMG / "01_quickstart_preview.png", dpi=110, bbox_inches="tight")
'''),
    ("md", '''
## Next

[02 · Reciprocal space](02_reciprocal_space.ipynb) — what the kernel is. [03 · Dislocations & contrast](03_dislocations_and_contrast.ipynb) — character, weak beam, COM ≈ −qi. [04 · Oblique & reflections](04_oblique_and_reflections.ipynb) — mounts, reflection tables, g·b invisibility. [05 · Identification at scale](05_identification_at_scale.ipynb) — labelled ML datasets and throughput.
'''),
]
```

- [ ] **Step 2: Build it**

Run: `$PY C:\Users\borgi\Documents\GM-reworked\m5_scratch\build_01.py`
Expected: `wrote ...examples\01_quickstart.ipynb`

- [ ] **Step 3: Execute via nbmake**

Run (worktree root): `$PY -m pytest --nbmake --nbmake-timeout=600 examples/01_quickstart.ipynb -q`
Expected: `1 passed` (first run includes kernel bootstrap + numba JIT; allow ~2–4 min).
If it FAILS: read the traceback, fix the builder CELLS, rebuild, re-run. Do not hand-edit the ipynb.

- [ ] **Step 4: Inspect the preview**

Read `examples/img/01_quickstart_preview.png` (image read). Expected: left panel shows a dislocation contrast feature; right panel near-uniform. If the left panel is blank/black, something is wrong — STOP and investigate (likely kernel mismatch), do not commit.

- [ ] **Step 5: Commit**

```bash
git add examples/01_quickstart.ipynb examples/img/01_quickstart_preview.png
git commit -m "feat: tutorial notebook 01 - quickstart (empty TOML to first image)"
```

---

### Task 4: Notebook 02 — reciprocal space

**Files:**
- Create: `m5_scratch\build_02.py`, `examples/02_reciprocal_space.ipynb`, `examples/img/02_reciprocal_space_preview.png`

- [ ] **Step 1: Write `m5_scratch\build_02.py`** with CELLS:

```python
CELLS = [
    ("md", '''
# 02 · Reciprocal space — the resolution kernel

The kernel Res(**q**) is the instrument's acceptance: the probability that a ray scattered with reciprocal-space deviation **q** = (q₁, q₂, q₃) clears condenser, objective and bandwidth to reach the detector. `dfxm-bootstrap` estimates it by Monte-Carlo ray sampling ([Poulsen 2021](../docs/references.md#poulsen-2021), [Poulsen 2017](../docs/references.md#poulsen-2017)); the forward model then weights every voxel's local q against this table ([Borgi 2024](../docs/references.md#borgi-2024) §3).
'''),
    ("code", '''
%matplotlib inline
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import dfxm_geo.direct_space.forward_model as fm

HERE = Path.cwd()
assert HERE.name == "examples", "Run this notebook from the examples/ folder"
IMG, OUT, KERNEL_DIR = HERE / "img", HERE / "out_02", HERE / "kernel"
for d in (IMG, OUT, KERNEL_DIR):
    d.mkdir(exist_ok=True)
fm.pkl_fpath = KERNEL_DIR.resolve().as_posix() + "/"

KERNEL_FILE = KERNEL_DIR / "Resq_i_h-1_k1_l-1_17keV_tutorial.npz"
if not KERNEL_FILE.exists():
    boot = OUT / "bootstrap.toml"
    boot.write_text("[reciprocal]\\nNrays = 2_000_000\\nseed = 0\\nbeamstop = false\\n", encoding="utf-8")
    subprocess.run(
        [sys.executable, "-c",
         "from dfxm_geo.reciprocal_space.kernel import cli_main; cli_main()",
         "--config", str(boot), "--output", str(KERNEL_FILE)],
        check=True,
    )
print("kernel:", KERNEL_FILE.name)
'''),
    ("md", '''
## Inside the npz

The kernel ships its full provenance — reflection, energy, Bragg angle, ray count, grid geometry and instrument parameters — alongside the 3-D lookup table itself.
'''),
    ("code", '''
k = np.load(KERNEL_FILE)
for key in ("hkl", "keV", "theta", "Nrays", "npoints1", "npoints2", "npoints3",
            "qi1_range", "qi2_range", "qi3_range", "beamstop", "seed"):
    print(f"{key:12s} = {k[key]}")
R = k["Resq_i"]
print("LUT shape:", R.shape, "| nonzero voxels:", int((R > 0).sum()))
'''),
    ("md", '''
## The acceptance as a point cloud

Scattering the occupied voxels shows the familiar elongated acceptance volume (compare `docs/img/reciprocal_pointcloud_50k.png`, rendered at 10⁸ rays).
'''),
    ("code", '''
q1 = np.linspace(-k["qi1_range"], k["qi1_range"], R.shape[0])
q2 = np.linspace(-k["qi2_range"], k["qi2_range"], R.shape[1])
q3 = np.linspace(-k["qi3_range"], k["qi3_range"], R.shape[2])
idx = np.argwhere(R > 0.01 * R.max())
rng = np.random.default_rng(0)
idx = idx[rng.choice(len(idx), size=min(50_000, len(idx)), replace=False)]
w = R[idx[:, 0], idx[:, 1], idx[:, 2]]

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(projection="3d")
ax.scatter(q1[idx[:, 0]], q2[idx[:, 1]], q3[idx[:, 2]], c=w, s=1, cmap="viridis", alpha=0.3)
ax.set_xlabel("$q_1$")
ax.set_ylabel("$q_2$")
ax.set_zlabel("$q_3$")
fig.savefig(IMG / "02_reciprocal_space_preview.png", dpi=110, bbox_inches="tight")
'''),
    ("md", '''
## MC kernel vs the closed-form (analytic) backend

For the no-beamstop regime there is also a closed-form backend (v2.1.0) — same physics, zero sampling noise, no kernel file; the ±140 µrad vertical-divergence clip both backends apply is the condenser's physical aperture ([Carlsen 2022](../docs/references.md#carlsen-2022)). At only 2 M rays the MC center-of-mass maps are visibly noisier — that is the trade, and `[reciprocal] backend` is the switch. See [docs/resolution-backend-comparison.md](../docs/resolution-backend-comparison.md).
'''),
    ("code", '''
from dfxm_geo.analysis.mosaicity import compute_com_maps
from dfxm_geo.io.hdf5 import load_h5_scan
from dfxm_geo.pipeline import SimulationConfig, run_simulation

BASE = """
[scan.phi]
range = 6e-4
steps = 9

[scan.chi]
range = 2e-3
steps = 9

[io]
include_perfect_crystal = false
write_strain_provenance = false

[postprocess]
enabled = false
"""
com = {}
for backend in ("mc", "analytic"):
    cfg_file = OUT / f"mosa_{backend}.toml"
    cfg_file.write_text(f'[reciprocal]\\nbackend = "{backend}"\\nbeamstop = false\\n' + BASE,
                        encoding="utf-8")
    res = run_simulation(SimulationConfig.from_toml(cfg_file), OUT / backend)
    _, stack, h, w_ = load_h5_scan(res["h5_path"], phi_steps=9, chi_steps=9)
    com[backend] = compute_com_maps(stack, phi_range=6e-4, phi_steps=9,
                                    chi_range=2e-3, chi_steps=9)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
vmax = 1e-4
im0 = axes[0].imshow(com["mc"][0], vmin=-vmax, vmax=vmax, cmap="RdBu_r")
axes[0].set_title("φ-COM, MC kernel (2 M rays)")
axes[1].imshow(com["analytic"][0], vmin=-vmax, vmax=vmax, cmap="RdBu_r")
axes[1].set_title("φ-COM, analytic backend")
axes[2].imshow(com["mc"][0] - com["analytic"][0], vmin=-vmax, vmax=vmax, cmap="RdBu_r")
axes[2].set_title("difference (MC sampling noise)")
for ax in axes:
    ax.axis("off")
fig.colorbar(im0, ax=axes, shrink=0.8, label="rad")
'''),
    ("md", '''
Production kernels (10⁸ rays) drive that difference toward zero — the parity study lives in [docs/resolution-backend-comparison.md](../docs/resolution-backend-comparison.md). Next: [03 · Dislocations & contrast](03_dislocations_and_contrast.ipynb).
'''),
]
```

- [ ] **Step 2: Build:** `$PY m5_scratch\build_02.py` → `wrote ...02_reciprocal_space.ipynb`
- [ ] **Step 3: Execute:** `$PY -m pytest --nbmake --nbmake-timeout=900 examples/02_reciprocal_space.ipynb -q` → `1 passed` (~3–5 min: 81 frames × 2 backends).
- [ ] **Step 4: Inspect** `examples/img/02_reciprocal_space_preview.png` — expect a 3-D elongated point cloud.
- [ ] **Step 5: Commit**

```bash
git add examples/02_reciprocal_space.ipynb examples/img/02_reciprocal_space_preview.png
git commit -m "feat: tutorial notebook 02 - resolution kernel + MC vs analytic"
```

---

### Task 5: Notebook 03 — dislocations and contrast

**Files:**
- Create: `m5_scratch\build_03.py`, `examples/03_dislocations_and_contrast.ipynb`, `examples/img/03_dislocations_preview.png`

Entirely analytic backend — **no kernel needed**. Three sections: (a) mixed-character α sweep via the identification engine, (b) weak-beam rocking frames, (c) the COM ≈ −qi README figure made executable.

**PHYSICS CAUTION:** do NOT label α endpoints as "edge" or "screw" — that naming is an open design question (deferred #8 rotation_deg). Call α "the line-direction rotation in the slip plane"; state only that the forward default `t = n×b` is the pure edge case.

- [ ] **Step 1: Write `m5_scratch\build_03.py`** with CELLS:

```python
CELLS = [
    ("md", '''
# 03 · Dislocations and contrast

A dislocation bends the lattice around its core, so nearby voxels diffract at slightly different angles — that misorientation field is what DFXM images. Displacement fields are the classical isotropic-elasticity expressions ([Hirth & Lothe 1982](../docs/references.md#hirth-lothe-1982)); the imaging physics is [Borgi 2024](../docs/references.md#borgi-2024) Figs. 3–5. Everything below uses the closed-form resolution backend, so this notebook needs no kernel file.
'''),
    ("code", '''
%matplotlib inline
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

HERE = Path.cwd()
assert HERE.name == "examples", "Run this notebook from the examples/ folder"
IMG, OUT = HERE / "img", HERE / "out_03"
for d in (IMG, OUT):
    d.mkdir(exist_ok=True)
'''),
    ("md", '''
## Dislocation character: the α sweep

The identification engine sweeps the dislocation line direction by an in-plane rotation α (the `rotation_deg` label in the HDF5 output); the forward default `t = n×b` is the pure edge configuration. Same Burgers vector, same (111) slip plane — only α changes, and the contrast morphology changes with it.
'''),
    ("code", '''
ID_TOML = """
mode = "single"

[reciprocal]
backend = "analytic"
beamstop = false

[crystal]
slip_plane_normal = [1, 1, 1]
sweep_all_slip_planes = false
b_vector_indices = [0]
angle_start_deg = 0.0
angle_stop_deg = 90.0
angle_step_deg = 45.0
exclude_invisibility = false

[scan.phi]
value = 1.25e-4

[noise]
poisson_noise = false

[io]
include_perfect_crystal = false
write_strain_provenance = false
"""
cfg_file = OUT / "alpha_sweep.toml"
cfg_file.write_text(ID_TOML, encoding="utf-8")
subprocess.run(
    [sys.executable, "-c",
     "from dfxm_geo.pipeline import cli_main_identify; cli_main_identify()",
     "--config", str(cfg_file), "--output", str(OUT / "alpha")],
    check=True,
)

alphas = [0, 45, 90]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
with h5py.File(OUT / "alpha" / "dfxm_identify.h5", "r") as f:
    for ax, scan, a in zip(axes, ("1.1", "2.1", "3.1"), alphas):
        ax.imshow(f[f"/{scan}/instrument/dfxm_sim_detector/data"][0],
                  cmap="magma", norm=PowerNorm(0.45))
        ax.set_title(f"α = {a}°" + ("  (pure edge)" if a == 0 else ""))
        ax.axis("off")
fig.tight_layout()
'''),
    ("md", '''
## Weak-beam contrast

Offsetting the goniometer from the Bragg peak suppresses the far-field background and leaves the steep strain near the core — frames far out on the rocking curve show the sparse bright-core "weak beam" look exploited by the identification pipeline.
'''),
    ("code", '''
from dfxm_geo.pipeline import SimulationConfig, run_simulation

ROCK_TOML = """
[reciprocal]
backend = "analytic"
beamstop = false

[scan.phi]
range = 1.25e-4
steps = 5

[io]
include_perfect_crystal = false
write_strain_provenance = false

[postprocess]
enabled = false
"""
cfg_file = OUT / "rocking.toml"
cfg_file.write_text(ROCK_TOML, encoding="utf-8")
res = run_simulation(SimulationConfig.from_toml(cfg_file), OUT / "rocking")
with h5py.File(res["h5_path"], "r") as f:
    frames = f["/1.1/instrument/dfxm_sim_detector/data"][...]

fig, axes = plt.subplots(1, frames.shape[0], figsize=(15, 3.2))
phis = np.linspace(-1.25e-4, 1.25e-4, frames.shape[0])
for ax, im, p in zip(axes, frames, phis):
    ax.imshow(im, cmap="magma", norm=PowerNorm(0.45))
    ax.set_title(f"φ = {p * 1e6:+.0f} µrad")
    ax.axis("off")
fig.tight_layout()
'''),
    ("md", '''
## COM ≈ −qi: the README figure, executable

For a wall of edge dislocations, the φ/χ centre-of-mass maps of a mosaicity scan reproduce (negated) the geometrical-optics qi fields at the sample plane — the self-consistency result of [Borgi 2024](../docs/references.md#borgi-2024). This is the repo README figure; scan ranges are the article's (φ ±600 µrad, χ ±2 mrad — radians in the TOML).
'''),
    ("code", '''
from dfxm_geo.pipeline import run_postprocess

WALL_TOML = """
[reciprocal]
backend = "analytic"
beamstop = false

[crystal]
mode = "wall"

[crystal.wall]
dis = 4
ndis = 151
sample_remount = "S1"

[scan.phi]
range = 6e-4
steps = 11

[scan.chi]
range = 2e-3
steps = 11

[io]
include_perfect_crystal = true

[postprocess]
enabled = true
"""
cfg_file = OUT / "wall.toml"
cfg_file.write_text(WALL_TOML, encoding="utf-8")
cfg = SimulationConfig.from_toml(cfg_file)
run_dir = OUT / "wall"
run_simulation(cfg, run_dir)
res = run_postprocess(run_dir, cfg)
phi_list, chi_list, qi_field = res["phi_list"], res["chi_list"], res["qi_field"]
'''),
    ("code", '''
v = 1e-4
panels = [
    ("$-COM_\\\\varphi$", -phi_list.T), ("$-COM_\\\\chi$", -chi_list.T),
    ("$qi_1$", qi_field[..., 0].T),     ("$qi_2$", qi_field[..., 1].T),
]
fig, axes = plt.subplots(2, 2, figsize=(9, 8))
for ax, (title, data) in zip(axes.ravel(), panels):
    pc = ax.imshow(data, vmin=-v, vmax=v, cmap="RdBu_r")
    ax.set_title(title)
    ax.axis("off")
fig.colorbar(pc, ax=axes, shrink=0.8, label="rad")
fig.savefig(IMG / "03_dislocations_preview.png", dpi=110, bbox_inches="tight")
'''),
    ("md", '''
Each top panel reproduces the one beneath it — the measured mosaicity maps recover the model's own input distortion field. Next: [04 · Oblique & reflections](04_oblique_and_reflections.ipynb).
'''),
]
```

**NOTE for executor (panel orientation):** the −COM/transpose convention above mirrors `scripts/render_readme_examples.py` lines 159–171 (`plot_mosaicity_maps` convention: COM maps negated + transposed so −COM_φ ≈ qi₁, −COM_χ ≈ qi₂). If the panels come out mirrored vs `docs/img/com_qi_comparison.png`, read that script's exact plotting block and match it; the README PNG is ground truth.

- [ ] **Step 2: Build:** `$PY m5_scratch\build_03.py`
- [ ] **Step 3: Execute:** `$PY -m pytest --nbmake --nbmake-timeout=900 examples/03_dislocations_and_contrast.ipynb -q` → `1 passed` (~3–6 min; the 11×11 wall mosa with perfect-crystal reference dominates).
- [ ] **Step 4: Inspect** `examples/img/03_dislocations_preview.png` against `docs/img/com_qi_comparison.png` — same 2×2 structure, top row visually matching bottom row.
- [ ] **Step 5: Commit**

```bash
git add examples/03_dislocations_and_contrast.ipynb examples/img/03_dislocations_preview.png
git commit -m "feat: tutorial notebook 03 - character, weak beam, COM=-qi reproduction"
```

---

### Task 6: Notebook 04 — oblique mounts and reflections

**Files:**
- Create: `m5_scratch\build_04.py`, `examples/04_oblique_and_reflections.ipynb`, `examples/img/04_reflections_preview.png`

Config is lifted verbatim from `tests/test_gb_invisibility_physics.py` (`_GB_INVISIBILITY_TOML`) — analytic backend, **no kernels**. Measured reference: contrast ratio g=200 vs g=111 ≈ 3.55× at steps=5.

- [ ] **Step 1: Write `m5_scratch\build_04.py`** with CELLS:

```python
CELLS = [
    ("md", '''
# 04 · Oblique mounts, reflection tables, and g·b invisibility

The simplified geometry hard-wires the (−1,1,−1)@17 keV mount; oblique mode (v2.3.0) takes any cubic mount (`mount_x/y/z`) and any Laue-reachable reflection, and multi-reflection configs (v2.6.0) sweep several reflections in one run. The classic kinematical criterion says a dislocation vanishes when g·b = 0 ([Borgi 2025](../docs/references.md#borgi-2025); Borgi 2024 App. A) — below we watch it vanish.
'''),
    ("code", '''
%matplotlib inline
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

HERE = Path.cwd()
assert HERE.name == "examples", "Run this notebook from the examples/ folder"
IMG, OUT = HERE / "img", HERE / "out_04"
for d in (IMG, OUT):
    d.mkdir(exist_ok=True)

CONFIG = """
[reciprocal]
keV = 19.1
backend = "analytic"
beamstop = false

[geometry]
mode = "oblique"

[crystal]
lattice = "cubic"
a       = 4.0493e-10
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]
mode = "centered"

[crystal.centered]
b = [-1, 1, 0]
n = [1,  1, 1]
t = [-1, -1, 2]

[scan.phi]
value = 0.0
range = 1.25e-4
steps = 5

[io]
include_perfect_crystal = true

[postprocess]
enabled = false

[[reflections]]
hkl = [1, 1, 1]

[[reflections]]
hkl = [2, 0, 0]
"""
cfg_file = OUT / "gb_invisibility.toml"
cfg_file.write_text(CONFIG, encoding="utf-8")
print(CONFIG)
'''),
    ("md", '''
## Which reflections can this mount even reach?

`dfxm-find-reflections` enumerates Laue-reachable reflections for a mount and energy, with Bragg angle and the two goniometer solutions per reflection.
'''),
    ("code", '''
table = subprocess.run(
    [sys.executable, "-c",
     "from dfxm_geo.find_reflections_cmd import cli_main; cli_main()",
     "--config", str(cfg_file), "--hkl-max", "2"],
    check=True, capture_output=True, text=True,
)
print(table.stdout)
'''),
    ("md", '''
## One run, two reflections, one invisible dislocation

The config's `[[reflections]]` blocks request g = (1,1,1) and g = (2,0,0) for the same b = ½[−110] dislocation: g·b = 0 for the first (invisible), g·b = −2 for the second (visible). The run writes one super-master with a per-reflection sub-tree each.
'''),
    ("code", '''
from dfxm_geo.crystal.burgers import gb_cos
from dfxm_geo.pipeline import SimulationConfig, run_simulation

cfg = SimulationConfig.from_toml(cfg_file)
result = run_simulation(cfg, OUT / "run")
print("reflections:", result["n_reflections"])

b = np.array([-1, 1, 0])
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
contrast = {}
for i, (ax, g) in enumerate(zip(axes, ([1, 1, 1], [2, 0, 0])), start=1):
    master = OUT / "run" / f"reflection_{i:03d}" / "dfxm_geo.h5"
    with h5py.File(master, "r") as f:
        strained = f["/1.1/instrument/dfxm_sim_detector/data"][2]
        perfect = f["/2.1/instrument/dfxm_sim_detector/data"][2]
    c = float(np.std(strained - perfect) / max(np.mean(perfect), 1e-10))
    contrast[tuple(g)] = c
    gb = int(np.dot(g, b))
    ax.imshow(strained, cmap="magma", norm=PowerNorm(0.45))
    ax.set_title(f"g={g}  g·b={gb}  contrast={c:.3f}")
    ax.axis("off")
fig.tight_layout()
fig.savefig(IMG / "04_reflections_preview.png", dpi=110, bbox_inches="tight")

ratio = contrast[(2, 0, 0)] / contrast[(1, 1, 1)]
print(f"contrast ratio (visible/invisible) = {ratio:.2f}")
assert ratio > 3.0, "g·b invisibility broke — expected >3x (measured 3.55x in the test suite)"
'''),
    ("md", '''
The g·b = 0 panel keeps only faint residual contrast (the criterion is exact for screw dislocations and approximate for edge), while g·b = −2 shows the full dislocation — the suite pins this ratio at ≈3.6×. The same `[[reflections]]` mechanism drives whole sweep campaigns; `exclude_invisibility` uses g·b to skip blind configurations in identification. Next: [05 · Identification at scale](05_identification_at_scale.ipynb).
'''),
]
```

(The `[2]` frame index = centre of the 5-step rocking curve. `gb_cos` is imported to make the API discoverable even though the inline dot product carries the demo; if `from dfxm_geo.crystal.burgers import gb_cos` fails, check the module path with Grep before changing anything.)

- [ ] **Step 2: Build:** `$PY m5_scratch\build_04.py`
- [ ] **Step 3: Execute:** `$PY -m pytest --nbmake --nbmake-timeout=900 examples/04_oblique_and_reflections.ipynb -q` → `1 passed` (~2–4 min: 2 reflections × 5 frames × 2 crystals, analytic).
- [ ] **Step 4: Inspect** `examples/img/04_reflections_preview.png` — left panel near-blank, right panel clear dislocation, titles carry g·b values.
- [ ] **Step 5: Commit**

```bash
git add examples/04_oblique_and_reflections.ipynb examples/img/04_reflections_preview.png
git commit -m "feat: tutorial notebook 04 - oblique mount, find-reflections, g.b invisibility"
```

---

### Task 7: Notebook 05 — identification at scale

**Files:**
- Create: `m5_scratch\build_05.py`, `examples/05_identification_at_scale.ipynb`, `examples/img/05_identification_preview.png`

Slim re-cut of `examples/identification_ml_tutorial/` (which stays the deep-dive; link it) + the throughput story from `docs/cluster-profiling.md`. Needs the MC kernel (identify multi runs the production MC path).

- [ ] **Step 1: Write `m5_scratch\build_05.py`** with CELLS:

```python
CELLS = [
    ("md", '''
# 05 · Identification at scale — labelled data for ML

`dfxm-identify` inverts the tutorial so far: instead of one known dislocation, it renders *labelled* images across slip systems, characters and positions — training data for identification models ([Borgi 2025](../docs/references.md#borgi-2025)). This is the slim cut; the full walkthrough is [identification_ml_tutorial](identification_ml_tutorial/dfxm_identify_ml_tutorial.ipynb).
'''),
    ("code", '''
%matplotlib inline
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

import dfxm_geo.direct_space.forward_model as fm

HERE = Path.cwd()
assert HERE.name == "examples", "Run this notebook from the examples/ folder"
IMG, OUT, KERNEL_DIR = HERE / "img", HERE / "out_05", HERE / "kernel"
for d in (IMG, OUT, KERNEL_DIR):
    d.mkdir(exist_ok=True)
fm.pkl_fpath = KERNEL_DIR.resolve().as_posix() + "/"

KERNEL_FILE = KERNEL_DIR / "Resq_i_h-1_k1_l-1_17keV_tutorial.npz"
if not KERNEL_FILE.exists():
    boot = OUT / "bootstrap.toml"
    boot.write_text("[reciprocal]\\nNrays = 2_000_000\\nseed = 0\\nbeamstop = false\\n", encoding="utf-8")
    subprocess.run(
        [sys.executable, "-c",
         "from dfxm_geo.reciprocal_space.kernel import cli_main; cli_main()",
         "--config", str(boot), "--output", str(KERNEL_FILE)],
        check=True,
    )

# Subprocesses don't inherit fm.pkl_fpath — prefix it into their bootstrap.
KERNEL_ENV = "import dfxm_geo.direct_space.forward_model as fm; fm.pkl_fpath = %r; " % (
    KERNEL_DIR.resolve().as_posix() + "/"
)
'''),
    ("md", '''
## A tiny labelled dataset

`mode = "multi"` drops two random dislocations per sample, renders the noisy combined image plus a noiseless per-dislocation render each (`render_per_dislocation`), and stores the labels (slip plane, Burgers vector, rotation, position) next to the images. Four samples are enough to see the shape of the data.
'''),
    ("code", '''
ID_TOML = """
mode = "multi"

[reciprocal]
hkl = [-1, 1, -1]
keV = 17.0

[scan.phi]
value = 1.25e-4

[noise]
poisson_noise = true
rng_seed = 3
intensity_scale = 7.0

[multi]
n_samples = 4
pos_std_um = 5.0
render_per_dislocation = true

[io]
include_perfect_crystal = false
write_strain_provenance = false
"""
cfg_file = OUT / "identify_multi.toml"
cfg_file.write_text(ID_TOML, encoding="utf-8")
master = OUT / "identify" / "dfxm_identify.h5"
if not master.exists():
    subprocess.run(
        [sys.executable, "-c",
         KERNEL_ENV + "from dfxm_geo.pipeline import cli_main_identify; cli_main_identify()",
         "--config", str(cfg_file), "--output", str(OUT / "identify"), "--mode", "multi"],
        check=True,
    )
print("master:", master)
'''),
    ("code", '''
fig, axes = plt.subplots(4, 3, figsize=(9, 12))
with h5py.File(master, "r") as f:
    scans = sorted(k for k in f.keys() if k.endswith(".1"))[:4]
    for row, scan in zip(axes, scans):
        combined = f[f"/{scan}/instrument/dfxm_sim_detector/data"][0]
        row[0].imshow(combined, cmap="magma", norm=PowerNorm(0.45))
        row[0].set_ylabel(scan)
        labels = []
        for di in sorted(f[f"/{scan}/dislocations"].keys()):
            g = f[f"/{scan}/dislocations/{di}"]
            n_ = g["slip_plane_normal"][...]
            b_ = g["burgers"][...]
            labels.append(f"n={n_.tolist()} b={b_.tolist()} α={float(g['rotation_deg'][()]):.0f}°")
        for col, di, lab in zip(row[1:], ("dis0", "dis1"), labels):
            solo = OUT / "identify" / f"scan{scans.index(scan) + 1:04d}" / f"dfxm_sim_detector_{di}_0000.h5"
            with h5py.File(solo, "r") as fs:
                col.imshow(fs["/entry_0000/dfxm_sim_detector/image"][0],
                           cmap="magma", norm=PowerNorm(0.45))
            col.set_title(lab, fontsize=7)
        for ax in row:
            ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("combined (noisy) | per-dislocation renders with labels")
fig.tight_layout()
fig.savefig(IMG / "05_identification_preview.png", dpi=110, bbox_inches="tight")
'''),
    ("md", '''
## Throughput: how 100k images become practical

Per-config cost is dominated by import/JIT and Hg geometry, so the fan-out launcher (`scripts/fanout.py`) keeps a persistent worker pool and the v2.5.1 fused kernels cut the geometry stage ~15×; `write_strain_provenance = false` is the storage lever (0.51 TB → ~3 GB per 100k images). Numbers below are the measured rows from [docs/cluster-profiling.md](../docs/cluster-profiling.md).
'''),
    ("code", '''
runs = ["v2.4.0\\nsubprocess", "v2.5.1\\nisolate", "v2.5.1\\npool"]
laptop = [703, 1309, 2717]          # configs/hour, 8 workers, 16-config manifest
cluster = [2254, 8173, 12993]       # 32-core LSF node; pool rows at 16x2 sizing

x = np.arange(3)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - 0.2, laptop, 0.4, label="laptop (8x1)")
ax.bar(x + 0.2, cluster, 0.4, label="cluster node (16x2)")
for i, (lv, cv) in enumerate(zip(laptop, cluster)):
    ax.text(i - 0.2, lv, str(lv), ha="center", va="bottom", fontsize=8)
    ax.text(i + 0.2, cv, str(cv), ha="center", va="bottom", fontsize=8)
ax.set_xticks(x, runs)
ax.set_ylabel("identify configs / hour")
ax.set_title("Fan-out throughput (docs/cluster-profiling.md; cluster DoD = 5.76x)")
ax.legend()
fig.tight_layout()
'''),
    ("md", '''
At the measured 12 993 configs/hour a 100k-image campaign is an overnight job on a handful of nodes. Recipes (LSF arrays, seed sharding, storage budgets): [identification_ml_tutorial](identification_ml_tutorial/dfxm_identify_ml_tutorial.ipynb) §8 and [docs/cluster-runs.md](../docs/cluster-runs.md). All references: [references page](../docs/references.md).
'''),
]
```

**NOTE for executor:** the label-reading cell mirrors `read_labels`/`read_image` from the ML tutorial (cells 12/14). If a key differs (e.g. `rotation_deg` dataset shape), open the ML tutorial notebook JSON and copy its exact access pattern — it is the working reference.

- [ ] **Step 2: Build:** `$PY m5_scratch\build_05.py`
- [ ] **Step 3: Execute:** `$PY -m pytest --nbmake --nbmake-timeout=900 examples/05_identification_at_scale.ipynb -q` → `1 passed` (~2–4 min).
- [ ] **Step 4: Inspect** `examples/img/05_identification_preview.png` — 4×3 grid, first column noisy, others clean single dislocations with label titles.
- [ ] **Step 5: Commit**

```bash
git add examples/05_identification_at_scale.ipynb examples/img/05_identification_preview.png
git commit -m "feat: tutorial notebook 05 - identification datasets + throughput"
```

---

### Task 8: `examples/README.md` index

**Files:**
- Modify: `examples/README.md`

- [ ] **Step 1: Rewrite the top of `examples/README.md`** — keep the existing notes about nbstripout and the two legacy example dirs, prepend the series index:

```markdown
# Examples

Tutorial series (run each from this `examples/` folder; notebooks are
committed output-stripped — the preview images below are what the key
figure looks like when executed):

| Notebook | Shows | Preview |
|---|---|---|
| [01 · Quickstart](01_quickstart.ipynb) | empty TOML → first image; the two-stage model | ![](img/01_quickstart_preview.png) |
| [02 · Reciprocal space](02_reciprocal_space.ipynb) | the resolution kernel; MC vs analytic backend | ![](img/02_reciprocal_space_preview.png) |
| [03 · Dislocations & contrast](03_dislocations_and_contrast.ipynb) | character sweep, weak beam, COM ≈ −qi | ![](img/03_dislocations_preview.png) |
| [04 · Oblique & reflections](04_oblique_and_reflections.ipynb) | mounts, reflection tables, g·b invisibility | ![](img/04_reflections_preview.png) |
| [05 · Identification at scale](05_identification_at_scale.ipynb) | labelled ML datasets, fan-out throughput | ![](img/05_identification_preview.png) |

Citations: [docs/references.md](../docs/references.md). CI executes
notebooks 01–03 on every push (`.github/workflows/notebooks.yml`).
```

(Adapt the retained tail text as needed so the file reads coherently — read the current 41-line file first.)

- [ ] **Step 2: Commit**

```bash
git add examples/README.md
git commit -m "docs: examples README - tutorial series index with previews"
```

---

### Task 9: Notebook CI workflow

**Files:**
- Create: `.github/workflows/notebooks.yml` (NEW file — never touch `ci.yml`)

- [ ] **Step 1: Read `.github/workflows/ci.yml`** (read-only) and note the `actions/checkout` + `actions/setup-python` versions it uses; mirror them below.

- [ ] **Step 2: Write `.github/workflows/notebooks.yml`**

```yaml
name: notebooks

on:
  push:
    branches: [main]
    paths: ["examples/**", "src/**", ".github/workflows/notebooks.yml"]
  pull_request:
    paths: ["examples/**", "src/**", ".github/workflows/notebooks.yml"]

jobs:
  execute:
    runs-on: ubuntu-latest
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@v4          # match ci.yml's version
      - uses: actions/setup-python@v5      # match ci.yml's version
        with:
          python-version: "3.11"
          cache: pip
      - name: Install package + nbmake
        run: python -m pip install -e ".[dev]" nbmake
      - name: Execute tutorial notebooks 01-03
        run: >
          python -m pytest -q --nbmake --nbmake-timeout=900
          examples/01_quickstart.ipynb
          examples/02_reciprocal_space.ipynb
          examples/03_dislocations_and_contrast.ipynb
```

(01 self-bootstraps the shared 2 M-ray kernel into `examples/kernel/`, which 02 reuses within the same job; 03 is kernel-free. 04/05 are deliberately CI-exempt per the spec.)

- [ ] **Step 3: Validate YAML parses**

Run: `$PY -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/notebooks.yml').read_text()); print('yaml ok')"`
Expected: `yaml ok`. (If PyYAML is missing in the venv: `$PY -m pip install pyyaml`.)

- [ ] **Step 4: Prove the CI command works from a clean state**

```bash
rm -rf examples/kernel examples/out_01 examples/out_02 examples/out_03
$PY -m pytest -q --nbmake --nbmake-timeout=900 examples/01_quickstart.ipynb examples/02_reciprocal_space.ipynb examples/03_dislocations_and_contrast.ipynb
```
Expected: `3 passed` with NO pre-existing kernel — this is exactly what the CI runner sees. Note the wall time; if > ~20 min, reduce 02's mosa to 7×7 steps in the builder and re-run Tasks 4 steps 2–5.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/notebooks.yml
git commit -m "ci: execute tutorial notebooks 01-03 via nbmake (standalone workflow)"
```

---

### Task 10: Final gates, cleanup, review

- [ ] **Step 1: Bootstrap a seeded kernel into the worktree's pkl_files** (recovers the kernel-dependent tests that SKIPPED in the 830-passed baseline):

```bash
$PY -c "from dfxm_geo.reciprocal_space.kernel import cli_main; cli_main()" --config configs/profile_identify_single.toml --seed 0 --if-missing
```
(Writes `reciprocal_space/pkl_files/Resq_i_h-1_k1_l-1_17keV_*.npz` in the worktree — gitignored, ~122 MB, deleted in Step 4.)

- [ ] **Step 2: Full gates**

```bash
$PY -m pytest -q
$PY -m mypy src/dfxm_geo/
```
Expected: failure set EMPTY (compare against the main-tree gate shape: ~898 passed / 1 skipped / 22 deselected / 1 xfailed; exact skip count depends on which kernel-dependent tests the Step 1 kernel re-enables) and `Success: no issues found in 40 source files`. Also re-run all five notebooks one last time:
```bash
$PY -m pytest -q --nbmake --nbmake-timeout=900 examples/0*.ipynb
```
Expected: `5 passed`.

- [ ] **Step 3: Verify the no-touch invariant**

Run: `git diff main --stat -- pyproject.toml .github/workflows/ci.yml src/`
Expected: EMPTY output. If anything shows here, the branch violates the spec — fix before review.

- [ ] **Step 4: Cleanup session-created intermediates >10 MB** (CLAUDE.md wrap-up rule)

```bash
rm -f reciprocal_space/pkl_files/Resq_i_*.npz
rm -rf examples/kernel examples/out_01 examples/out_02 examples/out_03 examples/out_04 examples/out_05
rm -rf direct_space/deformation_gradient_tensors/Fg_*.npy
```
(The wall-mode run in 03 creates an Fg cache ~100 MB; the tutorial kernel is ~128 MB. All regenerable.)

- [ ] **Step 5: Tick the spec's §6 gate checkboxes** in `docs/superpowers/specs/2026-06-12-m5-tutorial-notebooks-design.md` for every gate that passed, commit:

```bash
git add docs/superpowers/specs/2026-06-12-m5-tutorial-notebooks-design.md
git commit -m "docs: tick M5 arc-1 spec gates after final verification"
```

- [ ] **Step 6: Request review** — invoke superpowers:requesting-code-review for the whole branch (spec + quality), then superpowers:finishing-a-development-branch to present merge options to Sina. NO version tag, NO push without Sina's explicit nod.

---

## Plan self-review record

- **Spec coverage:** §1 deliverables → Tasks 3–7 (notebooks), 2 (references), 9 (CI), 8 (README index). §2 no-touch → working rules + Task 10 Step 3. §3 conventions (stripped/previews/flat/smoke/self-bootstrap) → builder pattern + per-task cells. §6 gates → Task 10. Spec amendments → Task 2 Step 3.
- **Known deviations from spec, by design:** 03/04 use the analytic backend (no kernel) rather than the smoke MC kernel — strictly simpler and deterministic; spec's "self-bootstrap" applies to 01/02/05 which do need it.
- **Type consistency:** all notebooks share the same setup-cell shape; preview filenames match between Tasks 3–7 and Task 8's README table; `$PY` is the worktree venv python everywhere.
