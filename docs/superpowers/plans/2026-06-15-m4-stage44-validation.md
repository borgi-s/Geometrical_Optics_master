# M4 Stage 4.4 — Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close out the M4 crystal-structures milestone on `abaf614` with locked per-structure golden snapshots, committed tutorial figures, and a validation report mapping every M4 DoD box to a test — without cutting v3.0.0.

**Architecture:** Extract the three deterministic showcase recipes (FCC Al / BCC W / HCP Ti, single pure-edge dislocation, analytic backend, ideal detector) into one shared in-repo module `scripts/render_structure_showcase.py`. A new slow test renders each structure in-process at the full default grid and locks it against a committed `.npy` golden. The same module renders the docs figure; the paper's `make_showcase.py` is refactored to import it so the recipe never drifts.

**Tech Stack:** Python, numpy, h5py, matplotlib (Agg), pytest, `dfxm_geo.pipeline` (`load_identification_config`, `run_identification`), `dfxm_geo.crystal.oblique` (`CrystalMount`, `compute_omega_eta`).

**Environment:** Use the venv python `C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe`. Working dir: `C:\Users\borgi\Documents\GM-reworked\Geometrical_Optics_master`. Branch already exists: `feature/m4-stage44-validation` (off `abaf614`).

**Spec:** `docs/superpowers/specs/2026-06-15-m4-stage44-validation-design.md`

---

## File structure

| File | Responsibility |
|---|---|
| `scripts/render_structure_showcase.py` | **Create.** Single source of truth: the 3 recipe builders, η computation, in-process `render_raw(tag)`, figure-assembly helpers, and a `--figures`/`--golden` CLI. |
| `tests/data/golden/structure_showcase/{fcc,bcc,hcp}.npy` | **Create (generated).** Locked raw float32 detector images, one per structure. |
| `tests/test_structure_goldens.py` | **Create.** Slow test: render each structure in-process, assert `np.allclose` vs golden + finite/contrast; FCC 2-run determinism. |
| `docs/img/showcase_fcc_bcc_hcp.png` + `showcase_{fcc,bcc,hcp}.png` | **Create (generated).** Committed tutorial figures. |
| `docs/crystal-structures.md` | **Modify.** Add a short "Forward contrast across crystal systems" figure block. |
| `papers/2026-06-dfxm-geo-software-paper/scripts/make_showcase.py` | **Modify.** Import recipes/render/assembly from the shared module (drop the inline copies); paper figure output unchanged in content. |
| `docs/m4-validation-report.md` | **Create.** DoD-to-test mapping + recorded gate numbers + deferred-release punch list. |

---

### Task 1: Shared recipe + render module

**Files:**
- Create: `scripts/render_structure_showcase.py`

- [ ] **Step 1: Write the module**

```python
#!/usr/bin/env python
"""Single source of truth for the FCC Al / BCC W / HCP Ti crystal-structure
showcase: deterministic weak-beam DFXM render of one pure-edge dislocation per
structure on the analytic backend.

Used by:
  * tests/test_structure_goldens.py    (locks the raw render per structure)
  * docs figure generation             (--figures -> docs/img/)
  * papers/.../scripts/make_showcase.py (the published paper figure)

All three share the recipe builders below so the golden, the docs figure, and
the paper figure can never drift. Run with the project venv python:

    python scripts/render_structure_showcase.py --figures   # docs/img/*.png|pdf
    python scripts/render_structure_showcase.py --golden    # tests/.../*.npy
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent          # Geometrical_Optics_master/
DOCS_IMG = _REPO_ROOT / "docs" / "img"
GOLDEN_DIR = _REPO_ROOT / "tests" / "data" / "golden" / "structure_showcase"

KEV = 17.0
TAGS = ("fcc", "bcc", "hcp")
PANEL_TITLES = {
    "fcc": ("FCC Al", r"$(\bar{1}1\bar{1})$, edge"),
    "bcc": ("BCC W", r"$(200)$, edge"),
    "hcp": ("HCP Ti", r"$(10\bar{1}1)$, edge"),
}

RESBLOCK = (
    'backend = "analytic"\nbeamstop = false\naperture = false\n'
    "zeta_v_fwhm = 5.3e-4\nzeta_h_fwhm = 0.0\n"
    "NA_rms = 3.1106382978723403e-4\neps_rms = 6.0e-5\n"
)
IO_BLOCK = (
    '[io]\nftype = ".h5"\ndislocs_dirname = "dis"\nperfect_dirname = "ignored"\n'
    "include_perfect_crystal = false\nwrite_strain_provenance = false\n"
)
DET_IDEAL = '[detector]\nmodel = "ideal"\n'


def _vec(t) -> str:
    return f"[{int(t[0])}, {int(t[1])}, {int(t[2])}]"


def sweep_block(plane) -> str:
    # angle 0 deg == pure edge; a single configuration, no slip-plane sweep,
    # no RNG draw -> deterministic raw render.
    return (
        f"slip_plane_normal = {_vec(plane)}\n"
        "sweep_all_slip_planes = false\n"
        "angle_start_deg = 0.0\nangle_stop_deg = 0.0\nangle_step_deg = 30.0\n"
        "exclude_invisibility = false\n"
    )


def eta_for(mount, hkl):
    from dfxm_geo.crystal.oblique import compute_omega_eta

    g = compute_omega_eta(mount, hkl, KEV)
    for eta in (g.eta_1, g.eta_2):
        if not np.isnan(eta):
            return float(eta)
    return None


def fcc_toml() -> str:
    return (
        'mode = "single"\n\n'
        f"[reciprocal]\nhkl = {_vec((-1, 1, -1))}\nkeV = {KEV}\n{RESBLOCK}\n"
        '[geometry]\nmode = "simplified"\n\n'
        # Classic validated FCC path: no structure_type / material in simplified
        # mode (the bare default is FCC aluminium, nu = 0.334).
        '[crystal]\nlattice = "cubic"\na = 4.05e-10\n'
        f"{sweep_block((1, 1, 1))}\n"
        "[scan.phi]\nvalue = 1.75e-4\n\n"
        f"{DET_IDEAL}\n{IO_BLOCK}"
    )


def bcc_toml(eta) -> str:
    return (
        'mode = "single"\n\n'
        f"[reciprocal]\nhkl = {_vec((2, 0, 0))}\nkeV = {KEV}\n{RESBLOCK}\n"
        f'[geometry]\nmode = "oblique"\neta = {eta!r}\n\n'
        '[crystal]\nlattice = "cubic"\na = 3.1652e-10\nstructure_type = "bcc"\n'
        'material = "W"\npoisson_ratio = 0.28\n'
        "mount_x = [1, 0, 0]\nmount_y = [0, 1, 0]\nmount_z = [0, 0, 1]\n"
        f"{sweep_block((1, -1, 0))}\n"
        "[scan.phi]\nvalue = 1.75e-4\n\n"
        f"{DET_IDEAL}\n{IO_BLOCK}"
    )


def hcp_toml(eta) -> str:
    return (
        'mode = "single"\n\n'
        f"[reciprocal]\nhkl = {_vec((1, 0, -1))}\nkeV = {KEV}\n{RESBLOCK}\n"
        f'[geometry]\nmode = "oblique"\neta = {eta!r}\n\n'
        '[crystal]\nlattice = "hexagonal"\na = 2.9505e-10\nc = 4.6826e-10\n'
        'structure_type = "hcp"\nmaterial = "Ti"\npoisson_ratio = 0.32\n'
        "mount_x = [2, -1, 0]\nmount_y = [0, 1, 0]\nmount_z = [0, 0, 1]\n"
        f"{sweep_block((1, 0, 1))}\n"
        "[scan.phi]\nvalue = 1.75e-4\n\n"
        f"{DET_IDEAL}\n{IO_BLOCK}"
    )


def build_toml(tag: str) -> str:
    """Return the TOML text for a structure tag, computing eta where needed."""
    from dfxm_geo.crystal.oblique import CrystalMount

    if tag == "fcc":
        return fcc_toml()
    if tag == "bcc":
        mount = CrystalMount(lattice="cubic", a=3.1652e-10, structure_type="bcc",
                             mount_x=(1, 0, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1))
        eta = eta_for(mount, (2, 0, 0))
        if eta is None:
            raise RuntimeError("BCC W (200) reflection not reachable at 17 keV")
        return bcc_toml(eta)
    if tag == "hcp":
        mount = CrystalMount(lattice="hexagonal", a=2.9505e-10, c=4.6826e-10,
                             structure_type="hcp",
                             mount_x=(2, -1, 0), mount_y=(0, 1, 0), mount_z=(0, 0, 1))
        eta = eta_for(mount, (1, 0, -1))
        if eta is None:
            raise RuntimeError("HCP Ti (10-1) reflection not reachable at 17 keV")
        return hcp_toml(eta)
    raise ValueError(f"unknown structure tag {tag!r} (expected one of {TAGS})")


def render_raw(tag: str, workdir: Path) -> np.ndarray:
    """Render one structure in-process and return the raw 2-D detector image.

    Deterministic: single pure-edge dislocation, analytic backend, ideal
    detector. Reads the single frame from the identify master's external link.
    """
    from dfxm_geo.pipeline import load_identification_config, run_identification

    workdir = Path(workdir)
    cfg_path = workdir / f"{tag}.toml"
    cfg_path.write_text(build_toml(tag), encoding="utf-8")
    out = workdir / f"out_{tag}"
    cfg = load_identification_config(cfg_path)
    run_identification(cfg, out)
    with h5py.File(out / "dfxm_identify.h5", "r") as f:
        img = f["/1.1/instrument/dfxm_sim_detector/data"][0].astype(np.float64)
    return img


# --- figure assembly (publication render) ----------------------------------

def crop_to_feature(img, margin=22, min_h=120, min_w=120):
    bg = float(np.median(img))
    dev = np.abs(img.astype(float) - bg)
    thr = np.percentile(dev, 99.0)
    ys, xs = np.where(dev >= max(thr, 1e-12))
    h, w = img.shape
    if len(ys) < 12:
        s = min(h, w)
        cy, cx = h // 2, w // 2
        return img[max(0, cy - s // 2):cy + s // 2, max(0, cx - s // 2):cx + s // 2]
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half_h = max((y1 - y0) // 2 + margin, min_h // 2)
    half_w = max((x1 - x0) // 2 + margin, min_w // 2)
    return img[max(0, cy - half_h):min(h, cy + half_h),
               max(0, cx - half_w):min(w, cx + half_w)]


def norm01(crop):
    lo, hi = np.percentile(crop, 1.0), np.percentile(crop, 99.5)
    if hi <= lo:
        hi = lo + 1e-12
    return np.clip((crop.astype(float) - lo) / (hi - lo), 0.0, 1.0)


def assemble_figure(images: dict, out_pdf: Path) -> None:
    """Assemble the multi-panel showcase figure + per-panel PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.dpi": 300})
    tags = [t for t in TAGS if images.get(t) is not None]
    n = len(tags)
    fig, axes = plt.subplots(1, n, figsize=(2.45 * n, 2.9))
    if n == 1:
        axes = [axes]
    last_im = None
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    for ax, tag in zip(axes, tags):
        disp = norm01(crop_to_feature(images[tag]))
        last_im = ax.imshow(disp, cmap="gray", origin="lower", vmin=0.0, vmax=1.0,
                            aspect="equal", interpolation="bilinear")
        name, sub = PANEL_TITLES[tag]
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}\n{sub}", fontsize=9.5)
        # Per-panel PNG (for notebook/doc reuse).
        pfig, pax = plt.subplots(figsize=(2.6, 2.6))
        pax.imshow(disp, cmap="gray", origin="lower", vmin=0.0, vmax=1.0,
                   aspect="equal", interpolation="bilinear")
        pax.set_xticks([]); pax.set_yticks([])
        pfig.savefig(out_pdf.parent / f"showcase_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close(pfig)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, fraction=0.045, pad=0.03, ticks=[0, 0.5, 1.0])
        cbar.set_label("normalized weak-beam intensity", fontsize=8)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Weak-beam DFXM contrast of an edge dislocation across crystal "
                 "systems (17 keV, analytic backend)", fontsize=9, y=1.04)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", action="store_true", help="render docs/img figures")
    ap.add_argument("--golden", action="store_true", help="(re)generate test goldens")
    args = ap.parse_args()
    if not (args.figures or args.golden):
        ap.error("pass --figures and/or --golden")

    work = Path(tempfile.mkdtemp(prefix="dfxm_showcase_"))
    print("workdir:", work)
    images = {}
    for tag in TAGS:
        print(f"[{tag}] rendering ...")
        try:
            images[tag] = render_raw(tag, work)
            print(f"[{tag}] shape={images[tag].shape} std={images[tag].std():.4g}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{tag}] FAILED: {exc}")
            images[tag] = None

    if args.golden:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        for tag in TAGS:
            if images[tag] is not None:
                np.save(GOLDEN_DIR / f"{tag}.npy", images[tag].astype(np.float32))
                print("wrote", GOLDEN_DIR / f"{tag}.npy")
    if args.figures:
        assemble_figure(images, DOCS_IMG / "showcase_fcc_bcc_hcp.pdf")
        print("wrote", DOCS_IMG / "showcase_fcc_bcc_hcp.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-run the module to generate goldens (also proves `render_raw` works)**

Run:
```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" scripts/render_structure_showcase.py --golden
```
Expected: prints a shape `(510, 170)` and non-zero `std` for each of `fcc`/`bcc`/`hcp`, and writes three `.npy` files under `tests/data/golden/structure_showcase/`. Runtime a few minutes (three full-grid analytic renders). If any structure prints `FAILED`, stop and debug the recipe before continuing.

- [ ] **Step 3: Verify the goldens are sane and small**

Run:
```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -c "import numpy as np,glob,os;[print(os.path.basename(p), np.load(p).shape, round(float(np.load(p).std()),5), os.path.getsize(p)) for p in glob.glob('tests/data/golden/structure_showcase/*.npy')]"
```
Expected: three files, each shape `(510, 170)`, non-zero std, size ~350 KB (well under 10 MB).

- [ ] **Step 4: Commit**

```bash
git add scripts/render_structure_showcase.py tests/data/golden/structure_showcase/
git commit -m "feat(validation): shared crystal-structure showcase recipe + golden baselines"
```

---

### Task 2: Golden regression test

**Files:**
- Create: `tests/test_structure_goldens.py`

- [ ] **Step 1: Write the test**

```python
"""M4 Stage 4.4 golden regression: lock the per-structure forward render.

Renders one pure-edge dislocation per crystal system (FCC Al / BCC W / HCP Ti)
on the analytic backend and compares the raw float32 detector image against a
committed golden. The recipe is shared with the docs/paper figure via
scripts/render_structure_showcase.py, so this test guards against silent
forward-physics drift across crystal systems.

Slow (three full-grid analytic renders); not run in the default CI suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# scripts/ is not on sys.path by default; the shared recipe module lives there.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import render_structure_showcase as rss  # noqa: E402

_GOLDEN = Path(__file__).resolve().parent / "data" / "golden" / "structure_showcase"


@pytest.mark.slow
@pytest.mark.parametrize("tag", rss.TAGS)
def test_structure_render_matches_golden(tag: str, tmp_path: Path) -> None:
    golden = np.load(_GOLDEN / f"{tag}.npy").astype(np.float64)
    img = rss.render_raw(tag, tmp_path)

    assert img.shape == golden.shape, f"{tag}: shape {img.shape} != golden {golden.shape}"
    assert np.isfinite(img).all(), f"{tag}: render has non-finite pixels"
    assert float(img.std()) > 0.0, f"{tag}: render has no contrast"

    # Analytic backend is pure-numpy deterministic; rtol guards platform float
    # noise only. atol scaled to the golden's dynamic range so near-zero
    # background pixels don't fail on relative tolerance alone.
    atol = 1e-9 + 1e-6 * float(np.abs(golden).max())
    assert np.allclose(img, golden, rtol=1e-6, atol=atol), (
        f"{tag}: render drifted from golden "
        f"(max abs diff {float(np.abs(img - golden).max()):.3e}, atol {atol:.3e})"
    )


@pytest.mark.slow
def test_fcc_render_is_deterministic(tmp_path: Path) -> None:
    """Same recipe -> bit-identical pixels on the same machine (independent of
    the stored golden's tolerance)."""
    a = rss.render_raw("fcc", tmp_path / "a")
    b = rss.render_raw("fcc", tmp_path / "b")
    assert np.array_equal(a, b), "FCC render is non-deterministic across two runs"
```

- [ ] **Step 2: Run the test to verify it passes against the committed goldens**

Run:
```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest tests/test_structure_goldens.py -v -m slow
```
Expected: 4 PASS (`fcc`/`bcc`/`hcp` golden matches + FCC determinism). Runtime several minutes. (The test is meaningless if the goldens from Task 1 were not committed first — they were.)

- [ ] **Step 3: Sanity-check that the test actually locks (mutation check)**

Temporarily edit the test's `atol`/`rtol` to `0` and confirm at least the determinism test still passes but the golden compares become exact; revert. (Optional 1-minute confidence check — do NOT commit the edit.)

- [ ] **Step 4: Commit**

```bash
git add tests/test_structure_goldens.py
git commit -m "test(validation): lock FCC/BCC/HCP forward renders against goldens"
```

---

### Task 3: Render and commit the docs figures

**Files:**
- Create (generated): `docs/img/showcase_fcc_bcc_hcp.png`, `docs/img/showcase_fcc_bcc_hcp.pdf`, `docs/img/showcase_{fcc,bcc,hcp}.png`
- Modify: `docs/crystal-structures.md`

- [ ] **Step 1: Generate the figures**

Run:
```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" scripts/render_structure_showcase.py --figures
```
Expected: writes `docs/img/showcase_fcc_bcc_hcp.png` + `.pdf` + three per-panel `showcase_{fcc,bcc,hcp}.png`. Open the combined PNG and confirm three labelled greyscale panels each showing a dislocation contrast feature.

- [ ] **Step 2: Reference the figure in `docs/crystal-structures.md`**

Find a sensible location (after the structure-type / slip-family overview, before the limitations section) and insert:

```markdown
## Forward contrast across crystal systems

The three supported structure families render end to end on the kernel-free
analytic backend. The figure below shows one weak-beam DFXM image of a pure-edge
dislocation in FCC aluminium, BCC tungsten and HCP titanium, all at 17 keV
(reproduce with `python scripts/render_structure_showcase.py --figures`):

![Weak-beam DFXM contrast of an edge dislocation in FCC Al, BCC W and HCP Ti](img/showcase_fcc_bcc_hcp.png)

These renders are locked as golden regressions in
`tests/test_structure_goldens.py` (Stage 4.4 validation).
```

- [ ] **Step 3: Commit**

```bash
git add docs/img/showcase_fcc_bcc_hcp.png docs/img/showcase_fcc_bcc_hcp.pdf docs/img/showcase_fcc.png docs/img/showcase_bcc.png docs/img/showcase_hcp.png docs/crystal-structures.md
git commit -m "docs(validation): commit FCC/BCC/HCP showcase figures + crystal-structures reference"
```

---

### Task 4: Refactor `make_showcase.py` onto the shared recipe

**Files:**
- Modify: `papers/2026-06-dfxm-geo-software-paper/scripts/make_showcase.py`

- [ ] **Step 1: Replace the script body with a thin wrapper over the shared module**

Overwrite `papers/2026-06-dfxm-geo-software-paper/scripts/make_showcase.py` with:

```python
#!/usr/bin/env python
"""Render figures/showcase_fcc_bcc_hcp.pdf for the dfxm-geo software paper.

Thin wrapper over the in-repo single source of truth
scripts/render_structure_showcase.py, so the paper figure, the docs figure and
the Stage 4.4 golden regression all use the identical recipe and renderer.

Run with the project venv python from anywhere:
    python make_showcase.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Locate the repo's scripts/ dir (…/Geometrical_Optics_master/scripts) and import
# the shared showcase module from it.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3] / "Geometrical_Optics_master"   # papers/<paper>/scripts/ -> repo
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import render_structure_showcase as rss  # noqa: E402

OUT_PDF = _THIS.parent.parent / "figures" / "showcase_fcc_bcc_hcp.pdf"


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="dfxm_showcase_"))
    print("workdir:", work)
    images = {}
    for tag in rss.TAGS:
        print(f"[{tag}] rendering ...")
        images[tag] = rss.render_raw(tag, work)
        print(f"[{tag}] shape={images[tag].shape} std={images[tag].std():.4g}")
    rss.assemble_figure(images, OUT_PDF)
    print("wrote", OUT_PDF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: this resolves the repo as a sibling of the `papers/` tree
(`papers/<paper>/scripts/make_showcase.py` → `parents[3]` is the `GM-reworked`
root → `/Geometrical_Optics_master`). If the paper folder is ever relocated,
this single path line is the only thing to adjust.

- [ ] **Step 2: Regenerate the paper figure and confirm it is unchanged in content**

Run:
```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" papers\2026-06-dfxm-geo-software-paper\scripts\make_showcase.py
```
Expected: writes `papers/2026-06-dfxm-geo-software-paper/figures/showcase_fcc_bcc_hcp.pdf` (+ `.png`) with three labelled panels — visually equivalent to the prior committed figure (same recipe, same physics; the render now goes in-process instead of via the CLI, which calls the same orchestrator).

- [ ] **Step 3: Commit**

```bash
git add papers/2026-06-dfxm-geo-software-paper/scripts/make_showcase.py papers/2026-06-dfxm-geo-software-paper/figures/showcase_fcc_bcc_hcp.pdf papers/2026-06-dfxm-geo-software-paper/figures/showcase_fcc_bcc_hcp.png
git commit -m "refactor(paper): make_showcase imports the shared showcase recipe (no drift)"
```

---

### Task 5: Baseline gate run + validation report

**Files:**
- Create: `docs/m4-validation-report.md`

- [ ] **Step 1: Clean the stale `Fg` caches (documented hazard before any full-suite run)**

Run:
```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -c "import glob,os;[os.remove(p) for p in glob.glob('direct_space/deformation_gradient_tensors/Fg_*.npy')];print('cleaned Fg caches')"
```

- [ ] **Step 2: Run the default suite and record the totals**

Run:
```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q
```
Record the printed summary line (e.g. `NNNN passed, N skipped, N xfailed`).

- [ ] **Step 3: Run the slow suite and record the totals**

Run:
```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m pytest -q -m slow
```
Record the summary. Any failures here must be triaged: the three historically-known pre-existing slow failures are the two cumulative-RAM `test_analytic_backend_integration` (pass in isolation) and the detector-OOM `test_pipeline::…sample_remount_S2` — the latter should now PASS given the `apply()` chunking in `9d5705d`; if it still OOMs, record it as a known environmental limit, NOT a 4.4 regression. The new `test_structure_goldens.py` cases must pass.

- [ ] **Step 4: Run mypy and record the result**

Run:
```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -m mypy src/dfxm_geo/
```
Record the count (expected `0 errors`).

- [ ] **Step 5: Write `docs/m4-validation-report.md`**

Write the report, filling the three `<…>` measurement fields with the numbers recorded in Steps 2–4:

```markdown
# M4 (CIF Crystal Structures) — Validation Report

**Date:** 2026-06-15
**Commit:** `feature/m4-stage44-validation` off `abaf614`
**Milestone:** M4, Stage 4.4 (validation-only; v3.0.0 release deferred).

## Definition-of-done coverage

| # | M4 DoD item | Proven by |
|---|---|---|
| 1 | `[crystal] cif=…` + a BCC reflection produces a forward image and an identify library with BCC slip-system labels | `tests/test_bcc_e2e.py::test_bcc_via_fe_cif`, `::test_bcc_identify_has_bcc_slip_labels`, `::test_bcc_forward_runs` |
| 2 | Cubic/FCC path bit-identical to v2.4.0 | `tests/test_fcc_bit_identity.py`, `tests/test_cubic_bit_identity.py` (slip-order lock, wall + random-dislocation determinism, forward/identify ordering distinction) |
| 3 | Forbidden reflections rejected with explanatory error | `tests/test_extinction_rules.py`, `tests/test_reflections_extinct.py`, `tests/test_ceramic_acceptance_al2o3.py` |
| 4 | Isotropic-elasticity limitation documented prominently | `docs/crystal-structures.md` (limitations section) |

## Stage 4.4 additions

| Artifact | Location |
|---|---|
| Per-structure golden regression (FCC Al / BCC W / HCP Ti) | `tests/test_structure_goldens.py` + `tests/data/golden/structure_showcase/*.npy` |
| Shared deterministic recipe (single source of truth) | `scripts/render_structure_showcase.py` (also drives the docs + paper figures) |
| Tutorial figure | `docs/img/showcase_fcc_bcc_hcp.png`, referenced from `docs/crystal-structures.md` |

## CIF round-trips per structure (already covered)

| Structure | CIF route |
|---|---|
| FCC | `tests/data/cif/al_fm3m.cif` via `tests/test_cif_loader.py` |
| BCC | inline Fe `Im-3m` CIF in `tests/test_bcc_e2e.py::test_bcc_via_fe_cif` |
| HCP | inline Ti `P6_3/mmc` CIF in `tests/test_hcp_e2e.py::test_hcp_via_ti_cif` |

## Gate numbers on `abaf614` (Fg caches cleaned first)

- Default suite: `<paste from Step 2>`
- Slow suite: `<paste from Step 3>`
- mypy `src/dfxm_geo/`: `<paste from Step 4>`

Known pre-existing slow failures (NOT 4.4 regressions): two cumulative-RAM
`test_analytic_backend_integration` (pass in isolation); the detector-OOM
`test_pipeline::…sample_remount_S2` (mitigated by `apply()` chunking in
`9d5705d`).

## Deferred to the v3.0.0 release (separate, user-triggered step)

- [ ] Version bump `pyproject.toml` 2.5.1 → 3.0.0 + tag.
- [ ] Detector `counts_scale` realism decision (still provisional `1.0e4`).
- [ ] conda-forge recipe sync: add the `dfxm-find-reflections` entry point to
      `build.python.entry_points`; document `gemmi` as an optional extra (do
      NOT add to `run` deps).
- [ ] PyPI publish (gated on the `pypi` GitHub Environment approval).
```

- [ ] **Step 6: Commit**

```bash
git add docs/m4-validation-report.md
git commit -m "docs(validation): M4 validation report — DoD coverage + recorded gates on abaf614"
```

---

### Task 6: Wrap-up

- [ ] **Step 1: Clean regenerable large intermediates this session created**

Run:
```
& "C:\Users\borgi\Documents\GM-reworked\.venv\Scripts\python.exe" -c "import glob,os;[os.remove(p) for p in glob.glob('direct_space/deformation_gradient_tensors/Fg_*.npy')];print('cleaned Fg caches')"
```
(The committed goldens are ~350 KB each and stay; only the >100 MB `Fg_*.npy` scratch caches are removed.)

- [ ] **Step 2: Confirm the branch state**

Run:
```
git log --oneline abaf614..HEAD
git status
```
Expected: 5 new commits (Tasks 1–5), clean working tree apart from pre-existing untracked scratch. Do NOT push, do NOT tag — the branch awaits Sina's merge call.

- [ ] **Step 3: Report completion** with the recorded gate numbers and a one-line pointer to `docs/m4-validation-report.md`.

---

## Self-review notes

- **Spec coverage:** recipe module (spec §1) = Task 1; golden tests (§2) = Task 2; docs figures (§3) = Task 3; `make_showcase` dedup (§1) = Task 4; validation report (§4) + baseline gate (§5) = Task 5; `Fg` cache cleanup + no-push/no-tag = Task 6. CIF-round-trip cataloguing is in the report (Task 5), not new tests, per spec.
- **Determinism:** spec asks for a 2-run determinism assertion; implemented as `test_fcc_render_is_deterministic` (one representative structure) to keep the slow test from rendering every structure twice; the stored-golden `allclose` covers BCC/HCP regression.
- **Grid size:** full default `Npixels=510` (no safe shrink knob; keeps golden == figure raw image). Test is `slow`, consistent with the existing BCC/HCP e2e tests; not in default CI.
- **Type/name consistency:** `render_raw(tag, workdir)`, `build_toml(tag)`, `assemble_figure(images, out_pdf)`, `TAGS`, `PANEL_TITLES`, `GOLDEN_DIR`, `DOCS_IMG` are defined in Task 1 and used identically in Tasks 2–4.
```
