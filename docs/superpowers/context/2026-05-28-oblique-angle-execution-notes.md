# Execution notes for the oblique-angle Phase A plan

**Read this BEFORE starting `superpowers:subagent-driven-development` on
`docs/superpowers/plans/2026-05-28-oblique-angle-phase-a.md`.**

This document holds the context that lived in the brainstorming session
(spec → plan) but isn't checked into the repo's normal docs. It's pinned
here so the executing session can pick up cold without the user having to
re-relay everything.

---

## 1. What this arc is and how it got here

**Goal:** add oblique-angle DFXM geometry to `dfxm_geo` end-to-end, matching
the arXiv:2503.22022v1 paper's §3.3 + Appendix A formalism. Reproduce paper
Figure 3B as the v2.3.0 ship gate. Multi-reflection iteration is Phase B
(separate plan, v2.4.0). Multi-reflection and oblique angle are physically
bundled (the paper's oblique geometry is what makes multi-reflection from a
*common* crystalline volume possible — μ=0, η=η_C constant), but they ship
in two phases for tight scientific validation per phase.

**How we got here:** the user's earlier-planned "Phase 3 persistent-worker
pool + multi-reflection iteration" arc (see auto-memory
`session-handoff-2026-05-27-phase2-float32-fanout`) got reframed during a
brainstorm. The user said: "Yes let's get started on the multi-reflection
and let's combine it with oblique angle implementation." The persistent-
worker pool is now a separate, later arc.

**Sequencing the user chose:** "B then A" — scientific capability first
(this Phase A + B), throughput/ML-dataset second (persistent worker pool
later).

## 2. Critical scope decisions locked in the spec

These are answered during the brainstorm — don't re-litigate them:

1. **Scope:** Phase A = single-reflection oblique infrastructure (v2.3.0).
   Phase B = `[[reflections]]` list + `dfxm-find-reflections` CLI (v2.4.0,
   separate plan after A ships).
2. **`mu` motor:** NOT in scope. `eta` only this arc; `mu` deferred to
   v3.0.0 or later. The paper's oblique mode locks μ=0.
3. **Backends:** BOTH the MC LUT (`reciprocal_res_func`) and the v2.1.0
   analytic closed-form (`AnalyticResolution`) gain `eta`. Eta=0 →
   bit-identical to v2.2.0 / v2.1.0 numerics respectively, gated by
   regression tests at every commit.
4. **Config shapes supported:** single `[reflection]` block (current
   schema, default behavior). `[[reflections]]` list mode is Phase B.
5. **Pipeline coverage:** both `forward` AND `identify` gain eta-aware
   geometry. Identify has 3 sub-modes (single/multi/zscan) — all gain
   plumbing but exercised only at eta=0 in Phase A (bit-identical
   regression).
6. **Crystal lattice:** `[crystal] a` becomes config-driven (cubic only).
   `.cif` parsing + non-cubic deferred to v3.0.0. See
   `followups_cif_crystal_structures.md` in user's auto-memory (also
   reproduced below).
7. **eta input model:** "both, validated" — user provides `[geometry] eta`
   directly, AND bootstrap cross-checks against `compute_omega_eta(mount,
   hkl, keV)` per paper Appendix A. No auto-correct; refuse on mismatch.
8. **LUT cache key:** changes from per-hkl to per-`(θ, η, keV, optics)`.
   Reflections sharing `(η, θ)` share ONE LUT — the paper's key insight,
   pays off in Phase B's multi-reflection efficiency.

## 3. THE η/θ gotcha — important, don't get it backwards

Throughout much of the brainstorm I (Claude, prior session) had η and θ
swapped. The paper §6.1 and Table A.2 are unambiguous:

- **θ = 15.416-15.417°  = 0.2691 rad** (Bragg angle)
- **η = 20.233°         = 0.3531 rad** (azimuthal angle)

This applies to Table A.2's group 1 (Al, mount (100,010,001), 19.1 keV,
reflections `{(1̄1̄3), (1̄13), (113), (11̄3)}`). The spec + plan are now
correct. If you see anything claiming "η = 15.417°" anywhere it's wrong
and should be flagged.

## 4. Reference materials and where they live

| What | Where |
|---|---|
| Spec | `docs/superpowers/specs/2026-05-28-multi-reflection-oblique-angle-design.md` (on this branch) |
| Phase A plan | `docs/superpowers/plans/2026-05-28-oblique-angle-phase-a.md` (on this branch) |
| Paper | `arXiv:2503.22022v1`. Key sections: §3.3 (oblique geometry intro), §6.1 (Al phantom params), §6.2 (microscope params), Appendix A (search algorithm — eq A.5-A.13), Appendix F (closed-form Gaussian resolution), Table A.2 (reference reflection groups for Al at 19.1 keV), Figure 3B (single-image reproduction target) |
| darkmod (reference impl) | `https://github.com/AxelHenningsson/darkmod` — `darkmod/laue.py:get_eta_angle` confirms the η sign convention; `darkmod/resolution.py:TruncatedPentaGauss` is the analytic backend we're paralleling |
| User's prototype | NOT on this branch (lives at `C:\Users\borgi\Documents\Oblique_Angle\` on the user's laptop). Already absorbed into the spec/plan as relevant — you don't need to read it directly. |

## 5. Cluster-vs-laptop environment differences

The Phase A plan was authored on the user's Windows laptop. Commands inside
the plan hardcode the laptop venv path:

```
& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe" -m pytest ...
```

On the DTU HPC cluster this becomes:

```bash
# Activate the conda env (per dtu_hpc_lsf_specifics auto-memory):
conda activate dfxm-geo
python -m pytest ...
```

**Substitution rule for executor:** anywhere the plan invokes the laptop
venv python, replace with whatever Python the executing host uses. If
running on the cluster:

- `& "C:/Users/borgi/Documents/GM-reworked/.venv/Scripts/python.exe"` →
  `python` (assuming `dfxm-geo` conda env is activated)
- `C:/Users/borgi/Documents/GM-reworked/Geometrical_Optics_master` → the
  cluster clone path (e.g. `~/Geometrical_Optics_master` or wherever the
  user has it; from the DTU HPC LSF auto-memory the clone is on `zhome`).

The plan's commands work as-is on the laptop; adjust for the cluster
based on the actual paths.

## 6. Project conventions that aren't in the plan

These come from the user's CLAUDE.md (intentionally not checked in) but
are load-bearing for the executor:

- **No direct pushes to `main`.** Work goes feature-branch → PR → merge.
  The Phase A plan's Task 26 (merge + tag + push) is **user-gated** — do
  not push without explicit confirmation from the user via the operating
  session.
- **Use `git push origin --delete` only with explicit consent.** No
  destructive remote operations without the user's nod.
- **Never skip pre-commit hooks** (`--no-verify`) or bypass signing
  unless the user explicitly says so.
- **Confirm before pushing or opening PRs.** Local commits are fine to
  do freely; surface what's about to leave the machine.
- **Clean up large intermediates at wrap-up.** Files >10 MB this session
  created (commonly `Fg_*.npy` deformation-gradient caches) get deleted
  at end. Tiny `_vars.txt` sidecars + small scratch plots/scripts can
  stay. Tracked goldens in `tests/data/golden/` are NEVER deleted.
- **Scale down smoke tests.** DFXM pipeline e2e tests must use small
  scan grids (5×5 not 61×61) and small Npixels (64-128 not 510). 4D
  scans: 5×5×3×3 not 61×61×21×5. Saves 10+ minutes per run.
- **Parallelize subagents by default.** When dispatching multiple
  subagents (e.g. spec reviewer + code-quality reviewer after an
  implementer), emit them in a SINGLE assistant message with multiple
  `Agent` tool-use blocks side-by-side and `run_in_background: true`.
  Don't sequence reviewers — their context is independent. Sequence only
  when there's a real constraint (same-file edits, spec→quality review
  chain per task).
- **No concurrent sessions on one worktree.** Multiple Claude Code
  sessions doing git ops on the same clone share one HEAD and clobber
  each other's branch state. If running multiple in parallel: use git
  worktrees (the `superpowers:using-git-worktrees` skill) or pause all
  but one.
- **Smoke tests must stay green** before any refactor: run
  `python -m pytest -q` (with the project Python). The Fd_find golden in
  `tests/data/golden/Fd_find_smoke.npy` is the safety net under the whole
  cleanup.
- **mypy stays clean.** `mypy src/dfxm_geo/` should report 0 errors.

## 7. The "[crystal] without [geometry]" gap to close

The spec §6 has a rule the plan doesn't have a dedicated test for:
"Config has `[crystal]` block but no `[geometry]`: ValueError. Force
the user to be explicit." When wiring `cli_main` in plan Task 14, add an
enforcement line:

```python
mount_block = data.get("crystal")
geometry_block = data.get("geometry")
if mount_block is not None and geometry_block is None:
    raise ValueError(
        "[crystal] block requires [geometry] mode to be set explicitly. "
        "Use mode='simplified' for v2.2.0 behavior, mode='oblique' for "
        "oblique-angle geometry."
    )
```

And add the matching negative test to Task 24's
`tests/test_oblique_error_handling.py`:

```python
def test_crystal_block_without_geometry_block_rejected(tmp_path: Path) -> None:
    """Per spec §6: [crystal] requires [geometry] to be explicit about mode."""
    cfg = tmp_path / "bad.toml"
    cfg.write_text("""
[crystal]
lattice = "cubic"
a = 4.0e-10
mount_x = [1, 0, 0]
mount_y = [0, 1, 0]
mount_z = [0, 0, 1]

[reflection]
hkl = [-1, 1, -1]
keV = 17.0

[reciprocal]
Nrays = 1000
""")
    from dfxm_geo.reciprocal_space.kernel import cli_main
    rc = cli_main(["--config", str(cfg)])
    # Expect non-zero exit code + ValueError message on stderr; details
    # depend on whether cli_main raises or returns rc. Adapt to actual API.
```

## 8. What's deferred (don't try to do these in Phase A)

- Multi-reflection iteration `[[reflections]]` — Phase B, v2.4.0, separate plan.
- `dfxm-find-reflections` CLI — Phase B. (The Python function
  `find_reflections` IS implemented in Phase A but kept unwired.)
- `.cif` parsing, non-cubic lattices — v3.0.0. The user explicitly noted
  this is a future feature: "I want to at some point be able to use
  .cif files for different crystal structures." See section 9 below.
- Goniometer `μ` motor — v3.0.0 or later.
- Dropping `simplified` geometry mode — v3.0.0 breaking change.
- Persistent-worker pool / ML dataset throughput — separate later arc.

## 9. Future `.cif` follow-up (background)

(Reproduced from `followups_cif_crystal_structures.md` so it's
discoverable from the branch.)

User wants `dfxm_geo` to eventually consume `.cif` files so simulations
can run on arbitrary crystal structures, not just FCC.

**Why:** Currently the crystal structure is implicit — `_SLIP_SYSTEM_111`
covers FCC {111} only (and only 2 of 4 planes), the unit-cell / Burgers
vector inputs assume cubic. CIFs would let users drop in any structure
(BCC, HCP, hexagonal, orthorhombic, etc.) from a Materials-Project
download or experimental refinement, and have the pipeline derive lattice
vectors, slip systems, structure factors, and allowed reflections
automatically.

**How to apply:** Not in scope for this oblique-angle + multi-reflection
arc. Park it as a future feature. When that arc is designed, factor the
lattice/slip-system data path so a CIF reader can slot in later (don't
hardwire FCC assumptions any deeper than they already are). A library
like `pymatgen` or `gemmi` handles CIF parsing; symmetry-aware reflection
enumeration is the non-trivial part.

The current spec preserves the .cif extension hook: `[crystal] lattice =
"cubic"` is the only value accepted in v2.3.0, but the field exists for
v3.0.0 to widen to `lattice = "cif"` and read `[crystal] cif = "path"`.

## 10. Brainstorm trail (for if you need to reconstruct decisions)

The full Q&A trail with the user during brainstorm:

| Q | A |
|---|---|
| Q1 — Primary motivation? | B then A — scientific correctness first, ML dataset throughput later |
| Q2 — Axis scope? | C — start with `eta`, leave `mu` as a future hook |
| Q3 — Backend coverage? | A — both MC LUT + analytic |
| Q4 — Multi-reflection meaning? | C — both single-reflection (today) AND `[[reflections]]` list mode (Phase B) |
| Q5 — Identify coverage? | A — both forward + identify get eta-awareness |
| Q6 — Lattice scope? | A — config-driven cubic lattice now, .cif deferred |
| Q7 — Geometry-mode coverage (mid-brainstorm) | Keep `simplified` for back-compat through v2.x; drop it in v3.0.0 when .cif lands |
| Q8 — LUT cache key change? | Yes — change to per-(θ, η, keV, optics) so multi-reflection groups share one LUT |
| Q9 — Eta validation? | "Both, validated" — user provides explicit eta, bootstrap cross-checks against compute_omega_eta |
| Paper-figure target for Phase A ship gate | Figure 3B (§6.1 + §6.2 + Fig 3 caption have full params) |

---

**END OF EXECUTION NOTES.** Now read the spec and Phase A plan and begin
task 1.
