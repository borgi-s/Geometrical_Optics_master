# M2 decision: pixel-level R_x(eta) in forward() — implement or accept?

**Status: DECIDED 2026-06-11 — Option B (accept + document), per Sina.**
Accept-paragraph copied into the 2026-05-29 rescope spec as a closing note;
the M2 "decision recorded" DoD box is ticked. The analysis below is kept
as the record of the options. This was roadmap gaps G1.4/G1.5; the decision
changes M2's scope by ~a week and is required by the M2 DoD ("decision
recorded ... either implement pixel-level eta or document the ~4 %
approximation as accepted").

## Context

Since the v2.3.0 oblique Phase A arc, eta enters the simulation ONLY through
the resolution kernel (both backends: analytic_resolution.py and the MC LUT).
The direct-space `forward()` ray grid is NOT rotated by R_x(eta) at the pixel
level. This is an intentional Phase-A scope cut, validated at the time
against darkmod: the residual is ~3.9 % on the thinnest covariance
eigenvalue of the resolution function (see
docs/superpowers/specs/2026-05-29-v230-oblique-ship-gate-rescope.md and the
darkmod parity work in auto-memory darkmod_analytic_resolution_parity).

## Option A — implement pixel-level eta (G1.4), optionally eta-weave (G1.5)

- Apply R_x(eta) in the direct-space projection (forward_model.py, the
  ForwardContext geometry block makes this a contained change post-#16).
- Effort: 3-5 days including darkmod parity re-validation (G1.4); +2-3 days
  if the eta-weave refinement (G1.5) is wanted instead of post-hoc R_x.
- Buys: closes the last known physics gap vs darkmod for oblique
  reflections; needed if quantitative pixel-fidelity to paper Fig 3 (or any
  oblique pixel-level comparison against darkmod/experiment) becomes a goal.
- Risk: touches the hot forward kernel path right after the W3 fusion work;
  full parity gates required.

## Option B — accept + document (recommended)

- Record the ~4 % thinnest-eigenvalue approximation as ACCEPTED in the
  oblique spec; no code change.
- Rationale: (1) M3 multi-reflection sweeps consume oblique geometry only
  through theta/eta kernel selection — first-order B-prime model, already
  deliberately approximate at the same order; (2) the ML-training-data goal
  (perf arc) is contrast-statistics-driven, not pixel-fidelity-driven;
  (3) the gap is confined to the resolution function's thinnest axis, the
  direction least constrained by experiment.
- Revisit trigger: any future quantitative oblique comparison against
  darkmod or measured Fig-3-class data (e.g. a paper figure), or M4/CIF-era
  non-cubic reflections where the approximation is unvalidated.

## Decision

- [ ] Option A (implement G1.4; G1.5 yes/no separately) — declined 2026-06-11
- [x] **Option B (accept + document)** — DECIDED by Sina 2026-06-11

Once decided: if B, copy the "accept" paragraph into
docs/superpowers/specs/2026-05-29-v230-oblique-ship-gate-rescope.md as a
closing note and tick the M2 DoD box; if A, write a dedicated spec + plan
(it is NOT part of the current M2 test arc).

## G1.6 audit note (closed by this arc)

Remount (S1-S4, crystal frame) x oblique (lab frame) was confirmed
mechanically orthogonal and is now pinned by
tests/test_oblique_remount_wall.py. viz/mosaicity.py is geometry-agnostic
(consumes ctx-derived xl_start); no oblique-specific code needed there.

## Backlog note from the M2 test-arc reviews (2026-06-11)

The oblique identify e2e tests assert the geometry attrs round-trip
(geometry_mode/eta/theta/mount), but nothing guards a future desync between
`geometry.eta` (drives the provenance attr) and `reciprocal.eta` (drives the
analytic-backend physics). The TOML loader sets both from [geometry];
programmatic construction must set both by hand. A cheap future guard would
be an oblique-vs-simplified image-DIFFERENCE assertion in
tests/test_identification_oblique_e2e.py; filed here so it is not lost.
