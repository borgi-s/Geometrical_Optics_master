# DFXM v2.1.0 — Grid-free analytic reciprocal-space resolution backend

Released: 2026-05-25.

## Headline: closed-form resolution function replaces the Monte-Carlo LUT

`forward()` now has a grid-free, closed-form reciprocal-space resolution
backend (`AnalyticResolution` in
`src/dfxm_geo/reciprocal_space/analytic_resolution.py`). It evaluates the
resolution function `p_Q(q)` analytically — a truncated-Gaussian
erf-difference — instead of looking up a precomputed Monte-Carlo `Resq_i`
histogram. Because there is no histogram grid, there is nothing to quantize
against, so the closed-form path eliminates the COM / rocking-map
grid-quantization **banding** artifact at its source.

See `docs/img/analytic_vs_mc_com.png`: the analytic COM map is banding-free
where the MC-grid COM map shows the stair-step quantization.

## What changed

No-beamstop runs now **default** (`backend = "auto"`) to the closed-form
backend instead of the Monte-Carlo LUT histogram. The MC LUT path remains
for any configuration the closed form cannot represent (a beamstop).

Pipeline dispatch (`pipeline._load_resolution`):

- `backend = "auto"` (default) → analytic when the beamstop is off, MC when
  the beamstop is on.
- `backend = "analytic"` → forces the closed form. Raises if `beamstop =
  true` (the closed form cannot represent a beamstop).
- `backend = "mc"` → forces the Monte-Carlo LUT (the v2.0.x behavior).

Validation: analytic-vs-MC forward parity within shot noise
(corr ≈ 0.985), and the analytic COM map is banding-free against the MC grid.

## BREAKING CHANGE (why 2.1.0)

Configs with `beamstop = false` that previously used the MC kernel now use
the **analytic** backend by default. The resulting images differ from the
v2.0.x MC-grid output (the banding is gone), so this is a behavior change for
no-beamstop runs.

**To restore the old grid behavior**, set:

```toml
[reciprocal]
backend = "mc"
```

**Beamstop runs are unaffected** — they still use the MC LUT.
`backend = "analytic"` together with `beamstop = true` raises (the closed
form cannot represent a beamstop).

## New config keys in `[reciprocal]`

- `backend` — `"auto"` (default) / `"analytic"` / `"mc"`. Selects the
  resolution backend per the dispatch rules above.
- `beamstop` — flag; when off, `"auto"` resolves to the analytic backend.
- Instrument parameters feeding the analytic backend:
  - `zeta_v_fwhm` — vertical divergence FWHM.
  - `zeta_h_fwhm` — horizontal divergence FWHM.
  - `NA_rms` — objective numerical-aperture RMS.
  - `eps_rms` — energy-bandwidth RMS.
  - `zeta_v_clip` — vertical-divergence truncation (the condenser physical
    aperture; sets the erf-difference truncation limits).

## Migrated / updated

- `configs/default.toml`: documented the new `backend` key with an inline
  comment in `[reciprocal]`. The example config keeps `beamstop = true`, so
  it stays on the MC path; existing values are unchanged.

## Unchanged

- Beamstop forward runs (still MC).
- Bit-equivalence safety net (`tests/data/golden/Fd_find_smoke.npy`).
- HDF5 schema.
- Collaborator branches (`Beam_Stop`, `CDD_inc`, `ESRF_DTU`,
  `Purdue_Paper`, `dislocation_identification`) — ground-truth references,
  never touched.

## Upgrade

```bash
pip install --upgrade dfxm-geo==2.1.0
```

If you run no-beamstop forward simulations and depend on the exact v2.0.x
MC-grid output, set `[reciprocal] backend = "mc"`. Otherwise, regenerate
those outputs — the analytic backend removes the COM/rocking-map banding.
