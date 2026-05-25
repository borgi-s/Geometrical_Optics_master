# Comparing the MC and analytic resolution backends

`scripts/compare_resolution_backends.py` renders a 2×5 figure that overlays the
Monte-Carlo reciprocal-space resolution function against the closed-form
analytic backend (v2.1.0), so you can confirm by eye and by number that the two
agree before merging the analytic backend.

- **Top row:** Monte-Carlo central-slice profile of each q-component.
- **Bottom row:** the analytic density evaluated along the same axis.
- **Columns:** `qrock | qroll | qpar` (crystal frame) and `qrock' | q2θ`
  (imaging frame). `qroll` is shared by both frames.
- The analytic curve is also overlaid (red dashed) on each MC panel, and each
  column is annotated with its peak `max|MC − analytic|`. A summary table also
  prints to stdout.

Both backends share **one** Bragg angle and **one** set of instrument
parameters (pulled from `ReciprocalConfig` / `generate_kernel`), so the two
sides cannot silently drift apart.

> **Beamstop is OFF.** The analytic backend only exists for the no-beamstop
> regime — it raises if a beamstop is configured. The comparison therefore runs
> with the beamstop disabled; that is the regime the analytic backend is used in
> by the pipeline anyway.

## 1. Get the branch onto the cluster

```bash
cd <your repo checkout>
git fetch origin
git checkout feature/analytic-resolution      # or: git pull, if already on it
pip install -e ".[dev]"                        # editable reinstall -> v2.1.0
python -c "import dfxm_geo, importlib.metadata as m; print(m.version('dfxm_geo'))"
#   -> 2.1.0
```

## 2. Run it

A high ray count is exactly what makes the central slices smooth. Start with
5e8 and push to 1e9 if a node has the headroom:

```bash
python scripts/compare_resolution_backends.py --nrays 1e9 --chunk 1e8
```

This writes `docs/img/resolution_backend_comparison.png` and prints the residual
table. Useful flags:

| flag | default | meaning |
|------|---------|---------|
| `--nrays` | `1e8` | total MC rays |
| `--chunk` | `1e8` | rays per chunk (memory is flat in this, not in `--nrays`) |
| `--nbins` | `151` | histogram bins per component |
| `--slab-frac` | `0.05` | central-slice half-thickness, as a fraction of each held-out axis half-width. Smaller ⇒ truer slice but noisier; larger ⇒ smoother but a thicker slab. |
| `--out` | `docs/img/resolution_backend_comparison.png` | output PNG |
| `--hkl` / `--keV` | `-1 1 -1` / `17` | reflection (θ is recomputed) |
| `--seed` | `0` | RNG seed |

**Memory:** rays are sampled in chunks of `--chunk`, histogrammed, and thrown
away, so peak RAM is set by the chunk size, not the total. A chunk of `1e8`
needs roughly **10–12 GB**; if a node is tight, drop to `--chunk 5e7` or `2e7` —
the result is identical, it just loops more times.

**Runtime:** dominated by the MC sampling (`scipy` truncated normals). 1e9 rays
is ~10 chunks; budget a few minutes per 1e8 chunk depending on the node.

## 3. View the PNG over SSH

The script uses matplotlib's headless `Agg` backend — it never opens a window,
it just writes a `.png`. To look at it from your laptop:

- **VSCode Remote-SSH (easiest).** Connect to the cluster in VSCode
  (`Remote-SSH: Connect to Host…`), open the repo folder, and click the PNG in
  the Explorer — it renders inline in the editor. Re-run and click again to
  refresh.
- **`scp` it back.** From your laptop:
  ```bash
  scp user@cluster:/path/to/repo/docs/img/resolution_backend_comparison.png .
  ```
  then open it locally.
- **X11 forwarding** (`ssh -X`) works in principle, but only if you switch the
  script off `Agg` and call `plt.show()` — it's slow and flaky over a WAN. Saving
  the PNG and using one of the two options above is strongly preferred.

## 4. Reading the result

At 1e9 rays the MC profiles should sit essentially on top of the analytic
curves, with the residuals dropping well into the low single-digit percent (the
2e6-ray smoke test shows ~15–28%, which is almost entirely shot noise in the
thin central-slice tubes — that is what the high ray count is for).

Two differences are **expected and not bugs**, and are noted on the figure:

1. **NA square aperture.** The analytic backend drops the objective NA aperture
   (`phys_aper`), validated to <1% vs MC. You will see slightly fuller tails in
   the MC `qroll`/`q2θ` panels where the aperture clips the Monte-Carlo rays.
2. **ζ_v truncation.** The vertical divergence is hard-cut at ±140 µrad in both
   backends, which is what gives `qrock`/`qrock'` their flat-topped shape.

If the *cores* (near q = 0) of the top and bottom rows disagree by more than a
percent or two at 1e9 rays, that's worth investigating; tail differences in
`qroll`/`q2θ` are the aperture approximation and are fine.
