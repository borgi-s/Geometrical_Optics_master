# Legacy entry points

Files in this directory are preserved verbatim from before the Phase 6 / 9.2
refactor. They are not maintained — use them as historical reference, not as
recommended workflows.

- `init_forward.py` — the single-file demo script that was the original
  entry point. Reproduce the same paper figures via:

      dfxm-forward --config configs/default.toml --output output/

  See `docs/reproducibility.md` for the supported flow.
