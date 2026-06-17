# DFXM v3.0.1 — stdio-server-safe git provenance

Released: 2026-06-17.

A patch release. No API, configuration, dependency, or CLI changes — a drop-in
upgrade from v3.0.0.

---

## Fix

- **`io.hdf5._get_git_sha_and_dirty()` no longer hangs behind an stdio server.**
  On every HDF5 write, `dfxm-geo` records `(git_sha, git_dirty)` by shelling out
  to `git rev-parse HEAD` / `git status --porcelain` via `subprocess.check_output`.
  Those calls did not isolate the child's **stdin**, so the spawned `git`
  inherited the parent process's stdin. When `dfxm-geo` runs embedded behind a
  stdio server (e.g. the `dfxm-geo-mcp` MCP transport), the parent's stdin is a
  live JSON-RPC pipe; the `git` child inheriting that handle made
  `subprocess.communicate()` block forever, hanging the entire write.

  Both `git` calls now pass `stdin=subprocess.DEVNULL`. **Provenance values are
  unchanged** (git still returns the same output); this is pure
  embedded/stdio-safety hardening with no effect on CLI or library behaviour.

  Regression test: `tests/test_hdf5_provenance.py::test_get_git_sha_and_dirty_isolates_subprocess_stdin`.

---

## Upgrade

```bash
pip install --upgrade dfxm-geo==3.0.1
# CIF support: pip install "dfxm-geo[cif]"
```

## Packaging note

No `[project.scripts]`, dependency, or `requires-python` change since v3.0.0, so
the conda-forge feedstock needs no hand-edit — the regro autotick bot's
version + sha256 bump applies as-is (keep `noarch: python`, `gemmi` as the `[cif]`
extra). The PyPI publish remains gated on the `pypi` GitHub Environment manual
approval.
