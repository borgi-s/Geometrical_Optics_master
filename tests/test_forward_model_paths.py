"""Default kernel & Fg-cache paths must resolve relative to the repo root.

The previous implementation used ``sys.path[0]``, which silently broke when
the module was imported from outside the repo root (e.g. via the installed
``dfxm-forward`` console entry point or a fresh ``python -c`` invocation).
This regression test pins the new ``__file__``-relative derivation.
"""

from __future__ import annotations

from pathlib import Path

from dfxm_geo.direct_space import forward_model as fm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pkl_fpath_resolves_relative_to_repo_root():
    """pkl_fpath must point at <repo>/reciprocal_space/pkl_files/ regardless of cwd."""
    expected = REPO_ROOT / "reciprocal_space" / "pkl_files"
    # forward_model normalises with a trailing slash for legacy compat.
    assert Path(fm.pkl_fpath).resolve() == expected.resolve()


def test_repo_root_is_a_real_directory():
    """The derived repo root must actually exist on the filesystem."""
    repo_root = Path(fm.__file__).resolve().parents[3]
    assert repo_root.is_dir()
    # Sanity: pyproject.toml is in the repo root.
    assert (repo_root / "pyproject.toml").is_file()
