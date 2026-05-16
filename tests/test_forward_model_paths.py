"""Default kernel & Fg-cache paths must resolve relative to the repo root.

The previous implementation used ``sys.path[0]``, which silently broke when
the module was imported from outside the repo root (e.g. via the installed
``dfxm-forward`` console entry point or a fresh ``python -c`` invocation).
This regression test pins the new ``__file__``-relative derivation.
"""

from __future__ import annotations

import os
import subprocess
import sys
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


def test_pkl_fpath_independent_of_cwd(tmp_path):
    """Importing forward_model from an unrelated cwd must still resolve
    pkl_fpath to <repo>/reciprocal_space/pkl_files/.

    Stronger regression guard than the same-process check above: spawns a
    subprocess in an unrelated tmp_path cwd (mimicking the original failure
    mode where `dfxm-forward` was invoked via the installed entry point or
    `python -c` from /tmp). With the pre-cleanup `pkl_fpath = sys.path[0]
    + "/reciprocal_space/pkl_files/"` derivation, this would resolve to
    absolute `/reciprocal_space/...` and forward() would silently skip the
    kernel autoload.
    """
    # Capture both the discovered _REPO_ROOT and pkl_fpath from the subprocess.
    # Use a `|` separator so we can tolerate Windows backslashes in the paths.
    code = (
        "import dfxm_geo.direct_space.forward_model as fm; "
        "print(fm._REPO_ROOT, fm.pkl_fpath, sep='|')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    # forward_model prints a "Loading kernel from" or "default kernel npz not
    # found" line during import; strip it by parsing only the last stdout line.
    stdout_line = [line for line in result.stdout.strip().splitlines() if "|" in line][-1]
    repo_root_str, pkl_fpath = stdout_line.split("|", 1)

    repo_root = Path(repo_root_str)
    # The derived repo root must contain the project marker, not be /tmp/etc.
    assert (repo_root / "pyproject.toml").is_file(), (
        f"_REPO_ROOT {repo_root!r} doesn't contain pyproject.toml — "
        "derivation is cwd-dependent, regressing the sys.path[0] fix."
    )
    expected_pkl_fpath = str(repo_root / "reciprocal_space" / "pkl_files") + os.sep
    assert pkl_fpath == expected_pkl_fpath, (
        f"pkl_fpath {pkl_fpath!r} != expected {expected_pkl_fpath!r}; "
        "derivation broke under a non-repo cwd."
    )
