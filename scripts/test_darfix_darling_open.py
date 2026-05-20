"""Open-it test for v1.1.0 HDF5 outputs in silx + darfix + darling.

The PR #10 sign-off gate: confirm that `dfxm_geo.h5` files produced by
`dfxm-forward` can be loaded by the ESRF analysis stack without patches.

Three layers, ordered by what they actually prove:

  1. silx       — the HDF5/BLISS reader that darfix and darling are built
                  on. If silx opens the file with the expected structure,
                  the layout is correct. This is the schema gate.
  2. darfix     — the higher-level dataset wrapper. Opens are blocked by
                  darfix's own undeclared dependencies in some versions
                  (pip install scikit-image if you hit that).
  3. darling    — separate DFXM analysis library.

Usage (from repo root, with the venv active):
    python scripts/test_darfix_darling_open.py
    python scripts/test_darfix_darling_open.py <h5_1> <h5_2> [...]

Exit code: 0 if no actual FAIL on any installed layer for any file
(SKIP for missing libraries is fine), 1 otherwise.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Result:
    library: str
    path: Path
    status: str  # "OK", "FAIL", "SKIP"
    detail: str


def find_default_paths() -> list[Path]:
    """Pick up dfxm_geo*.h5 from the most recent laptop_* and cluster_* run
    dirs under output/. Tolerates dfxm_geo.h5 and dfxm_geo_N.h5 naming."""
    laptops = sorted(Path("output").glob("laptop_*/dfxm_geo*.h5"))
    clusters = sorted(Path("output").glob("cluster_*/dfxm_geo*.h5"))
    paths: list[Path] = []
    if laptops:
        paths.append(laptops[-1])
    if clusters:
        paths.append(clusters[-1])
    if not paths:
        raise FileNotFoundError("no laptop_*/dfxm_geo*.h5 or cluster_*/dfxm_geo*.h5 under output/")
    return paths


def try_silx(path: Path) -> Result:
    """Schema gate: silx.io.open + verify BLISS-style structure."""
    try:
        import silx.io  # type: ignore[import-not-found]
    except ImportError:
        return Result("silx", path, "SKIP", "silx not installed")
    try:
        with silx.io.open(str(path)) as f:
            scans = sorted(k for k in f if k.endswith(".1"))
            if "1.1" not in scans:
                return Result("silx", path, "FAIL", f"no /1.1 scan group; top-level={list(f)}")
            entry = f["/1.1"]
            nx_class = entry.attrs.get("NX_class")
            if nx_class != "NXentry" and nx_class != b"NXentry":
                return Result(
                    "silx",
                    path,
                    "FAIL",
                    f"/1.1 has NX_class={nx_class!r}, expected NXentry",
                )
            det = entry["instrument/dfxm_sim_detector/data"]
            shape = tuple(det.shape)
            return Result(
                "silx",
                path,
                "OK",
                f"scans={scans} detector_shape={shape} dtype={det.dtype}",
            )
    except Exception as e:
        return Result("silx", path, "FAIL", f"{type(e).__name__}: {e}")


def try_darfix(path: Path) -> Result:
    """Open via whichever darfix API resolves. Tries several known
    entry points across darfix major versions."""
    try:
        import darfix  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return Result(
            "darfix",
            path,
            "SKIP",
            "darfix not installed (pip install darfix)",
        )

    # Common pip-name overrides for ModuleNotFoundError. The import name is
    # NOT the pip name for these.
    pip_name = {
        "skimage": "scikit-image",
        "cv2": "opencv-python",
        "sklearn": "scikit-learn",
        "PIL": "Pillow",
    }

    def _make_image_dataset() -> object:
        """darfix 2.x: build an ImageDataset from a DataUrl pointing at /1.1's
        detector data. Requires a working dir for darfix's bookkeeping."""
        import tempfile

        from darfix.core.dataset import ImageDataset  # type: ignore[import-not-found]
        from silx.io.url import DataUrl  # type: ignore[import-not-found]

        url = DataUrl(
            file_path=str(path),
            data_path="/1.1/instrument/dfxm_sim_detector/data",
            scheme="silx",
        )
        # Use a fresh tmp dir per call; darfix scribbles state into it.
        tmp_dir = tempfile.mkdtemp(prefix="darfix_open_it_")
        return ImageDataset(_dir=tmp_dir, detector_url=url)

    def _legacy_open_h5() -> object:
        from darfix.io.utils import open_h5  # type: ignore[import-not-found]

        return open_h5(str(path), "/1.1")

    # Try the actual current API first, then older fallbacks.
    candidates = [
        ("darfix.core.dataset.ImageDataset (via DataUrl)", _make_image_dataset),
        ("darfix.io.utils.open_h5 (pre-2.x)", _legacy_open_h5),
    ]

    last_err = None
    for api_name, make in candidates:
        try:
            obj = make()
            return Result("darfix", path, "OK", f"via {api_name} -> {type(obj).__name__}")
        except ModuleNotFoundError as e:
            # Missing transitive dep (e.g. skimage). Worth surfacing — not the
            # same as "library not installed".
            missing = e.name or "?"
            install_name = pip_name.get(missing, missing)
            last_err = f"{api_name}: missing dep {missing!r} (try: pip install {install_name})"
        except (ImportError, AttributeError) as e:
            last_err = f"{api_name}: {type(e).__name__}: {e}"
        except Exception as e:
            last_err = f"{api_name}: {type(e).__name__}: {e}"

    return Result("darfix", path, "FAIL", last_err or "no API matched")


def try_darling(path: Path) -> Result:
    try:
        import darling  # type: ignore[import-not-found]
    except ImportError:
        return Result(
            "darling",
            path,
            "SKIP",
            "darling not installed (pip install git+https://github.com/AxelHenningsson/darling.git)",
        )
    try:
        d = darling.DataSet(str(path), scan_id="1.1")
        motors = list(d.motors) if hasattr(d, "motors") else []
        shape = tuple(d.data.shape) if hasattr(d, "data") else None
        return Result("darling", path, "OK", f"data {shape}, motors {motors}")
    except Exception as e:
        return Result("darling", path, "FAIL", f"{type(e).__name__}: {e}")


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else find_default_paths()

    missing = [p for p in paths if not p.is_file()]
    if missing:
        for p in missing:
            print(f"ERROR: file not found: {p}")
        return 1

    print("Targets:")
    for p in paths:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p}  ({size_mb:.0f} MB)")
    print()

    libraries: list[tuple[str, Callable[[Path], Result]]] = [
        ("silx (schema gate)", try_silx),
        ("darfix", try_darfix),
        ("darling", try_darling),
    ]

    results: list[Result] = []
    for lib_name, runner in libraries:
        print(f"--- {lib_name} ---")
        for p in paths:
            r = runner(p)
            results.append(r)
            marker = {"OK": "PASS", "FAIL": "FAIL", "SKIP": "skip"}[r.status]
            print(f"  [{marker}] {p.name:30s}  {r.detail}")
        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'library':12s} {'file':40s} {'verdict':10s}")
    print("-" * 70)
    for r in results:
        print(f"{r.library:12s} {r.path.name:40s} {r.status:10s}")
    print()

    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_skip = sum(1 for r in results if r.status == "SKIP")
    n_ok = sum(1 for r in results if r.status == "OK")
    print(f"{n_ok} ok, {n_fail} failed, {n_skip} skipped")

    if n_fail:
        print()
        print("Failing details (paste these back if reporting):")
        for r in results:
            if r.status == "FAIL":
                print(f"  {r.library} on {r.path}:")
                print(f"    {r.detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
