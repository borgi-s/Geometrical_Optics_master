"""Run the all-pairs identifiability study over identify master(s).

Example:
    python scripts/run_identify_study.py path/to/masters_dir out/ --plots
"""

from __future__ import annotations

import argparse

from dfxm_geo.scoring import Identifier, load_library


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="DFXM dislocation identifiability study")
    p.add_argument("library", help="identify master .h5 or a directory of masters")
    p.add_argument("out_dir", help="output directory for matrix/metrics/plots")
    p.add_argument("--normalize", default="symmetric", choices=["symmetric", "diagonal", "none"])
    p.add_argument(
        "--class-key",
        default="plane_burgers",
        choices=["plane_burgers", "burgers", "plane_burgers_alpha"],
    )
    p.add_argument("--include-invisible", action="store_true")
    p.add_argument("--backend", default="auto", choices=["auto", "numpy", "torch"])
    p.add_argument("--plots", action="store_true")
    a = p.parse_args(argv)

    lib = load_library(a.library, include_invisible=a.include_invisible)
    ident = Identifier(lib, normalize=a.normalize, backend=a.backend, class_key_mode=a.class_key)
    res = ident.study()
    res.save(a.out_dir, plots=a.plots)
    print(f"top-1 accuracy: {res.top1_accuracy:.4f} over {len(lib)} candidates")


if __name__ == "__main__":
    main()
