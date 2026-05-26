"""Entry point for `dfxm-init`: scaffold the bundled config templates.

Copies the templates shipped inside `dfxm_geo.data` into a destination
directory (default `./configs/`) so pip-only users have an editable starting
point without cloning the repo.
"""

from __future__ import annotations


def cli_main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    from pathlib import Path

    from dfxm_geo.data import iter_config_files

    parser = argparse.ArgumentParser(
        prog="dfxm-init",
        description=(
            "Write the bundled DFXM config templates (default.toml, "
            "identification_*.toml, variants/*) into a directory you can edit. "
            "Existing files are left untouched unless --force is given."
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("configs"),
        help="Destination directory for the templates (default: ./configs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite template files that already exist at the destination.",
    )
    args = parser.parse_args(argv)

    written = 0
    skipped = 0
    for rel, src in iter_config_files():
        target = args.dest / rel
        if target.exists() and not args.force:
            print(f"skip (exists): {target}")
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(src.read_bytes())
        print(f"wrote: {target}")
        written += 1

    summary = f"dfxm-init: wrote {written} file(s), skipped {skipped}."
    if skipped and not args.force:
        summary += " Re-run with --force to overwrite skipped files."
    print(summary, file=sys.stderr)
    return 0
