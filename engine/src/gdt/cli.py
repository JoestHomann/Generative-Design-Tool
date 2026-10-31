"""Command line: gdt SETUP.json --out DIR"""

from __future__ import annotations

import argparse
import sys

from . import api
from .setup import SetupError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gdt", description="Optimize a 2D part and certify its optimality gap.")
    parser.add_argument("setup", help="problem setup file (JSON)")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--materials", help="material library (default: data/materials.json)")
    parser.add_argument("--quiet", action="store_true", help="no per-iteration progress")
    args = parser.parse_args(argv)

    def progress(it: int, compliance: float, volume: float, change: float) -> None:
        if not args.quiet:
            print(f"iteration {it:4d}  compliance {compliance:12.6g} N mm  "
                  f"volume {volume:6.3f}  change {change:6.3f}", flush=True)

    try:
        result = api.run(args.setup, args.out, material_library=args.materials,
                         progress=progress)
    except SetupError as err:
        print("The setup cannot be solved:", file=sys.stderr)
        for issue in err.issues:
            print(f"  - {issue}", file=sys.stderr)
        return 2
    b = result.certificate["bounds"]
    if b["status"] == "certified":
        print(f"Certified: compliance {b['upper_bound_Nmm']:.6g} N mm, gap "
              f"{b['gap_percent']:.2f} % (target "
              f"{'met' if b['target']['met'] else 'not met'}).")
    else:
        print(f"Status: {b['status']}")
    print(f"Results written to {args.out}")
    return 0 if b["status"] == "certified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
