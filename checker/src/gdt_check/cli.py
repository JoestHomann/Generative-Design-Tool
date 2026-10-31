"""Command line: gdt-check CERTIFICATE.json"""

from __future__ import annotations

import argparse

from . import check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gdt-check", description="Independently re-verify a design certificate.")
    parser.add_argument("certificate", help="certificate file (JSON)")
    args = parser.parse_args(argv)
    report = check(args.certificate)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
