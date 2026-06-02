"""간단 CLI 엔트리포인트."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="skku-vqa")
    ap.add_argument("--version", action="store_true")
    args = ap.parse_args()
    if args.version:
        from . import __version__

        print(__version__)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
