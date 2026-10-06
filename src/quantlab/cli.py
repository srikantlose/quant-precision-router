"""QuantLab command-line entry point.

Sub-commands are registered here as later phases add them (build-seeds,
build-safety, setup-llama, download-models, verify-models, bench, profile-quants,
kl, analyze, audit-refusals, adversarial, serve). For now this is a stub that
parses no sub-commands yet.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantlab")
    subparsers = parser.add_subparsers(dest="command")
    # Sub-commands are added by later phases.
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
