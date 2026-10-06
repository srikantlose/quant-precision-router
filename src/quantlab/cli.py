"""QuantLab command-line entry point.

Sub-commands are registered here as later phases add them (build-safety,
setup-llama, download-models, verify-models, bench, profile-quants, kl,
analyze, audit-refusals, adversarial, serve).
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config


def _cmd_build_seeds(args: argparse.Namespace) -> int:
    from .tasks import SeedConsolidationError, build_seeds

    cfg = load_config(args.config)
    try:
        tasks = build_seeds(
            cheaproute_root=cfg["paths"]["cheaproute_root"],
            seed_out=args.out,
            summary_out=args.summary_out,
        )
    except SeedConsolidationError as exc:
        print(f"build-seeds failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(tasks)} seed tasks -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantlab")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    subparsers = parser.add_subparsers(dest="command")

    p_seeds = subparsers.add_parser(
        "build-seeds", help="consolidate the CheapRoute seed tasks into data/tasks/seed.jsonl")
    p_seeds.add_argument("--out", default="data/tasks/seed.jsonl")
    p_seeds.add_argument("--summary-out", default="results/seed_summary.json")
    p_seeds.set_defaults(func=_cmd_build_seeds)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
