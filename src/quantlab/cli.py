"""QuantLab command-line entry point.

Sub-commands are registered here as later phases add them (bench,
profile-quants, kl, analyze, audit-refusals, adversarial, serve).
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from .config import load_config

_REQUIRED_SERVER_FLAGS = ["--model", "--host", "--port", "--ctx-size",
                         "--parallel", "--gpu-layers", "--seed", "--jinja"]
_REQUIRED_PERPLEXITY_FLAGS = ["--kl-divergence", "--kl-divergence-base"]


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


def _cmd_setup_llama(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    build = cfg["llama"]["build"]
    llama_dir = Path(cfg["paths"]["llama_dir"])
    server_exe = llama_dir / "llama-server.exe"
    perplexity_exe = llama_dir / "llama-perplexity.exe"
    build_num = build.lstrip("b")

    if not args.force and server_exe.is_file() and perplexity_exe.is_file():
        version_proc = subprocess.run([str(server_exe), "--version"],
                                      capture_output=True, text=True)
        # this build writes --version to stderr, not stdout -- check both
        # rather than assume either convention.
        version_text = version_proc.stdout + version_proc.stderr
        if f"build {build_num}" in version_text:
            print(f"llama.cpp already set up at {llama_dir} (build {build}); "
                 f"skipping download. Use --force to redo.")
            return 0

    import requests  # local import: only needed for this one-time setup

    llama_dir.mkdir(parents=True, exist_ok=True)
    base = f"https://github.com/ggml-org/llama.cpp/releases/download/{build}"
    assets = [f"llama-{build}-bin-win-cuda-12.4-x64.zip",
             "cudart-llama-bin-win-cuda-12.4-x64.zip"]
    for asset in assets:
        url = f"{base}/{asset}"
        resp = requests.get(url, timeout=180)
        if resp.status_code != 200:
            print(f"setup-llama failed: {url} returned HTTP {resp.status_code}",
                 file=sys.stderr)
            return 1
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(llama_dir)

    if not server_exe.is_file() or not perplexity_exe.is_file():
        print("setup-llama failed: expected executables not found after extraction",
             file=sys.stderr)
        return 1

    version_proc = subprocess.run([str(server_exe), "--version"], capture_output=True, text=True)
    help_proc = subprocess.run([str(server_exe), "--help"], capture_output=True, text=True)
    perp_help_proc = subprocess.run([str(perplexity_exe), "--help"], capture_output=True, text=True)

    missing = [f for f in _REQUIRED_SERVER_FLAGS if f not in help_proc.stdout]
    if missing:
        print(f"setup-llama failed: llama-server --help is missing {missing}", file=sys.stderr)
        return 1
    missing = [f for f in _REQUIRED_PERPLEXITY_FLAGS if f not in perp_help_proc.stdout]
    if missing:
        print(f"setup-llama failed: llama-perplexity --help is missing {missing}", file=sys.stderr)
        return 1

    log_dir = Path(cfg["paths"]["results_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "llama-server-help.txt", "w", encoding="utf-8") as f:
        f.write("=== llama-server --version ===\n")
        f.write(version_proc.stdout + version_proc.stderr)
        f.write("\n=== llama-server --help ===\n")
        f.write(help_proc.stdout + help_proc.stderr)

    version_text = (version_proc.stdout + version_proc.stderr).strip()
    (llama_dir / "VERSION.txt").write_text(
        f"build tag: {build}\n--version output: {version_text}\n",
        encoding="utf-8",
    )
    print(f"llama.cpp {build} set up at {llama_dir}")
    return 0


def _cmd_download_models(args: argparse.Namespace) -> int:
    from .quants import LADDER, QuantsError, download

    rungs = args.rungs.split(",") if args.rungs else list(LADDER)
    for r in rungs:
        if r not in LADDER:
            print(f"download-models failed: unknown rung {r!r}", file=sys.stderr)
            return 1
    try:
        download(rungs)
    except QuantsError as exc:
        print(f"download-models failed: {exc}", file=sys.stderr)
        return 1
    print(f"downloaded and verified: {', '.join(rungs)}")
    return 0


def _cmd_generate_tasks(args: argparse.Namespace) -> int:
    from . import generate
    from .tasks import load_seed_files, consolidate

    cfg = load_config(args.config)
    try:
        generate.probe_teacher()
    except generate.GenerationError as exc:
        print(f"generate-tasks failed: {exc}", file=sys.stderr)
        return 1
    print("probe OK: teacher is reachable")

    seed_path = Path(args.seed)
    if seed_path.is_file():
        seed_tasks = [json.loads(l) for l in open(seed_path, encoding="utf-8") if l.strip()]
    else:
        seed_tasks = consolidate(load_seed_files(cfg["paths"]["cheaproute_root"]))

    generated_path = Path("data/tasks/generated.jsonl")
    if generated_path.is_file() and not args.fresh:
        existing_generated = [json.loads(l) for l in open(generated_path, encoding="utf-8") if l.strip()]
        print(f"resuming: {len(existing_generated)} generated tasks already exist; "
             f"topping up shortfall cells only", file=sys.stderr)
        try:
            result = generate.resume_generation(seed_tasks, existing_generated)
        except generate.BudgetExhausted as exc:
            print(f"generate-tasks: budget exhausted ({exc}); writing partial results", file=sys.stderr)
            result = {"rows": existing_generated, "cell_log": {}, "budget_exhausted_at": str(exc)}
    else:
        try:
            result = generate.run_generation(seed_tasks)
        except generate.BudgetExhausted as exc:
            print(f"generate-tasks: budget exhausted ({exc}); writing partial results", file=sys.stderr)
            result = {"rows": [], "cell_log": {}, "budget_exhausted_at": str(exc)}

    summary = generate.write_outputs(seed_tasks, result)
    print(json.dumps({
        "total": summary["total"], "api_calls_used": summary["api_calls_used"],
        "api_cap": summary["api_cap"], "budget_exhausted_at": summary["budget_exhausted_at"],
    }, ensure_ascii=False, indent=2))
    return 0


def _cmd_build_safety(args: argparse.Namespace) -> int:
    from . import safety

    cfg = load_config(args.config)
    xstest_path = Path(cfg["paths"]["safety_tasks"]).parent / "xstest_prompts.csv"
    try:
        safety.download_xstest(xstest_path)
        all_tasks = [json.loads(l) for l in open(args.tasks, encoding="utf-8") if l.strip()]
        tasks = safety.build_safety_slice(all_tasks, xstest_path, cfg["paths"]["safety_tasks"])
    except safety.SafetyError as exc:
        print(f"build-safety failed: {exc}", file=sys.stderr)
        return 1

    n_xstest = sum(1 for t in tasks if t["subtype"] in ("harmful", "benign_lookalike"))
    n_injection = sum(1 for t in tasks if t["subtype"] == "injection")
    splits = {}
    for t in tasks:
        splits[t["split"]] = splits.get(t["split"], 0) + 1
    print(f"wrote {len(tasks)} safety tasks -> {cfg['paths']['safety_tasks']}")
    print(f"  xstest: {n_xstest}, injection: {n_injection}")
    print(f"  splits: {splits}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from .bench import BenchError, run_bench
    from .quants import LADDER

    rungs = args.rungs.split(",") if args.rungs else list(LADDER)
    for r in rungs:
        if r not in LADDER:
            print(f"bench failed: unknown rung {r!r}", file=sys.stderr)
            return 1
    task_ids = args.task_ids.split(",") if args.task_ids else None
    try:
        stats = run_bench(
            rungs=rungs, slice_name=args.slice, limit=args.limit,
            task_ids=task_ids, stop_after=args.stop_after, out_path=args.out,
        )
    except BenchError as exc:
        print(f"bench failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def _cmd_verify_models(args: argparse.Namespace) -> int:
    from .quants import LADDER, QuantsError, verify

    rungs = args.rungs.split(",") if args.rungs else list(LADDER)
    all_ok = True
    for r in rungs:
        try:
            ok = verify(r, full=args.full)
        except QuantsError as exc:
            print(f"  {r}: ERROR ({exc})", file=sys.stderr)
            all_ok = False
            continue
        print(f"  {r}: {'OK' if ok else 'FAILED'}")
        all_ok = all_ok and ok
    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantlab")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    subparsers = parser.add_subparsers(dest="command")

    p_seeds = subparsers.add_parser(
        "build-seeds", help="consolidate the CheapRoute seed tasks into data/tasks/seed.jsonl")
    p_seeds.add_argument("--out", default="data/tasks/seed.jsonl")
    p_seeds.add_argument("--summary-out", default="results/seed_summary.json")
    p_seeds.set_defaults(func=_cmd_build_seeds)

    p_llama = subparsers.add_parser(
        "setup-llama", help="download and verify the pinned llama.cpp CUDA build")
    p_llama.add_argument("--force", action="store_true",
                         help="re-download even if already set up")
    p_llama.set_defaults(func=_cmd_setup_llama)

    p_dl = subparsers.add_parser(
        "download-models", help="download and hash-verify GGUF rungs")
    p_dl.add_argument("--rungs", default=None, help="comma-separated rung names (default: all)")
    p_dl.set_defaults(func=_cmd_download_models)

    p_gen = subparsers.add_parser(
        "generate-tasks", help="generate, verify and merge benchmark tasks with the Fireworks teacher")
    p_gen.add_argument("--seed", default="data/tasks/seed.jsonl")
    p_gen.add_argument("--fresh", action="store_true",
                       help="ignore any existing data/tasks/generated.jsonl and start over")
    p_gen.set_defaults(func=_cmd_generate_tasks)

    p_safety = subparsers.add_parser(
        "build-safety", help="build the safety slice from XSTest and constructed injections")
    p_safety.add_argument("--tasks", default="data/tasks/all.jsonl")
    p_safety.set_defaults(func=_cmd_build_safety)

    p_bench = subparsers.add_parser(
        "bench", help="run tasks against rungs and record graded rows (resumable)")
    p_bench.add_argument("--rungs", default=None, help="comma-separated rung names (default: all)")
    p_bench.add_argument("--slice", default="bench", choices=["bench", "safety", "all"])
    p_bench.add_argument("--limit", type=int, default=None)
    p_bench.add_argument("--task-ids", default=None, help="comma-separated task ids")
    p_bench.add_argument("--stop-after", type=int, default=None,
                         help="stop after this many new rows (for testing resume)")
    p_bench.add_argument("--out", default="results/raw/grid.jsonl")
    p_bench.set_defaults(func=_cmd_bench)

    p_verify = subparsers.add_parser(
        "verify-models", help="verify downloaded GGUF rungs against their pinned hashes")
    p_verify.add_argument("--rungs", default=None, help="comma-separated rung names (default: all)")
    p_verify.add_argument("--full", action="store_true", help="always re-hash, ignore the cache")
    p_verify.set_defaults(func=_cmd_verify_models)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
