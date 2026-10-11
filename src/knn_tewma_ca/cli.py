"""Command line entry point.

    knn-tewma-ca                # everything: reproduce + corrected, reports and charts
    knn-tewma-ca reproduce      # only the Round 2 presentation numbers
    knn-tewma-ca corrected      # only the multi-seed, allocated-end version
    knn-tewma-ca --seeds 10 --no-plots
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from . import experiments as ex
from . import report

DEFAULT_OUT = Path("results")


def _log(quiet: bool):
    def log(msg: str) -> None:
        if not quiet:
            print(msg, file=sys.stderr, flush=True)

    return log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knn-tewma-ca",
        description="Run the kNN-TEWMA-CA scheduling experiments (E1-E4).",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("all", "reproduce", "corrected"),
        help="what to run (default: all)",
    )
    parser.add_argument("--seeds", type=int, default=ex.DEFAULT_SEED_COUNT,
                        help=f"seeds per experiment for the corrected runs (default: {ex.DEFAULT_SEED_COUNT})")
    parser.add_argument("--first-seed", type=int, default=0, help="first seed of the corrected runs (default: 0)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory (default: results)")
    parser.add_argument("--no-plots", action="store_true", help="skip chart generation")
    parser.add_argument("--quiet", action="store_true", help="only print the final tables")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log = _log(args.quiet)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    started = time.perf_counter()

    all_results = {}
    reproduced = None
    if args.command in ("all", "reproduce"):
        log("Reproducing the Round 2 presentation...")
        reproduced = ex.run_mode(ex.REPRODUCE, log)
        all_results[ex.REPRODUCE.key] = reproduced
        text = report.reproduce_markdown(reproduced)
        (out / "reproduce.md").write_text(text + "\n")
        print(text)
        if not args.no_plots:
            from .plotting import reproduce_figures

            for p in reproduce_figures(reproduced, figures):
                log(f"  wrote {p}")

    if args.command in ("all", "corrected"):
        mode = ex.corrected(args.seeds, args.first_seed)
        if reproduced is None:
            reproduced = ex.run_mode(ex.REPRODUCE)
        log("Presentation seeds with allocated-end reservations...")
        same_seeds = ex.run_mode(ex.SAME_SEEDS_ALLOCATED_END, log)
        log(f"Corrected methodology over {args.seeds} seeds...")
        main_results = ex.run_mode(mode, log)
        all_results[ex.SAME_SEEDS_ALLOCATED_END.key] = same_seeds
        all_results[mode.key] = main_results

        corrected_text = report.corrected_markdown(main_results, mode)
        comparison_text = report.comparison_markdown(reproduced, same_seeds, main_results, mode)
        (out / "corrected.md").write_text(corrected_text + "\n")
        (out / "comparison.md").write_text(comparison_text + "\n")
        print()
        print(comparison_text)
        if not args.no_plots:
            from .plotting import corrected_figures

            for p in corrected_figures(main_results, mode, reproduced, figures):
                log(f"  wrote {p}")

    report.write_runs_csv(out / "runs.csv", all_results)
    log(f"Done in {time.perf_counter() - started:.1f}s. Results are in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
