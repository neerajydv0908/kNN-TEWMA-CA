"""Command line entry point.

    knn-tewma-ca                # everything: reproduce + corrected + ablation, reports and charts
    knn-tewma-ca reproduce      # only the Round 2 presentation numbers
    knn-tewma-ca corrected      # only the corrected, multi-seed version (with the ablation)
    knn-tewma-ca --seeds 10 --workers 4 --no-plots
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
        description="Run the kNN-TEWMA-CA scheduling experiments (E1-E5 and the ablation).",
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
    parser.add_argument("--workers", type=int, default=ex.default_workers(),
                        help="parallel worker processes (default: number of CPUs, up to 8)")
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

    log("Reproducing the Round 2 presentation...")
    reproduced = ex.run_mode(ex.REPRODUCE, workers=1)
    csv_groups = [reproduced]
    if args.command in ("all", "reproduce"):
        text = report.reproduce_markdown(reproduced)
        (out / "reproduce.md").write_text(text + "\n")
        print(text)
        if not args.no_plots:
            from .plotting import reproduce_figures

            for p in reproduce_figures(reproduced, figures):
                log(f"  wrote {p}")

    if args.command in ("all", "corrected"):
        mode = ex.corrected(args.seeds, args.first_seed)
        log(f"Corrected methodology over {args.seeds} seeds ({args.workers} workers)...")
        main_results = ex.run_mode(mode, args.workers, log)
        log("Ablation: applying the fixes one at a time...")
        ablation = ex.run_ablation(mode.e1_seeds, args.workers, final=main_results, progress=log)
        csv_groups.append(main_results)
        csv_groups += [ablation[key] for key, _l, _m in ex.ABLATION_STEPS if key != "corrected"]

        corrected_text = report.corrected_markdown(main_results, mode)
        comparison_text = report.comparison_markdown(reproduced, main_results, ablation, mode)
        (out / "corrected.md").write_text(corrected_text + "\n")
        (out / "comparison.md").write_text(comparison_text + "\n")
        print()
        print(comparison_text)
        if not args.no_plots:
            from .plotting import ablation_figure, corrected_figures

            for p in corrected_figures(main_results, mode, reproduced, figures):
                log(f"  wrote {p}")
            log(f"  wrote {ablation_figure(ablation, mode.seed_count, figures)}")

    if args.command in ("all", "corrected"):
        # Only with the corrected runs, so it always matches corrected.md.
        report.write_runs_csv(out / "runs.csv", csv_groups)
    log(f"Done in {time.perf_counter() - started:.1f}s. Results are in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
