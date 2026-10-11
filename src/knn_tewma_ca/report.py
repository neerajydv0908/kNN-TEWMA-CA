"""Markdown and CSV reports for experiment results."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .experiments import (
    ABLATION_STEPS,
    ALL_SCHEDULERS,
    CONFIDENCE_LEVELS,
    FIXED_PADDING,
    Mode,
    Run,
    select,
    values,
    xs_in,
)
from .metrics import HIGHER_IS_BETTER, METRIC_LABELS, METRIC_NAMES, SLIDE_METRICS, display_digits
from .schedulers import EasyBaseline, KnnTewmaCA, TewmaScheduler
from .stats import paired_comparison, summarize

Results = Dict[str, List[Run]]
Ablation = Dict[str, Dict[str, List[Run]]]

CA = KnnTewmaCA.name
TEWMA = TewmaScheduler.name
BASELINE = EasyBaseline.name

# Decimal places for multi-seed means and standard deviations.
_AGG_DIGITS = {
    "makespan": 1,
    "avg_wait": 1,
    "avg_turnaround": 1,
    "success_rate": 3,
    "utilization": 3,
    "avg_slowdown": 2,
    "allocation_ratio": 2,
}

GUARANTEE_NOTE = (
    "Reservation violations are not reported as a result. Every backfilled job is "
    "allocated a runtime that ends by the head job's reservation and is killed at its "
    "allocation, so backfilling cannot delay a reservation under any of the three "
    "schedulers. The simulator re-checks this after every backfill round and raises "
    "`ReservationGuaranteeError` if it ever fails."
)


# -- formatting ---------------------------------------------------------------
def _fixed(value: float, metric: str) -> str:
    return f"{value:.{display_digits(metric)}f}"


def _pm(vals: Sequence[float], metric: str) -> str:
    s = summarize(vals)
    d = _AGG_DIGITS[metric]
    if s.n == 1:
        return f"{s.mean:.{d}f}"
    return f"{s.mean:.{d}f} ± {s.std:.{d}f}"


def _mean(vals: Sequence[float], metric: str) -> str:
    return f"{summarize(vals).mean:.{_AGG_DIGITS[metric]}f}"


def _pct(fraction: float) -> str:
    return f"{fraction * 100:+.1f}%"


def _ci(summary, metric: str) -> str:
    d = _AGG_DIGITS[metric]
    return f"{summary.mean:+.{d}f} [{summary.ci_low:+.{d}f}, {summary.ci_high:+.{d}f}]"


def _ci_pct(summary) -> str:
    return f"{summary.mean * 100:+.1f}% [{summary.ci_low * 100:+.1f}, {summary.ci_high * 100:+.1f}]"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], align: Optional[str] = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    marks = {"l": ":--", "r": "--:", "c": ":-:"}
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(marks[a] for a in align) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _seed_text(seeds: Sequence[int]) -> str:
    seeds = sorted(seeds)
    if len(seeds) > 2 and seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}–{seeds[-1]}"
    return ", ".join(str(s) for s in seeds)


def _rounded(run: Run, metric: str) -> float:
    return run.metrics.rounded()[metric]


def _better(pc, metric: str) -> str:
    return "—" if HIGHER_IS_BETTER[metric] is None else f"{pc.better}/{pc.n}"


# -- reproduce ----------------------------------------------------------------
def reproduce_markdown(results: Results) -> str:
    """Tables in the same shape as the Round 2 slides and write-up."""
    out: List[str] = [
        "# Round 2 presentation, reproduced",
        "",
        "Round 2 methodology (reservations planned from true finish times, schedulers told "
        "killed jobs' true runtimes, at most one queue-head start per event with no backfilling "
        "at that event, kNN stacked on the TEWMA bias correction) and one seed per experiment (E1 and E2: 42, E3: 99, E4: 7). These are "
        "the numbers in the slides and the write-up, produced by `knn-tewma-ca reproduce`.",
        "",
        "## E1 · Baseline comparison (300 jobs, 8 nodes, p = 0.80)",
        "",
    ]
    e1 = results["E1"]
    by = {name: select(e1, name)[0] for name in ALL_SCHEDULERS}
    rows = []
    for m in SLIDE_METRICS:
        b, t, c = (_rounded(by[n], m) for n in ALL_SCHEDULERS)
        rel = (c - b) / b if b else math.nan
        rows.append([METRIC_LABELS[m], _fixed(b, m), _fixed(t, m), _fixed(c, m), _pct(rel)])
    out += [_table(["Metric", BASELINE, TEWMA, CA, "CA vs Baseline"], rows), ""]
    out += [
        "Reservation violations (kNN-TEWMA-CA): 0. The slides list this as a result; "
        "it is guaranteed by the backfill rule for all three schedulers (see the README).",
        "",
        "## E2 · Confidence sweep (kNN-TEWMA-CA, 300 jobs)",
        "",
    ]
    rows = []
    for p in CONFIDENCE_LEVELS:
        r = select(results["E2"], CA, p)[0]
        rows.append(
            [f"p{int(round(p * 100))}"]
            + [_fixed(_rounded(r, m), m) for m in ("success_rate", "utilization", "avg_slowdown")]
        )
    out += [_table(["Confidence", "Success rate", "Utilization", "Avg slowdown"], rows), ""]

    out += ["## E3 · Load sensitivity (kNN-TEWMA-CA, p = 0.80)", ""]
    rows = []
    for n in xs_in(results["E3"]):
        r = select(results["E3"], CA, n)[0]
        rows.append([str(int(n))] + [_fixed(_rounded(r, m), m) for m in ("success_rate", "avg_wait", "avg_slowdown")])
    out += [_table(["Jobs", "Success rate", "Avg wait", "Avg slowdown"], rows), ""]

    out += [
        "## E4 · High-contention stress (150 jobs, λ = 1.2, bias = −0.25, noise × 0.8)",
        "",
    ]
    t_run, c_run = select(results["E4"], TEWMA)[0], select(results["E4"], CA)[0]
    rows = []
    for m in SLIDE_METRICS:
        tv, cv = _rounded(t_run, m), _rounded(c_run, m)
        rows.append([METRIC_LABELS[m], _fixed(tv, m), _fixed(cv, m), _pct((cv - tv) / tv)])
    out += [_table(["Metric", TEWMA, CA, "Δ"], rows), ""]
    return "\n".join(out)


# -- corrected ----------------------------------------------------------------
def corrected_markdown(results: Results, mode: Mode) -> str:
    n = mode.seed_count
    out: List[str] = [
        f"# Main results: corrected methodology, {n} seeds",
        "",
        f"Each cell is the mean ± sample standard deviation over {n} workloads "
        f"(seeds {_seed_text(mode.e1_seeds)}). Every scheduler runs on the same workloads, so the "
        "comparisons are paired by seed. Brackets are 95% confidence intervals for the mean "
        "difference (Student t). \"Better in\" counts the seeds where kNN-TEWMA-CA is strictly "
        "better. \"Allocated / needed node-time\" is sum(allocated x nodes) / sum(actual x nodes): "
        "how much capacity a scheduler reserves relative to what jobs needed (neither direction is "
        "simply better).",
        "",
        "## E1 · Baseline comparison (300 jobs, 8 nodes, p = 0.80)",
        "",
    ]
    e1 = results["E1"]
    rows = [
        [METRIC_LABELS[m]] + [_pm(values(select(e1, s), m), m) for s in ALL_SCHEDULERS]
        for m in METRIC_NAMES
    ]
    out += [_table(["Metric"] + list(ALL_SCHEDULERS), rows), ""]
    out += ["### kNN-TEWMA-CA minus the other schedulers, same seed", ""]
    out += [_paired_table(e1, CA, (BASELINE, TEWMA)), ""]

    out += ["## E2 · Confidence sweep (kNN-TEWMA-CA, 300 jobs)", ""]
    e2_metrics = ("success_rate", "utilization", "avg_wait", "avg_slowdown", "allocation_ratio")
    rows = []
    for p in xs_in(results["E2"]):
        runs = select(results["E2"], CA, p)
        rows.append([f"p{int(round(p * 100))}"] + [_pm(values(runs, m), m) for m in e2_metrics])
    out += [_table(["Confidence"] + [METRIC_LABELS[m] for m in e2_metrics], rows), ""]

    out += ["## E3 · Load sensitivity (p = 0.80)", ""]
    e3_metrics = ("success_rate", "utilization", "avg_wait", "avg_slowdown")
    rows = []
    for size in xs_in(results["E3"]):
        for s in mode.e3_schedulers:
            runs = select(results["E3"], s, size)
            rows.append([str(int(size)), s] + [_pm(values(runs, m), m) for m in e3_metrics])
    out += [_table(["Jobs", "Scheduler"] + [METRIC_LABELS[m] for m in e3_metrics], rows, "llrrrr"), ""]

    out += [
        "## E4 · High-contention stress (150 jobs, λ = 1.2, bias = −0.25, noise × 0.8)",
        "",
    ]
    e4 = results["E4"]
    rows = [
        [METRIC_LABELS[m]] + [_pm(values(select(e4, s), m), m) for s in mode.e4_schedulers]
        for m in METRIC_NAMES
    ]
    out += [_table(["Metric"] + list(mode.e4_schedulers), rows), ""]
    out += ["### kNN-TEWMA-CA minus the other schedulers, same seed", ""]
    refs = tuple(s for s in (TEWMA, BASELINE) if s in mode.e4_schedulers)
    out += [_paired_table(e4, CA, refs), ""]

    if "E5" in results:
        out += [e5_markdown(results["E5"]), ""]

    total = sum(len(v) for v in results.values())
    out += [
        "## Reservation guarantee",
        "",
        GUARANTEE_NOTE + f" It held in all {total} simulations above, as it must.",
        "",
    ]
    return "\n".join(out)


def e5_markdown(runs: Sequence[Run]) -> str:
    out = [
        "## E5 · What success costs",
        "",
        "Each row is one setting, averaged over seeds. \"Fixed padding\" is the baseline with "
        "every allocation multiplied by a constant factor; TEWMA and kNN-TEWMA-CA are swept over "
        "their base confidence p. Comparing rows with similar allocated node-time shows how much "
        "success each approach gets for the capacity it reserves; comparing rows with similar success "
        "shows what that success costs in queue time and utilization.",
        "",
    ]
    for workload, title in (("default", "E1 workload (unbiased predictions)"),
                            ("stress", "E4 workload (users under-predict by 25%)")):
        rows = []
        for name in (FIXED_PADDING, TEWMA, CA):
            sel = select(runs, name, workload=workload)
            for x in xs_in(sel):
                cell = select(sel, x=x)
                setting = f"× {x:g}" if name == FIXED_PADDING else f"p = {x:.2f}"
                rows.append(
                    [name, setting]
                    + [_mean(values(cell, m), m) for m in ("allocation_ratio", "success_rate", "utilization", "avg_wait")]
                )
        out += [
            f"### {title}",
            "",
            _table(["Scheduler", "Setting", "Allocated / needed node-time", "Success rate", "Utilization",
                    "Avg wait"],
                   rows, "llrrrr"),
            "",
        ]
    return "\n".join(out)


def _paired_table(runs: Sequence[Run], candidate: str, references: Sequence[str]) -> str:
    header = ["Metric"]
    for ref in references:
        header += [f"vs {ref}: mean diff [95% CI]", "relative", "better in"]
    rows = []
    cand = select(runs, candidate)
    for m in METRIC_NAMES:
        row = [METRIC_LABELS[m]]
        for ref in references:
            pc = paired_comparison(values(cand, m), values(select(runs, ref), m), bool(HIGHER_IS_BETTER[m]))
            row += [_ci(pc.difference, m), _pct(pc.relative.mean), _better(pc, m)]
        rows.append(row)
    return _table(header, rows)


# -- comparison ---------------------------------------------------------------
_HEADLINE = (
    ("E1", "success_rate", BASELINE),
    ("E1", "utilization", BASELINE),
    ("E1", "avg_wait", BASELINE),
    ("E1", "avg_slowdown", BASELINE),
    ("E4", "success_rate", TEWMA),
    ("E4", "utilization", TEWMA),
    ("E4", "avg_wait", TEWMA),
)


def comparison_markdown(reproduced: Results, main: Results, ablation: Ablation, mode: Mode) -> str:
    n = mode.seed_count
    out: List[str] = [
        "# Presentation numbers vs corrected methodology",
        "",
        "* **Slides**: the Round 2 numbers (one seed, Round 2 methodology).",
        f"* **Corrected**: every methodology fix applied, mean over {n} seeds "
        f"({_seed_text(mode.e1_seeds)}), with the 95% CI of the paired per-seed change.",
        "",
        "## Headline claims",
        "",
    ]
    rows = []
    for exp, metric, ref in _HEADLINE:
        slide_c = _rounded(select(reproduced[exp], CA)[0], metric)
        slide_r = _rounded(select(reproduced[exp], ref)[0], metric)
        pc = paired_comparison(
            values(select(main[exp], CA), metric),
            values(select(main[exp], ref), metric),
            bool(HIGHER_IS_BETTER[metric]),
        )
        rows.append(
            [
                f"{exp} {METRIC_LABELS[metric].lower()}, CA vs {ref}",
                _pct((slide_c - slide_r) / slide_r),
                _ci_pct(pc.relative),
                f"{pc.better}/{pc.n}",
            ]
        )
    rows.append(["Reservation violations", "0 (reported as a result)",
                 "0 for every scheduler, by construction", "n/a"])
    out += [_table(["Claim", "Slides", f"Corrected, {n} seeds: mean [95% CI]", "CA better in"], rows), ""]

    out += [
        "## What each fix changes",
        "",
        f"The fixes applied one at a time, cumulatively, each over the same {n} seeds. "
        "The first row is the Round 2 code on 30 seeds instead of one, so the gap between it and "
        "the slides is the effect of the seed alone. Cells are means.",
        "",
    ]
    for exp, title in (("E1", "E1 workload"), ("E4", "E4 stress workload")):
        rows = []
        for key, label, _m in ABLATION_STEPS:
            runs = ablation[key][exp]
            row = [label]
            for m in ("success_rate", "utilization", "avg_wait", "allocation_ratio"):
                row += [_mean(values(select(runs, s), m), m) for s in ALL_SCHEDULERS]
            rows.append(row)
        header = ["Step"]
        for m in ("Success", "Utilization", "Avg wait", "Alloc / needed"):
            header += [f"{m}: B", "T", "CA"]
        out += [f"### {title}", "", "B = Baseline, T = TEWMA, CA = kNN-TEWMA-CA.", "",
                _table(header, rows), ""]

    out += ["## Every metric", ""]
    rows = []
    for exp in ("E1", "E4"):
        for m in METRIC_NAMES:
            for s in ALL_SCHEDULERS:
                rep = select(reproduced[exp], s)
                main_runs = select(main[exp], s)
                if not main_runs:
                    continue
                rows.append(
                    [
                        f"{exp} {METRIC_LABELS[m]}",
                        s,
                        _fixed(_rounded(rep[0], m), m) if rep else "—",
                        _pm(values(main_runs, m), m),
                    ]
                )
    out += [_table(["Metric", "Scheduler", "Round 2 (slide seed)", f"Corrected, {n} seeds"], rows, "llrr"), ""]
    return "\n".join(out)


# -- raw data -------------------------------------------------------------------
def write_runs_csv(path: Path, groups: Sequence[Results]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mode", "experiment", "scheduler", "workload", "seed", "x"] + list(METRIC_NAMES))
        for results in groups:
            for runs in results.values():
                for r in runs:
                    x = "" if math.isnan(r.x) else f"{r.x:g}"
                    metrics = r.metrics.as_dict()
                    writer.writerow(
                        [r.mode, r.experiment, r.scheduler, r.workload, r.seed, x]
                        + [f"{metrics[m]:.6f}" for m in METRIC_NAMES]
                    )
