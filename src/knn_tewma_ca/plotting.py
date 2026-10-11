"""Charts for the experiment results.

Uses matplotlib's object API (``Figure``) rather than pyplot, so importing
this module never changes the global backend or opens windows. PNGs are
written without the default ``Software`` metadata tag.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import ScaledTranslation

from .experiments import (
    ABLATION_STEPS,
    ALL_SCHEDULERS,
    FIXED_PADDING,
    Mode,
    Run,
    select,
    values,
    xs_in,
)
from .metrics import METRIC_LABELS
from .schedulers import KnnTewmaCA, TewmaScheduler
from .stats import summarize

Results = Dict[str, List[Run]]

# Palette ------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_COLORS = {
    "Baseline": "#2a78d6",
    "TEWMA": "#eb6834",
    "kNN-TEWMA-CA": "#1baf7a",
    FIXED_PADDING: INK_2,
}

_VALUE_FORMAT = {
    "makespan": "{:.0f}",
    "avg_wait": "{:.1f}",
    "avg_turnaround": "{:.1f}",
    "success_rate": "{:.3f}",
    "utilization": "{:.3f}",
    "avg_slowdown": "{:.2f}",
    "allocation_ratio": "{:.2f}",
}

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
}

SLIDES_LABEL = "Round 2 code, slide seed"


def _figure(ncols: int, title: str, subtitle: str = "", width: float = 3.6, height: float = 3.4,
            nrows: int = 1) -> tuple:
    fig = Figure(figsize=(width * ncols, height * nrows), facecolor=SURFACE, layout="constrained")
    grid = fig.subplots(nrows, ncols, squeeze=False)
    axes = grid[0] if nrows == 1 else grid
    for ax in grid.flat:
        _style(ax)
    # The suptitle gets a blank second line so constrained layout reserves room
    # for the subtitle, which is then drawn one title-line below the top.
    heading = title if not subtitle else f"{title}\n"
    fig.suptitle(heading, x=0.01, ha="left", color=INK, fontsize=12,
                 fontweight="bold", linespacing=1.5)
    if subtitle:
        drop = ScaledTranslation(0, -12 * 1.5 / 72, fig.dpi_scale_trans)
        fig.text(0.01, 0.98, subtitle, ha="left", va="top", color=INK_2, fontsize=9,
                 transform=fig.transFigure + drop)
    return fig, axes


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, colors=MUTED, labelcolor=INK_2, labelsize=8.5, pad=4)


def _panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", color=INK, fontsize=10, pad=8)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE, metadata={"Software": None})
    return path


def _legend(fig: Figure, handles: Sequence[Line2D]) -> None:
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False,
               fontsize=8.5, labelcolor=INK_2, handlelength=1.6, columnspacing=1.6)


def _jitter(n: int, seed: int, width: float = 0.17) -> List[float]:
    rng = random.Random(seed)
    return [rng.uniform(-width, width) for _ in range(n)]


# Generic panels --------------------------------------------------------------
def _strip_panel(
    ax,
    metric: str,
    groups: Sequence[str],
    samples: Dict[str, List[float]],
    reference: Optional[Dict[str, float]] = None,
) -> None:
    """Every seed as a faint dot, the mean as a bar with its 95% CI, the slides value as a diamond."""
    fmt = _VALUE_FORMAT[metric]
    for i, name in enumerate(groups):
        vals = samples[name]
        color = SERIES_COLORS[name]
        offsets = _jitter(len(vals), seed=i)
        ax.scatter([i + o for o in offsets], vals, s=13, color=color, alpha=0.35, linewidths=0, zorder=2)
        s = summarize(vals)
        ax.plot([i, i], [s.ci_low, s.ci_high], color=color, linewidth=2, solid_capstyle="round", zorder=3)
        ax.plot([i - 0.24, i + 0.24], [s.mean, s.mean], color=color, linewidth=2.6,
                solid_capstyle="round", zorder=3)
        ax.annotate(fmt.format(s.mean), (i + 0.27, s.mean), va="center", ha="left",
                    fontsize=8, color=INK, zorder=5)
        if reference and name in reference:
            ax.scatter([i], [reference[name]], marker="D", s=30, facecolors=SURFACE,
                       edgecolors=INK, linewidths=1.1, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_xlim(-0.55, len(groups) - 0.25)
    _panel_title(ax, METRIC_LABELS[metric])


def _line_panel(
    ax,
    metric: str,
    xs: Sequence[float],
    samples: Dict[str, List[List[float]]],
    reference: Optional[Dict[str, List[float]]] = None,
    log2_x: bool = False,
) -> None:
    """Mean across seeds per x with a shaded 95% CI band; slides values as diamonds."""
    for name, per_x in samples.items():
        color = SERIES_COLORS[name]
        sums = [summarize(v) for v in per_x]
        if any(s.n > 1 for s in sums):
            ax.fill_between(xs, [s.ci_low for s in sums], [s.ci_high for s in sums],
                            color=color, alpha=0.14, linewidth=0, zorder=1)
        ax.plot(xs, [s.mean for s in sums], color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.4, solid_capstyle="round", zorder=3)
    if reference:
        for name, ref in reference.items():
            ax.plot(xs, ref, color=INK_2, linewidth=1, marker="D", markersize=4.5,
                    markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=1.0, zorder=4)
    if log2_x:
        ax.set_xscale("log", base=2)
    ax.set_xticks(list(xs))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.tick_params(axis="x", which="minor", length=0)
    _panel_title(ax, METRIC_LABELS[metric])


def _bar_panel(ax, metric: str, groups: Sequence[str], vals: Dict[str, float]) -> None:
    fmt = _VALUE_FORMAT[metric]
    heights = [vals[g] for g in groups]
    ax.bar(range(len(groups)), heights, width=0.55, color=[SERIES_COLORS[g] for g in groups], zorder=2)
    top = max(heights)
    for i, h in enumerate(heights):
        ax.annotate(fmt.format(h), (i, h), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_ylim(0, top * 1.15)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    _panel_title(ax, METRIC_LABELS[metric])


def _series_handles(names: Sequence[str]) -> List[Line2D]:
    return [Line2D([], [], color=SERIES_COLORS[n], linewidth=2, marker="o", markersize=5,
                   markeredgecolor=SURFACE, label=n) for n in names]


def _strip_handles(n_seeds: int, with_reference: bool) -> List[Line2D]:
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=5, color=MUTED, alpha=0.5,
               label=f"One seed (n = {n_seeds})"),
        Line2D([], [], color=INK_2, linewidth=2.4, label="Mean and 95% CI"),
    ]
    if with_reference:
        handles.append(Line2D([], [], linestyle="none", marker="D", markersize=5,
                              markerfacecolor=SURFACE, markeredgecolor=INK, label=SLIDES_LABEL))
    return handles


def _reference_handle() -> Line2D:
    return Line2D([], [], color=INK_2, linewidth=1, marker="D", markersize=4.5,
                  markerfacecolor=SURFACE, markeredgecolor=INK, label=SLIDES_LABEL)


# Multi-seed figures ------------------------------------------------------------
def _single(runs: Sequence[Run], scheduler: str, metric: str, x: Optional[float] = None) -> Optional[float]:
    sel = select(runs, scheduler, x)
    return sel[0].metrics.as_dict()[metric] if sel else None


def corrected_figures(main: Results, mode: Mode, reproduced: Optional[Results], out_dir: Path) -> List[Path]:
    """Charts for the multi-seed results, with the slides values overlaid when given."""
    import matplotlib

    paths: List[Path] = []
    n = mode.seed_count
    ca = KnnTewmaCA.name
    with matplotlib.rc_context(_RC):
        # E1 ------------------------------------------------------------------
        for name, metrics in (
            ("e1_efficiency.png", ("success_rate", "utilization", "allocation_ratio")),
            ("e1_time.png", ("avg_wait", "avg_turnaround", "avg_slowdown")),
        ):
            fig, axes = _figure(len(metrics), "E1 · Baseline comparison",
                                f"300 jobs on 8 nodes, p = 0.80, {n} seeds")
            for ax, m in zip(axes, metrics):
                samples = {s: values(select(main["E1"], s), m) for s in ALL_SCHEDULERS}
                ref = ({s: _single(reproduced["E1"], s, m) for s in ALL_SCHEDULERS} if reproduced else None)
                _strip_panel(ax, m, ALL_SCHEDULERS, samples, ref)
            _legend(fig, _strip_handles(n, reproduced is not None))
            paths.append(_save(fig, out_dir / name))

        # E2 ------------------------------------------------------------------
        metrics = ("success_rate", "utilization", "avg_slowdown")
        xs = xs_in(main["E2"])
        fig, axes = _figure(len(metrics), "E2 · Confidence sweep",
                            f"kNN-TEWMA-CA, 300 jobs, {n} seeds (band: 95% CI of the mean); x-axis is the base confidence p")
        for ax, m in zip(axes, metrics):
            samples = {ca: [values(select(main["E2"], ca, x), m) for x in xs]}
            ref = ({ca: [_single(reproduced["E2"], ca, m, x) for x in xs]} if reproduced else None)
            _line_panel(ax, m, xs, samples, ref)
        handles = _series_handles([ca]) + ([_reference_handle()] if reproduced else [])
        _legend(fig, handles)
        paths.append(_save(fig, out_dir / "e2_confidence.png"))

        # E3 ------------------------------------------------------------------
        metrics = ("success_rate", "utilization", "avg_wait")
        xs = xs_in(main["E3"])
        scheds = [s for s in ALL_SCHEDULERS if s in mode.e3_schedulers]
        fig, axes = _figure(len(metrics), "E3 · Load sensitivity",
                            f"p = 0.80, {n} seeds (bands: 95% CI of the mean); x-axis is the number of jobs, log scale")
        for ax, m in zip(axes, metrics):
            samples = {s: [values(select(main["E3"], s, x), m) for x in xs] for s in scheds}
            ref = ({ca: [_single(reproduced["E3"], ca, m, x) for x in xs]} if reproduced else None)
            _line_panel(ax, m, xs, samples, ref, log2_x=True)
        handles = _series_handles(scheds) + ([_reference_handle()] if reproduced else [])
        _legend(fig, handles)
        paths.append(_save(fig, out_dir / "e3_load.png"))

        # E4 ------------------------------------------------------------------
        metrics = ("success_rate", "utilization", "avg_wait", "allocation_ratio")
        scheds = [s for s in ALL_SCHEDULERS if s in mode.e4_schedulers]
        fig, axes = _figure(len(metrics), "E4 · High-contention stress",
                            f"150 jobs, 8x arrival rate, users under-predict by 25%, {n} seeds")
        for ax, m in zip(axes, metrics):
            samples = {s: values(select(main["E4"], s), m) for s in scheds}
            ref = None
            if reproduced:
                ref = {s: v for s in scheds if (v := _single(reproduced["E4"], s, m)) is not None}
            _strip_panel(ax, m, scheds, samples, ref)
        _legend(fig, _strip_handles(n, reproduced is not None))
        paths.append(_save(fig, out_dir / "e4_stress.png"))

        if "E5" in main:
            paths.append(_e5_figure(main, n, out_dir / "e5_allocation.png"))
    return paths


def _e5_figure(main: Results, n: int, path: Path) -> Path:
    """Success against reserved capacity (top) and against queue time (bottom), both workloads."""
    fig, grid = _figure(2, "E5 · What success costs",
                        f"Mean over {n} seeds. Each line sweeps one knob: the padding factor, or the base "
                        "confidence p. Large dots: the settings used in E1 and E4.",
                        width=4.6, height=3.5, nrows=2)
    workloads = (("default", "E1 workload (unbiased predictions)", "E1"),
                 ("stress", "E4 workload (users under-predict by 25%)", "E4"))
    for row, (x_metric, x_label) in zip(
        grid, (("allocation_ratio", "Allocated / needed node-time"), ("avg_wait", "Average wait"))
    ):
        for ax, (workload, title, experiment) in zip(row, workloads):
            for name in (FIXED_PADDING, TewmaScheduler.name, KnnTewmaCA.name):
                sel = select(main["E5"], name, workload=workload)
                xs = xs_in(sel)
                xv = [summarize(values(select(sel, x=x), x_metric)).mean for x in xs]
                succ = [summarize(values(select(sel, x=x), "success_rate")).mean for x in xs]
                color = SERIES_COLORS[name]
                hollow = name == FIXED_PADDING
                ax.plot(xv, succ, color=color, linewidth=1.4 if hollow else 2, marker="o", markersize=4.5,
                        markerfacecolor=SURFACE if hollow else color,
                        markeredgecolor=color if hollow else SURFACE, markeredgewidth=1.2, zorder=3)
                if hollow:
                    for x, a, sv in zip(xs, xv, succ):
                        if x in (1.0, 1.5, 2.0):
                            ax.annotate(f"×{x:g}", (a, sv), xytext=(5, -10), textcoords="offset points",
                                        fontsize=7.5, color=INK_2)
            # Operating points from the main experiment.
            for name in ALL_SCHEDULERS:
                runs = select(main[experiment], name)
                if not runs:
                    continue
                xv = summarize(values(runs, x_metric)).mean
                sv = summarize(values(runs, "success_rate")).mean
                ax.scatter([xv], [sv], s=70, color=SERIES_COLORS[name], edgecolors=SURFACE, linewidths=2, zorder=5)
            ax.set_xlabel(x_label, color=INK_2, fontsize=8.5)
            if x_metric == "allocation_ratio":
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}×"))
            ax.set_ylabel("Success rate", color=INK_2, fontsize=8.5)
            _panel_title(ax, title)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
    handles = [
        Line2D([], [], color=INK_2, linewidth=1.4, marker="o", markersize=4.5, markerfacecolor=SURFACE,
               markeredgecolor=INK_2, label="Fixed padding (Baseline × 1.0–2.0)"),
        Line2D([], [], color=SERIES_COLORS[TewmaScheduler.name], linewidth=2, marker="o", markersize=4.5,
               markeredgecolor=SURFACE, label="TEWMA (p 0.60–0.99)"),
        Line2D([], [], color=SERIES_COLORS[KnnTewmaCA.name], linewidth=2, marker="o", markersize=4.5,
               markeredgecolor=SURFACE, label="kNN-TEWMA-CA (p 0.55–0.90)"),
        Line2D([], [], linestyle="none", marker="o", markersize=8, color=SERIES_COLORS["Baseline"],
               markeredgecolor=SURFACE, label="Baseline (adaptive padding)"),
    ]
    _legend(fig, handles)
    return _save(fig, path)


ABLATION_SHORT = ("Round 2\ncode", "+ allocated\nend", "+ all heads\nstart", "+ censored\nlearning", "+ kNN\nfix")


def ablation_figure(ablation: Dict[str, Results], n: int, out_dir: Path) -> Path:
    """How each methodology fix moves the main metrics, E1 and E4 workloads."""
    import matplotlib

    metrics = ("success_rate", "utilization", "avg_wait")
    with matplotlib.rc_context(_RC):
        fig, grid = _figure(len(metrics), "What each fix changes",
                            f"Fixes applied cumulatively, left to right; mean over {n} seeds with 95% CI. "
                            "Top: E1 workload. Bottom: E4 stress workload.", width=3.9, height=3.0, nrows=2)
        xs = list(range(len(ABLATION_STEPS)))
        for row, exp in zip(grid, ("E1", "E4")):
            for ax, m in zip(row, metrics):
                samples = {
                    s: [values(select(ablation[key][exp], s), m) for key, _l, _m in ABLATION_STEPS]
                    for s in ALL_SCHEDULERS
                }
                _line_panel(ax, m, xs, samples)
                ax.set_xticks(xs)
                ax.set_xticklabels(ABLATION_SHORT, fontsize=7.5)
                ax.set_title(f"{METRIC_LABELS[m]} · {exp}", loc="left", color=INK, fontsize=10, pad=8)
        _legend(fig, _series_handles(ALL_SCHEDULERS))
        return _save(fig, out_dir / "ablation.png")


# Single-seed (reproduce) figures --------------------------------------------------
def reproduce_figures(reproduced: Results, out_dir: Path) -> List[Path]:
    """The presentation charts, redrawn from the reproduced single-seed runs."""
    import matplotlib

    paths: List[Path] = []
    ca, tewma = KnnTewmaCA.name, TewmaScheduler.name
    with matplotlib.rc_context(_RC):
        for name, metrics in (
            ("reproduce_e1_efficiency.png", ("success_rate", "utilization")),
            ("reproduce_e1_time.png", ("makespan", "avg_wait", "avg_turnaround", "avg_slowdown")),
        ):
            fig, axes = _figure(len(metrics), "E1 · Baseline comparison (Round 2, reproduced)",
                                "300 jobs on 8 nodes, p = 0.80, seed 42")
            for ax, m in zip(axes, metrics):
                _bar_panel(ax, m, ALL_SCHEDULERS, {s: _single(reproduced["E1"], s, m) for s in ALL_SCHEDULERS})
            paths.append(_save(fig, out_dir / name))

        metrics = ("success_rate", "avg_slowdown")
        xs = xs_in(reproduced["E2"])
        fig, axes = _figure(len(metrics), "E2 · Confidence sweep (Round 2, reproduced)",
                            "kNN-TEWMA-CA, 300 jobs, seed 42")
        for ax, m in zip(axes, metrics):
            _line_panel(ax, m, xs, {ca: [[_single(reproduced["E2"], ca, m, x)] for x in xs]})
        paths.append(_save(fig, out_dir / "reproduce_e2_confidence.png"))

        metrics = ("success_rate", "avg_wait", "avg_slowdown")
        xs = xs_in(reproduced["E3"])
        fig, axes = _figure(len(metrics), "E3 · Load sensitivity (Round 2, reproduced)",
                            "kNN-TEWMA-CA, p = 0.80, seed 99")
        for ax, m in zip(axes, metrics):
            _line_panel(ax, m, xs, {ca: [[_single(reproduced["E3"], ca, m, x)] for x in xs]}, log2_x=True)
        paths.append(_save(fig, out_dir / "reproduce_e3_load.png"))

        metrics = ("success_rate", "utilization", "avg_wait", "avg_slowdown")
        fig, axes = _figure(len(metrics), "E4 · High-contention stress (Round 2, reproduced)",
                            "150 jobs, 8x arrival rate, users under-predict by 25%, seed 7")
        for ax, m in zip(axes, metrics):
            _bar_panel(ax, m, (tewma, ca), {s: _single(reproduced["E4"], s, m) for s in (tewma, ca)})
        paths.append(_save(fig, out_dir / "reproduce_e4_stress.png"))
    return paths
