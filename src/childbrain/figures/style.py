"""Plotting style of the paper figures: palettes, age-group bars and curves, brackets."""

from __future__ import annotations

import string
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sstats

from . import data
from .stats import mannwhitney, p_to_stars, permute_cluster, sig_to_plot, wilcoxon_greater

# Editable text in vector outputs.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

RLABEL = r"$\mathcal{R}$"
#: Resolution of the rasterized parts (cortex) of vector figures.
DPI = 400
#: Window of the mean decoding score (s), and of peak metrics.
AVG_TMIN, AVG_TMAX = -1.0, 2.0
METRIC_TMIN, METRIC_TMAX = -2.0, 3.0
#: Two-sided p below which a statistic is written in red.
P_RED_TEXT = 0.06


def get_colors(n: int, cmap_name: str = "Reds", skip: int = 5) -> list:
    """``n`` colours of a sequential colormap, skipping its ``skip`` lightest."""
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i) for i in np.linspace(0, 1, n + skip)][skip:]


#: Age-group colours (young -> old = light -> dark) of each ROI.
ROI_CMAPS = dict(zip(data.ROIS, ["Greys", "Blues", "Reds", "Greens", "Purples", "Oranges", "PuRd"]))
ROI_COLORS = {roi: get_colors(3, cmap, skip=1) for roi, cmap in ROI_CMAPS.items()}
#: Age-group colormap of each linguistic level and decoded feature group.
LEVEL_CMAPS = {"acoustic": "Oranges", "phonetic": "Blues", "lexical": "Reds",
               "supralexical": "Purples"}


def palette(panel: str = "whole_brain", n: int = 3, skip: int = 1) -> list:
    """Age-group colours of a level (``LEVEL_CMAPS``), an ROI or a colormap name."""
    cmap = LEVEL_CMAPS.get(panel) or ROI_CMAPS.get(panel) or panel
    return get_colors(n, cmap, skip=skip)


def format_tick(value: float, precision: int = 3) -> str:
    """Tick label without the leading 0 (``.05``)."""
    if np.isclose(value, 0.0, atol=10 ** -(precision + 1)):
        return "0"
    text = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
    if text.startswith("-0."):
        return "-" + text[2:]
    return text[1:] if text.startswith("0.") else text


def minimal_yaxis(ax, ymax: float | None = None, ylabel: str | None = RLABEL, precision: int = 3,
                  show_zero: bool = True, pad_frac: float = 0.01) -> None:
    """Ticks at 0 and ``ymax`` only, dashed zero line, despined."""
    ymin, top = ax.get_ylim()
    ymax = top if ymax is None else ymax
    if show_zero and ymax > 0:
        ax.set_ylim(-ymax * pad_frac, ymax)
        ticks = [0.0, float(ymax)]
        ax.axhline(0, color="lightgrey", linestyle="--", zorder=-1)
    else:
        ticks = list(ax.get_ylim())
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: format_tick(v, precision)))
    ax.set_ylabel(ylabel, rotation=0)
    sns.despine(ax=ax)


def age_xticks(ax, colors: list, labels: list[str] = data.GROUPS) -> None:
    ax.set_xlabel(None)
    ax.set_xticks(range(len(labels)), labels)
    for tick, c in zip(ax.get_xticklabels(), colors):
        tick.set_color(c)


def draw_bracket(ax, x1: float, x2: float, y: float, stars: str, color: str = "0.15",
                 linewidth: float = 1.8, fontsize: float = 10) -> None:
    ax.plot([x1, x2], [y, y], lw=linewidth, c=color, clip_on=False, zorder=6, solid_capstyle="round")
    ax.text((x1 + x2) / 2, y, stars, ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
            color=color, clip_on=False, zorder=7)


def _values(df: pd.DataFrame, group: str, col: str, group_col: str = "group") -> np.ndarray:
    return df.loc[df[group_col].astype(str) == group, col].dropna().to_numpy()


def wilcoxon_stars(ax, df: pd.DataFrame, col: str, groups: list[str] = data.GROUPS,
                   group_col: str = "group", fontsize: float = 10, x: list[float] | None = None,
                   y: float = 0, color: str = "white",
                   to_stars: Callable[[float], str] = p_to_stars) -> dict[str, float]:
    """One-sided Wilcoxon against 0 of each group: stars on the bars (at ``y``)."""
    out = {}
    for i, g in enumerate(groups):
        p = wilcoxon_greater(_values(df, g, col, group_col))
        out[g] = p
        if stars := to_stars(p):
            ax.text(i if x is None else x[i], y, stars, ha="center", va="center", fontweight="bold",
                    color=color, fontsize=fontsize, clip_on=False, zorder=8)
    return out


def age_brackets(ax, df: pd.DataFrame, col: str, groups: list[str] = data.GROUPS,
                 group_col: str = "group", step_frac: float = 0.08, fontsize: float = 10,
                 x: list[float] | None = None, alternative: str = "less") -> dict[str, float]:
    """Mann-Whitney U between age groups (younger < older): brackets above the axes top."""
    ymin, ymax = ax.get_ylim()
    step = ymax * step_frac
    n, out = 0, {}
    for i, g1 in enumerate(groups):
        for j in range(i + 1, len(groups)):
            g2 = groups[j]
            p = mannwhitney(_values(df, g1, col, group_col), _values(df, g2, col, group_col),
                            alternative=alternative)
            out[f"{g1}<{g2}"] = p
            if stars := p_to_stars(p):
                n += 1
                draw_bracket(ax, i if x is None else x[i], j if x is None else x[j],
                             ymax + step * n, stars, fontsize=fontsize)
    if n:
        ax.set_ylim(ymin, ymax + step * (n + 0.5))
    return out


def age_bars(ax, df: pd.DataFrame, panel: str, col: str = "score", title: str | None = None,
             ylabel: str | None = RLABEL, stats: bool = True, ylim: tuple[float, float] | None = None,
             fontsize: float = 10) -> dict:
    """Mean and SEM across participants of each age group; Wilcoxon stars on the bars,
    Mann-Whitney brackets between groups. Returns the p-values."""
    colors = palette(panel)
    sns.barplot(data=df.assign(group=df.group.astype(str)), x="group", y=col, order=data.GROUPS,
                hue="group", hue_order=data.GROUPS, palette=colors, dodge=False, errorbar="se",
                ax=ax, legend=False)
    out: dict = {}
    if stats:
        out["wilcoxon"] = wilcoxon_stars(ax, df, col, fontsize=fontsize)
        out["mannwhitney"] = age_brackets(ax, df, col, fontsize=fontsize)
    age_xticks(ax, colors)
    ymax = ylim[1] if ylim is not None else ax.get_ylim()[1]
    minimal_yaxis(ax, ymax=ymax, ylabel=ylabel)
    if ylim is not None:
        ax.set_ylim(-ylim[1] * 0.01, ax.get_ylim()[1])
    if title:
        ax.set_title(title, fontsize=8, color=colors[1])
    return out


def age_curves(ax, df: pd.DataFrame, panel: str, col: str = "score", time_col: str = "time",
               title: str | None = None, ylabel: str | None = RLABEL, sig_dots: bool = True,
               ymax: float | None = None) -> list:
    """Mean and SEM time course of each age group, with cluster-permutation dots
    (times significantly above 0) below the curves."""
    colors = palette(panel)
    sns.lineplot(data=df.assign(group=df.group.astype(str)), x=time_col, y=col, hue="group",
                 hue_order=data.GROUPS, palette=colors, errorbar="se", ax=ax, legend=False)
    minimal_yaxis(ax, ymax=ymax, ylabel=ylabel)
    if sig_dots:
        top = ax.get_ylim()[1]
        cluster_dots(ax, df, colors, a=top * 0.05, c=top * 0.015, col=col, time_col=time_col)
    ax.set_xlabel("Time (s)")
    if title:
        ax.set_title(title, fontsize=8, color=colors[1])
    return colors


def cluster_dots(ax, df: pd.DataFrame, colors: list, a: float, c: float, col: str = "score",
                 time_col: str = "time", groups: list[str] = data.GROUPS, s: float = 4) -> dict:
    """Dots at the times where each group's scores are above 0 (cluster permutation),
    oldest group lowest. Returns the significance masks."""
    times = np.sort(df[time_col].unique())
    out = {}
    for j, (g, color) in enumerate(zip(groups[::-1], colors[::-1])):
        sub = df[df.group.astype(str) == g]
        sig = permute_cluster(sub, time_col=time_col, value_col=col)
        out[g] = sig
        if (dots := sig_to_plot(sig)) is not None:
            ax.scatter(times, -(dots * a) + (j - 1) * c, color=color, s=s, zorder=5)
    return out


def age_legend(ax, colors: list, labels: list[str] = data.GROUPS, fontsize: float = 9,
               loc: str = "upper left") -> None:
    """Legend in reversed order (oldest first), label text in the group colour."""
    handles = [plt.Line2D([0], [0], color=c, lw=2, label=lab) for c, lab in zip(colors, labels)]
    ax.legend(handles[::-1], labels[::-1], frameon=False, loc=loc, labelcolor=colors[::-1],
              handlelength=1.2, labelspacing=0.1, borderpad=0, borderaxespad=0, fontsize=fontsize)


def spearman_text(ax, x, y, alternative: str = "greater", show_ns: bool = False) -> tuple[float, float]:
    """One-sided Spearman r and p written on the axes (red when p <= P_RED_TEXT)."""
    res = sstats.spearmanr(x, y, alternative=alternative)
    r, p = float(res.statistic), float(res.pvalue)
    if not np.isfinite(p):
        return r, p
    sig = p <= P_RED_TEXT
    text = f"r = {r:.2f}, p = {'<0.001' if p < 0.001 else f'{p:.3f}'}"
    if show_ns:
        ax.text(1.0, 1.01, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color="red" if sig else "black")
    elif sig:
        ax.text(0.97, 0.97, text, transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color="red", bbox=dict(boxstyle="round,pad=0.35", facecolor="none", edgecolor="red",
                                       linewidth=2.5), zorder=2)
    return r, p


#: Bounds (years) of the released age ranges on the age axes; 40: display end of 18+.
AGE_RANGE_BOUNDS = [1, 3, 6, 9, 12, 18, 40]


def age_axis_positions(seed: int = 0) -> pd.Series:
    """Position of each participant on the age axes (0-1): a random point within its age
    range, each range having the same width. Uses the released age ranges only, so the
    plots do not reveal exact ages."""
    part = data.participants().sort_index()
    edges = np.linspace(0, 1, len(data.AGE_RANGES) + 1)
    i = part.age_range.map({r: k for k, r in enumerate(data.AGE_RANGES)}).to_numpy()
    u = np.random.default_rng(seed).uniform(0.08, 0.92, len(part))
    return pd.Series(edges[i] + u * (edges[i + 1] - edges[i]), index=part.index)


def age_to_axis(age: np.ndarray) -> np.ndarray:
    """Exact ages (years) on the age axes: piecewise-linear in log(age) within each range."""
    edges = np.linspace(0, 1, len(data.AGE_RANGES) + 1)
    return np.interp(np.log(age), np.log(AGE_RANGE_BOUNDS), edges)


def age_axis(ax) -> None:
    """Limits of the age axes, ticks at the bounds of the age groups (6 and 12 years)."""
    edges = np.linspace(0, 1, len(data.AGE_RANGES) + 1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([edges[data.AGE_RANGES.index("6-8")], edges[data.AGE_RANGES.index("12-17")]],
                  ["6", "12"])
    ax.minorticks_off()


def panel_letters(axes: list, fontsize: float = 18, x: float = -0.12, y: float = 1.08,
                  letters: str = string.ascii_uppercase) -> None:
    for letter, ax in zip(letters, axes):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize, fontweight="bold",
                va="bottom", ha="right")


def save(fig, path: str | Path, png: bool = False, dpi: int = DPI) -> list[Path]:
    """Vector PDF (rasterized artists at ``dpi``), and a PNG if ``png``."""
    path = Path(path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    out = [path]
    if png:
        fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        out.append(path.with_suffix(".png"))
    plt.close(fig)
    return out
