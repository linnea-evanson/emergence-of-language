"""Fig. 4: decoding of each linguistic level by age group, whole brain and ROIs.

Rows: phonetic, lexical, supralexical decoding. A, D, G: mean score between -1 and 2 s
per age group (white stars: Wilcoxon against 0; brackets: Mann-Whitney U, younger <
older). B, E, H: whole-brain time courses (dots: cluster permutation against 0; inset:
the ROI electrodes). C, F, I: time courses in each ROI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import brain, data, style
from .output import Output
from .stats import mannwhitney, permute_cluster, sig_to_plot, window_mean

NAME = "fig4"
#: ROI grid of C, F, I (2 x 3).
ROI_GRID = [data.ROIS[1:4], data.ROIS[4:]]
PAIRS = [(0, 1), (0, 2), (1, 2)]


def level_means(scores: pd.DataFrame) -> pd.DataFrame:
    """Per participant: mean score between ``AVG_TMIN`` and ``AVG_TMAX``."""
    return window_mean(scores, style.AVG_TMIN, style.AVG_TMAX, ["participant", "age", "group"])


def group_mwu(avg: pd.DataFrame, col: str = "score") -> dict[str, float]:
    """One-sided Mann-Whitney U p (younger < older) of each pair of age groups."""
    out = {}
    for i, j in PAIRS:
        g1, g2 = data.GROUPS[i], data.GROUPS[j]
        out[f"{g1}<{g2}"] = mannwhitney(avg.loc[avg.group == g1, col].to_numpy(),
                                        avg.loc[avg.group == g2, col].to_numpy())
    return out


def group_counts(scores: pd.DataFrame) -> dict[str, int]:
    n = scores.groupby("group", observed=False).participant.nunique()
    return {g: int(n[g]) for g in data.GROUPS}


def sig_spans(times: np.ndarray, mask: np.ndarray) -> list[list[float]]:
    """[first, last] time of each run of significant times."""
    spans: list[list[float]] = []
    for t, prev, m in zip(times, np.r_[False, mask[:-1]], mask):
        if m and not prev:
            spans.append([float(t), float(t)])
        elif m:
            spans[-1][1] = float(t)
    return spans


def group_curves(scores: pd.DataFrame, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    """Mean, SEM and n of each age group per time, with the significance mask."""
    out = scores.groupby(["group", "time"], observed=True).score.agg(["mean", "sem", "count"])
    out = out.rename(columns={"count": "n"}).reset_index().astype({"group": str})
    times = np.sort(scores.time.unique())
    sig = pd.concat([pd.DataFrame({"group": g, "time": times, "significant": m}) for g, m in masks.items()])
    return out.merge(sig, on=["group", "time"], how="left")


def group_masks(scores: pd.DataFrame) -> dict[str, np.ndarray]:
    """Cluster-permutation mask (above 0) of each age group."""
    return {g: permute_cluster(scores[scores.group == g]) for g in data.GROUPS}


def whole_brain_curves(ax, scores: pd.DataFrame, level: str) -> tuple[list, dict[str, np.ndarray]]:
    """Age-group time courses, cluster dots below the curves."""
    colors = style.age_curves(ax, scores, level, title="Whole brain", sig_dots=False)
    ymax = ax.get_ylim()[1]
    ymin = -0.16 * ymax
    c = 0.05 * ymax
    a = -ymin - 2 * c
    times = np.sort(scores.time.unique())
    masks = group_masks(scores)
    for j, (g, color) in enumerate(zip(data.GROUPS[::-1], colors[::-1])):
        if (dots := sig_to_plot(masks[g])) is not None:
            ax.scatter(times, -(dots * a) + (j - 1) * c, color=color, s=4, zorder=5)
    ax.set_ylim(ymin, ymax)
    return colors, masks


def roi_ymax(scores: pd.DataFrame) -> float:
    """1.1 x the highest age-group mean curve."""
    top = float(scores.groupby(["group", "time"], observed=True).score.mean().max())
    return 1.1 * top if top > 0 else 0.01


def roi_panel(ax, scores: pd.DataFrame, roi: str) -> dict[str, np.ndarray]:
    """Age-group mean curves in the ROI colours, cluster dots below."""
    colors = style.ROI_COLORS[roi]
    ymax = roi_ymax(scores)
    times = np.sort(scores.time.unique())
    for g, color in zip(data.GROUPS, colors):
        mean = scores[scores.group == g].groupby("time").score.mean()
        ax.plot(mean.index, mean.to_numpy(), color=color, lw=1.2)
    c = 0.06 * ymax
    ymin = -4 * c
    masks = group_masks(scores)
    for j, (g, color) in enumerate(zip(data.GROUPS[::-1], colors[::-1])):
        if (dots := sig_to_plot(masks[g])) is not None:
            ax.scatter(times, dots * (ymin + (j + 0.5) * c), color=color, s=2, zorder=5)
    style.minimal_yaxis(ax, ymax=ymax, ylabel=None)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(None)
    ax.set_title(data.ROI_TITLES[roi], color=colors[1], fontsize=11)
    return masks


def roi_grid(fig, spec, level: str, bottom: bool) -> tuple:
    """2 x 3 ROI panels of one level; returns the first axes, source data and stats."""
    sub_gs = spec.subgridspec(2, 3, hspace=0.55, wspace=0.35)
    first, rows, stats = None, [], {}
    for i, row in enumerate(ROI_GRID):
        for j, roi in enumerate(row):
            ax = fig.add_subplot(sub_gs[i, j])
            first = first or ax
            scores = data.decoding_levels(roi, [level])
            masks = roi_panel(ax, scores, roi)
            if bottom and i == 1:
                ax.set_xlabel("Time (s)")
            times = np.sort(scores.time.unique())
            rows.append(group_curves(scores, masks).assign(roi=roi))
            stats[roi] = {"n": group_counts(scores), "mannwhitney": group_mwu(level_means(scores)),
                          "significant_s": {g: sig_spans(times, m) for g, m in masks.items()}}
    return first, pd.concat(rows, ignore_index=True), stats


def make(out: Path) -> None:
    o = Output(out, NAME)
    fig = plt.figure(figsize=(17, 4.6 * len(data.LEVELS)), layout="constrained")
    gs = fig.add_gridspec(len(data.LEVELS), 3, width_ratios=[0.55, 1.1, 1.9])
    letters = iter("ABCDEFGHI")
    letter_axes = []
    for r, level in enumerate(data.LEVELS):
        scores = data.decoding_levels("whole_brain", [level])
        ax_bar = fig.add_subplot(gs[r, 0])
        ax_curve = fig.add_subplot(gs[r, 1])
        avg = level_means(scores)
        bar_stats = style.age_bars(ax_bar, avg, level, title="Whole brain")
        ax_bar.annotate(data.LEVEL_TITLES[level], xy=(0, 0.5), xycoords=ax_bar.yaxis.label,
                        xytext=(-10, 0), textcoords="offset points", ha="right", va="center",
                        rotation=90, fontsize=14)
        colors, masks = whole_brain_curves(ax_curve, scores, level)
        for ax in (ax_bar, ax_curve):
            ax.title.set_fontsize(12)
        if r == 0:
            style.age_legend(ax_curve, colors, fontsize=11)
            brain.roi_inset(ax_curve.inset_axes((0.55, 0.5, 0.47, 0.5)))
        ax_roi, roi_source, roi_stats = roi_grid(fig, gs[r, 2], level, bottom=r == len(data.LEVELS) - 1)
        letter_axes += [ax_bar, ax_curve, ax_roi]

        a, b, c = next(letters), next(letters), next(letters)
        o.source(f"{a}_{level}_whole_brain_bars", avg)
        o.source(f"{b}_{level}_whole_brain_curves", group_curves(scores, masks))
        o.source(f"{c}_{level}_roi_curves", roi_source)
        times = np.sort(scores.time.unique())
        o.stat(level, {
            "whole_brain": {
                "n": group_counts(scores),
                "mean": avg.groupby("group", observed=True).score.mean().to_dict(),
                "sem": avg.groupby("group", observed=True).score.sem().to_dict(),
                "wilcoxon_greater": bar_stats["wilcoxon"],
                "mannwhitney": bar_stats["mannwhitney"],
                "significant_s": {g: sig_spans(times, m) for g, m in masks.items()},
            },
            "rois": roi_stats,
        })
    style.panel_letters(letter_axes)
    brain.rasterize_large(fig)
    o.figure(fig)
    o.write_stats()
