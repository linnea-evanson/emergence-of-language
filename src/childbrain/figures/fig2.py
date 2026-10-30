"""Fig. 2: encoding of the phonetic and lexical features, R(phonemes+words), by age group.

A, per electrode R(phonetic+lexical) on the left lateral cortex for each age group,
coloured by ROI (white: outside the figure ROIs; small black dots: not significant,
permutation p with BH-FDR across every electrode). B, per-participant mean R in each
ROI (whole brain: every electrode, located or not), mean and SEM by age group; white
stars, one-sided Wilcoxon > 0 within a group; black brackets, one-sided Mann-Whitney U
between groups. The B y axes are shared in tiers: STG; STS + ATL; the others.

Also the statistics of the text: label permutation of the group differences, numbers of
significant electrodes of the three nested TRFs, and the supralexical gain ΔR(supralexical).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import brain, data, style
from .output import Output
from .stats import label_permutation as permutation_p, mannwhitney, paper_bar_stars, wilcoxon_greater

#: B panels sharing a y axis (the STG keeps its own).
YLIM_TIERS = [["STS", "ATL"], ["whole_brain", "IPL", "L/IFC", "OCC"]]
#: Drawing order of the A electrodes (later on top).
ROI_ORDER = ["whole_brain", "STG", "STS", "ATL", "L/IFC", "IPL", "OCC"]
PAIRS = [(0, 1), (0, 2), (1, 2)]
STAR_FS = 15


def _bars(ax, means: pd.DataFrame, roi: str, line_y: float | None = None,
          spacing: float = 0.1) -> dict:
    """One B panel; brackets stacked from ``line_y`` (default:
    the axes top) by ``spacing`` x ``line_y``. Returns the p-values."""
    colors = style.ROI_COLORS[roi]
    vals = means[means.roi == roi].assign(group=lambda d: d.group.astype(str))
    sns.barplot(vals, x="group", y="score", errorbar="se", order=data.GROUPS, ax=ax, palette=colors,
                hue="group", hue_order=data.GROUPS, dodge=False, legend=False)
    ax.set_title(data.ROI_TITLES[roi], color=colors[1])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    out: dict = {}
    for x, g in enumerate(data.GROUPS):
        v = vals.score[vals.group == g].to_numpy()
        p = wilcoxon_greater(v)
        out[g] = dict(R=float(v.mean()), p=p, n=len(v))
        if s := paper_bar_stars(p):
            ax.text(x, 0, s, ha="center", va="center", fontweight="bold", color="white", fontsize=STAR_FS)
    j = 0
    for a, b in PAIRS:
        ga, gb = data.GROUPS[a], data.GROUPS[b]
        p = mannwhitney(vals.score[vals.group == ga], vals.score[vals.group == gb])
        out[f"{ga}<{gb}"] = p
        if s := paper_bar_stars(p):
            top = ax.get_ylim()[1] if line_y is None else line_y
            y = top + top * spacing * j
            ax.hlines(y=y, xmin=a, xmax=b, color="k", lw=2)
            ax.text((a + b) / 2, y - 0.01 * spacing, s, ha="center", fontweight="bold", fontsize=STAR_FS)
            j += 1
    ymax = ax.get_ylim()[1]
    ax.set_ylim(-ymax * 0.01, ymax)
    _ticks(ax, [t for t in ax.get_yticks() if 0 <= t <= ymax])
    sns.despine(ax=ax)
    return out


def _ticks(ax, ticks: list[float]) -> None:
    ax.set_yticks(ticks, labels=[str(np.round(t, 3)).replace("0.", ".") if t in (0, ticks[-1]) else ""
                                 for t in ticks])


def _brain(ax, values: pd.DataFrame, selected: set[tuple[str, str]]) -> pd.DataFrame:
    """Electrodes coloured by ROI, sized by R (0-0.5), on the left lateral cortex."""
    el = data.electrodes()
    el = el[el.x.notna() & ~el.medial]
    df = values.merge(el[["participant_id", "electrode_id", "y", "z", "roi"]],
                      left_on=["participant", "electrode"], right_on=["participant_id", "electrode_id"])
    df["roi"] = df.roi.fillna("whole_brain")
    df["color"] = [(1.0, 1.0, 1.0, 1.0) if r == "whole_brain" else style.ROI_COLORS[r][2 if r == "IPL" else 1]
                   for r in df.roi]
    df["s"] = brain.get_sizes(df.score.to_numpy(), 0, 0.5)
    df["selected"] = [k in selected for k in zip(df.participant, df.electrode)]
    df["order"] = df.roi.map(ROI_ORDER.index)
    df = df.sort_values(["order", "s"], kind="stable")
    brain.brain_background(ax)
    ns, sig = df[~df.selected], df[df.selected]
    ax.scatter(-ns.y, ns.z, color="grey", s=2, edgecolors="k", linewidth=0.5)
    ax.scatter(-sig.y, sig.z, c=list(sig.color), s=sig.s, edgecolors="k", linewidth=0.5)
    ax.set_axis_off()
    return df


def label_permutation(means: pd.DataFrame, n_perm: int = 10_000, seed: int = 0) -> dict:
    """Label permutation of the B group differences (younger < older); one generator."""
    rng = np.random.default_rng(seed)
    out: dict = {}
    for roi in data.ROIS:
        d = means[means.roi == roi]
        out[roi] = {}
        for a, b in PAIRS:
            x = d.score[d.group == data.GROUPS[a]].to_numpy()
            y = d.score[d.group == data.GROUPS[b]].to_numpy()
            out[roi][f"{data.GROUPS[a]}<{data.GROUPS[b]}"] = permutation_p(x, y, rng, n_perm)
    return out


def _test(v: pd.Series) -> dict:
    return dict(mean=float(v.mean()), n_pos=int((v > 0).sum()), n=len(v), p=wilcoxon_greater(v.to_numpy()))


def make(out_dir: Path) -> None:
    out = Output(out_dir, "fig2")
    trf = data.trf_linguistic()
    values = data.trf_scores(trf, data.TRF_LEXICAL)
    selected = data.trf_significant(trf, data.TRF_LEXICAL)
    means = data.roi_means(values)
    means["group"] = means.group.astype(str)

    row1: list = [f"e{i}" for i in range(3) for _ in range(7)]
    row2: list = [f"f{j}" for j in range(7) for _ in range(3)]
    fig, axs = plt.subplot_mosaic([row1, row1, row2], figsize=(17, 7.5),
                                  gridspec_kw={"wspace": 0.5, "hspace": 0.3})
    shown = []
    for i, group in enumerate(data.GROUPS):
        ax = axs[f"e{i}"]
        df = _brain(ax, values[values.group.astype(str) == group], selected)
        ax.set_aspect(1.09, adjustable="datalim")  # proportions of the brains of Figs. 1, 3 and 5
        ax.set_title(f"{group} years old", fontsize=18)
        for coll in ax.collections:
            coll.set_rasterized(True)
        shown.append(df.assign(group=group))
    shown_df = pd.concat(shown, ignore_index=True)
    out.source("A_electrodes", shown_df[["participant", "electrode", "group", "age", "y", "z", "roi",
                                         "score", "selected"]].rename(columns={"score": "r", "selected": "significant"}))

    # Brackets are stacked by a fraction of each panel's own height, which crushes them in the
    # low panels of a tier: dry run for the heights, then the same spacing within each tier.
    dry, dry_axs = plt.subplots(1, 7)
    tops = {}
    for ax, roi in zip(dry_axs, data.ROIS):
        _bars(ax, means, roi)
        tops[roi] = ax.get_ylim()[1]
    plt.close(dry)
    spacing = {roi: 0.1 * max(tops[r] for r in tier) / tops[roi] for tier in YLIM_TIERS for roi in tier}
    bar_stats = {}
    for i, roi in enumerate(data.ROIS):
        ax = axs[f"f{i}"]
        bar_stats[roi] = _bars(ax, means, roi, line_y=tops[roi] if roi in spacing else None,
                               spacing=spacing.get(roi, 0.1))
        ax.set_xticks(range(3), data.GROUPS)
        ax.tick_params(axis="x", which="both", length=0)
        for label, c in zip(ax.get_xticklabels(), style.ROI_COLORS[roi]):
            label.set_color(c)
    axs["f0"].set_ylabel(style.RLABEL, rotation=0)
    for tier in YLIM_TIERS:
        tier_axs = [axs[f"f{data.ROIS.index(roi)}"] for roi in tier]
        ymax = max(ax.get_ylim()[1] for ax in tier_axs)
        ticks = [t for t in plt.MaxNLocator(5).tick_values(0, ymax) if 0 <= t <= ymax]
        for ax in tier_axs:
            ax.set_ylim(-0.01 * ymax, ymax)
            _ticks(ax, ticks)
    # Narrow the B panels from the left until every y tick label clears the panel to its left.
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    shift = 0.0
    for i in range(1, 7):
        ax, prev = axs[f"f{i}"], axs[f"f{i - 1}"]
        boxes = [lab.get_window_extent(renderer) for lab in ax.get_yticklabels() if lab.get_text()]
        if boxes:
            x0 = fig.transFigure.inverted().transform((min(b.x0 for b in boxes), 0))[0]
            shift = max(shift, prev.get_position().x1 + 0.004 - x0)
    for i in range(7):
        pos = axs[f"f{i}"].get_position()
        axs[f"f{i}"].set_position((pos.x0 + shift, pos.y0, pos.width - shift, pos.height))

    legend_ax = axs["e0"].inset_axes((-0.1, 0, 0.1, 0.2))
    for s, lab in [(102, ".5"), (50, ".25"), (10, ".01")]:
        legend_ax.scatter([], [], s=s, color="white", label=lab, edgecolors="k", linewidth=0.5)
    legend_ax.scatter([], [], s=2, color="k", label="n.s.", edgecolors="k", linewidth=0.5)
    legend_ax.legend(title=style.RLABEL, loc="lower left", bbox_to_anchor=(-0.1, 0), fontsize=10,
                     title_fontsize=10, frameon=False)
    legend_ax.set_axis_off()
    fig.text(0.1, 0.9, "A", fontsize=16, fontweight="bold")
    fig.text(0.1, 0.34, "B", fontsize=16, fontweight="bold")
    out.figure(fig, png=True)

    bars = means.groupby(["roi", "group"], sort=False).score.agg(["mean", "sem", "count"]).reset_index()
    out.source("B_participant_means", means[["roi", "participant", "age", "group", "score"]]
               .rename(columns={"score": "r"}))
    out.source("B_bars", bars.rename(columns={"count": "n"}))

    out.stat("B_roi_groups", bar_stats)
    out.stat("label_permutation", label_permutation(means))
    out.stat("n_electrodes", int(len(values)))
    out.stat("n_participants", int(values.participant.nunique()))
    out.stat("n_located_participants", int(data.located(values).participant.nunique()))
    out.stat("n_significant", {m: len(data.trf_significant(trf, m))
                               for m in (data.TRF_PHONETIC, data.TRF_LEXICAL, data.TRF_SUPRA)})
    gain = data.trf_scores(trf, data.TRF_SUPRA, data.TRF_LEXICAL)
    gain_means = data.roi_means(gain)
    out.stat("supra_gain", {roi: _test(gain_means.score[gain_means.roi == roi]) for roi in data.ROIS})
    out.write_stats()
