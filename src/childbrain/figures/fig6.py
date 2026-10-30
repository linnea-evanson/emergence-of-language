"""Fig. 6: decoding of the best layer of each model across age groups, and training gain.

* A/F: model schematic (as Fig. 5).
* B/G: per-participant whole-brain decoding score of the best-decoded layer (wav2vec 2.0
  ``0.8``, Llama 3.1 ``0.1``), averaged between -1 and 2 s; one-sided Wilcoxon against 0
  (white stars) and one-sided Mann-Whitney U between age groups (brackets).
* C-D/H-I: mean decoding time course of each age group, whole brain and per ROI, with
  cluster-permutation significance dots.
* E/J: training gain ΔT R = R(layer) - R(Rand. layer) of both layers (0.1, 0.8), mean over
  each participant's STG electrodes, per age group: one-sided Wilcoxon against 0 (white
  stars) and one-sided Mann-Whitney U between age groups within a layer (brackets).

The statistics of the text also give the training gain over the whole brain.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sstats

from . import brain, data, style
from .fig5 import BEST, GAIN_ROI, feature, group_pairs, intervals, model_schematic
from .output import Output
from .stats import mannwhitney, paper_bar_stars, wilcoxon_greater

#: Layers of the training-gain panels (E/J).
GAIN_LAYERS = ["0.1", "0.8"]


def _ylims(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Y limits of each ROI's curves: from the extreme group means."""
    ylims = {}
    for roi, g in df.groupby("roi"):
        m = g.groupby(["group", "time"], observed=True).score.mean()
        hi = 1.15 * m.max()
        ylims[str(roi)] = (min(1.1 * m.min(), -0.2 * hi), hi)
    return ylims


def roi_curves(ax, df: pd.DataFrame, roi: str, ylim: tuple[float, float], legend: bool,
               fs: float = 14) -> tuple[pd.DataFrame, dict]:
    """Mean time course of each age group, with cluster dots."""
    colors = style.ROI_COLORS[roi]
    sns.lineplot(df.assign(group=df.group.astype(str)), x="time", y="score", hue="group",
                 hue_order=data.GROUPS, palette=colors, errorbar=None, ax=ax, legend=False)
    ax.set_title(data.ROI_TITLES[roi], color=colors[1])
    c = ylim[1] * 0.05
    masks = style.cluster_dots(ax, df, colors, a=-ylim[0] - 2 * c, c=c)
    if legend:
        handles = [plt.Line2D([0], [0], color=c, label=f"{g}yrs") for c, g in zip(colors, data.GROUPS)]
        ax.legend(handles=handles[::-1], frameon=False, labelspacing=0.1, loc="upper left",
                  bbox_to_anchor=(0.06, 1), borderpad=0, borderaxespad=0, labelcolor=colors[::-1], handlelength=0,
                  handletextpad=0, fontsize=fs)
    ax.set_ylabel(None)
    ax.set_xlabel("Time (s)")
    sns.despine(ax=ax)
    ticks = [t for t in ax.get_yticks() if ylim[0] <= t <= ylim[1]]
    ax.set_yticks(ticks, labels=[str(np.round(t, 3)).replace("0.", ".")
                                 if t in (0, max(ticks)) else "" for t in ticks])
    ax.axhline(0, color="lightgrey", linestyle="--", zorder=-1)
    ax.set_ylim(ylim)
    times = np.sort(df.time.unique())
    curves = (df.groupby(["group", "time"], observed=True).score.mean().rename("mean_score")
              .reset_index().assign(roi=roi))
    curves["significant"] = [bool(masks[str(g)][np.searchsorted(times, t)])
                             for g, t in zip(curves.group, curves.time)]
    return curves, {g: intervals(times, m) for g, m in masks.items()}


def bars(ax, df: pd.DataFrame, title: str | None, space: float = 0.1, star_fs: float = 20) -> dict:
    """Age-group bars of ``score``: one-sided Wilcoxon against 0
    (white stars) and Mann-Whitney U between groups (brackets stacked above the axis top);
    the axis extends below 0 if a bar does."""
    colors = style.palette("whole_brain")
    sns.barplot(df.assign(group=df.group.astype(str)), x="group", y="score", order=data.GROUPS,
                hue="group", hue_order=data.GROUPS, palette=colors, dodge=False, errorbar="se",
                ax=ax, legend=False)
    st: dict = {"wilcoxon": style.wilcoxon_stars(ax, df, "score", fontsize=star_fs,
                                                to_stars=paper_bar_stars),
                "mannwhitney": group_pairs(df)}
    sig = [(k, p) for k, p in st["mannwhitney"].items() if p < 0.05]
    for j, (pair, p) in enumerate(sig):
        x1, x2 = (data.GROUPS.index(g) for g in pair.split("<"))
        y = ax.get_ylim()[1] * (1 + space * j)
        ax.hlines(xmin=x1, xmax=x2, y=y, color="k", lw=2)
        ax.text((x1 + x2) / 2, y - 0.01 * space, paper_bar_stars(p), ha="center", fontweight="bold",
                fontsize=star_fs)
    style.age_xticks(ax, colors)
    sns.despine(ax=ax)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(-0.01 * ymax, ymax)
    ticks = [t for t in ax.get_yticks() if 0 <= t <= ymax]
    ax.set_yticks(ticks, labels=[str(np.round(t, 3)).replace("0.", ".")
                                 if t in (0, max(ticks)) else "" for t in ticks])
    if title:
        ax.set_title(title, color=colors[1])
    summary = df.groupby("group", observed=True).score.agg(["mean", "sem"])
    lo = (summary["mean"] - summary["sem"]).min()
    if lo < 0:
        hi = max(ymax, (summary["mean"] + summary["sem"]).max())
        ax.set_ylim(1.15 * lo, hi)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_yticks([lo, 0, hi], labels=[style.format_tick(lo), "0", style.format_tick(hi)])
    st["n"] = {g: int((df.group.astype(str) == g).sum()) for g in data.GROUPS}
    st["mean"] = summary["mean"].to_dict()
    st["sem"] = summary["sem"].to_dict()
    return st


def training_gain(trf: pd.DataFrame, model: str, roi: str, layer: str) -> pd.DataFrame:
    """ΔT R of ``layer``, per-participant mean over the ROI's electrodes."""
    return data.roi_means(data.trf_scores(trf, feature(model, layer),
                                          feature(model, f"Rand. {layer}")), [roi])


def gain_layers(ax, trf: pd.DataFrame, model: str, fs: float = 12) -> tuple[pd.DataFrame, dict]:
    """E/J: STG training gain of each layer per age group; white stars, one-sided Wilcoxon
    against 0; brackets, one-sided Mann-Whitney U between groups within a layer."""
    df = pd.concat([training_gain(trf, model, GAIN_ROI, layer).assign(layer=layer)
                    for layer in GAIN_LAYERS], ignore_index=True)
    df["group"] = df.group.astype(str)
    sns.barplot(df, x="layer", y="score", hue="group", order=GAIN_LAYERS, hue_order=data.GROUPS,
                palette=style.palette("whole_brain"), errorbar="se", ax=ax, legend=False, width=0.8)
    st: dict = {layer: {"wilcoxon": {}, "mean": {}, "sem": {}, "n": {}, "mannwhitney": {}}
                for layer in GAIN_LAYERS}
    tops = {}
    for g, container in zip(data.GROUPS, ax.containers):
        for layer, patch in zip(GAIN_LAYERS, container.patches):
            v = df.score[(df.group == g) & (df.layer == layer)].to_numpy()
            sem = float(v.std(ddof=1) / np.sqrt(len(v)))
            p = wilcoxon_greater(v)
            st[layer]["wilcoxon"][g], st[layer]["mean"][g] = p, float(v.mean())
            st[layer]["sem"][g], st[layer]["n"][g] = sem, len(v)
            x = patch.get_x() + patch.get_width() / 2
            tops[(layer, g)] = (x, v.mean() + sem)
            if stars := paper_bar_stars(p):
                ax.text(x, 0, stars, ha="center", va="bottom", color="white", fontweight="bold",
                        fontsize=fs * 0.8)
    lo, hi = ax.get_ylim()
    span, top_y = hi - lo, hi
    for layer in GAIN_LAYERS:
        top, k = max(tops[(layer, g)][1] for g in data.GROUPS), 0
        for i, g1 in enumerate(data.GROUPS):
            for g2 in data.GROUPS[i + 1:]:
                p = mannwhitney(df.score[(df.group == g1) & (df.layer == layer)],
                                df.score[(df.group == g2) & (df.layer == layer)])
                st[layer]["mannwhitney"][f"{g1}<{g2}"] = p
                if p < 0.05:
                    y = top + span * (0.05 + 0.12 * k)
                    x1, x2 = tops[(layer, g1)][0], tops[(layer, g2)][0]
                    ax.hlines(y, x1, x2, color="k", lw=2)
                    ax.text((x1 + x2) / 2, y, paper_bar_stars(p), ha="center", va="bottom",
                            fontweight="bold", fontsize=fs * 1.3)
                    top_y = max(top_y, y + span * 0.14)
                    k += 1
    ax.set_ylim(min(lo, 0), top_y)
    ax.set_xticks(range(len(GAIN_LAYERS)), GAIN_LAYERS)
    ax.set_xlabel("Layer")
    ymax = ax.get_ylim()[1]
    ax.set_yticks([0, ymax], ["0", style.format_tick(ymax)])
    sns.despine(ax=ax)
    return df, st


def gain_age(gain: pd.DataFrame) -> dict:
    """Text: Wilcoxon against 0 within each group, age-group tests, and one-sided Spearman
    correlation of the gain with exact age (:func:`data.has_exact_age`)."""
    out = dict(n=int(len(gain)), mannwhitney=group_pairs(gain), wilcoxon={
        g: wilcoxon_greater(gain.score[gain.group.astype(str) == g].to_numpy()) for g in data.GROUPS})
    if data.has_exact_age():
        res = sstats.spearmanr(gain.age, gain.score, alternative="greater")
        out.update(spearman_r=float(res.statistic), spearman_p_greater=float(res.pvalue))
    return out


def make(out_dir: Path) -> None:
    out = Output(out_dir, "fig6")
    trf = data.trf_embeddings()
    dec = data.decoding_embeddings()
    fig, axs = plt.subplot_mosaic(
        [["A", "B", "C0", "C1", "C2", "C3", "D"], ["A", "B", "C0", "C4", "C5", "C6", "D"],
         ["G", "H", "I0", "I1", "I2", "I3", "J"], ["G", "H", "I0", "I4", "I5", "I6", "J"]], figsize=(20, 6.5), width_ratios=(1, 1, 2, 1, 1, 1, 1),
                                  gridspec_kw={"wspace": 0.2, "hspace": 0.4})
    # model, paper letters (bars, curves, ROI curves, gain), mosaic keys (schematic, bars, curves, gain)
    for model, letters, (kA, kB, kC, kD) in [("wav2vec", "BCDE", ("A", "B", "C", "D")),
                                             ("llama", "GHIJ", ("G", "H", "I", "J"))]:
        model_schematic(axs[kA], model)
        d = dec[dec.embedding == feature(model, BEST[model])]
        ylims = _ylims(d)
        curves, masks = [], {}
        for i, roi in enumerate(data.ROIS):
            ax = axs[f"{kC}{i}"]
            cv, masks[roi] = roi_curves(ax, d[d.roi == roi], roi, ylims[roi],
                                        legend=roi == "whole_brain" and model == "wav2vec")
            curves.append(cv)
            if (model == "wav2vec" and i) or (model == "llama" and i in (1, 2, 3)):
                ax.set_xlabel("")
                ax.set_xticklabels([])
        axs[f"{kC}0"].set_ylabel(style.RLABEL, rotation=0)
        axs[f"{kC}0"].yaxis.set_label_coords(-0.04, 0.5)
        curves_df = pd.concat(curves, ignore_index=True)
        out.source(f"{letters[1]}_{model}_decoding_whole_brain",
                   curves_df[curves_df.roi == "whole_brain"])
        out.source(f"{letters[2]}_{model}_decoding_rois", curves_df[curves_df.roi != "whole_brain"])
        out.stat(f"{letters[1]}{letters[2]}_{model}_significant_times", masks)

        wb = d[(d.roi == "whole_brain") & d.time.between(style.AVG_TMIN, style.AVG_TMAX)]
        means = data.add_age(wb.groupby("participant", as_index=False).score.mean())
        out.stat(f"{letters[0]}_{model}_decoding_age", bars(axs[kB], means, "Whole brain", space=0.08))
        out.source(f"{letters[0]}_{model}_decoding_mean", means)
        axs[kB].set_ylabel(style.RLABEL, rotation=0)
        axs[kB].yaxis.set_label_coords(-0.07, 0.5)  # next to its axis (ticks only at the ends)

        gain, gain_stats = gain_layers(axs[kD], trf, model)
        out.stat(f"{letters[3]}_{model}_training_gain_{GAIN_ROI}", gain_stats)
        out.source(f"{letters[3]}_{model}_training_gain", gain)
        axs[kD].set_ylabel(r"$\Delta_T\mathcal{R}$", rotation=0)
        axs[kD].yaxis.set_label_coords(-0.14, 0.5)
        out.stat(f"text_{model}_training_gain", {
            layer: {roi: gain_age(training_gain(trf, model, roi, layer))
                    for roi in ("whole_brain", GAIN_ROI)} for layer in GAIN_LAYERS})
        if model == "wav2vec":  # tick labels on the bottom row only
            for k in (kB, f"{kC}0", kD):
                axs[k].set_xlabel("")
                axs[k].set_xticklabels([])
    brain.roi_inset(axs["C0"].inset_axes((0.55, 0.55, 0.45, 0.45)))
    axs["D"].set_title(f"Training Gain ({GAIN_ROI})\n"
                       + r"$\Delta_T\mathcal{R}=\mathcal{R}_{train}-\mathcal{R}_{rand.}$")
    for x, y, letter in [(0.12, 0.925, "A"), (0.2, 0.925, "B"), (0.31, 0.925, "C"),
                         (0.49, 0.925, "D"), (0.785, 0.925, "E"), (0.12, 0.48, "F"),
                         (0.2, 0.48, "G"), (0.31, 0.48, "H"), (0.49, 0.48, "I"), (0.785, 0.48, "J")]:
        fig.text(x, y, letter, fontsize=16, fontweight="bold")
    out.figure(fig)
    out.write_stats()
