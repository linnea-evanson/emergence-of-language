"""Fig. 3: anatomical, temporal and developmental bases of the language hierarchy.

Rows: phonetic, lexical, supralexical. A, D, G: TRF encoding map (R of the phonetic
model; gain of adding lexical, then supralexical features), significant electrodes
coloured. B, E, H: whole-brain decoding time course of each feature, averaged over
participants (dots: cluster permutation against 0). C, F, I: mean score between -1
and 2 s of each feature within each age group (stars: Wilcoxon against 0; brackets:
Mann-Whitney U on the mean over features, younger < older).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import brain, data, style
from .fig4 import group_mwu, sig_spans
from .output import Output
from .stats import p_to_stars, permute_cluster, sig_to_plot, wilcoxon_greater, window_mean

NAME = "fig3"
FEATURE_LABELS = {
    "approximant": "Approx.", "fricative": "Fricative", "high": "High", "labial": "Labial",
    "low": "Low", "mid": "Mid", "nasal": "Nasal", "plosive": "Plosive", "trill": "Trill",
    "velar": "Velar", "voice": "Voice", "vowel": "Vowel",
    "ADJ": "Adj", "ADP": "Adp", "ADV": "Adv", "CONJ": "Conj", "DET": "Det", "NOUN": "Noun",
    "PRON": "Pron", "VERB": "Verb", "wordfreq": "Zipf freq.",
    "treedepth": "Tree depth", "wordpos": "Word position", "nclosing": "Closing nodes",
    "nopening": "Opening nodes", "deplength": "Dep. length",
}
#: Brain map of each row: (TRF model, baseline or None, legend title, colormap, vmax).
BRAIN_PANELS = {
    "phonetic": (data.TRF_PHONETIC, None, style.RLABEL, "Blues_r", 0.5),
    "lexical": (data.TRF_LEXICAL, data.TRF_PHONETIC, r"$\Delta\mathcal{R}$", "Reds_r", 0.05),
    "supralexical": (data.TRF_SUPRA, data.TRF_LEXICAL, r"$\Delta\mathcal{R}$", "Purples_r", 0.05),
}


def feature_colors(level: str, n: int) -> list:
    return style.get_colors(n, style.LEVEL_CMAPS[level], skip=5)


def feature_means(feats: pd.DataFrame) -> pd.DataFrame:
    """Per participant and feature: mean score between -1 and 2 s."""
    return window_mean(feats, style.AVG_TMIN, style.AVG_TMAX, ["participant", "age", "group", "feature"])


def trf_brain(ax, trf: pd.DataFrame, level: str) -> tuple[pd.DataFrame, int]:
    """Encoding map of one row; returns the electrodes drawn and the number significant."""
    model, baseline, title, cmap, vmax = BRAIN_PANELS[level]
    values = data.trf_scores(trf, model, baseline)
    selected = data.trf_significant(trf, model, baseline)
    drawn = brain.plot_brain_values(ax, values, selected, vmax=vmax, cmap=cmap, title=title,
                                    fs=10, anchor=(0.75, -0.12))
    return drawn, len(selected)


def feature_curves(ax, feats: pd.DataFrame, level: str, order: list[str]) -> tuple[pd.DataFrame, dict]:
    """Mean time course of each feature, one row of cluster dots per feature."""
    colors = feature_colors(level, len(order))
    mean = feats.groupby(["feature", "time"]).score.agg(["mean", "sem"]).reset_index()
    for feature, color in zip(order, colors):
        sub = mean[mean.feature == feature]
        ax.plot(sub.time, sub["mean"], color=color, lw=1.4)
    style.minimal_yaxis(ax)
    ymax = ax.get_ylim()[1]
    times = np.sort(feats.time.unique())
    a, c = 0.15 * ymax, 0.015 * ymax
    masks = {}
    for j, (feature, color) in enumerate(zip(order[::-1], colors[::-1])):
        masks[feature] = permute_cluster(feats[feats.feature == feature])
        if (dots := sig_to_plot(masks[feature])) is not None:
            ax.scatter(times, -(dots * a) + j * c, color=color, s=1, zorder=5)
    ax.set_ylim(-a - c, ymax)
    ax.set_xlabel("Time (s)")
    ax.set_title(data.LEVEL_TITLES[level], color=colors[len(colors) // 2], fontsize=13)
    ax.legend([plt.Line2D([0], [0], color=c, lw=0) for c in colors[::-1]],
              [FEATURE_LABELS.get(f, f) for f in order[::-1]], labelcolor=colors[::-1],
              frameon=False, ncol=2 if len(order) > 6 else 1, loc="upper right", handlelength=0,
              handletextpad=0, columnspacing=0.8, labelspacing=0.1, fontsize=9)
    sig = pd.concat([pd.DataFrame({"feature": f, "time": times, "significant": m}) for f, m in masks.items()])
    return mean.merge(sig, on=["feature", "time"]), masks


def feature_bars(ax, avg: pd.DataFrame, level: str, order: list[str]) -> tuple[pd.DataFrame, dict]:
    """Mean and SEM of each feature within each age group; Wilcoxon stars on the bars,
    Mann-Whitney brackets on the per-participant mean over features."""
    colors = feature_colors(level, len(order))
    n = len(order)
    centers, rows = [], []
    top = 0.0  # highest bar + star, so brackets clear the stars
    for g, group in enumerate(data.GROUPS):
        x0 = g * (n + 1.5)
        centers.append(x0 + (n - 1) / 2)
        for i, (feature, color) in enumerate(zip(order, colors)):
            vals = avg.loc[(avg.group == group) & (avg.feature == feature), "score"].dropna()
            m, se = vals.mean(), vals.sem()
            top = max(top, m + 1.2 * se)
            ax.bar(x0 + i, m, width=1.0, color=color)
            ax.plot([x0 + i] * 2, [m - se, m + se], color="0.25", lw=1)
            p = wilcoxon_greater(vals.to_numpy())
            if stars := p_to_stars(p):
                ax.text(x0 + i, m + 1.2 * se, stars, ha="center", va="center", fontweight="bold",
                        color="black", fontsize=11)
            rows.append(dict(group=group, feature=feature, mean=m, sem=se, n=len(vals), wilcoxon_p=p))
    style.minimal_yaxis(ax)
    per_participant = avg.groupby(["participant", "group"], observed=True).score.mean().reset_index()
    mwu = group_mwu(per_participant)
    ymin, ymax = ax.get_ylim()
    step = ymax * 0.08
    ymax = max(ymax, top * 1.08)
    n_brackets = 0
    for (g1, g2), p in zip([(0, 1), (0, 2), (1, 2)], mwu.values()):
        if stars := p_to_stars(p):
            n_brackets += 1
            style.draw_bracket(ax, centers[g1], centers[g2], ymax + step * n_brackets, stars)
    ax.set_ylim(ymin, ymax + step * (n_brackets + 0.5) if n_brackets else ymax)
    ax.set_xticks(centers, data.GROUPS)
    ax.set_title(data.LEVEL_TITLES[level], color=colors[len(colors) // 2], fontsize=13)
    return pd.DataFrame(rows), mwu


def make(out: Path) -> None:
    o = Output(out, NAME)
    trf = data.trf_linguistic()
    feats_all = data.decoding_features("whole_brain")
    fig = plt.figure(figsize=(15, 4.0 * len(data.LEVELS)), layout="constrained")
    gs = fig.add_gridspec(len(data.LEVELS), 3, width_ratios=[1.0, 1.1, 1.1])
    letters = iter("ABCDEFGHI")
    letter_axes = []
    for r, level in enumerate(data.LEVELS):
        feats = feats_all[feats_all.level == level]
        avg = feature_means(feats)
        order = avg.groupby("feature").score.mean().sort_values().index.tolist()
        ax_brain, ax_curve, ax_bar = (fig.add_subplot(gs[r, i]) for i in range(3))
        drawn, n_sig = trf_brain(ax_brain, trf, level)
        curves, masks = feature_curves(ax_curve, feats, level, order)
        bars, mwu = feature_bars(ax_bar, avg, level, order)
        for ax in (ax_curve, ax_bar):
            ax.set_ylabel(style.RLABEL, rotation=0)
        letter_axes += [ax_brain, ax_curve, ax_bar]

        a, b, c = next(letters), next(letters), next(letters)
        o.source(f"{a}_{level}_trf_brain", drawn.drop(columns="s"))
        o.source(f"{b}_{level}_feature_curves", curves)
        o.source(f"{c}_{level}_feature_bars", avg)
        level_curve = data.decoding_levels("whole_brain", [level]).groupby("time").score.mean()
        times = np.sort(feats.time.unique())
        o.stat(level, {
            "trf_n_significant": n_sig,
            "trf_n_drawn": int(len(drawn)),
            "trf_n_drawn_significant": int(drawn.selected.sum()),
            "peak_ms": int(round(1000 * level_curve.idxmax())),
            "feature_order": order,
            "significant_s": {f: sig_spans(times, m) for f, m in masks.items()},
            "wilcoxon_greater": {g: dict(zip(sub.feature, sub.wilcoxon_p))
                                 for g, sub in bars.groupby("group", sort=False)},
            "mannwhitney_feature_mean": mwu,
        })
    style.panel_letters(letter_axes)
    brain.rasterize_large(fig)
    o.figure(fig, png=True)
    o.write_stats()
