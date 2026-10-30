"""Fig. S collinearity: predictor correlation (A), feature-to-feature decoding of the
stimulus at word onset (B) and over time (C).

Data: ``stimulus/predictor_correlation.csv``,
``decoding/stimulus_feature_to_feature.csv.gz``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm

from . import data, style
from .output import Output

#: Feature families of the stimulus decoding (``source``, ``target_group``), in order.
FAMILIES = ["melspec", "phoneme_features", "wordfreq", "pos", "treedepth", "nopening", "nclosing",
            "deplength", "wordpos", "wav2vec_L0.1", "wav2vec_L0.8", "llama_L0.1", "llama_L0.8"]
FAMILY_LABELS = {
    "melspec": "Mel spectrogram", "phoneme_features": "Phoneme", "wordfreq": "Zipf freq.",
    "pos": "Part of speech", "treedepth": "Tree depth", "wordpos": "Word position",
    "nclosing": "Closing nodes", "nopening": "Opening nodes", "deplength": "Dep. length",
    "wav2vec_L0.1": "wav2vec 2.0 0.1", "wav2vec_L0.8": "wav2vec 2.0 0.8",
    "llama_L0.1": "Llama 3.1 0.1", "llama_L0.8": "Llama 3.1 0.8",
}
#: Linguistic level of each linguistic family (colour cmap of ``style.LEVEL_CMAPS``).
FAMILY_LEVELS = {
    "acoustic": ["melspec"], "phonetic": ["phoneme_features"], "lexical": ["pos", "wordfreq"],
    "supralexical": ["treedepth", "nopening", "nclosing", "deplength", "wordpos"],
}
MODEL_FAMILIES = ["llama_L0.1", "llama_L0.8", "wav2vec_L0.1", "wav2vec_L0.8"]

#: Predictor-correlation boxes: families boxed together, and box colours.
PREDICTOR_GROUPS = {
    **dict.fromkeys(["Zipf freq.", "Part of speech"], "Lexical"),
    **dict.fromkeys(["Tree depth", "Word position", "Opening nodes", "Closing nodes",
                     "Dep. length"], "Supralexical"),
    **dict.fromkeys(["Llama 3.1 0.1", "Llama 3.1 0.8"], "Llama 3.1"),
    **dict.fromkeys(["wav2vec 2.0 0.1", "wav2vec 2.0 0.8"], "wav2vec 2.0"),
}
BOX_COLORS = {"Mel spectrogram": "#1f77b4", "Phoneme": "#ff7f0e", "Lexical": "#d62728",
              "Supralexical": "#8c564b", "Llama 3.1": "#08519c", "wav2vec 2.0": "#6a3d9a"}


def family_color(name: str) -> tuple:
    """Shade of the family's level colormap (models: Blues)."""
    for level, fams in FAMILY_LEVELS.items():
        if name in fams:
            return style.get_colors(len(fams), style.LEVEL_CMAPS[level], skip=2)[fams.index(name)]
    return style.get_colors(len(MODEL_FAMILIES), "Blues", skip=2)[MODEL_FAMILIES.index(name)]


def target_curves(stim: pd.DataFrame) -> pd.DataFrame:
    """Per source, target family and time: mean and SEM across the family's target features."""
    g = stim.groupby(["source", "target_group", "time"]).score
    return g.agg(score="mean", sem="sem", n_targets="size").reset_index()


def predictor_corr(out: Path) -> None:
    o = Output(out, "figS_predictor_corr")
    df = data.predictor_correlation()
    labels = df.predictor.tolist()
    families = df.family.tolist()
    R = df[labels].to_numpy(float)
    n = len(R)

    fig, ax = plt.subplots(figsize=(11, 9.5))
    bounds = [b / 10 for b in range(-10, 11) if not -1 < b < 1]
    cmap = plt.get_cmap("bwr", len(bounds) - 1)
    lower = np.where(np.tril(np.ones_like(R, dtype=bool)), R, np.nan)
    edges = np.arange(n + 1) - 0.5
    im = ax.pcolormesh(edges, edges, np.ma.masked_invalid(lower), cmap=cmap,
                       norm=BoundaryNorm(bounds, cmap.N))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(n), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n), labels, fontsize=7)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    boxes = [PREDICTOR_GROUPS.get(f, f) for f in families]
    start = 0
    for i in range(1, n + 1):
        if i == n or boxes[i] != boxes[start]:
            color = BOX_COLORS.get(boxes[start], "k")
            ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5), i - start, i - start,
                                       fill=False, lw=1.8, edgecolor=color, zorder=5))
            ax.text(i - 0.2, start - 0.3, boxes[start], color=color, fontsize=9,
                    fontweight="bold", va="bottom", ha="left", zorder=6)
            start = i
    ticks = [b for b in bounds if abs(round(b * 10)) % 2 == 0 or abs(b) == 0.1]
    cbar = fig.colorbar(im, ax=ax, shrink=0.45, ticks=ticks, pad=0.02, anchor=(0.0, 1.0),
                        panchor=(1.0, 1.0))
    cbar.solids.set_rasterized(False)  # type: ignore[union-attr]
    cbar.set_label("Pearson correlation", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Predictor cross-correlation", fontsize=11)
    o.figure(fig)
    o.source("A_predictor_correlation", df)


def feature_matrix(out: Path, curves: pd.DataFrame) -> None:
    """Decoding score at the time sample closest to word onset."""
    o = Output(out, "figS_feature_to_feature_matrix")
    times = curves.time.unique()
    t0 = float(times[np.argmin(np.abs(times))])
    mat = (curves[curves.time == t0].pivot(index="source", columns="target_group", values="score")
           .loc[FAMILIES, FAMILIES])
    fig, ax = plt.subplots(figsize=(0.65 * mat.shape[1] + 2.5, 0.55 * mat.shape[0] + 2.0),
                           constrained_layout=True)
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=max(float(np.nanmax(mat)), 0.05),
                linewidths=0.5, xticklabels=[FAMILY_LABELS[c] for c in mat.columns],
                yticklabels=[FAMILY_LABELS[r] for r in mat.index],
                cbar_kws={"label": style.RLABEL, "shrink": 0.8}, ax=ax)
    ax.collections[0].colorbar.solids.set_rasterized(False)  # type: ignore[attr-defined,union-attr]
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
    ax.set_xlabel("Target feature")
    ax.set_ylabel("Source feature")
    ax.set_title("Feature-to-feature decoding at t = 0 s", fontsize=11)
    o.figure(fig)
    o.source("B_matrix", mat.rename_axis(index="source", columns=None).reset_index()
             .assign(time=t0))
    o.stat("time_s", t0)
    o.stat("diagonal", {f: float(mat.loc[f, f]) for f in FAMILIES})
    o.write_stats()


def feature_curves(out: Path, curves: pd.DataFrame, stim: pd.DataFrame) -> None:
    """Source x target grid of time courses (mean +- SEM across the target's features)."""
    o = Output(out, "figS_feature_to_feature_curves")
    ymax = max(float(stim.score.max()), 0.05)
    n = len(FAMILIES)
    fig, axes = plt.subplots(n, n, figsize=(1.15 * n + 1.8, 0.85 * n + 1.5), sharex=True,
                             sharey=True, constrained_layout=True)
    for i, source in enumerate(FAMILIES):
        for j, target in enumerate(FAMILIES):
            ax = axes[i, j]
            c = curves[(curves.source == source) & (curves.target_group == target)]
            color = family_color(target)
            ax.plot(c.time, c.score, color=color, lw=1.2)
            sem = c["sem"].fillna(0)
            ax.fill_between(c.time, c.score - sem, c.score + sem, color=color, alpha=0.2, lw=0)
            ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.45)
            ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.65)
            ax.set_ylim(-0.02 * ymax, ymax * 1.05)
            sns.despine(ax=ax)
            if i == 0:
                ax.set_title(FAMILY_LABELS[target], fontsize=9, pad=4)
            if j == 0:
                ax.annotate(textwrap.fill(FAMILY_LABELS[source], 12), xy=(-0.45, 0.5),
                            xycoords="axes fraction", ha="right", va="center", fontsize=8)
                ax.set_ylabel(style.RLABEL if i == n // 2 else "", fontsize=9)
            else:
                ax.tick_params(labelleft=False)
            if i < n - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Time (s)", fontsize=8)
    fig.supxlabel("Time relative to word onset (s)", fontsize=10, y=-0.02)
    fig.suptitle("Feature-to-feature decoding time courses", fontsize=11, y=1.02)
    o.figure(fig)
    o.source("C_curves", curves)


def make(out: Path) -> None:
    predictor_corr(out)
    stim = data.stimulus_decoding()
    curves = target_curves(stim)
    feature_matrix(out, curves)
    feature_curves(out, curves, stim)
