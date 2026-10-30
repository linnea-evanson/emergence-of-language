"""Fig. S (age trends): whole-brain decoding of each level as a function of age.

Rows: phonetic, lexical, supralexical. Left: mean score between -1 and 2 s; right: time
of peak decoding between -2 and 3 s; one point per participant at a random position within
its age range (exact ages are not shown), linear fit with 95% CI and one-sided Spearman
(higher scores, earlier peaks with age) computed with exact ages. Also the
Spearman correlation of age with the duration of significant decoding (quoted with the
decoding results).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sstats

from . import data, style
from .output import Output
from .stats import window_mean

NAME = "figS_age_trends"
#: Time step of the decoding time courses (s).
DURATION_STEP = 0.05


def peak_times(scores: pd.DataFrame) -> pd.DataFrame:
    """Per participant: time and score of the maximum between -2 and 3 s."""
    w = scores[(scores.time >= style.METRIC_TMIN) & (scores.time <= style.METRIC_TMAX)]
    peak = w.loc[w.groupby("participant").score.idxmax(), ["participant", "age", "group", "time", "score"]]
    return peak.rename(columns={"time": "peak_time", "score": "peak_score"}).reset_index(drop=True)


def significant_duration(pvalues: pd.DataFrame, level: str, alpha: float = 0.05) -> pd.Series:
    """Per participant: longest run of consecutive times (s) at which any feature of the
    level is decoded significantly (whole brain, p < ``alpha``)."""
    sig = pvalues[(pvalues.level == level) & (pvalues.pvalue < alpha)]
    out = {}
    for participant, g in sig.groupby("participant"):
        t = np.sort(g.time.unique())
        best = run = 1
        for d in np.diff(t):
            run = run + 1 if d <= 1.5 * DURATION_STEP else 1
            best = max(best, run)
        out[participant] = best * DURATION_STEP
    return pd.Series(out, dtype=float)


def public(df: pd.DataFrame) -> pd.DataFrame:
    """Source data without exact ages: age range and plotted position instead."""
    return df.drop(columns="age").assign(
        age_range=df.participant.map(data.participants().age_range),
        x=df.participant.map(style.age_axis_positions()))


def age_panel(ax, df: pd.DataFrame, col: str, level: str, alternative: str,
              show_zero: bool) -> tuple[float, float]:
    """Points at their position on the age axis (:func:`style.age_axis_positions`), coloured
    by age group; Spearman with exact log(age) above; linear fit on the exact ages mapped onto
    the axis (same ranks), drawn across the whole axis."""
    colors = style.palette(level)
    log_age = np.log(df.age.to_numpy(float))
    y = df[col].to_numpy()
    sns.scatterplot(x=df.participant.map(style.age_axis_positions()).to_numpy(), y=y,
                    hue=df.group.astype(str).to_numpy(), hue_order=data.GROUPS, palette=colors, ax=ax,
                    legend=False, s=22, alpha=0.85, edgecolor="0.2", linewidth=0.4)
    style.age_axis(ax)
    sns.regplot(x=style.age_to_axis(np.exp(log_age)), y=y, ax=ax, scatter=False, color=colors[1],
                seed=0, truncate=False, line_kws={"linewidth": 1.5})
    style.minimal_yaxis(ax, ylabel=None, show_zero=show_zero)
    if not show_zero:
        ax.axhline(0, color="lightgrey", linestyle=":", linewidth=0.8, alpha=0.65, zorder=-1)
    style.age_axis(ax)
    ax.set_xlabel(None)
    return style.spearman_text(ax, log_age, y, alternative=alternative, show_ns=True)


def make(out: Path) -> None:
    if data.skip_without_exact_age(f"{NAME} (Fig. S5)"):
        return
    o = Output(out, NAME)
    pvalues = data.decoding_pvalues()
    fig, axes = plt.subplots(len(data.LEVELS), 2, figsize=(8, 2.6 * len(data.LEVELS)),
                             layout="constrained", squeeze=False)
    letters = iter("ABCDEF")
    for r, level in enumerate(data.LEVELS):
        scores = data.decoding_levels("whole_brain", [level])
        avg = window_mean(scores, style.AVG_TMIN, style.AVG_TMAX, ["participant", "age", "group"])
        peak = peak_times(scores)
        r_score = age_panel(axes[r, 0], avg, "score", level, "greater", show_zero=True)
        r_peak = age_panel(axes[r, 1], peak, "peak_time", level, "less", show_zero=False)
        axes[r, 0].set_ylabel(data.LEVEL_TITLES[level], fontsize=13, rotation=90,
                              color=sns.color_palette(style.LEVEL_CMAPS[level])[3])
        dur = avg[["participant", "age", "group"]].assign(
            duration_s=avg.participant.map(significant_duration(pvalues, level)).fillna(0.0))
        res = sstats.spearmanr(np.log(dur.age), dur.duration_s, alternative="greater")
        o.source(f"{next(letters)}_{level}_score_vs_age", public(avg))
        o.source(f"{next(letters)}_{level}_peak_time_vs_age", public(peak))
        o.source(f"text_{level}_significant_duration", public(dur))
        o.stat(level, {
            "n": int(len(avg)),
            "spearman_score": list(r_score),
            "spearman_peak_time": list(r_peak),
            "spearman_duration": [float(res.statistic), float(res.pvalue)],
        })
    for ax in axes[-1]:
        ax.set_xlabel("Age (years)")
    axes[0, 0].set_title(style.RLABEL, fontsize=11)
    axes[0, 1].set_title("Peak time", fontsize=11)
    style.panel_letters(axes.flatten().tolist(), fontsize=14)
    o.figure(fig)
    o.write_stats()
