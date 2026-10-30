"""Supplementary encoding figures.

* ``figS_prop_sig_electrodes``: per participant and ROI, % of electrodes significant for
  R(phonemes+words) (permutation p, BH-FDR across every electrode, as Fig. 2 A); mean and
  SEM by age group, one-sided Mann-Whitney U between groups; one y axis for every panel.
  Electrode counts per participant: Kruskal-Wallis across age groups.
* ``figS_single_subject_predictions``: for the participant with the best electrode of each
  age group, R(phonemes+words) on the cortex and 10 s of measured and held-out predicted
  activity of the three best electrodes.
* ``figS_attention``: phonetic TRF score in the first vs the second half of each audiobook
  segment (first 5 vs last 5 of its 10 contiguous CV splits).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats as sstats

from . import brain, data, style
from .output import Output
from .stats import mannwhitney, p_to_stars, wilcoxon_greater

PAIRS = [(0, 1), (0, 2), (1, 2)]
GROUP_GREYS = dict(zip(data.GROUPS, ["#bdbdbd", "#737373", "#000000"]))
N_TOP = 3


def _pair(a: int, b: int) -> str:
    return f"{data.GROUPS[a]}<{data.GROUPS[b]}"


# ── proportion of significant electrodes ─────────────────────────────────────


def prop_sig_table() -> pd.DataFrame:
    """Per ROI and participant: n electrodes, k significant, ``score`` = 100 k / n."""
    trf = data.trf_linguistic()
    pw = data.trf_scores(trf, data.TRF_LEXICAL)
    sig = data.trf_significant(trf, data.TRF_LEXICAL)
    pw["sig"] = [k in sig for k in zip(pw.participant, pw.electrode)]
    rows = []
    for roi in data.ROIS:
        d = pw if roi == "whole_brain" else pw.merge(data.roi_electrodes(roi), on=["participant", "electrode"])
        m = d.groupby("participant", as_index=False).agg(n=("sig", "size"), k=("sig", "sum"))
        rows.append(m.assign(roi=roi, score=100 * m.k / m.n))
    out = data.add_age(pd.concat(rows, ignore_index=True))
    out["group"] = out.group.astype(str)
    return out


def _prop_bars(ax, d: pd.DataFrame, roi: str) -> dict:
    """Bars of one ROI (no Wilcoxon)."""
    colors = style.ROI_COLORS[roi]
    st = d.groupby("group").score.agg(["mean", "sem"]).reindex(data.GROUPS)
    ax.bar(range(3), st["mean"], yerr=st["sem"], color=colors, width=0.8, error_kw=dict(ecolor="0.25", lw=1))
    ax.axhline(0, color="0.5", lw=0.6)
    lo = min(0, (st["mean"] - st["sem"]).min())
    hi = max(0, (st["mean"] + st["sem"]).max())
    span = hi - lo or 1e-3
    out, k = {}, 0
    for i, j in PAIRS:
        p = mannwhitney(d.score[d.group == data.GROUPS[i]], d.score[d.group == data.GROUPS[j]])
        out[_pair(i, j)] = p
        if s := p_to_stars(p):
            k += 1
            y = hi + span * 0.12 * k
            ax.plot([i, j], [y, y], color="k", lw=1.2)
            ax.text((i + j) / 2, y, s, ha="center", va="bottom", fontsize=9)
    ax.set_ylim(lo - 0.05 * span, hi + span * (0.15 + 0.12 * k))
    ax.set_xticks(range(3), data.GROUPS, fontsize=7)
    for label, c in zip(ax.get_xticklabels(), colors):
        label.set_color(c)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(data.ROI_TITLES[roi], color=colors[1])
    return out


def fig_prop_sig(out_dir: Path) -> None:
    out = Output(out_dir, "figS_prop_sig_electrodes")
    per = prop_sig_table()
    fig, axes = plt.subplots(1, len(data.ROIS), figsize=(15, 2.6), layout="constrained")
    stats: dict[str, dict] = {}
    for ax, roi in zip(axes, data.ROIS):
        d = per[per.roi == roi]
        by = {g: d[d.group == g] for g in data.GROUPS}
        stats[roi] = dict(
            mannwhitney=_prop_bars(ax, d, roi),
            n={g: len(x) for g, x in by.items()},
            mean_pct={g: float(x.score.mean()) for g, x in by.items()},
            pooled={g: [int(x.k.sum()), int(x.n.sum())] for g, x in by.items()},
            median_n_electrodes={g: float(x.n.median()) for g, x in by.items()},
            kruskal_n_electrodes=float(sstats.kruskal(*[x.n for x in by.values()]).pvalue),
        )
    axes[0].set_ylabel("% significant\nelectrodes", fontsize=10)
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(plt.AutoLocator())
        top = max(t for t in ax.get_yticks() if 0 <= t <= min(100, ax.get_ylim()[1]))
        ax.set_yticks([0, top / 2, top])
    out.figure(fig)
    out.source("A_participants", per[["roi", "participant", "age", "group", "n", "k", "score"]]
               .rename(columns={"n": "n_electrodes", "k": "n_significant", "score": "pct_significant"}))
    bars = per.groupby(["roi", "group"], sort=False).score.agg(["mean", "sem", "count"]).reset_index()
    out.source("A_bars", bars.rename(columns={"count": "n"}))
    out.stat("roi_groups", stats)
    out.stat("kruskal_n_electrodes_min", min(s["kruskal_n_electrodes"] for s in stats.values()))
    out.write_stats()


# ── single-participant predictions ───────────────────────────────────────────


def fig_single_subject(out_dir: Path) -> None:
    out = Output(out_dir, "figS_single_subject_predictions")
    trf = data.trf_linguistic()
    pw = data.trf_scores(trf, data.TRF_LEXICAL)
    sig = data.trf_significant(trf, data.TRF_LEXICAL)
    pred = data.trf_predictions()
    roi_of = data.electrodes().set_index(["participant_id", "electrode_id"]).roi
    fig = plt.figure(figsize=(15, 4.5 + 2.9 * N_TOP))
    grid = fig.add_gridspec(1 + N_TOP, 3, height_ratios=[1.6] + [1] * N_TOP, hspace=0.55, wspace=0.15)
    stats, brains, traces = {}, [], []
    for col, group in enumerate(data.GROUPS):
        best = pw[pw.group.astype(str) == group].sort_values("score").participant.iloc[-1]
        mine = pw[pw.participant == best].sort_values("score", ascending=False)
        top = mine.head(N_TOP)
        age = mine.age.iloc[0]  # exact age or age range
        stats[group] = dict(participant=best, age=float(age) if data.has_exact_age() else age,
                            top_r=[round(float(r), 3) for r in top.score])
        ax = fig.add_subplot(grid[0, col])
        shown = brain.plot_brain_values(ax, mine, selected=sig, fs=10, anchor=(-0.05, -0.1))
        for coll in ax.collections:
            coll.set_rasterized(True)
        ax.set_title(f"Best predicted\n{group} yrs participant", fontsize=13)
        brains.append(shown[["participant", "electrode", "y", "z", "score", "selected"]].assign(group=group))
        for row, r in enumerate(top.itertuples(), start=1):
            tr = pred[(pred.participant == best) & (pred.electrode == r.electrode)].sort_values("time")
            assert len(tr) and int(tr["rank"].iloc[0]) == row, (best, r.electrode)
            roi = roi_of.get((best, r.electrode))
            roi = roi if isinstance(roi, str) else "other"
            traces.append(tr.assign(group=group, roi=roi)[
                ["group", "participant", "electrode", "rank", "roi", "r", "time", "measured", "predicted"]])
            ax = fig.add_subplot(grid[row, col])
            ax.plot(tr.time, tr.measured, color="0.45", lw=1, label="Measured")
            ax.plot(tr.time, tr.predicted, color="blue", lw=1.2, label="Prediction")
            ax.set_title(f"Top {row} electrode : {roi}", loc="left", fontsize=11)
            ax.text(0.99, 1.02, rf"{style.RLABEL}={r.score:.2f}", transform=ax.transAxes, ha="right", fontsize=11)
            ax.set_yticks([])
            ax.spines[["top", "right", "left"]].set_visible(False)
            if row == N_TOP:
                ax.set_xlabel("Time (s)")
            else:
                ax.set_xticklabels([])
            if row == 1 and col == 0:
                ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.45), ncol=2)
    out.figure(fig, png=True)
    out.source("A_electrodes", pd.concat(brains, ignore_index=True).rename(
        columns={"score": "r", "selected": "significant"}))
    out.source("B_traces", pd.concat(traces, ignore_index=True))
    out.stat("best_participants", stats)
    out.write_stats()


# ── first vs second half of the segments ─────────────────────────────────────


def fig_attention(out_dir: Path) -> None:
    out = Output(out_dir, "figS_attention")
    sp = data.trf_splits()  # phonetic TRF, mean over electrodes
    keys = ["participant", "segment"]
    sp = sp.assign(rel=sp.split / sp.groupby(keys).split.transform("max"))  # 0 -> 1
    sp["which"] = np.where(sp.rel < 0.5, "first", "second")
    per = (sp.groupby([*keys, "which"]).r.mean()  # over splits
           .groupby(["participant", "which"]).mean().unstack().reset_index())  # over segments
    per = data.add_age(per)
    slopes = data.add_age(sp.groupby("participant")[["rel", "r"]].apply(
        lambda d: np.polyfit(d.rel, d.r, 1)[0]).rename("slope").reset_index())
    stats: dict = {"kruskal_slope_groups": float(sstats.kruskal(
        *[slopes.slope[slopes.group == g] for g in data.GROUPS]).pvalue)}
    fig, ax = plt.subplots(figsize=(4.5, 3))
    bars = []
    for i, g in enumerate(data.GROUPS):
        d = per[per.group == g]
        st: dict = dict(n=len(d))
        for j, w in enumerate(("first", "second")):
            m, se = d[w].mean(), d[w].sem()
            x = i + (j - 0.5) * 0.36
            # The hatch takes the edge colour: white stripes on the black 12+ bars.
            ax.bar(x, m, 0.34, color=GROUP_GREYS[g], edgecolor="w" if g == data.GROUPS[-1] else "k", lw=0,
                   hatch="" if w == "first" else "//")
            ax.bar(x, m, 0.34, yerr=se, fill=False, edgecolor="k", lw=0.5, capsize=2, error_kw=dict(lw=0.8))
            p0 = wilcoxon_greater(d[w].to_numpy())
            st[f"{w}_mean"], st[f"{w}_p"] = float(m), p0
            bars.append(dict(group=g, half=w, mean=m, sem=se, n=len(d), p_wilcoxon=p0))
            ax.text(x, 0, p_to_stars(p0), ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
        st["p_first_vs_second"] = float(sstats.wilcoxon(d["first"], d["second"]).pvalue)
        st["p_slope"] = float(sstats.wilcoxon(slopes.slope[slopes.group == g]).pvalue)
        top = max(d[w].mean() + d[w].sem() for w in ("first", "second"))
        ax.plot([i - 0.18, i + 0.18], [top * 1.08] * 2, color="k", lw=0.8)
        ax.text(i, top * 1.1, f"p={st['p_first_vs_second']:.2f}", ha="center", va="bottom", fontsize=7)
        stats[g] = st
    ax.set_xticks(range(3), data.GROUPS)
    ax.set_ylabel(rf"{style.RLABEL}(phonemes)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend([Patch(facecolor="w", edgecolor="k"), Patch(facecolor="w", edgecolor="k", hatch="//")],
              ["first half", "second half"], frameon=False, fontsize=7, loc="upper left")
    out.figure(fig)
    out.source("A_participants", per.merge(slopes[["participant", "slope"]], on="participant")[
        ["participant", "age", "group", "first", "second", "slope"]])
    out.source("A_bars", pd.DataFrame(bars))
    out.stat("groups", stats)
    out.write_stats()


def make(out: Path) -> None:
    fig_prop_sig(out)
    fig_single_subject(out)
    fig_attention(out)
