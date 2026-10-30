"""Fig. 5 (``fig5_1``): encoding and decoding of the wav2vec 2.0 and Llama 3.1 embeddings.

* A/E: model schematic.
* B/F: per-participant mean encoding score over all electrodes (left) and over the STG
  electrodes (right) of the random shallow layer (``Rand. 0.1``), the trained shallow layer
  (``0.1``) and both layers (``0.1+0.8``); within each age group, one-sided Wilcoxon against 0
  (stars) and paired one-sided Wilcoxon between embeddings (brackets).
* C/G: R(Rand. 0.1), training gain ΔT R = R(0.1) - R(Rand. 0.1) and hierarchical gain
  ΔL R = R(0.1+0.8) - R(0.1) per located electrode (coloured: permutation p, BH-FDR over every
  electrode, and positive gain).
* D/H: whole-brain decoding time course of each embedding (mean over participants), with the
  half-max width of each mean curve and cluster-permutation significance dots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

from . import brain, data, style
from .output import Output
from .stats import mannwhitney, p_to_stars, permute_cluster, sig_to_plot, wilcoxon_greater

MODELS = {"wav2vec": "wav2vec 2.0", "llama": "Llama 3.1"}
#: Best-decoded layer of each model (Fig. 6, ROI decoding, training gain).
BEST = {"wav2vec": "0.8", "llama": "0.1"}
#: Encoding embeddings of panels B/F, with their bar colours.
ORDER = ["Rand. 0.1", "0.1", "0.1+0.8"]
COLORS = ["k", "blue", "red"]
#: Decoded embeddings of panels D/H.
DEC_ORDER = ["Rand. 0.1", "0.1", "0.8"]
#: ROI of the right bars of B/F and of the Fig. 6 training gain.
GAIN_ROI = "STG"


def feature(model: str, layer: str) -> str:
    """Release name of an embedding: ``Rand. 0.1`` -> ``wav2vec_L0.1_untrained``."""
    if layer.startswith("Rand. "):
        return f"{model}_L{layer.removeprefix('Rand. ')}_untrained"
    return f"{model}_L{layer}"


def group_pairs(df: pd.DataFrame, col: str = "score") -> dict[str, float]:
    """One-sided Mann-Whitney U (younger < older) between every pair of age groups."""
    vals = {g: df.loc[df.group.astype(str) == g, col].to_numpy() for g in data.GROUPS}
    return {f"{a}<{b}": mannwhitney(vals[a], vals[b])
            for i, a in enumerate(data.GROUPS) for b in data.GROUPS[i + 1:]}


def intervals(times: np.ndarray, mask: np.ndarray) -> list[list[float]]:
    """[start, end] times of the runs of True in ``mask``."""
    edges = np.diff(np.concatenate([[0], np.asarray(mask, int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return [[float(times[a]), float(times[b])] for a, b in zip(starts, ends)]


def speech_waveform(duration: float = 1.4, sfreq: int = 8000, seed: int = 0) -> np.ndarray:
    """Synthetic speech-like waveform (noise under a syllabic envelope) for the schematic.

    The audiobook itself is not part of the release.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sfreq)) / sfreq
    env = np.zeros_like(t)
    for onset, length, amp in [(0.05, 0.18, 0.5), (0.27, 0.12, 0.9), (0.42, 0.22, 1.0),
                               (0.7, 0.1, 0.6), (0.85, 0.2, 0.8), (1.12, 0.16, 0.45)]:
        inside = (t >= onset) & (t < onset + length)
        env[inside] = amp * np.sin(np.pi * (t[inside] - onset) / length) ** 2
    carrier = rng.standard_normal(len(t)) * (0.6 + 0.4 * np.sin(2 * np.pi * 140 * t))
    return env * carrier + 0.01 * rng.standard_normal(len(t))


def model_schematic(ax, model: str, fs: float = 10) -> None:
    """Model box fed with audio (wav2vec 2.0) or text (Llama 3.1), then PCA."""
    name = MODELS[model]
    h, w, pad, x = 1.0, 0.3, 0.05, 0.3
    bottom = h + 0.5
    def box(x0: float, y0: float, width: float, height: float) -> None:
        ax.add_patch(FancyBboxPatch((x0, y0), width, height, boxstyle=f"round,pad={pad}",
                                    facecolor="white", edgecolor="black", linewidth=1))

    box(x, bottom, w, h)
    ax.set_xlim(0.8 * x, x + w + 0.2 * x)
    ax.set_ylim(0, bottom + 2.5 * h)
    arrow = dict(arrowstyle="->", color="black", linewidth=3)
    ax.annotate("", xy=(x + w / 2, bottom - 0.01 * h), xytext=(x + w / 2, bottom - h / 2),
                arrowprops=arrow)
    if model == "wav2vec":
        inset = ax.inset_axes((0.25, bottom - h, 0.4, 0.5), transform=ax.transData)
        inset.plot(speech_waveform(), color="k", linewidth=0.5)
        inset.set_axis_off()
    else:
        ax.text(x + w / 2, bottom - h * 1.1 / 2, "Lorsque j'avais six ans", ha="center", va="top")
    ax.text(x + w / 2, bottom + h / 2, name, ha="center", va="center", fontsize=fs)
    ax.annotate("", xy=(x + w / 2, bottom + 1.6 * h), xytext=(x + w / 2, bottom + 1.1 * h),
                arrowprops=arrow)
    box(x + w / 4, bottom + 1.6 * h, w / 2, h / 2)
    ax.text(x + w / 2, bottom + 1.7 * h, "PCA$_{50}$", ha="center", va="bottom", fontsize=fs)
    ax.set_axis_off()


def _rounded_ticks(ax, lo: float, hi: float) -> None:
    """Only 0 and the top tick labelled, without the leading 0."""
    ticks = [t for t in ax.get_yticks() if lo <= t <= hi]
    ax.set_yticks(ticks, labels=[str(np.round(t, 3)).replace("0.", ".")
                                 if t in (0, max(ticks)) else "" for t in ticks])


def layer_bars(ax, trf: pd.DataFrame, model: str, roi: str, legend: bool) -> tuple[pd.DataFrame, dict]:
    """B/F: mean R of each embedding per age group; stars and brackets as in the module doc."""
    df = pd.concat([data.roi_means(data.trf_scores(trf, feature(model, layer)), [roi])
                    .assign(layer=layer) for layer in ORDER], ignore_index=True)
    df["group"] = df.group.astype(str)
    sns.barplot(df, x="group", y="score", hue="layer", order=data.GROUPS, hue_order=ORDER,
                palette=COLORS, errorbar="se", ax=ax)
    ax.set_ylabel(style.RLABEL if roi == "whole_brain" else None, rotation=0)
    ax.set_xlabel(None)
    ax.set_xticks(range(3), data.GROUPS)
    ax.set_title(data.ROI_TITLES[roi], fontsize=14, color="k")
    if legend:
        ax.legend(frameon=False, handlelength=1, labelspacing=0, loc="upper left")
    elif (leg := ax.get_legend()) is not None:
        leg.remove()
    sns.despine(ax=ax)
    offset = 0.8 / 3
    st: dict = {"n": {g: int((df[df.layer == ORDER[0]].group == g).sum()) for g in data.GROUPS},
                "mean": {}, "wilcoxon": {}, "paired": {}}
    rows = []
    for xpos, g in enumerate(data.GROUPS):
        sub = df[df.group == g].pivot(index="participant", columns="layer", values="score")
        for i, layer in enumerate(ORDER):
            p = wilcoxon_greater(sub[layer].to_numpy())
            st["mean"][f"{g}:{layer}"] = float(sub[layer].mean())
            st["wilcoxon"][f"{g}:{layer}"] = p
            if stars := p_to_stars(p):
                ax.text(xpos + (i - 1) * offset, 0, stars, ha="center", va="top",
                        fontweight="bold", fontsize=16)
        for i, f1 in enumerate(ORDER):
            for f2 in ORDER[i + 1:]:
                p = wilcoxon_greater((sub[f1] - sub[f2]).to_numpy(), alternative="less")
                st["paired"][f"{g}:{f1}<{f2}"] = p
                rows.append((g, f1, f2, p))
    tops = df.groupby(["group", "layer"]).score.agg(["mean", "sem"]).sum(axis=1)
    step = 0.08 * tops.max()
    y_max = tops.max()
    for g in data.GROUPS:
        sig = [r for r in rows if r[0] == g and r[3] < 0.05]
        for j, (_, f1, f2, p) in enumerate(sig):
            xpos = data.GROUPS.index(g)
            xmin = xpos - offset if f1 == ORDER[0] else xpos
            xmax = xpos + offset if f2 == ORDER[2] else xpos
            y = tops[g].max() + step * (j + 0.6)
            ax.hlines(xmin=xmin, xmax=xmax, y=y, color="k", lw=2)
            ax.text((xmin + xmax) / 2, y, p_to_stars(p), ha="center", fontweight="bold")
            y_max = max(y_max, y)
    top = y_max + step
    ax.set_ylim(-0.1 * top, top)
    _rounded_ticks(ax, 0, top)
    return df[["participant", "age", "group", "layer", "score"]], st


#: Brain panels C/G: (embedding, baseline, legend title, colormap, vmax, legend anchor).
BRAINS = [
    ("Rand. 0.1", None, r"$\mathcal{R}_{Rand. 0.1}$", "Greys_r", 0.3, (-0.15, -0.15)),
    ("0.1", "Rand. 0.1", r"$\Delta_T \mathcal{R}$", "Blues_r", 0.1, (-0.1, -0.15)),
    ("0.1+0.8", "0.1", r"$\Delta_L\mathcal{R}$", "Reds_r", 0.1, (-0.1, -0.15)),
]


def model_brains(axs, trf: pd.DataFrame, model: str, out: Output, letter: str) -> None:
    for ax, (layer, base, title, cmap, vmax, anchor) in zip(axs, BRAINS):
        feat, bfeat = feature(model, layer), base and feature(model, base)
        values = data.trf_scores(trf, feat, bfeat)
        drawn = brain.plot_brain_values(ax, values, data.trf_significant(trf, feat, bfeat),
                                        vmax=vmax, cmap=cmap, title=title, fs=9, anchor=anchor)
        for coll in [*ax.collections, *[c for a in ax.child_axes for c in a.collections]]:
            coll.set_rasterized(True)
        name = "R_rand01" if base is None else "deltaT" if base == "Rand. 0.1" else "deltaL"
        out.source(f"{letter}_{model}_{name}", drawn.rename(columns={"selected": "significant"})
                   [["participant", "electrode", "y", "z", "score", "significant"]])


def half_max_window(times: np.ndarray, curve: np.ndarray) -> tuple[float, float]:
    """First time steps around the peak where the mean curve is below half its max."""
    i = int(np.argmax(curve))
    below = curve < curve[i] / 2
    before = np.flatnonzero(below[:i])
    after = np.flatnonzero(below[i:])
    rise = times[before[-1]] if len(before) else times[0]
    fall = times[i + after[0]] if len(after) else times[-1]
    return float(rise), float(fall)


def decoding_curves(ax, model: str) -> tuple[pd.DataFrame, dict]:
    """D/H: whole-brain decoding of each embedding over time (mean over participants)."""
    dec = data.decoding_embeddings()
    dec = dec[dec.roi == "whole_brain"]
    df = pd.concat([dec[dec.embedding == feature(model, layer)].assign(layer=layer)
                    for layer in DEC_ORDER], ignore_index=True)
    sns.lineplot(df, x="time", y="score", hue="layer", hue_order=DEC_ORDER, palette=COLORS,
                 errorbar=None, ax=ax)
    times = np.sort(df.time.unique())
    means = df.groupby(["layer", "time"]).score.mean()
    ymax = means.max()
    a, c = 0.17 * ymax, 0.035 * ymax
    st: dict = {}
    curves = []
    for j, (layer, color) in enumerate(zip(DEC_ORDER[::-1], COLORS[::-1])):
        sub = df[df.layer == layer]
        mean = means[layer]
        rise, fall = half_max_window(mean.index.to_numpy(), mean.to_numpy())
        win = sub[sub.time.between(style.AVG_TMIN, style.AVG_TMAX)].groupby("participant").score.mean()
        sig = permute_cluster(sub)
        st[layer] = dict(half_max_rise=rise, half_max_fall=fall, half_max_width=fall - rise,
                         peak_time=float(mean.idxmax()), n=int(sub.participant.nunique()),
                         mean_R_window=float(win.mean()), wilcoxon_window=wilcoxon_greater(win.to_numpy()),
                         significant_times=intervals(times, sig))
        curves.append(pd.DataFrame(dict(layer=layer, time=mean.index, mean_score=mean.to_numpy(),
                                        significant=sig)))
        if (dots := sig_to_plot(sig)) is not None:
            ax.scatter(times, -(dots * a) + j * c, color=color, s=1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title=None, frameon=False, labelcolor=COLORS[::-1],
              handlelength=1, labelspacing=0)
    sns.despine(ax=ax)
    ax.set_ylabel(style.RLABEL, rotation=0)
    ax.set_xlabel("Time (s)")
    lo, hi = -a - c, ymax * 1.1
    ax.set_ylim(lo, hi)
    ticks = [t for t in ax.get_yticks() if lo <= t <= hi]
    ax.set_yticks(ticks, labels=[np.round(t, 3) if t in (0, max(ticks)) else "" for t in ticks])
    ax.axhline(0, color="lightgrey", linestyle="--", zorder=-1)
    return pd.concat(curves, ignore_index=True), st


def gain_stats(trf: pd.DataFrame, model: str) -> dict:
    """Text: whole-brain ΔT R (0.1 and 0.8) and ΔL R, per-participant mean over all electrodes,
    one-sided Wilcoxon > 0 across participants; ΔL R in the STG between age groups."""
    out = {}
    for name, (a, b) in {"delta_T": ("0.1", "Rand. 0.1"), "delta_T_08": ("0.8", "Rand. 0.8"),
                         "delta_L": ("0.1+0.8", "0.1")}.items():
        v = data.roi_means(data.trf_scores(trf, feature(model, a), feature(model, b)), ["whole_brain"])
        out[name] = dict(mean=float(v.score.mean()), p=wilcoxon_greater(v.score.to_numpy()),
                         n=int(len(v)))
    dl = data.roi_means(data.trf_scores(trf, feature(model, "0.1+0.8"), feature(model, "0.1")),
                        [GAIN_ROI])
    out[f"delta_L_{GAIN_ROI}_age_mannwhitney"] = group_pairs(dl)
    return out


def make(out_dir: Path) -> None:
    out = Output(out_dir, "fig5_1")
    trf = data.trf_embeddings()
    fig, axs = plt.subplot_mosaic(
        [["A", "B", "B2", "C", "D", "E", "F"], ["G", "H", "H2", "I", "J", "K", "L"]],
        figsize=(20.5, 6), layout="constrained", width_ratios=[0.5, 0.75, 0.75, 1, 1, 1, 1])
    # model, paper letters of its bars / brains / curves, mosaic keys of its panels
    for model, (B, C, D), row in [
            ("wav2vec", ("B", "C", "D"), dict(A="A", B="B", C="CDE", D="F")),
            ("llama", ("F", "G", "H"), dict(A="G", B="H", C="IJK", D="L"))]:
        model_schematic(axs[row["A"]], model)
        st: dict = {}
        for roi, key in [("whole_brain", row["B"]), (GAIN_ROI, row["B"] + "2")]:
            bars, st[roi] = layer_bars(axs[key], trf, model, roi,
                                       legend=model == "wav2vec" and roi == "whole_brain")
            out.source(f"{B}_{model}_{roi}", bars)
        out.stat(f"{B}_{model}_encoding_bars", st)
        model_brains([axs[k] for k in row["C"]], trf, model, out, C)
        curves, dst = decoding_curves(axs[row["D"]], model)
        out.source(f"{D}_{model}_decoding", curves)
        out.stat(f"{D}_{model}_decoding", dst)
        out.stat(f"text_{model}_gains", gain_stats(trf, model))
    fig.text(0.41, 1, "Random\n" + r"$\mathcal{R}_{Rand. 0.1}$", fontsize=16)
    fig.text(0.535, 1, "     Training gain\n"
             + r"$\Delta_T \mathcal{R} = \mathcal{R}_{0.1} - \mathcal{R}_{Rand. 0.1}$", fontsize=16)
    fig.text(0.695, 1, "  Hierarchical gain\n"
             + r"$\Delta_L \mathcal{R} = \mathcal{R}_{0.8+0.1} - \mathcal{R}_{0.1}$", fontsize=16)
    for x, y, letter in [(0, 1.04, "A"), (0.08, 1.04, "B"), (0.335, 1.04, "C"), (0.815, 1.04, "D"),
                         (0, 0.48, "E"), (0.08, 0.48, "F"), (0.335, 0.48, "G"), (0.815, 0.48, "H")]:
        fig.text(x, y, letter, fontsize=16, fontweight="bold")
    out.figure(fig, png=True)
    out.write_stats()
