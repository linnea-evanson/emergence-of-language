"""Fig. S model layers: linguistic content of the shallow and deep layers, and the
layer-wise rate of change of the brain decoding across age groups.

* ``figS_llm_layers_decode_features``: phonetic, lexical and supralexical features decoded
  from the layer 0.1 and 0.8 embeddings of each model (stimulus decoding).
* ``figS_layerwise_gain_by_age``: A-B, per age group, the
  change in whole-brain decoding score between consecutive layers (Delta R); FDR-corrected
  two-sided Wilcoxon (stars below bars) and Mann-Whitney (brackets) tests; mixed-design
  ANOVA inset, with a label-permutation p of its age group x layer interaction. C,
  inter-subject RSA of the Delta R profiles per ROI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist

from . import data, style
from .figS_stimulus import FAMILY_LEVELS
from .output import Output
from .stats import fdr_bh, p_to_stars

MODELS = ["wav2vec", "llama"]
MODEL_TITLES = {"wav2vec": "wav2vec 2.0", "llama": "Llama 3.1"}
N_LAYERS = 11
#: Delta R transition (index of its first layer) from the last convolutional block to the
#: first transformer layer of wav2vec 2.0 (relative depths 0.3 -> 0.4: of its 20 blocks,
#: 7 convolutional, depth 0.3 is conv block 6 and 0.4 transformer layer 1).
WAV2VEC_CONV_TRANS = 3
#: Label permutations: mixed-ANOVA interaction, inter-subject RSA.
N_PERM_ANOVA, N_PERM_RSA = 2000, 5000
GROUP_LABELS = [f"{g}yrs" for g in data.GROUPS]
STAT_COLOR, SIG_COLOR = "0.15", "red"

# ── layer content (stimulus decoding) ─────────────────────────────────────────

LAYER_LEVELS = ["phonetic", "lexical", "supralexical"]
SERIES_COLORS = {
    ("0.1", "phonetic"): "#9ecae1", ("0.8", "phonetic"): "#08519c",
    ("0.1", "lexical"): "#fcbba1", ("0.8", "lexical"): "#a50f15",
    ("0.1", "supralexical"): "#bcbddc", ("0.8", "supralexical"): "#54278f",
}
FAMILY_TITLES = {"llama": "Llama 3.1", "wav2vec": "Wav2Vec 2.0"}


def layer_content() -> pd.DataFrame:
    """Per model, layer (0.1, 0.8), level and time: decoding score averaged over the level's
    target features (each feature first averaged over its dimensions)."""
    stim = data.stimulus_decoding()
    level = {f: lev for lev, fams in FAMILY_LEVELS.items() for f in fams}
    df = stim.assign(level=stim.target_group.map(level))
    parts = df.source.str.extract(r"^(?P<model>wav2vec|llama)_L(?P<layer>[0-9.]+)$")
    df = df.assign(model=parts.model, layer=parts.layer)
    df = df[df.model.notna() & df.level.isin(LAYER_LEVELS)]
    return df.groupby(["model", "layer", "level", "time"], as_index=False).score.mean()


def llm_layers(out: Path) -> None:
    o = Output(out, "figS_llm_layers_decode_features")
    df = layer_content()
    ymax = max(float(df.score.max()), 0.06)
    fig, axes = plt.subplots(2, 1, figsize=(4.5, 4.8), sharex=True, sharey=True,
                             constrained_layout=True)
    for i, (ax, model) in enumerate(zip(axes, ["llama", "wav2vec"])):
        for level in LAYER_LEVELS:
            for layer in ("0.1", "0.8"):
                c = df.query("model == @model and layer == @layer and level == @level")
                ax.plot(c.time, c.score, color=SERIES_COLORS[(layer, level)])
        if i == 0:
            handles = [plt.Line2D([0], [0], color=SERIES_COLORS[(layer, level)], lw=2,
                                  label=f"L{layer[-1]} {data.LEVEL_TITLES[level]}")
                       for level in LAYER_LEVELS for layer in ("0.1", "0.8")]
            ax.legend(handles=handles, frameon=False, loc="upper right", handlelength=1.2,
                      labelspacing=0.15, borderpad=0, borderaxespad=0)
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.65)
        ax.set_title(FAMILY_TITLES[model])
        style.minimal_yaxis(ax, ymax=ymax)
        ax.set_ylim(top=ymax * 1.05)
    axes[-1].set_xlabel("Time relative to word/phoneme onset (s)")
    o.figure(fig)
    o.source("curves", df)
    o.stat("peak", {f"{m}_L{lay}_{lev}": dict(score=float(g.score.max()),
                                                time=float(g.time[g.score.idxmax()]))
                    for (m, lay, lev), g in df.groupby(["model", "layer", "level"])})
    o.write_stats()


# ── layer-wise rate of change ─────────────────────────────────────────────────


def layer_scores(roi: str) -> pd.DataFrame:
    """Per participant, model and layer (0-10): decoding score averaged over
    [AVG_TMIN, AVG_TMAX] (``avg_score``), with age and group."""
    df = data.decoding_layerwise().query("roi == @roi")
    df = df[(df.time >= style.AVG_TMIN) & (df.time <= style.AVG_TMAX)]
    out = df.groupby(["participant", "model", "depth"], as_index=False).score.mean()
    out["layer"] = (out.depth * 10).round().astype(int)
    assert (out.groupby(["model", "participant"]).layer.nunique() == N_LAYERS).all()
    return data.add_age(out.rename(columns={"score": "avg_score"}))


def delta_r(sl: pd.DataFrame) -> pd.DataFrame:
    """Delta R between consecutive layers; ``layer`` 0-9 indexes the transition L -> L+1."""
    sl = sl.sort_values(["participant", "model", "layer"])
    sl = sl.assign(delta_r=sl.groupby(["participant", "model"]).avg_score.diff())
    out = sl.dropna(subset=["delta_r"]).copy()
    out["layer"] -= 1
    return out


def mixed_anova(m: pd.DataFrame) -> dict:
    """Mixed-design ANOVA of ``delta_r``: group between participants, layer within
    (participants as the error term of the group effect), Greenhouse-Geisser corrected."""
    y = m.delta_r.to_numpy()
    gm = y.mean()
    L = m.layer.nunique()
    subj = m.groupby("participant").agg(group=("group", "first"), mean=("delta_r", "mean"))
    grp = m.groupby("group").delta_r.mean()
    ng = subj.group.value_counts()
    lay = m.groupby("layer").delta_r.mean()
    cell = m.groupby(["group", "layer"]).delta_r.mean()
    ss_g = sum(ng[g] * L * (grp[g] - gm) ** 2 for g in grp.index)
    ss_s = L * ((subj["mean"] - subj.group.map(grp)) ** 2).sum()
    ss_l = m.participant.nunique() * ((lay - gm) ** 2).sum()
    ss_gl = sum(ng[g] * (cell[(g, la)] - grp[g] - lay[la] + gm) ** 2 for g in grp.index for la in lay.index)
    ss_e = ((y - gm) ** 2).sum() - ss_g - ss_s - ss_l - ss_gl
    G, N = len(grp), m.participant.nunique()
    df_g, df_s, df_l, df_gl, df_e = G - 1, N - G, L - 1, (G - 1) * (L - 1), (N - G) * (L - 1)
    Fg = (ss_g / df_g) / (ss_s / df_s)
    Fl = (ss_l / df_l) / (ss_e / df_e)
    Fgl = (ss_gl / df_gl) / (ss_e / df_e)
    # Greenhouse-Geisser epsilon of the within factor (pooled within-group covariance)
    w = m.pivot(index="participant", columns="layer", values="delta_r")
    resid = w.sub(w.groupby(subj.group).transform("mean"))
    S = np.cov(resid.to_numpy().T, ddof=G)
    C = np.eye(L) - 1 / L
    Sc = C @ S @ C
    eps = np.trace(Sc) ** 2 / ((L - 1) * np.trace(Sc @ Sc))
    return dict(group=dict(F=Fg, df=(df_g, df_s), p=stats.f.sf(Fg, df_g, df_s)),
                layer=dict(F=Fl, df=(df_l, df_e), p_GG=stats.f.sf(Fl, df_l * eps, df_e * eps)),
                interaction=dict(F=Fgl, df=(df_gl, df_e), p=stats.f.sf(Fgl, df_gl, df_e),
                                 p_GG=stats.f.sf(Fgl, df_gl * eps, df_e * eps)),
                GG_epsilon=eps)


def mixed_anova_perm(m: pd.DataFrame, n_perm: int = N_PERM_ANOVA, seed: int = 0) -> dict:
    """:func:`mixed_anova`, with the interaction p also by permuting the participants' groups."""
    res = mixed_anova(m)
    rng = np.random.default_rng(seed)
    sg = m.groupby("participant").group.first()
    null = [mixed_anova(m.assign(group=m.participant.map(
        pd.Series(rng.permutation(sg.to_numpy()), index=sg.index))))["interaction"]["F"]
        for _ in range(n_perm)]
    res["interaction"]["p_perm"] = float((np.sum(np.array(null) >= res["interaction"]["F"]) + 1) / (n_perm + 1))
    return res


def layer_tests(m: pd.DataFrame) -> pd.DataFrame:
    """Per transition: two-sided Wilcoxon against 0 of each group and Mann-Whitney U between
    groups, BH-FDR across the 10 transitions of each group / pair."""
    rows = []
    for layer, d in m.groupby("layer"):
        vals = {g: d.delta_r[d.group.astype(str) == g].to_numpy() for g in data.GROUPS}
        for g in data.GROUPS:
            rows.append(dict(layer=layer, test="wilcoxon", groups=g,
                             p=float(stats.wilcoxon(vals[g], alternative="two-sided").pvalue)))
        for i, g1 in enumerate(data.GROUPS):
            for g2 in data.GROUPS[i + 1:]:
                rows.append(dict(layer=layer, test="mannwhitney", groups=f"{g1} vs {g2}",
                                 p=float(stats.mannwhitneyu(vals[g1], vals[g2], alternative="two-sided").pvalue)))
    df = pd.DataFrame(rows)
    df["q"] = df.groupby(["test", "groups"]).p.transform(lambda p: fdr_bh(p.to_numpy()))
    return df


def mantel(brain: np.ndarray, age: np.ndarray, rng: np.random.Generator,
           n_perm: int = N_PERM_RSA) -> tuple[float, float]:
    """Spearman correlation between the brain dissimilarities and |Delta log age| of every
    pair of participants; two-sided p by permuting the ages."""
    def unit_ranks(v: np.ndarray) -> np.ndarray:
        v = stats.rankdata(v)
        v = v - v.mean()
        return v / np.linalg.norm(v)

    b = unit_ranks(brain)
    log_age = np.log10(age)[:, None]
    obs = float(b @ unit_ranks(pdist(log_age, "cityblock")))
    null = np.array([b @ unit_ranks(pdist(rng.permutation(log_age), "cityblock"))
                     for _ in range(n_perm)])
    return obs, float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def isrsa(seed: int = 0) -> pd.DataFrame:
    """Per ROI and model: Mantel r and p of the Euclidean distance between the participants'
    Delta R profiles, each divided by its mean |Delta R|."""
    rng = np.random.default_rng(seed)
    rows = []
    for roi in data.ROIS:
        d = delta_r(layer_scores(roi))
        for model in MODELS:
            w = d[d.model == model].pivot_table(index=["participant", "age"], columns="layer",
                                                values="delta_r").sort_index(level="age")
            x = w.to_numpy()
            brain = pdist(x / np.abs(x).mean(1, keepdims=True), "euclidean")
            r, p = mantel(brain, w.index.get_level_values("age").to_numpy(float), rng)
            rows.append(dict(roi=roi, model=model, n=len(w), r=r, p=p))
    return pd.DataFrame(rows)


def _stats_box(ax, res: dict, x: float = 0.97, y0: float = 0.82, dy: float = 0.045,
               width: float = 0.2, fontsize: float = 8) -> None:
    items = [("Mixed ANOVA", STAT_COLOR)]
    for name, key, p_key in [("age", "group", "p"), ("layer", "layer", "p_GG"),
                             ("age × layer", "interaction", "p_GG")]:
        p = res[key][p_key]
        items.append((f"{name}: p{'<0.001' if p < 0.001 else f'={p:.3f}'}",
                      SIG_COLOR if p <= 0.05 else STAT_COLOR))
    height = len(items) * dy + 0.008
    ax.add_patch(plt.Rectangle((x - width, y0 - height), width, height, transform=ax.transAxes,
                               facecolor="white", alpha=0.92, edgecolor="none", zorder=4,
                               clip_on=False))
    for i, (line, color) in enumerate(items):
        ax.text(x, y0 - i * dy, line, transform=ax.transAxes, ha="right", va="top",
                fontsize=fontsize, color=color, zorder=5, clip_on=False)


def _clean_y_axis(ax, pad_frac: float = 0.06) -> None:
    """Ticks at the data min, 0 and max only ."""
    ymin, ymax = ax.get_ylim()
    ticks = sorted({ymin, ymax, 0.0} if ymin <= 0 <= ymax else {ymin, ymax})
    span = ymax - ymin
    precision = int(np.clip(np.ceil(-np.log10(span / 2)) + 1, 1, 3))
    ax.set_ylim(ymin - span * pad_frac, ymax + span * pad_frac)
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: style.format_tick(v, precision)))
    ax.tick_params(axis="y", direction="in", length=6, width=1.0, color="0.15")


def draw_bars(ax, m: pd.DataFrame, model: str, tests: pd.DataFrame, anova: dict,
              legend: bool) -> None:
    colors = style.palette("whole_brain")
    sns.barplot(data=m.assign(group=m.group.astype(str) + "yrs"), x="layer", y="delta_r",
                hue="group", hue_order=GROUP_LABELS, palette=colors, errorbar="se", capsize=0.08,
                err_kws={"linewidth": 1.0}, ax=ax)
    q = tests.set_index(["layer", "test", "groups"]).q
    layers = sorted(m.layer.unique())
    # stars below the bars (Wilcoxon), stacked downwards
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.03
    stars_drawn = []
    for li, layer in enumerate(layers):
        for gi, cont in enumerate(ax.containers):
            patch = cont.patches[li]  # type: ignore[attr-defined]
            stars = p_to_stars(q[(layer, "wilcoxon", data.GROUPS[gi])])
            x = patch.get_x() + patch.get_width() / 2
            y = min(patch.get_y(), patch.get_y() + patch.get_height()) - offset
            for k, star in enumerate(stars):
                trans = mtransforms.offset_copy(ax.transData, fig=ax.figure, x=0, y=-k * 9 * 0.68,
                                                units="points")
                stars_drawn.append(ax.text(x, y, star, transform=trans, ha="center", va="top",
                                           fontsize=9, fontweight="bold", color="0.15",
                                           clip_on=False, zorder=7))
    if stars_drawn:
        renderer = ax.figure.canvas.get_renderer()  # type: ignore[attr-defined]
        low = min(a.get_window_extent(renderer=renderer).transformed(ax.transData.inverted()).y0
                  for a in stars_drawn)
        ax.set_ylim(min(low, ymin), ymax)
    # brackets above the bars (Mann-Whitney)
    ymin, ymax = ax.get_ylim()
    step = (ymax - ymin) * 0.07
    top_needed = ymax
    for li, layer in enumerate(layers):
        d = m[m.layer == layer]
        xs, tops = [], []
        for gi, cont in enumerate(ax.containers):
            patch = cont.patches[li]  # type: ignore[attr-defined]
            v = d.delta_r[d.group.astype(str) == data.GROUPS[gi]]
            xs.append(patch.get_x() + patch.get_width() / 2)
            tops.append(max(v.mean() + v.sem(), patch.get_y() + patch.get_height()))
        level = 0
        for i in range(3):
            for j in range(i + 1, 3):
                stars = p_to_stars(q[(layer, "mannwhitney", f"{data.GROUPS[i]} vs {data.GROUPS[j]}")])
                if stars:
                    level += 1
                    y = max(tops[i], tops[j]) + step * level
                    top_needed = max(top_needed, y + (ymax - ymin) * 0.015 + step * 0.25)
                    style.draw_bracket(ax, xs[i], xs[j], y, stars)
    if top_needed > ymax:
        ax.set_ylim(ymin, top_needed + step * 0.4)
    _stats_box(ax, anova)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.45)
    if model == "wav2vec":
        t = WAV2VEC_CONV_TRANS
        ax.axvspan(t - 0.5, t + 0.5, color="0.92", zorder=0, lw=0)
        ax.text(t, 0.99, "conv → trans", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=8.5, color="0.35")
    ax.set_xlabel("Relative layer")
    ax.set_ylabel(r"$\Delta R$", rotation=0, labelpad=10)
    ax.set_xticks(range(10), [style.format_tick((i + 1) / 10, 1) for i in range(10)])
    if legend:
        ax.legend(loc="upper left", frameon=False, title=None)
    elif (leg := ax.get_legend()) is not None:
        leg.remove()
    sns.despine(ax=ax)
    _clean_y_axis(ax)


def draw_isrsa(ax, rsa: pd.DataFrame, width: float = 0.38) -> None:
    """Mantel r per ROI; lighter shade of the ROI palette: wav2vec 2.0, darker: Llama 3.1."""
    x = np.arange(len(data.ROIS), dtype=float)
    x[1:] += 0.5
    for k, model in enumerate(MODELS):
        r = rsa[rsa.model == model].set_index("roi").loc[data.ROIS]
        for xi, (roi, row) in zip(x, r.iterrows()):
            xb = xi + (k - 0.5) * width
            ax.bar(xb, row.r, width=width, color=style.ROI_COLORS[roi][k], edgecolor="white",
                   linewidth=0.6)
            if row.p < 0.05:
                ax.text(xb, max(row.r, 0) + 0.004, p_to_stars(row.p), ha="center",
                        va="bottom", fontsize=12)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x, [data.ROI_TITLES[r] for r in data.ROIS], fontsize=11)
    for lab, roi in zip(ax.get_xticklabels(), data.ROIS):
        lab.set_color(style.ROI_COLORS[roi][1])
        lab.set_fontweight("semibold")
    ax.set_ylim(min(0, rsa.r.min()) - 0.01, max(0.25, rsa.r.max()) + 0.03)
    ax.set_yticks([0, 0.25], ["0", "0.25"], fontsize=11)
    ax.set_ylabel("Spearman r\n(ΔR profile ~ age)", fontsize=11)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=style.ROI_COLORS["whole_brain"][k],
                                     edgecolor="white", label=MODEL_TITLES[m])
                       for k, m in enumerate(MODELS)], frameon=False, fontsize=10, loc="upper right")
    sns.despine(ax=ax)


def _letter(ax, letter: str) -> None:
    ax.text(-0.09, 1.04, letter, transform=ax.transAxes, fontsize=20, fontweight="bold",
            ha="right", va="bottom")


def layerwise_gain(out: Path) -> None:
    o = Output(out, "figS_layerwise_gain_by_age")
    sl = layer_scores("whole_brain")
    delta = delta_r(sl)
    delta["group"] = delta.group.astype(str)
    # C (inter-subject RSA) needs exact ages: without them, stats of A-B only, no figure.
    draw = not data.skip_without_exact_age("the Fig. S7 image and its panel C (inter-subject RSA)")
    rsa = isrsa() if draw else None
    o.stat("n_participants", int(delta.participant.nunique()))

    fig = plt.figure(figsize=(10, 6.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.9])
    for j, model in enumerate(MODELS):
        m = delta[delta.model == model]
        anova = mixed_anova_perm(m)
        tests = layer_tests(m)
        o.stat(f"mixed_anova_{model}", anova)
        o.stat(f"layer_tests_{model}", tests.to_dict(orient="records"))
        ax = fig.add_subplot(gs[0, j])
        draw_bars(ax, m, model, tests, anova, legend=j == 0)
        ax.set_title(f"{MODEL_TITLES[model]} (Whole brain)", fontsize=12)
        _letter(ax, "AB"[j])
    if rsa is None:
        plt.close(fig)
    else:
        ax = fig.add_subplot(gs[1, :].subgridspec(1, 3, width_ratios=[0.15, 0.7, 0.15])[0, 1])
        draw_isrsa(ax, rsa)
        ax.set_title("Inter-subject RSA", fontsize=12)
        _letter(ax, "C")
        o.figure(fig)
        o.stat("isrsa", rsa.to_dict(orient="records"))
        o.source("C_isrsa", rsa)

    summary = delta.groupby(["model", "layer", "group"]).delta_r.agg(["mean", "sem", "size"])
    o.source("AB_delta_r_per_participant",
             delta[["participant", "age", "group", "model", "layer", "delta_r"]].assign(
                 transition=lambda d: [f"{la / 10:g}->{(la + 1) / 10:g}" for la in d.layer]))
    o.source("AB_delta_r_group_mean_sem", summary.reset_index())
    o.write_stats()


def make(out: Path) -> None:
    llm_layers(out)
    layerwise_gain(out)
