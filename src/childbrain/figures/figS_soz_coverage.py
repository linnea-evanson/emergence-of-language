"""Fig. S seizure lateralization and hemispheric coverage across age groups.

The 45 participants with a seizure onset zone (SOZ) ranking (``participants.tsv``
``has_soz_ranking``).
A: electrodes of the left- and right-lateralized participants on each hemisphere;
B: electrodes per participant and hemisphere (participants with >= 1 electrode there),
two-sided Mann-Whitney U; C: lateralized participants per age group, chi-square with a
label-permutation p. With the full participants table, also the SOZ lobe x age test
quoted in the Discussion.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from . import data, style
from .brain import brain_background
from .output import Output

HEMIS = ["left", "right"]
#: Label permutations of each chi-square test.
N_PERM = 10_000


def _table(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Contingency table of two integer-coded label arrays."""
    t = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(t, (a, b), 1)
    return t


def chi2_permutation(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                     n_perm: int = N_PERM) -> dict:
    """Chi-square test of independence of the labels ``a`` and ``b``; p also by permuting ``b``.
    The tests of the figure share one generator, in a fixed order."""
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    chi2, p, dof, _ = stats.chi2_contingency(_table(a, b), correction=False)
    null = np.array([stats.chi2_contingency(_table(a, rng.permutation(b)), correction=False)[0]
                     for _ in range(n_perm)])
    return dict(chi2=float(chi2), dof=int(dof), p_asymptotic=float(p),
                p_permutation=float((np.sum(null >= chi2 - 1e-12) + 1) / (n_perm + 1)))


#: Seizure onset zone details of the full participants table.
SOZ_COLUMNS = ["soz_hemisphere", "soz_lobe"]


def coverage() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Localised electrodes of the SOZ participants (with ``lat``), and per participant:
    group, electrodes per hemisphere, lateralization (+ SOZ hemisphere and lobe, with the
    full participants table)."""
    part = data.participants()
    part = part[part.has_soz_ranking]
    el = data.electrodes()
    el = el[el.participant_id.isin(part.index) & el.x.notna()]
    n = el.groupby(["participant_id", "hemisphere"]).size().unstack(fill_value=0)[HEMIS]
    assert (n.left != n.right).all(), "tied hemisphere counts"
    soz = [c for c in SOZ_COLUMNS if c in part.columns]
    info = pd.DataFrame({
        "group": part.age_group.loc[n.index],
        "left": n.left, "right": n.right, "lat": np.where(n.left > n.right, "left", "right"),
        **{c: part[c].loc[n.index] for c in soz},
    }).rename_axis("participant").reset_index()
    info = data.add_age(info.drop(columns="group")).assign(group=info.group.to_numpy())
    info["total"] = info.left + info.right
    el = el.assign(lat=el.participant_id.map(info.set_index("participant").lat))
    return el, info


def make(out: Path) -> None:
    o = Output(out, "figS_soz_coverage")
    el, info = coverage()
    groups = info.group.astype(str)
    o.stat("n_participants", len(info))
    o.stat("n_electrodes", int(info.total.sum()))
    o.stat("lat_total", info.lat.value_counts().to_dict())
    tab = pd.crosstab(groups, info.lat).reindex(data.GROUPS)
    o.stat("lat_by_age", tab.to_dict(orient="index"))
    rng = np.random.default_rng(0)
    lat_test = chi2_permutation(groups.to_numpy(), info.lat.to_numpy(), rng)
    o.stat("lat_x_age_test", lat_test)
    o.stat("binom_left_vs_right", float(stats.binomtest(int((info.lat == "left").sum()), len(info)).pvalue))

    # electrodes per participant and hemisphere (participants with >= 1 electrode there)
    cnt = info.melt(id_vars=["participant", "group"], value_vars=HEMIS, var_name="hemisphere",
                    value_name="n_electrodes").query("n_electrodes > 0")
    cnt["group"] = cnt.group.astype(str)
    by_hemi = {}
    for g in data.GROUPS:
        left = cnt.query("group == @g and hemisphere == 'left'").n_electrodes.to_numpy()
        right = cnt.query("group == @g and hemisphere == 'right'").n_electrodes.to_numpy()
        sub = info[groups == g]
        by_hemi[g] = dict(
            n_participants=len(sub), n_bilateral=int(((sub.left > 0) & (sub.right > 0)).sum()),
            total_left=int(sub.left.sum()), total_right=int(sub.right.sum()),
            mean_left=float(left.mean()), sem_left=float(stats.sem(left)), n_left=len(left),
            mean_right=float(right.mean()), sem_right=float(stats.sem(right)), n_right=len(right),
            mannwhitney_p=float(stats.mannwhitneyu(left, right, alternative="two-sided").pvalue))
    o.stat("electrodes_by_hemisphere", by_hemi)
    o.stat("kruskal_total_across_age", dict(zip(("H", "p"), map(float, stats.kruskal(
        *[info.total[groups == g] for g in data.GROUPS])))))

    if "soz_lobe" in info:  # text statistics on the SOZ details
        o.stat("soz_hemisphere_x_age_test",
               chi2_permutation(groups.to_numpy(), info.soz_hemisphere.to_numpy(), rng))
        o.stat("soz_lobe_by_age",
               pd.crosstab(groups, info.soz_lobe).reindex(data.GROUPS).to_dict(orient="index"))
        o.stat("soz_lobe_x_age_test", chi2_permutation(groups.to_numpy(), info.soz_lobe.to_numpy(), rng))
        temporal = np.where(info.soz_lobe == "temporal", "temporal", "non-temporal")
        o.stat("temporal_by_age", {g: f"{int((temporal[groups == g] == 'temporal').sum())}/"
                                      f"{int((groups == g).sum())}" for g in data.GROUPS})
        o.stat("temporal_x_age_test", chi2_permutation(groups.to_numpy(), temporal, rng))
    else:
        print("  skipped the SOZ hemisphere / lobe statistics: needs the full participants table")

    with plt.rc_context({"font.size": 8}):
        o.figure(figure(el, info, cnt, tab, by_hemi, lat_test), png=True)
    o.source("A_electrodes", el[["participant_id", "electrode_id", "hemisphere", "x", "y", "z", "lat"]]
             .rename(columns={"participant_id": "participant", "electrode_id": "electrode"}))
    o.source("B_electrodes_per_hemisphere", cnt)
    o.source("C_lateralization", info[["participant", "group", "age", "left", "right", "total", "lat",
                                       *[c for c in SOZ_COLUMNS if c in info]]])
    o.write_stats()


def figure(el: pd.DataFrame, info: pd.DataFrame, cnt: pd.DataFrame, tab: pd.DataFrame,
           by_hemi: dict, lat_test: dict):
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 1.25, 1], hspace=0.3, wspace=0.25)
    colors = style.palette("whole_brain")
    lat_colors = {"left": colors[1], "right": colors[2]}
    for r, lat in enumerate(HEMIS):
        for c, hemi in enumerate(HEMIS):
            ax = fig.add_subplot(gs[r, 2 * c:2 * c + 2])
            sub = el[(el.hemisphere == hemi) & (el.lat == lat)]
            brain_background(ax, hemi=hemi)
            sign = -1 if hemi == "left" else 1
            ax.scatter(sign * sub.y, sub.z, s=3, color=lat_colors["left"], edgecolors="k",
                       linewidths=0.3)
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.set_title(f"{hemi.capitalize()} hemisphere: {len(sub)} electrodes, "
                         f"{sub.participant_id.nunique()} participants", fontsize=7)
    n_lat = info.lat.value_counts()
    for y, lat in zip((0.755, 0.47), HEMIS):
        fig.text(0.075, y, f"{lat.capitalize()}-lateralized SOZ\n(n={n_lat[lat]})", rotation=90,
                 ha="center", va="center", fontsize=8, color=lat_colors[lat])

    axB = fig.add_subplot(gs[2, 0:2])
    sns.barplot(data=cnt, x="group", y="n_electrodes", hue="hemisphere", order=data.GROUPS,
                hue_order=HEMIS, palette=[lat_colors[h] for h in HEMIS], errorbar="se", ax=axB,
                width=0.7)
    ymax = cnt.groupby(["group", "hemisphere"]).n_electrodes.agg(lambda v: v.mean() + stats.sem(v)).max()
    for i, g in enumerate(data.GROUPS):
        p = by_hemi[g]["mannwhitney_p"]
        lab = "n.s." if p >= 0.05 else ("*" if p >= 0.01 else "**")
        axB.plot([i - 0.175, i + 0.175], [ymax * 1.08] * 2, color="k", lw=0.6)
        axB.text(i, ymax * 1.1, f"{lab} (p={p:.2f})", ha="center", va="bottom", fontsize=5.5)
    axB.set_ylim(0, ymax * 1.25)
    axB.set_xlabel("Age group (years)")
    axB.set_ylabel("Electrodes per participant")
    axB.legend(title=None, frameon=False, fontsize=6, ncols=2, loc="lower left",
               bbox_to_anchor=(0, 1.0), borderaxespad=0)
    sns.despine(ax=axB)

    axC = fig.add_subplot(gs[2, 2:4])
    tl = tab.rename_axis(index="group", columns=None).reset_index().melt(
        id_vars="group", var_name="lat", value_name="n")
    sns.barplot(data=tl, x="group", y="n", hue="lat", order=data.GROUPS, hue_order=HEMIS,
                palette=[lat_colors[h] for h in HEMIS], ax=axC, width=0.7)
    for cont in axC.containers:
        axC.bar_label(cont, fontsize=6, padding=1)  # type: ignore[arg-type]
    axC.text(0.99, 0.99, f"lateralization × age: χ²({lat_test['dof']})={lat_test['chi2']:.2f}, "
             f"p={lat_test['p_permutation']:.2f}", transform=axC.transAxes, ha="right", va="top",
             fontsize=5.5)
    axC.set_ylim(0, tl.n.max() * 1.35)
    axC.set_xlabel("Age group (years)")
    axC.set_ylabel("Participants")
    axC.legend(title=None, frameon=False, fontsize=6, ncols=2, loc="lower left",
               bbox_to_anchor=(0, 1.0), borderaxespad=0)
    sns.despine(ax=axC)
    for x, y, letter in [(0.09, 0.9, "A"), (0.09, 0.31, "B"), (0.5, 0.31, "C")]:
        fig.text(x, y, letter, fontsize=12, fontweight="bold")
    return fig
