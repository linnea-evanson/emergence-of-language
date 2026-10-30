"""Fig. 1 (cohort and evoked responses), Fig. S2 (``fig1_phon``) and Fig. S1 (``glassbrain``).

* ``fig1``: A, the study schematic (artwork, ``assets/fig1_schematic.png``); B, number of
  electrodes per participant, at a random position within its age range; C, whole-brain word-evoked broadband response
  per age group (|evoked| averaged over the participant's electrodes, minus its mean in
  [-0.75, -0.5] s; dots: one-sided cluster permutation test > 0 across participants);
  D, |evoked| at 100 ms per located electrode, coloured when >= 1 µV.
* ``fig1_phon``: Fig. 1 D for phoneme onsets.
* ``glassbrain``: |evoked| at 100 ms after phoneme onset of every located electrode, both
  hemispheres, sagittal / axial / coronal views.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from . import brain, data, style
from .output import Output
from .stats import mannwhitney

SCHEMATIC = Path(__file__).parent / "assets" / "fig1_schematic.png"
#: Baseline subtracted from each participant's whole-brain curve (panel C).
BASELINE = (-0.75, -0.5)
#: Panel D: the response at 100 ms.
BRAIN_TIME = 0.1
#: Electrodes coloured on the brains (|µV|).
BRAIN_THRESHOLD = 1.0
#: Window of the mean response compared across age groups (Results text).
MEAN_WINDOW = (0.0, 1.0)
GROUP_GREYS = ["lightgrey", "grey", "black"]
BRAIN_CMAP = LinearSegmentedColormap.from_list("grey_blue", ["grey", "royalblue"])
UV = r"|$\mu$V|"


def evoked_curves(event: str = "word") -> pd.DataFrame:
    """Per participant and time: mean |evoked| over electrodes, minus its baseline mean."""
    ev = data.evoked(event)
    curves = (ev.assign(value=ev.value.abs())
              .groupby(["participant", "time"], as_index=False).value.mean())
    lo, hi = BASELINE
    base = curves[curves.time.between(lo, hi)].groupby("participant").value.mean()
    curves["value"] -= curves.participant.map(base)
    return data.add_age(curves)


def brain_values(event: str = "word") -> pd.DataFrame:
    """|evoked| at ``BRAIN_TIME`` of every located electrode, with its position."""
    ev = data.evoked(event)
    ev = ev[np.isclose(ev.time, BRAIN_TIME)]
    el = data.electrodes()
    el = el[el.x.notna()][["participant_id", "electrode_id", "hemisphere", "x", "y", "z", "medial"]]
    out = ev.merge(el, left_on=["participant", "electrode"],
                   right_on=["participant_id", "electrode_id"])
    out["abs_value"] = out.value.abs()
    out["coloured"] = out.abs_value >= BRAIN_THRESHOLD
    return out[["participant", "electrode", "age", "group", "hemisphere", "x", "y", "z", "medial",
                "abs_value", "coloured"]]


def cohort_counts() -> pd.DataFrame:
    """Electrodes analysed per participant (the encoding channels)."""
    trf = data.trf_linguistic()
    n = trf[trf.model == data.TRF_LEXICAL].groupby("participant").electrode.nunique()
    return data.add_age(n.rename("n_electrodes").reset_index())


def panel_counts(ax, counts: pd.DataFrame) -> None:
    """Electrodes per participant, each at a random position within its age range
    (:func:`style.age_axis_positions`: exact ages are not shown)."""
    x = counts.participant.map(style.age_axis_positions())
    for group, color in zip(data.GROUPS, GROUP_GREYS):
        g = counts.group == group
        label = f"{group}years old" if group == data.GROUPS[-1] else group
        ax.scatter(x[g], counts.n_electrodes[g], color=color, edgecolors="k", linewidth=0.5, s=25,
                   label=label)
    style.age_axis(ax)
    ax.set_yticks([counts.n_electrodes.min(), counts.n_electrodes.max()])
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("# electrodes", labelpad=-15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(bbox_to_anchor=(-0.15, -0.5), loc="upper left", frameon=False, ncol=3,
              columnspacing=0.1, borderaxespad=0.0, borderpad=0.0, handletextpad=0.1,
              labelspacing=0.0)


def panel_curves(ax, curves: pd.DataFrame) -> dict[str, np.ndarray]:
    """Group-mean curves (no error band) with the cluster-test dots; returns the masks."""
    colors = style.palette("whole_brain")
    m = curves.groupby(["group", "time"], observed=True).value.mean()
    ylim = [min(-0.1, 1.2 * m.min()), max(0.5, 1.15 * m.max())]
    sns.lineplot(data=curves.assign(group=curves.group.astype(str)), x="time", y="value",
                 hue="group", hue_order=data.GROUPS, palette=colors, errorbar=None, ax=ax,
                 legend=False)
    sns.despine(ax=ax)
    c = ylim[1] * 0.05
    sig = style.cluster_dots(ax, curves, colors, a=-ylim[0] - 2 * c, c=c, col="value")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(UV, rotation=0)
    ax.yaxis.set_label_coords(-0.12, 0.4)
    ticks = [t for t in ax.get_yticks() if ylim[0] <= t <= ylim[1]]
    ax.set_yticks(ticks, [str(np.round(t, 3)).replace("0.", ".") if t in (0, max(ticks)) else ""
                          for t in ticks])
    ax.axhline(0, color="lightgrey", linestyle="--", zorder=-1)
    ax.set_ylim(ylim)
    return sig


def evoked_brain(ax, values: pd.DataFrame, legend: bool) -> None:
    """|evoked| per electrode on the left lateral cortex (colour 0-10 µV, size 0-20 µV,
    grey dots below ``BRAIN_THRESHOLD``)."""
    df = values[~values.medial].assign(
        s=brain.get_sizes(values.abs_value[~values.medial].to_numpy(), 0, 20, s0=10))
    df = df.sort_values("s", kind="stable")
    brain.brain_background(ax)
    ns = df[~df.coloured]
    ax.scatter(-ns.y, ns.z, color="grey", s=2, edgecolors="k", linewidth=0.5)
    sel = df[df.coloured]
    scatter = ax.scatter(-sel.y, sel.z, c=sel.abs_value, s=sel.s, edgecolors="k", linewidth=0.5,
                         cmap=BRAIN_CMAP, vmin=0, vmax=10)
    if legend:
        vsmall = float(values.abs_value[values.coloured].min())  # medial ones included
        brain.plot_legend(ax, vsmall, scatter, 0, 10, fs=12, title=UV,
                          plot_ns=False, anchor=(-0.1, -0.1))
    ax.set_axis_off()


def brain_row(axes: list, values: pd.DataFrame, counts: pd.DataFrame, ylim_scale: float,
              equal: bool = False) -> None:
    """Panel D: one brain per age group; titles give every participant and electrode."""
    for i, (group, ax) in enumerate(zip(data.GROUPS, axes)):
        evoked_brain(ax, values[values.group == group], legend=i == 0)
        y0, y1 = ax.get_ylim()
        ax.set_ylim([y0, y1 * ylim_scale])
        if equal:
            ax.set_aspect("equal", adjustable="datalim")
        g = counts[counts.group == group]
        ax.text(0.25, 0.98, f"{group} years old", fontsize=20, transform=ax.transAxes)
        ax.text(0.25, 0.9, f"$n={len(g)}$ ({g.n_electrodes.sum():,} electrodes)", fontsize=14,
                transform=ax.transAxes)
        for coll in ax.collections:
            coll.set_rasterized(True)


def group_counts(counts: pd.DataFrame, values: pd.DataFrame) -> dict:
    """Participants and electrodes per age group: every one, and the located ones."""
    return {g: dict(participants=int((counts.group == g).sum()),
                    electrodes=int(counts.n_electrodes[counts.group == g].sum()),
                    located_participants=int(values.participant[values.group == g].nunique()),
                    located_electrodes=int((values.group == g).sum()))
            for g in data.GROUPS}


def sig_intervals(times: np.ndarray, mask: np.ndarray) -> list[list[float]]:
    """[first, last] time of each run of significant samples."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return [[float(times[a]), float(times[b - 1])] for a, b in zip(edges[::2], edges[1::2])]


def make_fig1(out: Path) -> None:
    o = Output(out, "fig1")
    counts = cohort_counts()
    curves = evoked_curves("word")
    values = brain_values("word")

    mosaic: list = [["A", "A", "B", "e0", "e0", "e1", "e1", "e2", "e2"],
              ["A", "A", "C", "e0", "e0", "e1", "e1", "e2", "e2"]]
    fig, axs = plt.subplot_mosaic(mosaic, figsize=(18, 4), layout="constrained", dpi=300)
    axs["A"].imshow(mpimg.imread(SCHEMATIC))
    axs["A"].set_axis_off()
    sig = panel_curves(axs["C"], curves)
    panel_counts(axs["B"], counts)
    brain_row([axs["e0"], axs["e1"], axs["e2"]], values, counts, ylim_scale=1.2)
    for x, y, letter in [(0.19, 0.95, "B"), (0.19, 0.4, "C"), (0.35, 0.95, "D")]:
        fig.text(x, y, letter, fontsize=20, fontweight="bold")
    o.figure(fig, png=True)

    times = np.sort(curves.time.unique())
    o.source("B_electrodes_per_participant", counts.assign(
        age_range=counts.participant.map(data.participants().age_range),
        x=counts.participant.map(style.age_axis_positions()))[
        ["participant", "age_range", "group", "n_electrodes", "x"]])
    o.source("C_participant_curves", curves[["participant", "age", "group", "time", "value"]])
    means = curves.groupby(["group", "time"], observed=True).value.mean().rename("mean")
    sig_df = pd.concat([pd.DataFrame(dict(group=g, time=times, significant=m)) for g, m in sig.items()])
    o.source("C_group_means", means.reset_index().merge(sig_df, on=["group", "time"]))
    o.source("D_electrodes", values)

    lo, hi = MEAN_WINDOW
    mean_resp = curves[curves.time.between(lo, hi)].groupby("participant").value.mean()
    mean_resp = data.add_age(mean_resp.rename("value").reset_index())
    vals = {g: mean_resp.value[mean_resp.group == g].to_numpy() for g in data.GROUPS}
    o.stat("n_participants", int(len(counts)))
    o.stat("n_electrodes", int(counts.n_electrodes.sum()))
    o.stat("electrodes_per_participant", [int(counts.n_electrodes.min()),
                                          int(counts.n_electrodes.max())])
    o.stat("n_located_participants", int(values.participant.nunique()))
    o.stat("n_located_electrodes", int(len(values)))
    o.stat("groups", group_counts(counts, values))
    o.stat("C_cluster_significant_s", {g: sig_intervals(times, m) for g, m in sig.items()})
    o.stat("C_mean_response_0_1s", {g: float(v.mean()) for g, v in vals.items()})
    o.stat("C_mannwhitney_0_1s_less", {
        f"{a}<{b}": mannwhitney(vals[a], vals[b]) for i, a in enumerate(data.GROUPS)
        for b in data.GROUPS[i + 1:]})
    o.stat("D_n_coloured_electrodes", int((values.coloured & ~values.medial).sum()))
    o.write_stats()


def make_fig1_phon(out: Path, values: pd.DataFrame) -> None:
    o = Output(out, "fig1_phon")
    counts = cohort_counts()
    mosaic: list = [["e0", "e1", "e2"]]
    fig, axs = plt.subplot_mosaic(mosaic, figsize=(15.6, 4.8), layout="constrained",
                                  dpi=300)
    brain_row([axs["e0"], axs["e1"], axs["e2"]], values, counts, ylim_scale=1.3, equal=True)
    o.figure(fig, png=True)
    o.source("electrodes", values)
    o.stat("n_located_participants", int(values.participant.nunique()))
    o.stat("n_located_electrodes", int(len(values)))
    o.stat("groups", group_counts(counts, values))
    o.stat("n_coloured_electrodes", int((values.coloured & ~values.medial).sum()))
    o.write_stats()


def make_glassbrain(out: Path, values: pd.DataFrame) -> None:
    """Dot size: |evoked| scaled between its 5th and 95th percentiles."""
    o = Output(out, "glassbrain")
    d = values.copy()
    s = d.abs_value.fillna(0)
    p5, p95 = np.percentile(s, [5, 95])
    d["size"] = ((((s - p5) / (p95 - p5)).clip(0, 1) * 10) ** 2) / 50 * 8
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), gridspec_kw={"wspace": 0.05, "hspace": 0.05})
    views = [("y", "z"), ("x", "y"), ("x", "z")]  # sagittal, axial, coronal
    for row, hemi, surf in zip(axes, ("left", "right"), brain.pial_surfaces()):
        h = d[d.hemisphere == hemi]
        for ax, (a, b) in zip(row, views):
            ia, ib = "xyz".index(a), "xyz".index(b)
            ax.scatter(surf[::5, ia], surf[::5, ib], color="0.6", s=40, zorder=-2, rasterized=True)
            ax.scatter(surf[::5, ia], surf[::5, ib], color="white", s=12, zorder=-1, rasterized=True)
            ax.scatter(h[a], h[b], s=h["size"], c="k", linewidths=0)
            ax.set_aspect("equal")
            ax.set_axis_off()
    o.figure(fig, png=True)
    o.source("electrodes", d)
    o.stat("n_electrodes", int(len(d)))
    o.stat("size_percentiles_uV", [float(p5), float(p95)])
    o.write_stats()


def make(out: Path) -> None:
    make_fig1(out)
    phon = brain_values("phoneme")
    make_fig1_phon(out, phon)
    make_glassbrain(out, phon)
