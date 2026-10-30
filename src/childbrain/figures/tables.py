"""Tables of the paper: ROI coverage (Table S1), encoding (Table S2, ``ci_table``) and
decoding (Table S3) group means and age effects, and the cohort counts quoted in the text.

Group means: 95% bootstrap CI over participants (10,000 resamples), one-sided Wilcoxon
test > 0. Age effects: rank-biserial correlation r (> 0: the older group scores higher)
with a 90% bootstrap CI (each group resampled), one-sided Mann-Whitney U test. The
bootstraps are seeded and draw participants in ``participant`` order.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, wilcoxon

from . import data
from .output import Output
from .style import AVG_TMAX, AVG_TMIN

G = data.GROUPS
PAIRS = [(0, 1), (0, 2), (1, 2)]
N_BOOT = 10_000
MIN_N = 5  # cells with a smaller group are flagged with a dagger
LABELS = {**data.ROI_TITLES, "OCC": r"\draft{Occipital}"}
LEVEL_NAMES = {"phonetic": "Phonetic", "lexical": "Lexical", "supralexical": "Supralexical"}
HEADER = ("\\begin{tabular}{lcccccc}\n\\hline\n"
          " & \\multicolumn{3}{c}{Mean $R$ [95\\% CI]} & \\multicolumn{3}{c}{Age effect: $r$ [90\\% CI]} \\\\\n"
          "ROI & " + " & ".join(f"{g} years" for g in G) + " & "
          + " & ".join(f"{G[i]} vs {G[j]}" for i, j in PAIRS) + " \\\\\n\\hline\n")
FOOTER = "\n\\hline\n\\end{tabular}\n"


def boot_ci(v: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean."""
    means = np.random.default_rng(seed).choice(v, size=(N_BOOT, len(v)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """> 0 when b (older) tends to score higher than a (younger)."""
    return float(2 * mannwhitneyu(b, a).statistic / (len(a) * len(b)) - 1)


def age_effect(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict:
    """Rank-biserial r of b vs a, 90% bootstrap CI, one-sided Mann-Whitney U p (b > a)."""
    rng = np.random.default_rng(seed)
    idx = [(rng.integers(0, len(a), len(a)), rng.integers(0, len(b), len(b))) for _ in range(N_BOOT)]
    # Mann-Whitney U of each resample, vectorised: rank sum of b minus its minimum.
    ranks = rankdata(np.array([np.concatenate([a[i], b[j]]) for i, j in idx]), axis=1)
    u = ranks[:, len(a):].sum(1) - len(b) * (len(b) + 1) / 2
    lo, hi = np.percentile(2 * u / (len(a) * len(b)) - 1, [5, 95])
    return dict(r=rank_biserial(a, b), ci90=[float(lo), float(hi)], n=[len(a), len(b)],
                p=float(mannwhitneyu(b, a, alternative="greater").pvalue))


def by_group(df: pd.DataFrame, col: str = "score") -> dict[str, np.ndarray]:
    df = df.sort_values("participant")
    return {g: df[df.group == g][col].dropna().to_numpy(float) for g in G}


def group_means(vals: dict[str, np.ndarray]) -> tuple[dict, list[str]]:
    """Per group: mean, 95% bootstrap CI, one-sided Wilcoxon p; and the table cells."""
    entry: dict = {}
    cells: list[str] = []
    for g in G:
        v = vals[g]
        p = float(wilcoxon(v, alternative="greater").pvalue) if len(v) else np.nan
        lo, hi = boot_ci(v) if len(v) > 1 else (np.nan, np.nan)
        entry[g] = dict(R=float(v.mean()), ci95=[lo, hi], n=len(v), p=p)
        cells.append(f"{v.mean():.3f} [{lo:.3f}, {hi:.3f}]{'*' if p < 0.05 else ''}")
    age = {f"{G[i]}:{G[j]}": age_effect(vals[G[i]], vals[G[j]])
           for i, j in PAIRS if len(vals[G[i]]) > 1 and len(vals[G[j]]) > 1}
    entry["age"] = age
    cells += [age_cell(e) for e in age.values()]
    return entry, cells


def age_cell(e: dict) -> str:
    flag = r"$^\dagger$" if min(e["n"]) < MIN_N else ""
    return f"{e['r']:.2f} [{e['ci90'][0]:.2f}, {e['ci90'][1]:.2f}]{'*' if e['p'] < 0.05 else ''}{flag}"


def typeset(table: str) -> str:
    """LaTeX typography: en dashes in age ranges, minus signs."""
    table = re.sub(r"\b(1|6)-(5|11)\b", r"\1--\2", table.replace("-0.000", "0.000"))
    return re.sub(r"(?<![\w$-])-(\d)", r"$-$\1", table)


def cohort(out: Output, pw: pd.DataFrame) -> str:
    """Cohort counts (text) and the ROI coverage rows (Table S1, without "All ages")."""
    part = data.participants()
    located = data.located(pw)
    unlocated = part.loc[sorted(set(pw.participant) - set(located.participant)), "age_group"]
    groups = part.loc[sorted(set(pw.participant)), "age_group"]
    out.stat("n_participants", pw.participant.nunique())
    out.stat("n_electrodes", len(pw))
    out.stat("n_localised", located.participant.nunique())
    out.stat("groups", {g: int((groups == g).sum()) for g in G})
    out.stat("unlocated_groups", {g: int((unlocated == g).sum()) for g in G})
    rows, lines = [], []
    for roi in data.ROIS:
        sub = pw if roi == "whole_brain" else located.merge(
            data.roi_electrodes(roi), on=["participant", "electrode"])
        per = sub.groupby("participant", as_index=False).size().pipe(data.add_age)
        n_g = [int((per.group == g).sum()) for g in G]
        rows.append(dict(roi=roi, n_electrodes=len(sub), n_participants=len(per),
                         **{f"n_{g}": n for g, n in zip(G, n_g)}))
        label = "Occipital" if roi == "OCC" else data.ROI_TITLES[roi]
        lines.append(f"{label} & {len(sub)} & {len(per)} & " + " & ".join(map(str, n_g)) + " \\\\")
    cov = pd.DataFrame(rows)
    out.source("roi_coverage", cov)
    out.stat("roi_coverage", cov.set_index("roi").to_dict("index"))
    return "\n".join(lines) + "\n"


def encoding(out: Output, pw: pd.DataFrame) -> str:
    """Table S2: R(phonemes+words) per ROI and age group (Fig. 2B) and age effects."""
    means = data.roi_means(pw)
    out.source("ci_table", means[["roi", "participant", "age", "group", "score"]])
    stats, lines = {}, []
    for roi in data.ROIS:
        stats[roi], cells = group_means(by_group(means[means.roi == roi]))
        lines.append(f"{LABELS[roi]} & " + " & ".join(cells) + r" \\")
    out.stat("encoding", stats)
    return typeset(HEADER + "\n".join(lines) + FOOTER)


def decoding(out: Output) -> str:
    """Table S3: decoding score averaged over [AVG_TMIN, AVG_TMAX] per level, ROI, group."""
    avgs: list[pd.DataFrame] = []
    stats: dict = {}
    cells: dict = {}
    for roi in data.ROIS:
        lv = data.decoding_levels(roi, data.LEVELS)
        lv = lv[(lv.time >= AVG_TMIN) & (lv.time <= AVG_TMAX)]
        avg = lv.groupby(["participant", "level"], as_index=False).score.mean().pipe(data.add_age)
        avgs.append(avg.assign(roi=roi))
        stats[roi] = {}
        for level in data.LEVELS:
            stats[roi][level], cells[roi, level] = group_means(by_group(avg[avg.level == level]))
    out.source("decoding_effects_table",
               pd.concat(avgs)[["level", "roi", "participant", "age", "group", "score"]])
    out.stat("decoding", stats)
    lines = []
    for level in data.LEVELS:
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{LEVEL_NAMES[level]} features}}}} \\\\")
        lines += [f"\\quad {LABELS[roi]} & " + " & ".join(cells[roi, level]) + r" \\"
                  for roi in data.ROIS]
    return typeset(HEADER + "\n".join(lines) + FOOTER)


def make(out_dir: Path) -> None:
    out = Output(out_dir, "tables")
    pw = data.trf_scores(data.trf_linguistic(), data.TRF_LEXICAL)
    out.text("roi_coverage.tex", cohort(out, pw))
    out.text("ci_table.tex", encoding(out, pw))
    out.text("decoding_effects_table.tex", decoding(out))
    out.write_stats()
