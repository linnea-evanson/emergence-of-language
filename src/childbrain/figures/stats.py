"""Statistics of the figures: cluster permutation over time, group tests, FDR, bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN kept)."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    pv = p[valid]
    order = np.argsort(pv)
    m = len(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0.0, 1.0)
    out = np.empty(m)
    out[order] = ranked
    q[valid] = out
    return q


def cluster_mask(data: np.ndarray, n_permutations: int = 1000, tail: int = 1,
                 alpha: float = 0.05, threshold: float | None = None,
                 seed: int | None = 2) -> np.ndarray:
    """Cluster-based one-sample permutation test against 0 over time (MNE).

    ``data``: participants x times. Returns the boolean mask of significant times.
    1,000 sign-flip permutations, with a fixed ``seed`` for reproducibility.
    """
    from mne.stats import permutation_cluster_1samp_test

    t_obs, clusters, cluster_p, _ = permutation_cluster_1samp_test(
        data, n_permutations=n_permutations, threshold=threshold, tail=tail,
        out_type="mask", rng=seed, verbose=False)
    mask = np.zeros_like(t_obs, dtype=bool)
    for cluster, p in zip(clusters, cluster_p):
        if p < alpha:
            mask[cluster] = True
    return mask


def permute_cluster(df: pd.DataFrame, time_col: str = "time", value_col: str = "score",
                    subject_col: str = "participant", **kwargs) -> np.ndarray:
    """:func:`cluster_mask` of one time course per subject (fewer than 3: none)."""
    pivot = df.pivot_table(index=subject_col, columns=time_col, values=value_col,
                           aggfunc="mean", observed=True)
    data = pivot.to_numpy()
    if data.shape[0] < 3:
        return np.zeros(data.shape[1], dtype=bool)
    return cluster_mask(data, **kwargs)


def window_mean(df: pd.DataFrame, tmin: float, tmax: float, keys: list[str],
                time_col: str = "time", value_col: str = "score") -> pd.DataFrame:
    """Mean of ``value_col`` over ``tmin <= time <= tmax`` per ``keys``."""
    sub = df[(df[time_col] >= tmin) & (df[time_col] <= tmax)]
    return sub.groupby(keys, as_index=False, observed=True)[value_col].mean()


def sig_to_plot(sig: np.ndarray) -> np.ndarray | None:
    """1 where significant, NaN elsewhere (None if nothing is significant)."""
    if not np.any(sig):
        return None
    return np.where(sig, 1.0, np.nan)


def wilcoxon_greater(values: np.ndarray, alternative: str = "greater") -> float:
    """One-sample Wilcoxon signed-rank p against 0 (NaN with fewer than 3 values)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.nan
    try:
        return float(stats.wilcoxon(values, alternative=alternative).pvalue)
    except ValueError:
        return np.nan


def mannwhitney(a: np.ndarray, b: np.ndarray, alternative: str = "less") -> float:
    """Mann-Whitney U p (``less``: a < b, i.e. the younger group lower)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    try:
        return float(stats.mannwhitneyu(a, b, alternative=alternative).pvalue)
    except ValueError:
        return np.nan


def label_permutation(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                      n_perm: int = 10_000) -> float:
    """One-sided permutation p of mean(y) - mean(x) > 0, shuffling the group labels."""
    both = np.concatenate([x, y])
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(both)
        null[i] = perm[len(x):].mean() - perm[:len(x)].mean()
    return float((np.sum(null >= y.mean() - x.mean()) + 1) / (n_perm + 1))


def p_to_stars(p: float, threshold: float = 0.05) -> str:
    if not np.isfinite(p) or p > threshold:
        return ""
    return "***" if p <= 0.001 else "**" if p <= 0.01 else "*"


def paper_bar_stars(p: float) -> str:
    """Stars of the bars of Figs. 2 and 6: ``*`` p < 0.05, ``**`` p < 0.01."""
    return "" if not np.isfinite(p) or p >= 0.05 else "**" if p < 0.01 else "*"


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial correlation of b vs a (> 0: b larger), from the Mann-Whitney U."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    u = stats.mannwhitneyu(b, a, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def bootstrap_ci(values: np.ndarray, stat=np.mean, n: int = 10_000, level: float = 0.95,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval of ``stat`` (resampling ``values``)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    boot = stat(values[idx], axis=1)
    lo = (1 - level) / 2
    return float(np.quantile(boot, lo)), float(np.quantile(boot, 1 - lo))
