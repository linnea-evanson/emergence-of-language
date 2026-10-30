"""Metrics, splitter, ridge and per-subject scoring shared by decoding and encoding."""

import logging
import typing as tp

import exca
import neuralset as ns
import numpy as np
import pandas as pd
import pydantic
import scipy.stats
from sklearn import linear_model, preprocessing
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline

logger = logging.getLogger(__name__)


class Correlation(pydantic.BaseModel):
    """Pearson correlation per column of ``(samples, dims)`` arrays."""

    model_config = pydantic.ConfigDict(extra="forbid")

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
        SX2 = (X**2).sum(0) ** 0.5
        SY2 = (Y**2).sum(0) ** 0.5
        SXY = (X * Y).sum(0)
        prod = SX2 * SY2
        non_zero = prod != 0
        R = np.zeros_like(prod)
        R[non_zero] = SXY[non_zero] / prod[non_zero]
        return R


class SpearmanR(pydantic.BaseModel):
    """Spearman correlation per column of ``(samples, dims)`` arrays."""

    model_config = pydantic.ConfigDict(extra="forbid")

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.array(
            [scipy.stats.spearmanr(x[:, f], y[:, f])[0] for f in range(x.shape[1])]
        )


class Cv(pydantic.BaseModel):
    """Contiguous, non-shuffled K-fold over the trigger events."""

    n_splits: int = 10
    model_config = pydantic.ConfigDict(extra="forbid")

    def build(self, events: pd.DataFrame) -> tp.Iterator[tuple[np.ndarray, np.ndarray]]:
        return KFold(self.n_splits, shuffle=False).split(events)


class RidgeCVConfig(pydantic.BaseModel):
    """``sklearn.linear_model.RidgeCV`` (efficient leave-one-out over ``alphas``)."""

    alphas: list[float]
    alpha_per_target: bool
    model_config = pydantic.ConfigDict(extra="forbid")

    def build(self) -> Pipeline:
        """``StandardScaler`` then ``RidgeCV``."""
        ridge = linear_model.RidgeCV(
            alphas=self.alphas, alpha_per_target=self.alpha_per_target
        )
        return make_pipeline(preprocessing.StandardScaler(), ridge)


class BaseData(pydantic.BaseModel):
    """Study, neuro and feature extractors, segmented by ``_segmenter``."""

    study: ns.Step
    neuro: ns.extractors.BaseExtractor
    extractor: ns.extractors.BaseExtractor
    event_type: str
    model_config = pydantic.ConfigDict(extra="forbid")
    _segmenter: ns.dataloader.Segmenter


class BaseExperiment(pydantic.BaseModel):
    """Scores one subject at a time.

    Subclasses define ``infra`` and ``_iter_subject_score`` (``@infra.apply`` is
    bound at class definition), caching one item per subject.
    """

    data: BaseData
    cv: Cv = Cv()
    infra: exca.MapInfra
    model_config = pydantic.ConfigDict(extra="forbid")

    def _iter_subject_score(self, subjects: list[str]) -> tp.Iterator[pd.DataFrame]:
        raise NotImplementedError

    def get_scores(self, subjects: list[str]) -> pd.DataFrame:
        """Scores of every subject, one row per (subject, split, dim, …)."""
        events = self.data.study.run()
        cached = set(self.infra.cache_dict.keys())
        missing = [s for s in subjects if self.infra.item_uid(s) not in cached]
        if missing:
            # Prepare shared extractors (e.g. model embeddings) across subjects first
            dset = self.data._segmenter.apply(events.loc[events.subject.isin(missing)])
            dset.prepare()
        return pd.concat(list(self._iter_subject_score(subjects)), ignore_index=True)


def wilcoxon_greater_pvalue(values: np.ndarray) -> float:
    """One-sided Wilcoxon signed-rank test: median > 0 across CV splits."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return float("nan")
    try:
        return float(scipy.stats.wilcoxon(values, alternative="greater").pvalue)
    except ValueError:
        return float("nan")
