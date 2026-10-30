"""Time-resolved decoding: features predicted from word-locked neural epochs."""

import logging
import typing as tp

import exca
import neuralset as ns
import numpy as np
import pandas as pd
from tqdm import tqdm

from .base import (
    BaseData,
    BaseExperiment,
    RidgeCVConfig,
    SpearmanR,
    wilcoxon_greater_pvalue,
)

logger = logging.getLogger(__name__)


def _stack_neuro_at_time(neuro: np.ndarray, t: int, radius: int) -> np.ndarray:
    """``(trials, channels, times)`` → ``(trials, channels * n)`` for times ``t ± radius``.

    The window is clipped at the epoch edges (two samples there for ``radius=1``).
    """
    if radius == 0:
        return neuro[:, :, t]
    n_times = neuro.shape[2]
    indices = list(range(max(0, t - radius), min(n_times, t + radius + 1)))
    return np.concatenate([neuro[:, :, i] for i in indices], axis=1)


class EpochData(BaseData):
    """Epochs from ``start`` to ``stop`` (s) around each ``event_type`` trigger.

    Triggers are further filtered by the extractor's ``event_query`` if any.
    """

    start: float
    stop: float

    def model_post_init(self, context: tp.Any) -> None:
        super().model_post_init(context)
        if self.extractor.aggregation != "trigger":
            raise ValueError(f"{self} only supports trigger extractor.")
        trigger_query = f'type=="{self.event_type}"'
        event_query = getattr(self.extractor, "event_query", None)
        if event_query:
            trigger_query = f"({trigger_query}) and ({event_query})"
        self._segmenter = ns.dataloader.Segmenter(
            start=self.start,
            duration=self.stop - self.start,
            trigger_query=trigger_query,
            extractors=dict(neuro=self.neuro, extractor=self.extractor),
            drop_incomplete=True,
        )


class SlidingWindow(BaseExperiment):
    """Decode the extractor from the neuro at each epoch time sample.

    At each time ``t``, a ridge is fitted on the training folds' neuro at
    ``t ± neuro_time_radius`` (channels stacked) and scored on the test fold,
    per feature dimension.
    """

    data: EpochData
    model: RidgeCVConfig
    scoring: SpearmanR = SpearmanR()
    neuro_time_radius: int = 0
    infra: exca.MapInfra = exca.MapInfra(version="1")

    @infra.apply(item_uid=str)
    def _iter_subject_score(self, subjects: list[str]) -> tp.Iterator[pd.DataFrame]:
        for subject in subjects:
            yield self.get_subject_score(subject)

    def get_subject_score(self, subject: str) -> pd.DataFrame:
        all_events = self.data.study.run()
        dset = self.data._segmenter.apply(all_events.query(f"subject == '{subject}'"))
        dset = dset.select(dset.triggers.subject == subject)
        # extractor prepare affects padding, channel names etc., so it comes
        # after selecting the subject
        dset.prepare()
        return self.score_batch(dset.load_all(), subject=subject)

    def score_batch(self, batch: tp.Any, subject: str) -> pd.DataFrame:
        neuro = batch.data["neuro"].numpy()
        extractor = batch.data["extractor"].numpy()
        return self.score_arrays(neuro, extractor, batch.triggers, subject)

    def score_arrays(
        self,
        neuro: np.ndarray,
        extractor: np.ndarray,
        events: pd.DataFrame,
        subject: str,
    ) -> pd.DataFrame:
        """Scores from ``neuro`` (events, channels, times) and ``extractor`` (events, dims)."""
        logger.info("Fitting on neuro %s, extractor %s", neuro.shape, extractor.shape)
        n_events, _, n_times = neuro.shape
        if extractor.ndim != 2 or len(extractor) != n_events:
            raise ValueError(f"extractor should be (events, dims), got {extractor.shape}")
        times = np.linspace(self.data.start, self.data.stop, n_times)
        radius = self.neuro_time_radius

        scores = []
        splits = list(enumerate(self.cv.build(events)))
        for (split, (train, test)), t in tqdm(
            [(s, t) for s in splits for t in range(n_times)], desc="training..."
        ):
            model = self.model.build()
            model.fit(_stack_neuro_at_time(neuro[train], t, radius), extractor[train])
            preds = model.predict(_stack_neuro_at_time(neuro[test], t, radius))
            preds = np.atleast_2d(preds).T if preds.ndim == 1 else preds
            for dim, score in enumerate(self.scoring(extractor[test], preds)):
                scores.append(
                    dict(score=score, split=split, time=times[t], dim=dim, subject=subject)
                )
        return pd.DataFrame(scores)


def scores_to_df(
    scores: pd.DataFrame, feat_key: str, dim_names: list[str] | None = None
) -> pd.DataFrame:
    """Mean over CV splits per (subject, dim, time), with Wilcoxon p-values across splits.

    With ``dim_names`` (e.g. POS tags), ``feature`` is the dimension's name and
    each is tested separately; otherwise dimensions are averaged within each
    split before testing.
    """
    scores = scores.dropna(subset=["score"])
    df = scores.groupby(["subject", "dim", "time"], as_index=False)["score"].mean()
    df["feature_group"] = feat_key
    if dim_names is not None:
        n_dims = int(df["dim"].max()) + 1
        if len(dim_names) != n_dims:
            raise ValueError(f"{len(dim_names)} dim_names for {n_dims} dims of {feat_key}")
        df["feature"] = df["dim"].map(lambda d: dim_names[int(d)])
        df["feat_dim"] = df["feature"]
        splits = scores.assign(feature=scores["dim"].map(lambda d: dim_names[int(d)]))
    else:
        df["feature"] = feat_key
        df["feat_dim"] = feat_key + df["dim"].astype(str)
        splits = scores.groupby(["subject", "split", "time"], as_index=False)["score"].mean()
        splits["feature"] = feat_key
    pvalues = (
        splits.groupby(["subject", "feature", "time"])["score"]
        .apply(wilcoxon_greater_pvalue)
        .reset_index(name="pvalue")
    )
    return df.merge(pvalues, on=["subject", "feature", "time"], how="left")
