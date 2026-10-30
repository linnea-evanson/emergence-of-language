"""TRF encoding: continuous features → iEEG, fitted per chapter (``mne.decoding.ReceptiveField``).

For each chapter:

* regressors and neuro are z-scored over the whole chapter;
* ``train_test_split(test_size=0.2, shuffle=False)``: per-channel ridge alphas
  are fitted once on the trailing 20% (``RidgeCV``, leave-one-out);
* a non-shuffled 10-fold ``KFold`` over the leading 80%, with those alphas
  fixed, scores held-out correlations on the samples where the lagged design
  matrix is fully defined (``ReceptiveField.valid_samples_``).

Scores are averaged over folds within a chapter, then over chapters. The
per-channel permutation null (1000 permutations of the held-out signal) uses the
same statistic on the held-out predictions.
"""

import logging
import os
import typing as tp

import exca
import neuralset as ns
import numpy as np
import pandas as pd
import pydantic
import torch
from mne.decoding import ReceptiveField
from neuralset.dataloader import Batch, SegmentDataset
from neuralset.events.utils import extract_events
from neuralset.segments import find_incomplete_segments
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .base import BaseData, BaseExperiment, Correlation, wilcoxon_greater_pvalue

logger = logging.getLogger(__name__)


class _TimeSegmentDataset(SegmentDataset):
    """SegmentDataset whose batches concatenate segments along time."""

    pad_duration: None = None

    def collate_fn(self, batches: list[Batch]) -> Batch:
        if not batches:
            return Batch(data={}, segments=[])
        if not batches[0].data:
            raise ValueError(f"No extractor in first batch: {batches[0]}")
        extractors = {}
        for name in batches[0].data:
            extractors[name] = torch.cat([b.data[name] for b in batches], dim=-1)
        dummy_segments = [ns.segments.Segment(start=0, duration=1, timeline="")] * len(
            extractors[name]
        )
        return Batch(data=extractors, segments=dummy_segments)

    def _check_padding(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        self._pad_lengths: dict[str, int | None]  # type: ignore
        self._pad_lengths = {k: None for k in self.extractors.keys()}


def _load_data(dset: SegmentDataset) -> tuple[np.ndarray, np.ndarray]:
    """(regressors, neuro), each ``(times, dims)``, the segments concatenated."""
    dset_ = _TimeSegmentDataset(segments=dset.segments, extractors=dset.extractors)
    batch = dset_.load_all(num_workers=max(1, os.cpu_count() or 1))
    neuro = batch.data["neuro"].numpy()
    extractor = batch.data["extractor"].numpy()
    if neuro.shape[-1] != extractor.shape[-1]:
        raise RuntimeError("`extractor` and `neuro` extractors have different durations.")
    return extractor[0].T, neuro[0].T


def dset_channels(dset: SegmentDataset) -> list[str]:
    """Neuro channel names of ``dset``, in the first recording's order.

    With several recordings, only the channels present in all of them.
    """
    neuro = dset.extractors["neuro"]
    events = neuro._event_types_helper.extract(extract_events(dset.segments))
    names: list[str] | None = None
    for ta in neuro._get_data(events):
        present = set(ta.ch_names)
        names = list(ta.ch_names) if names is None else [ch for ch in names if ch in present]
    if not names:
        raise ValueError("no neuro channels found for this chapter")
    return names


class TRFData(BaseData):
    """Chapters cut into contiguous ``duration``-long segments.

    Word-less segments are kept (their regressor is zero), so
    ``drop_incomplete=False``; segments without neuro are dropped by the
    experiment.
    """

    duration: pydantic.PositiveFloat
    stride: pydantic.PositiveFloat

    def model_post_init(self, context: tp.Any) -> None:
        super().model_post_init(context)
        self._segmenter = ns.dataloader.Segmenter(
            start=0.0,
            duration=self.duration,
            trigger_query=f'type=="{self.event_type}"',
            extractors=dict(neuro=self.neuro, extractor=self.extractor),
            drop_incomplete=False,
            stride=self.stride,
            stride_drop_incomplete=True,
        )


class TRFEncoding(BaseExperiment):
    """Per-chapter TRF encoding scores of every channel, one subject per item.

    ``chapter_pca_components``: StandardScaler + PCA of each chapter's
    regressors before the TRF (model embeddings)
    (``IncrementalPCA`` with ``chapter_pca_incremental``).
    """

    data: TRFData
    tmin: float
    tmax: float
    alphas: list[float]
    scoring: Correlation = Correlation()
    alpha_holdout: float = 0.2
    n_permutations: int = 0
    null_seed: int = 0
    chapter_pca_components: int | None = None
    chapter_pca_incremental: bool = False
    infra: exca.MapInfra = exca.MapInfra(version="1")

    _held_out: list[tuple[str, np.ndarray, np.ndarray]] = pydantic.PrivateAttr([])
    _held_out_channels: list[str] = pydantic.PrivateAttr([])

    def model_post_init(self, context: tp.Any) -> None:
        super().model_post_init(context)
        if self.data.neuro.frequency != self.data.extractor.frequency:
            raise ValueError(
                f"neuro frequency ({self.data.neuro.frequency}) must equal extractor "
                f"frequency ({self.data.extractor.frequency})"
            )
        if self.data.extractor.aggregation == "trigger":
            raise ValueError("extractor.aggregation must be `average` or `sum`, got `trigger`")
        if not 0.0 < self.alpha_holdout < 1.0:
            raise ValueError(f"alpha_holdout must be in (0, 1), got {self.alpha_holdout}")

    @infra.apply(item_uid=str)
    def _iter_subject_score(self, subjects: list[str]) -> tp.Iterator[pd.DataFrame]:
        for subject in subjects:
            yield self.get_subject_score(subject)

    def _receptive_field(self, frequency: float, estimator: tp.Any, **kwargs: tp.Any) -> ReceptiveField:
        return ReceptiveField(
            self.tmin, self.tmax, frequency, estimator=estimator, n_jobs=-1,
            scoring="corrcoef", **kwargs,
        )

    def _fit_predict(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        frequency: float,
        alphas: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Held-out (prediction, signal) on the samples ``ReceptiveField.score`` keeps."""
        model = self._receptive_field(frequency, Ridge(alpha=alphas), patterns=True)
        model.fit(X, Y)
        valid = model.valid_samples_
        return np.asarray(model.predict(X_test))[valid], np.asarray(Y_test)[valid]

    def _chapter_folds(
        self, X: np.ndarray, Y: np.ndarray, frequency: float, n_splits: int, desc: str
    ) -> tp.Iterator[tuple[int, np.ndarray, np.ndarray]]:
        X_cv, X_alpha, Y_cv, Y_alpha = train_test_split(
            X, Y, test_size=self.alpha_holdout, shuffle=False
        )
        rf = self._receptive_field(
            frequency, RidgeCV(alphas=self.alphas, cv=None, alpha_per_target=True)
        )
        rf.fit(X_alpha, Y_alpha)
        alphas = rf.estimator_.alpha_
        splits = KFold(n_splits=n_splits, shuffle=False).split(X_cv)
        for split, (train, test) in tqdm(enumerate(splits), total=n_splits, desc=desc):
            pred, actual = self._fit_predict(
                X_cv[train], Y_cv[train], X_cv[test], Y_cv[test], frequency, alphas
            )
            yield split, pred, actual

    def _fold_mean_statistic(
        self,
        folds: list[tuple[str, np.ndarray, np.ndarray]],
        shuffle: tp.Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Per-channel score: mean over folds within a chapter, then over chapters."""
        by_chapter: dict[str, list[np.ndarray]] = {}
        for timeline, pred, actual in folds:
            if shuffle is not None:
                actual = shuffle(actual)
            by_chapter.setdefault(timeline, []).append(np.asarray(self.scoring(pred, actual)))
        return np.mean([np.mean(s, axis=0) for s in by_chapter.values()], axis=0)

    def _permutation_null(
        self, folds: list[tuple[str, np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-channel observed score and ``(count + 1) / (n + 1)`` permutation p."""
        rng = np.random.default_rng(self.null_seed)
        observed = self._fold_mean_statistic(folds)
        null = np.empty((self.n_permutations, len(observed)))
        for i in range(self.n_permutations):
            null[i] = self._fold_mean_statistic(folds, shuffle=rng.permutation)
        pvalues = ((null >= observed).sum(0) + 1) / (self.n_permutations + 1)
        return observed, pvalues

    def _chapter_pca(self, X: np.ndarray) -> np.ndarray:
        pca_cls = IncrementalPCA if self.chapter_pca_incremental else PCA
        pca = make_pipeline(
            StandardScaler(), pca_cls(n_components=self.chapter_pca_components, whiten=False)
        )
        return pca.fit_transform(X)

    def _subject_segments(self, subject: str) -> tuple:
        """(dataset, triggers, subject mask, chapters, channels common to all chapters)."""
        events = self.data.study.run()
        all_dset = self.data._segmenter.apply(events)
        no_neuro = find_incomplete_segments(
            all_dset.segments, list(self.data.neuro._event_types_helper.classes)
        )
        if no_neuro:
            logger.info("dropping %d segment(s) with no neuro event", len(no_neuro))
            keep = np.ones(len(all_dset.segments), dtype=bool)
            keep[list(no_neuro)] = False
            all_dset = all_dset.select(keep)
        triggers = all_dset.triggers.reset_index(drop=True)
        in_subject = triggers.subject == subject
        timelines = sorted(triggers.loc[in_subject, "timeline"].unique())
        if not timelines:
            raise ValueError(f"no segments for subject {subject!r}")

        # A chapter's recording can lack a contact that another chapter has
        chapter_channels = {
            timeline: dset_channels(
                all_dset.select((in_subject & (triggers.timeline == timeline)).to_numpy())
            )
            for timeline in timelines
        }
        common = set.intersection(*(set(c) for c in chapter_channels.values()))
        channels = [ch for ch in chapter_channels[timelines[0]] if ch in common]
        if not channels:
            raise ValueError(f"{subject}: no neuro channel is shared by all chapters")
        for timeline, names in chapter_channels.items():
            if len(names) != len(channels):
                logger.warning(
                    "%s / %s: dropping channel(s) absent from other chapters: %s",
                    subject, timeline, sorted(set(names) - common),
                )
        return all_dset, triggers, in_subject, timelines, channels

    def get_subject_score(self, subject: str) -> pd.DataFrame:
        """One row per (chapter, fold, channel); ``perm_r`` / ``perm_p`` with the null."""
        all_dset, triggers, in_subject, timelines, channels = self._subject_segments(subject)
        out: list[dict] = []
        held_out: list[tuple[str, np.ndarray, np.ndarray]] = []
        for timeline in timelines:
            mask = (in_subject & (triggers.timeline == timeline)).to_numpy()
            dset = all_dset.select(mask)
            n_segments = int(mask.sum())
            n_splits = min(self.cv.n_splits, n_segments)
            if n_splits < 2:
                logger.warning("%s / %s: %d segment(s), skipped", subject, timeline, n_segments)
                continue
            dset.prepare()
            frequency = dset.extractors["neuro"].frequency
            X, Y = _load_data(dset)
            Y = Y[:, dset.extractors["neuro"]._get_channels(channels)]
            if self.chapter_pca_components is not None:
                X = self._chapter_pca(X)
            # z-scored over the whole chapter before splitting
            X = StandardScaler().fit_transform(X)
            Y = StandardScaler().fit_transform(Y)
            if len(X) % n_segments:
                raise RuntimeError(
                    f"{subject} / {timeline}: {len(X)} samples is not a multiple of "
                    f"{n_segments} segments"
                )
            for split, pred, actual in self._chapter_folds(
                X, Y, frequency, n_splits, f"Cross-val {timeline}"
            ):
                held_out.append((timeline, pred, actual))
                for dim, score in enumerate(np.atleast_1d(self.scoring(pred, actual))):
                    out.append(dict(dim=dim, score=score, split=split, timeline=timeline))

        self._held_out, self._held_out_channels = held_out, channels
        if not out:
            raise ValueError(f"no chapter of {subject!r} had enough segments to cross-validate")
        df = pd.DataFrame(out)
        if self.n_permutations:
            observed, pvalues = self._permutation_null(held_out)
            df["perm_r"] = observed[df["dim"]]
            df["perm_p"] = pvalues[df["dim"]]
        df["subject"] = subject
        df["channel"] = [channels[d] for d in df["dim"]]
        return df

    def held_out_predictions(self, subject: str) -> dict[str, np.ndarray]:
        """Refit ``subject`` without caching: every fold's held-out prediction and signal."""
        trf = self.model_copy(update={"n_permutations": 0})
        trf.get_subject_score(subject)
        arrays: dict[str, np.ndarray] = {"channels": np.array(trf._held_out_channels)}
        for i, (timeline, pred, actual) in enumerate(trf._held_out):
            arrays[f"{i}/pred"] = pred.astype(np.float32)
            arrays[f"{i}/actual"] = actual.astype(np.float32)
            arrays[f"{i}/timeline"] = np.array(timeline)
        return arrays


def trf_scores_to_df(scores: pd.DataFrame, feat_key: str) -> pd.DataFrame:
    """One row per (subject, channel): mean over folds within a chapter, then over chapters.

    ``pvalue``: one-sided Wilcoxon over every (chapter, fold) score of the channel.
    """
    long = scores.dropna(subset=["score"])
    keep = ["subject", "dim"]
    per_chapter = long.groupby(keep + ["timeline"])["score"].mean().reset_index()
    df = per_chapter.groupby(keep)["score"].agg(score="mean", n_chapters="size").reset_index()
    df["feature_group"] = feat_key
    df["feature"] = feat_key
    df["feat_dim"] = feat_key + df["dim"].astype(str)
    pvalues = (
        long.groupby(keep)["score"]
        .apply(lambda s: wilcoxon_greater_pvalue(s.to_numpy()))
        .reset_index(name="pvalue")
    )
    df = df.merge(pvalues, on=keep, how="left")
    channel_cols = ["subject", "dim", "channel"]
    channel_cols += [c for c in ("perm_r", "perm_p") if c in scores.columns]
    channels = scores[channel_cols].drop_duplicates(subset=keep)
    return df.merge(channels, on=keep, how="left")
