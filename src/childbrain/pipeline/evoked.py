"""Word- (or phoneme-) evoked broadband responses (Fig. 1, Fig. S1-S2).

* signal: bipolar, notch at 50 Hz and harmonics, 0.05–20 Hz, 20 Hz, no scaler;
* epochs: every event, ``tmin`` … ``tmax`` s around its onset, no baseline;
  chapters concatenated on the channels they all share;
* ``do_clipping(epochs, 5)``: each (channel, time) clipped to ±5 SD across epochs;
* evoked: mean over epochs, signed, in µV.
"""

import typing as tp

import exca
import neuralset as ns
import numpy as np
import pandas as pd
import pydantic

from .decoding import EpochData
from .encoding import dset_channels


class EvokedResponses(pydantic.BaseModel):
    """Average event-locked response per bipolar channel, one subject per item.

    ``data`` defines the events: its extractor's trigger events are epoched and
    only its ``neuro`` is read. ``data.stop`` should be ``tmax`` plus one sample
    so that the epoch includes ``tmax``, as ``mne.Epochs``.
    """

    data: EpochData
    clip_std: float | None = 5.0
    infra: exca.MapInfra = exca.MapInfra(version="1")
    model_config = pydantic.ConfigDict(extra="forbid")

    @infra.apply(item_uid=str)
    def _iter_subject_score(self, subjects: list[str]) -> tp.Iterator[pd.DataFrame]:
        for subject in subjects:
            yield self.get_subject_evoked(subject)

    def get_scores(self, subjects: list[str]) -> pd.DataFrame:
        return pd.concat(list(self._iter_subject_score(subjects)), ignore_index=True)

    def get_subject_evoked(self, subject: str) -> pd.DataFrame:
        events = self.data.study.run()
        dset = self.data._segmenter.apply(events.loc[events.subject == subject])
        triggers = dset.triggers.reset_index(drop=True)
        chapters: list[tuple[list[str], np.ndarray]] = []
        for timeline in sorted(triggers.timeline.unique()):
            chapter = dset.select((triggers.timeline == timeline).to_numpy())
            chapter.prepare()
            names = dset_channels(chapter)
            neuro = chapter.extractors["neuro"]
            assert isinstance(neuro, ns.extractors.neuro.MneRaw)
            cols = neuro._get_channels(names)
            chapters.append((names, chapter.load_all().data["neuro"].numpy()[:, cols]))
        if not chapters:
            raise ValueError(f"no epochs for {subject!r}")

        common = set.intersection(*(set(names) for names, _ in chapters))
        channels = [ch for ch in chapters[0][0] if ch in common]
        epochs = np.concatenate(
            [x[:, [names.index(ch) for ch in channels]] for names, x in chapters]
        )
        if self.clip_std is not None:
            threshold = self.clip_std * epochs.std(axis=0)
            epochs = np.clip(epochs, -threshold, threshold)
        evoked = epochs.mean(axis=0) * 1e6
        frequency = float(self.data.neuro.frequency)
        times = self.data.start + np.arange(evoked.shape[1]) / frequency
        return pd.DataFrame(
            dict(
                subject=subject,
                channel=np.repeat(channels, len(times)),
                times=np.tile(np.round(times, 6), len(channels)),
                val=evoked.ravel(),
                n_words=len(epochs),
            )
        )
